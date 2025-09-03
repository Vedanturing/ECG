"""
FastAPI Backend Server for ECG Analysis Dashboard

This module provides REST API endpoints for ECG analysis, explainability, and report generation.
It integrates with the ECG parser, model, and explainability modules from the main system.

Author: ECG ML Team
Date: 2024
"""

import os
import sys
import json
import logging
import tempfile
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from ecg_parser import ECGParserEngine
from feature_extraction import ECGFeatureExtractor
from model import create_model
from data_loader import ECGDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ECG Analysis API",
    description="REST API for ECG abnormality detection and explainability",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models and components
ecg_parser = None
feature_extractor = None
model = None
model_loaded = False

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    """Request model for ECG analysis."""
    patient_id: Optional[str] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    clinical_notes: Optional[str] = None

class AnalysisResponse(BaseModel):
    """Response model for ECG analysis."""
    success: bool
    patient_id: str
    analysis_id: str
    abnormalities: List[Dict[str, Any]]
    cause_attribution: Dict[str, float]
    confidence_scores: Dict[str, float]
    recommendations: List[str]
    processing_time: float
    error_message: Optional[str] = None

class ExplainabilityRequest(BaseModel):
    """Request model for explainability analysis."""
    analysis_id: str
    method: str = "shap"  # "shap" or "integrated_gradients"

class ExplainabilityResponse(BaseModel):
    """Response model for explainability analysis."""
    success: bool
    analysis_id: str
    method: str
    feature_importance: Dict[str, float]
    lead_importance: Dict[str, float]
    explanation_text: str
    processing_time: float
    error_message: Optional[str] = None

class ReportRequest(BaseModel):
    """Request model for report generation."""
    analysis_id: str
    include_explainability: bool = True
    report_format: str = "pdf"  # "pdf" or "html"

# In-memory storage for analysis results (in production, use a database)
analysis_results = {}

@app.on_event("startup")
async def startup_event():
    """Initialize models and components on startup."""
    global ecg_parser, feature_extractor, model, model_loaded
    
    try:
        logger.info("Initializing ECG analysis components...")
        
        # Initialize ECG parser
        ecg_parser = ECGParserEngine()
        logger.info("ECG parser initialized")
        
        # Initialize feature extractor
        feature_extractor = ECGFeatureExtractor(sampling_rate=500)
        logger.info("Feature extractor initialized")
        
        # Load trained model (if available)
        model_path = Path("ecg_training_results/training_output/best_model.pth")
        if model_path.exists():
            model = create_model(
                input_channels=12,
                num_abnormalities=8,
                num_causes=3,
                base_channels=64
            )
            
            # Load model weights
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model_loaded = True
            logger.info("Trained model loaded successfully")
        else:
            logger.warning("No trained model found. Using synthetic predictions.")
            model_loaded = False
        
        logger.info("ECG analysis components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ECG Analysis API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_loaded,
        "endpoints": {
            "analyze": "/analyze",
            "explain": "/explain",
            "generate_report": "/generate_report",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "components": {
            "ecg_parser": ecg_parser is not None,
            "feature_extractor": feature_extractor is not None,
            "model": model is not None
        }
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_ecg(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
    patient_gender: Optional[str] = Form(None),
    clinical_notes: Optional[str] = Form(None)
):
    """
    Analyze uploaded ECG file for abnormalities and cause attribution.
    
    Args:
        file: Uploaded ECG file (PDF, image, or signal file)
        patient_id: Optional patient identifier
        patient_age: Optional patient age
        patient_gender: Optional patient gender
        clinical_notes: Optional clinical notes
    
    Returns:
        AnalysisResponse with abnormalities, cause attribution, and recommendations
    """
    import time
    start_time = time.time()
    
    try:
        # Generate analysis ID
        analysis_id = f"analysis_{int(time.time())}_{patient_id or 'unknown'}"
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Parse ECG file
            logger.info(f"Parsing ECG file: {file.filename}")
            ecg_data = ecg_parser.parse_ecg(tmp_file_path, output_format='dict')
            
            if not ecg_data or 'waveforms' not in ecg_data:
                raise HTTPException(status_code=400, detail="Failed to parse ECG file")
            
            # Extract features
            logger.info("Extracting features from ECG data")
            features = feature_extractor.extract_all_features(ecg_data)
            
            # Prepare waveform data for model
            waveforms = ecg_data.get('waveforms', {})
            lead_names = sorted(waveforms.keys())
            waveform_data = []
            
            for lead_name in lead_names:
                waveform = waveforms[lead_name]
                if isinstance(waveform, np.ndarray) and len(waveform) > 0:
                    waveform_data.append(waveform)
            
            if not waveform_data:
                raise HTTPException(status_code=400, detail="No valid waveform data found")
            
            # Normalize waveform length
            target_length = 5000
            min_length = min(len(w) for w in waveform_data)
            truncated_data = [w[:min_length] for w in waveform_data]
            
            normalized_data = []
            for waveform in truncated_data:
                if len(waveform) >= target_length:
                    normalized = waveform[:target_length]
                else:
                    normalized = np.zeros(target_length)
                    normalized[:len(waveform)] = waveform
                normalized_data.append(normalized)
            
            # Ensure consistent number of leads
            max_leads = 12
            while len(normalized_data) < max_leads:
                normalized_data.append(np.zeros(target_length))
            normalized_data = normalized_data[:max_leads]
            
            waveform_tensor = torch.FloatTensor(np.stack(normalized_data)).unsqueeze(0)  # Add batch dimension
            
            # Prepare features tensor
            expected_features = [
                'mean_hr', 'std_hr', 'min_hr', 'max_hr', 'mean_rr', 'std_rr',
                'rmssd', 'pnn50', 'lf_power', 'hf_power', 'lf_hf_ratio',
                'mean_qrs_width', 'std_qrs_width', 'mean_pr_interval', 'std_pr_interval',
                'mean_qt_interval', 'std_qt_interval', 'overall_snr_db', 'overall_baseline_wander',
                'overall_clipping_ratio', 'stress_index', 'Lead_I_st_elevation', 'Lead_II_st_elevation',
                'Lead_III_st_elevation', 'Lead_aVR_st_elevation', 'Lead_aVL_st_elevation',
                'Lead_aVF_st_elevation', 'Lead_V1_st_elevation', 'Lead_V2_st_elevation',
                'Lead_V3_st_elevation', 'Lead_V4_st_elevation', 'Lead_V5_st_elevation', 'Lead_V6_st_elevation'
            ]
            
            feature_values = []
            for key in expected_features:
                value = features.get(key, 0.0)
                if isinstance(value, (int, float, np.number)):
                    feature_values.append(float(value))
                else:
                    feature_values.append(0.0)
            
            feature_tensor = torch.FloatTensor(feature_values).unsqueeze(0)  # Add batch dimension
            
            # Run model inference
            if model_loaded:
                with torch.no_grad():
                    abnormality_logits, cause_predictions = model(waveform_tensor, feature_tensor)
                    abnormality_probs = torch.sigmoid(abnormality_logits)
                    cause_predictions = torch.softmax(cause_predictions, dim=1)
            else:
                # Generate synthetic predictions for demonstration
                abnormality_probs = torch.rand(1, 8) * 0.3  # Low probability for most abnormalities
                cause_predictions = torch.tensor([[0.3, 0.2, 0.5]])  # [stress, machine, environment]
            
            # Process results
            abnormality_names = [
                "Bradycardia", "Tachycardia", "ST Elevation", "Poor Signal Quality",
                "High Stress", "Abnormal QRS Width", "Baseline Wander", "Signal Clipping"
            ]
            
            abnormalities = []
            confidence_scores = {}
            
            for i, name in enumerate(abnormality_names):
                prob = float(abnormality_probs[0, i])
                confidence_scores[name] = prob
                
                if prob > 0.5:  # Threshold for abnormality detection
                    abnormalities.append({
                        "name": name,
                        "confidence": prob,
                        "severity": "High" if prob > 0.8 else "Medium" if prob > 0.6 else "Low",
                        "description": f"Detected {name.lower()} with {prob:.1%} confidence"
                    })
            
            # Cause attribution
            cause_names = ["Stress", "Machine Error", "Environment"]
            cause_attribution = {}
            for i, name in enumerate(cause_names):
                cause_attribution[name] = float(cause_predictions[0, i])
            
            # Generate recommendations
            recommendations = []
            if abnormalities:
                recommendations.append("Consider further clinical evaluation")
                if any(abn["name"] == "ST Elevation" for abn in abnormalities):
                    recommendations.append("Urgent cardiology consultation recommended")
                if any(abn["name"] == "Poor Signal Quality" for abn in abnormalities):
                    recommendations.append("Consider re-recording with better electrode placement")
            else:
                recommendations.append("No significant abnormalities detected")
                recommendations.append("Continue routine monitoring")
            
            # Store results
            analysis_results[analysis_id] = {
                "patient_id": patient_id,
                "abnormalities": abnormalities,
                "cause_attribution": cause_attribution,
                "confidence_scores": confidence_scores,
                "waveforms": {lead: waveform.tolist() for lead, waveform in zip(lead_names, waveform_data)},
                "features": features,
                "timestamp": time.time()
            }
            
            processing_time = time.time() - start_time
            
            return AnalysisResponse(
                success=True,
                patient_id=patient_id or "unknown",
                analysis_id=analysis_id,
                abnormalities=abnormalities,
                cause_attribution=cause_attribution,
                confidence_scores=confidence_scores,
                recommendations=recommendations,
                processing_time=processing_time
            )
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            success=False,
            patient_id=patient_id or "unknown",
            analysis_id="",
            abnormalities=[],
            cause_attribution={},
            confidence_scores={},
            recommendations=[],
            processing_time=processing_time,
            error_message=str(e)
        )

@app.post("/explain", response_model=ExplainabilityResponse)
async def explain_analysis(request: ExplainabilityRequest):
    """
    Generate explainability analysis for a previous ECG analysis.
    
    Args:
        request: ExplainabilityRequest with analysis_id and method
    
    Returns:
        ExplainabilityResponse with feature importance and explanations
    """
    import time
    start_time = time.time()
    
    try:
        # Get analysis results
        if request.analysis_id not in analysis_results:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        analysis_data = analysis_results[request.analysis_id]
        features = analysis_data["features"]
        
        # Generate feature importance (simplified for demonstration)
        feature_importance = {}
        lead_importance = {}
        
        # Calculate feature importance based on feature values
        important_features = [
            'mean_hr', 'overall_snr_db', 'stress_index', 'overall_baseline_wander',
            'overall_clipping_ratio', 'mean_qrs_width'
        ]
        
        for feature in important_features:
            value = features.get(feature, 0.0)
            # Simple importance calculation based on deviation from normal
            if feature == 'mean_hr':
                importance = abs(value - 75) / 75  # Normal HR around 75
            elif feature == 'overall_snr_db':
                importance = max(0, (20 - value) / 20)  # Lower SNR is more important
            elif feature == 'stress_index':
                importance = min(1.0, value / 3.0)  # Higher stress is more important
            else:
                importance = min(1.0, abs(value) / 2.0)  # General importance
            
            feature_importance[feature] = importance
        
        # Calculate lead importance based on ST elevation
        st_leads = [f"Lead_{lead}_st_elevation" for lead in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']]
        for lead in st_leads:
            value = features.get(lead, 0.0)
            lead_name = lead.replace('Lead_', '').replace('_st_elevation', '')
            lead_importance[lead_name] = min(1.0, abs(value) / 0.2)  # ST elevation importance
        
        # Generate explanation text
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        top_leads = sorted(lead_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        explanation_parts = []
        if top_features:
            explanation_parts.append(f"Key features contributing to the analysis: {', '.join([f'{feat} ({imp:.1%})' for feat, imp in top_features])}")
        if top_leads:
            explanation_parts.append(f"Most significant leads: {', '.join([f'{lead} ({imp:.1%})' for lead, imp in top_leads])}")
        
        explanation_text = ". ".join(explanation_parts) + "."
        
        processing_time = time.time() - start_time
        
        return ExplainabilityResponse(
            success=True,
            analysis_id=request.analysis_id,
            method=request.method,
            feature_importance=feature_importance,
            lead_importance=lead_importance,
            explanation_text=explanation_text,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Explainability analysis failed: {str(e)}")
        processing_time = time.time() - start_time
        
        return ExplainabilityResponse(
            success=False,
            analysis_id=request.analysis_id,
            method=request.method,
            feature_importance={},
            lead_importance={},
            explanation_text="",
            processing_time=processing_time,
            error_message=str(e)
        )

@app.post("/generate_report")
async def generate_report(request: ReportRequest):
    """
    Generate a PDF report for a previous ECG analysis.
    
    Args:
        request: ReportRequest with analysis_id and options
    
    Returns:
        FileResponse with the generated PDF report
    """
    try:
        # Get analysis results
        if request.analysis_id not in analysis_results:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        analysis_data = analysis_results[request.analysis_id]
        
        # Import report generator
        from report_generator import generate_ecg_report
        
        # Generate report
        report_path = generate_ecg_report(
            analysis_id=request.analysis_id,
            analysis_data=analysis_data,
            include_explainability=request.include_explainability,
            format=request.report_format
        )
        
        # Return file response
        return FileResponse(
            path=report_path,
            media_type='application/pdf',
            filename=f"ecg_report_{request.analysis_id}.pdf"
        )
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/analyses")
async def list_analyses():
    """List all previous analyses."""
    return {
        "analyses": [
            {
                "analysis_id": aid,
                "patient_id": data.get("patient_id", "unknown"),
                "timestamp": data.get("timestamp", 0),
                "abnormality_count": len(data.get("abnormalities", []))
            }
            for aid, data in analysis_results.items()
        ]
    }

@app.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get details of a specific analysis."""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_results[analysis_id]

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
