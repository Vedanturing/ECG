"""
ECG Model Evaluation Module

This module implements comprehensive evaluation for the ECG abnormality detection system:
1. Classification metrics for abnormality detection
2. Regression metrics for cause attribution
3. Model comparison and benchmarking
4. Clinical interpretation and reporting

Author: ECG ML Team
Date: 2024
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.manifold import TSNE
import torch
import warnings


class ECGEvaluator:
    """
    Comprehensive evaluator for ECG abnormality detection models.
    
    Evaluates both deep learning and baseline ML approaches,
    providing detailed metrics and clinical interpretations.
    """
    
    def __init__(self, 
                 output_dir: str = "evaluation_results",
                 config: Optional[Dict] = None):
        """
        Initialize the ECG evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
            config: Evaluation configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.config = config or {}
        
        # Evaluation results storage
        self.deep_learning_results = {}
        self.baseline_results = {}
        self.comparison_results = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Clinical thresholds for interpretation
        self.clinical_thresholds = {
            'classification_accuracy': 0.85,  # 85% accuracy threshold
            'classification_f1': 0.80,        # 80% F1 score threshold
            'regression_mae': 15.0,           # 15% MAE threshold for cause attribution
            'regression_r2': 0.70,            # 70% R² threshold
            'stress_detection_mae': 20.0,     # 20% MAE for stress detection
            'machine_error_mae': 25.0,        # 25% MAE for machine error detection
            'environment_mae': 20.0           # 20% MAE for environment detection
        }
    
    def evaluate_deep_learning_model(self, 
                                   model_predictions: Dict,
                                   ground_truth: Dict) -> Dict:
        """
        Evaluate deep learning model performance.
        
        Args:
            model_predictions: Model predictions dictionary
            ground_truth: Ground truth labels dictionary
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            self.logger.info("Evaluating deep learning model...")
            
            # Extract predictions and labels
            abnormality_predictions = model_predictions.get('abnormality_predictions', [])
            cause_predictions = model_predictions.get('cause_predictions', [])
            abnormality_labels = ground_truth.get('abnormality_labels', [])
            cause_labels = ground_truth.get('cause_labels', [])
            
            if not (len(abnormality_predictions) > 0 and len(cause_predictions) > 0 and 
                   len(abnormality_labels) > 0 and len(cause_labels) > 0):
                raise ValueError("Missing required prediction or label data")
            
            # Convert to numpy arrays if needed
            abnormality_predictions = np.array(abnormality_predictions)
            cause_predictions = np.array(cause_predictions)
            abnormality_labels = np.array(abnormality_labels)
            cause_labels = np.array(cause_labels)
            
            # Evaluate abnormality classification
            classification_metrics = self._evaluate_classification(
                abnormality_predictions, abnormality_labels
            )
            
            # Evaluate cause attribution regression
            regression_metrics = self._evaluate_regression(
                cause_predictions, cause_labels
            )
            
            # Combine metrics
            dl_results = {
                'classification': classification_metrics,
                'regression': regression_metrics,
                'overall': self._calculate_overall_metrics(
                    classification_metrics, regression_metrics
                )
            }
            
            self.deep_learning_results = dl_results
            
            # Save results
            self._save_evaluation_results('deep_learning', dl_results)
            
            self.logger.info("Deep learning model evaluation completed")
            return dl_results
            
        except Exception as e:
            self.logger.error(f"Deep learning evaluation failed: {str(e)}")
            return {}
    
    def evaluate_baseline_models(self, 
                               baseline_predictions: Dict,
                               ground_truth: Dict) -> Dict:
        """
        Evaluate baseline ML models performance.
        
        Args:
            baseline_predictions: Baseline model predictions
            ground_truth: Ground truth labels
            
        Returns:
            Dictionary containing evaluation metrics for baseline models
        """
        try:
            self.logger.info("Evaluating baseline ML models...")
            
            baseline_results = {}
            
            for model_name, predictions in baseline_predictions.items():
                self.logger.info(f"Evaluating {model_name}...")
                
                if 'cause_predictions' in predictions:
                    # Evaluate cause attribution
                    cause_predictions = np.array(predictions['cause_predictions'])
                    cause_labels = np.array(ground_truth['cause_labels'])
                    
                    regression_metrics = self._evaluate_regression(
                        cause_predictions, cause_labels
                    )
                    
                    baseline_results[model_name] = {
                        'regression': regression_metrics,
                        'overall': self._calculate_overall_metrics(
                            {}, regression_metrics  # No classification for baseline
                        )
                    }
            
            self.baseline_results = baseline_results
            
            # Save results
            self._save_evaluation_results('baseline', baseline_results)
            
            self.logger.info("Baseline models evaluation completed")
            return baseline_results
            
        except Exception as e:
            self.logger.error(f"Baseline evaluation failed: {str(e)}")
            return {}
    
    def _evaluate_classification(self, 
                               predictions: np.ndarray,
                               labels: np.ndarray) -> Dict:
        """
        Evaluate classification performance.
        
        Args:
            predictions: Model predictions (probabilities)
            labels: Ground truth labels (binary)
            
        Returns:
            Dictionary containing classification metrics
        """
        # Convert probabilities to binary predictions
        threshold = 0.5
        binary_predictions = (predictions > threshold).astype(int)
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(labels.flatten(), binary_predictions.flatten())
        
        # Per-class metrics
        num_classes = predictions.shape[1]
        class_names = [f"abnormality_class_{i}" for i in range(num_classes)]
        
        for i in range(num_classes):
            class_name = class_names[i]
            
            # Calculate metrics for each class
            try:
                metrics[f"{class_name}_precision"] = precision_score(
                    labels[:, i], binary_predictions[:, i], zero_division=0
                )
                metrics[f"{class_name}_recall"] = recall_score(
                    labels[:, i], binary_predictions[:, i], zero_division=0
                )
                metrics[f"{class_name}_f1"] = f1_score(
                    labels[:, i], binary_predictions[:, i], zero_division=0
                )
                
                # ROC AUC (requires probabilities)
                if len(np.unique(labels[:, i])) > 1:
                    metrics[f"{class_name}_auc"] = roc_auc_score(
                        labels[:, i], predictions[:, i]
                    )
                else:
                    metrics[f"{class_name}_auc"] = 0.5
                    
            except Exception as e:
                self.logger.warning(f"Failed to calculate metrics for {class_name}: {str(e)}")
                metrics[f"{class_name}_precision"] = 0.0
                metrics[f"{class_name}_recall"] = 0.0
                metrics[f"{class_name}_f1"] = 0.0
                metrics[f"{class_name}_auc"] = 0.5
        
        # Average metrics across classes
        metrics['macro_precision'] = np.mean([
            metrics[f"{class_name}_precision"] for class_name in class_names
        ])
        metrics['macro_recall'] = np.mean([
            metrics[f"{class_name}_recall"] for class_name in class_names
        ])
        metrics['macro_f1'] = np.mean([
            metrics[f"{class_name}_f1"] for class_name in class_names
        ])
        metrics['macro_auc'] = np.mean([
            metrics[f"{class_name}_auc"] for class_name in class_names
        ])
        
        # Weighted metrics (accounting for class imbalance)
        metrics['weighted_precision'] = precision_score(
            labels, binary_predictions, average='weighted', zero_division=0
        )
        metrics['weighted_recall'] = recall_score(
            labels, binary_predictions, average='weighted', zero_division=0
        )
        metrics['weighted_f1'] = f1_score(
            labels, binary_predictions, average='weighted', zero_division=0
        )
        
        return metrics
    
    def _evaluate_regression(self, 
                           predictions: np.ndarray,
                           labels: np.ndarray) -> Dict:
        """
        Evaluate regression performance for cause attribution.
        
        Args:
            predictions: Model predictions (percentages)
            labels: Ground truth labels (percentages)
            
        Returns:
            Dictionary containing regression metrics
        """
        metrics = {}
        
        # Cause type names
        cause_names = ['stress', 'machine', 'environment']
        
        # Per-cause metrics
        for i, cause_name in enumerate(cause_names):
            try:
                mae = mean_absolute_error(labels[:, i], predictions[:, i])
                rmse = np.sqrt(mean_squared_error(labels[:, i], predictions[:, i]))
                r2 = r2_score(labels[:, i], predictions[:, i])
                
                metrics[f"{cause_name}_mae"] = mae
                metrics[f"{cause_name}_rmse"] = rmse
                metrics[f"{cause_name}_r2"] = r2
                
                # Percentage error
                mape = np.mean(np.abs((labels[:, i] - predictions[:, i]) / labels[:, i])) * 100
                metrics[f"{cause_name}_mape"] = mape
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate metrics for {cause_name}: {str(e)}")
                metrics[f"{cause_name}_mae"] = float('inf')
                metrics[f"{cause_name}_rmse"] = float('inf')
                metrics[f"{cause_name}_r2"] = 0.0
                metrics[f"{cause_name}_mape"] = float('inf')
        
        # Overall regression metrics
        metrics['overall_mae'] = np.mean([
            metrics[f"{cause_name}_mae"] for cause_name in cause_names
        ])
        metrics['overall_rmse'] = np.mean([
            metrics[f"{cause_name}_rmse"] for cause_name in cause_names
        ])
        metrics['overall_r2'] = np.mean([
            metrics[f"{cause_name}_r2"] for cause_name in cause_names
        ])
        metrics['overall_mape'] = np.mean([
            metrics[f"{cause_name}_mape"] for cause_name in cause_names
        ])
        
        return metrics
    
    def _calculate_overall_metrics(self, 
                                 classification_metrics: Dict,
                                 regression_metrics: Dict) -> Dict:
        """
        Calculate overall performance metrics.
        
        Args:
            classification_metrics: Classification performance metrics
            regression_metrics: Regression performance metrics
            
        Returns:
            Dictionary containing overall metrics
        """
        overall_metrics = {}
        
        # Classification scores
        if classification_metrics:
            overall_metrics['classification_score'] = (
                classification_metrics.get('macro_f1', 0) * 0.4 +
                classification_metrics.get('macro_auc', 0) * 0.3 +
                classification_metrics.get('accuracy', 0) * 0.3
            )
        else:
            overall_metrics['classification_score'] = 0.0
        
        # Regression scores
        if regression_metrics:
            # Convert MAE to score (lower MAE = higher score)
            mae_score = max(0, 1 - regression_metrics.get('overall_mae', 100) / 100)
            r2_score = max(0, regression_metrics.get('overall_r2', 0))
            
            overall_metrics['regression_score'] = mae_score * 0.6 + r2_score * 0.4
        else:
            overall_metrics['regression_score'] = 0.0
        
        # Combined score
        overall_metrics['combined_score'] = (
            overall_metrics['classification_score'] * 0.6 +
            overall_metrics['regression_score'] * 0.4
        )
        
        # Performance level assessment
        combined_score = overall_metrics['combined_score']
        if combined_score >= 0.9:
            performance_level = 'Excellent'
        elif combined_score >= 0.8:
            performance_level = 'Good'
        elif combined_score >= 0.7:
            performance_level = 'Fair'
        elif combined_score >= 0.6:
            performance_level = 'Poor'
        else:
            performance_level = 'Very Poor'
        
        overall_metrics['performance_level'] = performance_level
        
        return overall_metrics
    
    def compare_models(self) -> Dict:
        """
        Compare deep learning and baseline model performance.
        
        Returns:
            Dictionary containing comparison results
        """
        try:
            self.logger.info("Comparing model performance...")
            
            if not self.deep_learning_results or not self.baseline_results:
                raise ValueError("Both deep learning and baseline results required for comparison")
            
            comparison = {}
            
            # Compare cause attribution performance
            dl_regression = self.deep_learning_results.get('regression', {})
            
            for model_name, baseline_result in self.baseline_results.items():
                baseline_regression = baseline_result.get('regression', {})
                
                model_comparison = {}
                
                # Compare each cause type
                for cause_name in ['stress', 'machine', 'environment']:
                    dl_mae = dl_regression.get(f"{cause_name}_mae", float('inf'))
                    baseline_mae = baseline_regression.get(f"{cause_name}_mae", float('inf'))
                    
                    if dl_mae < baseline_mae:
                        improvement = ((baseline_mae - dl_mae) / baseline_mae) * 100
                        model_comparison[f"{cause_name}_improvement"] = improvement
                        model_comparison[f"{cause_name}_winner"] = "Deep Learning"
                    else:
                        improvement = ((dl_mae - baseline_mae) / dl_mae) * 100
                        model_comparison[f"{cause_name}_improvement"] = -improvement
                        model_comparison[f"{cause_name}_winner"] = "Baseline"
                
                # Overall comparison
                dl_overall_mae = dl_regression.get('overall_mae', float('inf'))
                baseline_overall_mae = baseline_regression.get('overall_mae', float('inf'))
                
                if dl_overall_mae < baseline_overall_mae:
                    overall_improvement = ((baseline_overall_mae - dl_overall_mae) / baseline_overall_mae) * 100
                    model_comparison['overall_improvement'] = overall_improvement
                    model_comparison['overall_winner'] = "Deep Learning"
                else:
                    overall_improvement = ((dl_overall_mae - baseline_overall_mae) / dl_overall_mae) * 100
                    model_comparison['overall_improvement'] = -overall_improvement
                    model_comparison['overall_winner'] = "Baseline"
                
                comparison[model_name] = model_comparison
            
            self.comparison_results = comparison
            
            # Save comparison results
            self._save_evaluation_results('comparison', comparison)
            
            self.logger.info("Model comparison completed")
            return comparison
            
        except Exception as e:
            self.logger.error(f"Model comparison failed: {str(e)}")
            return {}
    
    def generate_clinical_report(self) -> Dict:
        """
        Generate clinical interpretation report.
        
        Returns:
            Dictionary containing clinical assessment
        """
        try:
            self.logger.info("Generating clinical report...")
            
            if not self.deep_learning_results:
                raise ValueError("Deep learning results required for clinical report")
            
            clinical_report = {
                'model_performance': {},
                'clinical_interpretation': {},
                'recommendations': []
            }
            
            # Model performance assessment
            overall_metrics = self.deep_learning_results.get('overall', {})
            classification_metrics = self.deep_learning_results.get('classification', {})
            regression_metrics = self.deep_learning_results.get('regression', {})
            
            # Performance level
            performance_level = overall_metrics.get('performance_level', 'Unknown')
            clinical_report['model_performance']['overall_level'] = performance_level
            
            # Classification performance
            classification_score = overall_metrics.get('classification_score', 0)
            if classification_score >= 0.9:
                classification_assessment = "Excellent - Suitable for clinical use"
            elif classification_score >= 0.8:
                classification_assessment = "Good - Suitable for screening with validation"
            elif classification_score >= 0.7:
                classification_assessment = "Fair - Requires improvement before clinical use"
            else:
                classification_assessment = "Poor - Not suitable for clinical use"
            
            clinical_report['model_performance']['classification_assessment'] = classification_assessment
            
            # Regression performance
            regression_score = overall_metrics.get('regression_score', 0)
            if regression_score >= 0.8:
                regression_assessment = "Excellent - Reliable cause attribution"
            elif regression_score >= 0.7:
                regression_assessment = "Good - Reasonable cause attribution"
            elif regression_score >= 0.6:
                regression_assessment = "Fair - Limited reliability"
            else:
                regression_assessment = "Poor - Unreliable cause attribution"
            
            clinical_report['model_performance']['regression_assessment'] = regression_assessment
            
            # Clinical interpretation
            clinical_report['clinical_interpretation'] = self._interpret_clinical_metrics(
                classification_metrics, regression_metrics
            )
            
            # Generate recommendations
            clinical_report['recommendations'] = self._generate_clinical_recommendations(
                classification_metrics, regression_metrics
            )
            
            # Save clinical report
            self._save_evaluation_results('clinical_report', clinical_report)
            
            self.logger.info("Clinical report generated successfully")
            return clinical_report
            
        except Exception as e:
            self.logger.error(f"Clinical report generation failed: {str(e)}")
            return {}
    
    def _interpret_clinical_metrics(self, 
                                  classification_metrics: Dict,
                                  regression_metrics: Dict) -> Dict:
        """
        Interpret metrics from a clinical perspective.
        
        Args:
            classification_metrics: Classification performance metrics
            regression_metrics: Regression performance metrics
            
        Returns:
            Dictionary containing clinical interpretations
        """
        interpretation = {}
        
        # Classification interpretation
        if classification_metrics:
            # Overall classification performance
            macro_f1 = classification_metrics.get('macro_f1', 0)
            if macro_f1 >= 0.9:
                interpretation['abnormality_detection'] = "Highly reliable for detecting ECG abnormalities"
            elif macro_f1 >= 0.8:
                interpretation['abnormality_detection'] = "Reliable for detecting ECG abnormalities"
            elif macro_f1 >= 0.7:
                interpretation['abnormality_detection'] = "Moderately reliable for detecting ECG abnormalities"
            else:
                interpretation['abnormality_detection'] = "Limited reliability for detecting ECG abnormalities"
            
            # Per-class interpretation
            class_interpretations = {}
            for i in range(8):  # 8 abnormality classes
                class_name = f"abnormality_class_{i}"
                f1_score = classification_metrics.get(f"{class_name}_f1", 0)
                
                if f1_score >= 0.8:
                    class_interpretations[f"Class_{i}"] = "Reliable detection"
                elif f1_score >= 0.6:
                    class_interpretations[f"Class_{i}"] = "Moderate detection"
                else:
                    class_interpretations[f"Class_{i}"] = "Poor detection"
            
            interpretation['per_class_performance'] = class_interpretations
        
        # Regression interpretation
        if regression_metrics:
            # Stress detection
            stress_mae = regression_metrics.get('stress_mae', float('inf'))
            if stress_mae <= 15:
                interpretation['stress_assessment'] = "Accurate stress level assessment"
            elif stress_mae <= 25:
                interpretation['stress_assessment'] = "Moderately accurate stress assessment"
            else:
                interpretation['stress_assessment'] = "Limited accuracy in stress assessment"
            
            # Machine error detection
            machine_mae = regression_metrics.get('machine_mae', float('inf'))
            if machine_mae <= 20:
                interpretation['machine_error_detection'] = "Reliable machine error identification"
            elif machine_mae <= 30:
                interpretation['machine_error_detection'] = "Moderate machine error identification"
            else:
                interpretation['machine_error_detection'] = "Limited machine error identification"
            
            # Environmental factor assessment
            environment_mae = regression_metrics.get('environment_mae', float('inf'))
            if environment_mae <= 20:
                interpretation['environmental_assessment'] = "Accurate environmental factor assessment"
            elif environment_mae <= 30:
                interpretation['environmental_assessment'] = "Moderate environmental factor assessment"
            else:
                interpretation['environmental_assessment'] = "Limited environmental factor assessment"
        
        return interpretation
    
    def _generate_clinical_recommendations(self, 
                                         classification_metrics: Dict,
                                         regression_metrics: Dict) -> List[str]:
        """
        Generate clinical recommendations based on model performance.
        
        Args:
            classification_metrics: Classification performance metrics
            regression_metrics: Regression performance metrics
            
        Returns:
            List of clinical recommendations
        """
        recommendations = []
        
        # Classification recommendations
        if classification_metrics:
            macro_f1 = classification_metrics.get('macro_f1', 0)
            
            if macro_f1 < 0.8:
                recommendations.append("Improve abnormality detection accuracy before clinical deployment")
                recommendations.append("Consider additional training data or model architecture modifications")
            
            if macro_f1 < 0.7:
                recommendations.append("Model requires significant improvement - not suitable for clinical use")
                recommendations.append("Investigate class imbalance and data quality issues")
        
        # Regression recommendations
        if regression_metrics:
            stress_mae = regression_metrics.get('stress_mae', float('inf'))
            machine_mae = regression_metrics.get('machine_mae', float('inf'))
            environment_mae = regression_metrics.get('environment_mae', float('inf'))
            
            if stress_mae > 20:
                recommendations.append("Improve stress level assessment accuracy for better clinical utility")
            
            if machine_mae > 25:
                recommendations.append("Enhance machine error detection for reliable quality assessment")
            
            if environment_mae > 25:
                recommendations.append("Improve environmental factor assessment for comprehensive analysis")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Model performance meets clinical standards")
            recommendations.append("Proceed with clinical validation studies")
            recommendations.append("Monitor performance in real-world settings")
        
        return recommendations
    
    def create_evaluation_visualizations(self) -> bool:
        """
        Create comprehensive evaluation visualizations.
        
        Returns:
            True if visualizations created successfully, False otherwise
        """
        try:
            self.logger.info("Creating evaluation visualizations...")
            
            # Create plots directory
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Classification performance plots
            if self.deep_learning_results.get('classification'):
                self._create_classification_plots(plots_dir)
            
            # Regression performance plots
            if self.deep_learning_results.get('regression'):
                self._create_regression_plots(plots_dir)
            
            # Model comparison plots
            if self.comparison_results:
                self._create_comparison_plots(plots_dir)
            
            # Overall performance summary
            if self.deep_learning_results.get('overall'):
                self._create_overall_summary_plot(plots_dir)
            
            self.logger.info(f"Evaluation visualizations saved to {plots_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {str(e)}")
            return False
    
    def _create_classification_plots(self, plots_dir: Path):
        """Create classification performance plots."""
        try:
            classification_metrics = self.deep_learning_results['classification']
            
            # F1 scores by class
            plt.figure(figsize=(12, 6))
            class_names = [f"Class {i}" for i in range(8)]
            f1_scores = [classification_metrics.get(f"abnormality_class_{i}_f1", 0) for i in range(8)]
            
            bars = plt.bar(class_names, f1_scores, color='skyblue', alpha=0.7)
            plt.title('F1 Scores by Abnormality Class', fontsize=14, fontweight='bold')
            plt.ylabel('F1 Score', fontsize=12)
            plt.xlabel('Abnormality Class', fontsize=12)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, f1_scores):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "classification_f1_scores.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # ROC AUC scores by class
            plt.figure(figsize=(12, 6))
            auc_scores = [classification_metrics.get(f"abnormality_class_{i}_auc", 0) for i in range(8)]
            
            bars = plt.bar(class_names, auc_scores, color='lightgreen', alpha=0.7)
            plt.title('ROC AUC Scores by Abnormality Class', fontsize=14, fontweight='bold')
            plt.ylabel('ROC AUC Score', fontsize=12)
            plt.xlabel('Abnormality Class', fontsize=12)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, auc_scores):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "classification_auc_scores.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create classification plots: {str(e)}")
    
    def _create_regression_plots(self, plots_dir: Path):
        """Create regression performance plots."""
        try:
            regression_metrics = self.deep_learning_results['regression']
            
            # MAE by cause type
            plt.figure(figsize=(10, 6))
            cause_names = ['Stress', 'Machine Error', 'Environment']
            mae_scores = [
                regression_metrics.get('stress_mae', 0),
                regression_metrics.get('machine_mae', 0),
                regression_metrics.get('environment_mae', 0)
            ]
            
            bars = plt.bar(cause_names, mae_scores, color=['red', 'orange', 'blue'], alpha=0.7)
            plt.title('Mean Absolute Error by Cause Type', fontsize=14, fontweight='bold')
            plt.ylabel('MAE (%)', fontsize=12)
            plt.xlabel('Cause Type', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, mae_scores):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{score:.1f}%', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "regression_mae_by_cause.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # R² scores by cause type
            plt.figure(figsize=(10, 6))
            r2_scores = [
                regression_metrics.get('stress_r2', 0),
                regression_metrics.get('machine_r2', 0),
                regression_metrics.get('environment_r2', 0)
            ]
            
            bars = plt.bar(cause_names, r2_scores, color=['red', 'orange', 'blue'], alpha=0.7)
            plt.title('R² Scores by Cause Type', fontsize=14, fontweight='bold')
            plt.ylabel('R² Score', fontsize=12)
            plt.xlabel('Cause Type', fontsize=12)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, r2_scores):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "regression_r2_by_cause.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create regression plots: {str(e)}")
    
    def _create_comparison_plots(self, plots_dir: Path):
        """Create model comparison plots."""
        try:
            # Overall MAE comparison
            plt.figure(figsize=(12, 6))
            
            model_names = list(self.comparison_results.keys())
            improvements = [self.comparison_results[model]['overall_improvement'] 
                          for model in model_names]
            
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            bars = plt.bar(model_names, improvements, color=colors, alpha=0.7)
            
            plt.title('Deep Learning vs Baseline Models: Overall MAE Improvement', 
                     fontsize=14, fontweight='bold')
            plt.ylabel('Improvement (%)', fontsize=12)
            plt.xlabel('Baseline Model', fontsize=12)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, improvement in zip(bars, improvements):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., 
                        height + (1 if height > 0 else -1),
                        f'{improvement:.1f}%', ha='center', 
                        va='bottom' if height > 0 else 'top', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "model_comparison_overall.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create comparison plots: {str(e)}")
    
    def _create_overall_summary_plot(self, plots_dir: Path):
        """Create overall performance summary plot."""
        try:
            overall_metrics = self.deep_learning_results['overall']
            
            # Performance scores
            plt.figure(figsize=(10, 6))
            score_names = ['Classification', 'Regression', 'Combined']
            scores = [
                overall_metrics.get('classification_score', 0),
                overall_metrics.get('regression_score', 0),
                overall_metrics.get('combined_score', 0)
            ]
            
            colors = ['skyblue', 'lightgreen', 'gold']
            bars = plt.bar(score_names, scores, color=colors, alpha=0.7)
            
            plt.title('Overall Model Performance Scores', fontsize=14, fontweight='bold')
            plt.ylabel('Score', fontsize=12)
            plt.xlabel('Performance Type', fontsize=12)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "overall_performance_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create overall summary plot: {str(e)}")
    
    def _save_evaluation_results(self, result_type: str, results: Dict):
        """Save evaluation results to file."""
        try:
            result_file = self.output_dir / f"{result_type}_results.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"{result_type} results saved to {result_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save {result_type} results: {str(e)}")
    
    def generate_evaluation_summary(self) -> str:
        """
        Generate a comprehensive evaluation summary.
        
        Returns:
            String containing evaluation summary
        """
        try:
            summary = []
            summary.append("=" * 80)
            summary.append("ECG ABNORMALITY DETECTION MODEL EVALUATION SUMMARY")
            summary.append("=" * 80)
            summary.append("")
            
            if self.deep_learning_results:
                summary.append("DEEP LEARNING MODEL PERFORMANCE:")
                summary.append("-" * 40)
                
                overall = self.deep_learning_results.get('overall', {})
                summary.append(f"Overall Performance Level: {overall.get('performance_level', 'Unknown')}")
                summary.append(f"Combined Score: {overall.get('combined_score', 0):.3f}")
                summary.append(f"Classification Score: {overall.get('classification_score', 0):.3f}")
                summary.append(f"Regression Score: {overall.get('regression_score', 0):.3f}")
                summary.append("")
                
                # Classification details
                if 'classification' in self.deep_learning_results:
                    cls_metrics = self.deep_learning_results['classification']
                    summary.append("Classification Performance:")
                    summary.append(f"  Overall Accuracy: {cls_metrics.get('accuracy', 0):.3f}")
                    summary.append(f"  Macro F1: {cls_metrics.get('macro_f1', 0):.3f}")
                    summary.append(f"  Macro AUC: {cls_metrics.get('macro_auc', 0):.3f}")
                    summary.append("")
                
                # Regression details
                if 'regression' in self.deep_learning_results:
                    reg_metrics = self.deep_learning_results['regression']
                    summary.append("Cause Attribution Performance:")
                    summary.append(f"  Overall MAE: {reg_metrics.get('overall_mae', 0):.1f}%")
                    summary.append(f"  Overall R²: {reg_metrics.get('overall_r2', 0):.3f}")
                    summary.append("")
            
            if self.baseline_results:
                summary.append("BASELINE MODELS PERFORMANCE:")
                summary.append("-" * 40)
                
                for model_name, results in self.baseline_results.items():
                    summary.append(f"{model_name.upper()}:")
                    reg_metrics = results.get('regression', {})
                    summary.append(f"  Overall MAE: {reg_metrics.get('overall_mae', 0):.1f}%")
                    summary.append(f"  Overall R²: {reg_metrics.get('overall_r2', 0):.3f}")
                    summary.append("")
            
            if self.comparison_results:
                summary.append("MODEL COMPARISON:")
                summary.append("-" * 40)
                
                for model_name, comparison in self.comparison_results.items():
                    summary.append(f"{model_name.upper()}:")
                    summary.append(f"  Overall Improvement: {comparison.get('overall_improvement', 0):.1f}%")
                    summary.append(f"  Winner: {comparison.get('overall_winner', 'Unknown')}")
                    summary.append("")
            
            summary.append("=" * 80)
            
            # Save summary to file
            summary_file = self.output_dir / "evaluation_summary.txt"
            with open(summary_file, 'w') as f:
                f.write('\n'.join(summary))
            
            return '\n'.join(summary)
            
        except Exception as e:
            self.logger.error(f"Failed to generate evaluation summary: {str(e)}")
            return "Error generating evaluation summary"


if __name__ == "__main__":
    # Example usage and testing
    print("Testing ECG Model Evaluator...")
    
    # Create sample evaluation data
    np.random.seed(42)
    
    # Sample predictions and labels
    num_samples = 100
    num_classes = 8
    num_causes = 3
    
    # Classification predictions (probabilities)
    abnormality_predictions = np.random.random((num_samples, num_classes))
    abnormality_labels = np.random.randint(0, 2, (num_samples, num_classes))
    
    # Cause attribution predictions (percentages)
    cause_predictions = np.random.uniform(0, 100, (num_samples, num_causes))
    # Normalize to sum to 100%
    cause_predictions = cause_predictions / cause_predictions.sum(axis=1, keepdims=True) * 100
    
    cause_labels = np.random.uniform(0, 100, (num_samples, num_causes))
    cause_labels = cause_labels / cause_labels.sum(axis=1, keepdims=True) * 100
    
    # Create evaluator
    evaluator = ECGEvaluator(output_dir="test_evaluation")
    
    # Evaluate deep learning model
    dl_predictions = {
        'abnormality_predictions': abnormality_predictions,
        'cause_predictions': cause_predictions
    }
    
    ground_truth = {
        'abnormality_labels': abnormality_labels,
        'cause_labels': cause_labels
    }
    
    print("Evaluating deep learning model...")
    dl_results = evaluator.evaluate_deep_learning_model(dl_predictions, ground_truth)
    
    # Create sample baseline results
    baseline_predictions = {
        'random_forest': {
            'cause_predictions': np.random.uniform(0, 100, (num_samples, num_causes))
        },
        'xgboost': {
            'cause_predictions': np.random.uniform(0, 100, (num_samples, num_causes))
        }
    }
    
    # Normalize baseline predictions
    for model_name in baseline_predictions:
        preds = baseline_predictions[model_name]['cause_predictions']
        preds = preds / preds.sum(axis=1, keepdims=True) * 100
        baseline_predictions[model_name]['cause_predictions'] = preds
    
    print("Evaluating baseline models...")
    baseline_results = evaluator.evaluate_baseline_models(baseline_predictions, ground_truth)
    
    # Compare models
    print("Comparing models...")
    comparison_results = evaluator.compare_models()
    
    # Generate clinical report
    print("Generating clinical report...")
    clinical_report = evaluator.generate_clinical_report()
    
    # Create visualizations
    print("Creating visualizations...")
    evaluator.create_evaluation_visualizations()
    
    # Generate summary
    print("Generating evaluation summary...")
    summary = evaluator.generate_evaluation_summary()
    print(summary)
    
    print("\nEvaluation testing completed successfully!")
