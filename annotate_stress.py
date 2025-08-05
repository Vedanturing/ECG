#!/usr/bin/env python3
"""
ECG Stress Annotation and Noise Detection
Analyzes ECG waveforms for stress indicators and signal quality issues
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# ECG processing libraries
try:
    import neurokit2 as nk
    from biosppy.signals import ecg as biosppy_ecg
    import wfdb
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install neurokit2 biosppy wfdb")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stress_annotation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ECGStressAnnotator:
    """Main class for ECG stress analysis and annotation"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
        # Stress detection thresholds
        self.stress_thresholds = {
            'rmssd_low': 20.0,      # Low HRV = high stress
            'rmssd_high': 100.0,    # High HRV = anxiety
            'hr_high': 100,         # High heart rate
            'hr_low': 50,           # Low heart rate
            'lf_hf_ratio_high': 2.0, # Sympathetic dominance
            'snr_min': 20.0,        # Minimum SNR in dB
            'flatline_threshold': 0.01, # Variance threshold for flatlines
            'clipping_threshold': 0.95  # Amplitude clipping threshold
        }
        
        # HRV analysis parameters
        self.hrv_params = {
            'sampling_rate': 500,
            'window_size': 300,  # 5 minutes at 500 Hz
            'overlap': 0.5,      # 50% overlap between windows
            'min_rr': 0.3,       # Minimum RR interval (200 BPM)
            'max_rr': 2.0        # Maximum RR interval (30 BPM)
        }
    
    def load_ecg_data(self, file_path: Path) -> Dict[str, Any]:
        """Load ECG data from various formats"""
        try:
            if file_path.suffix == '.npy':
                signal_data = np.load(file_path)
                metadata_file = file_path.with_suffix('.json')
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {
                        'sampling_rate': 500,
                        'signal_names': ['ECG'],
                        'record_name': file_path.stem
                    }
                
                return {'signal_data': signal_data, 'metadata': metadata}
            
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                signal_data = np.array(data['ecg_signal'])
                metadata = {
                    'sampling_rate': data['sampling_rate'],
                    'signal_names': ['ECG'],
                    'record_name': file_path.stem,
                    'stress_level': data.get('stress_level', 'unknown'),
                    'subject_id': data.get('subject_id', file_path.stem),
                    'metadata': data.get('metadata', {})
                }
                
                return {'signal_data': signal_data, 'metadata': metadata}
            
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def detect_r_peaks(self, ecg_signal: np.ndarray, sampling_rate: int) -> Tuple[np.ndarray, Dict]:
        """Detect R-peaks using multiple algorithms for robustness"""
        try:
            # Method 1: NeuroKit2
            try:
                signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
                r_peaks_nk = info['ECG_R_Peaks']
                nk_confidence = len(r_peaks_nk) / (len(ecg_signal) / sampling_rate * 60)
            except:
                r_peaks_nk = np.array([])
                nk_confidence = 0
            
            # Method 2: BioSPPy
            try:
                biosppy_result = biosppy_ecg.ecg(ecg_signal, sampling_rate=sampling_rate, show=False)
                r_peaks_biosppy = biosppy_result['rpeaks']
                biosppy_confidence = len(r_peaks_biosppy) / (len(ecg_signal) / sampling_rate * 60)
            except:
                r_peaks_biosppy = np.array([])
                biosppy_confidence = 0
            
            # Choose the method with higher confidence
            if nk_confidence > biosppy_confidence and len(r_peaks_nk) > 0:
                r_peaks = r_peaks_nk
                method = 'neurokit2'
                confidence = nk_confidence
            elif len(r_peaks_biosppy) > 0:
                r_peaks = r_peaks_biosppy
                method = 'biosppy'
                confidence = biosppy_confidence
            else:
                r_peaks = self._simple_r_peak_detection(ecg_signal, sampling_rate)
                method = 'threshold'
                confidence = len(r_peaks) / (len(ecg_signal) / sampling_rate * 60)
            
            detection_info = {
                'method': method,
                'confidence': confidence,
                'num_peaks': len(r_peaks),
                'expected_peaks': len(ecg_signal) / sampling_rate * 60
            }
            
            return r_peaks, detection_info
            
        except Exception as e:
            logger.error(f"R-peak detection failed: {e}")
            return np.array([]), {'method': 'failed', 'confidence': 0, 'num_peaks': 0}
    
    def _simple_r_peak_detection(self, ecg_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Simple threshold-based R-peak detection as fallback"""
        try:
            # Bandpass filter
            nyquist = sampling_rate / 2
            low = 5 / nyquist
            high = 15 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, ecg_signal)
            
            # Find peaks above threshold
            threshold = np.percentile(filtered_signal, 90)
            peaks, _ = signal.find_peaks(filtered_signal, height=threshold, distance=int(0.5 * sampling_rate))
            
            return peaks
            
        except Exception as e:
            logger.error(f"Simple R-peak detection failed: {e}")
            return np.array([])
    
    def compute_hrv_features(self, r_peaks: np.ndarray, sampling_rate: int) -> Dict[str, float]:
        """Compute HRV features from R-peak locations"""
        try:
            if len(r_peaks) < 10:
                return self._get_default_hrv_features()
            
            # Convert to RR intervals
            rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # Convert to ms
            
            # Filter RR intervals
            rr_intervals = rr_intervals[
                (rr_intervals >= self.hrv_params['min_rr'] * 1000) & 
                (rr_intervals <= self.hrv_params['max_rr'] * 1000)
            ]
            
            if len(rr_intervals) < 5:
                return self._get_default_hrv_features()
            
            # Time domain features
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
            pnn50 = nn50 / len(rr_intervals) * 100 if len(rr_intervals) > 0 else 0
            
            # Frequency domain features
            try:
                # Interpolate RR intervals to regular time grid
                t_rr = np.cumsum(rr_intervals) / 1000  # Convert to seconds
                t_regular = np.arange(t_rr[0], t_rr[-1], 1/sampling_rate)
                rr_interpolated = np.interp(t_regular, t_rr, rr_intervals)
                
                # Compute power spectral density
                freqs, psd = signal.welch(rr_interpolated, fs=sampling_rate, nperseg=min(256, len(rr_interpolated)//4))
                
                # Define frequency bands
                vlf_mask = (freqs >= 0.0033) & (freqs < 0.04)
                lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                hf_mask = (freqs >= 0.15) & (freqs < 0.4)
                
                vlf_power = np.trapz(psd[vlf_mask], freqs[vlf_mask])
                lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
                hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])
                
                total_power = vlf_power + lf_power + hf_power
                lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
                lf_nu = lf_power / (lf_power + hf_power) * 100 if (lf_power + hf_power) > 0 else 0
                hf_nu = hf_power / (lf_power + hf_power) * 100 if (lf_power + hf_power) > 0 else 0
                
            except Exception as e:
                logger.warning(f"Frequency domain analysis failed: {e}")
                vlf_power = lf_power = hf_power = total_power = 0
                lf_hf_ratio = lf_nu = hf_nu = 0
            
            return {
                'mean_rr': float(mean_rr),
                'std_rr': float(std_rr),
                'rmssd': float(rmssd),
                'nn50': int(nn50),
                'pnn50': float(pnn50),
                'vlf_power': float(vlf_power),
                'lf_power': float(lf_power),
                'hf_power': float(hf_power),
                'total_power': float(total_power),
                'lf_hf_ratio': float(lf_hf_ratio),
                'lf_nu': float(lf_nu),
                'hf_nu': float(hf_nu),
                'heart_rate': float(60000 / mean_rr if mean_rr > 0 else 0)
            }
            
        except Exception as e:
            logger.error(f"HRV feature computation failed: {e}")
            return self._get_default_hrv_features()
    
    def _get_default_hrv_features(self) -> Dict[str, float]:
        """Return default HRV features when computation fails"""
        return {
            'mean_rr': 0.0, 'std_rr': 0.0, 'rmssd': 0.0, 'nn50': 0, 'pnn50': 0.0,
            'vlf_power': 0.0, 'lf_power': 0.0, 'hf_power': 0.0, 'total_power': 0.0,
            'lf_hf_ratio': 0.0, 'lf_nu': 0.0, 'hf_nu': 0.0, 'heart_rate': 0.0
        }
    
    def detect_signal_quality_issues(self, ecg_signal: np.ndarray, sampling_rate: int) -> Dict[str, Any]:
        """Detect various signal quality issues"""
        try:
            issues = {
                'flatlines': [], 'clipping': [], 'noise': [], 'baseline_wander': [],
                'overall_quality_score': 1.0
            }
            
            # Check for flatlines
            window_size = int(2 * sampling_rate)  # 2-second windows
            for i in range(0, len(ecg_signal) - window_size, window_size):
                window = ecg_signal[i:i+window_size]
                variance = np.var(window)
                
                if variance < self.stress_thresholds['flatline_threshold']:
                    issues['flatlines'].append({
                        'start_time': i / sampling_rate,
                        'end_time': (i + window_size) / sampling_rate,
                        'duration': window_size / sampling_rate,
                        'variance': variance
                    })
            
            # Check for amplitude clipping
            max_val = np.max(ecg_signal)
            min_val = np.min(ecg_signal)
            clipping_threshold = self.stress_thresholds['clipping_threshold']
            
            if np.any(ecg_signal >= max_val * clipping_threshold) or np.any(ecg_signal <= min_val * clipping_threshold):
                issues['clipping'].append({
                    'max_amplitude': float(max_val),
                    'min_amplitude': float(min_val),
                    'clipping_percentage': float(np.sum((ecg_signal >= max_val * clipping_threshold) | 
                                                       (ecg_signal <= min_val * clipping_threshold)) / len(ecg_signal) * 100)
                })
            
            # Estimate noise level
            try:
                nyquist = sampling_rate / 2
                high = 40 / nyquist
                b, a = signal.butter(4, high, btype='high')
                noise_signal = signal.filtfilt(b, a, ecg_signal)
                
                noise_rms = np.sqrt(np.mean(noise_signal ** 2))
                signal_rms = np.sqrt(np.mean(ecg_signal ** 2))
                snr_db = 20 * np.log10(signal_rms / noise_rms) if noise_rms > 0 else 100
                
                if snr_db < self.stress_thresholds['snr_min']:
                    issues['noise'].append({
                        'snr_db': float(snr_db),
                        'noise_rms': float(noise_rms),
                        'signal_rms': float(signal_rms)
                    })
                
            except Exception as e:
                logger.warning(f"Noise estimation failed: {e}")
            
            # Calculate overall quality score
            quality_penalties = 0.0
            if issues['flatlines']: quality_penalties += 0.3
            if issues['clipping']: quality_penalties += 0.2
            if issues['noise']: quality_penalties += 0.3
            if issues['baseline_wander']: quality_penalties += 0.2
            
            issues['overall_quality_score'] = max(0.0, 1.0 - quality_penalties)
            
            return issues
            
        except Exception as e:
            logger.error(f"Signal quality detection failed: {e}")
            return {'flatlines': [], 'clipping': [], 'noise': [], 'baseline_wander': [], 'overall_quality_score': 0.0}
    
    def classify_stress_level(self, hrv_features: Dict[str, float]) -> Dict[str, float]:
        """Classify stress level based on HRV features"""
        try:
            stress_scores = {'stress': 0.0, 'machine': 0.0, 'env': 0.0}
            
            rmssd = hrv_features.get('rmssd', 0)
            heart_rate = hrv_features.get('heart_rate', 0)
            lf_hf_ratio = hrv_features.get('lf_hf_ratio', 0)
            
            # Stress indicators
            if rmssd < self.stress_thresholds['rmssd_low']:
                stress_scores['stress'] += 0.4  # Low HRV = high stress
            elif rmssd > self.stress_thresholds['rmssd_high']:
                stress_scores['stress'] += 0.3  # High HRV = anxiety
            
            if heart_rate > self.stress_thresholds['hr_high']:
                stress_scores['stress'] += 0.3  # High heart rate
            elif heart_rate < self.stress_thresholds['hr_low']:
                stress_scores['stress'] += 0.2  # Low heart rate
            
            if lf_hf_ratio > self.stress_thresholds['lf_hf_ratio_high']:
                stress_scores['stress'] += 0.2  # Sympathetic dominance
            
            # Normalize scores
            total_score = sum(stress_scores.values())
            if total_score > 0:
                stress_scores = {k: v / total_score for k, v in stress_scores.items()}
            
            return stress_scores
            
        except Exception as e:
            logger.error(f"Stress classification failed: {e}")
            return {'stress': 0.0, 'machine': 0.0, 'env': 0.0}
    
    def analyze_ecg_file(self, file_path: Path) -> Dict[str, Any]:
        """Complete analysis of a single ECG file"""
        try:
            logger.info(f"Analyzing: {file_path}")
            
            # Load data
            data = self.load_ecg_data(file_path)
            if data is None:
                return None
            
            signal_data = data['signal_data']
            metadata = data['metadata']
            
            # Use lead II if available, otherwise first lead
            if signal_data.ndim > 1 and len(signal_data.shape) > 1:
                if 'II' in metadata.get('signal_names', []):
                    lead_idx = metadata['signal_names'].index('II')
                    ecg_signal = signal_data[:, lead_idx]
                else:
                    ecg_signal = signal_data[:, 0]  # Use first lead
            else:
                ecg_signal = signal_data
            
            sampling_rate = metadata['sampling_rate']
            
            # Detect R-peaks
            r_peaks, detection_info = self.detect_r_peaks(ecg_signal, sampling_rate)
            
            # Compute HRV features
            hrv_features = self.compute_hrv_features(r_peaks, sampling_rate)
            
            # Detect signal quality issues
            quality_issues = self.detect_signal_quality_issues(ecg_signal, sampling_rate)
            
            # Classify stress level
            stress_classification = self.classify_stress_level(hrv_features)
            
            # Generate annotations
            annotations = self._generate_annotations(
                ecg_signal, r_peaks, hrv_features, quality_issues, 
                stress_classification, sampling_rate
            )
            
            # Compile results
            results = {
                'subject_id': metadata.get('subject_id', metadata.get('record_name', file_path.stem)),
                'lead': 'II' if 'II' in metadata.get('signal_names', []) else metadata.get('signal_names', ['ECG'])[0],
                'sampling_rate': sampling_rate,
                'signal_length': len(ecg_signal),
                'duration_seconds': len(ecg_signal) / sampling_rate,
                'r_peak_detection': detection_info,
                'hrv_features': hrv_features,
                'signal_quality': quality_issues,
                'stress_classification': stress_classification,
                'annotations': annotations,
                'metadata': metadata
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed for {file_path}: {e}")
            return None
    
    def _generate_annotations(self, ecg_signal: np.ndarray, r_peaks: np.ndarray, 
                            hrv_features: Dict, quality_issues: Dict, 
                            stress_classification: Dict, sampling_rate: int) -> List[Dict]:
        """Generate detailed annotations for the ECG signal"""
        annotations = []
        
        try:
            # Add stress-related annotations
            if stress_classification['stress'] > 0.5:
                annotations.append({
                    'timestamp': 0.0,
                    'event': 'High Stress Detected',
                    'cause_label': stress_classification,
                    'confidence': stress_classification['stress'],
                    'hrv_rmssd': hrv_features.get('rmssd', 0),
                    'heart_rate': hrv_features.get('heart_rate', 0)
                })
            
            # Add quality issue annotations
            for flatline in quality_issues['flatlines']:
                annotations.append({
                    'timestamp': flatline['start_time'],
                    'event': 'Flatline Detected',
                    'cause_label': {'stress': 0.0, 'machine': 0.8, 'env': 0.2},
                    'confidence': 0.9,
                    'duration': flatline['duration'],
                    'variance': flatline['variance']
                })
            
            for clipping in quality_issues['clipping']:
                annotations.append({
                    'timestamp': 0.0,
                    'event': 'Amplitude Clipping',
                    'cause_label': {'stress': 0.0, 'machine': 0.9, 'env': 0.1},
                    'confidence': 0.8,
                    'clipping_percentage': clipping['clipping_percentage']
                })
            
            for noise in quality_issues['noise']:
                annotations.append({
                    'timestamp': 0.0,
                    'event': 'High Noise Level',
                    'cause_label': {'stress': 0.1, 'machine': 0.2, 'env': 0.7},
                    'confidence': 0.7,
                    'snr_db': noise['snr_db']
                })
            
        except Exception as e:
            logger.error(f"Annotation generation failed: {e}")
        
        return annotations
    
    def process_dataset(self, dataset_name: str) -> None:
        """Process all files in a dataset directory"""
        dataset_dir = self.data_dir / dataset_name
        
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return
        
        # Find all ECG files
        ecg_files = []
        for ext in ['.npy', '.json', '.dat']:
            ecg_files.extend(list(dataset_dir.rglob(f"*{ext}")))
        
        if not ecg_files:
            logger.warning(f"No ECG files found in {dataset_dir}")
            return
        
        logger.info(f"Found {len(ecg_files)} files to analyze in {dataset_name}")
        
        # Process each file
        results = []
        for file_path in ecg_files:
            try:
                result = self.analyze_ecg_file(file_path)
                if result:
                    results.append(result)
                    
                    # Save individual result
                    output_file = file_path.with_suffix('.annotation.json')
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                        
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        # Save dataset summary
        if results:
            summary_file = dataset_dir / f"{dataset_name}_stress_analysis.json"
            with open(summary_file, 'w') as f:
                json.dump({
                    'dataset_name': dataset_name,
                    'total_files': len(ecg_files),
                    'processed_files': len(results),
                    'analysis_timestamp': pd.Timestamp.now().isoformat(),
                    'results': results
                }, f, indent=2)
            
            logger.info(f"Stress analysis completed for {dataset_name}: {len(results)} files processed")
    
    def analyze_all_datasets(self) -> None:
        """Analyze all datasets in the data directory"""
        logger.info("Starting ECG stress analysis...")
        
        # Find all dataset directories
        dataset_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            try:
                self.process_dataset(dataset_name)
            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_name}: {e}")
                continue
        
        logger.info("ECG stress analysis completed!")

def main():
    """Main execution function"""
    print("🧠 ECG Stress Analysis Pipeline")
    print("=" * 50)
    
    # Initialize annotator
    annotator = ECGStressAnnotator()
    
    # Analyze all datasets
    annotator.analyze_all_datasets()
    
    print("\n✅ Stress analysis completed!")
    print("📁 Check individual .annotation.json files for detailed results")

if __name__ == "__main__":
    main() 