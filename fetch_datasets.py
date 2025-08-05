#!/usr/bin/env python3
"""
ECG Dataset Fetcher and Preprocessor
Downloads and parses multiple ECG datasets from PhysioNet
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
from tqdm import tqdm
import zipfile
import tarfile

# ECG processing libraries
try:
    import wfdb
    import neurokit2 as nk
    from biosppy.signals import ecg
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install wfdb neurokit2 biosppy")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ECGDatasetFetcher:
    """Main class for downloading and preprocessing ECG datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'mitbih': {
                'url': 'https://physionet.org/files/mitdb/1.0.0/mitdb-1.0.0.zip',
                'local_dir': self.data_dir / 'mitbih',
                'format': 'dat',
                'description': 'MIT-BIH Arrhythmia Database'
            },
            'ptb': {
                'url': 'https://physionet.org/files/ptbdb/1.0.0/ptbdb-1.0.0.zip',
                'local_dir': self.data_dir / 'ptb',
                'format': 'dat',
                'description': 'PTB Diagnostic ECG Database'
            },
            'incart': {
                'url': 'https://physionet.org/files/incartdb/1.0.0/incartdb-1.0.0.zip',
                'local_dir': self.data_dir / 'incart',
                'format': 'dat',
                'description': 'INCART 12-lead Arrhythmia Database'
            }
        }
        
        # Stress drivers dataset (simulated for demo)
        self.datasets['stress_drivers'] = {
            'url': None,  # Will generate synthetic data
            'local_dir': self.data_dir / 'stress_drivers',
            'format': 'csv',
            'description': 'Stress Recognition in Drivers (Synthetic)'
        }
    
    def download_file(self, url: str, local_path: Path, chunk_size: int = 8192) -> bool:
        """Download file with progress bar and resume capability"""
        try:
            # Check if file already exists
            if local_path.exists():
                logger.info(f"File already exists: {local_path}")
                return True
            
            # Create directory if it doesn't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_dir: Path) -> bool:
        """Extract zip or tar archive"""
        try:
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif archive_path.suffix in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                logger.error(f"Unsupported archive format: {archive_path.suffix}")
                return False
            
            logger.info(f"Successfully extracted: {archive_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {e}")
            return False
    
    def generate_synthetic_stress_data(self, output_dir: Path, num_subjects: int = 50) -> None:
        """Generate synthetic stress recognition data for demonstration"""
        logger.info("Generating synthetic stress recognition data...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for subject_id in range(num_subjects):
            # Generate 5-minute ECG data at 500 Hz
            duration = 300  # 5 minutes
            fs = 500  # 500 Hz
            t = np.linspace(0, duration, int(duration * fs))
            
            # Base heart rate with stress variation
            stress_level = np.random.choice(['low', 'medium', 'high'])
            base_hr = {'low': 60, 'medium': 80, 'high': 100}[stress_level]
            
            # Generate realistic ECG signal
            signal = self._generate_realistic_ecg(t, base_hr, stress_level, fs)
            
            # Add noise based on stress level
            noise_level = {'low': 0.01, 'medium': 0.02, 'high': 0.03}[stress_level]
            signal += np.random.normal(0, noise_level, len(signal))
            
            # Save data
            data = {
                'timestamp': t.tolist(),
                'ecg_signal': signal.tolist(),
                'sampling_rate': fs,
                'stress_level': stress_level,
                'subject_id': f'stress_subject_{subject_id:03d}',
                'metadata': {
                    'age': np.random.randint(25, 65),
                    'gender': np.random.choice(['M', 'F']),
                    'driving_condition': np.random.choice(['normal', 'traffic', 'highway']),
                    'time_of_day': np.random.choice(['morning', 'afternoon', 'evening'])
                }
            }
            
            output_file = output_dir / f'stress_subject_{subject_id:03d}.json'
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Generated {num_subjects} synthetic stress datasets")
    
    def _generate_realistic_ecg(self, t: np.ndarray, base_hr: float, stress_level: str, fs: int) -> np.ndarray:
        """Generate realistic ECG signal with stress-induced variations"""
        signal = np.zeros_like(t)
        
        # Heart rate variability based on stress
        hrv_std = {'low': 0.05, 'medium': 0.1, 'high': 0.15}[stress_level]
        
        # Generate R-peak times with HRV
        rr_intervals = np.random.normal(60/base_hr, hrv_std, int(len(t) * base_hr / 60))
        rr_intervals = np.clip(rr_intervals, 0.3, 2.0)  # Reasonable RR intervals
        
        r_peaks = np.cumsum(rr_intervals)
        r_peaks = r_peaks[r_peaks < t[-1]]
        
        # Generate ECG components
        for r_peak in r_peaks:
            # P wave
            p_start = r_peak - 0.2
            p_end = r_peak - 0.08
            p_mask = (t >= p_start) & (t <= p_end)
            signal[p_mask] += 0.15 * np.sin(np.pi * (t[p_mask] - p_start) / (p_end - p_start))
            
            # QRS complex
            qrs_start = r_peak - 0.08
            qrs_end = r_peak + 0.08
            qrs_mask = (t >= qrs_start) & (t <= qrs_end)
            signal[qrs_mask] += 1.5 * np.exp(-((t[qrs_mask] - r_peak) / 0.02)**2)
            
            # T wave
            t_start = r_peak + 0.08
            t_end = r_peak + 0.3
            t_mask = (t >= t_start) & (t <= t_end)
            signal[t_mask] += 0.3 * np.exp(-((t[t_mask] - (t_start + t_end)/2) / 0.05)**2)
        
        return signal
    
    def parse_wfdb_record(self, record_path: Path) -> Dict:
        """Parse WFDB record and extract metadata"""
        try:
            # Read record
            record = wfdb.rdrecord(str(record_path))
            annotations = wfdb.rdann(str(record_path), 'atr')
            
            # Extract signal data
            signal_data = record.p_signal
            sampling_rate = record.fs
            signal_names = record.sig_name
            
            # Extract metadata
            metadata = {
                'sampling_rate': sampling_rate,
                'signal_names': signal_names,
                'signal_length': len(signal_data),
                'duration_minutes': len(signal_data) / (sampling_rate * 60),
                'num_leads': len(signal_names),
                'record_name': record_path.name
            }
            
            # Add annotation data if available
            if hasattr(annotations, 'sample') and len(annotations.sample) > 0:
                metadata['annotations'] = {
                    'sample': annotations.sample.tolist(),
                    'symbol': annotations.symbol,
                    'aux_note': annotations.aux_note if hasattr(annotations, 'aux_note') else []
                }
            
            return {
                'signal_data': signal_data,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to parse record {record_path}: {e}")
            return None
    
    def process_dataset(self, dataset_name: str) -> None:
        """Process a specific dataset"""
        dataset_config = self.datasets[dataset_name]
        local_dir = dataset_config['local_dir']
        
        logger.info(f"Processing dataset: {dataset_name}")
        
        if dataset_name == 'stress_drivers':
            # Generate synthetic data
            self.generate_synthetic_stress_data(local_dir)
            return
        
        # Download dataset
        if dataset_config['url']:
            archive_path = local_dir / f"{dataset_name}.zip"
            if not self.download_file(dataset_config['url'], archive_path):
                logger.error(f"Failed to download {dataset_name}")
                return
            
            # Extract archive
            if not self.extract_archive(archive_path, local_dir):
                logger.error(f"Failed to extract {dataset_name}")
                return
        
        # Find and process WFDB records
        self._process_wfdb_records(local_dir, dataset_name)
    
    def _process_wfdb_records(self, data_dir: Path, dataset_name: str) -> None:
        """Process all WFDB records in a directory"""
        # Find all .dat files
        dat_files = list(data_dir.rglob("*.dat"))
        
        if not dat_files:
            logger.warning(f"No .dat files found in {data_dir}")
            return
        
        logger.info(f"Found {len(dat_files)} records to process")
        
        processed_records = []
        
        for dat_file in tqdm(dat_files, desc=f"Processing {dataset_name}"):
            # Get record path (without .dat extension)
            record_path = dat_file.with_suffix('')
            
            # Parse record
            parsed_data = self.parse_wfdb_record(record_path)
            
            if parsed_data:
                # Save processed data
                output_file = record_path.with_suffix('.npy')
                np.save(output_file, parsed_data['signal_data'])
                
                # Save metadata
                metadata_file = record_path.with_suffix('.json')
                with open(metadata_file, 'w') as f:
                    json.dump(parsed_data['metadata'], f, indent=2)
                
                processed_records.append({
                    'record_name': record_path.name,
                    'signal_file': str(output_file),
                    'metadata_file': str(metadata_file)
                })
        
        # Save dataset summary
        summary_file = data_dir / f"{dataset_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'dataset_name': dataset_name,
                'total_records': len(processed_records),
                'processed_records': processed_records,
                'processing_timestamp': pd.Timestamp.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Processed {len(processed_records)} records for {dataset_name}")
    
    def fetch_all_datasets(self) -> None:
        """Download and process all datasets"""
        logger.info("Starting ECG dataset collection...")
        
        for dataset_name in self.datasets.keys():
            try:
                self.process_dataset(dataset_name)
            except Exception as e:
                logger.error(f"Failed to process {dataset_name}: {e}")
                continue
        
        logger.info("Dataset collection completed!")
        
        # Generate overall summary
        self._generate_overall_summary()
    
    def _generate_overall_summary(self) -> None:
        """Generate summary of all collected datasets"""
        summary = {
            'total_datasets': len(self.datasets),
            'datasets': {},
            'collection_timestamp': pd.Timestamp.now().isoformat()
        }
        
        for dataset_name, config in self.datasets.items():
            local_dir = config['local_dir']
            summary_file = local_dir / f"{dataset_name}_summary.json"
            
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    dataset_summary = json.load(f)
                    summary['datasets'][dataset_name] = {
                        'description': config['description'],
                        'format': config['format'],
                        'total_records': dataset_summary.get('total_records', 0),
                        'status': 'completed'
                    }
            else:
                summary['datasets'][dataset_name] = {
                    'description': config['description'],
                    'format': config['format'],
                    'total_records': 0,
                    'status': 'failed'
                }
        
        # Save overall summary
        summary_file = self.data_dir / "overall_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Overall summary saved to {summary_file}")

def main():
    """Main execution function"""
    print("🚀 ECG Dataset Collection Pipeline")
    print("=" * 50)
    
    # Initialize fetcher
    fetcher = ECGDatasetFetcher()
    
    # Fetch all datasets
    fetcher.fetch_all_datasets()
    
    print("\n✅ Dataset collection completed!")
    print("📁 Check the 'data' directory for downloaded datasets")
    print("📋 Review 'data/overall_summary.json' for collection statistics")

if __name__ == "__main__":
    main() 