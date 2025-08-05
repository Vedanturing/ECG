# ECG Clinical & Technical Research Notes

## 12-Lead ECG Structure

### Core Waveform Components

#### P Wave
- **Definition**: Atrial depolarization wave
- **Duration**: 80-120ms
- **Amplitude**: 0.05-0.25mV
- **Morphology**: Smooth, rounded, positive deflection
- **Clinical significance**: Indicates atrial activity and conduction

#### QRS Complex
- **Definition**: Ventricular depolarization
- **Duration**: 60-100ms (narrow) to >120ms (wide)
- **Amplitude**: 0.5-3.0mV (varies by lead)
- **Components**:
  - Q wave: Initial negative deflection (if present)
  - R wave: First positive deflection
  - S wave: Negative deflection following R wave
- **Clinical significance**: Primary indicator of ventricular function

#### T Wave
- **Definition**: Ventricular repolarization
- **Duration**: 160-200ms
- **Amplitude**: 0.1-0.5mV
- **Morphology**: Asymmetric, rounded
- **Clinical significance**: Repolarization abnormalities indicate ischemia/injury

#### ST Segment
- **Definition**: Period between QRS end and T wave start
- **Duration**: 80-120ms
- **Baseline**: Should be isoelectric (flat)
- **Clinical significance**: ST elevation/depression indicates myocardial ischemia

#### RR Interval
- **Definition**: Time between consecutive R peaks
- **Normal range**: 600-1000ms (60-100 BPM)
- **Variability**: Heart Rate Variability (HRV) analysis
- **Clinical significance**: Autonomic nervous system function

### Sampling Frequency & Ranges

#### Standard Sampling Rates
- **Clinical ECG**: 250-500 Hz
- **Research ECG**: 1000-2000 Hz
- **High-fidelity**: 4000-10000 Hz

#### Amplitude Ranges
- **P wave**: 0.05-0.25mV
- **QRS complex**: 0.5-3.0mV
- **T wave**: 0.1-0.5mV
- **Baseline noise**: <0.05mV

## Deviation & Error Causes

### Physiological Deviations

#### Stress-Induced Changes
- **Heart Rate Variability (HRV)**:
  - **Low HRV**: Stress indicator (RMSSD < 20ms)
  - **High RMSSD**: Parasympathetic dominance, anxiety marker
  - **LF/HF ratio**: Sympathetic/parasympathetic balance
- **T-wave flattening**: Stress-related repolarization changes
- **ST-segment depression**: Exercise or emotional stress

#### Anxiety Markers
- **Elevated heart rate**: >100 BPM at rest
- **Reduced HRV**: Decreased autonomic flexibility
- **Irregular rhythms**: Premature beats, sinus arrhythmia

### Environmental Interference

#### Electromagnetic (EM) Interference
- **Power line noise**: 50/60 Hz interference
- **Radio frequency**: Mobile devices, WiFi
- **Electromagnetic fields**: MRI, electrical equipment
- **Mitigation**: Shielding, filtering, grounding

#### Electrode Issues
- **Misplacement**: Incorrect anatomical positioning
- **Poor contact**: Insufficient skin preparation
- **Movement artifacts**: Patient motion during recording
- **Dried gel**: Reduced signal quality

### Machine-Based Errors

#### Flatlines
- **Causes**: Disconnected leads, amplifier failure
- **Detection**: Zero variance over time window
- **Impact**: Complete signal loss

#### Amplitude Clipping
- **Causes**: Signal exceeds ADC range
- **Detection**: Saturation at maximum/minimum values
- **Impact**: Signal distortion, loss of detail

#### Dropped Leads
- **Causes**: Hardware failure, connection issues
- **Detection**: Missing data channels
- **Impact**: Incomplete 12-lead analysis

## Signal Quality Assessment

### Noise Metrics
- **Signal-to-Noise Ratio (SNR)**: >20dB for clinical use
- **Baseline wander**: <0.1Hz frequency components
- **Muscle artifact**: High-frequency noise >100Hz
- **Motion artifact**: Irregular baseline shifts

### Quality Indicators
- **R-peak detection confidence**: >95% accuracy
- **Signal continuity**: <1% missing data
- **Amplitude stability**: <10% variation
- **Frequency response**: Flat within 0.5-40Hz band

## Clinical Interpretation Guidelines

### Normal Variants
- **Respiratory sinus arrhythmia**: Normal HRV with breathing
- **Early repolarization**: Benign ST elevation in young adults
- **Athlete's heart**: Sinus bradycardia, voltage criteria

### Pathological Patterns
- **ST elevation**: Acute myocardial infarction
- **ST depression**: Ischemia, strain
- **T-wave inversion**: Ischemia, hypertrophy
- **Q waves**: Previous infarction, hypertrophy

## Technical Considerations for AI Processing

### Preprocessing Requirements
- **Baseline correction**: Remove low-frequency drift
- **Noise filtering**: Bandpass 0.5-40Hz
- **R-peak detection**: Robust algorithm selection
- **Signal segmentation**: Fixed-length windows

### Feature Engineering
- **Morphological features**: Wave amplitudes, durations
- **Temporal features**: RR intervals, HRV metrics
- **Frequency features**: Power spectral density
- **Statistical features**: Mean, variance, skewness

### Validation Standards
- **Clinical correlation**: Expert cardiologist review
- **Cross-validation**: Multiple dataset testing
- **Performance metrics**: Sensitivity, specificity, AUC
- **Interpretability**: Explainable AI requirements 