"""
helpers/data_generators.py

This module contains a library of functions for generating synthetic datasets
that simulate various scenarios encountered in biotech R&D and manufacturing.

Each function is designed to be a "pure function," meaning its output depends
only on its inputs. The use of a random number generator (RNG) seeded by an
optional 'seed' parameter ensures reproducibility for testing and
consistent demonstrations. This avoids the anti-pattern of hardcoding
np.random.seed() inside functions.

Author: AI Engineering SME
Version: 23.1 (Commercial Grade Refactor)
Date: 2023-10-26
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

# --- Constants for realistic data simulation ---
# Using named constants instead of "magic numbers" improves readability and
# maintainability. It provides context for the default values.
DEFAULT_PROCESS_MEAN = 20.0
DEFAULT_PROCESS_STD = 1.5
DEFAULT_SHIFT_POINT = 75
DEFAULT_SHIFT_MAGNITUDE_STD = 0.8
DEFAULT_SEED = 42

def _get_rng(seed: Optional[int] = DEFAULT_SEED) -> np.random.Generator:
    """Creates a NumPy random number generator instance for reproducibility."""
    return np.random.default_rng(seed)

# ==============================================================================
# SECTION 1: CORE PROCESS & STATISTICAL DATA
# ==============================================================================

def generate_process_data(mean: float, std_dev: float, size: int, seed: Optional[int] = DEFAULT_SEED) -> np.ndarray:
    """
    Generates a simple 1D array of normally distributed process data.

    :param mean: The mean of the normal distribution.
    :param std_dev: The standard deviation of the normal distribution.
    :param size: The number of data points to generate.
    :param seed: An optional seed for the random number generator for reproducibility.
    :return: A NumPy array of generated data.
    """
    rng = _get_rng(seed)
    return rng.normal(mean, std_dev, size)

def generate_nonlinear_data(size: int = 200, seed: Optional[int] = DEFAULT_SEED) -> pd.DataFrame:
    """
    Generates a dataset with a complex, non-linear relationship suitable for
    demonstrating advanced regression model performance over simple linear models.

    :param size: The number of data points to generate.
    :param seed: An optional seed for the random number generator.
    :return: A DataFrame with features and a target variable.
    """
    rng = _get_rng(seed)
    # Factor 1: A primary, influential parameter
    X1 = np.linspace(55, 65, size)
    # Factor 2: A parameter with a quadratic interaction
    X2 = 1.0 + 0.1 * (X1 - 60)**2 + rng.normal(0, 0.5, size)
    # Factor 3: A noise parameter with no real effect
    X3_noise = rng.standard_normal(size) * 5
    # Response variable with complex interactions and noise
    y = 70 - 0.5 * (X1 - 62)**2 + 10 * np.log(X2 + 1) + rng.normal(0, 3, size)
    
    return pd.DataFrame({
        'Annealing_Temp': X1,
        'Enzyme_Conc': X2,
        'Humidity_Noise': X3_noise,
        'On_Target_Rate': y
    })

def generate_control_chart_data(
    mean: float = DEFAULT_PROCESS_MEAN,
    std_dev: float = DEFAULT_PROCESS_STD,
    size: int = 150,
    shift_point: int = DEFAULT_SHIFT_POINT,
    shift_magnitude: float = DEFAULT_SHIFT_MAGNITUDE_STD,
    seed: Optional[int] = DEFAULT_SEED
) -> pd.DataFrame:
    """
    Generates data for control charts, simulating a process that starts in
    control and then experiences a downward shift.

    :param mean: The mean of the in-control process.
    :param std_dev: The standard deviation of the process.
    :param size: The total number of data points.
    :param shift_point: The index at which the process shift occurs.
    :param shift_magnitude: The size of the shift in terms of standard deviations.
    :param seed: An optional seed for the random number generator.
    :return: A DataFrame with Batch ID and a measured value.
    """
    rng = _get_rng(seed)
    in_control = rng.normal(mean, std_dev, shift_point)
    shifted_mean = mean - shift_magnitude * std_dev
    out_of_control = rng.normal(shifted_mean, std_dev, size - shift_point)
    
    return pd.DataFrame({
        'Batch_ID': np.arange(size),
        'Yield_ng': np.concatenate([in_control, out_of_control])
    })

def generate_doe_data(seed: Optional[int] = DEFAULT_SEED) -> pd.DataFrame:
    """
    Generates data for a 2^3 full factorial Design of Experiments, including
    main effects, a two-way interaction, and random noise.

    :param seed: An optional seed for the random number generator.
    :return: A DataFrame with coded factors and the measured response.
    """
    rng = _get_rng(seed)
    factors = [-1, 1]  # Coded units for low and high levels
    data = []
    # Define model coefficients for a realistic response
    BETA_0, BETA_1, BETA_2, BETA_3, BETA_23 = 80, 5, -12, 8, 6
    
    for f1 in factors:
        for f2 in factors:
            for f3 in factors:
                response = (
                    BETA_0 +
                    BETA_1 * f1 +
                    BETA_2 * f2 +
                    BETA_3 * f3 +
                    BETA_23 * f2 * f3 +
                    rng.normal(0, 2.5)  # Add random experimental noise
                )
                data.append([f1, f2, f3, response])
                
    return pd.DataFrame(
        data,
        columns=['Primer_Conc', 'Anneal_Temp', 'PCR_Cycles', 'Library_Yield']
    )

def generate_kano_data(seed: Optional[int] = DEFAULT_SEED) -> pd.DataFrame:
    """
    Generates data to illustrate the Kano model categories: Basic, Performance,
    and Excitement.

    :param seed: An optional seed for the random number generator.
    :return: A DataFrame with functionality, satisfaction, and category.
    """
    rng = _get_rng(seed)
    func = np.linspace(0, 10, 20)
    
    # Basic (Must-be): Logarithmic curve, plateaus quickly
    basic_sat = np.clip(np.log(func + 0.1) * 3 - 8, -10, 0) + rng.normal(0, 0.3, 20)
    basic_sat[func==0] = -10
    
    # Performance: Linear relationship
    perf_sat = np.linspace(-5, 5, 20) + rng.normal(0, 0.8, 20)
    
    # Excitement (Delighter): Exponential curve, takes off late
    excite_sat = np.clip(np.exp(func * 0.4) - 1.5, 0, 10) + rng.normal(0, 0.3, 20)
    excite_sat[func==0] = 0

    df_basic = pd.DataFrame({'functionality': func, 'satisfaction': basic_sat, 'category': 'Basic (Must-be)'})
    df_perf = pd.DataFrame({'functionality': func, 'satisfaction': perf_sat, 'category': 'Performance'})
    df_excite = pd.DataFrame({'functionality': func, 'satisfaction': excite_sat, 'category': 'Excitement (Delighter)'})
    
    return pd.concat([df_basic, df_perf, df_excite], ignore_index=True)

def generate_anova_data(means: list, stds: list, n: int, seed: Optional[int] = DEFAULT_SEED) -> pd.DataFrame:
    """
    Generates data for ANOVA by creating multiple groups with specified means
    and standard deviations.

    :param means: A list of mean values for each group.
    :param stds: A list of standard deviations for each group.
    :param n: The number of samples per group.
    :param seed: An optional seed for the random number generator.
    :return: A DataFrame with measured values and group labels.
    """
    rng = _get_rng(seed)
    data, groups = [], []
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        group_data = rng.normal(mean, std, n)
        data.extend(group_data)
        groups.extend([f'Lot {chr(65+i)}'] * n)
        
    return pd.DataFrame({'Library_Yield': data, 'Reagent_Lot': groups})

def generate_sensor_degradation_data(seed: Optional[int] = DEFAULT_SEED) -> pd.DataFrame:
    """
    Generates time-series data simulating sensor degradation over time, with
    some anomalous dips, suitable for predictive maintenance models.

    :param seed: An optional seed for the random number generator.
    :return: A DataFrame with run number and a sensor reading.
    """
    rng = _get_rng(seed)
    time = np.arange(0, 100)
    degradation = 100 * np.exp(-time * 0.015) + rng.normal(0, 0.5, 100)
    
    # Inject a few sudden failure points
    anomaly_indices = rng.choice(100, 3, replace=False)
    degradation[anomaly_indices] -= rng.normal(5, 1, 3)
    
    return pd.DataFrame({'Run_Number': time, 'Laser_Power_mW': degradation})

# ==============================================================================
# SECTION 2: QUALITATIVE & QMS DATA
# ==============================================================================

def generate_pareto_data() -> pd.DataFrame:
    """Generates static data for a Pareto chart of QC failures."""
    return pd.DataFrame({
        'QC_Failure_Mode': ['Low Library Yield', 'Adapter-Dimer Contamination', 'High Duplication Rate', 'Failed Positive Control', 'Low Q30 Score', 'Sample Mix-up'],
        'Frequency': [45, 22, 11, 6, 4, 2]
    })

def generate_fmea_data() -> pd.DataFrame:
    """Generates a sample FMEA table with calculated Risk Priority Numbers (RPN)."""
    data = [
        {'Failure Mode': 'Reagent Contamination', 'Severity': 10, 'Occurrence': 3, 'Detection': 5},
        {'Failure Mode': 'Incorrect Pipetting Volume', 'Severity': 8, 'Occurrence': 5, 'Detection': 3},
        {'Failure Mode': 'Thermal Cycler Malfunction', 'Severity': 9, 'Occurrence': 2, 'Detection': 7},
        {'Failure Mode': 'Sample Mis-labeling', 'Severity': 10, 'Occurrence': 1, 'Detection': 2}
    ]
    df = pd.DataFrame(data)
    df['RPN'] = df['Severity'] * df['Occurrence'] * df['Detection']
    return df.sort_values('RPN', ascending=False, ignore_index=True)

def generate_vsm_data() -> pd.DataFrame:
    """Generates data for a Value Stream Map, detailing process steps."""
    return pd.DataFrame([
        {"Step": "Accessioning", "CycleTime": 10, "WaitTime": 120, "ValueAdded": True},
        {"Step": "Extraction", "CycleTime": 90, "WaitTime": 30, "ValueAdded": True},
        {"Step": "Lib Prep", "CycleTime": 240, "WaitTime": 1200, "ValueAdded": True},
        {"Step": "Sequencing", "CycleTime": 1440, "WaitTime": 2880, "ValueAdded": True},
        {"Step": "Biofx", "CycleTime": 180, "WaitTime": 60, "ValueAdded": True},
        {"Step": "Reporting", "CycleTime": 30, "WaitTime": 240, "ValueAdded": True}
    ])

def generate_hotelling_data(seed: Optional[int] = DEFAULT_SEED) -> pd.DataFrame:
    """
    Generates multivariate data with an in-control group and an out-of-control
    group, suitable for Hotelling's T-squared chart.

    :param seed: An optional seed for the random number generator.
    :return: A DataFrame with two correlated process parameters.
    """
    rng = _get_rng(seed)
    mean_in, cov_in = [85, 15], [[4, -3], [-3, 4]]
    data_in = rng.multivariate_normal(mean_in, cov_in, 80)
    
    mean_out = [80, 22] # Shifted mean for the outlier group
    data_out = rng.multivariate_normal(mean_out, cov_in, 20)
    
    return pd.DataFrame(np.vstack((data_in, data_out)), columns=['Pct_Mapped', 'Pct_Duplication'])

def generate_rsm_data(seed: Optional[int] = DEFAULT_SEED) -> pd.DataFrame:
    """
    Generates data for a Response Surface Methodology plot, showing an
    optimal region for two factors.

    :param seed: An optional seed for the random number generator.
    :return: A DataFrame with two factors and a response variable.
    """
    rng = _get_rng(seed)
    temp = np.linspace(50, 70, 15)
    conc = np.linspace(1, 2, 15)
    T, C = np.meshgrid(temp, conc)
    
    # A quadratic model with an interaction term
    yield_val = 90 - 0.1*(T-60)**2 - 20*(C-1.5)**2 - 0.5*(T-60)*(C-1.5) + rng.normal(0, 2, T.shape)
    
    return pd.DataFrame({
        'Temperature': T.ravel(),
        'Concentration': C.ravel(),
        'Yield': yield_val.ravel()
    })

def generate_qfd_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates static data for a Quality Function Deployment matrix."""
    customer_reqs = ['High Sensitivity', 'High Specificity', 'Fast Turnaround', 'Low Cost']
    weights = pd.DataFrame({'Importance': [5, 5, 3, 4]}, index=customer_reqs)
    
    tech_chars = ['LOD (VAF %)', 'Specificity (%)', 'Hands-on Time (min)', 'Reagent Cost ($)']
    relationships = np.array([
        [9, 1, 3, 1],
        [1, 9, 1, 1],
        [1, 1, 9, 3],
        [1, 3, 3, 9]
    ])
    rel_df = pd.DataFrame(relationships, index=customer_reqs, columns=tech_chars)
    return weights, rel_df

def generate_dfmea_data() -> pd.DataFrame:
    """Generates a sample Design FMEA table."""
    data = [
        {'Potential Failure Mode': 'Incorrect material selected for sample well', 'Potential Effect': 'Sample Adsorption, low yield', 'Severity': 9, 'Potential Cause': 'Biocompatibility not verified', 'Occurrence': 3, 'Current Controls': 'Material Spec Sheet Review', 'Detection': 6},
        {'Potential Failure Mode': 'Fluidic channel geometry causes bubbles', 'Potential Effect': 'Flow obstruction, assay failure', 'Severity': 10, 'Potential Cause': 'Sharp corners in CAD model', 'Occurrence': 5, 'Current Controls': 'Visual Inspection', 'Detection': 7},
        {'Potential Failure Mode': 'Device housing cracks under stress', 'Potential Effect': 'Leakage, contamination', 'Severity': 7, 'Potential Cause': 'Low-grade polymer used', 'Occurrence': 2, 'Current Controls': 'Drop Test Protocol', 'Detection': 2}
    ]
    df = pd.DataFrame(data)
    df['RPN'] = df['Severity'] * df['Occurrence'] * df['Detection']
    return df.sort_values('RPN', ascending=False, ignore_index=True)

def generate_capa_data() -> pd.DataFrame:
    """Generates sample text data simulating CAPA and deviation logs for NLP."""
    data = {
        "ID": ["CAPA-001", "CAPA-002", "CAPA-003", "CAPA-004", "CAPA-005", "CAPA-006", "CAPA-007"],
        "Description": [
            'Batch 2023-45 showed low yield. Investigation found the enzyme from Lot A was degraded due to improper storage in Freezer B.',
            'Contamination detected in negative controls for run 2023-11-02. Root cause traced to aerosolization from adjacent high-titer sample.',
            'Run 2023-11-05 failed due to thermal cycler block temperature overshoot. The cycler requires recalibration.',
            'System flagged low Q30 scores for samples 1-8. Pipetting error during library prep suspected, inconsistent volumes.',
            'Positive control failed for batch 2023-48. The control was stored at the wrong temperature, leading to DNA degradation.',
            'Unexpected peaks observed in chromatography. Re-running sample with fresh mobile phase resolved the issue. Suspect mobile phase degradation.',
            'Pipetting robot failed to dispense into well A3. Service log indicates a clogged nozzle. Required preventative maintenance was overdue.'
        ]
    }
    return pd.DataFrame(data)

def generate_adverse_event_data() -> pd.DataFrame:
    """Generates synthetic adverse event narrative data for NLP clustering."""
    data = [
        'Patient experienced severe rash after starting treatment',
        'Acute liver enzyme elevation noted in patient bloodwork',
        'Patient reported mild headache and fatigue',
        'Nausea and dizziness reported within 1 hour of dosage',
        'Anaphylactic shock occurred; required epinephrine',
        'Minor injection site irritation observed'
    ]
    counts = [15, 10, 40, 25, 2, 8]
    descriptions = np.repeat(data, counts)
    
    return pd.DataFrame({
        "event_id": range(1, len(descriptions) + 1),
        "description": descriptions
    })

def generate_risk_signal_data(seed: Optional[int] = DEFAULT_SEED) -> pd.DataFrame:
    """
    Generates multivariate manufacturing data from different sources, including
    anomalous events, for risk signal detection via clustering.

    :param seed: An optional seed for the random number generator.
    :return: A DataFrame with process parameters and source labels.
    """
    rng = _get_rng(seed)
    # Line A: Normal operating conditions
    c1 = pd.DataFrame(rng.multivariate_normal([70, 2], [[5, -3], [-3, 3]], 30), columns=['Temp_C', 'Pressure_psi'])
    c1['Source'] = 'Manufacturing Line A'
    # Line B: Different operating conditions
    c2 = pd.DataFrame(rng.multivariate_normal([50, 5], [[4, 2], [2, 4]], 50), columns=['Temp_C', 'Pressure_psi'])
    c2['Source'] = 'Manufacturing Line B'
    # Anomalous Events
    outliers = pd.DataFrame([[85, 1], [40, 10], [75, 8]], columns=['Temp_C', 'Pressure_psi'])
    outliers['Source'] = 'Anomalous Events'
    
    return pd.concat([c1, c2, outliers], ignore_index=True)

def generate_pccp_data(seed: Optional[int] = DEFAULT_SEED) -> pd.DataFrame:
    """
    Generates data simulating AI/ML model performance degradation over time,
    as would be monitored under a Pre-determined Change Control Plan (PCCP).

    :param seed: An optional seed for the random number generator.
    :return: A DataFrame with time and model performance metric.
    """
    rng = _get_rng(seed)
    time = np.arange(100)
    # Performance degrades quadratically with some noise
    performance = 0.95 - 0.0001 * time - 0.000005 * time**2 + rng.normal(0, 0.005, 100)
    # A sudden drop due to a change in data distribution
    performance[70:] -= 0.05
    
    return pd.DataFrame({'Deployment_Day': time, 'Model_AUC': performance})

def generate_validation_data(seed: Optional[int] = DEFAULT_SEED) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Generates summary performance metrics and corresponding bootstrap samples
    for demonstrating validation with confidence intervals.

    :param seed: An optional seed for the random number generator.
    :return: A tuple containing a DataFrame of point estimates and a dictionary
             of bootstrapped sample distributions.
    """
    rng = _get_rng(seed)
    data = {'Metric': ['Accuracy', 'Sensitivity', 'Specificity'], 'Value': [0.95, 0.92, 0.97]}
    df = pd.DataFrame(data)
    
    bootstrap_samples = {
        'Accuracy': rng.normal(0.95, 0.02, 1000),
        'Sensitivity': rng.normal(0.92, 0.04, 1000),
        'Specificity': rng.normal(0.97, 0.015, 1000)
    }
    return df, bootstrap_samples
