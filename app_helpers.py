# app_helpers.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import graphviz
import shap

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import gaussian_kde, f_oneway
from scipy.stats import f as f_dist
from typing import List, Dict, Tuple

# ==============================================================================
# SECTION 1: VISUAL & STYLING CONFIGURATION
# ==============================================================================
COLORS = {
    "primary": "#0072B2", "secondary": "#009E73", "accent": "#D55E00",
    "neutral_yellow": "#F0E442", "neutral_pink": "#CC79A7", "background": "#F8F9FA",
    "text": "#212529", "light_gray": "#DEE2E6", "dark_gray": "#495057",
    "success": "#28A745", "warning": "#FFC107", "danger": "#DC3545"
}

def get_custom_css() -> str:
    """
    Returns the custom CSS string for a professional, clean Streamlit app theme.
    Uses robust `data-testid` selectors for stability across Streamlit versions.
    """
    return f"""
    <style>
        /* Main app styling */
        div[data-testid="stAppViewContainer"] > main {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
        }}
        
        /* Headers */
        h1, h2 {{
            color: {COLORS['dark_gray']};
            border-bottom: 2px solid {COLORS['light_gray']};
            padding-bottom: 10px;
        }}
        h3 {{ color: {COLORS['primary']}; }}
        h4, h5 {{ color: {COLORS['dark_gray']}; }}
        
        /* Style for containers created with st.container(border=True) */
        div[data-testid="stBlock"] {{
            border: 1px solid {COLORS['light_gray']};
            border-radius: 0.5rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
        }}
        
        /* Button styling */
        button[data-testid="stButton"] > button {{
            border-radius: 0.5rem;
        }}
    </style>
    """

def hex_to_rgba(h: str, a: float) -> str:
    """Converts a hex color string to an rgba string for Plotly compatibility."""
    return f"rgba({int(h[1:3], 16)}, {int(h[3:5], 16)}, {int(h[5:7], 16)}, {a})"

# ==============================================================================
# SECTION 2: SYNTHETIC DATA GENERATORS (UNCHANGED)
# ==============================================================================
def generate_process_data(mean: float, std_dev: float, size: int) -> np.ndarray:
    return np.random.normal(mean, std_dev, size)

def generate_nonlinear_data(size: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    X1 = np.linspace(55, 65, size)
    X2 = 1.0 + 0.1 * (X1 - 60)**2 + np.random.normal(0, 0.5, size)
    X3_noise = np.random.randn(size) * 5
    y = 70 - 0.5 * (X1 - 62)**2 + 10 * np.log(X2 + 1) + np.random.normal(0, 3, size)
    return pd.DataFrame({'Annealing_Temp': X1, 'Enzyme_Conc': X2, 'Humidity_Noise': X3_noise, 'On_Target_Rate': y})

def generate_control_chart_data(mean: float = 20.0, std_dev: float = 1.5, size: int = 150, shift_point: int = 75, shift_magnitude: float = 0.8) -> pd.DataFrame:
    np.random.seed(42)
    in_control = np.random.normal(mean, std_dev, shift_point)
    out_of_control = np.random.normal(mean - shift_magnitude * std_dev, std_dev, size - shift_point)
    return pd.DataFrame({'Batch_ID': np.arange(size), 'Yield_ng': np.concatenate([in_control, out_of_control])})

def generate_doe_data() -> pd.DataFrame:
    np.random.seed(42)
    factors = [-1, 1]
    data = []
    for f1 in factors:
        for f2 in factors:
            for f3 in factors:
                response = 80 + 5*f1 - 12*f2 + 8*f3 + 6*f2*f3 + np.random.randn() * 2.5
                data.append([f1, f2, f3, response])
    return pd.DataFrame(data, columns=['Primer_Conc', 'Anneal_Temp', 'PCR_Cycles', 'Library_Yield'])

def generate_kano_data() -> pd.DataFrame:
    np.random.seed(42)
    func = np.linspace(0, 10, 20)
    basic_sat = np.clip(np.log(func + 0.1) * 3 - 8, -10, 0) + np.random.normal(0, 0.3, 20)
    basic_sat[func==0] = -10
    perf_sat = np.linspace(-5, 5, 20) + np.random.normal(0, 0.8, 20)
    excite_sat = np.clip(np.exp(func * 0.4) - 1.5, 0, 10) + np.random.normal(0, 0.3, 20)
    excite_sat[func==0] = 0
    df_basic = pd.DataFrame({'functionality': func, 'satisfaction': basic_sat, 'category': 'Basic (Must-be)'})
    df_perf = pd.DataFrame({'functionality': func, 'satisfaction': perf_sat, 'category': 'Performance'})
    df_excite = pd.DataFrame({'functionality': func, 'satisfaction': excite_sat, 'category': 'Excitement (Delighter)'})
    return pd.concat([df_basic, df_perf, df_excite], ignore_index=True)

def generate_anova_data(means: list, stds: list, n: int) -> pd.DataFrame:
    data, groups = [], []
    for i, (mean, std) in enumerate(zip(means, stds)):
        data.extend(np.random.normal(mean, std, n))
        groups.extend([f'Lot {chr(65+i)}'] * n)
    return pd.DataFrame({'Library_Yield': data, 'Reagent_Lot': groups})

def generate_sensor_degradation_data() -> pd.DataFrame:
    np.random.seed(42)
    time = np.arange(0, 100)
    degradation = 100 * np.exp(-time * 0.015) + np.random.normal(0, 0.5, 100)
    degradation[np.random.choice(100, 3, replace=False)] -= np.random.normal(5, 1, 3)
    return pd.DataFrame({'Run_Number': time, 'Laser_Power_mW': degradation})

def generate_pareto_data() -> pd.DataFrame:
    return pd.DataFrame({
        'QC_Failure_Mode': ['Low Library Yield', 'Adapter-Dimer Contamination', 'High Duplication Rate', 'Failed Positive Control', 'Low Q30 Score', 'Sample Mix-up'],
        'Frequency': [45, 22, 11, 6, 4, 2]
    })

def generate_fmea_data() -> pd.DataFrame:
    return pd.DataFrame([
        {'Failure Mode': 'Reagent Contamination', 'Severity': 10, 'Occurrence': 3, 'Detection': 5},
        {'Failure Mode': 'Incorrect Pipetting Volume', 'Severity': 8, 'Occurrence': 5, 'Detection': 3},
        {'Failure Mode': 'Thermal Cycler Malfunction', 'Severity': 9, 'Occurrence': 2, 'Detection': 7},
        {'Failure Mode': 'Sample Mis-labeling', 'Severity': 10, 'Occurrence': 1, 'Detection': 2}
    ]).assign(RPN=lambda df: df.Severity * df.Occurrence * df.Detection).sort_values('RPN', ascending=False)

def generate_vsm_data() -> pd.DataFrame:
    return pd.DataFrame([
        {"Step": "Accessioning", "CycleTime": 10, "WaitTime": 120, "ValueAdded": True},
        {"Step": "Extraction", "CycleTime": 90, "WaitTime": 30, "ValueAdded": True},
        {"Step": "Lib Prep", "CycleTime": 240, "WaitTime": 1200, "ValueAdded": True},
        {"Step": "Sequencing", "CycleTime": 1440, "WaitTime": 2880, "ValueAdded": True},
        {"Step": "Biofx", "CycleTime": 180, "WaitTime": 60, "ValueAdded": True},
        {"Step": "Reporting", "CycleTime": 30, "WaitTime": 240, "ValueAdded": True}
    ])

def generate_hotelling_data() -> pd.DataFrame:
    np.random.seed(42)
    mean_in, cov_in = [85, 15], [[4, -3], [-3, 4]]
    data_in = np.random.multivariate_normal(mean_in, cov_in, 80)
    mean_out = [80, 22]
    data_out = np.random.multivariate_normal(mean_out, cov_in, 20)
    return pd.DataFrame(np.vstack((data_in, data_out)), columns=['Pct_Mapped', 'Pct_Duplication'])

def generate_causal_data(size: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    x1 = np.random.uniform(5, 15, size=size)
    x2 = 0.5 * x1 + np.random.normal(size=size) * 0.2
    x3 = -0.8 * x1 + np.random.normal(size=size) * 0.2
    x4 = 2 * x2 - 1.5 * x3 + np.random.normal(size=size) * 0.1
    return pd.DataFrame({'Ligation_Time': x1, 'Adapter_Dimer_Pct': x2, 'DNA_Input_ng': x3, 'Library_Yield_ng': x4})

# ==============================================================================
# SECTION 3: VISUALIZATION HELPERS (IMPROVED RENDERING)
# ==============================================================================

def plot_project_charter_visual() -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, fillcolor='white', line_width=0)
    fig.add_annotation(x=0.5, y=0.92, text="<b>Assay Development Plan: Liquid Biopsy for CRC</b>", showarrow=False, font=dict(size=22, color=COLORS['primary']))
    kpis = {"Analytical Sensitivity": ("LOD < 0.1% VAF", COLORS['success']),
            "Clinical Specificity": ("> 99.5%", COLORS['success']),
            "Turnaround Time": ("< 5 days", COLORS['success'])}
    for i, (k, (v, c)) in enumerate(kpis.items()):
        fig.add_annotation(x=0.2+i*0.3, y=0.75, text=f"<b>{k}</b>", showarrow=False, font=dict(size=14, color=COLORS['dark_gray']))
        fig.add_annotation(x=0.2+i*0.3, y=0.65, text=v, showarrow=False, font=dict(size=20, color=c))
    statements = {
        "Problem Statement": (0.05, 0.45, "Colorectal Cancer (CRC) requires earlier detection methods. Current methods are invasive or lack sensitivity for early-stage disease.", "left"),
        "Goal Statement": (0.95, 0.45, "Develop and validate a cfDNA-based NGS assay for early-stage CRC detection with >90% sensitivity at 99.5% specificity.", "right")
    }
    for title, (x, y, txt, anchor) in statements.items():
        fig.add_annotation(x=x, y=y, text=f"<b>{title}</b>", showarrow=False, align=anchor, xanchor=anchor, font_size=16)
        fig.add_annotation(x=x, y=y-0.1, text=txt, showarrow=False, align=anchor, xanchor=anchor, yanchor='top', width=400)
    fig.update_layout(xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[0, 1]), plot_bgcolor='white', margin=dict(t=20, b=20, l=20, r=20), height=350)
    return fig

def plot_sipoc_visual() -> go.Figure:
    header_values = ['<b>👤<br>Suppliers</b>', '<b>🧬<br>Inputs</b>', '<b>⚙️<br>Process</b>', '<b>📊<br>Outputs</b>', '<b>⚕️<br>Customers</b>']
    cell_values = [
        ['• Reagent Vendors<br>• Instrument Mfr.<br>• LIMS Provider'],
        ['• Patient Blood Sample<br>• Reagent Kits<br>• Lab Protocol (SOP)'],
        ['1. Sample Prep<br>2. Library Prep<br>3. NGS Sequencing<br>4. Bioinformatics<br>5. Reporting'],
        ['• VCF File<br>• QC Metrics Report<br>• Clinical Report'],
        ['• Oncologists<br>• Patients<br>• Pharma Partners']
    ]
    fig = go.Figure(data=[go.Table(
        header=dict(values=header_values, line_color=COLORS['light_gray'], fill_color=COLORS['light_gray'], align='center', font=dict(color=COLORS['primary'], size=14)),
        cells=dict(values=cell_values, line_color=COLORS['light_gray'], fill_color='white', align='left', font_size=12, height=150))
    ])
    fig.update_layout(title_text="<b>SIPOC Diagram:</b> NGS Assay Workflow", margin=dict(l=10, r=10, t=50, b=10))
    return fig

def plot_ctq_tree_visual() -> graphviz.Digraph:
    """
    FIX: Reworked for readability. Uses light fills and dark text for high contrast.
    """
    dot = graphviz.Digraph(comment='CTQ Tree')
    dot.attr('node', shape='box', style='rounded,filled', fontname="helvetica", fontcolor=COLORS['text'])
    dot.attr('edge', color=COLORS['dark_gray'])

    dot.node('Need', 'Clinician Need:\n"Reliable Early CRC Detection"', 
             fillcolor=hex_to_rgba(COLORS['accent'], 0.1), color=COLORS['accent'])
    
    dot.node('Driver1', 'High Sensitivity', 
             fillcolor=hex_to_rgba(COLORS['primary'], 0.1), color=COLORS['primary'])
    dot.node('Driver2', 'High Specificity', 
             fillcolor=hex_to_rgba(COLORS['primary'], 0.1), color=COLORS['primary'])
    dot.node('Driver3', 'Fast Turnaround', 
             fillcolor=hex_to_rgba(COLORS['primary'], 0.1), color=COLORS['primary'])

    dot.edge('Need', 'Driver1'); dot.edge('Need', 'Driver2'); dot.edge('Need', 'Driver3')

    dot.node('CTQ1', 'CTQ:\nLOD < 0.1% VAF', 
             fillcolor=hex_to_rgba(COLORS['secondary'], 0.1), color=COLORS['secondary'])
    dot.node('CTQ2', 'CTQ:\nAnalytical Specificity > 99.5%', 
             fillcolor=hex_to_rgba(COLORS['secondary'], 0.1), color=COLORS['secondary'])
    dot.node('CTQ3', 'CTQ:\nSample-to-Report < 5 days', 
             fillcolor=hex_to_rgba(COLORS['secondary'], 0.1), color=COLORS['secondary'])

    dot.edge('Driver1', 'CTQ1'); dot.edge('Driver2', 'CTQ2'); dot.edge('Driver3', 'CTQ3')
    return dot

def plot_causal_discovery_visual() -> graphviz.Digraph:
    """
    FIX: Reworked for readability. Uses light fills and dark text for high contrast.
    """
    dot = graphviz.Digraph(comment='Causal Graph', graph_attr={'rankdir': 'LR', 'splines': 'spline'})
    dot.attr('node', shape='box', style='rounded,filled', fontname="helvetica", fontcolor=COLORS['text'])
    dot.attr('edge', color=COLORS['dark_gray'], penwidth='1.5', fontcolor=COLORS['dark_gray'])
    
    dot.node('ReagentLot', 'Reagent<br>Lot', fillcolor=hex_to_rgba(COLORS['primary'], 0.1), color=COLORS['primary'])
    dot.node('DNAnq', 'DNA<br>Input (ng)', fillcolor=hex_to_rgba(COLORS['primary'], 0.1), color=COLORS['primary'])
    dot.node('LigationTime', 'Ligation<br>Time', fillcolor=hex_to_rgba(COLORS['secondary'], 0.1), color=COLORS['secondary'])
    dot.node('AdapterDimer', 'Adapter-Dimer<br>Formation (%)', 
             fillcolor=hex_to_rgba(COLORS['accent'], 0.15), color=COLORS['accent'], penwidth='2')
    
    dot.edge('ReagentLot', 'LigationTime'); dot.edge('DNAnq', 'LigationTime')
    dot.edge('LigationTime', 'AdapterDimer', label=" Drives")
    return dot

def plot_kano_visual() -> go.Figure:
    df = generate_kano_data()
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=10, fillcolor=hex_to_rgba(COLORS['success'], 0.1), line_width=0, layer='below')
    fig.add_shape(type="rect", x0=0, y0=-10, x1=10, y1=0, fillcolor=hex_to_rgba(COLORS['danger'], 0.1), line_width=0, layer='below')
    colors = {'Basic (Must-be)': COLORS['accent'], 'Performance': COLORS['primary'], 'Excitement (Delighter)': COLORS['secondary']}
    for cat, color in colors.items():
        subset = df[df['category'] == cat]
        fig.add_trace(go.Scatter(x=subset['functionality'], y=subset['satisfaction'], mode='lines', name=cat, line=dict(color=color, width=4)))
    fig.add_annotation(x=8, y=8, text="<b>Excitement</b><br>e.g., Detects new<br>resistance mutation", showarrow=True, arrowhead=1, ax=-50, ay=-40, font_color=COLORS['secondary'])
    fig.add_annotation(x=8, y=4, text="<b>Performance</b><br>e.g., VAF quantification<br>accuracy", showarrow=True, arrowhead=1, ax=0, ay=-40, font_color=COLORS['primary'])
    fig.add_annotation(x=8, y=-8, text="<b>Basic</b><br>e.g., Detects known<br>KRAS hotspot", showarrow=True, arrowhead=1, ax=0, ay=40, font_color=COLORS['accent'])
    fig.update_layout(title='<b>Kano Model:</b> Prioritizing Diagnostic Features',
                      xaxis_title='Feature Performance / Implementation →', yaxis_title='← Clinician Dissatisfaction ... Satisfaction →',
                      plot_bgcolor='white', legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'))
    return fig

def plot_voc_treemap() -> go.Figure:
    data = {'Category': ['Biomarkers', 'Biomarkers', 'Methodology', 'Methodology', 'Performance', 'Performance'],
            'Topic': ['EGFR Variants', 'KRAS Hotspots', 'ddPCR', 'Shallow WGS', 'LOD <0.1%', 'Specificity >99%'],
            'Count': [180, 150, 90, 60, 250, 210], 'Sentiment': [0.5, 0.4, -0.2, -0.4, 0.8, 0.7]}
    df = pd.DataFrame(data)
    fig = px.treemap(df, path=[px.Constant("All Publications"), 'Category', 'Topic'], values='Count', color='Sentiment',
                     color_continuous_scale='RdBu_r', color_continuous_midpoint=0, custom_data=['Sentiment'])
    fig.update_traces(textinfo='label+value', hovertemplate='<b>%{label}</b><br>Mentions: %{value}<br>Avg. Sentiment: %{customdata[0]:.2f}')
    fig.update_layout(title_text="<b>NLP on Literature:</b> Biomarker & Method Landscape", margin=dict(t=50, l=10, r=10, b=10))
    return fig

def plot_gage_rr_variance_components() -> go.Figure:
    df = pd.DataFrame({'Source': ['% Contribution'], 'Assay Variation': [92], 'Repeatability (Sequencer)': [5], 'Reproducibility (Operator)': [3]})
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df['Source'], x=df['Assay Variation'], name='Assay Variation', orientation='h', marker_color=COLORS['primary']))
    fig.add_trace(go.Bar(y=df['Source'], x=df['Repeatability (Sequencer)'], name='Repeatability (Sequencer)', orientation='h', marker_color=COLORS['warning']))
    fig.add_trace(go.Bar(y=df['Source'], x=df['Reproducibility (Operator)'], name='Reproducibility (Operator)', orientation='h', marker_color=COLORS['accent']))
    fig.update_layout(barmode='stack', title='<b>Gage R&R:</b> Assay vs. System Variation',
                      xaxis=dict(ticksuffix='%', range=[0, 100]), yaxis_visible=False, plot_bgcolor='white',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def plot_process_mining_graph() -> graphviz.Digraph:
    """
    FIX: Reworked for readability. Uses light fills and dark text for high contrast.
    """
    dot = graphviz.Digraph(comment='Process Mining', graph_attr={'rankdir': 'LR', 'splines': 'true'})
    dot.attr('node', shape='box', style='rounded,filled', fontname="Helvetica", fontcolor=COLORS['text'])
    dot.attr('edge', color=COLORS['dark_gray'], fontname="Helvetica", fontsize="10", fontcolor=COLORS['dark_gray'])
    
    dot.node('start', 'Sample\nReceived', shape='circle', style='filled', 
             fillcolor=hex_to_rgba(COLORS['success'], 0.1), color=COLORS['success'])
    dot.node('end', 'Report\nSent', shape='doublecircle', style='filled', 
             fillcolor=hex_to_rgba(COLORS['dark_gray'], 0.1), color=COLORS['dark_gray'])
    
    node_fill = hex_to_rgba(COLORS['primary'], 0.1)
    dot.node('A', 'DNA Extraction', fillcolor=node_fill, color=COLORS['primary'])
    dot.node('B', 'Library Prep', fillcolor=node_fill, color=COLORS['primary'])
    dot.node('C', 'Sequencing', fillcolor=node_fill, color=COLORS['primary'])
    dot.node('D', 'Bioinformatics', fillcolor=node_fill, color=COLORS['primary'])
    dot.node('E', 'QC Fail:\nRe-Prep', style='rounded,filled', 
             fillcolor=hex_to_rgba(COLORS['danger'], 0.15), color=COLORS['danger'])
    
    dot.edge('start', 'A', label='20 Samples'); dot.edge('A', 'B', label='20 Samples')
    dot.edge('B', 'C', label='18 Samples\nAvg. 5h', penwidth='3.0')
    dot.edge('C', 'D', label='18 Samples\nAvg. 26h', penwidth='3.5')
    dot.edge('D', 'end', label='18 Samples\nAvg. 4h')
    dot.edge('B', 'E', label='2 Samples (10%)\nLow Yield', color=COLORS['danger'], penwidth='1.5')
    dot.edge('E', 'B', label='Avg. 8h Delay', color=COLORS['danger'], penwidth='1.5', constraint='false')
    return dot

def plot_vsm() -> go.Figure:
    df = generate_vsm_data()
    total_lead_time = (df['CycleTime'] + df['WaitTime']).sum()
    va_time = df[df['ValueAdded']]['CycleTime'].sum()
    pce = (va_time / total_lead_time) * 100 if total_lead_time > 0 else 0
    fig = go.Figure()
    current_pos = 0
    for _, row in df.iterrows():
        cycle_pct = row['CycleTime'] / total_lead_time * 100
        wait_pct = row['WaitTime'] / total_lead_time * 100
        fig.add_shape(type="rect", x0=current_pos, x1=current_pos + cycle_pct, y0=1, y1=2,
                      fillcolor=COLORS['secondary'] if row['ValueAdded'] else COLORS['danger'], line_color=COLORS['dark_gray'])
        fig.add_annotation(x=current_pos + cycle_pct / 2, y=1.5, text=f"{row['Step']}<br>{row['CycleTime']/60:.1f}h", showarrow=False, font=dict(color='white'))
        current_pos += cycle_pct
        if row['WaitTime'] > 0:
            fig.add_shape(type="rect", x0=current_pos, x1=current_pos + wait_pct, y0=0, y1=1,
                          fillcolor=COLORS['warning'], line_color=COLORS['accent'], opacity=0.7)
            if wait_pct > 5:
                fig.add_annotation(x=current_pos + wait_pct / 2, y=0.5, text=f"{row['WaitTime']/60:.1f}h wait", showarrow=False)
            current_pos += wait_pct
    fig.update_layout(title=f"<b>Value Stream Map (Normalized):</b> Total TAT: {total_lead_time/1440:.1f} days | PCE: {pce:.1f}%",
                      xaxis=dict(title="Percentage of Total Lead Time", showgrid=False, range=[0, 100], ticksuffix="%"),
                      yaxis=dict(visible=False), plot_bgcolor='white', margin=dict(l=20, r=20, t=50, b=20), height=300)
    return fig

def plot_capability_analysis_pro(data: np.ndarray, lsl: float, usl: float) -> Tuple[go.Figure, float, float]:
    mean, std = np.mean(data), np.std(data, ddof=1)
    if std == 0: return go.Figure(), 0, 0
    cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
    cp = (usl - lsl) / (6 * std)
    x_range = np.linspace(min(lsl, data.min()) - 2 * std, max(usl, data.max()) + 2 * std, 500)
    try:
        kde_y = gaussian_kde(data)(x_range)
    except np.linalg.LinAlgError:
        kde_y = np.zeros_like(x_range)
        
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Histogram(x=data, name='Assay Output', marker_color=COLORS['primary'], opacity=0.7), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_range, y=kde_y, mode='lines', name='KDE of Output', line=dict(color=COLORS['accent'], width=3)), secondary_y=True)
    fig.add_vline(x=lsl, line=dict(color=COLORS['danger'], width=2, dash='dash'), name="LSL")
    fig.add_vline(x=usl, line=dict(color=COLORS['danger'], width=2, dash='dash'), name="USL")
    fig.add_vline(x=mean, line=dict(color=COLORS['dark_gray'], width=2, dash='dot'), name="Mean")
    fig.update_layout(title_text=f"<b>Assay Capability:</b> Performance vs. Specification",
                      xaxis_title="Assay Metric (e.g., Signal-to-Noise)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), plot_bgcolor='white')
    fig.update_yaxes(title_text="Count", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Probability Density", secondary_y=True, showgrid=False)
    return fig, cp, cpk

def plot_capability_metrics(cp: float, cpk: float) -> Tuple[go.Figure, go.Figure]:
    """
    FIX: Reworked to produce cleaner gauges. Titles are now handled in the page layout.
    """
    target = 1.33
    def get_gauge_color(v): return COLORS['danger'] if v < 1.0 else (COLORS['warning'] if v < target else COLORS['success'])
    
    fig_cp = go.Figure(go.Indicator(
        mode="gauge+number", value=round(cp, 2),
        gauge={'axis': {'range': [0, 2.5]}, 'bar': {'color': get_gauge_color(cp)},
               'steps': [{'range': [0, 1.0], 'color': hex_to_rgba(COLORS['danger'], 0.2)},
                         {'range': [1.0, target], 'color': hex_to_rgba(COLORS['warning'], 0.2)},
                         {'range': [target, 2.5], 'color': hex_to_rgba(COLORS['success'], 0.2)}]}
    ))
    fig_cp.update_layout(height=200, margin=dict(l=30, r=30, t=10, b=10))

    fig_cpk = go.Figure(go.Indicator(
        mode="gauge+number", value=round(cpk, 2),
        gauge={'axis': {'range': [0, 2.5]}, 'bar': {'color': get_gauge_color(cpk)},
               'steps': [{'range': [0, 1.0], 'color': hex_to_rgba(COLORS['danger'], 0.2)},
                         {'range': [1.0, target], 'color': hex_to_rgba(COLORS['warning'], 0.2)},
                         {'range': [target, 2.5], 'color': hex_to_rgba(COLORS['success'], 0.2)}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.9, 'value': target}}
    ))
    fig_cpk.update_layout(height=200, margin=dict(l=30, r=30, t=10, b=10))
    return fig_cp, fig_cpk

def plot_fishbone_diagram() -> graphviz.Digraph:
    """
    FIX: Reworked for readability. Uses light fills and dark text for high contrast.
    """
    dot = graphviz.Digraph(engine='neato', graph_attr={'splines': 'line'})
    dot.node('Effect', 'Low Library Yield', shape='box', style='filled', 
             fillcolor=hex_to_rgba(COLORS['danger'], 0.15), fontcolor=COLORS['text'], color=COLORS['danger'])
    
    cats = ['Reagents', 'Equipment', 'Method', 'Technician', 'Sample']
    for cat in cats:
        dot.node(cat, cat, shape='none', fontcolor=COLORS['primary'])
        dot.edge(cat, 'Effect', arrowhead='none', color=COLORS['dark_gray'])
    
    dot.node('c1', 'Enzyme Inactivity', shape='none', fontcolor=COLORS['dark_gray'])
    dot.edge('c1', 'Reagents', arrowhead='none')
    dot.node('c2', 'Pipette Out of Cal', shape='none', fontcolor=COLORS['dark_gray'])
    dot.edge('c2', 'Equipment', arrowhead='none')
    dot.node('c3', 'Incorrect Incubation Time', shape='none', fontcolor=COLORS['dark_gray'])
    dot.edge('c3', 'Method', arrowhead='none')
    dot.node('c4', 'Low DNA Input', shape='none', fontcolor=COLORS['dark_gray'])
    dot.edge('c4', 'Sample', arrowhead='none')
    return dot

def plot_shap_summary(model: RandomForestRegressor, X: pd.DataFrame) -> go.Figure:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    fig = go.Figure()
    for i, feature in enumerate(X.columns):
        y_jitter = np.random.uniform(-0.25, 0.25, len(shap_values))
        y_pos = np.full(len(shap_values), i) + y_jitter
        fig.add_trace(go.Scatter(
            x=shap_values.values[:, i], y=y_pos, mode='markers',
            marker=dict(color=shap_values.data[:, i], colorscale='RdBu_r', showscale=(i == 0),
                        colorbar=dict(title="Feature Value<br>High / Low", x=1.02, y=0.5, len=0.75),
                        symbol='circle', size=6, opacity=0.7),
            hoverinfo='text',
            hovertext=[f'<b>{feature}</b><br>Value: {val:.2f}<br>SHAP: {shap_val:.2f}'
                       for val, shap_val in zip(shap_values.data[:, i], shap_values.values[:, i])],
            showlegend=False))
    fig.update_layout(
        title="<b>XAI with SHAP:</b> Parameter Impact on On-Target Rate",
        xaxis_title="SHAP Value (Impact on Model Output)",
        yaxis=dict(tickmode='array', tickvals=list(range(len(X.columns))),
                   ticktext=[col.replace('_', ' ') for col in X.columns],
                   showgrid=True, gridcolor=COLORS['light_gray']),
        plot_bgcolor='white', margin=dict(l=150))
    return fig

def plot_pareto_chart() -> go.Figure:
    df = generate_pareto_data().sort_values('Frequency', ascending=False)
    df['Cumulative Percentage'] = df['Frequency'].cumsum() / df['Frequency'].sum() * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df['QC_Failure_Mode'], y=df['Frequency'], name='Failure Count', marker_color=COLORS['primary']), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['QC_Failure_Mode'], y=df['Cumulative Percentage'], name='Cumulative %', mode='lines+markers', line_color=COLORS['accent']), secondary_y=True)
    fig.add_hline(y=80, line=dict(color=COLORS['dark_gray'], dash='dot'), secondary_y=True)
    fig.update_layout(title_text="<b>Pareto Chart:</b> Identifying Top QC Failure Modes", xaxis_title="QC Failure Mode",
                      plot_bgcolor='white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Frequency", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Percentage", secondary_y=True, range=[0, 101], ticksuffix='%')
    return fig

def plot_anova_groups(df: pd.DataFrame) -> Tuple[go.Figure, float]:
    groups = df['Reagent_Lot'].unique()
    fig = go.Figure()
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['neutral_pink']]
    for i, group in enumerate(groups):
        fig.add_trace(go.Box(y=df[df['Reagent_Lot'] == group]['Library_Yield'], name=group, marker_color=colors[i % len(colors)]))
    group_data = [df[df['Reagent_Lot'] == g]['Library_Yield'] for g in groups]
    p_val = 1.0
    if len(group_data) > 1 and all(len(g) > 1 for g in group_data):
        _, p_val = f_oneway(*group_data)
    fig.update_layout(title=f'<b>ANOVA:</b> Comparing Reagent Lot Performance (p-value: {p_val:.4f})',
                      yaxis_title='Library Yield (ng/µL)', plot_bgcolor='white', showlegend=False)
    return fig, p_val

def plot_permutation_test(df: pd.DataFrame, n_permutations: int = 1000) -> go.Figure:
    groups = df['Reagent_Lot'].unique()
    if len(groups) < 2: return go.Figure(layout=dict(title="Need at least two groups for permutation test."))
    g1_data = df[df['Reagent_Lot'] == groups[0]]['Library_Yield'].dropna()
    g2_data = df[df['Reagent_Lot'] == groups[1]]['Library_Yield'].dropna()
    observed_diff = g1_data.mean() - g2_data.mean()
    concat_data = np.concatenate([g1_data, g2_data])
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(concat_data)
        perm_diffs.append(concat_data[:len(g1_data)].mean() - concat_data[len(g1_data):].mean())
    p_val = (np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) + 1) / (n_permutations + 1)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=perm_diffs, name='Permuted Differences', marker_color=COLORS['light_gray']))
    fig.add_vline(x=observed_diff, line=dict(color=COLORS['accent'], width=3, dash='dash'), name=f'Observed Diff ({observed_diff:.2f})')
    fig.update_layout(title=f'<b>Permutation Test:</b> {n_permutations} Shuffles (p-value: {p_val:.4f})',
                      xaxis_title=f'Difference in Mean Yield ({groups[0]} vs {groups[1]})', yaxis_title='Frequency', plot_bgcolor='white')
    return fig

def train_and_plot_regression_models(df: pd.DataFrame) -> Tuple[go.Figure, RandomForestRegressor, pd.DataFrame]:
    X, y = df.drop(columns=['On_Target_Rate']), df['On_Target_Rate']
    lin_reg = LinearRegression().fit(X, y)
    y_pred_lin = lin_reg.predict(X)
    r2_lin = lin_reg.score(X, y)
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True).fit(X, y)
    y_pred_rf = rf_reg.predict(X)
    r2_rf = rf_reg.oob_score_
    
    sort_idx = X['Annealing_Temp'].argsort()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X['Annealing_Temp'].iloc[sort_idx], y=y.iloc[sort_idx], mode='markers', name='Actual Data', marker=dict(color=COLORS['dark_gray'], opacity=0.4)))
    fig.add_trace(go.Scatter(x=X['Annealing_Temp'].iloc[sort_idx], y=y_pred_lin[sort_idx], mode='lines', name=f'Linear Model (R²={r2_lin:.2f})', line=dict(color=COLORS['primary'], width=3)))
    fig.add_trace(go.Scatter(x=X['Annealing_Temp'].iloc[sort_idx], y=y_pred_rf[sort_idx], mode='lines', name=f'Random Forest (OOB R²={r2_rf:.2f})', line=dict(color=COLORS['secondary'], width=3, dash='dot')))
    fig.update_layout(title_text="<b>Regression:</b> Modeling Assay Performance", xaxis_title="Primary Factor: Annealing Temp (°C)", yaxis_title="On-Target Rate (%)",
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), plot_bgcolor='white')
    return fig, rf_reg, X

def plot_doe_cube(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=[go.Scatter3d(
        x=df['Primer_Conc'], y=df['Anneal_Temp'], z=df['PCR_Cycles'], mode='markers+text',
        marker=dict(size=12, color=df['Library_Yield'], colorscale='Viridis', showscale=True, colorbar=dict(title='Yield (ng)')),
        text=[f"{y:.1f}" for y in df['Library_Yield']], textposition='top center'
    )])
    lines = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if np.sum(df.iloc[i, :3] != df.iloc[j, :3]) == 1:
                lines.append(go.Scatter3d(x=[df.iloc[i, 0], df.iloc[j, 0]], y=[df.iloc[i, 1], df.iloc[j, 1]], z=[df.iloc[i, 2], df.iloc[j, 2]],
                                          mode='lines', line=dict(color='grey', width=2), showlegend=False))
    fig.add_traces(lines)
    fig.update_layout(title="<b>DOE:</b> PCR Optimization Design Space", scene=dict(xaxis_title='A: Primer Conc', yaxis_title='B: Anneal Temp', zaxis_title='C: PCR Cycles'),
                      margin=dict(l=0, r=0, b=0, t=40))
    return fig

def plot_doe_effects(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    main_effects = {f: df.loc[df[f] == 1, 'Library_Yield'].mean() - df.loc[df[f] == -1, 'Library_Yield'].mean() for f in ['Primer_Conc', 'Anneal_Temp', 'PCR_Cycles']}
    fig_main = px.bar(x=list(main_effects.keys()), y=list(main_effects.values()), color=list(main_effects.keys()),
                      color_discrete_map={'Primer_Conc': COLORS['primary'], 'Anneal_Temp': COLORS['accent'], 'PCR_Cycles': COLORS['secondary']},
                      labels={'x': 'Factor', 'y': 'Effect on Library Yield (ng)'}, title="<b>DOE: Main Effects</b>")
    fig_main.update_layout(plot_bgcolor='white', showlegend=False)
    
    fig_int = go.Figure()
    for level in [-1, 1]:
        subset = df[df['Anneal_Temp'] == level]
        means = subset.groupby('PCR_Cycles')['Library_Yield'].mean()
        fig_int.add_trace(go.Scatter(x=means.index, y=means.values, mode='lines+markers', name=f'Anneal Temp at {level} (Low/High)'))
    fig_int.update_layout(title="<b>DOE: Interaction (Anneal Temp * PCR Cycles)</b>", xaxis_title="Factor C: PCR Cycles", yaxis_title="Mean Library Yield (ng)",
                          plot_bgcolor='white', legend_title_text='Factor B Level')
    return fig_main, fig_int

def plot_bayesian_optimization_interactive(true_func, x_range: np.ndarray, sampled_points: Dict[str, list]) -> Tuple[go.Figure, float]:
    X_sampled, y_sampled = np.array(sampled_points['x']).reshape(-1, 1), np.array(sampled_points['y'])
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    # Use std of sampled points for alpha to stabilize GP
    alpha = np.std(y_sampled)**2 if len(y_sampled) > 1 else 0.1**2
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=alpha, normalize_y=True)
    gp.fit(X_sampled, y_sampled)
    
    y_mean, y_std = gp.predict(x_range.reshape(-1, 1), return_std=True)
    ucb = y_mean + 1.96 * y_std 
    next_point_x = x_range[np.argmax(ucb)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([x_range, x_range[::-1]]), y=np.concatenate([y_mean - 1.96 * y_std, (y_mean + 1.96 * y_std)[::-1]]),
                               fill='toself', fillcolor=hex_to_rgba(COLORS["primary"], 0.2), line=dict(color='rgba(255,255,255,0)'), name='95% Confidence Interval'))
    fig.add_trace(go.Scatter(x=x_range, y=true_func(x_range), mode='lines', name='True Performance Curve (Hidden)', line=dict(color=COLORS['dark_gray'], width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=X_sampled.ravel(), y=y_sampled, mode='markers', name='Experiments Run', marker=dict(color=COLORS['accent'], size=12, symbol='x', line=dict(width=3))))
    fig.add_trace(go.Scatter(x=x_range, y=y_mean, mode='lines', name='GP Model of Assay', line=dict(color=COLORS['primary'], width=3)))
    fig.add_trace(go.Scatter(x=x_range, y=ucb, mode='lines', name='Acquisition Fn (UCB)', line=dict(color=COLORS['secondary'], width=2, dash='dot')))
    fig.add_vline(x=next_point_x, line=dict(color=COLORS['secondary'], width=3), name="Next Experiment to Run")
    fig.update_layout(title_text="<b>Bayesian Optimization:</b> Smart Search for Optimal Conditions",
                      xaxis_title="Parameter Setting (e.g., Enzyme Concentration)", yaxis_title="Assay Performance (e.g., On-Target %)",
                      plot_bgcolor='white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig, next_point_x

def plot_fmea_table() -> go.Figure:
    df = generate_fmea_data()
    colors = df['RPN'].apply(lambda v: hex_to_rgba(COLORS['danger'], 0.5) if v > 150 else (hex_to_rgba(COLORS['warning'], 0.5) if v > 80 else 'white'))
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color=COLORS['primary'], font=dict(color='white'), align='left'),
        cells=dict(values=[df[c] for c in df.columns],
                   fill=dict(color=[colors if c == 'RPN' else ['white'] * len(df) for c in df.columns]), align='left'))])
    fig.update_layout(title="<b>FMEA:</b> Risk Analysis of Lab Protocol", margin=dict(l=10, r=10, t=50, b=10))
    return fig

def plot_rul_prediction(df: pd.DataFrame) -> go.Figure:
    time, signal = df['Run_Number'].values, df['Laser_Power_mW'].values
    threshold, current_time = 80.0, 70
    
    model_time, model_signal = time[time < current_time], signal[time < current_time]
    p = np.polyfit(model_time, np.log(model_signal), 1)
    
    future_time = np.arange(current_time, 120)
    pred_signal = np.exp(p[1]) * np.exp(p[0] * future_time)
    
    failure_indices = np.where(pred_signal <= threshold)[0]
    if len(failure_indices) > 0:
        ttf = future_time[failure_indices[0]]
        rul_text = f"Predicted RUL: {ttf - current_time:.0f} Runs"
    else:
        ttf = 120
        rul_text = f"RUL: >{120-current_time} Runs"
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=signal, mode='markers', name='Actual Laser Power', marker=dict(color=COLORS['dark_gray'], opacity=0.7)))
    fig.add_trace(go.Scatter(x=future_time, y=pred_signal, mode='lines', name='Degradation Model', line=dict(color=COLORS['primary'], dash='dash')))
    fig.add_hline(y=threshold, line=dict(color=COLORS['danger'], width=2), name='Failure Threshold')
    fig.add_vrect(x0=current_time, x1=ttf, fillcolor=hex_to_rgba(COLORS['secondary'], 0.2), line_width=0, name='RUL Window')
    fig.add_vline(x=current_time, line=dict(color=COLORS['dark_gray'], width=2), name="Current Time")
    fig.add_annotation(x=(current_time + ttf) / 2, y=threshold + 5, text=rul_text, showarrow=False, bgcolor='rgba(255,255,255,0.7)')
    fig.update_layout(title='<b>Predictive Maintenance:</b> Sequencer Laser RUL', xaxis_title='Run Number', yaxis_title='Laser Power (mW)', plot_bgcolor='white', legend=dict(x=0.01, y=0.01))
    return fig

def plot_shewhart_chart(df: pd.DataFrame) -> go.Figure:
    mean, std_dev = df['Yield_ng'].iloc[:75].mean(), df['Yield_ng'].iloc[:75].std(ddof=1)
    ucl, lcl = mean + 3 * std_dev, mean - 3 * std_dev
    violations = df[(df['Yield_ng'] > ucl) | (df['Yield_ng'] < lcl)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Batch_ID'], y=df['Yield_ng'], mode='lines+markers', name='QC Control', line=dict(color=COLORS['primary'])))
    fig.add_trace(go.Scatter(x=[0, len(df) - 1], y=[ucl, ucl], mode='lines', name='UCL (Mean+3σ)', line=dict(color=COLORS['accent'], dash='dash')))
    fig.add_trace(go.Scatter(x=[0, len(df) - 1], y=[mean, mean], mode='lines', name='Center Line', line=dict(color=COLORS['dark_gray'], dash='dot')))
    fig.add_trace(go.Scatter(x=[0, len(df) - 1], y=[lcl, lcl], mode='lines', name='LCL (Mean-3σ)', line=dict(color=COLORS['accent'], dash='dash')))
    if not violations.empty:
        fig.add_trace(go.Scatter(x=violations['Batch_ID'], y=violations['Yield_ng'], mode='markers', name='Violation', marker=dict(color=COLORS['danger'], size=10, symbol='x')))
    fig.update_layout(title='<b>Levey-Jennings Chart:</b> Positive Control Monitoring', xaxis_title='Batch ID', yaxis_title='Yield (ng)', plot_bgcolor='white')
    return fig

def plot_ewma_chart(df: pd.DataFrame, lambda_val: float = 0.2) -> go.Figure:
    mean, std_dev = df['Yield_ng'].iloc[:75].mean(), df['Yield_ng'].iloc[:75].std(ddof=1)
    df['ewma'] = df['Yield_ng'].ewm(span=(2 / lambda_val) - 1).mean()
    n = np.arange(1, len(df) + 1)
    ucl_ewma = mean + 3 * std_dev * np.sqrt(lambda_val / (2 - lambda_val) * (1 - (1 - lambda_val) ** (2 * n)))
    lcl_ewma = mean - 3 * std_dev * np.sqrt(lambda_val / (2 - lambda_val) * (1 - (1 - lambda_val) ** (2 * n)))
    violations = df[(df['ewma'] > ucl_ewma) | (df['ewma'] < lcl_ewma)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Batch_ID'], y=df['Yield_ng'], mode='lines', name='Original QC Data', line=dict(color=COLORS['light_gray'], width=1)))
    fig.add_trace(go.Scatter(x=df['Batch_ID'], y=df['ewma'], mode='lines', name='EWMA', line=dict(color=COLORS['secondary'], width=2.5)))
    fig.add_trace(go.Scatter(x=df['Batch_ID'], y=ucl_ewma, mode='lines', name='EWMA UCL', line=dict(color=COLORS['accent'], dash='dash')))
    fig.add_trace(go.Scatter(x=df['Batch_ID'], y=lcl_ewma, mode='lines', name='EWMA LCL', line=dict(color=COLORS['accent'], dash='dash')))
    if not violations.empty:
        fig.add_trace(go.Scatter(x=violations['Batch_ID'], y=violations['ewma'], mode='markers', name='Violation', marker=dict(color=COLORS['danger'], size=10, symbol='x')))
    fig.update_layout(title=f'<b>EWMA Chart (λ={lambda_val}):</b> Detecting Small Drifts', xaxis_title='Batch ID', yaxis_title='Yield (ng)', plot_bgcolor='white')
    return fig

def plot_cusum_chart(df: pd.DataFrame, k: float = 0.5, h: float = 5.0) -> go.Figure:
    mean, std = df['Yield_ng'].iloc[:75].mean(), df['Yield_ng'].iloc[:75].std(ddof=1)
    target, k_val, h_val = mean, k * std, h * std
    sh, sl = np.zeros(len(df)), np.zeros(len(df))
    for i in range(1, len(df)):
        sh[i] = max(0, sh[i - 1] + df['Yield_ng'].iloc[i] - target - k_val)
        sl[i] = min(0, sl[i - 1] + df['Yield_ng'].iloc[i] - target + k_val)
    violations = np.where((sh > h_val) | (sl < -h_val))[0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Batch_ID'], y=sh, name='CUSUM High (Sh)', line=dict(color=COLORS['secondary'])))
    fig.add_trace(go.Scatter(x=df['Batch_ID'], y=sl, name='CUSUM Low (Sl)', line=dict(color=COLORS['primary'])))
    fig.add_hline(y=h_val, line=dict(color=COLORS['accent'], dash='dash'), name=f'Control Limit (H={h})')
    fig.add_hline(y=-h_val, line=dict(color=COLORS['accent'], dash='dash'), name=f'Control Limit (-H={h})')
    if len(violations) > 0:
        fig.add_vline(x=violations[0], line=dict(color=COLORS['danger'], width=2), name='First Detection')
    fig.update_layout(title='<b>CUSUM Chart:</b> Accumulating Small Deviations', xaxis_title='Batch ID', yaxis_title='Cumulative Sum', plot_bgcolor='white')
    return fig

def plot_hotelling_t2_chart() -> go.Figure:
    df = generate_hotelling_data()
    X_in_control = df.iloc[:80, :]
    mean_vec = X_in_control.mean().values
    inv_cov_mat = np.linalg.inv(np.cov(X_in_control.T))
    
    t_squared = [(df.iloc[i, :].values - mean_vec).T @ inv_cov_mat @ (df.iloc[i, :].values - mean_vec) for i in range(len(df))]
    
    n, p, alpha = X_in_control.shape[0], X_in_control.shape[1], 0.01
    ucl = (p * (n + 1) * (n - 1)) / (n * (n - p)) * f_dist.ppf(1 - alpha, p, n - p)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=t_squared, mode='lines+markers', name="T² Statistic", line_color=COLORS['primary']))
    fig.add_hline(y=ucl, line=dict(color=COLORS['danger'], dash='dash'), name=f'UCL (α={alpha})')
    fig.add_vrect(x0=80, x1=100, fillcolor=hex_to_rgba(COLORS['accent'], 0.2), line_width=0, name="Induced Shift")
    fig.update_layout(title="<b>Multivariate QC:</b> Hotelling's T² on NGS Metrics", xaxis_title="Sample Number", yaxis_title="T² Statistic", plot_bgcolor='white')
    return fig

def plot_control_plan() -> go.Figure:
    data = {'Process Step': ['Library Prep', 'Sequencing', 'Bioinformatics'],
            'Characteristic (X or Y)': ['Positive Control Yield (Y)', 'Sequencer Laser Power (X)', '% Mapped Reads (Y)'],
            'Specification': ['20 ± 5 ng', '> 80 mW', '> 85%'],
            'Tool': ['Fluorometer', 'Internal Sensor', 'FASTQC'],
            'Method': ['Levey-Jennings', 'EWMA Chart', 'Shewhart Chart'],
            'Frequency': ['Per Batch', 'Per Run', 'Per Sample'],
            'Reaction Plan': ['Re-prep Batch', 'Schedule Maint.', 'Review Alignment']}
    df = pd.DataFrame(data)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color=COLORS['dark_gray'], font=dict(color='white'), align='left', height=40),
        cells=dict(values=[df[c] for c in df.columns], align='left', height=30))])
    fig.update_layout(title="<b>Assay Control Plan:</b> Formalizing QC Procedures", margin=dict(l=10, r=10, t=50, b=10))
    return fig

def plot_comparison_radar() -> go.Figure:
    categories = ['Interpretability', 'Data Volume Needs', 'Scalability', 'Handling Complexity', 'Biomarker Discovery', 'Regulatory Ease']
    classical_scores = [5, 2, 1, 2, 1, 5]
    ml_scores = [2, 5, 5, 5, 5, 2]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=classical_scores + [classical_scores[0]], theta=categories + [categories[0]], fill='toself', name='Classical DOE/Stats', marker_color=COLORS['primary']))
    fig.add_trace(go.Scatterpolar(r=ml_scores + [ml_scores[0]], theta=categories + [categories[0]], fill='toself', name='ML / Bioinformatics', marker_color=COLORS['secondary']))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5.5])), showlegend=True,
                      title="<b>Strengths Profile:</b> Classical vs. ML for Biotech R&D",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
    return fig

def plot_verdict_barchart() -> go.Figure:
    data = {"Metric": ["Assay Parameter Optimization (DOE)", "Novel Biomarker Discovery", "High-Dimensional Data Analysis", "Analytical Validation (FDA)", "Proactive QC", "Protocol Interpretability"],
            "Winner": ["Classical", "ML", "ML", "Classical", "ML", "Classical"],
            "Score": [-1, 1, 1, -1, 1, -1]}
    df = pd.DataFrame(data).sort_values('Score')
    df['Color'] = df['Score'].apply(lambda x: COLORS['primary'] if x < 0 else COLORS['secondary'])
    fig = px.bar(df, x='Score', y='Metric', orientation='h', color='Color', color_discrete_map='identity', title="<b>Task-Specific Verdict:</b> Which Approach is Better?")
    fig.update_layout(xaxis=dict(tickvals=[-1, 1], ticktext=['<b>Winner: Classical Stats</b>', '<b>Winner: Machine Learning</b>'], tickfont=dict(size=14), range=[-1.5, 1.5]),
                      yaxis_title=None, plot_bgcolor='white', bargap=0.4, showlegend=False)
    return fig

def plot_synergy_diagram() -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="circle", x0=0, y0=0, x1=2, y1=2, line_color=COLORS['primary'], fillcolor=COLORS['primary'], opacity=0.6)
    fig.add_shape(type="circle", x0=1.2, y0=0, x1=3.2, y1=2, line_color=COLORS['secondary'], fillcolor=COLORS['secondary'], opacity=0.6)
    fig.add_annotation(x=1, y=1, text="<b>Classical Stats</b><br><i>Inference & Causality</i><br><i>Rigor & Validation</i>", showarrow=False, font=dict(color="white", size=12))
    fig.add_annotation(x=2.2, y=1, text="<b>Machine Learning</b><br><i>Prediction & Discovery</i><br><i>Complexity & Scale</i>", showarrow=False, font=dict(color="white", size=12))
    fig.add_annotation(x=1.6, y=1, text="<b>Bio-AI<br>Excellence</b>", showarrow=False, font=dict(color="black", size=18, family="Arial Black"))
    fig.update_layout(title="<b>The Hybrid Lab Philosophy:</b> Combining Strengths",
                      xaxis=dict(visible=False, range=[-0.5, 3.7]), yaxis=dict(visible=False, range=[-0.5, 2.5]),
                      plot_bgcolor='white', margin=dict(t=50, b=10, l=10, r=10))
    return fig

def get_guidance_data() -> Dict[str, Dict[str, str]]:
    return {
        "Validating an assay for FDA 510(k) submission": {"approach": "🏆 **Classical Stats** (DOE, LoD/LoB studies, Gage R&R)", "rationale": "Methods are traceable, validated, and follow CLSI/FDA guidelines, which is paramount for regulatory bodies. The focus is on rigorous inference and establishing performance characteristics beyond reproach."},
        "Discovering a new gene signature from RNA-Seq data": {"approach": "🏆 **Machine Learning** (Elastic Net, Random Forest with SHAP)", "rationale": "ML excels at feature selection from high-dimensional data (p >> n). It can identify a minimal, predictive set of genes from thousands of candidates, a task impossible for classical regression."},
        "Optimizing a 12-parameter cell culture media": {"approach": "🏆 **Hybrid:** ML Model + Bayesian Optimization", "rationale": "A full factorial DOE is impossible (2^12 = 4096 runs). Instead, run a small space-filling DOE to train a Gaussian Process model (the 'digital twin' of the culture), then use Bayesian Optimization to find the optimal media composition *in silico*."},
        "Monitoring daily QC for a clinical diagnostic lab": {"approach": "🏆 **Hybrid:** Levey-Jennings + EWMA + Multivariate Control", "rationale": "Use standard Levey-Jennings charts for regulatory compliance. Use more sensitive EWMA charts to detect slow reagent drift. Use a Hotelling's T² chart on the full QC profile to catch subtle, correlated shifts that individual charts would miss."},
        "Identifying sources of contamination in a clean room from microbiome data": {"approach": "🏆 **Bioinformatics & ML** (PCA, Clustering, Source Tracking)", "rationale": "These are high-dimensional, complex datasets. Unsupervised learning methods are required to cluster samples, identify outlier signatures, and trace them back to potential environmental or personnel sources."}
    }

def get_workflow_css() -> str:
    return f"""
    <style>
        .workflow-container{{display:flex;flex-direction:column;align-items:center;width:100%;}}
        .workflow-step{{background-color:#FFFFFF;border:1px solid {COLORS['light_gray']};border-radius:10px;padding:20px;margin-bottom:20px;width:95%;box-shadow:0 4px 6px rgba(0,0,0,0.05);border-left:5px solid;}}
        .workflow-arrow{{font-size:28px;color:{COLORS['dark_gray']};margin-bottom:20px;font-weight:bold;}}
        .step-define{{border-left-color:{COLORS['primary']};}}
        .step-measure{{border-left-color:{COLORS['secondary']};}}
        .step-analyze{{border-left-color:{COLORS['accent']};}}
        .step-improve{{border-left-color:{COLORS['neutral_yellow']};}}
        .step-control{{border-left-color:{COLORS['neutral_pink']};}}
        .workflow-step h4{{margin-top:0;margin-bottom:15px;font-size:1.5em;color:#333333;}}
        .workflow-step .tool-col{{padding:0 15px;}}
        .workflow-step .tool-col h5{{color:#555555;border-bottom:2px solid #EEEEEE;padding-bottom:5px;margin-bottom:10px;}}
        .workflow-step .tool-col ul{{padding-left:20px;margin:0;}}
        .workflow-step .tool-col li{{margin-bottom:5px;}}
        .tool-col-classical h5{{color:{COLORS['primary']};}}
        .tool-col-ml h5{{color:{COLORS['secondary']};}}
    </style>
    """

def render_workflow_step(phase_name: str, phase_class: str, classical_tools: List[str], ml_tools: List[str]) -> str:
    classical_list = "".join([f"<li>{tool}</li>" for tool in classical_tools])
    ml_list = "".join([f"<li>{tool}</li>" for tool in ml_tools])
    return f"""
    <div class="workflow-step {phase_class}">
        <h4>{phase_name}</h4>
        <div style="display:flex;justify-content:space-between;">
            <div style="flex:1;margin-right:10px;" class="tool-col tool-col-classical">
                <h5>Classical Tools (Rigor & Validation)</h5>
                <ul>{classical_list}</ul>
            </div>
            <div style="flex:1;margin-left:10px;" class="tool-col tool-col-ml">
                <h5>ML/Bio-AI Augmentation (Scale & Discovery)</h5>
                <ul>{ml_list}</ul>
            </div>
        </div>
    </div>
    """
