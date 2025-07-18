# app_helpers.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
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
    return f"""
    <style>
        div[data-testid="stAppViewContainer"] > main {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
        }}
        h1, h2 {{
            color: {COLORS['dark_gray']};
            border-bottom: 2px solid {COLORS['light_gray']};
            padding-bottom: 10px;
        }}
        h3 {{ color: {COLORS['primary']}; }}
        h4, h5 {{ color: {COLORS['dark_gray']}; }}
        div[data-testid="stBlock"], div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {{
            border: 1px solid {COLORS['light_gray']};
            border-radius: 0.5rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
        }}
        button[data-testid="stButton"] > button {{ border-radius: 0.5rem; }}
        .stTabs [data-baseweb="tab-list"] button {{
            background-color: transparent;
            border-bottom: 2px solid transparent;
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            border-bottom: 2px solid {COLORS['primary']};
            color: {COLORS['primary']};
        }}
    </style>
    """

def hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip('#')
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {a})"

# ==============================================================================
# SECTION 2: SYNTHETIC DATA GENERATORS (ALL-INCLUSIVE & RESTORED)
# ==============================================================================
def generate_process_data(mean: float, std_dev: float, size: int) -> np.ndarray: return np.random.normal(mean, std_dev, size)
def generate_nonlinear_data(size: int = 200) -> pd.DataFrame:
    np.random.seed(42); X1 = np.linspace(55, 65, size); X2 = 1.0 + 0.1 * (X1 - 60)**2 + np.random.normal(0, 0.5, size); X3_noise = np.random.randn(size) * 5
    y = 70 - 0.5 * (X1 - 62)**2 + 10 * np.log(X2 + 1) + np.random.normal(0, 3, size)
    return pd.DataFrame({'Annealing_Temp': X1, 'Enzyme_Conc': X2, 'Humidity_Noise': X3_noise, 'On_Target_Rate': y})
def generate_control_chart_data(mean: float = 20.0, std_dev: float = 1.5, size: int = 150, shift_point: int = 75, shift_magnitude: float = 0.8) -> pd.DataFrame:
    np.random.seed(42); in_control = np.random.normal(mean, std_dev, shift_point); out_of_control = np.random.normal(mean - shift_magnitude * std_dev, std_dev, size - shift_point)
    return pd.DataFrame({'Batch_ID': np.arange(size), 'Yield_ng': np.concatenate([in_control, out_of_control])})
def generate_doe_data() -> pd.DataFrame:
    np.random.seed(42); factors = [-1, 1]; data = []
    for f1 in factors:
        for f2 in factors:
            for f3 in factors:
                response = 80 + 5*f1 - 12*f2 + 8*f3 + 6*f2*f3 + np.random.randn() * 2.5
                data.append([f1, f2, f3, response])
    return pd.DataFrame(data, columns=['Primer_Conc', 'Anneal_Temp', 'PCR_Cycles', 'Library_Yield'])
def generate_kano_data() -> pd.DataFrame:
    np.random.seed(42); func = np.linspace(0, 10, 20)
    basic_sat = np.clip(np.log(func + 0.1) * 3 - 8, -10, 0) + np.random.normal(0, 0.3, 20); basic_sat[func==0] = -10
    perf_sat = np.linspace(-5, 5, 20) + np.random.normal(0, 0.8, 20)
    excite_sat = np.clip(np.exp(func * 0.4) - 1.5, 0, 10) + np.random.normal(0, 0.3, 20); excite_sat[func==0] = 0
    df_basic = pd.DataFrame({'functionality': func, 'satisfaction': basic_sat, 'category': 'Basic (Must-be)'}); df_perf = pd.DataFrame({'functionality': func, 'satisfaction': perf_sat, 'category': 'Performance'}); df_excite = pd.DataFrame({'functionality': func, 'satisfaction': excite_sat, 'category': 'Excitement (Delighter)'})
    return pd.concat([df_basic, df_perf, df_excite], ignore_index=True)
def generate_anova_data(means: list, stds: list, n: int) -> pd.DataFrame:
    data, groups = [], [];
    for i, (mean, std) in enumerate(zip(means, stds)): data.extend(np.random.normal(mean, std, n)); groups.extend([f'Lot {chr(65+i)}'] * n)
    return pd.DataFrame({'Library_Yield': data, 'Reagent_Lot': groups})
def generate_sensor_degradation_data() -> pd.DataFrame:
    np.random.seed(42); time = np.arange(0, 100); degradation = 100 * np.exp(-time * 0.015) + np.random.normal(0, 0.5, 100)
    degradation[np.random.choice(100, 3, replace=False)] -= np.random.normal(5, 1, 3)
    return pd.DataFrame({'Run_Number': time, 'Laser_Power_mW': degradation})
def generate_pareto_data() -> pd.DataFrame: return pd.DataFrame({'QC_Failure_Mode': ['Low Library Yield', 'Adapter-Dimer Contamination', 'High Duplication Rate', 'Failed Positive Control', 'Low Q30 Score', 'Sample Mix-up'],'Frequency': [45, 22, 11, 6, 4, 2]})
def generate_fmea_data() -> pd.DataFrame:
    return pd.DataFrame([
        {'Failure Mode': 'Reagent Contamination', 'Severity': 10, 'Occurrence': 3, 'Detection': 5}, {'Failure Mode': 'Incorrect Pipetting Volume', 'Severity': 8, 'Occurrence': 5, 'Detection': 3},
        {'Failure Mode': 'Thermal Cycler Malfunction', 'Severity': 9, 'Occurrence': 2, 'Detection': 7}, {'Failure Mode': 'Sample Mis-labeling', 'Severity': 10, 'Occurrence': 1, 'Detection': 2}
    ]).assign(RPN=lambda df: df.Severity * df.Occurrence * df.Detection).sort_values('RPN', ascending=False)
def generate_vsm_data() -> pd.DataFrame:
    return pd.DataFrame([
        {"Step": "Accessioning", "CycleTime": 10, "WaitTime": 120, "ValueAdded": True}, {"Step": "Extraction", "CycleTime": 90, "WaitTime": 30, "ValueAdded": True}, {"Step": "Lib Prep", "CycleTime": 240, "WaitTime": 1200, "ValueAdded": True},
        {"Step": "Sequencing", "CycleTime": 1440, "WaitTime": 2880, "ValueAdded": True}, {"Step": "Biofx", "CycleTime": 180, "WaitTime": 60, "ValueAdded": True}, {"Step": "Reporting", "CycleTime": 30, "WaitTime": 240, "ValueAdded": True}])
def generate_hotelling_data() -> pd.DataFrame:
    np.random.seed(42); mean_in, cov_in = [85, 15], [[4, -3], [-3, 4]]; data_in = np.random.multivariate_normal(mean_in, cov_in, 80)
    mean_out = [80, 22]; data_out = np.random.multivariate_normal(mean_out, cov_in, 20)
    return pd.DataFrame(np.vstack((data_in, data_out)), columns=['Pct_Mapped', 'Pct_Duplication'])
def generate_rsm_data() -> pd.DataFrame:
    np.random.seed(42); temp = np.linspace(50, 70, 15); conc = np.linspace(1, 2, 15)
    T, C = np.meshgrid(temp, conc); yield_val = 90 - 0.1*(T-60)**2 - 20*(C-1.5)**2 - 0.5*(T-60)*(C-1.5) + np.random.normal(0, 2, T.shape)
    return pd.DataFrame({'Temperature': T.ravel(), 'Concentration': C.ravel(), 'Yield': yield_val.ravel()})
def generate_qfd_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    customer_reqs = ['High Sensitivity', 'High Specificity', 'Fast Turnaround', 'Low Cost']; weights = pd.DataFrame({'Importance': [5, 5, 3, 4]}, index=customer_reqs)
    tech_chars = ['LOD (VAF %)', 'Specificity (%)', 'Hands-on Time (min)', 'Reagent Cost ($)']; relationships = np.array([[9, 1, 3, 1], [1, 9, 1, 1], [1, 1, 9, 3], [1, 3, 3, 9]])
    rel_df = pd.DataFrame(relationships, index=customer_reqs, columns=tech_chars); return weights, rel_df
def generate_dfmea_data() -> pd.DataFrame:
    return pd.DataFrame([
        {'Potential Failure Mode': 'Incorrect material selected for sample well', 'Potential Effect': 'Sample Adsorption, low yield', 'Severity': 9, 'Potential Cause': 'Biocompatibility not verified', 'Occurrence': 3, 'Current Controls': 'Material Spec Sheet Review', 'Detection': 6},
        {'Potential Failure Mode': 'Fluidic channel geometry causes bubbles', 'Potential Effect': 'Flow obstruction, assay failure', 'Severity': 10, 'Potential Cause': 'Sharp corners in CAD model', 'Occurrence': 5, 'Current Controls': 'Visual Inspection', 'Detection': 7},
        {'Potential Failure Mode': 'Device housing cracks under stress', 'Potential Effect': 'Leakage, contamination', 'Severity': 7, 'Potential Cause': 'Low-grade polymer used', 'Occurrence': 2, 'Current Controls': 'Drop Test Protocol', 'Detection': 2}
    ]).assign(RPN=lambda df: df.Severity * df.Occurrence * df.Detection).sort_values('RPN', ascending=False)
def generate_capa_data() -> pd.DataFrame:
    return pd.DataFrame({"ID": ["CAPA-001", "CAPA-002", "CAPA-003", "CAPA-004", "CAPA-005", "CAPA-006", "CAPA-007"],
        "Description": ['Batch 2023-45 showed low yield. Investigation found the enzyme from Lot A was degraded due to improper storage in Freezer B.','Contamination detected in negative controls for run 2023-11-02. Root cause traced to aerosolization from adjacent high-titer sample.','Run 2023-11-05 failed due to thermal cycler block temperature overshoot. The cycler requires recalibration.','System flagged low Q30 scores for samples 1-8. Pipetting error during library prep suspected, inconsistent volumes.','Positive control failed for batch 2023-48. The control was stored at the wrong temperature, leading to DNA degradation.','Unexpected peaks observed in chromatography. Re-running sample with fresh mobile phase resolved the issue. Suspect mobile phase degradation.','Pipetting robot failed to dispense into well A3. Service log indicates a clogged nozzle. Required preventative maintenance was overdue.']})
def generate_adverse_event_data() -> pd.DataFrame:
    return pd.DataFrame({"event_id": range(1, 101), "description": ['Patient experienced severe rash after starting treatment'] * 15 + ['Acute liver enzyme elevation noted in patient bloodwork'] * 10 + ['Patient reported mild headache and fatigue'] * 40 + ['Nausea and dizziness reported within 1 hour of dosage'] * 25 + ['Anaphylactic shock occurred; required epinephrine'] * 2 + ['Minor injection site irritation observed'] * 8})
def generate_risk_signal_data() -> pd.DataFrame:
    np.random.seed(42); c1 = pd.DataFrame(np.random.multivariate_normal([70, 2], [[5, -3], [-3, 3]], 30), columns=['Temp_C', 'Pressure_psi']); c1['Source'] = 'Manufacturing Line A'
    c2 = pd.DataFrame(np.random.multivariate_normal([50, 5], [[4, 2], [2, 4]], 50), columns=['Temp_C', 'Pressure_psi']); c2['Source'] = 'Manufacturing Line B'
    outliers = pd.DataFrame([[85, 1], [40, 10], [75, 8]], columns=['Temp_C', 'Pressure_psi']); outliers['Source'] = 'Anomalous Events'
    return pd.concat([c1, c2, outliers], ignore_index=True)
def generate_pccp_data() -> pd.DataFrame:
    np.random.seed(42); time = np.arange(100); performance = 0.95 - 0.0001 * time - 0.000005 * time**2 + np.random.normal(0, 0.005, 100)
    performance[70:] -= 0.05; return pd.DataFrame({'Deployment_Day': time, 'Model_AUC': performance})
def generate_validation_data() -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    np.random.seed(42); data = {'Metric': ['Accuracy', 'Sensitivity', 'Specificity'], 'Value': [0.95, 0.92, 0.97]}; df = pd.DataFrame(data)
    bootstrap_samples = {'Accuracy': np.random.normal(0.95, 0.02, 1000), 'Sensitivity': np.random.normal(0.92, 0.04, 1000), 'Specificity': np.random.normal(0.97, 0.015, 1000)}
    return df, bootstrap_samples

# ==============================================================================
# SECTION 3: VISUALIZATION HELPERS (ALL-INCLUSIVE & SME-GRADE)
# ==============================================================================

def _create_network_fig(height=400, x_range=None, y_range=None) -> go.Figure:
    fig = go.Figure(); fig.update_layout(showlegend=False, plot_bgcolor='white', height=height, xaxis=dict(showgrid=False, zeroline=False, visible=False, range=x_range), yaxis=dict(showgrid=False, zeroline=False, visible=False, range=y_range), margin=dict(t=40, b=20, l=20, r=20)); return fig
def _add_network_nodes_and_edges(fig: go.Figure, nodes: Dict, edges: List[Tuple], annotations: bool = True):
    for edge in edges: start_node, end_node = nodes[edge[0]], nodes[edge[1]]; fig.add_trace(go.Scatter(x=[start_node['x'], end_node['x']], y=[start_node['y'], end_node['y']], mode='lines', line=dict(color=COLORS['light_gray'], width=2), hoverinfo='none'))
    if annotations:
        for node_id, node_data in nodes.items(): fig.add_annotation(x=node_data['x'], y=node_data['y'], text=f"<b>{node_data['text'].replace('<br>', '<br>')}</b>", showarrow=False, font=dict(color=COLORS['text'], size=11, family="Arial"), bgcolor=hex_to_rgba(node_data.get('color', COLORS['primary']), 0.15), bordercolor=node_data.get('color', COLORS['primary']), borderwidth=2, borderpad=10, align="center")

# --- ALL VISUALIZATION FUNCTIONS, RESTORED AND VERIFIED ---

def plot_qfd_house_of_quality() -> go.Figure:
    weights, rel_df = generate_qfd_data()
    tech_importance = (rel_df.T * weights['Importance'].values).T.sum()
    fig = make_subplots(rows=2, cols=2, column_widths=[0.25, 0.75], row_heights=[0.75, 0.25], specs=[[{"type": "table"}, {"type": "heatmap"}], [None, {"type": "bar"}]], vertical_spacing=0.02, horizontal_spacing=0.02)
    fig.add_trace(go.Heatmap(z=rel_df.values, x=rel_df.columns, y=rel_df.index, colorscale='Blues', text=rel_df.values, texttemplate="%{text}", showscale=False), row=1, col=2)
    fig.add_trace(go.Table(header=dict(values=['<b>Customer Need</b>', '<b>Importance</b>'], fill_color=COLORS['dark_gray'], font=dict(color='white'), align='left'), cells=dict(values=[weights.index, weights.Importance], align='left', height=40, fill_color=[[hex_to_rgba(COLORS['light_gray'], 0.2), 'white']*len(weights)])), row=1, col=1)
    fig.add_trace(go.Bar(x=tech_importance.index, y=tech_importance.values, marker_color=COLORS['primary'], text=tech_importance.values, texttemplate='%{text:.0f}', textposition='outside'), row=2, col=2)
    fig.update_layout(title_text="<b>QFD 'House of Quality':</b> Translating VOC to Design Specs", plot_bgcolor='white', showlegend=False, margin=dict(l=10, r=10, t=50, b=10), height=500); fig.update_xaxes(showticklabels=False, row=1, col=2); fig.update_yaxes(title_text="<b>Technical Importance Score</b>", row=2, col=2, range=[0, max(tech_importance.values)*1.2]); return fig
def plot_dfmea_table() -> go.Figure:
    df = generate_dfmea_data(); column_widths = [4, 3, 1, 3, 1, 3, 1, 1]
    fig = go.Figure(data=[go.Table(columnorder=list(range(len(df.columns))), columnwidth=column_widths, header=dict(values=[f'<b>{col.replace(" ", "<br>")}</b>' for col in df.columns], fill_color=COLORS['dark_gray'], font=dict(color='white'), align='center', height=50), cells=dict(values=[df[c] for c in df.columns], align='left', height=60, fill_color=[[hex_to_rgba(COLORS['danger'], 0.5) if c == 'RPN' and v > 200 else (hex_to_rgba(COLORS['warning'], 0.5) if c == 'RPN' and v > 100 else 'white') for v in df[c]] for c in df.columns]))])
    fig.update_layout(title="<b>Design FMEA (DFMEA):</b> Proactive Risk Analysis of Device Design", margin=dict(l=10, r=10, t=50, b=10), height=350); return fig
def plot_risk_signal_clusters() -> go.Figure:
    df = generate_risk_signal_data(); model = DBSCAN(eps=3, min_samples=3).fit(df[['Temp_C', 'Pressure_psi']]); df['cluster'] = [str(c) for c in model.labels_]; df.loc[df['cluster'] == '-1', 'cluster'] = 'Outlier/Anomaly'
    fig = px.scatter(df, x='Temp_C', y='Pressure_psi', color='cluster', symbol='Source', color_discrete_map={'0': COLORS['primary'], '1': COLORS['secondary'], 'Outlier/Anomaly': COLORS['danger']}, title="<b>ML Clustering of Process Data for Risk Signal Detection</b>")
    fig.update_layout(plot_bgcolor='white', legend_title="Identified Group", xaxis_title="Temperature (°C)", yaxis_title="Pressure (psi)"); fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey'))); return fig
def plot_rsm_contour(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Contour(z=df['Yield'], x=df['Temperature'], y=df['Concentration'], colorscale='Viridis', contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=12, color='white'))))
    max_yield_point = df.loc[df['Yield'].idxmax()]; fig.add_trace(go.Scatter(x=[max_yield_point['Temperature']], y=[max_yield_point['Concentration']], mode='markers', marker=dict(color=COLORS['danger'], size=15, symbol='star'), name='Optimal Point'))
    fig.update_layout(title="<b>Response Surface Methodology (RSM):</b> Mapping the Design Space", xaxis_title="Temperature (°C)", yaxis_title="Enzyme Concentration", plot_bgcolor='white', showlegend=False); return fig
def plot_model_validation_ci() -> go.Figure:
    df, bootstrap_samples = generate_validation_data(); fig = go.Figure()
    for i, row in df.iterrows():
        metric = row['Metric']; samples = bootstrap_samples[metric]; ci_95 = np.percentile(samples, [2.5, 97.5])
        fig.add_trace(go.Box(y=samples, name=metric, marker_color=COLORS['primary'])); fig.add_annotation(x=metric, y=ci_95[1], text=f"<b>95% CI:</b> [{ci_95[0]:.3f}, {ci_95[1]:.3f}]", showarrow=False, yshift=20, font=dict(color=COLORS['dark_gray']))
    fig.update_layout(title="<b>ML Model Validation with Bootstrapped Confidence Intervals</b>", yaxis_title="Performance Metric Value", plot_bgcolor='white', showlegend=False); return fig
def plot_fault_tree_plotly() -> go.Figure:
    fig = _create_network_fig(height=500, x_range=[-0.5, 4.5], y_range=[-0.5, 5]); nodes = {'top':{'x': 2, 'y': 4.5, 'text': '<b>TOP EVENT</b><br>False Negative Result', 'color': COLORS['danger']}, 'or1':{'x': 2, 'y': 3.5, 'text': 'OR Gate', 'color': COLORS['dark_gray']}, 'and1':{'x': 0.5, 'y': 2, 'text': 'AND Gate', 'color': COLORS['dark_gray']}, 'assay':{'x': 3.5, 'y': 2.5, 'text': 'Assay Failure', 'color': COLORS['primary']}, 'reagent':{'x': 0, 'y': 1, 'text': 'Reagent Degraded<br>P=0.01', 'color': COLORS['secondary']}, 'storage':{'x': 1, 'y': 1, 'text': 'Improper Storage<br>P=0.05', 'color': COLORS['secondary']}, 'sample':{'x': 3.5, 'y': 1.5, 'text': 'Low DNA Input<br>P=0.02', 'color': COLORS['primary']}}
    edges = [('top', 'or1'), ('or1', 'and1'), ('or1', 'assay'), ('assay', 'sample'), ('and1', 'reagent'), ('and1', 'storage')]; _add_network_nodes_and_edges(fig, nodes, edges); fig.update_layout(title="<b>Fault Tree Analysis (FTA):</b> Top-Down Risk Assessment"); return fig
def plot_nlp_on_capa_logs() -> go.Figure:
    df = generate_capa_data(); topics = {"Reagent/Storage Issue": "enzyme|degradation|lot|freezer|stored|mobile phase", "Contamination": "contamination|aerosolization|negative control", "Hardware Failure": "thermal cycler|calibration|overshoot|robot|clogged|nozzle", "Human Error": "pipetting|inconsistent volumes"}
    topic_counts = {topic: df['Description'].str.contains(pattern, case=False).sum() for topic, pattern in topics.items()}
    fig = px.bar(x=list(topic_counts.values()), y=list(topic_counts.keys()), orientation='h', color=list(topic_counts.keys()), labels={'y': 'Identified Failure Theme', 'x': 'Frequency'}, title="<b>NLP Topic Modeling on CAPA & Deviation Logs</b>", color_discrete_map={"Reagent/Storage Issue": COLORS['primary'], "Contamination": COLORS['accent'], "Hardware Failure": COLORS['danger'], "Human Error": COLORS['warning']})
    fig.update_layout(plot_bgcolor='white', showlegend=False, yaxis={'categoryorder':'total ascending'}); return fig
def plot_5whys_diagram() -> go.Figure:
    fig = _create_network_fig(height=550, y_range=[-0.5, 5.5]); steps = [('Problem', 'Low Library Yield on Plate 4'), ('Why 1?', 'Reagents added improperly.'), ('Why 2?', 'Technician used a miscalibrated multi-channel pipette.'), ('Why 3?', 'Pipette was overdue for its 6-month calibration.'), ('Why 4?', 'Calibration tracking system is a manual spreadsheet, prone to error.'), ('Root Cause', 'The asset management system is not robust enough to prevent use of out-of-spec equipment.')]
    nodes, edges = {}, [];
    for i, (level, text) in enumerate(steps): y_pos = 5 - i; color = COLORS['danger'] if i == 0 else (COLORS['success'] if i == len(steps) - 1 else COLORS['primary']); nodes[f's{i}'] = {'x': 1, 'y': y_pos, 'text': f'<b>{level}</b><br>{text}', 'color': color};
    if i > 0: edges.append((f's{i-1}', f's{i}'))
    _add_network_nodes_and_edges(fig, nodes, edges); fig.update_layout(title="<b>5 Whys Analysis:</b> Drilling Down to the True Root Cause"); return fig
def plot_pccp_monitoring() -> go.Figure:
    df = generate_pccp_data(); fig = go.Figure(); fig.add_trace(go.Scatter(x=df['Deployment_Day'], y=df['Model_AUC'], mode='lines', name='Model Performance (AUC)', line=dict(color=COLORS['primary']))); fig.add_hline(y=0.90, line=dict(color=COLORS['danger'], dash='dash'), name='Performance Threshold'); fig.add_vrect(x0=70, x1=100, fillcolor=hex_to_rgba(COLORS['warning'], 0.2), line_width=0, name="Performance Degradation"); fig.add_annotation(x=85, y=0.87, text="<b>Retraining & Revalidation<br>Triggered per PCCP</b>", showarrow=True, arrowhead=1, ax=0, ay=-40, bgcolor="rgba(255,255,255,0.7)")
    fig.update_layout(title="<b>PCCP Monitoring for an AI/ML Device (SaMD)</b>", xaxis_title="Days Since Deployment", yaxis_title="Model Area Under Curve (AUC)", plot_bgcolor='white', legend=dict(x=0.01, y=0.01, yanchor='bottom')); return fig
def plot_adverse_event_clusters() -> go.Figure:
    df = generate_adverse_event_data(); vectorizer = TfidfVectorizer(stop_words='english'); X = vectorizer.fit_transform(df['description']); pca = PCA(n_components=2, random_state=42); X_pca = pca.fit_transform(X.toarray()); kmeans = KMeans(n_clusters=4, random_state=42, n_init=10); df['cluster'] = kmeans.fit_predict(X); df['x'] = X_pca[:, 0]; df['y'] = X_pca[:, 1]
    cluster_names = {0: "Neurological (Headache, Dizziness)", 1: "Allergic / Skin Reaction", 2: "Systemic (Liver, Anaphylaxis)", 3: "Gastrointestinal / Injection Site"}; df['cluster_name'] = df['cluster'].map(cluster_names)
    fig = px.scatter(df, x='x', y='y', color='cluster_name', hover_data=['description'], title="<b>ML Clustering of Adverse Event Narratives for Signal Detection</b>", labels={'color': 'Event Cluster'}); fig.update_layout(xaxis_title="PCA Component 1", yaxis_title="PCA Component 2", plot_bgcolor='white', xaxis=dict(showticklabels=False, zeroline=False, showgrid=False), yaxis=dict(showticklabels=False, zeroline=False, showgrid=False)); return fig
def train_and_plot_regression_models(df: pd.DataFrame) -> Tuple[go.Figure, RandomForestRegressor, pd.DataFrame]:
    X, y = df.drop(columns=['On_Target_Rate']), df['On_Target_Rate']; lin_reg = LinearRegression().fit(X, y); y_pred_lin = lin_reg.predict(X); r2_lin = lin_reg.score(X, y)
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True).fit(X, y); y_pred_rf = rf_reg.predict(X); r2_rf = rf_reg.oob_score_
    sort_idx = X['Annealing_Temp'].argsort(); fig = go.Figure(); fig.add_trace(go.Scatter(x=X['Annealing_Temp'].iloc[sort_idx], y=y.iloc[sort_idx], mode='markers', name='Actual Data', marker=dict(color=COLORS['dark_gray'], opacity=0.4))); fig.add_trace(go.Scatter(x=X['Annealing_Temp'].iloc[sort_idx], y=y_pred_lin[sort_idx], mode='lines', name=f'Linear Model (R²={r2_lin:.2f})', line=dict(color=COLORS['primary'], width=3))); fig.add_trace(go.Scatter(x=X['Annealing_Temp'].iloc[sort_idx], y=y_pred_rf[sort_idx], mode='lines', name=f'Random Forest (OOB R²={r2_rf:.2f})', line=dict(color=COLORS['secondary'], width=3, dash='dot')))
    fig.update_layout(title_text="<b>Regression:</b> Modeling Assay Performance", xaxis_title="Primary Factor: Annealing Temp (°C)", yaxis_title="On-Target Rate (%)", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), plot_bgcolor='white'); return fig, rf_reg, X
def plot_shap_summary(model: RandomForestRegressor, X: pd.DataFrame) -> go.Figure:
    explainer = shap.TreeExplainer(model); shap_values = explainer(X); fig = go.Figure()
    for i, feature in enumerate(X.columns):
        y_jitter = np.random.uniform(-0.25, 0.25, len(shap_values)); y_pos = np.full(len(shap_values), i) + y_jitter
        fig.add_trace(go.Scatter(x=shap_values.values[:, i], y=y_pos, mode='markers', marker=dict(color=shap_values.data[:, i], colorscale='RdBu_r', showscale=(i == 0), colorbar=dict(title="Feature Value<br>High / Low", x=1.02, y=0.5, len=0.75), symbol='circle', size=6, opacity=0.7), hoverinfo='text', hovertext=[f'<b>{feature}</b><br>Value: {val:.2f}<br>SHAP: {shap_val:.2f}' for val, shap_val in zip(shap_values.data[:, i], shap_values.values[:, i])], showlegend=False))
    fig.update_layout(title="<b>XAI with SHAP:</b> Parameter Impact on Outcome", xaxis_title="SHAP Value (Impact on Model Output)", yaxis=dict(tickmode='array', tickvals=list(range(len(X.columns))), ticktext=[col.replace('_', ' ') for col in X.columns], showgrid=True, gridcolor=COLORS['light_gray']), plot_bgcolor='white', margin=dict(l=150)); return fig
def plot_ctq_tree_plotly() -> go.Figure:
    fig = _create_network_fig(height=450); nodes = { 'Need':    {'x': 0, 'y': 2, 'text': 'Clinician Need<br>Reliable Early CRC Detection', 'color': COLORS['accent']}, 'Driver1': {'x': 1, 'y': 3, 'text': 'High Sensitivity', 'color': COLORS['primary']}, 'Driver2': {'x': 1, 'y': 2, 'text': 'High Specificity', 'color': COLORS['primary']}, 'Driver3': {'x': 1, 'y': 1, 'text': 'Fast Turnaround', 'color': COLORS['primary']}, 'CTQ1':    {'x': 2, 'y': 3, 'text': 'CTQ:<br>LOD < 0.1% VAF', 'color': COLORS['secondary']}, 'CTQ2':    {'x': 2, 'y': 2, 'text': 'CTQ:<br>Specificity > 99.5%', 'color': COLORS['secondary']}, 'CTQ3':    {'x': 2, 'y': 1, 'text': 'CTQ:<br>Sample-to-Report < 5 days', 'color': COLORS['secondary']} }; edges = [('Need', 'Driver1'), ('Need', 'Driver2'), ('Need', 'Driver3'), ('Driver1', 'CTQ1'), ('Driver2', 'CTQ2'), ('Driver3', 'CTQ3')]; _add_network_nodes_and_edges(fig, nodes, edges); return fig
def plot_causal_discovery_plotly() -> go.Figure:
    fig = _create_network_fig(height=350); nodes = { 'ReagentLot':   {'x': 0, 'y': 1.5, 'text': 'Reagent Lot', 'color': COLORS['primary']}, 'DNAnq':        {'x': 0, 'y': 0.5, 'text': 'DNA Input (ng)', 'color': COLORS['primary']}, 'LigationTime': {'x': 1.2, 'y': 1, 'text': 'Ligation Time', 'color': COLORS['secondary']}, 'AdapterDimer': {'x': 2.4, 'y': 1, 'text': 'Adapter-Dimer %', 'color': COLORS['accent']} }; edges = [('ReagentLot', 'LigationTime'), ('DNAnq', 'LigationTime'), ('LigationTime', 'AdapterDimer')]; _add_network_nodes_and_edges(fig, nodes, edges)
    for start, end in edges: fig.add_annotation(x=nodes[end]['x'], y=nodes[end]['y'], ax=nodes[start]['x'], ay=nodes[start]['y'], xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor=COLORS['dark_gray']); return fig
def plot_process_mining_plotly() -> go.Figure:
    fig = _create_network_fig(height=450, x_range=[-0.5, 5.5], y_range=[0.5, 2.5]); nodes = { 'start': {'x': 0.0, 'y': 2.0, 'text': 'Sample<br>Received', 'color': COLORS['success']}, 'A':     {'x': 1.0, 'y': 2.0, 'text': 'DNA Extraction', 'color': COLORS['primary']}, 'B':     {'x': 2.0, 'y': 2.0, 'text': 'Library Prep', 'color': COLORS['primary']}, 'E':     {'x': 2.0, 'y': 0.8, 'text': 'QC Fail:<br>Re-Prep', 'color': COLORS['danger']}, 'C':     {'x': 3.0, 'y': 2.0, 'text': 'Sequencing', 'color': COLORS['primary']}, 'D':     {'x': 4.0, 'y': 2.0, 'text': 'Bioinformatics', 'color': COLORS['primary']}, 'end':   {'x': 5.0, 'y': 2.0, 'text': 'Report<br>Sent', 'color': COLORS['dark_gray']} }; edges = [('start', 'A'), ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'end'), ('B', 'E'), ('E', 'B')]; _add_network_nodes_and_edges(fig, nodes, edges)
    edge_labels = { 'start-A': ('start', 'A', '20 Samples', 0.15), 'A-B': ('A', 'B', '20 Samples', 0.15), 'B-C': ('B', 'C', '18 Samples<br>Avg 5h', 0.15), 'C-D': ('C', 'D', '18 Samples<br>Avg 26h', 0.15), 'D-end': ('D', 'end', '18 Samples<br>Avg 4h', 0.15), 'B-E': ('B', 'E', '2 Samples (10%)', -0.15), 'E-B': ('E', 'B', 'Avg 8h Delay', 0.15) }
    for start, end, text, offset in edge_labels.values(): fig.add_annotation(x=(nodes[start]['x'] + nodes[end]['x']) / 2, y=(nodes[start]['y'] + nodes[end]['y']) / 2 + offset, text=text, showarrow=False, font=dict(size=9), bgcolor='rgba(255,255,255,0.7)'); return fig
def plot_fishbone_plotly() -> go.Figure:
    fig = _create_network_fig(height=500, x_range=[-1, 10], y_range=[0, 10]); fig.add_annotation(x=8.5, y=5, text="<b>Low Library<br>Yield</b>", showarrow=False, font=dict(color=COLORS['text'], size=14), bgcolor=hex_to_rgba(COLORS['danger'], 0.15), bordercolor=COLORS['danger'], borderwidth=2, borderpad=10, align="center"); fig.add_shape(type="line", x0=0, y0=5, x1=8.2, y1=5, line=dict(color=COLORS['dark_gray'], width=3))
    bones = { 'Reagents': {'pos': 1, 'causes': ['Enzyme Inactivity'], 'angle': 45}, 'Equipment': {'pos': 3, 'causes': ['Pipette Out of Cal'], 'angle': 45}, 'Method': {'pos': 5, 'causes': ['Incorrect Incubation Time'], 'angle': 45}, 'Technician': {'pos': 2, 'causes': ['Inconsistent Pipetting'], 'angle': -45}, 'Sample': {'pos': 4, 'causes': ['Low DNA Input'], 'angle': -45}, 'Environment': {'pos': 6, 'causes': ['High Humidity'], 'angle': -45} }
    for name, data in bones.items():
        angle_rad = np.deg2rad(data['angle']); x_start, y_start = data['pos'], 5; x_end = x_start + 2.5 * np.cos(angle_rad); y_end = y_start + 2.5 * np.sin(angle_rad)
        fig.add_shape(type="line", x0=x_start, y0=y_start, x1=x_end, y1=y_end, line=dict(color=COLORS['dark_gray'], width=1.5)); fig.add_annotation(x=x_end, y=y_end + 0.4 * np.sign(y_end - 5), text=f"<b>{name}</b>", showarrow=False, font=dict(color=COLORS['primary']))
        for i, cause in enumerate(data['causes']): sub_x_start = x_start + (1.2 + i*1.0) * np.cos(angle_rad); sub_y_start = y_start + (1.2 + i*1.0) * np.sin(angle_rad); sub_x_end = sub_x_start + 1.0 * np.cos(angle_rad); sub_y_end = sub_y_start + 1.0 * np.sin(angle_rad); text_x = sub_x_start + 0.6 * np.cos(angle_rad + np.pi/2); text_y = sub_y_start + 0.6 * np.sin(angle_rad + np.pi/2); fig.add_shape(type="line", x0=sub_x_start, y0=sub_y_start, x1=sub_x_end, y1=sub_y_end, line=dict(color='grey', width=1)); fig.add_annotation(x=text_x, y=text_y, text=cause, showarrow=False, font=dict(size=10, color=COLORS['text']))
    return fig
def plot_attribute_matrix() -> go.Figure:
    data = {'Attribute': ['Transparency', 'Data Requirements', 'Assumptions', 'Scalability', 'Implementation Cost', 'Auditability'], 'Classical Statistics (DOE, etc.)': ['High (e.g., regression coefficients, p-values)', 'Low (can work with small, structured datasets)', 'Many (e.g., normality, homoscedasticity)', 'Poor beyond 3-4 interacting variables', 'Low (Excel, Minitab)', 'High (standardized, validated methods)'], 'Machine Learning / AI': ["Often low ('black-box') unless using XAI like SHAP/LIME", 'High (more data typically improves generalization)', 'Fewer (nonparametric, flexible models)', 'Excellent with high-dimensional data', 'Higher (Python, cloud infra, data pipelines)', 'Lower (can be complex to validate model logic)']}
    df = pd.DataFrame(data); fig = go.Figure(data=[go.Table(columnorder=[0, 1, 2], columnwidth=[25, 40, 40], header=dict(values=[f'<b>{col}</b>' for col in df.columns], line_color=COLORS['dark_gray'], fill_color=COLORS['dark_gray'], align='center', font=dict(color='white', size=12), height=30), cells=dict(values=[df[k].tolist() for k in df.columns], line_color=COLORS['light_gray'], fill_color=[['white', hex_to_rgba(COLORS['light_gray'], 0.2)]*3], align='left', font=dict(color=COLORS['text'], size=11), height=50))]); fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380); return fig
def plot_project_charter_visual() -> go.Figure:
    fig = go.Figure(); fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, fillcolor='white', line_width=0); fig.add_annotation(x=0.5, y=0.92, text="<b>Assay Development Plan: Liquid Biopsy for CRC</b>", showarrow=False, font=dict(size=22, color=COLORS['primary']))
    kpis = {"Analytical Sensitivity": ("LOD < 0.1% VAF", COLORS['success']), "Clinical Specificity": ("> 99.5%", COLORS['success']), "Turnaround Time": ("< 5 days", COLORS['success'])};
    for i, (k, (v, c)) in enumerate(kpis.items()): fig.add_annotation(x=0.2+i*0.3, y=0.75, text=f"<b>{k}</b>", showarrow=False, font=dict(size=14, color=COLORS['dark_gray'])); fig.add_annotation(x=0.2+i*0.3, y=0.65, text=v, showarrow=False, font=dict(size=20, color=c))
    statements = { "Problem Statement": (0.05, 0.45, "Colorectal Cancer (CRC) requires earlier detection methods. Current methods are invasive or lack sensitivity for early-stage disease.", "left"), "Goal Statement": (0.95, 0.45, "Develop and validate a cfDNA-based NGS assay for early-stage CRC detection with >90% sensitivity at 99.5% specificity.", "right")}
    for title, (x, y, txt, anchor) in statements.items(): fig.add_annotation(x=x, y=y, text=f"<b>{title}</b>", showarrow=False, align=anchor, xanchor=anchor, font_size=16); fig.add_annotation(x=x, y=y-0.1, text=txt, showarrow=False, align=anchor, xanchor=anchor, yanchor='top', width=400)
    fig.update_layout(xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[0, 1]), plot_bgcolor='white', margin=dict(t=20, b=20, l=20, r=20), height=350); return fig
def plot_sipoc_visual() -> go.Figure:
    header_values = ['<b>👤<br>Suppliers</b>', '<b>🧬<br>Inputs</b>', '<b>⚙️<br>Process</b>', '<b>📊<br>Outputs</b>', '<b>⚕️<br>Customers</b>']; cell_values = [['• Reagent Vendors<br>• Instrument Mfr.<br>• LIMS Provider'], ['• Patient Blood Sample<br>• Reagent Kits<br>• Lab Protocol (SOP)'], ['1. Sample Prep<br>2. Library Prep<br>3. NGS Sequencing<br>4. Bioinformatics<br>5. Reporting'], ['• VCF File<br>• QC Metrics Report<br>• Clinical Report'], ['• Oncologists<br>• Patients<br>• Pharma Partners']]
    fig = go.Figure(data=[go.Table(header=dict(values=header_values, line_color=COLORS['light_gray'], fill_color=COLORS['light_gray'], align='center', font=dict(color=COLORS['primary'], size=14)), cells=dict(values=cell_values, line_color=COLORS['light_gray'], fill_color='white', align='left', font_size=12, height=150))]); fig.update_layout(title_text="<b>SIPOC Diagram:</b> NGS Assay Workflow", margin=dict(l=10, r=10, t=50, b=10)); return fig
def plot_kano_visual() -> go.Figure:
    df = generate_kano_data(); fig = go.Figure(); fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=10, fillcolor=hex_to_rgba(COLORS['success'], 0.1), line_width=0, layer='below'); fig.add_shape(type="rect", x0=0, y0=-10, x1=10, y1=0, fillcolor=hex_to_rgba(COLORS['danger'], 0.1), line_width=0, layer='below'); colors = {'Basic (Must-be)': COLORS['accent'], 'Performance': COLORS['primary'], 'Excitement (Delighter)': COLORS['secondary']}
    for cat, color in colors.items(): subset = df[df['category'] == cat]; fig.add_trace(go.Scatter(x=subset['functionality'], y=subset['satisfaction'], mode='lines', name=cat, line=dict(color=color, width=4)))
    fig.add_annotation(x=8, y=8, text="<b>Excitement</b><br>e.g., Detects new<br>resistance mutation", showarrow=True, arrowhead=1, ax=-50, ay=-40, font_color=COLORS['secondary']); fig.add_annotation(x=8, y=4, text="<b>Performance</b><br>e.g., VAF quantification<br>accuracy", showarrow=True, arrowhead=1, ax=0, ay=-40, font_color=COLORS['primary']); fig.add_annotation(x=8, y=-8, text="<b>Basic</b><br>e.g., Detects known<br>KRAS hotspot", showarrow=True, arrowhead=1, ax=0, ay=40, font_color=COLORS['accent']); fig.update_layout(title='<b>Kano Model:</b> Prioritizing Diagnostic Features', xaxis_title='Feature Performance / Implementation →', yaxis_title='← Clinician Dissatisfaction ... Satisfaction →', plot_bgcolor='white', legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)')); return fig
def plot_voc_bubble_chart() -> go.Figure:
    data = {'Category': ['Biomarkers', 'Biomarkers', 'Methodology', 'Methodology', 'Performance', 'Performance'], 'Topic': ['EGFR Variants', 'KRAS Hotspots', 'ddPCR', 'Shallow WGS', 'LOD <0.1%', 'Specificity >99%'], 'Count': [180, 150, 90, 60, 250, 210], 'Sentiment': [0.5, 0.4, -0.2, -0.4, 0.8, 0.7]}; df = pd.DataFrame(data); fig = px.scatter(df, x='Topic', y='Sentiment', size='Count', color='Category', hover_name='Topic', size_max=60, labels={"Sentiment": "Average Sentiment Score", "Topic": "Biomarker or Methodology", "Count": "Publication Volume"}, color_discrete_map={'Biomarkers': COLORS['primary'], 'Methodology': COLORS['secondary'], 'Performance': COLORS['accent']})
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey"); fig.update_layout(title="<b>NLP Landscape:</b> Scientific Literature Analysis", plot_bgcolor='white', yaxis=dict(range=[-1, 1], gridcolor=COLORS['light_gray']), xaxis=dict(showgrid=False), legend_title_text='Topic Category'); fig.update_traces(hovertemplate='<b>%{hovertext}</b><br>Publication Count: %{marker.size:,}<br>Avg. Sentiment: %{y:.2f}'); return fig
def plot_gage_rr_pareto() -> go.Figure:
    data = {'Source of Variation': ['Assay Variation (Biology)', 'Repeatability (Sequencer)', 'Reproducibility (Operator)'], 'Contribution (%)': [92, 5, 3]}; df = pd.DataFrame(data).sort_values('Contribution (%)', ascending=False).reset_index(drop=True); df['Cumulative Percentage'] = df['Contribution (%)'].cumsum(); fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df['Source of Variation'], y=df['Contribution (%)'], name='Contribution', marker_color=[COLORS['primary'], COLORS['warning'], COLORS['accent']]), secondary_y=False); fig.add_trace(go.Scatter(x=df['Source of Variation'], y=df['Cumulative Percentage'], name='Cumulative %', mode='lines+markers', line_color=COLORS['dark_gray']), secondary_y=True)
    fig.update_layout(title='<b>Gage R&R Pareto:</b> Identifying Largest Sources of Measurement Error', xaxis_title="Source of Variation", plot_bgcolor='white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); fig.update_yaxes(title_text="Percent Contribution", secondary_y=False, range=[0, 100], ticksuffix='%'); fig.update_yaxes(title_text="Cumulative Percentage", secondary_y=True, range=[0, 101], ticksuffix='%'); return fig
def plot_vsm() -> go.Figure:
    df = generate_vsm_data(); total_lead_time = (df['CycleTime'] + df['WaitTime']).sum(); va_time = df[df['ValueAdded']]['CycleTime'].sum(); pce = (va_time / total_lead_time) * 100 if total_lead_time > 0 else 0; fig = go.Figure(); current_pos = 0
    for _, row in df.iterrows():
        cycle_pct = row['CycleTime'] / total_lead_time * 100; wait_pct = row['WaitTime'] / total_lead_time * 100
        fig.add_shape(type="rect", x0=current_pos, x1=current_pos + cycle_pct, y0=1, y1=2, fillcolor=COLORS['secondary'] if row['ValueAdded'] else COLORS['danger'], line_color=COLORS['dark_gray']); fig.add_annotation(x=current_pos + cycle_pct / 2, y=1.5, text=f"{row['Step']}<br>{row['CycleTime']/60:.1f}h", showarrow=False, font=dict(color='white')); current_pos += cycle_pct
        if row['WaitTime'] > 0: fig.add_shape(type="rect", x0=current_pos, x1=current_pos + wait_pct, y0=0, y1=1, fillcolor=COLORS['warning'], line_color=COLORS['accent'], opacity=0.7);
        if wait_pct > 5: fig.add_annotation(x=current_pos + wait_pct / 2, y=0.5, text=f"{row['WaitTime']/60:.1f}h wait", showarrow=False); current_pos += wait_pct
    fig.update_layout(title=f"<b>Value Stream Map (Normalized):</b> Total TAT: {total_lead_time/1440:.1f} days | PCE: {pce:.1f}%", xaxis=dict(title="Percentage of Total Lead Time", showgrid=False, range=[0, 100], ticksuffix="%"), yaxis=dict(visible=False), plot_bgcolor='white', margin=dict(l=20, r=20, t=50, b=20), height=300); return fig
def plot_pareto_chart() -> go.Figure:
    df = generate_pareto_data().sort_values('Frequency', ascending=False); df['Cumulative Percentage'] = df['Frequency'].cumsum() / df['Frequency'].sum() * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]]); fig.add_trace(go.Bar(x=df['QC_Failure_Mode'], y=df['Frequency'], name='Failure Count', marker_color=COLORS['primary']), secondary_y=False); fig.add_trace(go.Scatter(x=df['QC_Failure_Mode'], y=df['Cumulative Percentage'], name='Cumulative %', mode='lines+markers', line_color=COLORS['accent']), secondary_y=True); fig.add_hline(y=80, line=dict(color=COLORS['dark_gray'], dash='dot'), secondary_y=True)
    fig.update_layout(title_text="<b>Pareto Chart:</b> Identifying Top QC Failure Modes", xaxis_title="QC Failure Mode", plot_bgcolor='white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); fig.update_yaxes(title_text="Frequency", secondary_y=False); fig.update_yaxes(title_text="Cumulative Percentage", secondary_y=True, range=[0, 101], ticksuffix='%'); return fig
def plot_shewhart_chart(df: pd.DataFrame) -> go.Figure:
    mean, std_dev = df['Yield_ng'].iloc[:75].mean(), df['Yield_ng'].iloc[:75].std(ddof=1); ucl, lcl = mean + 3 * std_dev, mean - 3 * std_dev; violations = df[(df['Yield_ng'] > ucl) | (df['Yield_ng'] < lcl)]; fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Batch_ID'], y=df['Yield_ng'], mode='lines+markers', name='QC Control', line=dict(color=COLORS['primary']))); fig.add_trace(go.Scatter(x=[0, len(df) - 1], y=[ucl, ucl], mode='lines', name='UCL (Mean+3σ)', line=dict(color=COLORS['accent'], dash='dash'))); fig.add_trace(go.Scatter(x=[0, len(df) - 1], y=[mean, mean], mode='lines', name='Center Line', line=dict(color=COLORS['dark_gray'], dash='dot'))); fig.add_trace(go.Scatter(x=[0, len(df) - 1], y=[lcl, lcl], mode='lines', name='LCL (Mean-3σ)', line=dict(color=COLORS['accent'], dash='dash')))
    if not violations.empty: fig.add_trace(go.Scatter(x=violations['Batch_ID'], y=violations['Yield_ng'], mode='markers', name='Violation', marker=dict(color=COLORS['danger'], size=10, symbol='x')))
    fig.update_layout(title='<b>Levey-Jennings Chart:</b> Positive Control Monitoring', xaxis_title='Batch ID', yaxis_title='Yield (ng)', plot_bgcolor='white'); return fig
def plot_control_plan() -> go.Figure:
    data = {'Process Step': ['Library Prep', 'Sequencing', 'Bioinformatics'], 'Characteristic (X or Y)': ['Positive Control Yield (Y)', 'Sequencer Laser Power (X)', '% Mapped Reads (Y)'], 'Specification': ['20 ± 5 ng', '> 80 mW', '> 85%'], 'Tool': ['Fluorometer', 'Internal Sensor', 'FASTQC'], 'Method': ['Levey-Jennings', 'EWMA Chart', 'Shewhart Chart'], 'Frequency': ['Per Batch', 'Per Run', 'Per Sample'], 'Reaction Plan': ['Re-prep Batch', 'Schedule Maint.', 'Review Alignment']}; df = pd.DataFrame(data)
    fig = go.Figure(data=[go.Table(header=dict(values=list(df.columns), fill_color=COLORS['dark_gray'], font=dict(color='white'), align='left', height=40), cells=dict(values=[df[c] for c in df.columns], align='left', height=30))]); fig.update_layout(title="<b>Assay Control Plan:</b> Formalizing QC Procedures", margin=dict(l=10, r=10, t=50, b=10)); return fig
def plot_comparison_radar() -> go.Figure:
    categories = ['Interpretability', 'Data Volume Needs', 'Scalability', 'Handling Complexity', 'Biomarker Discovery', 'Regulatory Ease']; classical_scores = [5, 2, 1, 2, 1, 5]; ml_scores = [2, 5, 5, 5, 5, 2]; fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=classical_scores + [classical_scores[0]], theta=categories + [categories[0]], fill='toself', name='Classical DOE/Stats', marker_color=COLORS['primary'])); fig.add_trace(go.Scatterpolar(r=ml_scores + [ml_scores[0]], theta=categories + [categories[0]], fill='toself', name='ML / Bioinformatics', marker_color=COLORS['secondary']))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5.5])), showlegend=True, title="<b>Strengths Profile:</b> Classical vs. ML for Biotech R&D", legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)); return fig
def plot_verdict_barchart() -> go.Figure:
    data = {"Metric": ["Assay Parameter Optimization (DOE)", "Novel Biomarker Discovery", "High-Dimensional Data Analysis", "Analytical Validation (FDA)", "Proactive QC", "Protocol Interpretability"], "Winner": ["Classical", "ML", "ML", "Classical", "ML", "Classical"], "Score": [-1, 1, 1, -1, 1, -1]}; df = pd.DataFrame(data).sort_values('Score'); df['Color'] = df['Score'].apply(lambda x: COLORS['primary'] if x < 0 else COLORS['secondary'])
    fig = px.bar(df, x='Score', y='Metric', orientation='h', color='Color', color_discrete_map='identity', title="<b>Task-Specific Verdict:</b> Which Approach is Better?"); fig.update_layout(xaxis=dict(tickvals=[-1, 1], ticktext=['<b>Winner: Classical Stats</b>', '<b>Winner: Machine Learning</b>'], tickfont=dict(size=14), range=[-1.5, 1.5]), yaxis_title=None, plot_bgcolor='white', bargap=0.4, showlegend=False); return fig
def plot_synergy_diagram() -> go.Figure:
    fig = go.Figure(); fig.add_shape(type="circle", x0=0, y0=0, x1=2, y1=2, line_color=COLORS['primary'], fillcolor=COLORS['primary'], opacity=0.6); fig.add_shape(type="circle", x0=1.2, y0=0, x1=3.2, y1=2, line_color=COLORS['secondary'], fillcolor=COLORS['secondary'], opacity=0.6); fig.add_annotation(x=1, y=1, text="<b>Classical Stats</b><br><i>Inference & Causality</i><br><i>Rigor & Validation</i>", showarrow=False, font=dict(color="white", size=12)); fig.add_annotation(x=2.2, y=1, text="<b>Machine Learning</b><br><i>Prediction & Discovery</i><br><i>Complexity & Scale</i>", showarrow=False, font=dict(color="white", size=12)); fig.add_annotation(x=1.6, y=1, text="<b>Bio-AI<br>Excellence</b>", showarrow=False, font=dict(color="black", size=18, family="Arial Black"))
    fig.update_layout(title="<b>The Hybrid Philosophy:</b> Combining Strengths", xaxis=dict(visible=False, range=[-0.5, 3.7]), yaxis=dict(visible=False, range=[-0.5, 2.5]), plot_bgcolor='white', margin=dict(t=50, b=10, l=10, r=10)); return fig
def get_guidance_data() -> Dict[str, Dict[str, str]]:
    return { "Validating an assay for FDA 510(k) submission": {"approach": "🏆 **Classical Stats** (DOE, LoD/LoB studies, Gage R&R)", "rationale": "Methods are traceable, validated, and follow CLSI/FDA guidelines, which is paramount for regulatory bodies. The focus is on rigorous inference and establishing performance characteristics beyond reproach."}, "Discovering a new gene signature from RNA-Seq data": {"approach": "🏆 **Machine Learning** (Elastic Net, Random Forest with SHAP)", "rationale": "ML excels at feature selection from high-dimensional data (p >> n). It can identify a minimal, predictive set of genes from thousands of candidates, a task impossible for classical regression."}, "Optimizing a 12-parameter cell culture media": {"approach": "🏆 **Hybrid:** ML Model + Bayesian Optimization", "rationale": "A full factorial DOE is impossible (2^12 = 4096 runs). Instead, run a small space-filling DOE to train a Gaussian Process model (the 'digital twin' of the culture), then use Bayesian Optimization to find the optimal media composition *in silico*."}, "Monitoring daily QC for a clinical diagnostic lab": {"approach": "🏆 **Hybrid:** Levey-Jennings + EWMA + Multivariate Control", "rationale": "Use standard Levey-Jennings charts for regulatory compliance. Use more sensitive EWMA charts to detect slow reagent drift. Use a Hotelling's T² chart on the full QC profile to catch subtle, correlated shifts that individual charts would miss."}, "Identifying sources of contamination in a clean room from microbiome data": {"approach": "🏆 **Bioinformatics & ML** (PCA, Clustering, Source Tracking)", "rationale": "These are high-dimensional, complex datasets. Unsupervised learning methods are required to cluster samples, identify outlier signatures, and trace them back to potential environmental or personnel sources."}}
def get_workflow_css() -> str: return f""" <style> .workflow-container{{display:flex;flex-direction:column;align-items:center;width:100%;}} .workflow-step{{background-color:#FFFFFF;border:1px solid {COLORS['light_gray']};border-radius:10px;padding:20px;margin-bottom:20px;width:95%;box-shadow:0 4px 6px rgba(0,0,0,0.05);border-left:5px solid;}} .workflow-arrow{{font-size:28px;color:{COLORS['dark_gray']};margin-bottom:20px;font-weight:bold;}} .step-define{{border-left-color:{COLORS['primary']};}} .step-measure{{border-left-color:{COLORS['secondary']};}} .step-analyze{{border-left-color:{COLORS['accent']};}} .step-improve{{border-left-color:{COLORS['neutral_yellow']};}} .step-control{{border-left-color:{COLORS['neutral_pink']};}} .workflow-step h4{{margin-top:0;margin-bottom:15px;font-size:1.5em;color:#333333;}} .workflow-step .tool-col{{padding:0 15px;}} .workflow-step .tool-col h5{{color:#555555;border-bottom:2px solid #EEEEEE;padding-bottom:5px;margin-bottom:10px;}} .workflow-step .tool-col ul{{padding-left:20px;margin:0;}} .workflow-step .tool-col li{{margin-bottom:5px;}} .tool-col-classical h5{{color:{COLORS['primary']};}} .tool-col-ml h5{{color:{COLORS['secondary']};}} </style> """
def render_workflow_step(phase_name: str, phase_class: str, classical_tools: List[str], ml_tools: List[str]) -> str:
    classical_list = "".join([f"<li>{tool}</li>" for tool in classical_tools]); ml_list = "".join([f"<li>{tool}</li>" for tool in ml_tools])
    return f""" <div class="workflow-step {phase_class}"> <h4>{phase_name}</h4> <div style="display:flex;justify-content:space-between;"> <div style="flex:1;margin-right:10px;" class="tool-col tool-col-classical"><h5>Classical Tools (Rigor & Validation)</h5><ul>{classical_list}</ul></div> <div style="flex:1;margin-left:10px;" class="tool-col tool-col-ml"><h5>ML/Bio-AI Augmentation (Scale & Discovery)</h5><ul>{ml_list}</ul></div> </div> </div> """
