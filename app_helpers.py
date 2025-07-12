# app_helpers.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde, f_oneway, ttest_ind
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
# SECTION 2: SYNTHETIC DATA GENERATORS (ALL-INCLUSIVE)
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
def generate_rsm_data() -> pd.DataFrame:
    np.random.seed(42)
    temp = np.linspace(50, 70, 15); conc = np.linspace(1, 2, 15)
    T, C = np.meshgrid(temp, conc)
    yield_val = 90 - 0.1*(T-60)**2 - 20*(C-1.5)**2 - 0.5*(T-60)*(C-1.5) + np.random.normal(0, 2, T.shape)
    return pd.DataFrame({'Temperature': T.ravel(), 'Concentration': C.ravel(), 'Yield': yield_val.ravel()})
def generate_qfd_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    customer_reqs = ['High Sensitivity', 'High Specificity', 'Fast Turnaround', 'Low Cost']
    weights = pd.DataFrame({'Importance': [5, 5, 3, 4]}, index=customer_reqs)
    tech_chars = ['LOD (VAF %)', 'Specificity (%)', 'Hands-on Time (min)', 'Reagent Cost ($)']
    relationships = np.array([[9, 1, 3, 1], [1, 9, 1, 1], [1, 1, 9, 3], [1, 3, 3, 9]])
    rel_df = pd.DataFrame(relationships, index=customer_reqs, columns=tech_chars)
    return weights, rel_df
def generate_dfmea_data() -> pd.DataFrame:
    return pd.DataFrame([
        {'Potential Failure Mode': 'Incorrect material selected for sample well', 'Potential Effect': 'Sample Adsorption, low yield', 'Severity': 9, 'Potential Cause': 'Biocompatibility not verified', 'Occurrence': 3, 'Current Controls': 'Material Spec Sheet Review', 'Detection': 6},
        {'Potential Failure Mode': 'Fluidic channel geometry causes bubbles', 'Potential Effect': 'Flow obstruction, assay failure', 'Severity': 10, 'Potential Cause': 'Sharp corners in CAD model', 'Occurrence': 5, 'Current Controls': 'Visual Inspection', 'Detection': 7},
        {'Potential Failure Mode': 'Device housing cracks under stress', 'Potential Effect': 'Leakage, contamination', 'Severity': 7, 'Potential Cause': 'Low-grade polymer used', 'Occurrence': 2, 'Current Controls': 'Drop Test Protocol', 'Detection': 2}
    ]).assign(RPN=lambda df: df.Severity * df.Occurrence * df.Detection).sort_values('RPN', ascending=False)
def generate_capa_data() -> pd.DataFrame:
    return pd.DataFrame({
        "ID": ["CAPA-001", "CAPA-002", "CAPA-003", "CAPA-004", "CAPA-005", "CAPA-006", "CAPA-007"],
        "Description": [
            "Batch 2023-45 showed low yield. Investigation found the enzyme from Lot A was degraded due to improper storage in Freezer B.",
            "Contamination detected in negative controls for run 2023-11-02. Root cause traced to aerosolization from adjacent high-titer sample.",
            "Run 2023-11-05 failed due to thermal cycler block temperature overshoot. The cycler requires recalibration.",
            "System flagged low Q30 scores for samples 1-8. Pipetting error during library prep suspected, inconsistent volumes.",
            "Positive control failed for batch 2023-48. The control was stored at the wrong temperature, leading to DNA degradation.",
            "Unexpected peaks observed in chromatography. Re-running sample with fresh mobile phase resolved the issue. Suspect mobile phase degradation.",
            "Pipetting robot failed to dispense into well A3. Service log indicates a clogged nozzle. Required preventative maintenance was overdue."
        ]
    })
def generate_adverse_event_data() -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": range(1, 101),
        "description": 
            ['Patient experienced severe rash after starting treatment'] * 15 +
            ['Acute liver enzyme elevation noted in patient bloodwork'] * 10 +
            ['Patient reported mild headache and fatigue'] * 40 +
            ['Nausea and dizziness reported within 1 hour of dosage'] * 25 +
            ['Anaphylactic shock occurred; required epinephrine'] * 2 +
            ['Minor injection site irritation observed'] * 8
    })
def generate_risk_signal_data() -> pd.DataFrame:
    np.random.seed(42)
    # Cluster 1: High temp, low pressure
    c1 = pd.DataFrame(np.random.multivariate_normal([70, 2], [[5, -3], [-3, 3]], 30), columns=['Temp_C', 'Pressure_psi'])
    c1['Source'] = 'Manufacturing Line A'
    # Cluster 2: Med temp, med pressure
    c2 = pd.DataFrame(np.random.multivariate_normal([50, 5], [[4, 2], [2, 4]], 50), columns=['Temp_C', 'Pressure_psi'])
    c2['Source'] = 'Manufacturing Line B'
    # Outliers
    outliers = pd.DataFrame([[85, 1], [40, 10], [75, 8]], columns=['Temp_C', 'Pressure_psi'])
    outliers['Source'] = 'Anomalous Events'
    return pd.concat([c1, c2, outliers], ignore_index=True)
def generate_pccp_data() -> pd.DataFrame:
    np.random.seed(42); time = np.arange(100)
    performance = 0.95 - 0.0001 * time - 0.000005 * time**2 + np.random.normal(0, 0.005, 100)
    performance[70:] -= 0.05 # Simulate a sudden degradation event
    return pd.DataFrame({'Deployment_Day': time, 'Model_AUC': performance})
def generate_validation_data() -> pd.DataFrame:
    np.random.seed(42)
    data = {'Metric': ['Accuracy', 'Sensitivity', 'Specificity'],
            'Value': [0.95, 0.92, 0.97]}
    df = pd.DataFrame(data)
    # Simulate bootstrap results
    bootstrap_samples = {
        'Accuracy': np.random.normal(0.95, 0.02, 1000),
        'Sensitivity': np.random.normal(0.92, 0.04, 1000),
        'Specificity': np.random.normal(0.97, 0.015, 1000)
    }
    return df, bootstrap_samples
def generate_anova_data(means: list, stds: list, n: int) -> pd.DataFrame:
    data, groups = [], [];
    for i, (mean, std) in enumerate(zip(means, stds)):
        data.extend(np.random.normal(mean, std, n))
        groups.extend([f'Lot {chr(65+i)}'] * n)
    return pd.DataFrame({'Library_Yield': data, 'Reagent_Lot': groups})
def generate_pareto_data() -> pd.DataFrame:
    return pd.DataFrame({
        'QC_Failure_Mode': ['Low Library Yield', 'Adapter-Dimer Contamination', 'High Duplication Rate', 'Failed Positive Control', 'Low Q30 Score', 'Sample Mix-up'],
        'Frequency': [45, 22, 11, 6, 4, 2]
    })
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

# ==============================================================================
# SECTION 3: VISUALIZATION HELPERS (ALL-INCLUSIVE & SME-GRADE)
# ==============================================================================

def _create_network_fig(height=400, x_range=None, y_range=None) -> go.Figure:
    """Helper to create a blank, styled Plotly figure for network graphs."""
    fig = go.Figure()
    fig.update_layout(showlegend=False, plot_bgcolor='white', height=height,
                      xaxis=dict(showgrid=False, zeroline=False, visible=False, range=x_range),
                      yaxis=dict(showgrid=False, zeroline=False, visible=False, range=y_range),
                      margin=dict(t=40, b=20, l=20, r=20))
    return fig

def _add_network_nodes_and_edges(fig: go.Figure, nodes: Dict, edges: List[Tuple], annotations: bool = True):
    """SME Overhaul: Uses annotations as nodes for auto-sizing and professional styling."""
    for edge in edges:
        start_node, end_node = nodes[edge[0]], nodes[edge[1]]
        fig.add_trace(go.Scatter(x=[start_node['x'], end_node['x']], y=[start_node['y'], end_node['y']],
                                 mode='lines', line=dict(color=COLORS['light_gray'], width=2), hoverinfo='none'))
    if annotations:
        for node_id, node_data in nodes.items():
            fig.add_annotation(x=node_data['x'], y=node_data['y'],
                               text=f"<b>{node_data['text'].replace('<br>', '<br>')}</b>",
                               showarrow=False, font=dict(color=COLORS['text'], size=11, family="Arial"),
                               bgcolor=hex_to_rgba(node_data.get('color', COLORS['primary']), 0.15),
                               bordercolor=node_data.get('color', COLORS['primary']), borderwidth=2, borderpad=10, align="center")

def plot_qfd_house_of_quality() -> go.Figure:
    weights, rel_df = generate_qfd_data()
    tech_importance = (rel_df.T * weights['Importance'].values).T.sum()
    
    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.25, 0.75], row_heights=[0.75, 0.25],
        specs=[[{"type": "table"}, {"type": "heatmap"}], [None, {"type": "bar"}]],
        vertical_spacing=0.02, horizontal_spacing=0.02
    )
    # Heatmap
    fig.add_trace(go.Heatmap(z=rel_df.values, x=rel_df.columns, y=rel_df.index,
                             colorscale='Blues', text=rel_df.values, texttemplate="%{text}", showscale=False), row=1, col=2)
    # Customer Importance
    fig.add_trace(go.Table(
        header=dict(values=['<b>Customer Need</b>', '<b>Importance</b>'], fill_color=COLORS['dark_gray'], font=dict(color='white'), align='left'),
        cells=dict(values=[weights.index, weights.Importance], align='left', height=40,
                   fill_color=[[hex_to_rgba(COLORS['light_gray'], 0.2), 'white']*len(weights)])
    ), row=1, col=1)
    # Technical Importance
    fig.add_trace(go.Bar(x=tech_importance.index, y=tech_importance.values, marker_color=COLORS['primary'],
                         text=tech_importance.values, texttemplate='%{text:.0f}', textposition='outside'), row=2, col=2)
    
    fig.update_layout(
        title_text="<b>QFD 'House of Quality':</b> Translating VOC to Design Specs",
        plot_bgcolor='white', showlegend=False, margin=dict(l=10, r=10, t=50, b=10), height=500
    )
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text="<b>Technical Importance Score</b>", row=2, col=2, range=[0, max(tech_importance.values)*1.2])
    return fig

def plot_dfmea_table() -> go.Figure:
    df = generate_dfmea_data()
    column_widths = [4, 3, 1, 3, 1, 3, 1, 1]
    fig = go.Figure(data=[go.Table(
        columnorder=list(range(len(df.columns))), columnwidth=column_widths,
        header=dict(values=[f'<b>{col.replace(" ", "<br>")}</b>' for col in df.columns],
                    fill_color=COLORS['dark_gray'], font=dict(color='white'), align='center', height=50),
        cells=dict(values=[df[c] for c in df.columns], align='left', height=60,
                   fill_color=[[hex_to_rgba(COLORS['danger'], 0.5) if c == 'RPN' and v > 200 else
                               (hex_to_rgba(COLORS['warning'], 0.5) if c == 'RPN' and v > 100 else 'white')
                               for v in df[c]] for c in df.columns])
    )])
    fig.update_layout(title="<b>Design FMEA (DFMEA):</b> Proactive Risk Analysis of Device Design",
                      margin=dict(l=10, r=10, t=50, b=10), height=350)
    return fig

def plot_risk_signal_clusters() -> go.Figure:
    df = generate_risk_signal_data()
    model = DBSCAN(eps=3, min_samples=3).fit(df[['Temp_C', 'Pressure_psi']])
    df['cluster'] = [str(c) for c in model.labels_]
    df.loc[df['cluster'] == '-1', 'cluster'] = 'Outlier/Anomaly'

    fig = px.scatter(df, x='Temp_C', y='Pressure_psi', color='cluster', symbol='Source',
                     color_discrete_map={'0': COLORS['primary'], '1': COLORS['secondary'], 'Outlier/Anomaly': COLORS['danger']},
                     title="<b>ML Clustering of Process Data for Risk Signal Detection</b>")
    fig.update_layout(plot_bgcolor='white', legend_title="Identified Group",
                      xaxis_title="Temperature (°C)", yaxis_title="Pressure (psi)")
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    return fig

def plot_rsm_contour(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Contour(
        z=df['Yield'], x=df['Temperature'], y=df['Concentration'],
        colorscale='Viridis',
        contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=12, color='white'))
    ))
    max_yield_point = df.loc[df['Yield'].idxmax()]
    fig.add_trace(go.Scatter(x=[max_yield_point['Temperature']], y=[max_yield_point['Concentration']],
                             mode='markers', marker=dict(color=COLORS['danger'], size=15, symbol='star'),
                             name='Optimal Point'))
    fig.update_layout(title="<b>Response Surface Methodology (RSM):</b> Mapping the Design Space",
                      xaxis_title="Temperature (°C)", yaxis_title="Enzyme Concentration",
                      plot_bgcolor='white', showlegend=False)
    return fig
    
def plot_model_validation_ci() -> go.Figure:
    df, bootstrap_samples = generate_validation_data()
    fig = go.Figure()
    for i, row in df.iterrows():
        metric = row['Metric']
        samples = bootstrap_samples[metric]
        ci_95 = np.percentile(samples, [2.5, 97.5])
        fig.add_trace(go.Box(y=samples, name=metric, marker_color=COLORS['primary']))
        fig.add_annotation(x=metric, y=ci_95[1],
                           text=f"<b>95% CI:</b> [{ci_95[0]:.3f}, {ci_95[1]:.3f}]",
                           showarrow=False, yshift=20, font=dict(color=COLORS['dark_gray']))
    
    fig.update_layout(
        title="<b>ML Model Validation with Bootstrapped Confidence Intervals</b>",
        yaxis_title="Performance Metric Value",
        plot_bgcolor='white', showlegend=False
    )
    return fig
    
def plot_fault_tree_plotly() -> go.Figure:
    fig = _create_network_fig(height=500, x_range=[-0.5, 4.5], y_range=[-0.5, 5])
    nodes = {
        'top':     {'x': 2, 'y': 4.5, 'text': '<b>TOP EVENT</b><br>False Negative Result', 'color': COLORS['danger']},
        'or1':     {'x': 2, 'y': 3.5, 'text': 'OR Gate', 'color': COLORS['dark_gray']},
        'and1':    {'x': 0.5, 'y': 2, 'text': 'AND Gate', 'color': COLORS['dark_gray']},
        'assay':   {'x': 3.5, 'y': 2.5, 'text': 'Assay Failure', 'color': COLORS['primary']},
        'reagent': {'x': 0, 'y': 1, 'text': 'Reagent Degraded<br>P=0.01', 'color': COLORS['secondary']},
        'storage': {'x': 1, 'y': 1, 'text': 'Improper Storage<br>P=0.05', 'color': COLORS['secondary']},
        'sample':  {'x': 3.5, 'y': 1.5, 'text': 'Low DNA Input<br>P=0.02', 'color': COLORS['primary']}
    }
    edges = [('top', 'or1'), ('or1', 'and1'), ('or1', 'assay'), ('assay', 'sample'), ('and1', 'reagent'), ('and1', 'storage')]
    _add_network_nodes_and_edges(fig, nodes, edges)
    fig.update_layout(title="<b>Fault Tree Analysis (FTA):</b> Top-Down Risk Assessment")
    return fig
    
def plot_nlp_on_capa_logs() -> go.Figure:
    df = generate_capa_data()
    # Simulate topic modeling
    topics = {
        "Reagent/Storage Issue": "enzyme|degradation|lot|freezer|stored|mobile phase",
        "Contamination": "contamination|aerosolization|negative control",
        "Hardware Failure": "thermal cycler|calibration|overshoot|robot|clogged|nozzle",
        "Human Error": "pipetting|inconsistent volumes"
    }
    topic_counts = {topic: df['Description'].str.contains(pattern, case=False).sum() for topic, pattern in topics.items()}
    
    fig = px.bar(
        x=list(topic_counts.values()), 
        y=list(topic_counts.keys()),
        orientation='h',
        color=list(topic_counts.keys()),
        labels={'y': 'Identified Failure Theme', 'x': 'Frequency'},
        title="<b>NLP Topic Modeling on CAPA & Deviation Logs</b>",
        color_discrete_map={
            "Reagent/Storage Issue": COLORS['primary'], "Contamination": COLORS['accent'],
            "Hardware Failure": COLORS['danger'], "Human Error": COLORS['warning']
        }
    )
    fig.update_layout(plot_bgcolor='white', showlegend=False, yaxis={'categoryorder':'total ascending'})
    return fig

def plot_5whys_diagram() -> go.Figure:
    fig = _create_network_fig(height=550, y_range=[-0.5, 5.5])
    steps = [
        ('Problem', 'Low Library Yield on Plate 4'),
        ('Why 1?', 'Reagents added improperly.'),
        ('Why 2?', 'Technician used a miscalibrated multi-channel pipette.'),
        ('Why 3?', 'Pipette was overdue for its 6-month calibration.'),
        ('Why 4?', 'Calibration tracking system is a manual spreadsheet, prone to error.'),
        ('Root Cause', 'The asset management system is not robust enough to prevent use of out-of-spec equipment.')
    ]
    nodes, edges = {}, []
    for i, (level, text) in enumerate(steps):
        y_pos = 5 - i
        color = COLORS['danger'] if i == 0 else (COLORS['success'] if i == len(steps) - 1 else COLORS['primary'])
        nodes[f's{i}'] = {'x': 1, 'y': y_pos, 'text': f'<b>{level}</b><br>{text}', 'color': color}
        if i > 0: edges.append((f's{i-1}', f's{i}'))

    _add_network_nodes_and_edges(fig, nodes, edges)
    fig.update_layout(title="<b>5 Whys Analysis:</b> Drilling Down to the True Root Cause")
    return fig
    
def plot_pccp_monitoring() -> go.Figure:
    df = generate_pccp_data()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Deployment_Day'], y=df['Model_AUC'], mode='lines', name='Model Performance (AUC)', line=dict(color=COLORS['primary'])))
    fig.add_hline(y=0.90, line=dict(color=COLORS['danger'], dash='dash'), name='Performance Threshold')
    fig.add_vrect(x0=70, x1=100, fillcolor=hex_to_rgba(COLORS['warning'], 0.2), line_width=0, name="Performance Degradation")
    fig.add_annotation(x=85, y=0.87, text="<b>Retraining & Revalidation<br>Triggered per PCCP</b>", showarrow=True, arrowhead=1, ax=0, ay=-40, bgcolor="rgba(255,255,255,0.7)")
    fig.update_layout(
        title="<b>PCCP Monitoring for an AI/ML Device (SaMD)</b>",
        xaxis_title="Days Since Deployment", yaxis_title="Model Area Under Curve (AUC)",
        plot_bgcolor='white', legend=dict(x=0.01, y=0.01, yanchor='bottom')
    )
    return fig

def plot_adverse_event_clusters() -> go.Figure:
    df = generate_adverse_event_data()
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['description'])
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray())
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    df['x'] = X_pca[:, 0]
    df['y'] = X_pca[:, 1]
    
    cluster_names = {
        0: "Neurological (Headache, Dizziness)", 1: "Allergic / Skin Reaction",
        2: "Systemic (Liver, Anaphylaxis)", 3: "Gastrointestinal / Injection Site",
    }
    df['cluster_name'] = df['cluster'].map(cluster_names)
    
    fig = px.scatter(
        df, x='x', y='y', color='cluster_name', hover_data=['description'],
        title="<b>ML Clustering of Adverse Event Narratives for Signal Detection</b>", labels={'color': 'Event Cluster'}
    )
    fig.update_layout(
        xaxis_title="PCA Component 1", yaxis_title="PCA Component 2", plot_bgcolor='white',
        xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
        yaxis=dict(showticklabels=False, zeroline=False, showgrid=False)
    )
    return fig

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
    fig.update_layout(title='<b>Kano Model:</b> Prioritizing Diagnostic Features', xaxis_title='Feature Performance / Implementation →', yaxis_title='← Clinician Dissatisfaction ... Satisfaction →', plot_bgcolor='white', legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'))
    return fig

def plot_pareto_chart() -> go.Figure:
    df = generate_pareto_data().sort_values('Frequency', ascending=False)
    df['Cumulative Percentage'] = df['Frequency'].cumsum() / df['Frequency'].sum() * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df['QC_Failure_Mode'], y=df['Frequency'], name='Failure Count', marker_color=COLORS['primary']), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['QC_Failure_Mode'], y=df['Cumulative Percentage'], name='Cumulative %', mode='lines+markers', line_color=COLORS['accent']), secondary_y=True)
    fig.add_hline(y=80, line=dict(color=COLORS['dark_gray'], dash='dot'), secondary_y=True)
    fig.update_layout(title_text="<b>Pareto Chart:</b> Identifying Top QC Failure Modes", xaxis_title="QC Failure Mode", plot_bgcolor='white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
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
    fig.update_layout(title=f'<b>ANOVA:</b> Comparing Reagent Lot Performance', yaxis_title='Library Yield (ng/µL)', plot_bgcolor='white', showlegend=False)
    return fig, p_val

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

def plot_synergy_diagram() -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="circle", x0=0, y0=0, x1=2, y1=2, line_color=COLORS['primary'], fillcolor=COLORS['primary'], opacity=0.6)
    fig.add_shape(type="circle", x0=1.2, y0=0, x1=3.2, y1=2, line_color=COLORS['secondary'], fillcolor=COLORS['secondary'], opacity=0.6)
    fig.add_annotation(x=1, y=1, text="<b>Classical Stats</b><br><i>Inference & Causality</i><br><i>Rigor & Validation</i>", showarrow=False, font=dict(color="white", size=12))
    fig.add_annotation(x=2.2, y=1, text="<b>Machine Learning</b><br><i>Prediction & Discovery</i><br><i>Complexity & Scale</i>", showarrow=False, font=dict(color="white", size=12))
    fig.add_annotation(x=1.6, y=1, text="<b>Bio-AI<br>Excellence</b>", showarrow=False, font=dict(color="black", size=18, family="Arial Black"))
    fig.update_layout(title="<b>The Hybrid Philosophy:</b> Combining Strengths", xaxis=dict(visible=False, range=[-0.5, 3.7]), yaxis=dict(visible=False, range=[-0.5, 2.5]), plot_bgcolor='white', margin=dict(t=50, b=10, l=10, r=10))
    return fig
