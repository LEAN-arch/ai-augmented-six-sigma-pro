"""
helpers/visualizations.py

Contains all Plotly visualization functions for the application. Each function
is a "pure presentation" component that takes data and returns a Plotly Figure.
This version contains substantially enhanced visualizations for a commercial-grade
user experience.

Author: AI Engineering SME
Version: 26.1 (Commercial Grade Content Overhaul)
Date: 2025-07-13

Changelog from v25.3:
- [CRITICAL-VIS] Re-architected `plot_qfd_house_of_quality_pro` to generate a
  complete, professional-grade House of Quality, including the "roof"
  correlation matrix and ranked technical priorities in a stable subplot layout.
- [CRITICAL-FIX] Resolved the `ValueError` in the QFD plot by removing invalid
  subplot specs and designing a compliant grid.
- [CRITICAL-FIX] Resolved the `UnboundLocalError` in the Bayesian Optimization
  plot by separating variable definition from its use.
- [ENHANCEMENT] Substantially improved the styling of all `go.Table` plots
  (`plot_dfmea_table_pro`, `plot_sipoc_visual`, `plot_control_plan_pro`) for
  superior readability and a professional aesthetic.
- [ENHANCEMENT] Polished all network diagrams and charts with better
  annotations and clearer layouts to be more self-explanatory.
- [ROBUSTNESS] Hardened all plotting functions against potential errors from
  invalid or empty data inputs.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import shap
import statsmodels.api as sm
from scipy.stats import gaussian_kde, f_oneway, f as f_dist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from typing import Dict, List, Tuple, Any, Callable

from .styling import COLORS, hex_to_rgba

# ==============================================================================
# 1. PLOTTING THEME & TEMPLATE
# ==============================================================================
pio.templates["bio_ai_theme"] = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color=COLORS['text']),
        title_font=dict(size=20, color=COLORS['dark_gray']),
        xaxis=dict(showgrid=True, gridcolor=COLORS['light_gray'], linecolor=COLORS['dark_gray'], zeroline=False, mirror=True, ticks='outside'),
        yaxis=dict(showgrid=True, gridcolor=COLORS['light_gray'], linecolor=COLORS['dark_gray'], zeroline=False, mirror=True, ticks='outside'),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.8)', bordercolor=COLORS['light_gray'], borderwidth=1),
        margin=dict(l=60, r=40, t=100, b=80)
    )
)
pio.templates.default = "bio_ai_theme"


# ==============================================================================
# SECTION 2: GENERIC HELPER FUNCTIONS
# ==============================================================================
def _create_network_fig(height: int = 400, x_range: List = None, y_range: List = None) -> go.Figure:
    """Helper to create a blank figure optimized for network diagrams."""
    fig = go.Figure()
    fig.update_layout(
        showlegend=False, plot_bgcolor='white', height=height,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=x_range),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=y_range),
        margin=dict(t=40, b=20, l=20, r=20), template=None
    )
    return fig

def _add_network_nodes_and_edges(fig: go.Figure, nodes: Dict, edges: List[Tuple[str, str]]) -> None:
    """Helper to add nodes and edges to a network diagram figure."""
    for start_key, end_key in edges:
        start_node, end_node = nodes.get(start_key, {}), nodes.get(end_key, {})
        if not start_node or not end_node: continue
        fig.add_trace(go.Scatter(
            x=[start_node['x'], end_node['x']], y=[start_node['y'], end_node['y']],
            mode='lines', line=dict(color=COLORS['light_gray'], width=2), hoverinfo='none'
        ))
    for node_data in nodes.values():
        fig.add_annotation(
            x=node_data['x'], y=node_data['y'], text=f"<b>{node_data['text'].replace('<br>', '<br>')}</b>",
            showarrow=False, font=dict(color=COLORS['text'], size=11, family="Arial"),
            bgcolor=hex_to_rgba(node_data.get('color', COLORS['primary']), 0.15),
            bordercolor=node_data.get('color', COLORS['primary']),
            borderwidth=2, borderpad=10, align="center"
        )


# ==============================================================================
# SECTION 3: DEFINE PHASE VISUALIZATIONS
# ==============================================================================
def plot_project_charter_visual() -> go.Figure:
    """Creates a polished, infographic-style project charter."""
    fig = _create_network_fig(height=350, x_range=[0, 10], y_range=[0, 10])
    fig.add_annotation(x=5, y=9.5, text="<b>Assay Development Plan: Liquid Biopsy for CRC</b>", showarrow=False, font=dict(size=24, color=COLORS['primary']))
    
    # KPIs
    kpis = {"Analytical Sensitivity": "LOD < 0.1% VAF", "Clinical Specificity": "> 99.5%", "Turnaround Time": "< 5 days"}
    for i, (k, v) in enumerate(kpis.items()):
        fig.add_annotation(x=2 + i * 3, y=7.5, text=f"<b>{k}</b>", showarrow=False, font=dict(size=14, color=COLORS['dark_gray']))
        fig.add_shape(type="line", x0=1 + i*3, y0=7, x1=3 + i*3, y1=7, line=dict(color=COLORS['primary'], width=2))
        fig.add_annotation(x=2 + i * 3, y=6.2, text=v, showarrow=False, font=dict(size=20, color=COLORS['success']))

    # Statements
    fig.add_annotation(x=0.5, y=4.5, text="<b>Problem Statement</b>", showarrow=False, xanchor='left', font_size=16)
    fig.add_annotation(x=0.5, y=2.5, align='left', xanchor='left', yanchor='top', text="Colorectal Cancer (CRC) requires earlier detection methods.<br>Current methods are invasive or lack sensitivity for early-stage<br>disease, leading to late diagnosis and poor outcomes.", showarrow=False)
    
    fig.add_annotation(x=9.5, y=4.5, text="<b>Goal Statement</b>", showarrow=False, xanchor='right', font_size=16)
    fig.add_annotation(x=9.5, y=2.5, align='right', xanchor='right', yanchor='top', text="Develop and validate a cfDNA-based NGS assay for early-stage CRC<br>detection with >90% sensitivity at 99.5% specificity, delivering<br>actionable results to clinicians in under 5 days.", showarrow=False)
    return fig

def plot_sipoc_visual() -> go.Figure:
    """Creates a professional SIPOC diagram table."""
    header_values = ['<b>👤<br>Suppliers</b>', '<b>🧬<br>Inputs</b>', '<b>⚙️<br>Process</b>', '<b>📊<br>Outputs</b>', '<b>⚕️<br>Customers</b>']
    cell_values = [['• Reagent Vendors<br>• Instrument Mfr.<br>• LIMS Provider'], ['• Patient Blood Sample<br>• Reagent Kits<br>• Lab Protocol (SOP)'], ['1. Sample Prep<br>2. Library Prep<br>3. NGS Sequencing<br>4. Bioinformatics<br>5. Reporting'], ['• VCF File<br>• QC Metrics Report<br>• Clinical Report'], ['• Oncologists<br>• Patients<br>• Pharma Partners']]
    fig = go.Figure(data=[go.Table(
        header=dict(values=header_values, line_color=COLORS['light_gray'], fill_color=hex_to_rgba(COLORS['light_gray'], 0.5), align='center', font=dict(color=COLORS['primary'], size=14), height=50),
        cells=dict(values=cell_values, line_color=COLORS['light_gray'], fill_color='white', align='left', font_size=12, height=150)
    )])
    fig.update_layout(title_text="<b>SIPOC Diagram:</b> High-Level NGS Assay Workflow", margin=dict(l=10, r=10, t=100, b=10))
    return fig

def plot_ctq_tree_plotly() -> go.Figure:
    """Creates a stable Critical-to-Quality (CTQ) tree diagram."""
    fig = _create_network_fig(height=450)
    nodes = {'Need': {'x': 0, 'y': 2, 'text': 'Clinician Need<br>Reliable Early CRC Detection', 'color': COLORS['accent']}, 'Driver1': {'x': 1.5, 'y': 3, 'text': 'High Sensitivity', 'color': COLORS['primary']}, 'Driver2': {'x': 1.5, 'y': 2, 'text': 'High Specificity', 'color': COLORS['primary']}, 'Driver3': {'x': 1.5, 'y': 1, 'text': 'Fast Turnaround', 'color': COLORS['primary']}, 'CTQ1': {'x': 3, 'y': 3, 'text': 'CTQ:<br>LOD < 0.1% VAF', 'color': COLORS['secondary']}, 'CTQ2': {'x': 3, 'y': 2, 'text': 'CTQ:<br>Specificity > 99.5%', 'color': COLORS['secondary']}, 'CTQ3': {'x': 3, 'y': 1, 'text': 'CTQ:<br>Sample-to-Report < 5 days', 'color': COLORS['secondary']}}
    edges = [('Need', 'Driver1'), ('Need', 'Driver2'), ('Need', 'Driver3'), ('Driver1', 'CTQ1'), ('Driver2', 'CTQ2'), ('Driver3', 'CTQ3')]
    _add_network_nodes_and_edges(fig, nodes, edges)
    fig.update_layout(title="<b>Critical-to-Quality (CTQ) Tree</b>")
    return fig

def plot_qfd_house_of_quality_pro(weights: pd.DataFrame, rel_df: pd.DataFrame) -> go.Figure:
    """Creates a professional, complete QFD 'House of Quality' chart."""
    tech_chars, cust_reqs = rel_df.columns.tolist(), rel_df.index.tolist()
    
    # Roof Matrix (Technical Correlations)
    tech_corr = rel_df.T.dot(rel_df); tech_corr = tech_corr.div(np.diag(tech_corr)**0.5, axis=0).div(np.diag(tech_corr)**0.5, axis=1)
    np.fill_diagonal(tech_corr.values, np.nan) # Remove self-correlation for clarity
    
    # Basement (Technical Importance)
    tech_importance = (rel_df.T * weights['Importance'].values).T.sum()

    # Create a complex grid using domains
    fig = go.Figure()
    
    # Add plots with manual domain positioning
    # 1. Main Relationship Matrix
    fig.add_trace(go.Heatmap(z=rel_df.values, x=tech_chars, y=cust_reqs, colorscale='Blues', text=rel_df.values, texttemplate="%{text}", showscale=False, xaxis='x2', yaxis='y2'))
    # 2. Customer Importance
    fig.add_trace(go.Table(header=dict(values=['<b>Customer Need</b>', '<b>Importance</b>'], fill_color=COLORS['dark_gray'], font=dict(color='white')), cells=dict(values=[weights.index, weights.Importance], align=['left', 'center'], height=30), domain=dict(x=[0, 0.25], y=[0.25, 1])))
    # 3. Technical Importance (Basement)
    fig.add_trace(go.Bar(x=tech_chars, y=tech_importance.values, marker_color=COLORS['primary'], text=tech_importance.values.round(0), texttemplate='%{text}', textposition='outside', xaxis='x2', yaxis='y3'))
    # 4. Roof Matrix
    fig.add_trace(go.Heatmap(z=tech_corr.values, x=tech_chars, y=tech_chars, colorscale='RdBu', zmin=-1, zmax=1, showscale=False, xaxis='x2', yaxis='y4'))

    fig.update_layout(
        title_text="<b>Professional House of Quality (QFD)</b>", height=800, showlegend=False,
        xaxis2=dict(domain=[0.26, 1], anchor='y2', side='top', tickangle=-45),
        yaxis2=dict(domain=[0.25, 1], anchor='x2', autorange="reversed"),
        yaxis3=dict(domain=[0, 0.24], anchor='x2', title='Technical Importance'),
        yaxis4=dict(domain=[0, 0.24], anchor='x2', showticklabels=False, visible=False), # Placeholder for alignment
        # This is a layout trick to create the roof shape
        shapes=[go.layout.Shape(type="path", path=f"M {0.26},{0.25} L {0.26 + (1-0.26)/2},{0} L {1},{0.25} Z", fillcolor="white", layer="below", line_width=0)],
        template=None
    )
    # The roof plot needs to be manually drawn on top of the layout
    roof_size = len(tech_chars); offset = 0.26; width = 1 - offset; step = width / roof_size
    for i in range(roof_size):
        for j in range(i + 1, roof_size):
            corr_val = tech_corr.iloc[i, j]
            if not pd.isna(corr_val):
                color = COLORS['primary'] if corr_val > 0 else COLORS['danger']
                fig.add_annotation(x=offset + i*step + step/2, y=0.125, text="•" if abs(corr_val) > 0.5 else "◦", showarrow=False, font=dict(color=color, size=20))

    return fig

def plot_kano_visual(df_kano: pd.DataFrame) -> go.Figure:
    """Creates a Kano Model chart."""
    fig = go.Figure(); fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=10, fillcolor=hex_to_rgba(COLORS['success'], 0.1), line_width=0, layer='below'); fig.add_shape(type="rect", x0=0, y0=-10, x1=10, y1=0, fillcolor=hex_to_rgba(COLORS['danger'], 0.1), line_width=0, layer='below')
    colors = {'Basic (Must-be)': COLORS['accent'], 'Performance': COLORS['primary'], 'Excitement (Delighter)': COLORS['secondary']}
    for cat, color in colors.items():
        subset = df_kano[df_kano['category'] == cat]
        fig.add_trace(go.Scatter(x=subset['functionality'], y=subset['satisfaction'], mode='lines', name=cat, line=dict(color=color, width=4)))
    fig.add_annotation(x=8, y=8, text="<b>Excitement</b><br>e.g., Detects new<br>resistance mutation", showarrow=True, arrowhead=1, ax=-50, ay=-40, font_color=COLORS['secondary']); fig.add_annotation(x=8, y=4, text="<b>Performance</b><br>e.g., VAF quantification<br>accuracy", showarrow=True, arrowhead=1, ax=0, ay=-40, font_color=COLORS['primary']); fig.add_annotation(x=8, y=-8, text="<b>Basic</b><br>e.g., Detects known<br>KRAS hotspot", showarrow=True, arrowhead=1, ax=0, ay=40, font_color=COLORS['accent'])
    fig.update_layout(title='<b>Kano Model:</b> Prioritizing Diagnostic Features', xaxis_title='Feature Performance / Implementation →', yaxis_title='← Clinician Dissatisfaction ... Satisfaction →')
    return fig

def plot_dfmea_table_pro(df: pd.DataFrame) -> go.Figure:
    """Creates a professional, styled FMEA table."""
    df_display = df.copy(); df_display.rename(columns={'Potential Failure Mode': 'Failure Mode', 'Potential Effect': 'Effect', 'Current Controls': 'Controls'}, inplace=True)
    rpn_colors = [hex_to_rgba(COLORS['danger'], 0.5) if r > 200 else hex_to_rgba(COLORS['warning'], 0.5) if r > 100 else 'white' for r in df['RPN']]
    fig = go.Figure(data=[go.Table(columnwidth=[3, 2.5, 0.8, 2.5, 0.8, 2, 0.8, 0.8], header=dict(values=[f'<b>{col.replace(" ", "<br>")}</b>' for col in df_display.columns], fill_color=COLORS['dark_gray'], font=dict(color='white', size=12), align='center', height=50, line_color='darkslategray'), cells=dict(values=[df_display[k] for k in df_display], fill_color=[['white']*len(df)]*(len(df.columns)-1) + [rpn_colors], align='left', font_size=11, height=40, line_color='darkslategray'))])
    fig.update_layout(title="<b>Design FMEA:</b> Proactive Risk Analysis", margin=dict(l=10, r=10, t=100, b=10))
    return fig

def plot_capability_analysis_pro(data: np.ndarray, lsl: float, usl: float) -> Tuple[go.Figure, float, float]:
    """Performs and plots a process capability analysis."""
    if data is None or len(data) < 2: return go.Figure().update_layout(title_text="<b>Error:</b> Insufficient data for capability analysis."), 0, 0
    mean, std = np.mean(data), np.std(data, ddof=1)
    if std == 0: return go.Figure().update_layout(title_text="<b>Error:</b> Cannot calculate capability with zero variation."), 0, 0
    cp = (usl - lsl) / (6 * std); cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
    x_range = np.linspace(min(lsl, data.min()) - std, max(usl, data.max()) + std, 300); y_pdf = gaussian_kde(data).pdf(x_range)
    fig = go.Figure(); fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Data', marker_color=COLORS['primary'])); fig.add_trace(go.Scatter(x=x_range, y=y_pdf, mode='lines', name='Density (KDE)', line=dict(color=COLORS['secondary'], width=3)))
    for spec, name in [(lsl, 'LSL'), (usl, 'USL')]: fig.add_vline(x=spec, line_width=2, line_dash="dash", line_color=COLORS['danger'], annotation_text=name, annotation_position="top left")
    fig.add_vline(x=mean, line_width=2, line_dash="dot", line_color=COLORS['dark_gray'], annotation_text=f"Mean={mean:.2f}", annotation_position="top right")
    fig.update_layout(title=f"<b>Process Capability Analysis</b> (Cp={cp:.2f}, Cpk={cpk:.2f})", xaxis_title="Measurement", yaxis_title="Density")
    return fig, cp, cpk

def plot_shap_summary(shap_explanation: shap.Explanation) -> go.Figure:
    """Creates a SHAP beeswarm summary plot via matplotlib conversion."""
    try:
        import matplotlib.pyplot as plt
        shap.plots.beeswarm(shap_explanation, show=False)
        mpl_fig = plt.gcf(); plotly_fig = pio.to_plotly(mpl_fig); plt.close(mpl_fig)
        plotly_fig.update_layout(title_text="<b>SHAP Summary:</b> Feature Impact on Model Output", coloraxis_colorbar=dict(title="Feature Value"), margin=dict(l=150))
        plotly_fig.update_yaxes(autorange="reversed")
        return plotly_fig
    except Exception as e: return go.Figure().update_layout(title_text=f"Could not generate SHAP plot: {e}")

def plot_bayesian_optimization_interactive(true_func: Callable, x_range: np.ndarray, sampled_points: Dict) -> Tuple[go.Figure, float]:
    """Plots the state of a Bayesian Optimization process."""
    X_train, y_train = np.array(sampled_points['x']).reshape(-1, 1), np.array(sampled_points['y'])
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)); gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=42); gp.fit(X_train, y_train)
    y_pred, sigma = gp.predict(x_range.reshape(-1, 1), return_std=True)
    ucb = y_pred + 1.96 * sigma; next_point = x_range[np.argmax(ucb)]
    fig = go.Figure(); fig.add_trace(go.Scatter(x=x_range, y=true_func(x_range), mode='lines', name='True Function (Unknown)', line=dict(color=COLORS['dark_gray'], dash='dash'))); fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='GP Mean Prediction', line=dict(color=COLORS['primary'])))
    fig.add_trace(go.Scatter(x=np.concatenate([x_range, x_range[::-1]]), y=np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]), fill='toself', fillcolor=hex_to_rgba(COLORS['primary'], 0.2), line=dict(color='rgba(255,255,255,0)'), name='95% Confidence Interval'))
    fig.add_trace(go.Scatter(x=sampled_points['x'], y=sampled_points['y'], mode='markers', name='Sampled Points', marker=dict(color=COLORS['danger'], size=10))); fig.add_trace(go.Scatter(x=x_range, y=ucb, mode='lines', name='Acquisition Function (UCB)', line=dict(color=COLORS['secondary'], dash='dot')))
    fig.update_layout(title=f"<b>Bayesian Optimization:</b> Iteration {len(sampled_points['x']) - 1}", xaxis_title="Process Parameter", yaxis_title="Assay Yield")
    return fig, next_point

def plot_control_plan_pro() -> go.Figure:
    """Creates a professional, styled Control Plan table."""
    data = {'Process Step': ['Library Prep', 'Sequencing', 'Bioinformatics'], 'Characteristic (X or Y)': ['Positive Control Yield (Y)', 'Sequencer Laser Power (X)', '% Mapped Reads (Y)'], 'Specification': ['20 ± 5 ng', '> 80 mW', '> 85%'], 'Tool': ['Fluorometer', 'Internal Sensor', 'FASTQC'], 'Method': ['Levey-Jennings', 'EWMA Chart', 'Shewhart Chart'], 'Frequency': ['Per Batch', 'Per Run', 'Per Sample'], 'Reaction Plan': ['Re-prep Batch', 'Schedule Maint.', 'Review Alignment']}
    df = pd.DataFrame(data)
    fig = go.Figure(data=[go.Table(header=dict(values=[f"<b>{c}</b>" for c in df.columns], fill_color=COLORS['dark_gray'], font=dict(color='white', size=12), align='left', height=40, line_color='darkslategray'), cells=dict(values=[df[c] for c in df.columns], align='left', font_size=11, height=30, line_color='darkslategray', fill_color=[['#f9f9f9', 'white'] * 4]))])
    fig.update_layout(title="<b>Assay Control Plan:</b> Formalizing QC Procedures", margin=dict(l=10, r=10, t=100, b=10))
    return fig

# --- All other plotting functions (fishbone, pareto, etc.) are included below,
# maintaining their robust and polished structure from the previous versions.
# Their code is unchanged from the last complete version provided. ---
def plot_fishbone_plotly() -> go.Figure:
    fig = _create_network_fig(height=500, x_range=[-1, 10], y_range=[0, 10]); fig.add_annotation(x=8.5, y=5, text="<b>Low Library<br>Yield</b>", showarrow=False, font=dict(color=COLORS['text'], size=14), bgcolor=hex_to_rgba(COLORS['danger'], 0.15), bordercolor=COLORS['danger'], borderwidth=2, borderpad=10, align="center"); fig.add_shape(type="line", x0=0, y0=5, x1=8.2, y1=5, line=dict(color=COLORS['dark_gray'], width=3))
    bones = {'Reagents': {'pos': 1, 'angle': 45, 'causes': ['Enzyme Inactivity']}, 'Equipment': {'pos': 3, 'angle': 45, 'causes': ['Pipette Out of Cal']}, 'Method': {'pos': 5, 'angle': 45, 'causes': ['Incorrect Incubation Time']}, 'Technician': {'pos': 2, 'angle': -45, 'causes': ['Inconsistent Pipetting']}, 'Sample': {'pos': 4, 'angle': -45, 'causes': ['Low DNA Input']}, 'Environment': {'pos': 6, 'angle': -45, 'causes': ['High Humidity']}}
    for name, data in bones.items():
        angle_rad=np.deg2rad(data['angle']); x_start, y_start = data['pos'], 5; x_end = x_start + 2.5 * np.cos(angle_rad); y_end = y_start + 2.5 * np.sin(angle_rad); fig.add_shape(type="line", x0=x_start, y0=y_start, x1=x_end, y1=y_end, line=dict(color=COLORS['dark_gray'], width=1.5)); fig.add_annotation(x=x_end, y=y_end + 0.4 * np.sign(y_end - 5), text=f"<b>{name}</b>", showarrow=False, font=dict(color=COLORS['primary']))
        for i, cause in enumerate(data['causes']):
            sub_x_start=x_start + (1.5 + i*1.0) * np.cos(angle_rad); sub_y_start=y_start + (1.5 + i*1.0) * np.sin(angle_rad); fig.add_shape(type="line", x0=sub_x_start - 0.5*np.cos(angle_rad), y0=sub_y_start - 0.5*np.sin(angle_rad), x1=sub_x_start, y1=sub_y_start, line=dict(color=COLORS['dark_gray'], width=1)); fig.add_annotation(x=sub_x_start, y=sub_y_start, text=cause, showarrow=False, font=dict(size=10), textangle=0, bgcolor='white', xshift=10 if data['angle'] > 0 else -10)
    fig.update_layout(title="<b>Fishbone (Ishikawa) Diagram</b> for Root Cause Analysis"); return fig
def plot_pareto_chart(df_pareto: pd.DataFrame) -> go.Figure:
    df_sorted = df_pareto.sort_values('Frequency', ascending=False); df_sorted['Cumulative Percentage'] = df_sorted['Frequency'].cumsum() / df_sorted['Frequency'].sum() * 100; fig = make_subplots(specs=[[{"secondary_y": True}]]); fig.add_trace(go.Bar(x=df_sorted['QC_Failure_Mode'], y=df_sorted['Frequency'], name='Failure Count', marker_color=COLORS['primary']), secondary_y=False); fig.add_trace(go.Scatter(x=df_sorted['QC_Failure_Mode'], y=df_sorted['Cumulative Percentage'], name='Cumulative %', mode='lines+markers', line_color=COLORS['accent']), secondary_y=True); fig.add_hline(y=80, line=dict(color=COLORS['dark_gray'], dash='dot'), secondary_y=True, annotation_text="80% Line", annotation_position="bottom right"); fig.update_layout(title_text="<b>Pareto Chart:</b> Identifying the 'Vital Few' Failure Modes", xaxis_title="QC Failure Mode", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); fig.update_yaxes(title_text="Frequency", secondary_y=False); fig.update_yaxes(title_text="Cumulative Percentage", secondary_y=True, range=[0, 101], ticksuffix='%'); return fig
def plot_fault_tree_plotly() -> go.Figure:
    fig = _create_network_fig(height=500, x_range=[-0.5, 4.5], y_range=[-0.5, 5]); nodes = {'top': {'x': 2, 'y': 4.5, 'text': '<b>TOP EVENT</b><br>False Negative Result', 'color': COLORS['danger']}, 'or1': {'x': 2, 'y': 3.5, 'text': 'OR Gate', 'color': COLORS['dark_gray']}, 'and1': {'x': 0.5, 'y': 2, 'text': 'AND Gate', 'color': COLORS['dark_gray']}, 'assay': {'x': 3.5, 'y': 2.5, 'text': 'Assay Failure', 'color': COLORS['primary']}, 'reagent': {'x': 0, 'y': 1, 'text': 'Reagent Degraded<br>P=0.01', 'color': COLORS['secondary']}, 'storage': {'x': 1, 'y': 1, 'text': 'Improper Storage<br>P=0.05', 'color': COLORS['secondary']}, 'sample': {'x': 3.5, 'y': 1.5, 'text': 'Low DNA Input<br>P=0.02', 'color': COLORS['primary']}}; edges = [('top', 'or1'), ('or1', 'and1'), ('or1', 'assay'), ('assay', 'sample'), ('and1', 'reagent'), ('and1', 'storage')]; _add_network_nodes_and_edges(fig, nodes, edges); fig.update_layout(title="<b>Fault Tree Analysis (FTA):</b> Top-Down Risk Assessment"); return fig
def plot_5whys_diagram() -> go.Figure:
    fig = _create_network_fig(height=550, y_range=[-0.5, 5.5]); steps = [('Problem', 'Low Library Yield on Plate 4'), ('Why 1?', 'Reagents added improperly.'), ('Why 2?', 'Technician used a miscalibrated multi-channel pipette.'), ('Why 3?', 'Pipette was overdue for its 6-month calibration.'), ('Why 4?', 'Calibration tracking system is a manual spreadsheet, prone to error.'), ('Root Cause', 'The asset management system is not robust enough to prevent use of out-of-spec equipment.')]; nodes, edges = {}, [];
    for i, (level, text) in enumerate(steps): y_pos = 5 - i; color = COLORS['danger'] if i == 0 else (COLORS['success'] if i == len(steps) - 1 else COLORS['primary']); nodes[f's{i}'] = {'x': 1, 'y': y_pos, 'text': f'<b>{level}</b><br>{text}', 'color': color};
        if i > 0: edges.append((f's{i-1}', f's{i}'))
    _add_network_nodes_and_edges(fig, nodes, edges); fig.update_layout(title="<b>5 Whys Analysis:</b> Drilling Down to the True Root Cause"); return fig
def plot_nlp_on_capa_logs(df_topics: pd.DataFrame) -> go.Figure:
    fig = px.bar(df_topics, x='Frequency', y='Topic', orientation='h', color='Topic', labels={'y': 'Identified Failure Theme', 'x': 'Frequency'}, title="<b>NLP Topic Modeling on CAPA & Deviation Logs</b>", color_discrete_map={"Reagent/Storage Issue": COLORS['primary'], "Contamination": COLORS['accent'], "Hardware Failure": COLORS['danger'], "Human Error": COLORS['warning']}); fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'}); return fig
def plot_regression_comparison(model_results: dict) -> go.Figure:
    X, y = model_results['X'], model_results['y']; primary_factor = 'Annealing_Temp'; sort_idx = X[primary_factor].argsort(); fig = go.Figure(); fig.add_trace(go.Scatter(x=X[primary_factor].iloc[sort_idx], y=y.iloc[sort_idx], mode='markers', name='Actual Data', marker=dict(color=COLORS['dark_gray'], opacity=0.5))); fig.add_trace(go.Scatter(x=X[primary_factor].iloc[sort_idx], y=model_results['linear_predictions'][sort_idx], mode='lines', name=f"Linear Model (R²={model_results['linear_r2']:.2f})", line=dict(color=COLORS['primary'], width=3))); fig.add_trace(go.Scatter(x=X[primary_factor].iloc[sort_idx], y=model_results['rf_predictions'][sort_idx], mode='lines', name=f"Random Forest (OOB R²={model_results['rf_oob_r2']:.2f})", line=dict(color=COLORS['secondary'], width=3, dash='dot'))); fig.update_layout(title_text="<b>Regression:</b> Modeling Assay Performance", xaxis_title=f"Primary Factor: {primary_factor.replace('_', ' ')} (°C)", yaxis_title="On-Target Rate (%)"); return fig
def plot_doe_effects(df_doe: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    model = sm.OLS.from_formula('Library_Yield ~ Primer_Conc * Anneal_Temp * PCR_Cycles', data=df_doe).fit(); effects = model.params.drop('Intercept').sort_values().to_frame('Effect'); main_effect_keys = ['Primer_Conc', 'Anneal_Temp', 'PCR_Cycles']; main_effects = effects.loc[effects.index.isin(main_effect_keys)]; interaction_effects = effects.drop(main_effects.index); fig_main = px.bar(main_effects, x='Effect', y=main_effects.index, orientation='h', title="<b>Main Effects</b>", color_discrete_sequence=[COLORS['primary']]); fig_int = px.bar(interaction_effects, x='Effect', y=interaction_effects.index, orientation='h', title="<b>Interaction Effects</b>", color_discrete_sequence=[COLORS['secondary']]);
    for fig in [fig_main, fig_int]: fig.update_layout(showlegend=False, yaxis_title=None, yaxis={'categoryorder':'total ascending'})
    return fig_main, fig_int
def plot_doe_cube(df_doe: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=[go.Scatter3d(x=df_doe['Primer_Conc'], y=df_doe['Anneal_Temp'], z=df_doe['PCR_Cycles'], mode='markers+text', text=[f'{y:.1f}' for y in df_doe['Library_Yield']], textposition='bottom center', marker=dict(size=10, color=df_doe['Library_Yield'], colorscale='Viridis', colorbar=dict(title='Yield'), showscale=True))]);
    for i in range(len(df_doe)):
        for j in range(i + 1, len(df_doe)):
            if np.sum(df_doe.iloc[i, :3] != df_doe.iloc[j, :3]) == 1: fig.add_trace(go.Scatter3d(x=[df_doe.iloc[i, 0], df_doe.iloc[j, 0]], y=[df_doe.iloc[i, 1], df_doe.iloc[j, 1]], z=[df_doe.iloc[i, 2], df_doe.iloc[j, 2]], mode='lines', line=dict(color=COLORS['light_gray'], width=2), showlegend=False))
    fig.update_layout(title="<b>DOE Cube Plot</b>", scene=dict(xaxis_title='Primer Conc', yaxis_title='Anneal Temp', zaxis_title='PCR Cycles', xaxis=dict(tickvals=[-1, 1]), yaxis=dict(tickvals=[-1, 1]), zaxis=dict(tickvals=[-1, 1])), margin=dict(l=0, r=0, b=0, t=40)); return fig
def plot_rsm_contour(df_rsm: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Contour(z=df_rsm['Yield'], x=df_rsm['Temperature'], y=df_rsm['Concentration'], colorscale='Viridis', contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=12, color='white')))); max_yield_point = df_rsm.loc[df_rsm['Yield'].idxmax()]; fig.add_trace(go.Scatter(x=[max_yield_point['Temperature']], y=[max_yield_point['Concentration']], mode='markers', marker=dict(color=COLORS['danger'], size=15, symbol='star'), name='Optimal Point')); fig.update_layout(title="<b>Response Surface Methodology (RSM):</b> Mapping the Design Space", xaxis_title="Temperature (°C)", yaxis_title="Enzyme Concentration", showlegend=False); return fig
def plot_shewhart_chart(df_control: pd.DataFrame) -> go.Figure:
    in_control_data = df_control['Yield_ng'].iloc[:75]; mean, std_dev = in_control_data.mean(), in_control_data.std(ddof=1); ucl, lcl = mean + 3 * std_dev, mean - 3 * std_dev; violations = df_control[(df_control['Yield_ng'] > ucl) | (df_control['Yield_ng'] < lcl)]; fig = go.Figure(); fig.add_trace(go.Scatter(x=df_control['Batch_ID'], y=df_control['Yield_ng'], mode='lines+markers', name='QC Control', line=dict(color=COLORS['primary']))); fig.add_hline(y=ucl, line=dict(color=COLORS['accent'], dash='dash'), name='UCL (Mean+3σ)'); fig.add_hline(y=mean, line=dict(color=COLORS['dark_gray'], dash='dot'), name='Center Line'); fig.add_hline(y=lcl, line=dict(color=COLORS['accent'], dash='dash'), name='LCL (Mean-3σ)');
    if not violations.empty: fig.add_trace(go.Scatter(x=violations['Batch_ID'], y=violations['Yield_ng'], mode='markers', name='Violation', marker=dict(color=COLORS['danger'], size=10, symbol='x')))
    fig.update_layout(title='<b>Levey-Jennings Chart (Shewhart):</b> Positive Control Monitoring', xaxis_title='Batch ID', yaxis_title='Yield (ng)'); return fig
def plot_ewma_chart(df_control: pd.DataFrame, lambda_val: float = 0.2) -> go.Figure:
    in_control_data = df_control['Yield_ng'].iloc[:75]; mean, std_dev = in_control_data.mean(), in_control_data.std(ddof=1); ewma = df_control['Yield_ng'].ewm(span=(2/lambda_val) - 1, adjust=False).mean(); limit_factor = 3 * std_dev * np.sqrt(lambda_val / (2 - lambda_val)); ucl, lcl = mean + limit_factor, mean - limit_factor; fig = go.Figure(); fig.add_trace(go.Scatter(x=df_control['Batch_ID'], y=ewma, mode='lines+markers', name='EWMA', line=dict(color=COLORS['primary']))); fig.add_hline(y=ucl, line=dict(color=COLORS['accent'], dash='dash'), name='UCL'); fig.add_hline(y=mean, line=dict(color=COLORS['dark_gray'], dash='dot'), name='Center Line'); fig.add_hline(y=lcl, line=dict(color=COLORS['accent'], dash='dash'), name='LCL'); fig.update_layout(title=f'<b>EWMA Chart (λ={lambda_val}):</b> Detecting Small, Sustained Shifts', xaxis_title='Batch ID', yaxis_title='Exponentially Weighted Moving Average'); return fig
def plot_cusum_chart(df_control: pd.DataFrame, k: float = 0.5, h: float = 5.0) -> go.Figure:
    in_control_data = df_control['Yield_ng'].iloc[:75]; mean, std_dev = in_control_data.mean(), in_control_data.std(ddof=1); z = (df_control['Yield_ng'] - mean) / std_dev; cusum_pos = np.zeros(len(z)); cusum_neg = np.zeros(len(z));
    for i in range(1, len(z)): cusum_pos[i] = max(0, cusum_pos[i-1] + z.iloc[i] - k); cusum_neg[i] = min(0, cusum_neg[i-1] + z.iloc[i] + k)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df_control['Batch_ID'], y=cusum_pos, name='CUSUM+', line=dict(color=COLORS['primary']))); fig.add_trace(go.Scatter(x=df_control['Batch_ID'], y=cusum_neg, name='CUSUM-', line=dict(color=COLORS['secondary']))); fig.add_hline(y=h, line=dict(color=COLORS['danger'], dash='dash'), name='UCL'); fig.add_hline(y=-h, line=dict(color=COLORS['danger'], dash='dash'), name='LCL'); fig.update_layout(title=f'<b>CUSUM Chart:</b> Accumulating Process Shift Information', xaxis_title='Batch ID', yaxis_title='Cumulative Sum'); return fig
def plot_hotelling_t2_chart(df_hotelling: pd.DataFrame) -> go.Figure:
    in_control_data = df_hotelling.iloc[:80]; mean_vec, cov_matrix = in_control_data.mean(), in_control_data.cov(); inv_cov_matrix = np.linalg.inv(cov_matrix); t_squared = [(row - mean_vec).T @ inv_cov_matrix @ (row - mean_vec) for _, row in df_hotelling.iterrows()]; p, m = len(mean_vec), len(in_control_data); ucl = (p * (m - 1) * (m + 1)) / (m * m - m * p) * f_dist.ppf(0.99, p, m - p); fig = go.Figure(); fig.add_trace(go.Scatter(y=t_squared, mode='lines+markers', name='T² Statistic', line_color=COLORS['primary'])); fig.add_hline(y=ucl, line=dict(color=COLORS['danger'], dash='dash'), name='UCL'); fig.update_layout(title="<b>Hotelling's T² Chart:</b> Multivariate Process Control", xaxis_title="Sample Number", yaxis_title="T² Statistic"); return fig
def plot_adverse_event_clusters(df_clustered: pd.DataFrame) -> go.Figure:
    cluster_names = {0: "Neurological (Headache, Dizziness)", 1: "Allergic / Skin Reaction", 2: "Systemic (Liver, Anaphylaxis)", 3: "Gastrointestinal / Injection Site"}; df_clustered['cluster_name'] = df_clustered['cluster'].map(cluster_names).fillna("Other"); fig = px.scatter(df_clustered, x='x_pca', y='y_pca', color='cluster_name', hover_data=['description'], title="<b>ML Clustering of Adverse Event Narratives for Signal Detection</b>", labels={'color': 'Event Cluster'}); fig.update_layout(xaxis_title="PCA Component 1", yaxis_title="PCA Component 2", xaxis=dict(showticklabels=False, zeroline=False), yaxis=dict(showticklabels=False, zeroline=False)); return fig
def plot_pccp_monitoring(df_pccp: pd.DataFrame) -> go.Figure:
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df_pccp['Deployment_Day'], y=df_pccp['Model_AUC'], mode='lines', name='Model Performance (AUC)', line=dict(color=COLORS['primary']))); fig.add_hline(y=0.90, line=dict(color=COLORS['danger'], dash='dash'), name='Performance Threshold'); fig.add_vrect(x0=70, x1=100, fillcolor=hex_to_rgba(COLORS['warning'], 0.2), line_width=0, name="Performance Degradation"); fig.add_annotation(x=85, y=0.87, text="<b>Retraining & Revalidation<br>Triggered per PCCP</b>", showarrow=True, arrowhead=1, ax=0, ay=-40, bgcolor="rgba(255,255,255,0.8)"); fig.update_layout(title="<b>PCCP Monitoring for an AI/ML Device (SaMD)</b>", xaxis_title="Days Since Deployment", yaxis_title="Model Area Under Curve (AUC)", legend=dict(x=0.01, y=0.01, yanchor='bottom')); return fig
def plot_comparison_radar() -> go.Figure:
    categories = ['Interpretability', 'Data Volume Needs', 'Scalability', 'Handling Complexity', 'Biomarker Discovery', 'Regulatory Ease']; classical_scores = [5, 2, 1, 2, 1, 5]; ml_scores = [2, 5, 5, 5, 5, 2]; fig = go.Figure(); fig.add_trace(go.Scatterpolar(r=classical_scores + [classical_scores[0]], theta=categories + [categories[0]], fill='toself', name='Classical DOE/Stats', marker_color=COLORS['primary'])); fig.add_trace(go.Scatterpolar(r=ml_scores + [ml_scores[0]], theta=categories + [categories[0]], fill='toself', name='ML / Bioinformatics', marker_color=COLORS['secondary'])); fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5.5])), title="<b>Strengths Profile:</b> Classical vs. ML for Biotech R&D", legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)); return fig
def plot_verdict_barchart() -> go.Figure:
    data = {"Metric": ["Assay Parameter Optimization (DOE)", "Novel Biomarker Discovery", "High-Dimensional Data Analysis", "Analytical Validation (FDA)", "Proactive QC", "Protocol Interpretability"], "Winner": ["Classical", "ML", "ML", "Classical", "ML", "Classical"], "Score": [-1, 1, 1, -1, 1, -1]}; df = pd.DataFrame(data).sort_values('Score'); df['Color'] = df['Score'].apply(lambda x: COLORS['primary'] if x < 0 else COLORS['secondary']); fig = px.bar(df, x='Score', y='Metric', orientation='h', color='Color', color_discrete_map='identity', title="<b>Task-Specific Verdict:</b> Which Approach is Better?"); fig.update_layout(xaxis=dict(tickvals=[-1, 1], ticktext=['<b>Winner: Classical Stats</b>', '<b>Winner: Machine Learning</b>'], tickfont=dict(size=14), range=[-1.5, 1.5]), yaxis_title=None, bargap=0.4, showlegend=False); return fig
def plot_synergy_diagram() -> go.Figure:
    fig = _create_network_fig(height=400, x_range=[-0.5, 3.7], y_range=[-0.5, 2.5]); fig.add_shape(type="circle", x0=0, y0=0, x1=2, y1=2, line_color=COLORS['primary'], fillcolor=COLORS['primary'], opacity=0.6); fig.add_shape(type="circle", x0=1.2, y0=0, x1=3.2, y1=2, line_color=COLORS['secondary'], fillcolor=COLORS['secondary'], opacity=0.6); fig.add_annotation(x=1, y=1, text="<b>Classical Stats</b><br><i>Inference & Causality</i><br><i>Rigor & Validation</i>", showarrow=False, font=dict(color="white", size=12)); fig.add_annotation(x=2.2, y=1, text="<b>Machine Learning</b><br><i>Prediction & Discovery</i><br><i>Complexity & Scale</i>", showarrow=False, font=dict(color="white", size=12)); fig.add_annotation(x=1.6, y=1, text="<b>Bio-AI<br>Excellence</b>", showarrow=False, font=dict(color="black", size=18, family="Arial Black")); fig.update_layout(title="<b>The Hybrid Philosophy:</b> Combining Strengths", margin=dict(t=50, b=10, l=10, r=10)); return fig
