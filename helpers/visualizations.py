"""
helpers/visualizations.py

The definitive, gold-standard visualization library for the Bio-AI Excellence
Framework. This module contains a complete suite of re-architected plotting
functions designed to deliver commercial-grade, information-dense, and
actionable visualizations.

Author: Bio-AI Excellence SME Collective
Version: 33.2 (QFD Subplot Hotfix)
Date: 2025-07-16

Changelog from v33.1:
- [CRITICAL-FIX] Corrected a `ValueError` in the elite QFD plot by adjusting
  the `row` and `col` arguments in `fig.add_trace` to account for the
  `rowspan` logic in `make_subplots`. The plot now renders correctly.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import shap
import statsmodels.api as sm
from scipy.stats import norm, probplot, f as f_dist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from typing import Dict, List, Tuple, Any, Callable

from .styling import COLORS, hex_to_rgba

# --- 1. PROFESSIONAL PLOTLY TEMPLATE & CONFIGURATION ---
pio.templates["bio_ai_pro_theme"] = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=14, color=COLORS['text']),
        title_font=dict(size=22, color=COLORS['dark_gray'], family="Arial Black, sans-serif"),
        title_x=0.5,
        xaxis=dict(showgrid=True, gridcolor=COLORS['light_gray'], gridwidth=1, linecolor=COLORS['dark_gray'], linewidth=2, mirror=True, zeroline=False, ticks='outside', tickfont=dict(size=12)),
        yaxis=dict(showgrid=True, gridcolor=COLORS['light_gray'], gridwidth=1, linecolor=COLORS['dark_gray'], linewidth=2, mirror=True, zeroline=False, ticks='outside', tickfont=dict(size=12)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=12)),
        margin=dict(l=80, r=40, t=100, b=80),
        colorway=px.colors.qualitative.T10
    )
)
pio.templates.default = "bio_ai_pro_theme"

# --- 2. ADVANCED HELPER & UTILITY FUNCTIONS ---
def _create_network_diagram(title: str, nodes: Dict[str, Dict], edges: List[Tuple[str, str]], height: int = 600, x_range: List = None, y_range: List = None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=f"<b>{title}</b>", showlegend=False, plot_bgcolor='white', height=height, xaxis=dict(showgrid=False, zeroline=False, visible=False, range=x_range), yaxis=dict(showgrid=False, zeroline=False, visible=False, range=y_range), margin=dict(t=80, b=20, l=20, r=20), template=None)
    for start_key, end_key in edges:
        start_node, end_node = nodes.get(start_key, {}), nodes.get(end_key, {})
        if not start_node or not end_node: continue
        fig.add_trace(go.Scatter(x=[start_node['x'], end_node['x']], y=[start_node['y'], end_node['y']], mode='lines', line=dict(color=COLORS['dark_gray'], width=2), hoverinfo='none'))
    for node_id, node_data in nodes.items():
        fig.add_annotation(x=node_data['x'], y=node_data['y'], text=f"<b>{node_data['text'].replace('<br>', '</b><br>')}", showarrow=False, font=dict(color=node_data.get('font_color', 'white'), size=11, family="Arial"), bgcolor=node_data.get('color', COLORS['primary']), bordercolor=hex_to_rgba(node_data.get('color', COLORS['primary']), 0.5), borderwidth=2, borderpad=10, align="center")
    return fig

def _get_spc_violations(data: pd.Series, mean: float, std_dev: float) -> pd.DataFrame:
    violations = []; rule1_mask = (data > mean + 3 * std_dev) | (data < mean - 3 * std_dev)
    for i in rule1_mask[rule1_mask].index: violations.append({'index': i, 'value': data[i], 'rule': 'Rule 1: Point > 3σ from Mean'})
    above = (data > mean).astype(int); below = (data < mean).astype(int)
    for i in range(len(data) - 8):
        if above.iloc[i:i+9].sum() == 9 or below.iloc[i:i+9].sum() == 9: violations.append({'index': i+8, 'value': data[i+8], 'rule': 'Rule 2: 9 points on one side'})
    return pd.DataFrame(violations).drop_duplicates(subset=['index'])

# --- 3. CORE VISUALIZATION FUNCTIONS (ALL IMPLEMENTED) ---

def generate_html_table(df: pd.DataFrame, title: str) -> str:
    styled_df = df.style
    if 'RPN' in df.columns: styled_df = styled_df.background_gradient(cmap='YlOrRd', subset=['RPN'], low=0.1, high=1.0).format({'RPN': '{:.0f}'})
    styled_df = styled_df.set_table_styles([{'selector': 'th', 'props': [('background-color', COLORS['dark_gray']), ('color', 'white'), ('font-size', '14px'), ('font-weight', 'bold'), ('text-align', 'center'), ('border', f"1px solid {COLORS['dark_gray']}") ]}, {'selector': 'td', 'props': [('padding', '10px 12px'), ('border', f"1px solid {COLORS['light_gray']}"), ('vertical-align', 'middle'), ('text-align', 'left')]}, {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]}, {'selector': 'tr:hover', 'props': [('background-color', hex_to_rgba(COLORS['primary'], 0.1))]}, {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-size', '20px'), ('font-weight', 'bold'), ('color', COLORS['dark_gray']), ('padding', '12px 0'), ('text-align', 'center')]} ]).set_properties(**{'font-size': '13px'}).set_caption(f"<b>{title}</b>").hide(axis="index")
    return styled_df.to_html()

def plot_project_charter_visual() -> go.Figure:
    fig = go.Figure(); fig.update_layout(title="<b>Project Charter:</b> Assay for Early-Stage CRC Detection", plot_bgcolor=COLORS['background'], height=450, xaxis=dict(visible=False, range=[0, 10]), yaxis=dict(visible=False, range=[0, 10]), margin=dict(t=80, b=20, l=20, r=20), template=None)
    fig.add_shape(type="rect", x0=0.2, y0=4.5, x1=4.8, y1=9.5, fillcolor="white", line=dict(color=COLORS['light_gray'])); fig.add_shape(type="rect", x0=5.2, y0=4.5, x1=9.8, y1=9.5, fillcolor="white", line=dict(color=COLORS['light_gray'])); fig.add_shape(type="rect", x0=0.2, y0=0.5, x1=9.8, y1=4.0, fillcolor="white", line=dict(color=COLORS['light_gray']))
    fig.add_annotation(x=2.5, y=9.0, text="<b>Problem Statement</b>", font=dict(color=COLORS['primary'], size=16), showarrow=False); fig.add_annotation(x=2.5, y=6.7, align='center', text="Current CRC detection methods are either<br>invasive or lack sensitivity for early-stage disease,<br>leading to poor patient outcomes.", showarrow=False, font_size=12)
    fig.add_annotation(x=7.5, y=9.0, text="<b>Goal Statement</b>", font=dict(color=COLORS['primary'], size=16), showarrow=False); fig.add_annotation(x=7.5, y=6.7, align='center', text="Develop & validate a cfDNA-based NGS assay<br>with >90% sensitivity @ 99.5% specificity,<br>delivering results in < 5 days.", showarrow=False, font_size=12)
    fig.add_annotation(x=5.0, y=3.5, text="<b>Key Performance Indicators (KPIs) & Targets</b>", font=dict(color=COLORS['primary'], size=16), showarrow=False)
    kpis = {"Analytical Sensitivity": "LOD < 0.1% VAF", "Clinical Specificity": "> 99.5%", "Turnaround Time": "< 5 days"};
    for i, (k, v) in enumerate(kpis.items()): fig.add_annotation(x=1.5 + i * 2.5, y=2.5, text=f"<b>{k}</b>", font_size=14, showarrow=False); fig.add_annotation(x=1.5 + i * 2.5, y=1.5, text=v, font=dict(color=COLORS['success'], size=20), showarrow=False)
    return fig

def plot_ctq_tree_plotly() -> go.Figure:
    nodes = {'need': {'x': 0, 'y': 3, 'text': 'NEED<br>Accurate & Fast<br>Cancer Test', 'color': COLORS['accent']}, 'driver1': {'x': 2, 'y': 4.5, 'text': 'DRIVER<br>High Sensitivity', 'color': COLORS['primary']}, 'driver2': {'x': 2, 'y': 1.5, 'text': 'DRIVER<br>Fast TAT', 'color': COLORS['primary']}, 'ctq1': {'x': 4, 'y': 5.5, 'text': 'CTQ<br>LOD < 0.1% VAF', 'color': COLORS['secondary']}, 'ctq2': {'x': 4, 'y': 3.5, 'text': 'CTQ<br>Specificity > 99%', 'color': COLORS['secondary']}, 'ctq3': {'x': 4, 'y': 1.5, 'text': 'CTQ<br>Sample-to-Report<br>< 5 Days', 'color': COLORS['secondary']}}
    edges = [('need', 'driver1'), ('need', 'driver2'), ('driver1', 'ctq1'), ('driver1', 'ctq2'), ('driver2', 'ctq3')]
    return _create_network_diagram("Critical-to-Quality (CTQ) Tree", nodes, edges, height=500, x_range=[-1, 5])

# In helpers/visualizations.py, replace the existing QFD function with this one.

def plot_qfd_house_of_quality_pro(weights: pd.DataFrame, rel_df: pd.DataFrame) -> go.Figure:
    """
    Creates a single, cohesive, and aesthetically superior House of Quality plot.
    This elite version uses a single figure canvas with carefully arranged subplots
    to build the classic 'house' structure, including the correlation roof,
    making it highly intuitive and actionable.
    """
    tech_chars, cust_reqs = rel_df.columns.tolist(), rel_df.index.tolist()
    
    # --- 1. Data Preparation ---
    # Convert relationship scores to a numerical scale for the heatmap
    rel_map = {9: 1.0, 3: 0.6, 1: 0.2, 0: 0.0}
    rel_heatmap_z = rel_df.map(lambda x: rel_map.get(x, 0))
    
    # Calculate the final technical importance scores
    tech_importance = (rel_heatmap_z.T * weights['Importance'].values).T.sum()
    
    # Create a placeholder correlation matrix for the 'roof'
    # In a real scenario, this would come from data analysis.
    corr_matrix = pd.DataFrame(np.identity(len(tech_chars)), index=tech_chars, columns=tech_chars)
    corr_matrix.iloc[0, 1] = 0.7  # Strong Positive
    corr_matrix.iloc[1, 0] = 0.7
    corr_matrix.iloc[2, 3] = -0.6 # Strong Negative
    corr_matrix.iloc[3, 2] = -0.6
    
    # Mask for the triangular roof
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # --- 2. Figure Construction with Subplots ---
    fig = make_subplots(
        rows=3, cols=2,
        shared_xaxes='columns',
        shared_yaxes='rows',
        column_widths=[0.25, 0.75], 
        row_heights=[0.25, 0.55, 0.2],
        specs=[
            [{"rowspan": 2}, {"type": "heatmap"}],
            [None, {"type": "heatmap"}],
            [None, {"type": "bar"}]
        ],
        horizontal_spacing=0.01,
        vertical_spacing=0.01
    )

    # --- 3. Populate Subplots ---
    # a) Correlation Roof (Top Right)
    fig.add_trace(go.Heatmap(
        z=corr_matrix.where(mask),
        x=tech_chars, y=tech_chars,
        colorscale='RdBu', zmin=-1, zmax=1, showscale=False,
        hovertemplate='Correlation: %{z:.2f}<extra></extra>'
    ), row=1, col=2)

    # b) Customer Importance (Left Bar Chart)
    fig.add_trace(go.Bar(
        y=cust_reqs, x=weights['Importance'], orientation='h',
        marker=dict(color=COLORS['primary'], opacity=0.7),
        text=weights['Importance'], textposition='auto'
    ), row=1, col=1)

    # c) Central Relationship Matrix (Main Heatmap)
    fig.add_trace(go.Heatmap(
        z=rel_heatmap_z, y=cust_reqs, x=tech_chars,
        colorscale='Blues', showscale=False, customdata=rel_df.values,
        hovertemplate='Requirement: %{y}<br>Tech Characteristic: %{x}<br><b>Relationship Strength: %{customdata}</b><extra></extra>'
    ), row=2, col=2)

    # d) Final Technical Priorities (Bottom Bar Chart)
    fig.add_trace(go.Bar(
        x=tech_chars, y=tech_importance,
        marker=dict(color=COLORS['secondary'])
    ), row=3, col=2)
    
    # --- 4. Final Layout and Styling ---
    fig.update_layout(
        title_text="<b>The House of Quality (QFD):</b> From Customer Needs to Technical Priorities",
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(t=100, b=20, l=20, r=20),
        bargap=0.2
    )
    # Style axes to create the 'house' effect
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_xaxes(title_text="Technical Characteristics", tickangle=-45, row=3, col=2)
    fig.update_yaxes(title_text="Customer Needs", autorange="reversed", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(title_text="Priority Score", row=3, col=2)
    fig.update_xaxes(title_text="Importance", autorange="reversed", row=1, col=1)
    
    return fig

def plot_kano_visual(df_kano: pd.DataFrame) -> go.Figure:
    fig = go.Figure(); fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=10, fillcolor=hex_to_rgba(COLORS['success'], 0.1), line_width=0, layer='below'); fig.add_shape(type="rect", x0=0, y0=-10, x1=10, y1=0, fillcolor=hex_to_rgba(COLORS['danger'], 0.1), line_width=0, layer='below')
    colors = {'Basic (Must-be)': COLORS['accent'], 'Performance': COLORS['primary'], 'Excitement (Delighter)': COLORS['secondary']}
    for cat, color in colors.items(): subset = df_kano[df_kano['category'] == cat]; fig.add_trace(go.Scatter(x=subset['functionality'], y=subset['satisfaction'], mode='lines+markers', name=cat, line=dict(color=color, width=4), marker=dict(size=6, color=color)))
    annotations = [dict(x=8, y=8.5, text="<b>Excitement / Delighter</b><br>e.g., Detects novel resistance mutation", ax=-60, ay=-40, font_color=COLORS['secondary']), dict(x=7, y=3.5, text="<b>Performance</b><br>e.g., Faster sample-to-answer time", ax=0, ay=-50, font_color=COLORS['primary']), dict(x=9, y=-8, text="<b>Basic / Must-Be</b><br>e.g., Detects known KRAS hotspot", ax=0, ay=50, font_color=COLORS['accent'])]
    for ann in annotations: fig.add_annotation(x=ann['x'], y=ann['y'], text=ann['text'], showarrow=True, arrowhead=2, arrowwidth=2, ax=ann['ax'], ay=ann['ay'], font=dict(size=12, color=ann['font_color']), bgcolor='rgba(255,255,255,0.7)')
    fig.update_layout(title='<b>Kano Model:</b> Prioritizing Diagnostic Features', xaxis_title='<b>← Dysfunctional ... Functional →</b><br>Feature Implementation', yaxis_title='<b>← Dissatisfaction ... Satisfaction →</b><br>Clinician Response', legend=dict(orientation='v', y=0.99, x=0.01, yanchor='top', xanchor='left', bgcolor='rgba(255,255,255,0.8)')); return fig

def plot_gage_rr_sunburst(df_gage: pd.DataFrame) -> go.Figure:
    """
    Creates an elegant and highly informative Sunburst chart to visualize the
    hierarchical decomposition of variance in a Gage R&R study.
    """
    # --- 1. Data Preparation ---
    part_var = df_gage.loc[df_gage['Source of Variation'] == 'Assay Variation (Biology)', 'Contribution (%)'].iloc[0]
    repeatability = df_gage.loc[df_gage['Source of Variation'] == 'Repeatability (Sequencer)', 'Contribution (%)'].iloc[0]
    reproducibility = df_gage.loc[df_gage['Source of Variation'] == 'Reproducibility (Operator)', 'Contribution (%)'].iloc[0]
    gage_rr_total = repeatability + reproducibility
    
    # Determine the verdict based on AIAG guidelines
    verdict = "Acceptable" if gage_rr_total < 10 else "Marginal" if gage_rr_total < 30 else "Unacceptable"
    verdict_color = COLORS['success'] if verdict == "Acceptable" else COLORS['warning'] if verdict == "Marginal" else COLORS['danger']

    # --- 2. Create Hierarchical Data Structure for Sunburst ---
    data = dict(
        ids=[
            "Total", 
            "Process Variation", "Measurement System", 
            "Repeatability", "Reproducibility"
        ],
        labels=[
            "Total Variation", 
            "Process Variation<br>(Good)", "Measurement System<br>(Gage R&R)", 
            "Repeatability<br>(Equipment)", "Reproducibility<br>(Operator)"
        ],
        parents=[
            "", 
            "Total", "Total", 
            "Measurement System", "Measurement System"
        ],
        values=[
            100, 
            part_var, gage_rr_total, 
            repeatability, reproducibility
        ]
    )

    # --- 3. Build the Figure ---
    fig = go.Figure()

    fig.add_trace(go.Sunburst(
        ids=data["ids"],
        labels=data["labels"],
        parents=data["parents"],
        values=data["values"],
        branchvalues="total",
        insidetextorientation='radial',
        hoverinfo='label+percent parent',
        marker=dict(
            colors=[
                # Define colors for each level
                COLORS['light_gray'],  # Total
                COLORS['primary'],     # Process Variation
                COLORS['danger'],      # Measurement System
                COLORS['accent'],      # Repeatability
                COLORS['warning'],     # Reproducibility
            ],
            line=dict(color='white', width=2)
        )
    ))

    # --- 4. Final Layout and Styling ---
    fig.update_layout(
        title=f"<b>Gage R&R Variance Decomposition — Verdict: <span style='color:{verdict_color};'>{verdict}</span> ({gage_rr_total:.1f}%)</b>",
        margin=dict(t=100, l=10, r=10, b=10),
    )

    return fig
def plot_capability_analysis_pro(data: np.ndarray, lsl: float, usl: float) -> Tuple[go.Figure, float, float]:
    if data is None or len(data) < 2: return go.Figure().update_layout(title_text="<b>Error:</b> Insufficient data for analysis."), 0, 0
    mean, std = np.mean(data), np.std(data, ddof=1);
    if std == 0: return go.Figure().update_layout(title_text="<b>Error:</b> Cannot calculate capability with zero variation."), 0, 0
    cp = (usl - lsl) / (6 * std); cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)); fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], specs=[[{"type": "xy"}, {"type": "xy"}]])
    x_range = np.linspace(min(lsl, data.min()) - std, max(usl, data.max()) + std, 400); y_pdf = norm.pdf(x_range, mean, std); fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Data', marker_color=COLORS['primary']), row=1, col=1); fig.add_trace(go.Scatter(x=x_range, y=y_pdf, mode='lines', name='Normal Fit', line=dict(color=COLORS['secondary'], width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_range[x_range < lsl], y=y_pdf[x_range < lsl], fill='tozeroy', fillcolor=hex_to_rgba(COLORS['danger'], 0.3), mode='none', name='< LSL'), row=1, col=1); fig.add_trace(go.Scatter(x=x_range[x_range > usl], y=y_pdf[x_range > usl], fill='tozeroy', fillcolor=hex_to_rgba(COLORS['danger'], 0.3), mode='none', name='> USL'), row=1, col=1)
    for spec, name in [(lsl, 'LSL'), (usl, 'USL')]: fig.add_vline(x=spec, line_width=2, line_dash="dash", line_color=COLORS['danger'], annotation_text=name, row=1, col=1)
    fig.add_vline(x=mean, line_width=2, line_dash="dot", line_color=COLORS['dark_gray'], annotation_text=f"Mean={mean:.2f}", row=1, col=1); (osm, osr), (slope, intercept, r) = probplot(data, dist="norm"); fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Data', marker=dict(color=COLORS['primary'])), row=1, col=2); fig.add_trace(go.Scatter(x=osm, y=slope*osm + intercept, mode='lines', name='Fit', line=dict(color=COLORS['danger'])), row=1, col=2)
    verdict = "Capable" if cpk >= 1.33 else ("Marginal" if cpk >= 1.0 else "Not Capable"); verdict_color = COLORS['success'] if verdict == 'Capable' else (COLORS['warning'] if verdict == 'Marginal' else COLORS['danger']); fig.update_layout(title=f"<b>Process Capability Report — Verdict: <span style='color:{verdict_color};'>{verdict}</span></b>", xaxis_title="Measurement", yaxis_title="Density", showlegend=False, xaxis2_title="Theoretical Quantiles", yaxis2_title="Sample Quantiles")
    metrics_text = (f"<b>Process Metrics</b><br>Mean: {mean:.3f}<br>Std Dev: {std:.3f}<br>Cp: <b>{cp:.3f}</b><br>Cpk: <b style='color:{verdict_color};'>{cpk:.3f}</b><br>Target Cpk: 1.33"); fig.add_annotation(text=metrics_text, align='left', showarrow=False, xref='paper', yref='paper', x=1.0, y=1.0, bgcolor="rgba(255,255,255,0.8)", bordercolor=COLORS['dark_gray'], borderwidth=1); return fig, cp, cpk

def plot_shap_summary(shap_explanation: shap.Explanation) -> go.Figure:
    shap_values = shap_explanation.values; feature_names = shap_explanation.feature_names; mean_abs_shap = np.abs(shap_values).mean(axis=0); importance_df = pd.DataFrame({'feature': feature_names, 'importance': mean_abs_shap}).sort_values('importance', ascending=True); fig = go.Figure()
    for i, feature in enumerate(importance_df['feature']):
        feature_idx = feature_names.index(feature); fig.add_trace(go.Box(x=shap_values[:, feature_idx], name=feature, orientation='h', marker_color=hex_to_rgba(COLORS['primary'], 0.5), line_color=hex_to_rgba(COLORS['primary'], 0.8), showlegend=False))
    fig.add_trace(go.Scatter(x=importance_df['importance'], y=importance_df['feature'], mode='markers', marker=dict(color=COLORS['secondary'], size=12, line=dict(width=2, color='white')), name='Mean |SHAP|', showlegend=False))
    for i, row in importance_df.iterrows(): fig.add_shape(type='line', x0=0, y0=row['feature'], x1=row['importance'], y1=row['feature'], line=dict(color=COLORS['secondary'], width=2))
    fig.update_layout(title="<b>Global Feature Importance & Impact Distribution</b>", xaxis_title="SHAP Value (Impact on Model Output)", yaxis_title=None, plot_bgcolor='white', xaxis=dict(gridcolor=COLORS['light_gray'], zerolinecolor=COLORS['dark_gray']), yaxis=dict(showgrid=False), margin=dict(l=150, t=100, b=50)); return fig

def plot_risk_signal_clusters(df_clustered: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df_clustered, x='Temp_C', y='Pressure_psi', color='cluster', title="<b>Risk Signal Clustering via DBSCAN</b>", labels={'cluster': 'Process Cluster', 'Temp_C': 'Temperature (°C)', 'Pressure_psi': 'Pressure (psi)'}, color_discrete_map={'0': COLORS['primary'], '1': COLORS['secondary'], 'Outlier/Anomaly': COLORS['danger']}, hover_data=['Source']); fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey'))); fig.update_layout(legend_title_text='Signal Cluster'); return fig

def plot_fishbone_plotly() -> go.Figure:
    nodes = {'effect': {'x': 8, 'y': 5, 'text': 'EFFECT<br>Low Library Yield', 'color': COLORS['danger']}, 'reagents': {'x': 3, 'y': 8, 'text': 'Reagents', 'color': COLORS['primary']}, 'c1': {'x': 1, 'y': 9, 'text': 'Enzyme Inactivity', 'color': COLORS['light_gray'], 'font_color': COLORS['text']}, 'equipment': {'x': 5, 'y': 8, 'text': 'Equipment', 'color': COLORS['primary']}, 'c2': {'x': 3, 'y': 9, 'text': 'Pipette out of cal', 'color': COLORS['light_gray'], 'font_color': COLORS['text']}, 'method': {'x': 7, 'y': 8, 'text': 'Method', 'color': COLORS['primary']}, 'c3': {'x': 5, 'y': 9, 'text': 'Wrong incubation time', 'color': COLORS['light_gray'], 'font_color': COLORS['text']}, 'technician': {'x': 3, 'y': 2, 'text': 'Technician', 'color': COLORS['primary']}, 'c4': {'x': 1, 'y': 1, 'text': 'Inconsistent pipetting', 'color': COLORS['light_gray'], 'font_color': COLORS['text']}, 'sample': {'x': 5, 'y': 2, 'text': 'Sample', 'color': COLORS['primary']}, 'c5': {'x': 3, 'y': 1, 'text': 'Low DNA input', 'color': COLORS['light_gray'], 'font_color': COLORS['text']}, 'environment': {'x': 7, 'y': 2, 'text': 'Environment', 'color': COLORS['primary']}, 'c6': {'x': 5, 'y': 1, 'text': 'High humidity', 'color': COLORS['light_gray'], 'font_color': COLORS['text']}}
    edges = [('effect', 'reagents'), ('reagents', 'c1'), ('effect', 'equipment'), ('equipment', 'c2'), ('effect', 'method'), ('method', 'c3'), ('effect', 'technician'), ('technician', 'c4'), ('effect', 'sample'), ('sample', 'c5'), ('effect', 'environment'), ('environment', 'c6')]
    return _create_network_diagram("Fishbone (Ishikawa) Diagram", nodes, edges, height=600, x_range=[0, 9], y_range=[0, 10])

def plot_pareto_chart(df_pareto: pd.DataFrame) -> go.Figure:
    df_sorted = df_pareto.sort_values('Frequency', ascending=False); df_sorted['Cumulative Percentage'] = df_sorted['Frequency'].cumsum() / df_sorted['Frequency'].sum() * 100; fig = make_subplots(specs=[[{"secondary_y": True}]]); fig.add_trace(go.Bar(x=df_sorted['QC_Failure_Mode'], y=df_sorted['Frequency'], name='Failure Count', marker_color=COLORS['primary']), secondary_y=False); fig.add_trace(go.Scatter(x=df_sorted['QC_Failure_Mode'], y=df_sorted['Cumulative Percentage'], name='Cumulative %', mode='lines+markers', line_color=COLORS['accent'], line_width=3), secondary_y=True); fig.add_hline(y=80, line=dict(color=COLORS['dark_gray'], dash='dot', width=2), secondary_y=True, annotation_text="80% Line", annotation_position="bottom right"); fig.update_layout(title_text="<b>Pareto Chart:</b> Identifying the 'Vital Few' Failure Modes", xaxis_title="QC Failure Mode", bargap=0.4); fig.update_yaxes(title_text="<b>Frequency</b>", secondary_y=False); fig.update_yaxes(title_text="<b>Cumulative Percentage</b>", secondary_y=True, range=[0, 101], ticksuffix='%'); return fig

def plot_regression_comparison(model_results: dict) -> go.Figure:
    X, y = model_results['X'], model_results['y']; primary_factor = 'Annealing_Temp'; sort_idx = X[primary_factor].argsort(); fig = go.Figure(); fig.add_trace(go.Scatter(x=X[primary_factor].iloc[sort_idx], y=y.iloc[sort_idx], mode='markers', name='Actual Data', marker=dict(color=COLORS['dark_gray'], opacity=0.5))); fig.add_trace(go.Scatter(x=X[primary_factor].iloc[sort_idx], y=model_results['linear_predictions'][sort_idx], mode='lines', name=f"Linear Model (R²={model_results['linear_r2']:.2f})", line=dict(color=COLORS['primary'], width=3))); fig.add_trace(go.Scatter(x=X[primary_factor].iloc[sort_idx], y=model_results['rf_predictions'][sort_idx], mode='lines', name=f"Random Forest (OOB R²={model_results['rf_oob_r2']:.2f})", line=dict(color=COLORS['secondary'], width=3, dash='dot'))); fig.update_layout(title_text="<b>Regression:</b> Modeling Assay Performance", xaxis_title=f"Primary Factor: {primary_factor.replace('_', ' ')} (°C)", yaxis_title="On-Target Rate (%)"); return fig

def plot_fault_tree_plotly() -> go.Figure:
    nodes = {'top': {'x': 2, 'y': 4.5, 'text': 'TOP EVENT<br>False Negative Result', 'color': COLORS['danger']}, 'or1': {'x': 2, 'y': 3.5, 'text': 'OR Gate', 'color': COLORS['dark_gray']}, 'and1': {'x': 0.5, 'y': 2, 'text': 'AND Gate', 'color': COLORS['dark_gray']}, 'assay': {'x': 3.5, 'y': 2.5, 'text': 'Assay Failure', 'color': COLORS['primary']}, 'reagent': {'x': 0, 'y': 1, 'text': 'Reagent Degraded<br>P=0.01', 'color': COLORS['secondary']}, 'storage': {'x': 1, 'y': 1, 'text': 'Improper Storage<br>P=0.05', 'color': COLORS['secondary']}, 'sample': {'x': 3.5, 'y': 1.5, 'text': 'Low DNA Input<br>P=0.02', 'color': COLORS['secondary']}}
    edges = [('top', 'or1'), ('or1', 'and1'), ('or1', 'assay'), ('assay', 'sample'), ('and1', 'reagent'), ('and1', 'storage')]
    return _create_network_diagram("Fault Tree Analysis (FTA)", nodes, edges, height=500, x_range=[-0.5, 4.5], y_range=[-0.5, 5])

def plot_5whys_diagram() -> go.Figure:
    steps = [('Problem', 'Low Library Yield on Plate 4', COLORS['danger']), ('Why 1?', 'Reagents added improperly.', COLORS['primary']), ('Why 2?', 'Technician used a miscalibrated pipette.', COLORS['primary']), ('Why 3?', 'Pipette was overdue for calibration.', COLORS['primary']), ('Why 4?', 'Manual tracking system for calibration failed.', COLORS['primary']), ('Root Cause', 'Asset management system is not robust.', COLORS['success'])]
    nodes, edges = {}, [];
    for i, (level, text, color) in enumerate(steps):
        nodes[f's{i}'] = {'x': 1, 'y': 5 - i, 'text': f'{level}<br>{text}', 'color': color};
        if i > 0: edges.append((f's{i-1}', f's{i}'))
    return _create_network_diagram("5 Whys Analysis", nodes, edges, height=600, y_range=[-0.5, 5.5])

def plot_nlp_on_capa_logs(df_topics: pd.DataFrame) -> go.Figure:
    df_topics['percent'] = (df_topics['Frequency'] / df_topics['Frequency'].sum()) * 100
    fig = px.treemap(df_topics, path=[px.Constant("All CAPAs"), 'Topic'], values='Frequency', color='Topic', color_discrete_map={'Reagent/Storage Issue': COLORS['primary'], 'Contamination': COLORS['accent'], 'Hardware Failure': COLORS['danger'], 'Human Error': COLORS['warning']}, title="<b>Systemic Failure Themes Revealed by NLP on CAPA Logs</b>", hover_data={'percent': ':.1f%'})
    fig.update_traces(textinfo='label+value+percent root', root_color='lightgrey'); fig.update_layout(margin=dict(t=80, l=10, r=10, b=10)); return fig

def plot_doe_effects(df_doe: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    model = sm.OLS.from_formula('Library_Yield ~ Primer_Conc * Anneal_Temp * PCR_Cycles', data=df_doe).fit(); effects = model.params.drop('Intercept').sort_values().to_frame('Effect'); main_effect_keys = ['Primer_Conc', 'Anneal_Temp', 'PCR_Cycles']; main_effects = effects.loc[effects.index.isin(main_effect_keys)]; interaction_effects = effects.drop(main_effects.index);
    fig_main = px.bar(main_effects, x='Effect', y=main_effects.index, orientation='h', title="<b>Main Effects</b>", color_discrete_sequence=[COLORS['primary']]); fig_int = px.bar(interaction_effects, x='Effect', y=interaction_effects.index, orientation='h', title="<b>Interaction Effects</b>", color_discrete_sequence=[COLORS['secondary']]);
    for fig in [fig_main, fig_int]: fig.update_layout(showlegend=False, yaxis_title=None, yaxis={'categoryorder':'total ascending'})
    return fig_main, fig_int

def plot_doe_cube(df_doe: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=[go.Scatter3d(x=df_doe['Primer_Conc'], y=df_doe['Anneal_Temp'], z=df_doe['PCR_Cycles'], mode='markers+text', text=[f'{y:.1f}' for y in df_doe['Library_Yield']], textposition='bottom center', marker=dict(size=12, color=df_doe['Library_Yield'], colorscale='Viridis', colorbar=dict(title='Yield'), showscale=True))]);
    for i in range(len(df_doe)):
        for j in range(i + 1, len(df_doe)):
            if np.sum(df_doe.iloc[i, :3] != df_doe.iloc[j, :3]) == 1: fig.add_trace(go.Scatter3d(x=[df_doe.iloc[i, 0], df_doe.iloc[j, 0]], y=[df_doe.iloc[i, 1], df_doe.iloc[j, 1]], z=[df_doe.iloc[i, 2], df_doe.iloc[j, 2]], mode='lines', line=dict(color=COLORS['light_gray'], width=3), showlegend=False))
    fig.update_layout(title="<b>DOE Cube Plot</b>", scene=dict(xaxis_title='Primer Conc', yaxis_title='Anneal Temp', zaxis_title='PCR Cycles', xaxis=dict(tickvals=[-1, 1]), yaxis=dict(tickvals=[-1, 1]), zaxis=dict(tickvals=[-1, 1])), margin=dict(l=0, r=0, b=0, t=40)); return fig

def plot_doe_3d_surface(df_doe: pd.DataFrame) -> go.Figure:
    formula = 'Library_Yield ~ Primer_Conc * Anneal_Temp'; model = sm.OLS.from_formula(formula, data=df_doe).fit()
    x_range = np.linspace(df_doe['Primer_Conc'].min(), df_doe['Primer_Conc'].max(), 20); y_range = np.linspace(df_doe['Anneal_Temp'].min(), df_doe['Anneal_Temp'].max(), 20); grid_x, grid_y = np.meshgrid(x_range, y_range); grid_df = pd.DataFrame({'Primer_Conc': grid_x.ravel(), 'Anneal_Temp': grid_y.ravel()})
    grid_df['predicted_yield'] = model.predict(grid_df); z_surface = grid_df['predicted_yield'].values.reshape(grid_x.shape)
    fig = go.Figure(data=[go.Surface(z=z_surface, x=x_range, y=y_range, colorscale='Viridis', colorbar=dict(title='Predicted Yield'), opacity=0.9)])
    fig.add_trace(go.Scatter3d(x=df_doe['Primer_Conc'], y=df_doe['Anneal_Temp'], z=df_doe['Library_Yield'], mode='markers', marker=dict(size=8, color=COLORS['danger'], symbol='circle'), name='Actual Experiments'))
    fig.update_layout(title="<b>3D Response Surface of Predicted Yield</b>", scene=dict(xaxis_title='Primer Conc', yaxis_title='Anneal Temp', zaxis_title='Library Yield'), margin=dict(l=0, r=0, b=0, t=80)); return fig

def plot_rsm_contour(df_rsm: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Contour(z=df_rsm['Yield'], x=df_rsm['Temperature'], y=df_rsm['Concentration'], colorscale='Viridis', contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=12, color='white')))); max_yield_point = df_rsm.loc[df_rsm['Yield'].idxmax()]; fig.add_trace(go.Scatter(x=[max_yield_point['Temperature']], y=[max_yield_point['Concentration']], mode='markers', marker=dict(color=COLORS['danger'], size=15, symbol='star'), name='Optimal Point')); fig.update_layout(title="<b>Response Surface Methodology (RSM)</b>", xaxis_title="Temperature (°C)", yaxis_title="Enzyme Concentration", showlegend=False); return fig

def plot_shewhart_chart(df_control: pd.DataFrame) -> go.Figure:
    data = df_control['Yield_ng']; mean, std_dev = data.iloc[:75].mean(), data.iloc[:75].std(ddof=1); fig = go.Figure();
    for i, alpha in [(3, 0.2), (2, 0.1), (1, 0.05)]: fig.add_shape(type="rect", xref="paper", yref="y", x0=0, y0=mean - i*std_dev, x1=1, y1=mean + i*std_dev, fillcolor=hex_to_rgba(COLORS['warning'], alpha), line_width=0, layer='below')
    fig.add_trace(go.Scatter(x=df_control['Batch_ID'], y=data, mode='lines+markers', name='QC Data', line=dict(color=COLORS['primary']))); fig.add_hline(y=mean, line=dict(color=COLORS['dark_gray'], dash='dot'));
    for i in [-3, 3]: fig.add_hline(y=mean + i*std_dev, line=dict(color=COLORS['danger'], dash='dash'))
    violations = _get_spc_violations(data, mean, std_dev);
    if not violations.empty: fig.add_trace(go.Scatter(x=violations['index'], y=violations['value'], mode='markers', name='Violation', marker=dict(color=COLORS['danger'], size=12, symbol='x', line_width=3), hovertext=violations['rule'], hoverinfo='text'))
    fig.update_layout(title='<b>Diagnostic Levey-Jennings Chart (Shewhart)</b>', xaxis_title='Batch ID', yaxis_title='Yield (ng)'); return fig

def plot_ewma_chart(df_control: pd.DataFrame, lambda_val: float = 0.2) -> go.Figure:
    data = df_control['Yield_ng']; mean, std_dev = data.iloc[:75].mean(), data.iloc[:75].std(ddof=1); ewma = data.ewm(span=(2/lambda_val) - 1, adjust=False).mean(); limit_factor = 3 * std_dev * np.sqrt(lambda_val / (2 - lambda_val)); ucl, lcl = mean + limit_factor, mean - limit_factor; fig = go.Figure(); fig.add_trace(go.Scatter(x=df_control['Batch_ID'], y=ewma, mode='lines+markers', name='EWMA', line=dict(color=COLORS['primary']))); fig.add_hline(y=ucl, line=dict(color=COLORS['accent'], dash='dash'), name='UCL'); fig.add_hline(y=mean, line=dict(color=COLORS['dark_gray'], dash='dot'), name='Center Line'); fig.add_hline(y=lcl, line=dict(color=COLORS['accent'], dash='dash'), name='LCL')
    violations = ewma[(ewma > ucl) | (ewma < lcl)];
    if not violations.empty: fig.add_trace(go.Scatter(x=violations.index, y=violations, mode='markers', name='Violation', marker=dict(color=COLORS['danger'], size=10, symbol='x')))
    fig.update_layout(title=f'<b>EWMA Chart (λ={lambda_val}):</b> Detecting Small Shifts', xaxis_title='Batch ID', yaxis_title='Exponentially Weighted Moving Average'); return fig

def plot_cusum_chart(df_control: pd.DataFrame, k: float = 0.5, h: float = 5.0) -> go.Figure:
    data = df_control['Yield_ng']; mean, std_dev = data.iloc[:75].mean(), data.iloc[:75].std(ddof=1); z = (data - mean) / std_dev; cusum_pos = np.zeros(len(z)); cusum_neg = np.zeros(len(z));
    for i in range(1, len(z)): cusum_pos[i] = max(0, cusum_pos[i-1] + z.iloc[i] - k); cusum_neg[i] = min(0, cusum_neg[i-1] + z.iloc[i] + k)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df_control['Batch_ID'], y=cusum_pos, name='CUSUM+', line=dict(color=COLORS['primary']))); fig.add_trace(go.Scatter(x=df_control['Batch_ID'], y=cusum_neg, name='CUSUM-', line=dict(color=COLORS['secondary']))); fig.add_hline(y=h, line=dict(color=COLORS['danger'], dash='dash'), name='UCL'); fig.add_hline(y=-h, line=dict(color=COLORS['danger'], dash='dash'), name='LCL'); fig.update_layout(title=f'<b>CUSUM Chart:</b> Accumulating Process Shift Info', xaxis_title='Batch ID', yaxis_title='Cumulative Sum'); return fig

def plot_hotelling_t2_chart(df_hotelling: pd.DataFrame) -> go.Figure:
    in_control_data = df_hotelling.iloc[:80]; mean_vec, cov_matrix = in_control_data.mean(), in_control_data.cov(); inv_cov_matrix = np.linalg.inv(cov_matrix); t_squared = [(row - mean_vec).T @ inv_cov_matrix @ (row - mean_vec) for _, row in df_hotelling.iterrows()]; p, m = len(mean_vec), len(in_control_data); ucl = (p * (m - 1) * (m + 1)) / (m * m - m * p) * f_dist.ppf(0.99, p, m - p); df_hotelling['T_Squared'] = t_squared; out_of_control_points = df_hotelling[df_hotelling['T_Squared'] > ucl]; fig = make_subplots(rows=2, cols=1, shared_xaxes=False, subplot_titles=("<b>When did it fail?</b> (T² Control Chart)", "<b>Why did it fail?</b> (Process Variable Scatter)"), vertical_spacing=0.15)
    fig.add_trace(go.Scatter(x=df_hotelling.index, y=df_hotelling['T_Squared'], mode='lines+markers', name='T² Statistic', line_color=COLORS['primary']), row=1, col=1); fig.add_hline(y=ucl, line=dict(color=COLORS['danger'], dash='dash'), name='UCL', row=1, col=1)
    if not out_of_control_points.empty: fig.add_trace(go.Scatter(x=out_of_control_points.index, y=out_of_control_points['T_Squared'], mode='markers', name='Violation', marker=dict(color=COLORS['danger'], size=10, symbol='x')), row=1, col=1)
    cols = df_hotelling.columns.drop('T_Squared'); fig.add_trace(go.Scatter(x=df_hotelling[cols[0]], y=df_hotelling[cols[1]], mode='markers', marker=dict(color=COLORS['dark_gray'], opacity=0.5), name='Process Data'), row=2, col=1)
    if not out_of_control_points.empty: fig.add_trace(go.Scatter(x=out_of_control_points[cols[0]], y=out_of_control_points[cols[1]], mode='markers', name='Violation', marker=dict(color=COLORS['danger'], size=10)), row=2, col=1)
    fig.update_layout(title="<b>Hotelling's T² Chart:</b> Linked Multivariate Process Control", showlegend=False); fig.update_xaxes(title_text="Sample Number", row=1, col=1); fig.update_yaxes(title_text="T² Statistic", row=1, col=1); fig.update_xaxes(title_text=cols[0], row=2, col=1); fig.update_yaxes(title_text=cols[1], row=2, col=1); return fig

def plot_adverse_event_clusters(df_clustered: pd.DataFrame) -> go.Figure:
    cluster_names = {0: "Neurological (Headache, Dizziness)", 1: "Allergic / Skin Reaction", 2: "Systemic (Liver, Anaphylaxis)", 3: "Gastrointestinal / Injection Site"}; df_clustered['cluster_name'] = df_clustered['cluster'].map(cluster_names).fillna("Other"); fig = px.scatter(df_clustered, x='x_pca', y='y_pca', color='cluster_name', hover_data=['description'], title="<b>ML Clustering of Adverse Event Narratives</b>", labels={'color': 'Event Cluster'}); fig.update_layout(xaxis_title="PCA Component 1", yaxis_title="PCA Component 2", xaxis=dict(showticklabels=False, zeroline=False), yaxis=dict(showticklabels=False, zeroline=False)); return fig

def plot_pccp_monitoring(df_pccp: pd.DataFrame) -> go.Figure:
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df_pccp['Deployment_Day'], y=df_pccp['Model_AUC'], mode='lines', name='Model Performance (AUC)', line=dict(color=COLORS['primary']))); fig.add_hline(y=0.90, line=dict(color=COLORS['danger'], dash='dash'), name='Performance Threshold'); fig.add_vrect(x0=70, x1=100, fillcolor=hex_to_rgba(COLORS['warning'], 0.2), line_width=0, name="Performance Degradation"); fig.add_annotation(x=85, y=0.87, text="<b>Retraining & Revalidation<br>Triggered per PCCP</b>", showarrow=True, arrowhead=1, ax=0, ay=-40, bgcolor="rgba(255,255,255,0.8)"); fig.update_layout(title="<b>PCCP Monitoring for an AI/ML Device (SaMD)</b>", xaxis_title="Days Since Deployment", yaxis_title="Model Area Under Curve (AUC)", legend=dict(x=0.01, y=0.01, yanchor='bottom')); return fig

def plot_comparison_radar() -> go.Figure:
    categories = ['Interpretability', 'Data Volume Needs', 'Scalability', 'Handling Complexity', 'Biomarker Discovery', 'Regulatory Ease']; classical_scores = [5, 2, 1, 2, 1, 5]; ml_scores = [2, 5, 5, 5, 5, 2]; fig = go.Figure(); fig.add_trace(go.Scatterpolar(r=classical_scores + [classical_scores[0]], theta=categories + [categories[0]], fill='toself', name='Classical DOE/Stats', marker_color=COLORS['primary'])); fig.add_trace(go.Scatterpolar(r=ml_scores + [ml_scores[0]], theta=categories + [categories[0]], fill='toself', name='ML / Bioinformatics', marker_color=COLORS['secondary'])); fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5.5])), title="<b>Strengths Profile:</b> Classical vs. ML for Biotech R&D", legend=dict(yanchor="bottom", y=-0.2, xanchor="center", x=0.5)); return fig

def plot_verdict_barchart() -> go.Figure:
    data = {"Metric": ["Assay Parameter Optimization (DOE)", "Novel Biomarker Discovery", "High-Dimensional Data Analysis", "Analytical Validation (FDA)", "Proactive QC", "Protocol Interpretability"], "Winner": ["Classical", "ML", "ML", "Classical", "ML", "Classical"], "Score": [-1, 1, 1, -1, 1, -1]}; df = pd.DataFrame(data).sort_values('Score'); df['Color'] = df['Score'].apply(lambda x: COLORS['primary'] if x < 0 else COLORS['secondary']); fig = px.bar(df, x='Score', y='Metric', orientation='h', color='Color', color_discrete_map='identity', title="<b>Task-Specific Verdict:</b> Which Approach is Better?"); fig.update_layout(xaxis=dict(tickvals=[-1, 1], ticktext=['<b>Winner: Classical Stats</b>', '<b>Winner: Machine Learning</b>'], tickfont=dict(size=14), range=[-1.5, 1.5]), yaxis_title=None, bargap=0.4, showlegend=False); return fig

def plot_synergy_diagram() -> go.Figure:
    fig = go.Figure(); fig.update_layout(height=400, showlegend=False, xaxis=dict(visible=False, range=[0, 3.2]), yaxis=dict(visible=False, range=[0,2]), margin=dict(t=80, b=10, l=10, r=10)); fig.add_shape(type="circle", x0=0, y0=0, x1=2, y1=2, line_color=COLORS['primary'], fillcolor=hex_to_rgba(COLORS['primary'], 0.6)); fig.add_shape(type="circle", x0=1.2, y0=0, x1=3.2, y1=2, line_color=COLORS['secondary'], fillcolor=hex_to_rgba(COLORS['secondary'], 0.6)); fig.add_annotation(x=1, y=1, text="<b>Classical Stats</b><br><i>Inference & Causality</i><br><i>Rigor & Validation</i>", showarrow=False, font=dict(color="white", size=12)); fig.add_annotation(x=2.2, y=1, text="<b>Machine Learning</b><br><i>Prediction & Discovery</i><br><i>Complexity & Scale</i>", showarrow=False, font=dict(color="white", size=12)); fig.add_annotation(x=1.6, y=1, text="<b>Bio-AI<br>Excellence</b>", showarrow=False, font=dict(color="black", size=18, family="Arial Black")); fig.update_layout(title="<b>The Hybrid Philosophy:</b> Combining Strengths"); return fig

def plot_bayesian_optimization_interactive(true_func, x_range, sampled_points) -> Tuple[go.Figure, float]:
    X_train, y_train = np.array(sampled_points['x']).reshape(-1, 1), np.array(sampled_points['y']); kernel = ConstantKernel(1.0) * RBF(length_scale=1.0); gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42); gp.fit(X_train, y_train); y_pred, sigma = gp.predict(x_range.reshape(-1, 1), return_std=True); ucb = y_pred + 1.96 * sigma; next_point = x_range[np.argmax(ucb)]
    fig = go.Figure(); fig.add_trace(go.Scatter(x=np.concatenate([x_range, x_range[::-1]]), y=np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]), fill='toself', fillcolor=hex_to_rgba(COLORS['primary'], 0.2), line=dict(color='rgba(255,255,255,0)'), name='95% Confidence Interval'))
    fig.add_trace(go.Scatter(x=x_range, y=true_func(x_range), mode='lines', name='True Function (Unknown to Model)', line=dict(color=COLORS['dark_gray'], dash='dash'))); fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='GP Mean Prediction', line=dict(color=COLORS['primary'], width=3))); fig.add_trace(go.Scatter(x=x_range, y=ucb, mode='lines', name='Acquisition Function (UCB)', line=dict(color=COLORS['secondary'], dash='dot', width=2))); fig.add_trace(go.Scatter(x=sampled_points['x'], y=sampled_points['y'], mode='markers', name='Sampled Points', marker=dict(color=COLORS['danger'], size=12, symbol='x', line_width=2))); fig.add_vline(x=next_point, line=dict(color=COLORS['accent'], dash='longdash', width=2), annotation_text="Next Experiment", annotation_position="top left")
    fig.update_layout(title=f"<b>Bayesian Optimization:</b> Iteration {len(sampled_points['x'])}", xaxis_title="Process Parameter (e.g., Annealing Temperature)", yaxis_title="Assay Yield / Performance", legend=dict(y=1.15)); return fig, next_point
