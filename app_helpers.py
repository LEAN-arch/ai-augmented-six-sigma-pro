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
from scipy.stats import norm, gaussian_kde, f_oneway, t, f
from typing import List, Dict, Any, Tuple

# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

COLORS = {
    "primary": "#0072B2",      # Muted Blue
    "secondary": "#009E73",    # Muted Green
    "accent": "#D55E00",       # Muted Orange
    "neutral_yellow": "#F0E442",# Muted Yellow
    "neutral_pink": "#CC79A7",  # Muted Pink
    "background": "#F8F9FA",   # Very Light Gray
    "text": "#212529",         # Dark Gray
    "light_gray": "#DEE2E6",   # Light Gray for grids/borders
    "dark_gray": "#495057",    # Medium-Dark Gray
    "success": "#28A745",      # Green for success
    "warning": "#FFC107",      # Yellow for warning
    "danger": "#DC3545",       # Red for danger
}

def get_custom_css() -> str:
    """Returns the custom CSS string for the Streamlit app."""
    css = """
    <style>
        /* General App Style */
        .stApp {{
            background-color: {background};
            color: {text};
        }}
        /* Main Headers */
        h1, h2 {{
            color: {dark_gray};
            border-bottom: 2px solid {light_gray};
            padding-bottom: 10px;
        }}
        h3 {{ color: {primary}; }}
        h4, h5 {{ color: {dark_gray}; }}

        /* Sidebar Style */
        .st-emotion-cache-16txtl3 {{
            background-color: #FFFFFF;
        }}

        /* Container Borders */
        .st-emotion-cache-1r4qj8v, .st-emotion-cache-1kyxreq {{
            border: 1px solid {light_gray};
            border-radius: 0.5rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }}

        /* Button Style for Custom Navigation */
        .stButton>button {{
            border-radius: 0.5rem;
        }}
    </style>
    """.format(**COLORS)
    return css

# ==============================================================================
# SECTION 2: DATA GENERATORS
# ==============================================================================

def generate_process_data(mean: float, std_dev: float, size: int, lsl: float, usl: float) -> np.ndarray:
    """Generates process data with potential outliers."""
    data = np.random.normal(mean, std_dev, size)
    return data

def generate_nonlinear_data(size: int = 200) -> pd.DataFrame:
    """Generates non-linear data for regression examples."""
    np.random.seed(42)
    X1 = np.linspace(-10, 10, size)
    X2 = 0.5 * X1**2 + np.random.normal(0, 5, size)
    X3_noise = np.random.randn(size) * 10
    y = 2 * X1 + 0.8 * X2 + np.random.normal(0, 8, size)
    return pd.DataFrame({'Temp': X1, 'Pressure': X2, 'Vibration_Noise': X3_noise, 'Yield': y})

def generate_control_chart_data(mean: float = 100, std_dev: float = 5, size: int = 150, shift_point: int = 75, shift_magnitude: float = 1.0) -> pd.DataFrame:
    """Generates data for control charts with a mean shift."""
    np.random.seed(42)
    in_control = np.random.normal(mean, std_dev, shift_point)
    out_of_control = np.random.normal(mean + shift_magnitude * std_dev, std_dev, size - shift_point)
    return pd.DataFrame({'Time': np.arange(size), 'Value': np.concatenate([in_control, out_of_control])})

def generate_doe_data() -> pd.DataFrame:
    """Generates data for a 2^3 factorial Design of Experiments."""
    np.random.seed(42)
    factors = [-1, 1]
    data = []
    for f1 in factors:
        for f2 in factors:
            for f3 in factors:
                response = 50 + 5*f1 - 8*f2 + 2*f3 + 4*f1*f2 + np.random.randn() * 1.5
                data.append([f1, f2, f3, response])
    return pd.DataFrame(data, columns=['Temp', 'Pressure', 'Time', 'Yield'])

def generate_kano_data() -> pd.DataFrame:
    """Generates data for plotting Kano model curves."""
    np.random.seed(42)
    func = np.linspace(0, 10, 20)
    basic_sat = np.clip(np.linspace(-8, 0, 20) + np.random.normal(0, 0.5, 20), -10, 0)
    basic_sat[func==0] = -10
    perf_sat = np.linspace(-5, 5, 20) + np.random.normal(0, 0.8, 20)
    excite_sat = np.clip(np.linspace(0, 8, 20) + np.random.normal(0, 0.5, 20), 0, 10)
    excite_sat[func==0] = 0
    
    basic = pd.DataFrame({'functionality': func, 'satisfaction': basic_sat, 'category': 'Basic (Must-be)'})
    performance = pd.DataFrame({'functionality': func, 'satisfaction': perf_sat, 'category': 'Performance'})
    excitement = pd.DataFrame({'functionality': func, 'satisfaction': excite_sat, 'category': 'Excitement (Delighter)'})
    return pd.concat([basic, performance, excitement], ignore_index=True)

def generate_anova_data(means: list, stds: list, n: int) -> pd.DataFrame:
    """Generates data for ANOVA examples."""
    data, groups = [], []
    for i, (mean, std) in enumerate(zip(means, stds)):
        data.extend(np.random.normal(mean, std, n))
        groups.extend([f'Supplier {chr(65+i)}'] * n)
    return pd.DataFrame({'Yield': data, 'Supplier': groups})

def generate_sensor_degradation_data() -> pd.DataFrame:
    """Generates sensor data showing degradation over time for RUL plots."""
    np.random.seed(42)
    time = np.arange(0, 100)
    degradation = 0.0015 * time**2.2 + np.random.normal(0, 0.3, 100) + 2.0
    # Add some noise spikes
    degradation[np.random.choice(100, 5, replace=False)] += np.random.normal(2, 0.5, 5)
    return pd.DataFrame({'Time': time, 'Vibration_Signal': degradation})

def generate_pareto_data() -> pd.DataFrame:
    """Generates sample defect data for Pareto charts."""
    return pd.DataFrame({
        'Defect Type': ['Wrong Component', 'Solder Joint Failure', 'PCB Crack', 'Missing Component', 'Cosmetic Flaw', 'Documentation Error'],
        'Count': [88, 42, 25, 12, 8, 4]
    })

def generate_fmea_data() -> pd.DataFrame:
    """Generates sample data for FMEA table."""
    return pd.DataFrame([
        {'Failure Mode': 'Power Supply Overheats', 'Severity': 9, 'Occurrence': 4, 'Detection': 6},
        {'Failure Mode': 'Incorrect Firmware Flashed', 'Severity': 10, 'Occurrence': 2, 'Detection': 8},
        {'Failure Mode': 'Case Tolerances Off', 'Severity': 5, 'Occurrence': 6, 'Detection': 3},
        {'Failure Mode': 'Keypad Button Sticks', 'Severity': 4, 'Occurrence': 7, 'Detection': 2},
    ]).assign(RPN=lambda df: df.Severity * df.Occurrence * df.Detection).sort_values('RPN', ascending=False)

def generate_vsm_data() -> pd.DataFrame:
    """Generates sample data for a Value Stream Map."""
    return pd.DataFrame([
        {"Step": "Order Entry", "CycleTime": 5, "WaitTime": 10, "ValueAdded": True},
        {"Step": "Credit Check", "CycleTime": 15, "WaitTime": 1440, "ValueAdded": False},
        {"Step": "Allocation", "CycleTime": 2, "WaitTime": 60, "ValueAdded": True},
        {"Step": "Picking", "CycleTime": 30, "WaitTime": 240, "ValueAdded": True},
        {"Step": "Packing", "CycleTime": 10, "WaitTime": 30, "ValueAdded": True},
        {"Step": "Shipping", "CycleTime": 5, "WaitTime": 720, "ValueAdded": True},
    ])

def generate_hotelling_data() -> pd.DataFrame:
    """Generates multivariate data for Hotelling T^2 chart."""
    np.random.seed(42)
    # In-control data
    mean_in = [10, 20]
    cov_in = [[1, 0.8], [0.8, 1]]
    data_in = np.random.multivariate_normal(mean_in, cov_in, 80)
    # Out-of-control data (mean shift)
    mean_out = [11, 21.5]
    data_out = np.random.multivariate_normal(mean_out, cov_in, 20)
    data = np.vstack((data_in, data_out))
    return pd.DataFrame(data, columns=['Temp', 'Pressure'])

def generate_causal_data(size: int = 500) -> pd.DataFrame:
    """Generates data with a known causal structure for Causal Discovery."""
    np.random.seed(42)
    x1 = np.random.uniform(size=size)
    x2 = 2 * x1 + np.random.normal(size=size) * 0.1
    x3 = -1 * x1 + np.random.normal(size=size) * 0.1
    x4 = 3 * x2 - 2 * x3 + np.random.normal(size=size) * 0.1
    return pd.DataFrame({'Setting A': x1, 'Sensor 1': x2, 'Sensor 2': x3, 'Output Y': x4})

# ==============================================================================
# SECTION 3: VISUALIZATION HELPERS
# ==============================================================================

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Converts a hex color string to an rgba string."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'

# --- ENHANCED DEFINE PHASE VISUALS ---

def plot_project_charter_visual() -> go.Figure:
    """Creates a dashboard-style visual summary of a Project Charter."""
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, fillcolor=COLORS['background'], line_width=0)
    fig.add_shape(type="rect", x0=0.02, y0=0.02, x1=0.98, y1=0.98, fillcolor='white', line=dict(color=COLORS['light_gray'], width=2))
    fig.add_annotation(x=0.5, y=0.9, text="<b>Project Charter: Reduce Delivery Time</b>", showarrow=False, font=dict(size=20, color=COLORS['primary']))
    kpis = {
        "Avg Delivery Time": ("72h → <36h", COLORS['danger']),
        "Customer Complaints": ("-50% 📉", COLORS['success']),
        "Annual Savings": ("$2M+ 💰", COLORS['success'])
    }
    for i, (k, (v, color)) in enumerate(kpis.items()):
        fig.add_annotation(x=0.2 + i*0.3, y=0.75, text=f"<b>{k}</b>", showarrow=False, font=dict(size=14, color=COLORS['dark_gray']))
        fig.add_annotation(x=0.2 + i*0.3, y=0.65, text=v, showarrow=False, font=dict(size=18, color=color))
    statements = {
        "Problem Statement": (0.05, 0.45, "Avg delivery time has increased by 150% in 6 months,<br>leading to a rise in complaints and order cancellations."),
        "Goal Statement": (0.55, 0.45, "Reduce average end-to-end delivery time from 72 hours<br>to 36 hours by the end of Q3, for all domestic orders."),
        "In Scope": (0.05, 0.2, "✅ Order processing<br>✅ Warehouse picking<br>✅ Local courier logistics"),
        "Out of Scope": (0.55, 0.2, "❌ International shipping<br>❌ Product manufacturing<br>❌ Supplier intake process")
    }
    for title, (x_pos, y_pos, text) in statements.items():
        fig.add_shape(type="rect", x0=x_pos, y0=y_pos-0.18, x1=x_pos+0.4, y1=y_pos+0.1, 
                      fillcolor=hex_to_rgba(COLORS['primary'], 0.1), 
                      line=dict(color=COLORS['primary'], width=1, dash='dot'))
        fig.add_annotation(x=x_pos+0.02, y=y_pos+0.08, text=f"<b>{title}</b>", showarrow=False, align='left', xanchor='left')
        fig.add_annotation(x=x_pos+0.02, y=y_pos-0.05, text=text, showarrow=False, align='left', xanchor='left')
    fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, visible=False), yaxis=dict(showgrid=False, zeroline=False, visible=False),
                      plot_bgcolor=COLORS['background'], margin=dict(t=20, b=20, l=20, r=20), height=400)
    return fig

def plot_sipoc_visual() -> go.Figure:
    """Creates an enhanced, icon-driven SIPOC diagram."""
    cats = ['🏭<br>Suppliers', '📝<br>Inputs', '⚙️<br>Process', '📦<br>Outputs', '👤<br>Customers']
    content = {'🏭<br>Suppliers': '• Component Fab<br>• Logistics Inc.<br>• Software Dev LLC',
               '📝<br>Inputs': '• Silicon Wafers<br>• Assembly Instructions<br>• Firmware v2.1',
               '⚙️<br>Process': '1. Receive Materials<br>2. Assemble Unit<br>3. Flash Firmware<br>4. Quality Test<br>5. Pack & Ship',
               '📦<br>Outputs': '• Assembled Product<br>• Quality Report<br>• Shipping Manifest',
               '👤<br>Customers': '• End Users<br>• Distributors<br>• Service Centers'}
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['neutral_yellow'], COLORS['neutral_pink']]
    fig = go.Figure()
    for i, cat in enumerate(cats):
        fig.add_shape(type="rect", x0=i+0.1, y0=0.1, x1=i+0.9, y1=0.9, line=dict(color=colors[i], width=3), fillcolor=hex_to_rgba(colors[i], 0.2))
        fig.add_annotation(x=i+0.5, y=0.95, text=f"<b>{cat}</b>", showarrow=False, font=dict(size=18, color=colors[i]))
        fig.add_annotation(x=i+0.5, y=0.5, text=content[cat], showarrow=False, align='left', font=dict(size=11))
    for i in range(len(cats) - 1):
        fig.add_annotation(x=i+1, y=0.5, ax=i+1, ay=0.5, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=2, arrowwidth=2.5, arrowcolor=COLORS['dark_gray'])
    fig.update_layout(title_text="<b>SIPOC Diagram:</b> An Icon-Driven View",
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 5]),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
                      plot_bgcolor='white', paper_bgcolor='white', margin=dict(t=50, b=20), height=300)
    return fig

def plot_causal_discovery_visual() -> graphviz.Digraph:
    """Creates a more realistic and visually distinct causal graph."""
    dot = graphviz.Digraph(comment='Causal Graph')
    dot.attr('node', shape='box', style='rounded,filled')
    dot.attr('edge', color=COLORS['dark_gray'], penwidth='1.5')
    dot.attr(rankdir='LR', splines='spline')
    dot.node('Region', 'Supplier<br>Region', fillcolor=hex_to_rgba(COLORS['primary'], 0.2), color=COLORS['primary'])
    dot.node('Complexity', 'Order<br>Complexity', fillcolor=hex_to_rgba(COLORS['primary'], 0.2), color=COLORS['primary'])
    dot.node('Method', 'Shipping<br>Method', fillcolor=hex_to_rgba(COLORS['primary'], 0.2), color=COLORS['primary'])
    dot.node('Inv', 'Inventory<br>Check', fillcolor=hex_to_rgba(COLORS['secondary'], 0.2), color=COLORS['secondary'])
    dot.node('Late', 'Late<br>Delivery', shape='ellipse', fillcolor=hex_to_rgba(COLORS['danger'], 0.3), color=COLORS['danger'], penwidth='2')
    dot.edge('Region', 'Inv'); dot.edge('Complexity', 'Inv'); dot.edge('Inv', 'Late'); dot.edge('Method', 'Late')
    return dot

def plot_ctq_tree_visual() -> graphviz.Digraph:
    """Creates an improved CTQ Tree with a top-down hierarchy and better styling."""
    dot = graphviz.Digraph(comment='CTQ Tree')
    dot.attr('node', shape='box', style='rounded,filled', fontname="helvetica")
    dot.attr(rankdir='TB')
    dot.node('Need', 'Fast & Hot Pizza', fillcolor=COLORS['secondary'], fontcolor='white')
    dot.node('D1', 'Short Total Time', fillcolor=hex_to_rgba(COLORS['primary'], 0.9))
    dot.node('D2', 'High Temp on Arrival', fillcolor=hex_to_rgba(COLORS['primary'], 0.9))
    dot.node('CTQ1', 'Total Time from Order to Door<br><b>< 30 minutes</b>', fillcolor=hex_to_rgba(COLORS['accent'], 0.7))
    dot.node('CTQ2', 'Pizza Internal Temperature<br><b>> 140°F</b>', fillcolor=hex_to_rgba(COLORS['accent'], 0.7))
    dot.edge('Need', 'D1'); dot.edge('Need', 'D2'); dot.edge('D1', 'CTQ1'); dot.edge('D2', 'CTQ2')
    return dot

def plot_kano_visual() -> go.Figure:
    """Creates a visually enhanced Kano plot with satisfaction zones and examples."""
    df = generate_kano_data()
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=10, fillcolor=hex_to_rgba(COLORS['success'], 0.15), line_width=0, layer='below', name='Delight Zone')
    fig.add_shape(type="rect", x0=0, y0=-10, x1=10, y1=0, fillcolor=hex_to_rgba(COLORS['danger'], 0.15), line_width=0, layer='below', name='Dissatisfaction Zone')
    colors = {'Basic (Must-be)': COLORS['accent'], 'Performance': COLORS['primary'], 'Excitement (Delighter)': COLORS['secondary']}
    for cat in df['category'].unique():
        subset = df[df['category'] == cat]
        fig.add_trace(go.Scatter(x=subset['functionality'], y=subset['satisfaction'], mode='lines', name=cat, line=dict(color=colors[cat], width=4)))
    fig.add_annotation(x=8, y=8, text="<b>Delighter</b><br>e.g., Free dessert", showarrow=True, arrowhead=2, ax=-40, ay=-40, font_color=COLORS['secondary'])
    fig.add_annotation(x=8, y=4, text="<b>Performance</b><br>e.g., Pizza taste", showarrow=True, arrowhead=2, ax=0, ay=-40, font_color=COLORS['primary'])
    fig.add_annotation(x=8, y=-1, text="<b>Basic</b><br>e.g., Correct toppings", showarrow=True, arrowhead=2, ax=0, ay=40, font_color=COLORS['accent'])
    fig.add_hline(y=0, line_width=1, line_color='black')
    fig.update_layout(title='<b>Kano Model:</b> Visualizing Customer Satisfaction Drivers',
                      xaxis_title='Functionality Provided →', yaxis_title='← Satisfaction →', plot_bgcolor='white', paper_bgcolor='white',
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'))
    return fig

def plot_voc_treemap() -> go.Figure:
    """Creates a treemap to visualize customer feedback topics and sentiment."""
    data = {'Category': ['Product', 'Product', 'Service', 'Service', 'Website'],
            'Topic': ['Reliability & Bugs', 'Feature Requests', 'Support Experience', 'Shipping Time', 'UI & Usability'],
            'Count': [150, 250, 180, 80, 90], 'Sentiment': [-0.8, 0.5, -0.4, -0.6, 0.1]}
    df = pd.DataFrame(data)
    fig = px.treemap(df, path=[px.Constant("All Feedback"), 'Category', 'Topic'], values='Count', color='Sentiment',
                     color_continuous_scale='RdBu', color_continuous_midpoint=0, custom_data=['Sentiment'])
    fig.update_traces(textinfo='label+value', hovertemplate='<b>%{label}</b><br>Mentions: %{value}<br>Avg. Sentiment: %{customdata[0]:.2f}')
    fig.update_layout(title_text="<b>ML-Powered VOC:</b> Feedback Treemap (Size=Volume, Color=Sentiment)",
                      margin=dict(t=50, l=10, r=10, b=10))
    return fig

# --- MEASURE PHASE PLOTS ---

def plot_gage_rr_variance_components() -> go.Figure:
    data = {'Source': ['% Contribution'], 'Part-to-Part': [85], 'Repeatability': [10], 'Reproducibility': [5]}
    df = pd.DataFrame(data)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df['Source'], x=df['Part-to-Part'], name='Process Variation', orientation='h', marker_color=COLORS['primary']))
    fig.add_trace(go.Bar(y=df['Source'], x=df['Repeatability'], name='Repeatability (Equipment)', orientation='h', marker_color=COLORS['warning']))
    fig.add_trace(go.Bar(y=df['Source'], x=df['Reproducibility'], name='Reproducibility (Operator)', orientation='h', marker_color=COLORS['accent']))
    fig.update_layout(
        barmode='stack', title='<b>Gage R&R:</b> Sources of Variation', xaxis_title='Percentage of Total Variation',
        xaxis=dict(range=[0, 100], ticksuffix='%'), yaxis_visible=False, plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def plot_process_mining_graph() -> graphviz.Digraph:
    dot = graphviz.Digraph(comment='Process Mining', graph_attr={'rankdir': 'LR', 'splines': 'true'})
    dot.attr('node', shape='box', style='rounded,filled', fillcolor=hex_to_rgba(COLORS['primary'], 0.2))
    dot.attr('edge', color=COLORS['dark_gray'], fontname="Helvetica", fontsize="10")
    dot.node('start', 'Start', shape='circle', style='filled', fillcolor=COLORS['secondary'])
    dot.node('end', 'End', shape='doublecircle', style='filled', fillcolor=COLORS['accent'])
    dot.node('A', 'Order Received'); dot.node('B', 'Check Inventory'); dot.node('C', 'Ship Product'); dot.node('D', 'Invoice Sent')
    dot.node('E', 'Request Restock', style='rounded,filled', fillcolor=hex_to_rgba(COLORS['accent'], 0.3))
    dot.node('F', 'Wait for Stock', style='rounded,filled', fillcolor=hex_to_rgba(COLORS['accent'], 0.3))
    dot.edge('start', 'A', label='  15 Cases'); dot.edge('A', 'B', label='  15 Cases')
    dot.edge('B', 'C', label='  10 Cases\nAvg. 5h 0m', penwidth='3.0')
    dot.edge('C', 'D', label='  15 Cases\nAvg. 0h 5m', penwidth='3.5')
    dot.edge('D', 'end', label='  15 Cases')
    dot.edge('B', 'E', label='  5 Cases\nAvg. 0h 5m', color=COLORS['danger'], penwidth='1.5')
    dot.edge('E', 'F', label='  5 Cases\nAvg. 25h 20m', color=COLORS['danger'], penwidth='1.5')
    dot.edge('F', 'C', label='  5 Cases\nAvg. 0h 15m', color=COLORS['danger'], penwidth='1.5', constraint='false')
    return dot

def plot_vsm() -> go.Figure:
    df = generate_vsm_data()
    df['EndTime'] = (df['CycleTime'] + df['WaitTime']).cumsum()
    df['StartTime'] = df['EndTime'] - df['CycleTime'] - df['WaitTime']
    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_shape(type="rect", x0=i, y0=2, x1=i+0.8, y1=3, fillcolor=COLORS['primary'] if row['ValueAdded'] else COLORS['accent'], line_color=COLORS['dark_gray'])
        fig.add_annotation(x=i+0.4, y=2.5, text=row['Step'], showarrow=False, font=dict(color='white'))
        fig.add_annotation(x=i+0.4, y=1.7, text=f"CT: {row['CycleTime']}m", showarrow=False)
        if row['WaitTime'] > 0:
            fig.add_annotation(x=i-0.1, y=1.7, text=f"Wait: {row['WaitTime']//60}h", showarrow=False, font_color=COLORS['danger'])
    total_time = df['EndTime'].max()
    va_time = df[df['ValueAdded']]['CycleTime'].sum()
    lead_time = df['CycleTime'].sum() + df['WaitTime'].sum()
    fig.add_shape(type="line", x0=-0.5, y0=0.5, x1=len(df), y1=0.5, line=dict(color=COLORS['dark_gray'], width=5))
    for i, row in df.iterrows():
        fig.add_shape(type="line", x0=row['StartTime']/total_time * len(df) -0.5, y0=0.5, x1=(row['StartTime'] + row['WaitTime'])/total_time * len(df) - 0.5, y1=0.5, line=dict(color=COLORS['accent'], width=10))
        fig.add_shape(type="line", x0=(row['StartTime'] + row['WaitTime'])/total_time * len(df) - 0.5, y0=0.5, x1=row['EndTime']/total_time * len(df) - 0.5, y1=0.5, line=dict(color=COLORS['secondary'], width=10))
    fig.update_layout(
        title=f"<b>Value Stream Map:</b> Lead Time: {lead_time/60:.1f}h | Value-Added Time: {va_time:.1f}m",
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[0, 4]),
        plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_capability_analysis_pro(data: np.ndarray, lsl: float, usl: float) -> Tuple[go.Figure, float, float]:
    mean, std = np.mean(data), np.std(data)
    if std == 0: return go.Figure(), 0, 0
    cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)); cp = (usl - lsl) / (6 * std)
    kde = gaussian_kde(data)
    x_range = np.linspace(min(lsl, data.min()) - std, max(usl, data.max()) + std, 500)
    kde_y = kde(x_range)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Histogram(x=data, name='Process Data (Count)', marker_color=COLORS['primary'], opacity=0.7), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_range, y=kde_y, mode='lines', name='ML: Kernel Density Estimate', line=dict(color=COLORS['accent'], width=3)), secondary_y=True)
    fig.add_vline(x=lsl, line=dict(color=COLORS['danger'], width=2, dash='dash'), name="LSL")
    fig.add_vline(x=usl, line=dict(color=COLORS['danger'], width=2, dash='dash'), name="USL")
    fig.add_vline(x=mean, line=dict(color=COLORS['dark_gray'], width=2, dash='dot'), name="Mean")
    fig.update_layout(title_text=f"<b>Capability Analysis:</b> Classical Metrics vs. ML Distributional View",
        xaxis_title="Measurement Value", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white', paper_bgcolor='white', font_color=COLORS['text'])
    fig.update_yaxes(title_text="Count (Histogram)", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Probability Density (KDE)", secondary_y=True, showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['light_gray'])
    return fig, cp, cpk

# --- ANALYZE PHASE PLOTS ---

def plot_fishbone_diagram() -> graphviz.Digraph:
    dot = graphviz.Digraph(engine='neato', graph_attr={'splines': 'line'})
    dot.node('Effect', 'Low Product Yield', shape='box', style='filled', fillcolor=hex_to_rgba(COLORS['accent'], 0.3))
    cats = ['Methods', 'Machines', 'Manpower', 'Materials', 'Measurement', 'Environment']
    for i, cat in enumerate(cats):
        dot.node(cat, cat, shape='none')
        dot.edge(cat, 'Effect', arrowhead='none')
    dot.node('c1', 'Poor Training', shape='none'); dot.edge('c1', 'Manpower', arrowhead='none')
    dot.node('c2', 'Inconsistent Setup', shape='none'); dot.edge('c2', 'Methods', arrowhead='none')
    dot.node('c3', 'Old Equipment', shape='none'); dot.edge('c3', 'Machines', arrowhead='none')
    dot.node('c4', 'Supplier Variation', shape='none'); dot.edge('c4', 'Materials', arrowhead='none')
    return dot

def plot_pareto_chart() -> go.Figure:
    df = generate_pareto_data().sort_values('Count', ascending=False)
    df['Cumulative Percentage'] = df['Count'].cumsum() / df['Count'].sum() * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df['Defect Type'], y=df['Count'], name='Defect Count', marker_color=COLORS['primary']), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Defect Type'], y=df['Cumulative Percentage'], name='Cumulative %', mode='lines+markers', line_color=COLORS['accent']), secondary_y=True)
    fig.update_layout(title_text="<b>Pareto Chart:</b> Identifying the 'Vital Few' Defects", xaxis_title="Defect Type",
        plot_bgcolor='white', paper_bgcolor='white', font_color=COLORS['text'], legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Percentage", secondary_y=True, range=[0, 101], ticksuffix='%')
    return fig

def plot_anova_groups(df: pd.DataFrame) -> Tuple[go.Figure, float]:
    groups = df['Supplier'].unique()
    fig = go.Figure()
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
    for i, group in enumerate(groups):
        fig.add_trace(go.Box(y=df[df['Supplier'] == group]['Yield'], name=group, marker_color=colors[i % len(colors)]))
    group_data = [df[df['Supplier'] == g]['Yield'] for g in groups]
    p_val = 1.0
    if len(group_data) > 1 and all(len(g) > 1 for g in group_data):
        _, p_val = f_oneway(*group_data)
    title = f'<b>ANOVA:</b> Comparing Supplier Yields (p-value: {p_val:.4f})'
    fig.update_layout(title=title, yaxis_title='Product Yield (%)', plot_bgcolor='white', paper_bgcolor='white', showlegend=False)
    return fig, p_val

def plot_permutation_test(df: pd.DataFrame, n_permutations: int = 1000) -> go.Figure:
    groups = df['Supplier'].unique()
    group_data = [df[df['Supplier'] == g]['Yield'] for g in groups]
    if len(group_data) < 2: return go.Figure()
    observed_diff = group_data[0].mean() - group_data[1].mean()
    concat_data = np.concatenate(group_data)
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(concat_data)
        new_g1 = concat_data[:len(group_data[0])]
        new_g2 = concat_data[len(group_data[0]):len(group_data[0])+len(group_data[1])]
        perm_diffs.append(new_g1.mean() - new_g2.mean())
    p_val = (np.sum(np.abs(perm_diffs) >= np.abs(observed_diff))) / n_permutations
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=perm_diffs, name='Permuted Differences', marker_color=COLORS['light_gray']))
    fig.add_vline(x=observed_diff, line=dict(color=COLORS['accent'], width=3, dash='dash'), name='Observed Difference')
    fig.update_layout(
        title=f'<b>Permutation Test:</b> {n_permutations} Shuffles (p-value: {p_val:.4f})',
        xaxis_title=f'Difference in Mean Yield ({groups[0]} - {groups[1]})', yaxis_title='Frequency',
        plot_bgcolor='white', paper_bgcolor='white')
    return fig

def plot_regression_comparison_pro(df: pd.DataFrame) -> Tuple[go.Figure, RandomForestRegressor, pd.DataFrame]:
    X = df[['Temp', 'Pressure', 'Vibration_Noise']]; y = df['Yield']
    lin_reg = LinearRegression().fit(X, y)
    y_pred_lin, r2_lin = lin_reg.predict(X), lin_reg.score(X, y)
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True).fit(X, y)
    y_pred_rf, r2_rf = rf_reg.predict(X), rf_reg.oob_score_
    sort_idx = X['Temp'].argsort()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X['Temp'].iloc[sort_idx], y=y.iloc[sort_idx], mode='markers', name='Actual Data', marker=dict(color=COLORS['dark_gray'], opacity=0.4)))
    fig.add_trace(go.Scatter(x=X['Temp'].iloc[sort_idx], y=y_pred_lin[sort_idx], mode='lines', name=f'Classical: Linear Reg (R²={r2_lin:.2f})', line=dict(color=COLORS['primary'], width=3)))
    fig.add_trace(go.Scatter(x=X['Temp'].iloc[sort_idx], y=y_pred_rf[sort_idx], mode='lines', name=f'ML: Random Forest (OOB R²={r2_rf:.2f})', line=dict(color=COLORS['secondary'], width=3, dash='dot')))
    fig.update_layout(
        title_text="<b>Regression Analysis:</b> Model Fit on Non-Linear Data",
        xaxis_title="Primary Feature: Temperature", yaxis_title="Process Yield",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        plot_bgcolor='white', paper_bgcolor='white')
    return fig, rf_reg, X

def plot_shap_summary(model: RandomForestRegressor, X: pd.DataFrame) -> go.Figure:
    explainer = shap.TreeExplainer(model); shap_values = explainer.shap_values(X)
    fig = go.Figure()
    for i, feature in enumerate(X.columns):
        y_vals = np.full(X.shape[0], i) + np.random.uniform(-0.2, 0.2, X.shape[0])
        fig.add_trace(go.Scatter(x=shap_values[:, i], y=y_vals, mode='markers',
            marker=dict(color=X[feature], colorscale='RdBu', showscale=True if i == 0 else False,
                        colorbar=dict(title="Feature Value", x=-0.3), cmin=X[feature].min(), cmax=X[feature].max()),
            showlegend=False, hovertext=[f"{feature}={val:.2f}" for val in X[feature]], hoverinfo='text'))
    fig.update_layout(title="<b>SHAP Summary Plot:</b> Feature Impact on Yield Prediction",
        xaxis_title="SHAP Value (Impact on model output)",
        yaxis=dict(tickmode='array', tickvals=list(range(len(X.columns))), ticktext=X.columns),
        plot_bgcolor='white', paper_bgcolor='white')
    return fig

# --- IMPROVE PHASE PLOTS ---

def plot_doe_cube(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=[go.Scatter3d(x=df['Temp'], y=df['Pressure'], z=df['Time'], mode='markers+text',
        marker=dict(size=12, color=df['Yield'], colorscale='Viridis', showscale=True, colorbar=dict(title='Yield')),
        text=[f"{y:.1f}" for y in df['Yield']], textposition='top center')])
    lines = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if np.sum(df.iloc[i, :3] != df.iloc[j, :3]) == 1:
                lines.append(go.Scatter3d(x=[df.iloc[i]['Temp'], df.iloc[j]['Temp']], y=[df.iloc[i]['Pressure'], df.iloc[j]['Pressure']],
                    z=[df.iloc[i]['Time'], df.iloc[j]['Time']], mode='lines', line=dict(color='grey', width=2), showlegend=False))
    fig.add_traces(lines)
    fig.update_layout(title="<b>DOE:</b> 2³ Factorial Design Cube Plot",
        scene=dict(xaxis_title='A: Temp', yaxis_title='B: Pressure', zaxis_title='C: Time'),
        margin=dict(l=0, r=0, b=0, t=40))
    return fig

def plot_doe_effects(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    main_effects = {factor: df.loc[df[factor] == 1, 'Yield'].mean() - df.loc[df[factor] == -1, 'Yield'].mean() for factor in ['Temp', 'Pressure', 'Time']}
    fig_main = go.Figure(go.Bar(x=list(main_effects.keys()), y=list(main_effects.values()), marker_color=[COLORS['primary'], COLORS['accent'], COLORS['secondary']]))
    fig_main.update_layout(title="<b>DOE:</b> Main Effects Plot", xaxis_title="Factor", yaxis_title="Effect on Yield", plot_bgcolor='white', paper_bgcolor='white')
    fig_int = go.Figure()
    for level in [-1, 1]:
        subset = df[df['Pressure'] == level]; means = subset.groupby('Temp')['Yield'].mean()
        fig_int.add_trace(go.Scatter(x=means.index, y=means.values, mode='lines+markers', name=f'Pressure at {level}'))
    fig_int.update_layout(title="<b>DOE:</b> Interaction Plot (Temp*Pressure)", xaxis_title="Factor: Temperature", yaxis_title="Mean Yield",
        plot_bgcolor='white', paper_bgcolor='white', legend_title_text='Pressure Level')
    return fig_main, fig_int

def plot_bayesian_optimization_interactive(true_func, x_range: np.ndarray, sampled_points: Dict[str, list]) -> Tuple[go.Figure, float]:
    X_sampled = np.array(sampled_points['x']).reshape(-1, 1); y_sampled = np.array(sampled_points['y'])
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True); gp.fit(X_sampled, y_sampled)
    y_mean, y_std = gp.predict(x_range.reshape(-1, 1), return_std=True)
    y_std = np.nan_to_num(y_std, 0); ucb = y_mean + 1.96 * y_std; next_point_x = x_range[np.argmax(ucb)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_range, y=y_mean - 1.96 * y_std, fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=x_range, y=y_mean + 1.96 * y_std, fill='tonexty', mode='lines', name='95% Confidence Interval',
        line=dict(color='rgba(0,0,0,0)'), fillcolor=hex_to_rgba(COLORS['primary'], 0.2)))
    fig.add_trace(go.Scatter(x=x_range, y=true_func(x_range), mode='lines', name='True Function (Hidden)', line=dict(color=COLORS['dark_gray'], width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=X_sampled.ravel(), y=y_sampled, mode='markers', name='Sampled Points', marker=dict(color=COLORS['accent'], size=12, symbol='x', line=dict(width=3))))
    fig.add_trace(go.Scatter(x=x_range, y=y_mean, mode='lines', name='GP Mean (Model Belief)', line=dict(color=COLORS['primary'], width=3)))
    fig.add_trace(go.Scatter(x=x_range, y=ucb, mode='lines', name='Acquisition Function (UCB)', line=dict(color=COLORS['secondary'], width=2, dash='dot')))
    fig.add_vline(x=next_point_x, line=dict(color=COLORS['secondary'], width=3, dash='solid'), name="Next Point to Sample")
    fig.update_layout(title_text="<b>Bayesian Optimization:</b> Intelligent Search for Optimum", xaxis_title="Parameter Setting", yaxis_title="Process Output",
        plot_bgcolor='white', paper_bgcolor='white', font_color=COLORS['text'], legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig, next_point_x

def plot_fmea_table() -> go.Figure:
    df = generate_fmea_data()
    def color_rpn(val):
        if val > 150: return COLORS['danger']
        if val > 80: return COLORS['warning']
        return COLORS['success']
    colors = [color_rpn(rpn) for rpn in df['RPN']]
    fig = go.Figure(data=[go.Table(header=dict(values=list(df.columns), fill_color=COLORS['primary'], font=dict(color='white'), align='left'),
        cells=dict(values=[df.iloc[:, i] for i in range(len(df.columns))], fill_color=[['#FFFFFF']*len(df)]*3 + [colors], align='left'))])
    fig.update_layout(title="<b>FMEA:</b> Risk Priority Number (RPN) Analysis", margin=dict(l=10, r=10, t=50, b=10))
    return fig

def plot_rul_prediction(df: pd.DataFrame) -> go.Figure:
    time, signal = df['Time'].values, df['Vibration_Signal'].values; threshold = 12.0
    model_time, model_signal = time[time < 70], signal[time < 70]
    p = np.polyfit(model_time, np.log(model_signal), 1)
    current_time = 75
    future_time = np.arange(current_time, 120); pred_signal = np.exp(p[1]) * np.exp(p[0] * future_time)
    try:
        failure_idx = np.where(pred_signal >= threshold)[0][0]; time_to_failure = future_time[failure_idx] - current_time
        rul_text = f"Predicted RUL: {time_to_failure:.1f} Hours"
    except IndexError:
        time_to_failure = np.inf; rul_text = "Predicted RUL: > 50 Hours"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=signal, mode='markers', name='Actual Sensor Signal', marker=dict(color=COLORS['dark_gray'], opacity=0.7)))
    fig.add_trace(go.Scatter(x=future_time, y=pred_signal, mode='lines', name='ML Degradation Model', line=dict(color=COLORS['primary'], dash='dash')))
    fig.add_hline(y=threshold, line=dict(color=COLORS['danger'], width=2, dash='solid'), name='Failure Threshold')
    fig.add_vrect(x0=current_time, x1=current_time + (time_to_failure if time_to_failure != np.inf else 50), fillcolor=hex_to_rgba(COLORS['secondary'], 0.2), line_width=0, name='RUL Window')
    fig.add_vline(x=current_time, line=dict(color=COLORS['dark_gray'], width=2), name="Current Time")
    fig.add_annotation(x=current_time, y=threshold + 2, text=rul_text, showarrow=True, arrowhead=1, ax=current_time+15, ay=threshold+5)
    fig.update_layout(title='<b>PHM:</b> Predicting Remaining Useful Life (RUL)', xaxis_title='Operating Time (Hours)', yaxis_title='Vibration Signal (g)',
        plot_bgcolor='white', paper_bgcolor='white', legend=dict(x=0.01, y=0.99))
    return fig

# --- CONTROL PHASE PLOTS ---

def plot_shewhart_chart(df: pd.DataFrame) -> go.Figure:
    mean = df['Value'].iloc[:75].mean(); std_dev = df['Value'].iloc[:75].std()
    ucl, lcl = mean + 3 * std_dev, mean - 3 * std_dev
    violations = df[(df['Value'] > ucl) | (df['Value'] < lcl)]
    first_violation_time = violations['Time'].min() if not violations.empty else None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Value'], mode='lines+markers', name='Data', line=dict(color=COLORS['primary'])))
    fig.add_hline(y=ucl, line=dict(color=COLORS['accent'], dash='dash'), name='UCL')
    fig.add_hline(y=mean, line=dict(color=COLORS['dark_gray'], dash='dot'), name='Center Line')
    fig.add_hline(y=lcl, line=dict(color=COLORS['accent'], dash='dash'), name='LCL')
    if first_violation_time:
        fig.add_vline(x=first_violation_time, line=dict(color=COLORS['danger'], width=2), name='First Detection')
        fig.add_trace(go.Scatter(x=violations['Time'], y=violations['Value'], mode='markers', name='Violation', marker=dict(color=COLORS['danger'], size=10, symbol='x')))
    fig.update_layout(title='<b>Shewhart Chart:</b> Detecting Large Shifts', xaxis_title='Time', yaxis_title='Value', plot_bgcolor='white', paper_bgcolor='white')
    return fig

def plot_ewma_chart(df: pd.DataFrame, lambda_val: float = 0.2) -> go.Figure:
    mean, std_dev = df['Value'].iloc[:75].mean(), df['Value'].iloc[:75].std()
    df['ewma'] = df['Value'].ewm(span=(2 / lambda_val) - 1).mean()
    n = np.arange(1, len(df) + 1); L = 3
    ucl_ewma = mean + L * std_dev * np.sqrt(lambda_val / (2 - lambda_val) * (1 - (1 - lambda_val)**(2 * n)))
    lcl_ewma = mean - L * std_dev * np.sqrt(lambda_val / (2 - lambda_val) * (1 - (1 - lambda_val)**(2 * n)))
    violations = df[(df['ewma'] > ucl_ewma) | (df['ewma'] < lcl_ewma)]
    first_violation_time = violations['Time'].min() if not violations.empty else None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Value'], mode='lines', name='Original Data', line=dict(color=COLORS['light_gray'], width=1)))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['ewma'], mode='lines', name='EWMA', line=dict(color=COLORS['secondary'], width=2.5)))
    fig.add_trace(go.Scatter(x=df['Time'], y=ucl_ewma, mode='lines', name='EWMA UCL', line=dict(color=COLORS['accent'], dash='dash')))
    fig.add_trace(go.Scatter(x=df['Time'], y=lcl_ewma, mode='lines', name='EWMA LCL', line=dict(color=COLORS['accent'], dash='dash')))
    if first_violation_time:
        fig.add_vline(x=first_violation_time, line=dict(color=COLORS['danger'], width=2), name='First Detection')
    fig.update_layout(title=f'<b>EWMA Chart (λ={lambda_val}):</b> Detecting Small, Sustained Shifts', xaxis_title='Time', yaxis_title='Value', plot_bgcolor='white', paper_bgcolor='white')
    return fig

def plot_cusum_chart(df: pd.DataFrame, k: float = 0.5, h: float = 5.0) -> go.Figure:
    mean, std = df['Value'].iloc[:75].mean(), df['Value'].iloc[:75].std()
    target = mean; k_val = k * std; h_val = h * std
    sh, sl = np.zeros(len(df)), np.zeros(len(df))
    for i in range(1, len(df)):
        sh[i] = max(0, sh[i-1] + df['Value'].iloc[i] - target - k_val)
        sl[i] = min(0, sl[i-1] + df['Value'].iloc[i] - target + k_val)
    violations = np.where((sh > h_val) | (sl < -h_val))[0]
    first_violation_time = violations[0] if len(violations) > 0 else None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=sh, name='CUSUM High (Sh)', line=dict(color=COLORS['secondary'])))
    fig.add_trace(go.Scatter(x=df['Time'], y=sl, name='CUSUM Low (Sl)', line=dict(color=COLORS['primary'])))
    fig.add_hline(y=h_val, line=dict(color=COLORS['accent'], dash='dash'), name='Control Limit (H)')
    fig.add_hline(y=-h_val, line=dict(color=COLORS['accent'], dash='dash'), name='Control Limit (-H)')
    if first_violation_time:
        fig.add_vline(x=first_violation_time, line=dict(color=COLORS['danger'], width=2), name='First Detection')
    fig.update_layout(title='<b>CUSUM Chart:</b> Accumulating Small Deviations', xaxis_title='Time', yaxis_title='Cumulative Sum', plot_bgcolor='white', paper_bgcolor='white')
    return fig

def plot_hotelling_t2_chart() -> go.Figure:
    df = generate_hotelling_data(); X = df.iloc[:80, :]
    mean_vec = X.mean().values; cov_mat = np.cov(X.T)
    inv_cov_mat = np.linalg.inv(cov_mat)
    t_squared = [(df.iloc[i, :].values - mean_vec).T @ inv_cov_mat @ (df.iloc[i, :].values - mean_vec) for i in range(len(df))]
    n, p = X.shape; alpha = 0.01
    ucl = (p * (n + 1) * (n - 1)) / (n * (n - p)) * f.ppf(1 - alpha, p, n - p)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=t_squared, mode='lines+markers', name="T² Statistic", line_color=COLORS['primary']))
    fig.add_hline(y=ucl, line=dict(color=COLORS['danger'], dash='dash'), name='UCL')
    rgba_fillcolor = hex_to_rgba(COLORS['accent'], 0.2)
    fig.add_vrect(x0=80, x1=100, fillcolor=rgba_fillcolor, line_width=0, name="Induced Shift")
    fig.update_layout(title="<b>ML Control:</b> Hotelling's T² Chart for Multivariate Data", xaxis_title="Sample Number", yaxis_title="T² Statistic", plot_bgcolor='white', paper_bgcolor='white')
    return fig

def plot_control_plan() -> go.Figure:
    data = {'Process Step': ['Curing', 'Curing', 'Final Assembly'], 'Characteristic (X or Y)': ['Oven Temperature (X)', 'Cure Time (Y)', 'Torque on Bolt A (X)'], 'Specification': ['150 ± 5 °C', '60 ± 2 min', '25 ± 1 Nm'], 'Tool': ['Thermocouple', 'Timer', 'Torque Wrench'], 'Method': ['EWMA Chart', 'X-bar Chart', 'SPC Chart'], 'Sample Size': [1, 5, 1], 'Frequency': ['Continuous', 'Per Batch', 'Per Unit'], 'Reaction Plan': ['Halt & Adjust', 'Investigate Batch', 'Recalibrate Wrench']}
    df = pd.DataFrame(data)
    fig = go.Figure(data=[go.Table(header=dict(values=list(df.columns), fill_color=COLORS['dark_gray'], font=dict(color='white'), align='left', height=40), cells=dict(values=[df[col] for col in df.columns], align='left', height=30))])
    fig.update_layout(title="<b>Control Plan:</b> Formalizing Process Controls", margin=dict(l=10, r=10, t=50, b=10))
    return fig

# --- VISUALIZATIONS FOR COMPARISON & HYBRID PAGES ---

def plot_comparison_radar() -> go.Figure:
    categories = ['Interpretability', 'Data Volume Needs', 'Scalability', 'Handling Complexity', 'Proactive Capability', 'Regulatory Ease']
    classical_scores = [5, 2, 1, 2, 1, 5]; ml_scores = [2, 5, 5, 5, 5, 2]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=classical_scores + [classical_scores[0]], theta=categories + [categories[0]], fill='toself', name='Classical Stats', marker_color=COLORS['primary']))
    fig.add_trace(go.Scatterpolar(r=ml_scores + [ml_scores[0]], theta=categories + [categories[0]], fill='toself', name='Machine Learning', marker_color=COLORS['secondary']))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True, title="<b>Strengths Profile:</b> Classical Stats vs. Machine Learning",
                      plot_bgcolor='white', paper_bgcolor='white', legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
    return fig

def plot_verdict_barchart() -> go.Figure:
    data = {"Metric": ["Interpretability & Simplicity", "Predictive Accuracy", "High-Dimensional Data", "Small-Data Rigor", "Proactive Control", "Auditability"], "Winner": ["Classical", "ML", "ML", "Classical", "ML", "Classical"], "Score": [-1, 1, 1, -1, 1, -1]}
    df = pd.DataFrame(data).sort_values('Score')
    df['Color'] = df['Score'].apply(lambda x: COLORS['primary'] if x < 0 else COLORS['secondary'])
    fig = go.Figure(go.Bar(x=df['Score'], y=df['Metric'], orientation='h', marker_color=df['Color']))
    fig.update_layout(title="<b>Task-Specific Verdict:</b> Which Approach is Better?",
        xaxis=dict(tickvals=[-1, 1], ticktext=['<b>Winner: Classical Stats</b>', '<b>Winner: Machine Learning</b>'], tickfont=dict(size=14), range=[-1.5, 1.5]),
        yaxis_title=None, plot_bgcolor='white', bargap=0.4)
    return fig

def plot_synergy_diagram() -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="circle", x0=0, y0=0, x1=2, y1=2, line_color=COLORS['primary'], fillcolor=COLORS['primary'], opacity=0.6)
    fig.add_shape(type="circle", x0=1.2, y0=0, x1=3.2, y1=2, line_color=COLORS['secondary'], fillcolor=COLORS['secondary'], opacity=0.6)
    fig.add_annotation(x=1, y=1, text="<b>Classical Stats</b><br><i>Inference & Causality</i><br><i>Rigor & Compliance</i>", showarrow=False, font=dict(color="white", size=12))
    fig.add_annotation(x=2.2, y=1, text="<b>Machine Learning</b><br><i>Prediction & Scale</i><br><i>Complexity & Automation</i>", showarrow=False, font=dict(color="white", size=12))
    fig.add_annotation(x=1.6, y=1, text="<b>AI-Augmented<br>Excellence</b>", showarrow=False, font=dict(color="white", size=16, family="Arial Black"))
    fig.update_layout(title="<b>The Hybrid Philosophy:</b> Combining Strengths",
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.5, 3.7]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.5, 2.5]),
        plot_bgcolor='white', margin=dict(t=50, b=10, l=10, r=10))
    return fig

# ==============================================================================
# SECTION 4: CONTENT & LAYOUT HELPERS
# ==============================================================================

def get_guidance_data() -> Dict[str, Dict[str, str]]:
    """Returns the data for the scenario guidance recommender as a dictionary."""
    return {
        "Validating a change for FDA/FAA compliance": {"approach": "🏆 **Classical Stats** (Hypothesis Testing, DOE)", "rationale": "Methods are traceable, validated, and legally defensible, which is paramount for regulatory bodies."},
        "Monitoring a semiconductor fab with 1000s of sensors": {"approach": "🏆 **Machine Learning + SPC** (Multivariate Anomaly Detection)", "rationale": "ML can detect subtle, correlated drifts across thousands of sensors that individual SPC charts would miss, preventing massive yield loss."},
        "Understanding why customers are churning by analyzing support emails": {"approach": "🏆 **Machine Learning NLP** (Topic Modeling & Sentiment Analysis)", "rationale": "NLP can process and extract actionable themes from millions of unstructured text entries, a task impossible for manual analysis."},
        "Optimizing a simple, 3-factor physical mixing process": {"approach": "🏆 **Classical DOE**", "rationale": "Simple, highly effective, and provides clear, interpretable results with a minimal number of experimental runs. The gold standard for this scale."},
        "Building a 'digital twin' of a chemical reactor": {"approach": "🏆 **Hybrid:** ML Model + Bayesian Optimization", "rationale": "ML builds the accurate simulation (the 'twin') from experimental or historical data; Bayesian Optimization then finds the peak efficiency in the vast parameter space without costly physical trials."},
        "Providing real-time operator guidance on a complex assembly line": {"approach": "🏆 **Machine Learning** (Real-time Predictive Model)", "rationale": "An ML model can continuously predict the outcome of the current settings and suggest optimal adjustments to the operator in real-time to prevent defects before they happen."}
    }

def get_workflow_css() -> str:
    """Returns the CSS for the unified workflow diagram."""
    css = f"""
    <style>
    .workflow-container {{ display: flex; flex-direction: column; align-items: center; width: 100%; }}
    .workflow-step {{ 
        background-color: #FFFFFF; border: 1px solid {COLORS['light_gray']}; border-radius: 10px; 
        padding: 20px; margin-bottom: 20px; width: 95%; box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
        border-left: 5px solid; 
    }}
    .workflow-arrow {{ font-size: 28px; color: {COLORS['dark_gray']}; margin-bottom: 20px; font-weight: bold; }}
    .step-define {{ border-left-color: {COLORS['primary']}; }}
    .step-measure {{ border-left-color: {COLORS['secondary']}; }}
    .step-analyze {{ border-left-color: {COLORS['accent']}; }}
    .step-improve {{ border-left-color: {COLORS['neutral_yellow']}; }}
    .step-control {{ border-left-color: {COLORS['neutral_pink']}; }}
    .workflow-step h4 {{ margin-top: 0; margin-bottom: 15px; font-size: 1.5em; color: #333333; }}
    .workflow-step .tool-col {{ padding: 0 15px; }}
    .workflow-step .tool-col h5 {{ color: #555555; border-bottom: 2px solid #EEEEEE; padding-bottom: 5px; margin-bottom: 10px; }}
    .workflow-step .tool-col ul {{ padding-left: 20px; margin: 0; }}
    .workflow-step .tool-col li {{ margin-bottom: 5px; }}
    .tool-col-classical h5 {{ color: {COLORS['primary']}; }}
    .tool-col-ml h5 {{ color: {COLORS['secondary']}; }}
    </style>
    """
    return css

def render_workflow_step(phase_name: str, phase_class: str, classical_tools: List[str], ml_tools: List[str]) -> str:
    """Renders a single step of the HTML workflow diagram."""
    classical_list = "".join([f"<li>{tool}</li>" for tool in classical_tools])
    ml_list = "".join([f"<li>{tool}</li>" for tool in ml_tools])
    return f"""
    <div class="workflow-step {phase_class}">
        <h4>{phase_name}</h4>
        <div style="display: flex;">
            <div style="flex: 1;" class="tool-col tool-col-classical">
                <h5>Classical Tools (The Rigor)</h5>
                <ul>{classical_list}</ul>
            </div>
            <div style="flex: 1;" class="tool-col tool-col-ml">
                <h5>ML Augmentation (The Scale)</h5>
                <ul>{ml_list}</ul>
            </div>
        </div>
    </div>
    """
