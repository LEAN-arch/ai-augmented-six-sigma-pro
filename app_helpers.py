# app_helpers.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import graphviz

# ***** BUG FIX 1: 'f_oneway' is now imported *****
from scipy.stats import norm, gaussian_kde, f, f_oneway

# --- START of config.py content ---
COLORS = {
    "primary": "#0072B2",   # Blue
    "secondary": "#009E73", # Green
    "accent": "#D55E00",    # Orange
    "neutral": "#F0E442",   # Yellow
    "background": "#FAFAFA",
    "text": "#333333",
    "light_gray": "#DDDDDD",
    "dark_gray": "#555555"
}

# ***** BUG FIX 3: 'get_custom_css' function is present and correct *****
def get_custom_css():
    return f"""
    <style>
        .stApp {{ background-color: {COLORS['background']}; color: {COLORS['text']}; }}
        h1, h2, h3 {{ color: {COLORS['dark_gray']}; }}
    </style>
    """
# --- END of config.py content ---


# --- START of data_generator.py content ---
def generate_process_data(mean, std_dev, size, lsl, usl):
    data = np.random.normal(mean, std_dev, size)
    num_outliers = int(size * 0.02)
    if num_outliers > 0:
        outliers = np.concatenate([
            np.random.uniform(lsl - 3 * std_dev, lsl, num_outliers // 2),
            np.random.uniform(usl, usl + 3 * std_dev, num_outliers - (num_outliers // 2))
        ])
        data[:len(outliers)] = outliers
    np.random.shuffle(data)
    return data

def generate_nonlinear_data(size=200):
    X1 = np.linspace(-10, 10, size)
    X2 = X1**2
    X3 = np.random.randn(size) * 5
    y = 2 * X1 + 0.5 * X2 + np.random.normal(0, 5, size)
    return pd.DataFrame({'Feature_1_Linear': X1, 'Feature_2_Quadratic': X2, 'Feature_3_Noise': X3, 'Output': y})

def generate_control_chart_data(mean=100, std_dev=5, size=150, shift_point=100, shift_magnitude=1.5):
    in_control = np.random.normal(mean, std_dev, shift_point)
    out_of_control = np.random.normal(mean + shift_magnitude * std_dev, std_dev, size - shift_point)
    return pd.DataFrame({'Time': np.arange(size), 'Value': np.concatenate([in_control, out_of_control])})

def generate_doe_data():
    factors = [-1, 1]
    data = []
    for f1 in factors:
        for f2 in factors:
            for f3 in factors:
                response = 10 + 2*f1 + 3*f2 - 1.5*f3 + 1.2*f1*f2 + np.random.randn() * 0.5
                data.append([f1, f2, f3, response])
    return pd.DataFrame(data, columns=['Temp', 'Pressure', 'Time', 'Yield'])

# NEW DATA GENERATORS for ENRICHED CONTENT
def generate_kano_data():
    np.random.seed(42)
    basic = pd.DataFrame({'functionality': np.linspace(0, 10, 20), 'satisfaction': np.clip(np.linspace(-8, 0, 20) + np.random.normal(0, 0.5, 20), -10, 0)})
    basic['category'] = 'Basic (Must-be)'
    performance = pd.DataFrame({'functionality': np.linspace(0, 10, 20), 'satisfaction': np.linspace(-5, 5, 20) + np.random.normal(0, 0.8, 20)})
    performance['category'] = 'Performance'
    excitement = pd.DataFrame({'functionality': np.linspace(0, 10, 20), 'satisfaction': np.clip(np.linspace(0, 8, 20) + np.random.normal(0, 0.5, 20), 0, 10)})
    excitement['category'] = 'Excitement (Delighter)'
    return pd.concat([basic, performance, excitement], ignore_index=True)

def generate_anova_data(means, stds, n):
    data, groups = [], []
    for i, (mean, std) in enumerate(zip(means, stds)):
        data.extend(np.random.normal(mean, std, n))
        groups.extend([f'Supplier {chr(65+i)}'] * n)
    return pd.DataFrame({'Yield': data, 'Supplier': groups})

def generate_sensor_degradation_data():
    time = np.arange(0, 100)
    degradation = 0.01 * time**2 + np.random.normal(0, 0.5, 100)
    degradation[85:] += np.random.normal(5, 2, 15)
    return pd.DataFrame({'Time': time, 'Vibration_Signal': degradation})
# --- END of data_generator.py content ---


# --- START of plotting_pro.py content ---
def plot_kano_model():
    df = generate_kano_data()
    fig = go.Figure()
    colors = {'Basic (Must-be)': COLORS['accent'], 'Performance': COLORS['primary'], 'Excitement (Delighter)': COLORS['secondary']}
    for cat in df['category'].unique():
        subset = df[df['category'] == cat]
        fig.add_trace(go.Scatter(x=subset['functionality'], y=subset['satisfaction'], mode='markers+lines', name=cat, marker_color=colors[cat], line_dash='dot'))
    fig.add_hline(y=0, line_width=1, line_color='black')
    fig.add_vline(x=5, line_width=1, line_color='black', line_dash='dash')
    fig.update_layout(title='<b>Kano Model:</b> Prioritizing Customer Requirements', xaxis_title='Degree of Functionality (Provided)', yaxis_title='Customer Satisfaction', plot_bgcolor='white', paper_bgcolor='white', legend=dict(x=0.01, y=0.99))
    return fig

def plot_process_mining_graph():
    dot = graphviz.Digraph(comment='Process Mining', graph_attr={'rankdir': 'LR', 'splines': 'ortho'})
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    dot.attr('edge', color='gray40', fontname="Helvetica", fontsize="10")
    dot.node('start', 'Start', shape='circle', style='filled', fillcolor=COLORS['secondary'])
    dot.node('end', 'End', shape='doublecircle', style='filled', fillcolor=COLORS['accent'])
    dot.edge('start', 'Order Received', label='  15 Cases')
    dot.edge('Order Received', 'Check Inventory', label='  15 Cases')
    dot.edge('Check Inventory', 'Ship Product', label='  10 Cases (Avg. 5h 0m)', penwidth='3.0')
    dot.edge('Ship Product', 'Invoice Sent', label='  15 Cases (Avg. 0h 5m)', penwidth='3.5')
    dot.edge('Invoice Sent', 'end', label='  15 Cases')
    dot.edge('Check Inventory', 'Request Restock', label='  5 Cases (Avg. 0h 5m)', color=COLORS['accent'], penwidth='1.5')
    dot.edge('Request Restock', 'Wait for Stock', label='  5 Cases (Avg. 25h 20m)', color=COLORS['accent'], penwidth='1.5')
    dot.edge('Wait for Stock', 'Ship Product', label='  5 Cases (Avg. 0h 15m)', color=COLORS['accent'], penwidth='1.5')
    return dot

def plot_anova_groups(df):
    groups = df['Supplier'].unique()
    fig = go.Figure()
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
    for i, group in enumerate(groups):
        fig.add_trace(go.Box(y=df[df['Supplier'] == group]['Yield'], name=group, marker_color=colors[i]))
    
    group_data = [df[df['Supplier'] == g]['Yield'] for g in groups]
    if len(group_data) > 1 and all(len(g) > 1 for g in group_data):
        f_val, p_val = f_oneway(*group_data)
        title = f'<b>ANOVA:</b> Comparing Supplier Yields (p-value: {p_val:.4f})'
    else:
        p_val = 1.0
        title = '<b>ANOVA:</b> Comparing Supplier Yields (Not enough data for test)'
    
    fig.update_layout(title=title, yaxis_title='Product Yield (%)', xaxis_title='Supplier', plot_bgcolor='white', paper_bgcolor='white', showlegend=False)
    return fig, p_val

def plot_rul_prediction(df):
    time, signal = df['Time'].values, df['Vibration_Signal'].values
    threshold = 12.0
    model = LinearRegression()
    early_time, early_signal = time[:50].reshape(-1, 1), signal[:50]
    model.fit(early_time, early_signal)
    full_pred = model.predict(time.reshape(-1, 1))
    current_time = 60
    current_pred_signal = model.predict(np.array([[current_time]]))[0]
    time_to_failure = (threshold - current_pred_signal) / model.coef_[0] if model.coef_[0] > 0 else float('inf')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=signal, mode='markers', name='Actual Sensor Signal', marker=dict(color=COLORS['dark_gray'], opacity=0.5)))
    fig.add_trace(go.Scatter(x=time, y=full_pred, mode='lines', name='ML Degradation Model', line=dict(color=COLORS['primary'], dash='dash')))
    fig.add_hline(y=threshold, line=dict(color='red', width=2, dash='solid'), name='Failure Threshold')
    
    if time_to_failure != float('inf') and time_to_failure > 0:
        fig.add_annotation(x=current_time, y=15, text=f"At T={current_time}, Predicted RUL: {time_to_failure:.1f} units", showarrow=True, arrowhead=1, ax=current_time+15, ay=20)
        fig.add_vrect(x0=current_time, x1=current_time + time_to_failure, fillcolor=COLORS['secondary'], opacity=0.2, line_width=0, name='RUL Window')
    
    fig.update_layout(title='<b>PHM:</b> Predicting Remaining Useful Life (RUL)', xaxis_title='Operating Time (Hours)', yaxis_title='Vibration Signal (g)', plot_bgcolor='white', paper_bgcolor='white')
    return fig

def plot_ewma_chart(df, lambda_val=0.2):
    mean, std_dev = df['Value'].iloc[:50].mean(), df['Value'].iloc[:50].std()
    df['ewma'] = df['Value'].ewm(span=(2/lambda_val)-1).mean()
    n = np.arange(1, len(df) + 1)
    L = 3
    ucl_ewma = mean + L * std_dev * np.sqrt(lambda_val / (2 - lambda_val) * (1 - (1 - lambda_val)**(2 * n)))
    lcl_ewma = mean - L * std_dev * np.sqrt(lambda_val / (2 - lambda_val) * (1 - (1 - lambda_val)**(2 * n)))
    violations = df[(df['ewma'] > ucl_ewma) | (df['ewma'] < lcl_ewma)]
    first_violation_time = violations['Time'].min() if not violations.empty else None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Value'], mode='lines', name='Original Data', line=dict(color=COLORS['light_gray'], width=1)))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['ewma'], mode='lines', name='EWMA', line=dict(color=COLORS['primary'], width=2.5)))
    fig.add_trace(go.Scatter(x=df['Time'], y=ucl_ewma, mode='lines', name='EWMA UCL', line=dict(color=COLORS['accent'], dash='dash')))
    fig.add_trace(go.Scatter(x=df['Time'], y=lcl_ewma, mode='lines', name='EWMA LCL', line=dict(color=COLORS['accent'], dash='dash')))
    
    if first_violation_time is not None:
        fig.add_vline(x=first_violation_time, line=dict(color='red', width=2), name='First Detection')
        
    fig.update_layout(title=f'<b>EWMA Chart (λ={lambda_val}):</b> Detecting Small, Sustained Shifts', xaxis_title='Time', yaxis_title='Value', plot_bgcolor='white', paper_bgcolor='white')
    return fig

def plot_voc_nlp_summary():
    topics = ['Pricing & Value', 'Feature Requests', 'Customer Support Experience', 'UI & Usability', 'Reliability & Bugs']; counts = [120, 250, 180, 90, 150]; df = pd.DataFrame({'Topic': topics, 'Count': counts}).sort_values('Count', ascending=True)
    fig = go.Figure(go.Bar(x=df['Count'], y=df['Topic'], orientation='h', marker_color=COLORS['primary'], text=df['Count'], textposition='outside')); fig.update_layout(title_text="<b>ML-Powered VOC:</b> Customer Feedback Topic Distribution", xaxis_title="Mention Count", yaxis_title=None, plot_bgcolor='white', paper_bgcolor='white', font_color=COLORS['text'], xaxis=dict(showgrid=True, gridcolor=COLORS['light_gray']), yaxis=dict(showgrid=False)); return fig

def plot_capability_analysis_pro(data, lsl, usl):
    mean, std = np.mean(data), np.std(data); cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else 0; cp = (usl - lsl) / (6 * std) if std > 0 else 0; kde = gaussian_kde(data)
    x_range = np.linspace(min(lsl, min(data)) - std, max(usl, max(data)) + std, 500); kde_y = kde(x_range); fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Histogram(x=data, name='Process Data (Count)', marker_color=COLORS['primary'], opacity=0.7), secondary_y=False); fig.add_trace(go.Scatter(x=x_range, y=kde_y, mode='lines', name='ML: Kernel Density Estimate', line=dict(color=COLORS['accent'], width=3)), secondary_y=True)
    fig.add_vline(x=lsl, line=dict(color='red', width=2, dash='dash'), name="LSL"); fig.add_vline(x=usl, line=dict(color='red', width=2, dash='dash'), name="USL"); fig.add_vline(x=mean, line=dict(color=COLORS['dark_gray'], width=2, dash='dot'), name="Mean")
    fig.update_layout(title_text=f"<b>Capability Analysis:</b> Classical Metrics vs. ML Distributional View", xaxis_title="Measurement Value", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), plot_bgcolor='white', paper_bgcolor='white', font_color=COLORS['text'])
    fig.update_yaxes(title_text="Count (Histogram)", secondary_y=False, showgrid=False); fig.update_yaxes(title_text="Probability Density (KDE)", secondary_y=True, showgrid=False); fig.update_xaxes(showgrid=True, gridcolor=COLORS['light_gray']); return fig, cp, cpk

def plot_regression_comparison_pro(df):
    X = df[['Feature_1_Linear', 'Feature_2_Quadratic', 'Feature_3_Noise']]; y = df['Output']; lin_reg = LinearRegression().fit(X, y); rf_reg = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y); y_pred_lin, r2_lin = lin_reg.predict(X), lin_reg.score(X, y); y_pred_rf, r2_rf = rf_reg.predict(X), rf_reg.score(X, y); sort_idx = X['Feature_1_Linear'].argsort()
    fig = go.Figure(); fig.add_trace(go.Scatter(x=X['Feature_1_Linear'].iloc[sort_idx], y=y.iloc[sort_idx], mode='markers', name='Actual Data', marker=dict(color=COLORS['dark_gray'], opacity=0.4)))
    fig.add_trace(go.Scatter(x=X['Feature_1_Linear'].iloc[sort_idx], y=y_pred_lin[sort_idx], mode='lines', name=f'Classical: Linear Regression (R²={r2_lin:.2f})', line=dict(color=COLORS['primary'], width=3))); fig.add_trace(go.Scatter(x=X['Feature_1_Linear'].iloc[sort_idx], y=y_pred_rf[sort_idx], mode='lines', name=f'ML: Random Forest (R²={r2_rf:.2f})', line=dict(color=COLORS['accent'], width=3, dash='dot')))
    fig.update_layout(title_text="<b>Regression Analysis:</b> Model Fit on Non-Linear Data", xaxis_title="Primary Feature Value", yaxis_title="Process Output", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), plot_bgcolor='white', paper_bgcolor='white', font_color=COLORS['text'], xaxis=dict(showgrid=True, gridcolor=COLORS['light_gray']), yaxis=dict(showgrid=True, gridcolor=COLORS['light_gray'])); return fig, rf_reg, X

def plot_bayesian_optimization_interactive(true_func, x_range, sampled_points):
    X_sampled = np.array(sampled_points['x']).reshape(-1, 1); y_sampled = np.array(sampled_points['y']); kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)); gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=0.1, normalize_y=True); gp.fit(X_sampled, y_sampled)
    y_mean, y_std = gp.predict(x_range.reshape(-1, 1), return_std=True); ucb = y_mean + 1.96 * y_std; next_point_x = x_range[np.argmax(ucb)]; fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_range, y=y_mean - 1.96 * y_std, fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False)); fig.add_trace(go.Scatter(x=x_range, y=y_mean + 1.96 * y_std, fill='tonexty', mode='lines', name='95% Confidence Interval', line=dict(color='rgba(0,0,0,0)'), fillcolor=f'rgba({int(COLORS["primary"][1:3], 16)}, {int(COLORS["primary"][3:5], 16)}, {int(COLORS["primary"][5:7], 16)}, 0.2)'))
    fig.add_trace(go.Scatter(x=x_range, y=true_func(x_range), mode='lines', name='True Function (Hidden)', line=dict(color=COLORS['dark_gray'], width=2, dash='dash'))); fig.add_trace(go.Scatter(x=X_sampled.ravel(), y=y_sampled, mode='markers', name='Sampled Points', marker=dict(color=COLORS['accent'], size=10, symbol='x-thin', line_width=2)))
    fig.add_trace(go.Scatter(x=x_range, y=y_mean, mode='lines', name='GP Mean (Model Belief)', line=dict(color=COLORS['primary'], width=3))); fig.add_trace(go.Scatter(x=x_range, y=ucb, mode='lines', name='Acquisition Function (UCB)', line=dict(color=COLORS['secondary'], width=2, dash='dot'))); fig.add_vline(x=next_point_x, line=dict(color=COLORS['secondary'], width=2, dash='solid'), name="Next Point to Sample")
    fig.update_layout(title_text="<b>Bayesian Optimization:</b> Intelligent Search for Optimum", xaxis_title="Parameter Setting", yaxis_title="Process Output", plot_bgcolor='white', paper_bgcolor='white', font_color=COLORS['text'], legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); return fig, next_point_x

def plot_control_chart_pro(df):
    mean, std_dev = df['Value'].iloc[:100].mean(), df['Value'].iloc[:100].std(); ucl, lcl = mean + 3 * std_dev, mean - 3 * std_dev; window = 10; df['RollingMean'] = df['Value'].rolling(window=window).mean(); df['RollingStd'] = df['Value'].rolling(window=window).std(); df['Z_Score'] = ((df['Value'] - df['RollingMean']) / df['RollingStd']).fillna(0)
    spc_violations = df[(df['Value'] > ucl) | (df['Value'] < lcl)]; ml_anomalies = df[df['Z_Score'].abs() > 2.5]; fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Value'], mode='lines+markers', name='Process Data', marker_color=COLORS['primary']), row=1, col=1); fig.add_hline(y=ucl, line=dict(color=COLORS['accent'], width=2, dash='dash'), name="UCL", row=1, col=1); fig.add_hline(y=mean, line=dict(color=COLORS['dark_gray'], width=2, dash='dot'), name="Center Line", row=1, col=1); fig.add_hline(y=lcl, line=dict(color=COLORS['accent'], width=2, dash='dash'), name="LCL", row=1, col=1); fig.add_trace(go.Scatter(x=spc_violations['Time'], y=spc_violations['Value'], mode='markers', name='SPC Violation', marker=dict(color=COLORS['accent'], size=12, symbol='x')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Z_Score'].abs(), mode='lines', name='ML Anomaly Score', line=dict(color=COLORS['secondary']), fill='tozeroy', fillcolor=f'rgba({int(COLORS["secondary"][1:3], 16)}, {int(COLORS["secondary"][3:5], 16)}, {int(COLORS["secondary"][5:7], 16)}, 0.2)'), row=2, col=1); fig.add_hline(y=2.5, line=dict(color=COLORS['accent'], width=2, dash='dash'), name="ML Threshold", row=2, col=1); fig.add_trace(go.Scatter(x=ml_anomalies['Time'], y=ml_anomalies['Z_Score'].abs(), mode='markers', name='ML Anomaly Detected', marker=dict(color=COLORS['accent'], size=10, symbol='star')), row=2, col=1)
    fig.update_layout(title_text="<b>Control Phase:</b> Classical SPC vs. Predictive ML Anomaly Detection", plot_bgcolor='white', paper_bgcolor='white', font_color=COLORS['text'], showlegend=False, margin=dict(t=50, l=10, r=10, b=10)); fig.update_yaxes(title_text="Measurement", row=1, col=1, showgrid=True, gridcolor=COLORS['light_gray']); fig.update_yaxes(title_text="Anomaly Score", row=2, col=1, showgrid=True, gridcolor=COLORS['light_gray']); fig.update_xaxes(title_text="Time / Sample Number", row=2, col=1, showgrid=True, gridcolor=COLORS['light_gray'])
    # ***** BUG FIX 2: This function now only returns one value *****
    return fig

def plot_doe_cube(df):
    fig = go.Figure(data=[go.Scatter3d(x=df['Temp'], y=df['Pressure'], z=df['Time'], mode='markers+text', marker=dict(size=12, color=df['Yield'], colorscale='Viridis', showscale=True, colorbar=dict(title='Yield')), text=[f"{y:.1f}" for y in df['Yield']], textposition='top center')])
    lines = [];
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if np.sum(df.iloc[i, :3] != df.iloc[j, :3]) == 1: lines.append(go.Scatter3d(x=[df.iloc[i]['Temp'], df.iloc[j]['Temp']], y=[df.iloc[i]['Pressure'], df.iloc[j]['Pressure']], z=[df.iloc[i]['Time'], df.iloc[j]['Time']], mode='lines', line=dict(color='grey', width=2), showlegend=False))
    fig.add_traces(lines); fig.update_layout(title="Classical DOE: 2³ Factorial Design Cube Plot", scene=dict(xaxis_title='Factor A: Temp', yaxis_title='Factor B: Pressure', zaxis_title='Factor C: Time'), margin=dict(l=0, r=0, b=0, t=40)); return fig
# --- END of plotting_pro.py content ---
