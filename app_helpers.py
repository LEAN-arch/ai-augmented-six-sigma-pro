# app_helpers.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm, gaussian_kde
import graphviz

# --- START of merged config.py ---
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

def get_custom_css():
    return f"""
    <style>
        .stApp {{ background-color: {COLORS['background']}; color: {COLORS['text']}; }}
        h1, h2, h3 {{ color: {COLORS['dark_gray']}; }}
    </style>
    """
# --- END of merged config.py ---


# --- START of merged data_generator.py ---
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
    in_control_data = np.random.normal(mean, std_dev, shift_point)
    out_of_control_data = np.random.normal(mean + shift_magnitude * std_dev, std_dev, size - shift_point)
    full_process_data = np.concatenate([in_control_data, out_of_control_data])
    time_index = np.arange(size)
    return pd.DataFrame({'Time': time_index, 'Value': full_process_data})

def generate_doe_data():
    factors = [-1, 1]
    data = []
    for f1 in factors:
        for f2 in factors:
            for f3 in factors:
                response = 10 + 2*f1 + 3*f2 - 1.5*f3 + 1.2*f1*f2 + np.random.randn() * 0.5
                data.append([f1, f2, f3, response])
    return pd.DataFrame(data, columns=['Temp', 'Pressure', 'Time', 'Yield'])
# --- END of merged data_generator.py ---


# --- START of merged plotting_pro.py ---
def plot_voc_nlp_summary():
    topics = ['Pricing & Value', 'Feature Requests', 'Customer Support Experience', 'UI & Usability', 'Reliability & Bugs']
    counts = [120, 250, 180, 90, 150]
    df = pd.DataFrame({'Topic': topics, 'Count': counts}).sort_values('Count', ascending=True)
    fig = go.Figure(go.Bar(x=df['Count'], y=df['Topic'], orientation='h', marker_color=COLORS['primary'], text=df['Count'], textposition='outside'))
    fig.update_layout(title_text="<b>ML-Powered VOC:</b> Customer Feedback Topic Distribution", xaxis_title="Mention Count", yaxis_title=None, plot_bgcolor='white', paper_bgcolor='white', font_color=COLORS['text'], xaxis=dict(showgrid=True, gridcolor=COLORS['light_gray']), yaxis=dict(showgrid=False))
    return fig

def plot_capability_analysis_pro(data, lsl, usl):
    mean, std = np.mean(data), np.std(data)
    cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else 0
    cp = (usl - lsl) / (6 * std) if std > 0 else 0
    kde = gaussian_kde(data)
    x_range = np.linspace(min(lsl, min(data)) - std, max(usl, max(data)) + std, 500)
    kde_y = kde(x_range)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Histogram(x=data, name='Process Data (Count)', marker_color=COLORS['primary'], opacity=0.7), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_range, y=kde_y, mode='lines', name='ML: Kernel Density Estimate', line=dict(color=COLORS['accent'], width=3)), secondary_y=True)
    fig.add_vline(x=lsl, line=dict(color='red', width=2, dash='dash'), name="LSL")
    fig.add_vline(x=usl, line=dict(color='red', width=2, dash='dash'), name="USL")
    fig.add_vline(x=mean, line=dict(color=COLORS['dark_gray'], width=2, dash='dot'), name="Mean")
    fig.update_layout(title_text=f"<b>Capability Analysis:</b> Classical Metrics vs. ML Distributional View", xaxis_title="Measurement Value", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), plot_bgcolor='white', paper_bgcolor='white', font_color=COLORS['text'])
    fig.update_yaxes(title_text="Count (Histogram)", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Probability Density (KDE)", secondary_y=True, showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['light_gray'])
    return fig, cp, cpk

def plot_regression_comparison_pro(df):
    X = df[['Feature_1_Linear', 'Feature_2_Quadratic', 'Feature_3_Noise']]
    y = df['Output']
    lin_reg = LinearRegression().fit(X, y)
    rf_reg = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
    y_pred_lin, r2_lin = lin_reg.predict(X), lin_reg.score(X, y)
    y_pred_rf, r2_rf = rf_reg.predict(X), rf_reg.score(X, y)
    sort_idx = X['Feature_1_Linear'].argsort()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X['Feature_1_Linear'].iloc[sort_idx], y=y.iloc[sort_idx], mode='markers', name='Actual Data', marker=dict(color=COLORS['dark_gray'], opacity=0.4)))
    fig.add_trace(go.Scatter(x=X['Feature_1_Linear'].iloc[sort_idx], y=y_pred_lin[sort_idx], mode='lines', name=f'Classical: Linear Regression (R²={r2_lin:.2f})', line=dict(color=COLORS['primary'], width=3)))
    fig.add_trace(go.Scatter(x=X['Feature_1_Linear'].iloc[sort_idx], y=y_pred_rf[sort_idx], mode='lines', name=f'ML: Random Forest (R²={r2_rf:.2f})', line=dict(color=COLORS['accent'], width=3, dash='dot')))
    fig.update_layout(title_text="<b>Regression Analysis:</b> Model Fit on Non-Linear Data", xaxis_title="Primary Feature Value", yaxis_title="Process Output", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), plot_bgcolor='white', paper_bgcolor='white', font_color=COLORS['text'], xaxis=dict(showgrid=True, gridcolor=COLORS['light_gray']), yaxis=dict(showgrid=True, gridcolor=COLORS['light_gray']))
    return fig, rf_reg, X

def plot_bayesian_optimization_interactive(true_func, x_range, sampled_points):
    X_sampled = np.array(sampled_points['x']).reshape(-1, 1)
    y_sampled = np.array(sampled_points['y'])
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=0.1, normalize_y=True)
    gp.fit(X_sampled, y_sampled)
    y_mean, y_std = gp.predict(x_range.reshape(-1, 1), return_std=True)
    ucb = y_mean + 1.96 * y_std
    next_point_x = x_range[np.argmax(ucb)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_range, y=y_mean - 1.96 * y_std, fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=x_range, y=y_mean + 1.96 * y_std, fill='tonexty', mode='lines', name='95% Confidence Interval', line=dict(color='rgba(0,0,0,0)'), fillcolor=f'rgba({int(COLORS["primary"][1:3], 16)}, {int(COLORS["primary"][3:5], 16)}, {int(COLORS["primary"][5:7], 16)}, 0.2)'))
    fig.add_trace(go.Scatter(x=x_range, y=true_func(x_range), mode='lines', name='True Function (Hidden)', line=dict(color=COLORS['dark_gray'], width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=X_sampled.ravel(), y=y_sampled, mode='markers', name='Sampled Points', marker=dict(color=COLORS['accent'], size=10, symbol='x-thin', line_width=2)))
    fig.add_trace(go.Scatter(x=x_range, y=y_mean, mode='lines', name='GP Mean (Model Belief)', line=dict(color=COLORS['primary'], width=3)))
    fig.add_trace(go.Scatter(x=x_range, y=ucb, mode='lines', name='Acquisition Function (UCB)', line=dict(color=COLORS['secondary'], width=2, dash='dot')))
    fig.add_vline(x=next_point_x, line=dict(color=COLORS['secondary'], width=2, dash='solid'), name="Next Point to Sample")
    fig.update_layout(title_text="<b>Bayesian Optimization:</b> Intelligent Search for Optimum", xaxis_title="Parameter Setting", yaxis_title="Process Output", plot_bgcolor='white', paper_bgcolor='white', font_color=COLORS['text'], legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig, next_point_x

def plot_control_chart_pro(df):
    mean, std_dev = df['Value'].iloc[:100].mean(), df['Value'].iloc[:100].std()
    ucl, lcl = mean + 3 * std_dev, mean - 3 * std_dev
    window = 10
    df['RollingMean'] = df['Value'].rolling(window=window).mean()
    df['RollingStd'] = df['Value'].rolling(window=window).std()
    df['Z_Score'] = ((df['Value'] - df['RollingMean']) / df['RollingStd']).fillna(0)
    spc_violations = df[(df['Value'] > ucl) | (df['Value'] < lcl)]
    ml_anomalies = df[df['Z_Score'].abs() > 2.5]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Value'], mode='lines+markers', name='Process Data', marker_color=COLORS['primary']), row=1, col=1)
    fig.add_hline(y=ucl, line=dict(color=COLORS['accent'], width=2, dash='dash'), name="UCL", row=1, col=1)
    fig.add_hline(y=mean, line=dict(color=COLORS['dark_gray'], width=2, dash='dot'), name="Center Line", row=1, col=1)
    fig.add_hline(y=lcl, line=dict(color=COLORS['accent'], width=2, dash='dash'), name="LCL", row=1, col=1)
    fig.add_trace(go.Scatter(x=spc_violations['Time'], y=spc_violations['Value'], mode='markers', name='SPC Violation', marker=dict(color=COLORS['accent'], size=12, symbol='x')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Z_Score'].abs(), mode='lines', name='ML Anomaly Score', line=dict(color=COLORS['secondary']), fill='tozeroy', fillcolor=f'rgba({int(COLORS["secondary"][1:3], 16)}, {int(COLORS["secondary"][3:5], 16)}, {int(COLORS["secondary"][5:7], 16)}, 0.2)'), row=2, col=1)
    fig.add_hline(y=2.5, line=dict(color=COLORS['accent'], width=2, dash='dash'), name="ML Threshold", row=2, col=1)
    fig.add_trace(go.Scatter(x=ml_anomalies['Time'], y=ml_anomalies['Z_Score'].abs(), mode='markers', name='ML Anomaly Detected', marker=dict(color=COLORS['accent'], size=10, symbol='star')), row=2, col=1)
    fig.update_layout(title_text="<b>Control Phase:</b> Classical SPC vs. Predictive ML Anomaly Detection", plot_bgcolor='white', paper_bgcolor='white', font_color=COLORS['text'], showlegend=False, margin=dict(t=50, l=10, r=10, b=10))
    fig.update_yaxes(title_text="Measurement", row=1, col=1, showgrid=True, gridcolor=COLORS['light_gray'])
    fig.update_yaxes(title_text="Anomaly Score", row=2, col=1, showgrid=True, gridcolor=COLORS['light_gray'])
    fig.update_xaxes(title_text="Time / Sample Number", row=2, col=1, showgrid=True, gridcolor=COLORS['light_gray'])
    return fig

def plot_doe_cube(df):
    fig = go.Figure(data=[go.Scatter3d(
        x=df['Temp'], y=df['Pressure'], z=df['Time'],
        mode='markers+text',
        marker=dict(size=12, color=df['Yield'], colorscale='Viridis', showscale=True, colorbar=dict(title='Yield')),
        text=[f"{y:.1f}" for y in df['Yield']],
        textposition='top center'
    )])
    lines = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if np.sum(df.iloc[i, :3] != df.iloc[j, :3]) == 1:
                lines.append(go.Scatter3d(
                    x=[df.iloc[i]['Temp'], df.iloc[j]['Temp']],
                    y=[df.iloc[i]['Pressure'], df.iloc[j]['Pressure']],
                    z=[df.iloc[i]['Time'], df.iloc[j]['Time']],
                    mode='lines', line=dict(color='grey', width=2), showlegend=False
                ))
    fig.add_traces(lines)
    fig.update_layout(
        title="Classical DOE: 2³ Factorial Design Cube Plot",
        scene=dict(xaxis_title='Factor A: Temp', yaxis_title='Factor B: Pressure', zaxis_title='Factor C: Time'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig
# --- END of merged plotting_pro.py ---
