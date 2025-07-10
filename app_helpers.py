# app_pages.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import f_oneway

# Import all necessary helper functions from the single helper file
# Using '*' is appropriate here as this module is designed to consume all helpers.
from app_helpers import *

# ==============================================================================
# PAGE 1: DEFINE PHASE
# ==============================================================================
def show_define_phase():
    st.title("🌀 Define Phase: Establishing the Foundation")
    st.markdown("""
    **Objective:** To clearly articulate the business problem, project goals, scope, and high-level process map. This phase is critical for ensuring the team is aligned and that the project is focused on a tangible, valuable business outcome.
    """)
    st.markdown("---")

    st.header("1. Understanding the Process & Scope")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classical Tool: SIPOC")
        st.info("""
        **What is it?** A qualitative, high-level map identifying **S**uppliers, **I**nputs, **P**rocess, **O**utputs, and **C**ustomers. It defines project boundaries.
        - **Strength:** Unmatched for simplicity and facilitating team alignment. It's the essential first step to get everyone on the same page.
        - **Caveat:** Its value is entirely dependent on the existing domain knowledge of the team. It cannot uncover relationships or variables that the team is not already aware of.
        """)
    with col2:
        st.subheader("ML Counterpart: Causal Discovery")
        st.info("""
        **What is it?** Algorithms (e.g., PC, LiNGAM) that analyze observational data to infer a graph of probable cause-and-effect relationships.
        - **Strength:** Objectively discovers potential causal links and latent interactions that human experts might overlook. Excellent for generating data-driven hypotheses about process drivers.
        - **Caveat:** Requires large, high-quality datasets and outputs hypotheses that need validation, not proven facts.
        """)

    st.header("2. Understanding Customer Needs")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classical Tool: Kano Model")
        st.info("""
        **What is it?** A framework for prioritizing customer requirements by categorizing them into **Basic** (must-haves), **Performance** (more is better), and **Excitement** (delighters).
        - **Strength:** Moves beyond a simple ranked list to understand the *non-linear impact* of features on satisfaction. Prevents over-investing in basic features.
        - **Caveat:** Requires structured survey data, which can be time-consuming to collect and analyze.
        """)
        st.plotly_chart(plot_kano_model(), use_container_width=True)
    with col2:
        st.subheader("ML Counterpart: NLP Topic Modeling")
        st.info("""
        **What is it?** Using algorithms (e.g., LDA, BERTopic) to analyze vast amounts of unstructured text data (reviews, support tickets) to automatically extract themes, sentiment, and key topics.
        - **Strength:** Massively scalable, real-time, and objective. Can analyze millions of data points to instantly uncover emerging trends and quantify issue prevalence.
        - **Caveat:** Requires access to text data; may miss the 'why' without deeper qualitative follow-up.
        """)
        st.plotly_chart(plot_voc_nlp_summary(), use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy:**
    1.  **Scope with SIPOC:** Start by drafting a SIPOC with the team to establish boundaries and a shared language.
    2.  **Quantify & Prioritize with NLP & Kano:** Use NLP on existing customer feedback to get a raw, data-driven list of customer issues. Then, use a targeted Kano-style survey on a smaller customer segment to correctly prioritize these issues based on their actual impact on satisfaction.
    3.  **Generate Hypotheses with Causal Graphs:** If historical process data exists, run causal discovery to generate a data-driven "proto-fishbone" diagram to guide the Measure phase.
    """)


# ==============================================================================
# PAGE 2: MEASURE PHASE
# ==============================================================================
def show_measure_phase():
    st.title("🔬 Measure Phase: Quantifying Process Performance")
    st.markdown("""
    **Objective:** To validate the measurement system's reliability and then establish a robust, data-driven baseline of the process's current performance. The mantra is "if you can't measure it, you can't improve it."
    """)
    st.markdown("---")

    with st.expander("⚠️ Foundational Prerequisite: Measurement System Analysis (MSA)"):
        st.warning("""
        An SME would never proceed without first validating the measurement system. Before trusting your data, you must ensure it is reliable.
        - **Classical MSA (Gage R&R):** A designed experiment to quantify how much variation in your data comes from the measurement system itself versus the actual process. It assesses **repeatability** (same operator, same part) and **reproducibility** (different operators, same part). The total Gage R&R variation should be <10% of the process tolerance.
        - **ML Approach (Uncertainty Quantification):** Bayesian models can learn a distribution for each measurement, directly modeling the measurement error as part of the overall process model. This is more dynamic but less standardized than a Gage R&R.
        """)

    st.header("1. Understanding the Process Flow")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classical Tool: Value Stream Mapping (VSM)")
        st.info("""
        **What is it?** A detailed flowcharting method that documents every step in a process, including critical metrics like cycle time, wait time, and value-added time. It helps identify bottlenecks and waste (Muda).
        - **Strength:** Provides a comprehensive, holistic view. Forces the team to physically walk the process and build consensus.
        - **Caveat:** A manual, time-consuming snapshot in time. It represents the process as it *should* be, not necessarily as it *is*, and struggles to capture complex deviations.
        """)
    with col2:
        st.subheader("ML Counterpart: Process Mining")
        st.info("""
        **What is it?** Algorithms that automatically discover and visualize a real process model directly from event logs in IT systems (e.g., an ERP or CRM).
        - **Strength:** Discovers the process as it *actually* happens, including all the unexpected deviations, rework loops, and true bottlenecks. It's objective, data-driven, and dynamic.
        - **Caveat:** Requires clean, structured event log data with a case ID, activity name, and timestamp.
        """)
        st.graphviz_chart(plot_process_mining_graph())
        st.caption("A process mining graph showing the main 'happy path' (thick lines) and a costly rework loop (red lines) with average cycle times.")

    st.header("2. Understanding the Process Capability")
    st.markdown("Capability analysis assesses whether a process is capable of consistently meeting customer specifications.")
    st.sidebar.header("Capability Simulator")
    st.sidebar.markdown("Adjust the process parameters to see how they affect capability.")
    lsl = st.sidebar.slider("LSL", 80.0, 95.0, 90.0, key="m_lsl")
    usl = st.sidebar.slider("USL", 105.0, 120.0, 110.0, key="m_usl")
    process_mean = st.sidebar.slider("Process Mean (μ)", 95.0, 105.0, 101.5, key="m_mean")
    process_std = st.sidebar.slider("Process Std Dev (σ)", 0.5, 5.0, 2.0, key="m_std")
    
    col3, col4 = st.columns([1, 2])
    with col3:
        st.subheader("Classical: Cp & Cpk")
        st.info("Industry-standard indices that summarize capability, assuming normality.")
        with st.expander("Formulas & Interpretation"):
            st.latex(r''' C_p = \frac{USL - LSL}{6\sigma} \quad (\text{Potential})''')
            st.latex(r''' C_{pk} = \min\left(\frac{USL - \mu}{3\sigma}, \frac{\mu - LSL}{3\sigma}\right) \quad (\text{Actual})''')
        data = generate_process_data(process_mean, process_std, 1000, lsl, usl)
        fig, cp, cpk = plot_capability_analysis_pro(data, lsl, usl)
        st.metric("Process Potential (Cp)", f"{cp:.2f}")
        st.metric("Process Capability (Cpk)", f"{cpk:.2f}")
        if cpk < 1.33: st.warning("Process is marginal or not capable.", icon="⚠️")
        else: st.success("Process is capable.", icon="✅")
    with col4:
        st.subheader("ML: Distributional View")
        st.info("Non-parametric methods like **Kernel Density Estimation (KDE)** visualize the *true* shape of the process data, revealing issues like skewness or bimodality that single-point indices hide.")
        st.plotly_chart(fig, use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy:**
    1.  **Discover with Process Mining:** Start by running process mining on system logs to get an objective map of the real process flow and identify major bottlenecks or rework loops.
    2.  **Detail with VSM:** Use the insights from process mining to guide a targeted, physical VSM exercise on the most problematic parts of the process.
    3.  **Baseline with Cpk, Diagnose with KDE:** After a successful MSA, report the official Cpk baseline. Use the KDE plot internally to diagnose the root cause of poor capability (shift, spread, or non-normality).
    """)


# ==============================================================================
# PAGE 3: ANALYZE PHASE
# ==============================================================================
def show_analyze_phase():
    st.title("📈 Analyze Phase: Discovering Root Causes")
    st.markdown("""
    **Objective:** To analyze the data to identify, validate, and quantify the root cause(s) of defects or variation. This involves moving from *what* is happening to *why* it is happening.
    """)
    st.markdown("---")

    st.header("1. Comparing Group Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classical: ANOVA")
        st.info("""
        **What is it?** Analysis of Variance (ANOVA) is a statistical test used to determine if there are any statistically significant differences between the means of two or more independent groups.
        - **Strength:** A rigorous, standard method for comparing groups (e.g., yield from different machines, shifts, or suppliers).
        - **Caveat:** Assumes normality and equal variances in the groups. Can be sensitive to outliers.
        """)
    with col2:
        st.subheader("ML Counterpart: Permutation Testing")
        st.info("""
        **What is it?** A non-parametric, computational method. It works by shuffling the group labels thousands of times and recalculating the statistic (e.g., difference in means) for each shuffle to build an empirical distribution of what's possible under the null hypothesis.
        - **Strength:** Makes no assumptions about the data's distribution. It's more robust and intuitive than classical tests.
        - **Caveat:** Can be computationally intensive for very large datasets.
        """)

    st.sidebar.header("ANOVA Simulator")
    st.sidebar.markdown("Adjust the means of three suppliers to see if the difference becomes statistically significant.")
    mean1 = st.sidebar.slider("Supplier A Mean Yield", 98.0, 102.0, 99.5, 0.1, key='a1')
    mean2 = st.sidebar.slider("Supplier B Mean Yield", 98.0, 102.0, 100.0, 0.1, key='a2')
    mean3 = st.sidebar.slider("Supplier C Mean Yield", 98.0, 102.0, 100.5, 0.1, key='a3')
    anova_data = generate_anova_data(means=[mean1, mean2, mean3], stds=[0.5, 0.5, 0.5], n=50)
    fig_anova, p_val = plot_anova_groups(anova_data)
    st.plotly_chart(fig_anova, use_container_width=True)
    if p_val < 0.05: st.error(f"P-value is {p_val:.4f}. We reject the null hypothesis: there is a significant difference between suppliers.", icon="🚨")
    else: st.success(f"P-value is {p_val:.4f}. We fail to reject the null hypothesis: no significant difference detected.", icon="✅")

    st.header("2. Identifying Variable Relationships")
    df_reg = generate_nonlinear_data()
    fig_reg, model, X_reg = plot_regression_comparison_pro(df_reg)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Classical: Linear Regression")
        st.info("Models the linear relationship between inputs and outputs. Simple and interpretable, but often fails to capture real-world complexity.")
        with st.expander("Model & Assumptions"):
            st.latex(r''' Y = \beta_0 + \beta_1X_1 + \dots + \epsilon ''')
            st.markdown("Assumes **L**inearity, **I**ndependence, **N**ormality, and **E**qual Variance.")
        st.plotly_chart(fig_reg, use_container_width=True)
    with col4:
        st.subheader("ML: Ensemble Models & Explainability")
        st.info("Ensemble models like **Random Forest** capture complex non-linear relationships. We then use explainers like **SHAP** to understand the 'black box' model.")
        with st.expander("How They Work"):
            st.markdown("- **Random Forest:** Builds hundreds of de-correlated decision trees and averages their predictions for high accuracy and stability.\n- **SHAP:** Uses game theory to compute the precise contribution of each feature to each prediction.")
        shap_vals = pd.DataFrame({'Feature': X_reg.columns, 'Importance': np.abs(model.feature_importances_ * 10)}).sort_values('Importance', ascending=True)
        fig_shap = go.Figure(go.Bar(x=shap_vals['Importance'], y=shap_vals['Feature'], orientation='h', marker_color=COLORS['secondary']))
        fig_shap.update_layout(title_text="<b>ML Root Cause:</b> Feature Importance", xaxis_title="Average impact on model output", plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig_shap, use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy:**
    1.  **Compare Groups with ANOVA:** For simple comparisons between groups, ANOVA is the standard and correct first step. If its assumptions are violated, use a non-parametric alternative like a permutation test.
    2.  **Model Relationships with Both:** Fit both a Linear Regression (for a simple baseline) and an ensemble ML model (for accuracy).
    3.  **Trust the Best Model:** If the ML model is significantly more accurate (higher R²), its feature importance rankings from SHAP are a more reliable guide to the true root causes than the coefficients from the poorly-fitting linear model.
    """)


# ==============================================================================
# PAGE 4: IMPROVE PHASE
# ==============================================================================
def show_improve_phase():
    st.title("⚙️ Improve Phase: Discovering Optimal Solutions")
    st.markdown("""
    **Objective:** To identify, test, and implement solutions that address the root causes. This involves finding the optimal settings for our critical process inputs (Xs) and proactively mitigating risks.
    """)
    st.markdown("---")
    
    st.header("1. Finding Optimal Process Settings")
    tab1, tab2 = st.tabs(["Classical: Design of Experiments (DOE)", "ML: Bayesian Optimization"])
    with tab1:
        st.subheader("Classical: Design of Experiments (DOE)")
        st.info("""
        **What is it?** A structured statistical method for efficiently changing multiple inputs to determine their individual (main) and combined (interaction) effects on the output.
        - **Strength:** The gold standard for physical experimentation. Statistically rigorous and highly informative.
        - **Caveat:** Suffers from the curse of dimensionality, making it impractical for >7 factors.
        """)
        doe_data = generate_doe_data()
        fig_doe = plot_doe_cube(doe_data)
        st.plotly_chart(fig_doe, use_container_width=True)
    with tab2:
        st.subheader("ML: Bayesian Optimization")
        st.info("""
        **What is it?** An intelligent search algorithm for finding the global optimum of an expensive-to-evaluate function (e.g., a 3-day simulation or a costly physical experiment).
        - **Strength:** Extremely sample-efficient, especially in high-dimensional spaces. It intelligently balances exploiting good solutions and exploring new ones.
        - **Caveat:** Can be sensitive to initial parameters and may struggle with very 'spiky' functions.
        """)
        with st.expander("Technical Deep Dive: Surrogate Model & Acquisition Function"):
            st.markdown("""
            1.  **Surrogate Model (The Brain):** A cheap probabilistic model (typically a Gaussian Process) that approximates the true function, providing a mean prediction (blue line) and an uncertainty estimate (blue shading).
            2.  **Acquisition Function (The Strategy):** A function (e.g., Upper Confidence Bound, UCB) that uses the surrogate's output to decide where to sample next. It finds a balance between high-predicted-mean areas (exploitation) and high-uncertainty areas (exploration).
            """)
        st.sidebar.header("Bayesian Opt. Simulator")
        @st.cache_data 
        def true_func(x): return (np.sin(x * 0.8) * 15) + (np.cos(x * 2.5)) * 5 - (x/10)**3
        x_range = np.linspace(0, 20, 200)
        if 'sampled_points' not in st.session_state: st.session_state.sampled_points = {'x': [2.0, 15.0], 'y': [true_func(2.0), true_func(15.0)]}
        if st.sidebar.button("Sample Next Best Point"): 
            _, next_point = plot_bayesian_optimization_interactive(true_func, x_range, st.session_state.sampled_points)
            st.session_state.sampled_points['x'].append(next_point)
            st.session_state.sampled_points['y'].append(true_func(next_point))
        if st.sidebar.button("Reset Simulation"): st.session_state.sampled_points = {'x': [2.0, 15.0], 'y': [true_func(2.0), true_func(15.0)]}
        fig_bo, _ = plot_bayesian_optimization_interactive(true_func, x_range, st.session_state.sampled_points)
        st.plotly_chart(fig_bo, use_container_width=True)

    st.header("2. Proactively Mitigating Risks")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Classical: FMEA")
        st.info("""
        **What is it?** Failure Mode and Effects Analysis is a structured, team-based risk assessment tool. It involves brainstorming potential failure modes, their effects, and their causes, then ranking them by a Risk Priority Number (RPN = Severity × Occurrence × Detection).
        - **Strength:** A powerful, systematic way to force a team to think about what *could* go wrong.
        - **Caveat:** The RPN scores are subjective, qualitative, and based on team consensus, not always on hard data.
        """)
    with col4:
        st.subheader("ML: Prognostics & Health Management (PHM)")
        st.info("""
        **What is it?** A data-driven approach where ML models are trained on sensor data (vibration, temperature, etc.) to predict equipment degradation and estimate its **Remaining Useful Life (RUL)** before a failure occurs.
        - **Strength:** Moves risk management from a qualitative exercise to a quantitative, predictive capability.
        - **Caveat:** Requires high-quality, high-frequency sensor data, including run-to-failure examples, which can be difficult to obtain.
        """)
        st.plotly_chart(plot_rul_prediction(generate_sensor_degradation_data()), use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy:**
    1.  **Mitigate with FMEA, Predict with PHM:** Use a classical FMEA to identify the highest-risk failure modes. For those top risks, investigate if sensor data is available to build a PHM/RUL model to create a data-driven, predictive control.
    2.  **Optimize with the Right Tool:** For physical processes with few variables, use **DOE**. For complex digital systems or simulations, use **Bayesian Optimization**.
    3.  **The Ultimate Hybrid ("Digital Twin"):** Use DOE data to train a highly accurate ML model of your process (a "digital twin"). Then, use Bayesian Optimization on this digital twin to find the global optimum virtually before a final confirmation run.
    """)

# ==============================================================================
# PAGE 5: CONTROL PHASE
# ==============================================================================
def show_control_phase():
    st.title("📡 Control Phase: Sustaining and Monitoring Gains")
    st.markdown("""
    **Objective:** To implement a robust system to monitor the improved process, ensuring it remains stable and that improvements are sustained. The goal is to create a control plan and move from reactive problem-solving to proactive process management.
    """)
    st.markdown("---")

    st.header("1. Detecting Process Shifts")
    st.markdown("Control charts are the primary tool for monitoring process stability over time.")

    st.sidebar.header("Control Simulator")
    st.sidebar.markdown("Introduce a small, sustained shift in the process.")
    shift_mag = st.sidebar.slider("Magnitude of Shift (in Std Devs)", 0.2, 2.0, 0.8, 0.1, key="ctrl_shift_mag")
    ewma_lambda = st.sidebar.slider("EWMA Lambda (λ)", 0.1, 0.5, 0.2, 0.05, help="Higher λ reacts faster but is more sensitive to noise.")
    
    chart_data = generate_control_chart_data(shift_point=75, shift_magnitude=shift_mag)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classical: Shewhart Chart (X-bar)")
        st.info("""
        **What is it?** The standard SPC chart. It plots individual or subgroup means and detects large shifts effectively. Each point is treated independently.
        - **Strength:** Simple, robust, and excellent for detecting large (>1.5σ) shifts.
        - **Limitation:** Slow to detect small, sustained shifts, as it has no 'memory' of past data points.
        """)
        # ***** CORRECTED LINE IS HERE *****
        fig_spc = plot_control_chart_pro(chart_data)
        st.plotly_chart(fig_spc, use_container_width=True)

    with col2:
        st.subheader("Advanced Classical: EWMA Chart")
        st.info("""
        **What is it?** An Exponentially Weighted Moving Average chart gives more weight to recent data points and less to older ones. It has memory, making it highly effective at detecting small, sustained shifts.
        - **Strength:** Significantly more sensitive to small shifts than a Shewhart chart.
        - **Limitation:** The choice of the weighting factor λ is a trade-off between sensitivity and false alarms.
        """)
        st.plotly_chart(plot_ewma_chart(chart_data, lambda_val=ewma_lambda), use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy:**
    1.  **Monitor CTQs with SPC:** Keep a classical Shewhart chart on your final Critical-to-Quality (CTQ) output for simple, robust monitoring.
    2.  **Monitor Critical Inputs with Advanced SPC/ML:** For the critical input variables (Xs) that drive your process, use more sensitive charts like **EWMA** or **CUSUM**.
    3.  **Create an Early Warning System with ML:** For the most complex, multivariate interactions, deploy an ML anomaly detection model (like an LSTM or Autoencoder) that learns the normal 'heartbeat' of your process across all sensor inputs. This model doesn't just detect shifts; it predicts them, enabling true proactive control.
    """)


# ==============================================================================
# PAGE 6: COMPARISON MATRIX
# ==============================================================================
def show_comparison_matrix():
    st.title("⚔️ Head-to-Head Comparison: Classical Stats vs. Machine Learning")
    st.markdown("A summary of the core philosophical and practical differences between the two approaches.")
    st.subheader("Attribute Comparison Matrix")
    comparison_data = pd.DataFrame({
        "Dimension": ["Primary Goal", "Data Requirements", "Assumptions", "Interpretability", "Scalability", "Causality", "Implementation Cost", "Auditability & Compliance", "Detection Method"],
        "Classical Stats": ["Inference & Hypothesis Testing", "Low (designed for small, clean samples)", "Many & Strict (normality, independence, etc.)", "High (coefficients have direct meaning)", "Poor (struggles with >5-7 variables)", "Inferred via designed experiments (DOE)", "Low (Excel, Minitab, established knowledge)", "High (standardized, validated methods)", "Reactive (detects shifts after they occur)"],
        "Machine Learning": ["Prediction & Pattern Recognition", "High (performance scales with data volume)", "Fewer & Flexible (non-parametric)", "Requires post-hoc tools (SHAP, LIME)", "Excellent (designed for high dimensionality)", "Can be inferred from observational data (Causal ML)", "Higher (Python, cloud infra, specialized skills)", "Lower (models can be complex to validate)", "Proactive (can predict shifts before they occur)"]
    })
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)

    st.subheader("🏁 The Verdict: Which is Better?")
    verdict_data = pd.DataFrame({
        "Metric": ["Interpretability & Simplicity", "Predictive Accuracy in Complex Systems", "Handling High-Dimensional Data", "Rigor in Small-Data, Physical Experiments", "Proactive & Early Warning Capability", "Regulatory Acceptance & Auditability"],
        "🏆 Winner": ["Classical Stats", "Machine Learning", "Machine Learning", "Classical Stats", "Machine Learning", "Classical Stats"],
        "Rationale": ["Models and outputs are simple, transparent, and defensible without extra tools.", "Natively captures non-linearities and complex interactions that classical models miss.", "Algorithms are designed to thrive in 'wide data' environments (many variables).", "DOE provides a statistically rigorous framework for establishing causality with minimal runs.", "Models are trained to recognize precursors to failure, enabling proactive control.", "Methods are standardized, well-documented, and accepted by regulatory bodies (e.g., FDA)."]
    })
    st.dataframe(verdict_data, use_container_width=True, hide_index=True)

# ==============================================================================
# PAGE 7: HYBRID STRATEGY
# ==============================================================================
def show_hybrid_strategy():
    st.title("🧠 The Hybrid Strategy: The Future of Quality")
    st.markdown("The most competitive organizations do not choose one over the other; they build an **AI-Augmented Six Sigma** program that fuses statistical rigor with machine learning's predictive power.")
    st.subheader("Scenario-Based Recommendations")
    guidance_data = pd.DataFrame({
        "Scenario": ["Validating a change for FDA/FAA compliance", "Monitoring a semiconductor fab with 1000s of sensors", "Understanding why customers are churning by analyzing support emails", "Optimizing a simple, 3-factor physical mixing process", "Building a 'digital twin' of a chemical reactor", "Providing real-time operator guidance"],
        "Recommended Approach": ["**Classical Stats** (Hypothesis Testing, DOE)", "**ML + SPC** (Multivariate Anomaly Detection)", "**ML NLP** (Topic Modeling & Sentiment Analysis)", "**Classical DOE**", "**Hybrid:** ML Model + Bayesian Opt.", "**ML** (Real-time Predictive Model)"],
        "Why?": ["Methods are traceable, validated, and legally defensible.", "Detects subtle, multivariate sensor drifts that individual SPC charts would miss.", "Processes and extracts actionable themes from massive, unstructured text data.", "Simple, highly effective, and provides clear, interpretable results with minimal setup.", "ML builds the accurate simulation; Bayesian Opt. finds the peak efficiency in the vast parameter space.", "Predicts the outcome of the current settings and suggests optimal adjustments to the operator."]
    })
    st.dataframe(guidance_data, use_container_width=True, hide_index=True)
    st.header("A Unified, Modern DMAIC Workflow")
    st.image("https://i.imgur.com/rS2Mtn1.png", caption="An integrated workflow where classical and ML tools support each other at every stage.")
    st.markdown("""
    #### 1. **Define: Scope with Clarity, Validate with Data**
       - **Classical:** Use SIPOC to align stakeholders and define project boundaries. Use the Kano Model to prioritize features.
       - **ML:** Use NLP on customer feedback to provide a data-driven business case and quantify the problem's impact. Use Causal Discovery to form initial hypotheses.
    #### 2. **Measure: Baseline with Rigor, Diagnose with Insight**
       - **Classical:** Use Gage R&R to validate the measurement system, then calculate Cpk to establish the official process baseline.
       - **ML:** Use Process Mining to discover the true process flow and KDE plots to diagnose the *nature* of poor capability.
    #### 3. **Analyze: Hypothesize with Experience, Discover with ML**
       - **Classical:** Use Fishbone diagrams and ANOVA to test hypotheses about known potential root causes.
       - **ML:** Use a powerful model (e.g., XGBoost) and SHAP explainability on historical data to identify and rank the most impactful, often non-obvious, process drivers.
    #### 4. **Improve: Experiment Physically with DOE, Proactively with FMEA/PHM**
       - **Classical:** Use FMEA to brainstorm risks. Use DOE on the vital few factors identified by SHAP to efficiently optimize physical processes.
       - **ML:** Use PHM/RUL models to predict failures. Use Bayesian Optimization to optimize complex digital systems or simulations ("digital twins").
    #### 5. **Control: Monitor Outputs with SPC, Predict with ML**
       - **Classical:** Use advanced SPC charts (EWMA/CUSUM) on the final CTQ output for robust, sensitive monitoring.
       - **ML:** Use multivariate anomaly detection on process inputs (Xs) as an early warning system to predict and prevent defects before they happen.
    """)
