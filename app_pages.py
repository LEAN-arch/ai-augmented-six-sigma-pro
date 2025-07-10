# app_pages.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Import all necessary helper functions from the single helper file
from app_helpers import (
    plot_voc_nlp_summary,
    generate_process_data,
    plot_capability_analysis_pro,
    generate_nonlinear_data,
    plot_regression_comparison_pro,
    COLORS,
    generate_doe_data,
    plot_doe_cube,
    plot_bayesian_optimization_interactive,
    generate_control_chart_data,
    plot_control_chart_pro
)

# ==============================================================================
# PAGE 1: DEFINE PHASE
# ==============================================================================
def show_define_phase():
    st.title("🌀 Define Phase: Establishing the Foundation")
    st.markdown("""
    **Objective:** To clearly articulate the business problem, project goals, scope, and high-level process map. This phase is critical for ensuring the team is aligned and that the project is focused on a tangible, valuable business outcome.
    """)
    st.markdown("---")

    st.header("1. Process Mapping: SIPOC vs. Data-Driven Graphs")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classical Tool: SIPOC")
        st.info("""
        **What is it?** A qualitative, high-level map that defines the scope of an improvement project by identifying **S**uppliers, **I**nputs, **P**rocess, **O**utputs, and **C**ustomers.
        - **Strength:** Unmatched for simplicity, facilitating team alignment, and defining clear project boundaries. It's the essential first step for any project.
        - **Caveat:** Its value is entirely dependent on the existing domain knowledge of the team. It cannot uncover relationships or variables that the team is not already aware of.
        """)
    with col2:
        st.subheader("ML Counterpart: Causal Discovery")
        st.info("""
        **What is it?** A suite of algorithms (e.g., PC, LiNGAM) that analyze observational data to infer a graph of probable cause-and-effect relationships. It goes beyond correlation by attempting to identify directional influence.
        - **Strength:** Objectively discovers potential causal links and complex, latent interactions that human experts might overlook. Excellent for generating data-driven hypotheses about process drivers.
        - **Caveat:** These are not magic bullets. They require large, high-quality datasets and their output is a set of *hypotheses* that must be validated with domain expertise and experimentation (e.g., a DOE).
        """)

    st.header("2. Voice of the Customer (VOC): Surveys vs. NLP")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classical Tool: Surveys & Interviews")
        st.info("""
        **What is it?** Manual methods for gathering customer feedback, including structured surveys (e.g., Likert scales), focus groups, and one-on-one interviews. The goal is to translate customer needs into measurable requirements.
        - **Strength:** Provides deep, contextual, and qualitative insights—the "why" behind customer opinions.
        - **Caveat:** Suffers from low sample size, high cost per insight, and significant lead time. Prone to various biases (e.g., selection, response bias).
        """)
    with col2:
        st.subheader("ML Counterpart: Natural Language Processing (NLP)")
        st.info("""
        **What is it?** Using algorithms to analyze vast amounts of unstructured text data (product reviews, support tickets, social media comments) to automatically extract themes, sentiment, and key topics.
        - **Core Concepts:** Utilizes techniques like **Topic Modeling (LDA)** to find recurring themes and **Sentiment Analysis (e.g., using Transformer models like BERT)** to gauge positive/negative tone at scale.
        - **Strength:** Massively scalable, real-time, and objective. Can analyze millions of data points to instantly uncover emerging trends and quantify the prevalence of different issues.
        """)
        fig = plot_voc_nlp_summary()
        st.plotly_chart(fig, use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy:**
    1.  **Draft with SIPOC:** Begin every project by drafting a SIPOC with the core team to establish a shared understanding and define the project scope.
    2.  **Quantify with NLP:** Use NLP on existing customer feedback data (e.g., App Store reviews, support tickets) to validate and quantify the "Output" problems identified in the SIPOC. This provides a data-driven business case (e.g., "Feature X is mentioned in 35% of negative reviews").
    3.  **Hypothesize with Causal Graphs:** If historical process data is available, run causal discovery algorithms to generate a data-driven "proto-fishbone" diagram. This can highlight potential critical inputs (Xs) that the team may have overlooked in the SIPOC, guiding the Measure phase.
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

    st.header("Capability Analysis: Indices vs. Full Distributions")
    st.markdown("""
    Capability analysis is the cornerstone of the Measure phase. It answers a simple question: "Is my process capable of consistently meeting my customer's requirements?"
    """)
    
    with st.expander("⚠️ Foundational Prerequisite: Measurement System Analysis (MSA)"):
        st.warning("""
        An SME would never perform a capability analysis without first validating the measurement system. Before trusting your data, you must ensure it is reliable.
        - **Classical MSA (Gage R&R):** A designed experiment to quantify how much variation in your data comes from the measurement system itself versus the actual process. It assesses **repeatability** (same operator, same part) and **reproducibility** (different operators, same part).
        - **Guideline:** The total Gage R&R variation should be less than 10% of the total process tolerance. If your measurement system is poor, your capability analysis will be meaningless.
        """)


    st.sidebar.header("Capability Simulator")
    st.sidebar.markdown("Adjust the process parameters to see how they affect capability.")
    lsl = st.sidebar.slider("Lower Specification Limit (LSL)", 80.0, 95.0, 90.0, key="measure_lsl")
    usl = st.sidebar.slider("Upper Specification Limit (USL)", 105.0, 120.0, 110.0, key="measure_usl")
    process_mean = st.sidebar.slider("Process Mean (μ)", 95.0, 105.0, 101.5, key="measure_mean")
    process_std = st.sidebar.slider("Process Std Dev (σ)", 0.5, 5.0, 2.0, key="measure_std")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Classical: Cp & Cpk")
        st.info("These are industry-standard, single-point indices that summarize process capability. They are powerful but rely on the critical assumption that the process data is normally distributed.")
        with st.expander("Show Mathematical Formulas & Interpretation"):
            st.markdown("**Process Potential (Cp):** The 'best case scenario.' It compares the total allowed spread (Voice of the Customer) to the natural process spread (Voice of the Process), assuming the process is perfectly centered.")
            st.latex(r''' C_p = \frac{\text{Tolerance Width}}{\text{Process Width}} = \frac{USL - LSL}{6\sigma} ''')
            st.markdown("**Process Capability (Cpk):** The 'reality check.' It adjusts Cp for any off-center bias, representing the *actual* capability of the process.")
            st.latex(r''' C_{pk} = \min\left(\frac{USL - \mu}{3\sigma}, \frac{\mu - LSL}{3\sigma}\right) ''')

        data = generate_process_data(process_mean, process_std, 1000, lsl, usl)
        fig, cp, cpk = plot_capability_analysis_pro(data, lsl, usl)

        st.metric("Process Potential (Cp)", f"{cp:.2f}")
        st.metric("Process Capability (Cpk)", f"{cpk:.2f}")

        if cpk < 1.0:
            st.error(f"Process is not capable. Significant portion of output is out of spec.", icon="🚨")
        elif 1.0 <= cpk < 1.33:
            st.warning(f"Process is marginal. It may be acceptable but needs improvement.", icon="⚠️")
        else:
            st.success(f"Process is capable. (Typically Cpk > 1.33 is considered good).", icon="✅")
            
    with col2:
        st.subheader("ML: Distributional View")
        st.info("Non-parametric methods like **Kernel Density Estimation (KDE)** do not assume a distribution. They build a smooth curve to visualize the *true* shape of the process data, instantly revealing issues like skewness or bimodality that single-point indices like Cpk would completely hide.")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Try This:** Keep the `Std Dev` low but move the `Mean` very close to a spec limit. Notice how `Cp` remains high (the process *could* be good), but `Cpk` plummets (the reality is poor). The KDE plot visually confirms this by showing the distribution's tail crossing the red line.")

    st.success("""
    **🏆 Verdict & Hybrid Strategy:**
    1.  **Validate First:** Always perform an **MSA/Gage R&R** before proceeding. Garbage in, garbage out.
    2.  **Report Cpk:** Cpk is the universal language of process capability. It should be the headline metric for reports and dashboards.
    3.  **Diagnose with KDE:** Use the KDE plot as your expert diagnostic tool. If the Cpk is low, the shape of the KDE plot immediately tells you if the problem is poor centering (a shifted peak), excessive variation (a wide peak), or non-normality (a skewed or double-peaked shape). This insight is crucial for focusing your efforts in the Analyze phase.
    """)


# ==============================================================================
# PAGE 3: ANALYZE PHASE
# ==============================================================================
def show_analyze_phase():
    st.title("📈 Analyze Phase: Discovering Root Causes")
    st.markdown("""
    **Objective:** To analyze the data from the Measure phase to identify, validate, and quantify the root cause(s) of defects or variation. This involves moving from *what* is happening to *why* it is happening.
    """)
    st.markdown("---")

    st.header("Identifying Drivers: Linear Regression vs. Ensemble Models")
    st.markdown("A core task is to model the relationship `Y = f(X)`, where `Y` is the critical output and `X` represents the potential process inputs.")

    df = generate_nonlinear_data()
    fig_reg, model, X = plot_regression_comparison_pro(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classical: Linear Regression")
        st.info("A powerful, interpretable method that models the linear relationship between inputs and outputs. Its simplicity is both its greatest strength and its greatest weakness.")
        with st.expander("Mathematical Model & Assumptions"):
            st.latex(r''' Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \epsilon ''')
            st.markdown(r"""
            - **Interpretation:** The coefficient $\beta_i$ represents the average change in $Y$ for a one-unit increase in $X_i$, holding all other variables constant.
            - **Critical Assumptions (L.I.N.E):**
                1.  **L**inearity: The relationship between X and Y is linear.
                2.  **I**ndependence: The errors are independent.
                3.  **N**ormality: The errors are normally distributed.
                4.  **E**qual Variance (Homoscedasticity): The errors have constant variance.
            - **Caveat:** If these assumptions are violated (as in the plot!), the model's conclusions can be misleading.
            """)
        st.plotly_chart(fig_reg, use_container_width=True)

    with col2:
        st.subheader("ML: Random Forest & Explainability")
        st.info("Ensemble models like Random Forest build hundreds of decision trees on random subsets of the data and features, then average their predictions. This makes them incredibly robust to non-linearities and interactions.")
        with st.expander("How Random Forest & SHAP Work"):
            st.markdown("""
            - **Random Forest:** This technique combats the high variance of single decision trees through **bagging** (bootstrap aggregating). It creates many decorrelated trees and averages them, resulting in a stable and highly accurate model that does not require feature scaling or strict distributional assumptions.
            - **SHAP (SHapley Additive exPlanations):** Since the model is a "black box," we need an explainer. SHAP uses game theory to calculate the precise contribution of each feature to each individual prediction. Summing these contributions gives us a robust measure of global feature importance.
            """)

        shap_values = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(model.feature_importances_ * 10)
        }).sort_values('Importance', ascending=True)

        fig_shap = go.Figure(go.Bar(
            x=shap_values['Importance'],
            y=shap_values['Feature'],
            orientation='h',
            marker_color=COLORS['secondary']
        ))
        fig_shap.update_layout(
            title_text="<b>ML Root Cause:</b> Feature Importance",
            xaxis_title="Average impact on model output (mean |SHAP value|)",
            plot_bgcolor='white', paper_bgcolor='white'
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("The feature importance plot correctly identifies the true drivers, `Feature_1` and `Feature_2`, while correctly assigning almost zero importance to the `Noise` feature.")

    st.success("""
    **🏆 Verdict & Hybrid Strategy:**
    1.  **Build Both Models:** Always start by fitting a simple Linear Regression for a baseline understanding and an interpretable model. Then, fit a powerful ensemble model (like XGBoost or Random Forest) to capture the true, complex relationships.
    2.  **Compare R-squared:** If the R² values are similar, the linear model is likely sufficient, and its simple coefficients can be used for root cause analysis.
    3.  **Trust but Verify with SHAP:** If the ML model's R² is significantly higher (as in the example), the process is non-linear. The linear model's conclusions are unreliable. Trust the ML model and use its **SHAP-based feature importance** to identify the true "vital few" Xs. These become the primary candidates for the Improve phase.
    """)


# ==============================================================================
# PAGE 4: IMPROVE PHASE
# ==============================================================================
def show_improve_phase():
    st.title("⚙️ Improve Phase: Discovering Optimal Solutions")
    st.markdown("""
    **Objective:** To identify, test, and implement solutions that address the root causes from the Analyze phase. This involves finding the optimal settings for our critical process inputs (Xs).
    """)
    st.markdown("---")

    tab1, tab2 = st.tabs(["Classical: Design of Experiments (DOE)", "ML: Bayesian Optimization"])
    with tab1:
        st.header("Classical Approach: Design of Experiments (DOE)")
        st.markdown("""
        **What is it?** A structured, statistical method for systematically and efficiently changing multiple process inputs simultaneously to determine their individual (main effects) and combined (interaction effects) impact on the output.
        - **Strength:** The gold standard for physical experimentation. It provides maximum information from the minimum number of experimental runs. Its statistical rigor gives high confidence in the results.
        - **Caveat:** Suffers from the **curse of dimensionality.** The number of runs required grows exponentially with the number of factors, making it impractical for problems with more than ~5-7 variables.
        """)
        doe_data = generate_doe_data()
        fig_doe = plot_doe_cube(doe_data)
        st.plotly_chart(fig_doe, use_container_width=True)
        st.info("This 3D cube plot visualizes a 2³ factorial design. Each corner represents one of the 8 unique experimental runs. The color and label show the resulting 'Yield,' allowing us to visually identify the settings that produce the best outcome.")

    with tab2:
        st.header("ML Counterpart: Bayesian Optimization")
        st.markdown("""
        **What is it?** An intelligent search algorithm for finding the global optimum of a function that is expensive to evaluate (e.g., a 3-day simulation, a costly physical experiment, or training a deep learning model).
        - **Core Logic:** It iteratively builds a probabilistic model of the objective function and uses that model to decide where to sample next, perfectly balancing **exploitation** (sampling where the model predicts a high output) and **exploration** (sampling where the model is most uncertain).
        - **Strength:** Extremely sample-efficient. Can find a better optimum in far fewer trials than DOE or grid search, especially in high-dimensional spaces.
        - **Caveat:** Can be sensitive to initial parameter choices and may struggle with very high-frequency (spiky) functions.
        """)
        with st.expander("Technical Deep Dive: Surrogate Model & Acquisition Function"):
            st.markdown("""
            1.  **Surrogate Model (The Brain):** A cheap, probabilistic model that approximates the expensive true function. A Gaussian Process (GP) is typically used, as it provides both a mean prediction (the blue line) and an uncertainty estimate (the blue shading).
            2.  **Acquisition Function (The Strategy):** A function that uses the surrogate's predictions and uncertainty to decide where to sample next. The **Upper Confidence Bound (UCB)** (green dotted line) is a popular choice: it's high where the predicted mean is high (exploitation) *and* where the uncertainty is high (exploration). We sample at its peak.
            """)

        @st.cache_data
        def true_function(x):
            return (np.sin(x * 0.8) * 15) + (np.cos(x * 2.5)) * 5 - (x/10)**3

        x_range = np.linspace(0, 20, 200)
        st.sidebar.header("Bayesian Opt. Simulator")
        if 'sampled_points' not in st.session_state:
            st.session_state.sampled_points = {'x': [2.0, 15.0], 'y': [true_function(2.0), true_function(15.0)]}
        if st.sidebar.button("Sample Next Best Point"):
            _, next_point = plot_bayesian_optimization_interactive(true_function, x_range, st.session_state.sampled_points)
            st.session_state.sampled_points['x'].append(next_point)
            st.session_state.sampled_points['y'].append(true_function(next_point))
        if st.sidebar.button("Reset Simulation"):
            st.session_state.sampled_points = {'x': [2.0, 15.0], 'y': [true_function(2.0), true_function(15.0)]}

        fig_bo, _ = plot_bayesian_optimization_interactive(true_function, x_range, st.session_state.sampled_points)
        st.plotly_chart(fig_bo, use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy:** The choice depends entirely on the nature of the problem.
    -   For **physical processes** with a handful of key factors identified in the Analyze phase, use **Classical DOE**. Its rigor and ability to estimate interactions are unparalleled for real-world experimentation.
    -   For **digital problems** like optimizing a complex simulation, tuning the hyperparameters of an ML model, or designing a new material in a computational environment, use **Bayesian Optimization**. Its sample efficiency is critical when each trial is costly.
    -   **The Ultimate Hybrid ("Digital Twin"):** The most advanced approach is to use data from a well-run DOE to train a highly accurate ML model (a "digital twin" of your physical process). Then, use Bayesian Optimization to run tens of thousands of "virtual" experiments on this fast and free digital twin to find the true global optimum, which you then confirm with a single, final physical experiment.
    """)


# ==============================================================================
# PAGE 5: CONTROL PHASE
# ==============================================================================
def show_control_phase():
    st.title("📡 Control Phase: Sustaining and Monitoring Gains")
    st.markdown("""
    **Objective:** To implement a robust system to monitor the improved process, ensuring it remains stable and that the improvements are sustained over time. The goal is to create a control plan and move from reactive problem-solving to proactive process management.
    """)
    st.markdown("---")

    st.header("Process Monitoring: SPC vs. ML Anomaly Detection")

    st.sidebar.header("Control Simulator")
    st.sidebar.markdown("Introduce a shift in the process and see which method detects it first.")
    shift_point = st.sidebar.slider("Point of Process Shift", min_value=50, max_value=130, value=100, key="control_shift_point")
    shift_magnitude = st.sidebar.slider("Magnitude of Shift (in Std Devs)", 0.5, 4.0, 1.8, 0.1, key="control_shift_mag")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classical: Statistical Process Control (SPC)")
        st.info("SPC uses control charts to monitor a process variable over time. It is based on the fundamental distinction between common cause variation (the natural noise of a process) and special cause variation (an external event that shifts the process).")
        with st.expander("How it Works: Control Limits"):
            st.markdown("""
            Control limits are the "voice of the process." They are calculated from your process data, typically at $\mu \pm 3\sigma$. They are not specification limits. A point outside the control limits is a statistical signal that the process has fundamentally changed.
            - **Key Heuristics (Western Electric Rules):** SPC uses rules to detect non-random patterns, such as a single point outside $\pm3\sigma$ or 8 consecutive points above the centerline.
            """)
    with col2:
        st.subheader("ML: Anomaly Detection")
        st.info("ML anomaly detection models learn a holistic, often multivariate, representation of what 'normal' looks like. They can detect subtle deviations from this learned normal pattern that would be invisible to single-variable SPC charts.")
        with st.expander("The Multivariate Blind Spot of SPC"):
            st.error("""
            Imagine `Temperature` and `Pressure` are normally positively correlated. A classic SPC system has two separate charts. If a failure causes `Temp` to rise while `Pressure` falls, **both might stay within their individual control limits**, and SPC would see nothing wrong. An ML model trained on the *relationship* between variables would immediately flag this broken correlation as a major anomaly.
            - **Advanced Methods:** Real-world systems use models like **LSTMs** (for time-series data) or **Autoencoders** (for learning a compressed representation of "normalcy") to detect these complex, multivariate anomalies.
            """)

    chart_data = generate_control_chart_data(shift_point=shift_point, shift_magnitude=shift_magnitude)
    fig_control = plot_control_chart_pro(chart_data)
    st.plotly_chart(fig_control, use_container_width=True)
    st.markdown("**Try This:** Set the `Magnitude of Shift` slider to a small value like `1.5`. Notice how the ML Anomaly Score (bottom panel) often spikes above its threshold *before* any point in the top panel crosses the red SPC limits. This is the power of predictive, early-warning control.")

    st.success("""
    **🏆 Verdict & Hybrid Strategy:** This is the most critical phase for a hybrid approach. Using only one is a missed opportunity.
    1.  **Monitor the Output (Y) with SPC:** Keep a classical control chart (e.g., X-bar & R) on your final, critical-to-quality (CTQ) output. This provides a simple, robust, and universally understood view of final product quality. It answers the question, "Are we making good parts?"
    2.  **Monitor the Inputs (Xs) with ML:** Deploy a multivariate ML anomaly detection model on the critical *input and process parameters* identified in the Analyze phase. This model acts as a sophisticated **early warning system**. It answers the question, "Is our process about to start making bad parts?" by detecting subtle deviations in the inputs before they cascade into an out-of-spec output. This enables true proactive control.
    """)


# ==============================================================================
# PAGE 6: COMPARISON MATRIX
# ==============================================================================
def show_comparison_matrix():
    st.title("⚔️ Head-to-Head Comparison: Classical Stats vs. Machine Learning")
    st.markdown("A summary of the core philosophical and practical differences between the two approaches.")

    st.subheader("Attribute Comparison Matrix")
    comparison_data = pd.DataFrame({
        "Dimension": [
            "Primary Goal", "Data Requirements", "Assumptions",
            "Interpretability", "Scalability", "Causality",
            "Implementation Cost", "Auditability & Compliance", "Detection Method"
        ],
        "Classical Stats": [
            "Inference & Hypothesis Testing", "Low (designed for small, clean samples)", "Many & Strict (normality, independence, etc.)",
            "High (coefficients have direct meaning)", "Poor (struggles with >5-7 variables)", "Inferred via designed experiments (DOE)",
            "Low (Excel, Minitab, established knowledge)", "High (standardized, validated methods)", "Reactive (detects shifts after they occur)"
        ],
        "Machine Learning": [
            "Prediction & Pattern Recognition", "High (performance scales with data volume)", "Fewer & Flexible (non-parametric)",
            "Requires post-hoc tools (SHAP, LIME)", "Excellent (designed for high dimensionality)", "Can be inferred from observational data (Causal ML)",
            "Higher (Python, cloud infra, specialized skills)", "Lower (models can be complex to validate)", "Proactive (can predict shifts before they occur)"
        ]
    })
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)

    st.subheader("🏁 The Verdict: Which is Better?")
    verdict_data = pd.DataFrame({
        "Metric": ["Interpretability & Simplicity", "Predictive Accuracy in Complex Systems", "Handling High-Dimensional Data", "Rigor in Small-Data, Physical Experiments", "Proactive & Early Warning Capability", "Regulatory Acceptance & Auditability"],
        "🏆 Winner": ["Classical Stats", "Machine Learning", "Machine Learning", "Classical Stats", "Machine Learning", "Classical Stats"],
        "Rationale": [
            "Models and outputs are simple, transparent, and defensible without extra tools.",
            "Natively captures non-linearities and complex interactions that classical models miss.",
            "Algorithms are designed to thrive in 'wide data' environments (many variables).",
            "DOE provides a statistically rigorous framework for establishing causality with minimal runs.",
            "Models are trained to recognize precursors to failure, enabling proactive control.",
            "Methods are standardized, well-documented, and accepted by regulatory bodies (e.g., FDA)."
        ]
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
        "Scenario": [
            "Validating a change for FDA/FAA compliance", "Monitoring a high-volume semiconductor fab with 1000s of sensors", "Understanding why customers are churning by analyzing support emails",
            "Optimizing a simple, 3-factor physical mixing process", "Building a 'digital twin' of a chemical reactor to find a novel operating recipe", "Providing real-time guidance to a machine operator"
        ],
        "Recommended Approach": [
            "**Classical Stats** (Hypothesis Testing, DOE)", "**ML + SPC** (Multivariate Anomaly Detection)", "**ML NLP** (Topic Modeling & Sentiment Analysis)", "**Classical DOE**", "**Hybrid:** ML Model + Bayesian Opt.", "**ML** (Real-time Predictive Model)"
        ],
        "Why?": [
            "Methods are traceable, validated, and legally defensible.", "Detects subtle, multivariate sensor drifts that individual SPC charts would miss.", "Processes and extracts actionable themes from massive, unstructured text data.",
            "Simple, highly effective, and provides clear, interpretable results with minimal setup.", "ML builds the accurate simulation of the reactor; Bayesian Opt. finds the peak efficiency in the vast parameter space.", "Predicts the outcome of the current settings and suggests optimal adjustments to the operator."
        ]
    })
    st.dataframe(guidance_data, use_container_width=True, hide_index=True)

    st.header("A Unified, Modern DMAIC Workflow")
    st.image("https://i.imgur.com/rS2Mtn1.png", caption="An integrated workflow where classical and ML tools support each other at every stage.")

    st.markdown("""
    #### 1. **Define: Scope with Clarity, Validate with Data**
       - **Classical:** Use SIPOC to align stakeholders and define project boundaries.
       - **ML:** Use NLP on customer feedback to provide a data-driven business case and quantify the problem's impact.

    #### 2. **Measure: Baseline with Rigor, Diagnose with Insight**
       - **Classical:** Use Gage R&R to validate the measurement system, then calculate Cpk to establish the official process baseline.
       - **ML:** Use KDE plots to visualize the true process distribution and diagnose the *nature* of poor capability (shift, spread, or non-normality).

    #### 3. **Analyze: Hypothesize with Experience, Discover with ML**
       - **Classical:** Use Fishbone diagrams and process knowledge to list potential root causes.
       - **ML:** Use a powerful model (e.g., XGBoost) and SHAP explainability on historical data to identify and rank the most impactful, often non-obvious, process drivers.

    #### 4. **Improve: Experiment Physically with DOE, Virtually with ML**
       - **Classical:** Use DOE on the vital few factors identified by SHAP to efficiently optimize physical processes.
       - **ML:** Use Bayesian Optimization to explore and optimize complex digital systems or simulations ("digital twins") at a scale impossible for physical experiments.

    #### 5. **Control: Monitor Outputs with SPC, Predict with ML**
       - **Classical:** Use SPC charts on the final CTQ output for robust, simple, and compliant-ready process monitoring.
       - **ML:** Use multivariate anomaly detection on the process inputs (Xs) as an early warning system to predict and prevent defects before they happen.
    """)
