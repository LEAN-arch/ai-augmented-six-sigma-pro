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
# PAGE 1: DEFINE PHASE (VISUALLY ENHANCED)
# ==============================================================================
def show_define_phase():
    st.title("🌀 Define Phase: Establishing the Project's North Star")
    st.markdown("""
    **Objective:** To clearly articulate the business problem, establish the project's mandate, define the scope, and understand what is truly Critical to Quality (CTQ) for the customer. 
    This phase ensures the team is aligned and focused on a tangible, valuable business outcome. Without a solid definition, a project is likely to fail.
    """)
    st.markdown("---")

    # --- Tool 1: Project Charter ---
    with st.container(border=True):
        st.subheader("1. The Mandate: Project Charter")
        st.markdown("""
        **What is it?** The Project Charter is the foundational document that formally authorizes a project. It serves as a contract between the project team and organizational leadership, outlining the core purpose, scope, and objectives.
        
        - **Strength:** Creates alignment and clarity from the outset. It provides a constant reference point to prevent "scope creep" and ensures everyone agrees on what "success" looks like.
        - **Caveat:** It's a static document. While it shouldn't change frequently, it must be revisited if major discoveries in later phases fundamentally alter the project's direction.
        """)
        # ENHANCED VISUALIZATION
        st.plotly_chart(plot_project_charter_visual(), use_container_width=True)

    # --- Tool 2: SIPOC & Causal Discovery ---
    with st.container(border=True):
        st.subheader("2. The Landscape: Mapping the Process & Hypotheses")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical Tool: SIPOC**")
            st.info("""
            **What is it?** A high-level process map that captures the **S**uppliers, **I**nputs, **P**rocess, **O**utputs, and **C**ustomers of a process. It's a qualitative tool designed to define the project's boundaries and get everyone on the same page.
            - **Strength:** Unmatched simplicity for facilitating team alignment. It's the essential first step to building a shared understanding of the process from a bird's-eye view.
            - **Caveat:** Its value is entirely dependent on the existing domain knowledge of the team. It cannot uncover relationships or variables the team isn't already aware of.
            """)
            # ENHANCED VISUALIZATION
            st.plotly_chart(plot_sipoc_visual(), use_container_width=True)
        with col2:
            st.markdown("##### **ML Counterpart: Causal Discovery**")
            st.info("""
            **What is it?** A family of algorithms (e.g., PC, FCI, LiNGAM) that analyze observational data to infer a graph of probable cause-and-effect relationships. It generates a data-driven map of potential process drivers.
            - **Strength:** Objectively discovers potential causal links and latent interactions that human experts might overlook. Excellent for generating data-driven hypotheses before the Analyze phase.
            - **Caveat:** Requires large, high-quality datasets and outputs *hypotheses* that require validation, not proven facts.
            """)
            # ENHANCED VISUALIZATION
            st.graphviz_chart(plot_causal_discovery_visual())

    # --- Tool 3: VOC, CTQ Tree, Kano, and NLP ---
    with st.container(border=True):
        st.subheader("3. The Target: Understanding Customer Needs")
        st.markdown("This is perhaps the most critical step: translating the vague 'Voice of the Customer' (VOC) into specific, measurable project metrics, known as 'Critical to Quality' (CTQ) requirements.")

        tab1, tab2, tab3 = st.tabs(["📊 CTQ Tree", "💖 Kano Model", "🤖 NLP for VOC"])
        with tab1:
            st.markdown("""
            ##### **Classical Tool: CTQ Tree**
            **What is it?** A visual diagram that breaks down broad customer needs into more specific, actionable, and, most importantly, *measurable* requirements. It's the bridge from VOC to data.
            - **Strength:** Provides a logical, structured way to ensure the project is focused on metrics that genuinely matter to the customer.
            - **Caveat:** Can be overly simplistic if not validated. The team's assumptions about drivers must be tested.
            """)
            # ENHANCED VISUALIZATION
            st.graphviz_chart(plot_ctq_tree_visual())
        with tab2:
            st.markdown("""
            ##### **Classical Tool: Kano Model**
            **What is it?** A framework for prioritizing customer requirements by categorizing them into **Basic** (must-haves), **Performance** (more is better), and **Excitement** (delighters).
            - **Strength:** Moves beyond a simple ranked list to understand the *non-linear impact* of features on satisfaction. Prevents over-investing in basic features while highlighting opportunities for delight.
            - **Caveat:** Requires structured survey data, which can be time-consuming to collect and analyze.
            """)
            # ENHANCED VISUALIZATION
            st.plotly_chart(plot_kano_visual(), use_container_width=True)
        with tab3:
            st.markdown("""
            ##### **ML Counterpart: NLP Topic Modeling**
            **What is it?** Using algorithms (e.g., BERTopic) to analyze vast amounts of unstructured text (reviews, support tickets, survey comments) to automatically extract themes, quantify their prevalence, and gauge associated sentiment.
            - **Strength:** Massively scalable, real-time, and objective. Can analyze millions of data points to instantly uncover emerging trends and quantify issue prevalence far beyond human capacity.
            - **Caveat:** Requires access to clean text data; may miss the 'why' without deeper qualitative follow-up.
            """)
            # ENHANCED VISUALIZATION
            st.plotly_chart(plot_voc_treemap(), use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy for the Define Phase:**
    1.  **Mandate with the Charter:** Start with a Project Charter to secure buy-in and formalize goals.
    2.  **Scope with SIPOC:** Use a team-based SIPOC session to establish project boundaries and create a shared language.
    3.  **Discover with NLP & Causal Graphs:** Run NLP on existing customer feedback (e.g., support tickets, reviews) to get a raw, data-driven list of customer issues. If historical process data exists, run causal discovery to generate a data-driven "proto-fishbone" diagram of potential drivers.
    4.  **Translate & Prioritize with CTQ & Kano:** Use the NLP output to build a CTQ Tree, translating vague complaints into measurable metrics. Then, use a targeted Kano-style survey to correctly prioritize these CTQs based on their actual impact on customer satisfaction.
    """)


# ==============================================================================
# PAGE 2: MEASURE PHASE
# ==============================================================================
def show_measure_phase():
    st.title("🔬 Measure Phase: Quantifying the Current State")
    st.markdown("""
    **Objective:** To validate the measurement system's reliability, collect data, and establish a robust, data-driven baseline of the process's current performance. The mantra is **"if you can't measure it, you can't improve it."**
    """)
    st.markdown("---")

    # --- Tool 1: Measurement System Analysis (MSA) ---
    with st.container(border=True):
        st.subheader("1. Foundational Prerequisite: Measurement System Analysis (MSA)")
        st.warning("""
        **You cannot trust your data until you trust your measurement system.** An SME would never proceed without first validating their "ruler." MSA quantifies how much variation in your data comes from the measurement system itself versus the actual process.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("##### **Classical Tool: Gage R&R**")
            st.info("""
            **What is it?** A designed experiment to assess a measurement system's **Repeatability** (variation from one operator using the same tool) and **Reproducibility** (variation between different operators using the same tool).
            - **Strength:** A standardized, rigorous, and universally accepted method for qualifying a measurement system.
            - **Caveat:** Requires a planned, often disruptive, experiment. Can be costly and time-consuming to execute.
            """)
        with col2:
            st.plotly_chart(plot_gage_rr_variance_components(), use_container_width=True)

    # --- Tool 2: Process Mapping ---
    with st.container(border=True):
        st.subheader("2. Understanding the Process Flow")
        tab1, tab2 = st.tabs(["🗺️ Value Stream Mapping (VSM)", "🤖 Process Mining"])
        with tab1:
            st.markdown("##### **Classical Tool: Value Stream Mapping (VSM)**")
            st.info("""
            **What is it?** A detailed flowcharting method that documents every step in a process, capturing critical metrics like cycle time, wait time, uptime, and identifying value-added vs. non-value-added activities.
            - **Strength:** Provides a comprehensive, holistic view. Forces the team to physically "walk the gemba" (the real place of work) and build consensus. Excellent for identifying the 8 wastes of Lean.
            - **Caveat:** A manual, time-consuming snapshot in time. It represents the process as it *should* be, not necessarily as it *is*, and struggles to capture complex deviations.
            """)
            st.plotly_chart(plot_vsm(), use_container_width=True)
        with tab2:
            st.markdown("##### **ML Counterpart: Process Mining**")
            st.info("""
            **What is it?** Algorithms that automatically discover and visualize a real process model directly from event logs in IT systems (e.g., an ERP, CRM, or MES).
            - **Strength:** Discovers the process as it *actually* happens, including all unexpected deviations, rework loops, and true bottlenecks. It's objective, data-driven, and dynamic.
            - **Caveat:** Requires clean, structured event log data with three key elements: a **Case ID**, an **Activity Name**, and a **Timestamp**. Data quality is paramount.
            """)
            st.graphviz_chart(plot_process_mining_graph())

    # --- Tool 3: Process Capability ---
    with st.container(border=True):
        st.subheader("3. Understanding Process Capability")
        st.markdown("Capability analysis answers the fundamental question: **Is our process capable of consistently meeting customer specifications?** It compares the Voice of the Process (what it's actually doing) with the Voice of the Customer (what we need it to do).")
        
        st.sidebar.header("Capability Simulator")
        st.sidebar.markdown("Adjust the process and specification limits to see how they affect capability.")
        lsl = st.sidebar.slider("Lower Spec Limit (LSL)", 80.0, 95.0, 90.0, key="m_lsl")
        usl = st.sidebar.slider("Upper Spec Limit (USL)", 105.0, 120.0, 110.0, key="m_usl")
        process_mean = st.sidebar.slider("Process Mean (μ)", 95.0, 105.0, 101.5, key="m_mean")
        process_std = st.sidebar.slider("Process Std Dev (σ)", 0.5, 5.0, 2.0, key="m_std")
        
        col3, col4 = st.columns([1, 2])
        with col3:
            st.markdown("##### **Classical: Cp & Cpk**")
            st.info("Industry-standard indices that summarize capability into a single number, assuming the data is normally distributed.")
            data = generate_process_data(process_mean, process_std, 1000, lsl, usl)
            fig_cap, cp, cpk = plot_capability_analysis_pro(data, lsl, usl)
            st.metric("Process Potential (Cp)", f"{cp:.2f}")
            st.metric("Process Capability (Cpk)", f"{cpk:.2f}")
            if cpk < 1.0: st.error("Process is not capable.", icon="🚨")
            elif cpk < 1.33: st.warning("Process is marginal.", icon="⚠️")
            else: st.success("Process is capable.", icon="✅")
        with col4:
            st.markdown("##### **ML: Distributional View**")
            st.info("While Cpk is a useful summary, it can be misleading. Non-parametric methods like **Kernel Density Estimation (KDE)** visualize the *true* shape of the process data, revealing issues like skewness or multiple modes (bimodality) that single-point indices hide.")
            st.plotly_chart(fig_cap, use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy for the Measure Phase:**
    1.  **Validate with Gage R&R:** Before any data collection, perform an MSA (like a Gage R&R) to ensure your measurement system is reliable.
    2.  **Discover with Process Mining:** Start by running process mining on system logs to get an objective map of the real process flow and identify major bottlenecks or rework loops.
    3.  **Detail with VSM:** Use the insights from process mining to guide a targeted, physical VSM exercise on the most problematic parts of the process. This adds the "gemba" context that logs alone lack.
    4.  **Baseline with Cpk, Diagnose with KDE:** After a successful MSA, calculate and report the official Cpk baseline to stakeholders. Use the KDE plot internally to diagnose the *reason* for poor capability (e.g., a shifted mean, excessive spread, or non-normality).
    """)


# ==============================================================================
# PAGE 3: ANALYZE PHASE
# ==============================================================================
def show_analyze_phase():
    st.title("📈 Analyze Phase: Uncovering the Root Causes")
    st.markdown("""
    **Objective:** To analyze data to identify, validate, and quantify the root cause(s) of defects or variation. This is the detective work of Six Sigma, moving from *what* is happening to *why* it is happening.
    """)
    st.markdown("---")
    
    # --- Tool 1: Qualitative Analysis ---
    with st.container(border=True):
        st.subheader("1. Structuring the Brainstorm: Qualitative Root Cause Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical Tool: Fishbone (Ishikawa) Diagram**")
            st.info("""
            **What is it?** A structured brainstorming tool used to visually organize potential causes of a specific problem (the "effect"). Causes are grouped into major categories.
            - **Strength:** Promotes comprehensive, systematic thinking. Excellent for team collaboration and ensuring all possible avenues are explored.
            - **Caveat:** It's a qualitative tool that generates *potential* causes. Each hypothesized cause must be validated with data.
            """)
            st.graphviz_chart(plot_fishbone_diagram())
        with col2:
            st.markdown("##### **Classical Tool: Pareto Chart**")
            st.info("""
            **What is it?** A bar chart that displays the frequency of problems in descending order, combined with a line graph showing the cumulative percentage. It visualizes the "80/20 Rule."
            - **Strength:** Clearly separates the "vital few" problems from the "trivial many," allowing teams to focus their efforts on the issues with the biggest impact.
            - **Caveat:** Only shows the frequency or cost of a problem, not its root cause. It tells you *what* to work on first, not *how* to fix it.
            """)
            st.plotly_chart(plot_pareto_chart(), use_container_width=True)

    # --- Tool 2: Comparing Groups ---
    with st.container(border=True):
        st.subheader("2. Proving the Difference: Comparing Group Performance")
        st.markdown("Once you have hypotheses (e.g., 'Supplier A is worse than Supplier B'), you need statistical proof.")

        st.sidebar.header("Group Comparison Simulator")
        st.sidebar.markdown("Adjust the means of three suppliers to see if the difference becomes statistically significant.")
        mean1 = st.sidebar.slider("Supplier A Mean Yield", 98.0, 102.0, 99.5, 0.1, key='a1')
        mean2 = st.sidebar.slider("Supplier B Mean Yield", 98.0, 102.0, 100.0, 0.1, key='a2')
        mean3 = st.sidebar.slider("Supplier C Mean Yield", 98.0, 102.0, 100.5, 0.1, key='a3')
        anova_data = generate_anova_data(means=[mean1, mean2, mean3], stds=[0.5, 0.5, 0.5], n=50)

        tab1, tab2 = st.tabs(["🔬 Classical: ANOVA", "💻 ML: Permutation Testing"])
        with tab1:
            st.markdown("##### **Classical: ANOVA**")
            st.info("""
            **What is it?** Analysis of Variance (ANOVA) is a statistical test used to determine if there are any statistically significant differences between the means of two or more independent groups.
            - **Strength:** A rigorous, standard method for comparing groups (e.g., yield from different machines, shifts, or suppliers).
            - **Caveat:** Assumes the data within each group is normally distributed and has equal variances. Can be sensitive to outliers.
            """)
            fig_anova, p_val = plot_anova_groups(anova_data)
            st.plotly_chart(fig_anova, use_container_width=True)
            if p_val < 0.05: st.error(f"P-value is {p_val:.4f}. Reject the null hypothesis: a significant difference exists.", icon="🚨")
            else: st.success(f"P-value is {p_val:.4f}. Fail to reject the null hypothesis: no significant difference detected.", icon="✅")
        with tab2:
            st.markdown("##### **ML Counterpart: Permutation Testing**")
            st.info("""
            **What is it?** A non-parametric, computational method. It works by shuffling the group labels thousands of times and recalculating the statistic (e.g., difference in means). This builds an empirical distribution of what's possible by pure chance.
            - **Strength:** Makes no assumptions about the data's distribution (normality, etc.). It's more robust, intuitive, and often more reliable than classical tests, especially with complex data.
            - **Caveat:** Can be computationally intensive for very large datasets.
            """)
            st.plotly_chart(plot_permutation_test(anova_data), use_container_width=True)

    # --- Tool 3: Finding Relationships ---
    with st.container(border=True):
        st.subheader("3. Finding the Drivers: Identifying Variable Relationships")
        st.markdown("This is the core of root cause analysis: finding the specific input variables (X's) that drive the output (Y).")
        df_reg = generate_nonlinear_data()
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### **Classical: Linear Regression**")
            st.info("Models the linear relationship between inputs (X) and an output (Y). Simple, highly interpretable, but often fails to capture real-world complexity.")
            fig_reg, _, _ = plot_regression_comparison_pro(df_reg)
            st.plotly_chart(fig_reg, use_container_width=True)
        with col4:
            st.markdown("##### **ML: Ensemble Models & Explainability (SHAP)**")
            st.info("Ensemble models like **Random Forest** or **Gradient Boosting** capture complex, non-linear relationships with high accuracy. We then use explainability tools like **SHAP** to understand the 'black box' model.")
            _, model, X_reg = plot_regression_comparison_pro(df_reg)
            fig_shap = plot_shap_summary(model, X_reg)
            st.plotly_chart(fig_shap, use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy for the Analyze Phase:**
    1.  **Structure with Fishbone & Pareto:** Use a Fishbone diagram in a team setting to brainstorm all potential causes. Then, use a Pareto chart on defect data to identify the most frequent problem types to investigate first.
    2.  **Verify with ANOVA (or Permutation):** For comparing group performance (e.g., from the Pareto chart), ANOVA is the standard first step. If its assumptions are violated (check with a normality test), use a more robust permutation test.
    3.  **Model with Both, Trust the Best:** Fit both a Linear Regression (for a simple baseline) and an ensemble ML model. If the ML model is significantly more accurate (check R²), its feature importance rankings from SHAP are a more reliable guide to the true root causes than the coefficients from a poorly-fitting linear model.
    """)


# ==============================================================================
# PAGE 4: IMPROVE PHASE
# ==============================================================================
def show_improve_phase():
    st.title("⚙️ Improve Phase: Discovering and Implementing Solutions")
    st.markdown("""
    **Objective:** To identify, test, and implement solutions that address the root causes discovered in the Analyze phase. This involves moving from analysis to action, finding the optimal settings for our critical process inputs (X's), and proactively mitigating risks.
    """)
    st.markdown("---")
    
    # --- Tool 1: Finding Optimal Settings ---
    with st.container(border=True):
        st.subheader("1. Finding Optimal Process Settings")
        st.markdown("Once we know which X's are critical, we need to find their optimal settings to maximize our Y.")

        tab1, tab2 = st.tabs(["🧪 Classical: Design of Experiments (DOE)", "🤖 ML: Bayesian Optimization"])
        with tab1:
            st.markdown("##### **Classical: Design of Experiments (DOE)**")
            st.info("""
            **What is it?** A structured statistical method for efficiently changing multiple inputs simultaneously to determine their individual (main) and combined (interaction) effects on the output.
            - **Strength:** The gold standard for physical experimentation. Statistically rigorous, highly informative, and the most reliable way to establish causality.
            - **Caveat:** Suffers from the "curse of dimensionality." The number of runs required grows exponentially with the number of factors, making it impractical for more than ~5-7 factors.
            """)
            doe_data = generate_doe_data()
            fig_doe_main, fig_doe_interaction = plot_doe_effects(doe_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_doe_cube(doe_data), use_container_width=True)
            with col2:
                st.plotly_chart(fig_doe_main, use_container_width=True)
                st.plotly_chart(fig_doe_interaction, use_container_width=True)

        with tab2:
            st.markdown("##### **ML: Bayesian Optimization**")
            st.info("""
            **What is it?** An intelligent search algorithm for finding the global optimum of an expensive-to-evaluate function (e.g., a 3-day simulation, a costly physical experiment, or tuning a complex ML model).
            - **Strength:** Extremely sample-efficient, especially in high-dimensional spaces. It intelligently balances exploiting known good solutions and exploring new, uncertain areas.
            - **Caveat:** Can be sensitive to initial parameters and may struggle with very 'spiky', discontinuous functions.
            """)
            st.sidebar.header("Bayesian Opt. Simulator")
            st.sidebar.markdown("Click the button to let the algorithm intelligently choose the next best point to sample.")
            # Define the true function to optimize (cached for performance)
            @st.cache_data 
            def true_func(x): return (np.sin(x * 0.8) * 15) + (np.cos(x * 2.5)) * 5 - (x/10)**3
            
            x_range = np.linspace(0, 20, 200)
            
            # Initialize session state for interactive sampling
            if 'sampled_points' not in st.session_state:
                st.session_state.sampled_points = {'x': [2.0, 18.0], 'y': [true_func(2.0), true_func(18.0)]}
            
            if st.sidebar.button("Sample Next Best Point"): 
                _, next_point = plot_bayesian_optimization_interactive(true_func, x_range, st.session_state.sampled_points)
                st.session_state.sampled_points['x'].append(next_point)
                st.session_state.sampled_points['y'].append(true_func(next_point))
            
            if st.sidebar.button("Reset Simulation"): 
                st.session_state.sampled_points = {'x': [2.0, 18.0], 'y': [true_func(2.0), true_func(18.0)]}
            
            fig_bo, _ = plot_bayesian_optimization_interactive(true_func, x_range, st.session_state.sampled_points)
            st.plotly_chart(fig_bo, use_container_width=True)

    # --- Tool 2: Mitigating Risks ---
    with st.container(border=True):
        st.subheader("2. Proactively Mitigating Risks")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### **Classical: FMEA**")
            st.info("""
            **What is it?** Failure Mode and Effects Analysis is a structured, team-based risk assessment tool. It involves brainstorming potential failure modes, their effects, and their causes, then ranking them by a **Risk Priority Number (RPN)**.
            - **Strength:** A powerful, systematic way to force a team to think about what *could* go wrong and prioritize preventative actions.
            - **Caveat:** The RPN scores are subjective, qualitative, and based on team consensus, not always on hard data. It can be prone to bias.
            """)
            st.plotly_chart(plot_fmea_table(), use_container_width=True)
        with col4:
            st.markdown("##### **ML: Prognostics & Health Management (PHM)**")
            st.info("""
            **What is it?** A data-driven approach where ML models are trained on sensor data (vibration, temperature, pressure etc.) to predict equipment degradation and estimate its **Remaining Useful Life (RUL)** before a failure occurs.
            - **Strength:** Moves risk management from a qualitative exercise to a quantitative, predictive capability, enabling condition-based maintenance instead of scheduled maintenance.
            - **Caveat:** Requires high-quality, high-frequency sensor data, including run-to-failure examples, which can be difficult or expensive to obtain.
            """)
            st.plotly_chart(plot_rul_prediction(generate_sensor_degradation_data()), use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy for the Improve Phase:**
    1.  **Optimize with the Right Tool:** For physical processes with few (<7) variables where experiments are feasible, use **DOE**. For complex digital systems, simulations, or when experiments are very expensive, use **Bayesian Optimization**.
    2.  **Mitigate with FMEA, Predict with PHM:** Use a classical FMEA to identify the highest-risk failure modes based on team expertise. For the top risks identified, investigate if sensor data is available to build a PHM/RUL model. This turns a qualitative risk into a quantitative, predictive control.
    3.  **The Ultimate Hybrid ("Digital Twin"):** Use DOE data to train a highly accurate ML model of your process (a "surrogate model" or "digital twin"). Then, use Bayesian Optimization on this fast, cheap digital twin to find the global optimum virtually before performing one final confirmation run in the real world.
    """)

# ==============================================================================
# PAGE 5: CONTROL PHASE
# ==============================================================================
def show_control_phase():
    st.title("📡 Control Phase: Sustaining and Monitoring the Gains")
    st.markdown("""
    **Objective:** To implement a robust system to monitor the improved process, ensuring it remains stable and that improvements are sustained. The goal is to create a Control Plan and move from reactive problem-solving to proactive process management.
    """)
    st.markdown("---")

    # --- Tool 1: Control Charts ---
    with st.container(border=True):
        st.subheader("1. Monitoring for Stability: Statistical Process Control (SPC)")
        st.markdown("Control charts are the primary tool for monitoring process stability over time, distinguishing between 'common cause' (natural) variation and 'special cause' (assignable) variation that requires investigation.")
        
        st.sidebar.header("Control Simulator")
        st.sidebar.markdown("Introduce a small, sustained shift in the process and see which chart detects it faster.")
        shift_mag = st.sidebar.slider("Magnitude of Shift (in Std Devs)", 0.2, 3.0, 1.0, 0.1, key="ctrl_shift_mag")
        ewma_lambda = st.sidebar.slider("EWMA Lambda (λ)", 0.1, 0.5, 0.2, 0.05, help="Higher λ reacts faster but is more sensitive to noise.")
        chart_data = generate_control_chart_data(shift_point=75, shift_magnitude=shift_mag)

        tab1, tab2, tab3 = st.tabs(["📊 Classical: Shewhart Chart", "📈 Advanced Classical: EWMA/CUSUM", "🤖 ML: Multivariate & Anomaly Detection"])
        with tab1:
            st.markdown("##### **Classical: Shewhart Chart (X-bar)**")
            st.info("""
            **What is it?** The standard SPC chart. It plots individual measurements or subgroup means over time and uses control limits set at ±3 standard deviations from the center line. Each point is treated independently.
            - **Strength:** Simple, robust, and excellent for detecting large (>1.5σ) shifts quickly.
            - **Limitation:** Slow to detect small, sustained shifts (e.g., <1.5σ), as it has no 'memory' of past data points.
            """)
            st.plotly_chart(plot_shewhart_chart(chart_data), use_container_width=True)
        with tab2:
            st.markdown("##### **Advanced Classical: EWMA & CUSUM Charts**")
            st.info("""
            **What are they?** These charts have 'memory', making them highly effective at detecting small, sustained shifts that Shewhart charts would miss.
            - **EWMA (Exponentially Weighted Moving Average):** Gives more weight to recent data points and exponentially less to older ones.
            - **CUSUM (Cumulative Sum):** Accumulates deviations from the mean over time.
            - **Strength:** Significantly more sensitive to small shifts than a Shewhart chart.
            - **Limitation:** Can be more complex to set up and interpret. The choice of parameters (λ for EWMA, k for CUSUM) is a trade-off between sensitivity and false alarms.
            """)
            st.plotly_chart(plot_ewma_chart(chart_data, lambda_val=ewma_lambda), use_container_width=True)
            st.plotly_chart(plot_cusum_chart(chart_data), use_container_width=True)
        with tab3:
            st.markdown("##### **ML: Multivariate Anomaly Detection**")
            st.info("""
            **What is it?** While classical SPC monitors one variable at a time, many processes have dozens of correlated inputs. ML models can monitor the 'health' of the entire system at once.
            - **Hotelling's T² Chart:** A statistical method for monitoring two or more correlated variables.
            - **Autoencoders / LSTMs:** Unsupervised deep learning models that learn the normal 'heartbeat' of a process across all sensor inputs. They then flag any pattern of behavior that deviates from this learned norm.
            - **Strength:** Detects subtle, multivariate drifts that individual SPC charts would miss. Can provide a single, holistic process health score.
            - **Caveat:** Can be a 'black box'. When an anomaly is flagged, it may require further analysis (e.g., using SHAP) to identify which variable(s) caused the deviation.
            """)
            st.plotly_chart(plot_hotelling_t2_chart(), use_container_width=True)

    # --- Tool 2: The Control Plan ---
    with st.container(border=True):
        st.subheader("2. Formalizing the Gains: The Control Plan")
        st.info("""
        **What is it?** A living document that details the methods, responsibilities, and reaction plan for maintaining control of the improved process. It's the "hand-off" document that ensures the gains are sustained long after the project team disbands.
        - **Strength:** Provides a clear, actionable plan for process owners. It operationalizes the findings of the entire DMAIC project.
        - **Caveat:** It's only effective if it's used, reviewed, and updated regularly. It cannot be a "file and forget" document.
        """)
        st.plotly_chart(plot_control_plan(), use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy for the Control Phase:**
    1.  **Monitor CTQs with Shewhart:** Keep a classical Shewhart chart on your final Critical-to-Quality (CTQ) output for simple, robust, and easily interpretable monitoring.
    2.  **Monitor Critical Inputs with Advanced SPC:** For the critical input variables (X's) that drive your process, use more sensitive charts like **EWMA** or **CUSUM** to detect small drifts before they impact the final output.
    3.  **Create an Early Warning System with ML:** For the most complex, high-stakes processes, deploy a multivariate ML anomaly detection model (like a Hotelling's T² or an Autoencoder) that learns the normal 'heartbeat' of your process across all sensor inputs. This provides a holistic early warning system.
    4.  **Codify Everything in a Control Plan:** The plan must document which charts are used for which variables, the control limits, the measurement frequency, and the exact reaction plan for any out-of-control signal.
    """)


# ==============================================================================
# PAGE 6: COMPARISON MATRIX (VISUALLY ENHANCED)
# ==============================================================================
def show_comparison_matrix():
    st.title("⚔️ Head-to-Head: Classical vs. Machine Learning")
    st.markdown("A visual comparison of the core philosophies and practical strengths of the two approaches.")
    st.markdown("---")

    # --- Radar Chart Visualization ---
    with st.container(border=True):
        st.subheader("Approach DNA: A Multi-Dimensional View")
        st.markdown("""
        This radar chart provides an at-a-glance comparison of the two methodologies across key attributes. 
        A larger area for a given approach indicates a greater strength in those dimensions. Notice the different "shapes" of their capabilities.
        """)
        st.plotly_chart(plot_comparison_radar(), use_container_width=True)

    # --- Diverging Bar Chart ---
    with st.container(border=True):
        st.subheader("The Verdict: Who Wins for Which Task?")
        st.markdown("""
        This chart visualizes which approach is generally superior for specific, common tasks in process improvement. 
        The direction and color of the bar indicate the winner, providing a clear, decisive verdict for each use case.
        """)
        st.plotly_chart(plot_verdict_barchart(), use_container_width=True)

# ==============================================================================
# PAGE 7: HYBRID STRATEGY (VISUALLY ENHANCED)
# ==============================================================================
def show_hybrid_strategy():
    st.title("🤝 The Hybrid Manifesto: The Future of Process Excellence")
    st.markdown("The most competitive organizations do not choose one over the other; they build an **AI-Augmented Six Sigma** program that fuses statistical rigor with machine learning's predictive power. This is not about replacement; it's about augmentation.")
    st.markdown("---")

    # --- Synergy Diagram ---
    with st.container(border=True):
        st.subheader("The Philosophy of Synergy")
        st.markdown("""
        Neither methodology is a silver bullet. The true power lies in their integration. Classical statistics provides the **rigor for inference and causality**, while machine learning provides the **power for prediction and scale**. This diagram illustrates their complementary nature, with the greatest value found in the overlap.
        """)
        st.plotly_chart(plot_synergy_diagram(), use_container_width=True)
    
    # --- Interactive Recommender ---
    with st.container(border=True):
        st.subheader("Interactive Solution Recommender")
        st.info("💡 Select a common business scenario to see the recommended approach and rationale.")
        
        guidance_data = get_guidance_data()
        scenarios = list(guidance_data.keys())
        
        selected_scenario = st.selectbox("Choose your scenario:", scenarios, label_visibility="collapsed")
        
        if selected_scenario:
            recommendation = guidance_data[selected_scenario]['approach']
            rationale = guidance_data[selected_scenario]['rationale']
            
            st.markdown(f"##### Recommended Approach: {recommendation}")
            st.markdown(f"**Why?** {rationale}")

    # --- Existing Workflow Diagram ---
    with st.container(border=True):
        st.subheader("A Unified, Modern DMAIC Workflow")
        st.markdown("This workflow demonstrates how to embed ML augmentation at each step of the traditional DMAIC cycle.")
        st.markdown(get_workflow_css(), unsafe_allow_html=True)
        st.markdown('<div class="workflow-container">', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="🌀 1. Define",
            phase_class="step-define",
            classical_tools=["Project Charter", "SIPOC", "Kano Model", "CTQ Tree"],
            ml_tools=["NLP for VOC at Scale", "Causal Discovery", "Automated Customer Segmentation"]
        ), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="🔬 2. Measure",
            phase_class="step-measure",
            classical_tools=["Gage R&R (MSA)", "Process Capability (Cp, Cpk)", "Value Stream Mapping"],
            ml_tools=["Process Mining", "Kernel Density Estimation (KDE)", "Uncertainty Quantification"]
        ), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="📈 3. Analyze",
            phase_class="step-analyze",
            classical_tools=["Hypothesis Testing (ANOVA)", "Pareto Analysis", "Fishbone Diagram", "Linear Regression"],
            ml_tools=["Feature Importance (SHAP)", "Ensemble Models (Random Forest)", "Permutation Testing"]
        ), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="⚙️ 4. Improve",
            phase_class="step-improve",
            classical_tools=["Design of Experiments (DOE)", "FMEA", "Pilot Testing"],
            ml_tools=["Bayesian Optimization", "Prognostics (PHM/RUL)", "Simulation & Digital Twins"]
        ), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="📡 5. Control",
            phase_class="step-control",
            classical_tools=["Control Charts (SPC, EWMA)", "Control Plan", "Standard Operating Procedures (SOPs)"],
            ml_tools=["Multivariate Anomaly Detection", "Real-time Predictive Models", "Automated Alerting"]
        ), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
