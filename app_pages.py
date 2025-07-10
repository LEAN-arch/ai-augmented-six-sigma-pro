# app_pages.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Import all necessary helper functions from the single helper file.
# Using '*' is appropriate here as this module is a dedicated consumer of the app_helpers toolkit.
from app_helpers import *

# ==============================================================================
# PAGE 0: WELCOME & OVERVIEW (NEW)
# ==============================================================================
def show_welcome_page():
    st.title("🚀 Welcome to the AI-Augmented Process Excellence Framework")
    st.markdown("##### An interactive playbook for the modern, data-driven improvement professional.")
    st.markdown("---")

    st.info("""
    **This application is designed for a technically proficient audience** (e.g., Master Black Belts, Data Scientists, Process Engineers).
    It moves beyond introductory concepts to demonstrate a powerful, unified framework that fuses the **inferential rigor of classical Six Sigma** with the **predictive and scaling power of modern Machine Learning**.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classical Six Sigma")
        st.markdown("""
        The gold standard for process improvement, built on a foundation of statistical inference, hypothesis testing, and designed experiments.
        - **Core Strength:** Establishing causality and ensuring statistical rigor.
        - **Primary Focus:** Variance reduction, defect elimination, and building deep process understanding.
        - **Best Suited For:** Problems with structured data, limited variables, and where interpretability and regulatory compliance are paramount.
        """)

    with col2:
        st.subheader("Machine Learning Augmentation")
        st.markdown("""
        A suite of computational techniques that excel at finding patterns, making predictions, and automating decisions from large, complex, and high-dimensional data.
        - **Core Strength:** Prediction, pattern recognition, and scalability.
        - **Primary Focus:** Optimizing outcomes, proactive control, and handling complexity beyond human capacity.
        - **Best Suited For:** Problems with high-dimensional or unstructured data, non-linear interactions, and where predictive accuracy is the key objective.
        """)

    st.subheader("The Hybrid Philosophy: Augmentation, Not Replacement")
    st.markdown("""
    The central thesis of this application is that pitting these two disciplines against each other is a false dichotomy. The most effective path to process excellence in the 21st century lies in their **synergistic integration**.
    
    Use the navigation panel on the left to explore the traditional **DMAIC (Define, Measure, Analyze, Improve, Control)** cycle. Each phase will present:
    1.  **Classical Tools:** The trusted, foundational methods.
    2.  **ML Counterparts:** The modern techniques that augment and scale the classical approach.
    3.  **Hybrid Strategy:** A prescriptive guide on how to combine them for superior results.
    """)
    st.success("Click on a phase in the sidebar to begin your exploration.")


# ==============================================================================
# PAGE 1: DEFINE PHASE
# ==============================================================================
def show_define_phase():
    st.title("🌀 Define Phase: Establishing the Project's North Star")
    st.markdown("""
    **Objective:** To clearly articulate the business problem, establish the project's mandate through a formal charter, define the scope with precision, and translate the qualitative 'Voice of the Customer' (VOC) into quantifiable, 'Critical to Quality' (CTQ) metrics.
    """)
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. The Mandate: Project Charter")
        st.markdown("""
        The Project Charter is the foundational document that formally authorizes a project, acting as a contract between the team and leadership. Its primary technical function is to prevent "scope creep" by creating an immutable reference for project objectives, deliverables, and boundaries.
        """)
        st.plotly_chart(plot_project_charter_visual(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. The Landscape: Mapping the Process & Generating Hypotheses")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical Tool: SIPOC Diagram**")
            st.info("""
            A high-level process map capturing **S**uppliers, **I**nputs, **P**rocess, **O**utputs, and **C**ustomers.
            - **Function:** A qualitative, expert-driven tool for defining project boundaries and fostering team alignment. It visualizes the existing domain knowledge.
            - **Limitation:** Cannot discover relationships or variables the team isn't already aware of. It represents a static, consensus view of the process.
            """)
            st.plotly_chart(plot_sipoc_visual(), use_container_width=True)
        with col2:
            st.markdown("##### **ML Augmentation: Causal Discovery**")
            st.info("""
            Algorithms (e.g., PC, FCI, LiNGAM) that analyze observational data to infer a graph of probable cause-and-effect relationships.
            - **Function:** Objectively generates data-driven hypotheses about process drivers, potentially uncovering latent interactions that human experts might overlook.
            - **Limitation:** Requires large, high-quality datasets. It outputs a *hypothesis graph* that requires experimental validation (e.g., via DOE), not a proven causal map. Correlation is not causation, but this points to where to look for it.
            """)
            st.graphviz_chart(plot_causal_discovery_visual())

    with st.container(border=True):
        st.subheader("3. The Target: Translating the Voice of the Customer (VOC)")
        st.markdown("This is where we translate vague customer needs into specific, measurable project metrics (CTQs).")
        tab1, tab2, tab3 = st.tabs(["📊 CTQ Tree", "💖 Kano Model", "🤖 NLP for Scaled VOC"])
        with tab1:
            st.markdown("##### **Classical Tool: CTQ Tree**")
            st.info("""A decomposition tool to break down broad customer needs into specific and measurable requirements. It provides a logical bridge from the qualitative VOC to quantitative project metrics.""")
            st.graphviz_chart(plot_ctq_tree_visual())
        with tab2:
            st.markdown("##### **Classical Tool: Kano Model**")
            st.info("""A framework for prioritizing customer requirements by categorizing them into **Basic**, **Performance**, and **Excitement** needs. Its strength lies in understanding the non-linear impact of features on satisfaction, preventing over-investment in basic attributes.""")
            st.plotly_chart(plot_kano_visual(), use_container_width=True)
        with tab3:
            st.markdown("##### **ML Augmentation: NLP Topic Modeling & Sentiment Analysis**")
            st.info("""Using algorithms (e.g., BERTopic, Latent Dirichlet Allocation) to analyze vast amounts of unstructured text (reviews, support tickets, survey comments) to automatically extract themes, quantify their prevalence, and gauge associated sentiment. This scales VOC analysis beyond human capacity.""")
            st.plotly_chart(plot_voc_treemap(), use_container_width=True)

    st.success("""
    **🏆 Hybrid Strategy for the Define Phase:**
    1.  **Mandate & Scope (Classical):** Begin with a formal **Project Charter** and a team-based **SIPOC** session to establish clear boundaries and alignment based on expert knowledge.
    2.  **Discover at Scale (ML):** Deploy **NLP Topic Modeling** on all available customer text data (e.g., support tickets, online reviews) to generate a raw, data-driven list of customer pain points. If historical process data exists, run **Causal Discovery** to generate a data-driven "proto-fishbone" diagram of potential drivers.
    3.  **Translate & Prioritize (Hybrid):** Use the high-frequency topics from the NLP output as primary inputs for building a **CTQ Tree**. This ensures the tree is grounded in data, not just assumptions. If necessary, use a targeted **Kano** survey to correctly prioritize these data-driven CTQs based on their non-linear impact on customer satisfaction.
    """)


# ==============================================================================
# PAGE 2: MEASURE PHASE
# ==============================================================================
def show_measure_phase():
    st.title("🔬 Measure Phase: Quantifying the Current State")
    st.markdown("""
    **Objective:** To validate the measurement system's reliability (MSA), collect data, and establish a robust, data-driven baseline of the process's current performance. The mantra is **"if you can't measure it, you can't improve it."**
    """)
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Prerequisite: Measurement System Analysis (MSA)")
        st.warning("""
        **Data integrity is paramount.** Before any process data is analyzed, the measurement system itself must be validated. MSA quantifies the error contributed by the measurement system and ensures that the observed process variation is not being masked by measurement noise.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("##### **Classical Tool: Gage R&R**")
            st.info("""
            A designed experiment to partition measurement system variance into its components: **Repeatability** (equipment variation) and **Reproducibility** (appraiser variation).
            - **Function:** Provides a standardized, rigorous method for qualifying a measurement system, typically assessing its percent contribution to total variance.
            - **Limitation:** Requires a planned, often disruptive, experiment.
            """)
        with col2:
            st.plotly_chart(plot_gage_rr_variance_components(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Understanding the Process Flow")
        tab1, tab2 = st.tabs(["🗺️ Value Stream Mapping (VSM)", "🤖 Process Mining"])
        with tab1:
            st.markdown("##### **Classical Tool: Value Stream Mapping (VSM)**")
            st.info("""
            A manual, observational flowcharting method to document every step, capturing metrics like cycle time and wait time to identify value-added vs. non-value-added activities.
            - **Function:** Excellent for building team consensus and identifying the 8 wastes of Lean. Forces the team to "walk the gemba" (the real place of work).
            - **Limitation:** Represents a static snapshot in time, often showing the process as it *should* be, not necessarily as it *is*, struggling to capture complex deviations.
            """)
            st.plotly_chart(plot_vsm(), use_container_width=True)
        with tab2:
            st.markdown("##### **ML Augmentation: Process Mining**")
            st.info("""
            Algorithms that automatically discover and visualize a real process model directly from event logs in IT systems (e.g., ERP, CRM, MES).
            - **Function:** Discovers the process as it *actually* happens, including all unexpected deviations, rework loops, and true bottlenecks. It is objective, data-driven, and dynamic.
            - **Limitation:** Requires clean, structured event log data with three key elements: a **Case ID**, an **Activity Name**, and a **Timestamp**. Data quality is the primary barrier.
            """)
            st.graphviz_chart(plot_process_mining_graph())

    with st.container(border=True):
        st.subheader("3. Establishing Process Capability")
        st.markdown("Capability analysis answers: **Is our process capable of consistently meeting customer specifications?** It compares the Voice of the Process (its actual distribution) with the Voice of the Customer (the specification limits).")

        st.sidebar.header("Capability Simulator")
        st.sidebar.markdown("Adjust the process parameters and spec limits to see the impact on capability indices and the distributional fit.")
        lsl = st.sidebar.slider("Lower Spec Limit (LSL)", 80.0, 95.0, 90.0, key="m_lsl")
        usl = st.sidebar.slider("Upper Spec Limit (USL)", 105.0, 120.0, 110.0, key="m_usl")
        process_mean = st.sidebar.slider("Process Mean (μ)", 95.0, 105.0, 101.5, key="m_mean")
        process_std = st.sidebar.slider("Process Std Dev (σ)", 0.5, 5.0, 2.0, key="m_std")

        data = generate_process_data(process_mean, process_std, 2000, lsl, usl)
        fig_cap, cp, cpk = plot_capability_analysis_pro(data, lsl, usl)

        col3, col4 = st.columns([1, 2])
        with col3:
            st.markdown("##### **Classical Indices: Cp & Cpk**")
            st.info("Industry-standard indices that summarize capability, assuming the data is normally distributed. **Cp** measures potential, while **Cpk** accounts for centering.")
            st.metric("Process Potential (Cp)", f"{cp:.2f}")
            st.metric("Process Capability (Cpk)", f"{cpk:.2f}", help="A Cpk < 1.33 is generally considered not capable for most industries.")
            if cpk < 1.0: st.error("Process is not capable.", icon="🚨")
            elif cpk < 1.33: st.warning("Process is marginal.", icon="⚠️")
            else: st.success("Process is capable.", icon="✅")
        with col4:
            st.markdown("##### **ML Augmentation: Distributional View**")
            st.info("While Cpk is a useful summary, it can be misleading. Non-parametric methods like **Kernel Density Estimation (KDE)** visualize the *true* shape of the process distribution, revealing issues like skewness or multimodality that single-point indices hide.")
            st.plotly_chart(fig_cap, use_container_width=True)

    st.success("""
    **🏆 Hybrid Strategy for the Measure Phase:**
    1.  **Validate (Classical):** Always perform a **Gage R&R** or other MSA before any data collection to ensure the measurement system is reliable. This is a non-negotiable prerequisite.
    2.  **Discover (ML):** Begin by running **Process Mining** on system event logs. This provides an objective, data-driven map of the real process flow, immediately highlighting major bottlenecks, rework loops, and compliance issues.
    3.  **Detail (Classical):** Use the insights from process mining to guide a targeted, physical **VSM** exercise. Focus on the most problematic areas identified by the data to add the crucial "gemba" context that logs alone lack.
    4.  **Baseline & Diagnose (Hybrid):** After a successful MSA, calculate and report the official **Cpk** baseline for stakeholder communication. Internally, use the **KDE plot** to diagnose the *reason* for poor capability (e.g., a shifted mean, excessive spread, or non-normality) which provides a richer diagnostic picture than the index alone.
    """)


# ==============================================================================
# PAGE 3: ANALYZE PHASE
# ==============================================================================
def show_analyze_phase():
    st.title("📈 Analyze Phase: Uncovering Root Causes")
    st.markdown("""
    **Objective:** To analyze data to identify, validate, and quantify the root cause(s) of defects or variation. This is the core "detective work," moving from correlation to causation.
    """)
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Qualitative Root Cause Analysis & Prioritization")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical Tool: Fishbone Diagram**")
            st.info("A structured brainstorming tool to visually organize potential causes of a specific problem. Its value lies in promoting systematic, comprehensive team-based thinking.")
            st.graphviz_chart(plot_fishbone_diagram())
        with col2:
            st.markdown("##### **Classical Tool: Pareto Chart**")
            st.info("""A bar chart that displays problem frequency in descending order, combined with a line graph showing the cumulative percentage. It visualizes the "80/20 Rule," allowing teams to focus on the "vital few" issues.""")
            st.plotly_chart(plot_pareto_chart(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Proving the Difference: Comparing Group Performance")
        st.markdown("Once hypotheses are formed (e.g., 'Supplier A's material is causing lower yield'), statistical proof is required.")
        st.sidebar.header("Group Comparison Simulator")
        st.sidebar.markdown("Adjust supplier means to see how classical and computational tests detect significance.")
        mean1 = st.sidebar.slider("Supplier A Mean Yield", 98.0, 102.0, 99.5, 0.1, key='a1')
        mean2 = st.sidebar.slider("Supplier B Mean Yield", 98.0, 102.0, 100.0, 0.1, key='a2')
        mean3 = st.sidebar.slider("Supplier C Mean Yield", 98.0, 102.0, 100.5, 0.1, key='a3')
        anova_data = generate_anova_data(means=[mean1, mean2, mean3], stds=[0.8, 0.8, 0.8], n=50)

        tab1, tab2 = st.tabs(["🔬 Classical: ANOVA", "💻 ML Augmentation: Permutation Testing"])
        with tab1:
            st.markdown("##### **Classical: Analysis of Variance (ANOVA)**")
            st.info("""
            A statistical test to determine if significant differences exist between the means of two or more groups.
            - **Function:** A standard, rigorous method for comparing groups.
            - **Assumptions:** Assumes data within groups is normally distributed and has equal variances. Can be sensitive to outliers.
            """)
            fig_anova, p_val = plot_anova_groups(anova_data)
            st.plotly_chart(fig_anova, use_container_width=True)
            if p_val < 0.05: st.error(f"P-value is {p_val:.4f}. Reject the null hypothesis: A statistically significant difference exists.", icon="🚨")
            else: st.success(f"P-value is {p_val:.4f}. Fail to reject the null hypothesis: No significant difference detected.", icon="✅")
        with tab2:
            st.markdown("##### **ML Augmentation: Permutation Testing**")
            st.info("""
            A non-parametric, computational method. It shuffles group labels thousands of times to build an empirical distribution of what's possible by chance, then compares the observed statistic.
            - **Function:** More robust and intuitive than classical tests, as it makes **no assumptions** about the data's distribution (normality, etc.).
            - **Limitation:** Can be computationally intensive for very large datasets or many groups.
            """)
            st.plotly_chart(plot_permutation_test(anova_data), use_container_width=True)

    with st.container(border=True):
        st.subheader("3. Finding the Drivers: Identifying Variable Relationships (Y = f(x))")
        st.markdown("This is the core of root cause analysis: finding the specific input variables (X's) that mathematically drive the output (Y).")
        df_reg = generate_nonlinear_data()
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### **Classical: Multiple Linear Regression**")
            st.info("Models the linear relationship between inputs (X) and an output (Y). Simple, highly interpretable, but fundamentally assumes linearity and can fail to capture real-world complexity.")
            fig_reg, _, _ = plot_regression_comparison_pro(df_reg)
            st.plotly_chart(fig_reg, use_container_width=True)
        with col4:
            st.markdown("##### **ML Augmentation: Ensemble Models & XAI**")
            st.info("Ensemble models (**Random Forest, Gradient Boosting**) capture complex, non-linear relationships with high accuracy. We then use **eXplainable AI (XAI)** tools like **SHAP** to understand the 'black box' model's logic.")
            _, model, X_reg = plot_regression_comparison_pro(df_reg)
            fig_shap = plot_shap_summary(model, X_reg)
            st.plotly_chart(fig_shap, use_container_width=True)

    st.success("""
    **🏆 Hybrid Strategy for the Analyze Phase:**
    1.  **Structure & Prioritize (Classical):** Use a **Fishbone** diagram to brainstorm potential causes and a **Pareto** chart to identify the most frequent problem areas to investigate first.
    2.  **Verify Group Differences (Hybrid):** For comparing group performance (e.g., suppliers from the Pareto chart), **ANOVA** is the standard first step. However, always validate its assumptions (e.g., using a Shapiro-Wilk test for normality). If assumptions are violated, default to the more robust **Permutation Test**.
    3.  **Model Relationships (Hybrid):** Fit both a **Linear Regression** (for a simple, interpretable baseline) and an **Ensemble ML model** (e.g., Random Forest). Compare their performance (e.g., R², MAE). If the ML model is significantly more accurate, its feature importance rankings from **SHAP** are a more reliable guide to the true root causes than the coefficients from a poorly-fitting linear model. The linear model's failure *is itself* an important finding.
    """)


# ==============================================================================
# PAGE 4: IMPROVE PHASE
# ==============================================================================
def show_improve_phase():
    st.title("⚙️ Improve Phase: Discovering and Implementing Solutions")
    st.markdown("""
    **Objective:** To identify, test, and implement solutions that address the validated root causes. This involves moving from analysis to action, finding the optimal settings for critical process inputs (X's), and proactively mitigating risks.
    """)
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Finding Optimal Process Settings")
        st.markdown("Once we know which X's are critical, we need to find their optimal settings to maximize our Y.")
        tab1, tab2 = st.tabs(["🧪 Classical: Design of Experiments (DOE)", "🤖 ML Augmentation: Bayesian Optimization"])
        with tab1:
            st.markdown("##### **Classical: Design of Experiments (DOE)**")
            st.info("""
            A structured statistical method for efficiently changing multiple inputs simultaneously to determine their main and interaction effects on the output.
            - **Function:** The gold standard for physical experimentation. Establishes causality with statistical rigor.
            - **Limitation:** Suffers from the "curse of dimensionality." The number of runs required grows exponentially with the number of factors, making it impractical for high-dimensional problems.
            """)
            doe_data = generate_doe_data()
            fig_doe_main, fig_doe_interaction = plot_doe_effects(doe_data)
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(plot_doe_cube(doe_data), use_container_width=True)
            with col2:
                st.plotly_chart(fig_doe_main, use_container_width=True)
                st.plotly_chart(fig_doe_interaction, use_container_width=True)
        with tab2:
            st.markdown("##### **ML Augmentation: Bayesian Optimization**")
            st.info("""
            An intelligent search algorithm for finding the global optimum of an expensive-to-evaluate function (e.g., a multi-day simulation, a costly physical experiment, or tuning a complex ML model). It builds a probabilistic model of the objective function and uses an acquisition function (like UCB, shown here) to decide where to sample next.
            - **Function:** Extremely sample-efficient, especially in high-dimensional spaces. Intelligently balances exploiting known good solutions and exploring uncertain areas.
            - **Limitation:** Can be sensitive to kernel/hyperparameter choices and may struggle with very noisy or discontinuous functions.
            """)
            st.sidebar.header("Bayesian Opt. Simulator")
            st.sidebar.markdown("Let the algorithm intelligently choose the next best point to sample to find the global maximum.")
            @st.cache_data 
            def true_func(x): return (np.sin(x * 0.8) * 15) + (np.cos(x * 2.5)) * 5 - (x/10)**3
            x_range = np.linspace(0, 20, 400)
            if 'sampled_points' not in st.session_state: st.session_state.sampled_points = {'x': [2.0, 18.0], 'y': [true_func(2.0), true_func(18.0)]}
            if st.sidebar.button("Sample Next Best Point", key='bo_sample'): 
                _, next_point = plot_bayesian_optimization_interactive(true_func, x_range, st.session_state.sampled_points)
                st.session_state.sampled_points['x'].append(next_point)
                st.session_state.sampled_points['y'].append(true_func(next_point))
            if st.sidebar.button("Reset Simulation", key='bo_reset'): 
                st.session_state.sampled_points = {'x': [2.0, 18.0], 'y': [true_func(2.0), true_func(18.0)]}
            fig_bo, _ = plot_bayesian_optimization_interactive(true_func, x_range, st.session_state.sampled_points)
            st.plotly_chart(fig_bo, use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Proactively Mitigating Risks")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### **Classical: FMEA**")
            st.info("""
            Failure Mode and Effects Analysis is a systematic, team-based risk assessment. It involves brainstorming failure modes and ranking them by a **Risk Priority Number (RPN = Severity × Occurrence × Detection)**.
            - **Function:** A powerful, structured way to force a team to think about what *could* go wrong and prioritize preventative actions.
            - **Limitation:** The RPN scores are subjective and can be prone to team bias. It is a qualitative, not quantitative, tool.
            """)
            st.plotly_chart(plot_fmea_table(), use_container_width=True)
        with col4:
            st.markdown("##### **ML Augmentation: Prognostics & Health Management (PHM)**")
            st.info("""
            A data-driven approach where ML models are trained on sensor data (vibration, temperature, etc.) to predict equipment degradation and estimate its **Remaining Useful Life (RUL)** before a failure occurs.
            - **Function:** Moves risk management from a qualitative exercise to a quantitative, predictive capability, enabling condition-based maintenance.
            - **Limitation:** Requires high-quality, high-frequency sensor data, including run-to-failure examples, which can be difficult or expensive to obtain.
            """)
            st.plotly_chart(plot_rul_prediction(generate_sensor_degradation_data()), use_container_width=True)

    st.success("""
    **🏆 Hybrid Strategy for the Improve Phase:**
    1.  **Optimize with the Right Tool:** For physical processes with few (<7) variables where experiments are feasible, **DOE** is the unparalleled gold standard for establishing causality and finding optima. For complex digital systems, simulations, or when experiments are prohibitively expensive/slow, use **Bayesian Optimization** for its superior sample efficiency.
    2.  **Mitigate Risks (Hybrid):** Use a classical **FMEA** to identify the highest-risk failure modes based on team expertise. For the top risks identified, investigate if sensor data is available to build a **PHM/RUL model**. This turns a qualitative risk into a quantitative, predictive control.
    3.  **The Ultimate Hybrid ("Digital Twin"):** Use data from a **DOE** to train a highly accurate ML model of your process (a "surrogate model" or "digital twin"). Then, use **Bayesian Optimization** on this fast, cheap digital twin to find the global optimum virtually before performing one final confirmation run in the real world. This combines the rigor of DOE with the efficiency of Bayesian search.
    """)

# ==============================================================================
# PAGE 5: CONTROL PHASE
# ==============================================================================
def show_control_phase():
    st.title("📡 Control Phase: Sustaining and Monitoring the Gains")
    st.markdown("""
    **Objective:** To implement a robust system to monitor the improved process, ensuring it remains stable and that improvements are sustained. This involves creating a formal Control Plan and moving from reactive problem-solving to proactive, real-time process management.
    """)
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Monitoring for Stability: Statistical Process Control (SPC)")
        st.markdown("Control charts are the primary tool for monitoring process stability, distinguishing between 'common cause' (natural) variation and 'special cause' (assignable) variation that requires investigation.")
        st.sidebar.header("Control Chart Simulator")
        st.sidebar.markdown("Introduce a process shift and see which chart detects it faster.")
        shift_mag = st.sidebar.slider("Magnitude of Shift (in Std Devs)", 0.2, 3.0, 1.0, 0.1, key="ctrl_shift_mag")
        ewma_lambda = st.sidebar.slider("EWMA Lambda (λ)", 0.1, 0.5, 0.2, 0.05, help="Higher λ reacts faster but is more sensitive to noise.")
        chart_data = generate_control_chart_data(shift_point=75, shift_magnitude=shift_mag)

        tab1, tab2, tab3 = st.tabs(["📊 Classical: Shewhart Chart", "📈 Advanced Classical: EWMA/CUSUM", "🤖 ML Augmentation: Multivariate Control"])
        with tab1:
            st.markdown("##### **Classical: Shewhart Chart (X-bar)**")
            st.info("""
            The standard SPC chart, plotting data over time with control limits at ±3σ. Each point is treated independently.
            - **Strength:** Simple, robust, and excellent for detecting large (>1.5σ) shifts quickly.
            - **Limitation:** Slow to detect small, sustained shifts, as it has no 'memory' of past data points.
            """)
            st.plotly_chart(plot_shewhart_chart(chart_data), use_container_width=True)
        with tab2:
            st.markdown("##### **Advanced Classical: EWMA & CUSUM Charts**")
            st.info("""
            These charts have 'memory,' making them highly effective at detecting small, sustained shifts that Shewhart charts would miss. **EWMA** gives more weight to recent data, while **CUSUM** accumulates deviations from the target.
            - **Strength:** Significantly more sensitive to small shifts than a Shewhart chart.
            - **Limitation:** More complex to set up; parameter choice (λ for EWMA, k for CUSUM) is a trade-off between sensitivity and false alarms.
            """)
            st.plotly_chart(plot_ewma_chart(chart_data, lambda_val=ewma_lambda), use_container_width=True)
        with tab3:
            st.markdown("##### **ML Augmentation: Multivariate & Anomaly Detection**")
            st.info("""
            While classical SPC monitors one variable at a time, ML can monitor the 'health' of the entire system at once.
            - **Hotelling's T² Chart:** A statistical method for monitoring two or more correlated variables, plotting a single statistic that represents their joint deviation.
            - **Autoencoders / LSTMs (Advanced):** Unsupervised deep learning models that learn the normal 'heartbeat' of a process across all sensor inputs and flag any pattern that deviates from this learned norm.
            - **Strength:** Detects subtle, multivariate drifts that individual SPC charts would miss.
            """)
            st.plotly_chart(plot_hotelling_t2_chart(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Formalizing the Gains: The Control Plan")
        st.info("""
        A living document that details the methods, responsibilities, and reaction plan for maintaining control of the improved process. It operationalizes the findings of the entire DMAIC project and is the key to sustaining gains.
        """)
        st.plotly_chart(plot_control_plan(), use_container_width=True)

    st.success("""
    **🏆 Hybrid Strategy for the Control Phase:**
    1.  **Monitor the Output (Y) with Shewhart:** Keep a classical **Shewhart chart** on your final Critical-to-Quality (CTQ) output for simple, robust, and easily interpretable monitoring for stakeholders.
    2.  **Monitor Key Inputs (X's) with Advanced SPC:** For the critical input variables (X's) identified in the Analyze phase, use more sensitive charts like **EWMA** or **CUSUM** to detect small drifts *before* they impact the final output.
    3.  **Create an Early Warning System (ML):** For the most complex, high-stakes processes, deploy a **multivariate ML model** (like Hotelling's T² or an Autoencoder) that monitors all sensor inputs simultaneously. This provides a holistic 'process health' score as an early warning system.
    4.  **Codify Everything in a Control Plan:** The plan must document which charts are used for which variables, the control limits, the measurement frequency, and the exact, pre-defined reaction plan for any out-of-control signal from any chart.
    """)


# ==============================================================================
# PAGE 6: COMPARISON MATRIX
# ==============================================================================
def show_comparison_matrix():
    st.title("⚔️ Head-to-Head: Classical Statistics vs. Machine Learning")
    st.markdown("A visual comparison of the core philosophies and practical strengths of the two approaches, tailored for specific process improvement tasks.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("Strengths Profile: A Multi-Dimensional View")
        st.markdown("This radar chart compares the two methodologies across key attributes. The different 'shapes' of their capabilities highlight their complementary nature.")
        st.plotly_chart(plot_comparison_radar(), use_container_width=True)

    with st.container(border=True):
        st.subheader("The Verdict: Which Approach Excels for Which Task?")
        st.markdown("This chart provides a clear, decisive verdict for common use cases, visualizing which approach is generally superior.")
        st.plotly_chart(plot_verdict_barchart(), use_container_width=True)


# ==============================================================================
# PAGE 7: HYBRID STRATEGY
# ==============================================================================
def show_hybrid_strategy():
    st.title("🤝 The Hybrid Manifesto: The Future of Process Excellence")
    st.markdown("The most competitive organizations do not choose one over the other; they build an **AI-Augmented Six Sigma** program that fuses statistical rigor with machine learning's predictive power.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("The Philosophy of Synergy: Inference + Prediction")
        st.markdown("""
        Neither methodology is a silver bullet. True power lies in their integration. Classical statistics provides the **rigor for inference and causality**, while machine learning provides the **power for prediction and scale**.
        """)
        st.plotly_chart(plot_synergy_diagram(), use_container_width=True)

    with st.container(border=True):
        st.subheader("Interactive Solution Recommender")
        st.info("💡 Select a common business scenario to see the recommended hybrid approach and expert rationale.")
        guidance_data = get_guidance_data()
        scenarios = list(guidance_data.keys())
        selected_scenario = st.selectbox("Choose your scenario:", scenarios, label_visibility="collapsed")
        if selected_scenario:
            recommendation = guidance_data[selected_scenario]['approach']
            rationale = guidance_data[selected_scenario]['rationale']
            st.markdown(f"##### Recommended Approach: {recommendation}")
            st.markdown(f"**Rationale:** {rationale}")

    with st.container(border=True):
        st.subheader("A Unified, Modern DMAIC Workflow")
        st.markdown("This workflow demonstrates how to embed ML augmentation at each step of the traditional DMAIC cycle, creating a powerful, integrated methodology.")
        st.markdown(get_workflow_css(), unsafe_allow_html=True)
        st.markdown('<div class="workflow-container">', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="🌀 1. Define", phase_class="step-define",
            classical_tools=["Project Charter", "SIPOC", "Kano Model", "CTQ Tree"],
            ml_tools=["NLP for VOC at Scale", "Causal Discovery", "Automated Customer Segmentation"]), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="🔬 2. Measure", phase_class="step-measure",
            classical_tools=["Gage R&R (MSA)", "Process Capability (Cp, Cpk)", "Value Stream Mapping"],
            ml_tools=["Process Mining", "Kernel Density Estimation (KDE)", "Uncertainty Quantification"]), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="📈 3. Analyze", phase_class="step-analyze",
            classical_tools=["Hypothesis Testing (ANOVA)", "Pareto Analysis", "Fishbone Diagram", "Linear Regression"],
            ml_tools=["Feature Importance (XAI/SHAP)", "Ensemble Models", "Permutation Testing"]), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="⚙️ 4. Improve", phase_class="step-improve",
            classical_tools=["Design of Experiments (DOE)", "FMEA", "Pilot Testing"],
            ml_tools=["Bayesian Optimization", "Prognostics (PHM/RUL)", "Simulation & Digital Twins"]), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="📡 5. Control", phase_class="step-control",
            classical_tools=["Control Charts (SPC, EWMA)", "Control Plan", "Standard Operating Procedures (SOPs)"],
            ml_tools=["Multivariate Anomaly Detection", "Real-time Predictive Control", "Automated Alerting"]), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
