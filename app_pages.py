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
# PAGE 1: DEFINE PHASE (BIOTECH & GENOMICS FOCUS)
# ==============================================================================
def show_define_phase():
    st.title("🌀 Define Phase: Characterizing the Assay & Project Goals")
    st.markdown("""
    **Objective:** To clearly articulate the scientific or clinical problem, establish the project's scope for assay development or improvement, and define what is truly **Critical to Quality (CTQ)** for the assay's performance. 
    This phase ensures the team is aligned on a tangible, scientifically valid, and valuable outcome.
    """)
    st.markdown("---")

    # --- Tool 1: Project Charter ---
    with st.container(border=True):
        st.subheader("1. The Mandate: Project Charter for Assay Improvement")
        st.markdown("""
        **What is it?** In a biotech context, the Project Charter formalizes the mission to develop a new assay or improve an existing one. It acts as a contract between the R&D team, clinical lab operations, and management, defining success in measurable scientific terms.
        
        - **Strength:** Prevents "science experiments" from endlessly expanding. It aligns the team on specific performance targets (e.g., sensitivity, specificity, CV%) and timelines.
        - **Caveat:** Must remain a living document. A breakthrough discovery or a newly identified interference in the Analyze phase may require a formal charter update.
        """)
        st.plotly_chart(plot_project_charter_visual(), use_container_width=True)

    # --- Tool 2: SIPOC & Causal Discovery ---
    with st.container(border=True):
        st.subheader("2. The Landscape: Mapping the Lab Workflow & Hypothesizing Drivers")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical Tool: SIPOC**")
            st.info("""
            **What is it?** A high-level map of the entire lab and analysis workflow, identifying **S**uppliers (of reagents/samples), **I**nputs, the lab **P**rocess, **O**utputs (data/reports), and **C**ustomers (clinicians/researchers).
            - **Strength:** Creates a shared, holistic understanding of the entire process, from sample receipt to final report, highlighting interdependencies between the wet lab and data analysis.
            - **Caveat:** It reflects the *intended* workflow. It cannot, by itself, identify the hidden loops (e.g., sample re-runs) or process deviations that Process Mining can uncover.
            """)
            st.plotly_chart(plot_sipoc_visual(), use_container_width=True)
        with col2:
            st.markdown("##### **ML Counterpart: Causal Discovery**")
            st.info("""
            **What is it?** Algorithms that analyze historical instrument data, reagent lot numbers, and run metadata to generate a graph of probable cause-and-effect relationships on assay performance.
            - **Strength:** Objectively discovers potential drivers of assay failure (e.g., a specific reagent lot correlating with low signal) that might be missed by human intuition, generating data-driven hypotheses.
            - **Caveat:** Outputs are *hypotheses*, not proof. A correlation between a reagent lot and poor performance requires a validation experiment (like a DOE) to confirm causality.
            """)
            st.graphviz_chart(plot_causal_discovery_visual())

    # --- Tool 3: VOC, CTQ Tree, Kano, and NLP ---
    with st.container(border=True):
        st.subheader("3. The Target: Translating User Needs to Assay Specifications")
        st.markdown("This is the critical translation of a clinician's or researcher's need (the 'Voice of the Customer') into quantifiable, testable assay performance metrics ('Critical to Quality').")

        tab1, tab2, tab3 = st.tabs(["📊 CTQ Tree", "💖 Kano Model for Features", "🤖 NLP for Lab Insights"])
        with tab1:
            st.markdown("""
            ##### **Classical Tool: CTQ Tree**
            **What is it?** A diagram that breaks down a high-level need (e.g., "Reliable cancer variant detection") into specific, measurable performance characteristics for the assay.
            - **Strength:** Ensures the team's technical work is directly tied to a clinical or research requirement. It translates "good" into numbers (e.g., Limit of Detection < 1% VAF).
            - **Caveat:** The initial "drivers" are assumptions. They must be validated to ensure they truly capture the end-user's primary need.
            """)
            st.graphviz_chart(plot_ctq_tree_visual())
        with tab2:
            st.markdown("""
            ##### **Classical Tool: Kano Model**
            **What is it?** A framework for prioritizing features of a medical device or software platform by their impact on user satisfaction.
            - **Strength:** Helps distinguish "must-have" features (e.g., sample traceability) from "performance" features (e.g., analysis speed) and "delighters" (e.g., an automated report generator). Prevents over-engineering of basic needs.
            - **Caveat:** Relies on structured surveys of users (e.g., clinical lab scientists), which can be time-consuming to arrange and analyze.
            """)
            st.plotly_chart(plot_kano_visual(), use_container_width=True)
        with tab3:
            st.markdown("""
            ##### **ML Counterpart: NLP on Lab Deviations & Feedback**
            **What is it?** Using algorithms to analyze unstructured text from LIMS deviation reports, support tickets, and user feedback forms to automatically identify recurring failure modes or usability issues.
            - **Strength:** Massively scalable for large labs. Can instantly quantify that "Reagent Kit A fails in high humidity" is the most common complaint this month, long before manual reviews would spot the trend.
            - **Caveat:** Quality of insight depends entirely on the quality and consistency of the text data entered by lab personnel.
            """)
            st.plotly_chart(plot_voc_treemap(), use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy for the Define Phase:**
    1.  **Mandate with a Charter:** Formalize the assay development/improvement goals, scope, and key performance indicators in a Project Charter.
    2.  **Scope with SIPOC:** Map the end-to-end process from sample to report with the entire team to build a shared understanding.
    3.  **Generate Hypotheses with NLP & Causal Discovery:** Analyze lab deviation reports with NLP to find themes. Concurrently, use Causal Discovery on instrument/run data to identify potential drivers of poor performance (e.g., specific reagent lots, temperature fluctuations).
    4.  **Translate & Prioritize with CTQ & Kano:** Use the insights to build a data-driven CTQ tree that translates user needs into measurable assay specifications. For device/software features, use Kano surveys to prioritize development.
    """)

# ==============================================================================
# PAGE 2: MEASURE PHASE
# ==============================================================================
def show_measure_phase():
    st.title("🔬 Measure Phase: Quantifying Assay Performance")
    st.markdown("""
    **Objective:** To validate the reliability of our measurement systems (e.g., pipettes, plate readers, sequencers) and then establish a robust, data-driven performance baseline for the assay. The mantra is **"if you can't measure it, you can't improve it."**
    """)
    st.markdown("---")

    # --- Tool 1: Measurement System Analysis (MSA) ---
    with st.container(border=True):
        st.subheader("1. Foundational Prerequisite: Measurement System Analysis (MSA)")
        st.warning("""
        **You cannot trust your assay data until you trust your instruments.** Before analyzing results, one must quantify how much variability comes from the measurement system itself (e.g., the plate reader, the sequencing instrument) versus the true biological/chemical process.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("##### **Classical Tool: Gage R&R**")
            st.info("""
            **What is it?** A designed experiment to assess a measurement system's **Repeatability** (variation from one operator using the same instrument) and **Reproducibility** (variation between different lab technicians using the same instrument).
            - **Strength:** The gold standard for qualifying instruments and operators, essential for CLIA/CAP and FDA compliance.
            - **Caveat:** Requires a planned experiment that consumes time and expensive reagents.
            """)
        with col2:
            st.plotly_chart(plot_gage_rr_variance_components(), use_container_width=True)

    # --- Tool 2: Process Mapping ---
    with st.container(border=True):
        st.subheader("2. Understanding the Lab Workflow")
        tab1, tab2 = st.tabs(["🗺️ Value Stream Mapping (VSM)", "🤖 Process Mining"])
        with tab1:
            st.markdown("##### **Classical Tool: Value Stream Mapping (VSM)**")
            st.info("""
            **What is it?** A detailed flowchart of the lab workflow, documenting every step from sample accessioning to final report. It captures metrics like hands-on time, wait time (e.g., incubation, sequencing run), and identifies value-added vs. non-value-added steps.
            - **Strength:** Forces the team to physically walk through the lab workflow ("gemba walk"), building consensus on bottlenecks and sources of waste (e.g., waiting for a centrifuge).
            - **Caveat:** A manual snapshot of the *intended* process. It struggles to capture the complexity of parallel runs, batching, and unexpected rework loops.
            """)
            st.plotly_chart(plot_vsm(), use_container_width=True)
        with tab2:
            st.markdown("##### **ML Counterpart: Process Mining**")
            st.info("""
            **What is it?** Algorithms that automatically discover the *real* lab workflow by analyzing event logs from a Laboratory Information Management System (LIMS).
            - **Strength:** Objectively discovers how samples *actually* flow through the lab, highlighting all unexpected deviations, true bottlenecks (e.g., the QC review step), and costly rework loops that are often invisible to management.
            - **Caveat:** Requires a well-structured LIMS with clean data, including a unique **Sample ID (Case ID)**, **Process Step (Activity)**, and a **Timestamp**.
            """)
            st.graphviz_chart(plot_process_mining_graph())

    # --- Tool 3: Process Capability ---
    with st.container(border=True):
        st.subheader("3. Baselining Assay Capability")
        st.markdown("Capability analysis answers: **Is our assay capable of consistently meeting its performance specifications?** (e.g., delivering a positive control value within ±3 standard deviations).")
        
        st.sidebar.header("Assay Capability Simulator")
        st.sidebar.markdown("Adjust the assay's performance to see its capability against QC specifications.")
        lsl = st.sidebar.slider("Lower Spec Limit (LSL)", 80.0, 95.0, 90.0, key="m_lsl")
        usl = st.sidebar.slider("Upper Spec Limit (USL)", 105.0, 120.0, 110.0, key="m_usl")
        process_mean = st.sidebar.slider("Assay Mean (μ)", 95.0, 105.0, 101.5, key="m_mean")
        process_std = st.sidebar.slider("Assay Std Dev (σ)", 0.5, 5.0, 2.0, key="m_std")
        
        col3, col4 = st.columns([1, 2])
        with col3:
            st.markdown("##### **Classical: Cp & Cpk**")
            st.info("Industry-standard indices that summarize an assay's capability, assuming the output (e.g., fluorescence units) is normally distributed.")
            data = generate_assay_data(process_mean, process_std, 1000, lsl, usl)
            fig_cap, cp, cpk = plot_capability_analysis_pro(data, lsl, usl)
            st.metric("Process Potential (Cp)", f"{cp:.2f}")
            st.metric("Process Capability (Cpk)", f"{cpk:.2f}")
            if cpk < 1.0: st.error("Assay is not capable.", icon="🚨")
            elif cpk < 1.33: st.warning("Assay is marginal.", icon="⚠️")
            else: st.success("Assay is capable.", icon="✅")
        with col4:
            st.markdown("##### **ML: Distributional View**")
            st.info("Cpk can be misleading if the data isn't perfectly normal. **Kernel Density Estimation (KDE)** visualizes the *true* distribution of assay results, revealing skewness or bimodality (e.g., due to a faulty instrument channel) that a single Cpk value would hide.")
            st.plotly_chart(fig_cap, use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy for the Measure Phase:**
    1.  **Validate with Gage R&R:** First, qualify all critical instruments and operators with a Gage R&R study to ensure trustworthy data.
    2.  **Discover with Process Mining:** Analyze LIMS data with Process Mining to get an objective map of the lab's true workflow and identify systemic bottlenecks.
    3.  **Detail with VSM:** Use the insights from Process Mining to guide a targeted VSM of the most problematic workflow segment (e.g., the library preparation process).
    4.  **Baseline with Cpk, Diagnose with KDE:** Report the Cpk of key QC metrics. Use the KDE plot to diagnose the underlying cause of poor capability (e.g., a shift in instrument calibration, an increase in reagent variability, or a non-normal failure mode).
    """)


# ==============================================================================
# PAGE 3: ANALYZE PHASE
# ==============================================================================
def show_analyze_phase():
    st.title("📈 Analyze Phase: Discovering Root Causes of Assay Variation")
    st.markdown("""
    **Objective:** To analyze data to identify, validate, and quantify the root cause(s) of assay failure or high variability. This is the core scientific investigation, moving from *what* is failing to *why* it is failing.
    """)
    st.markdown("---")
    
    # --- Tool 1: Qualitative Analysis ---
    with st.container(border=True):
        st.subheader("1. Structuring the Brainstorm: Qualitative Root Cause Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical Tool: Fishbone (Ishikawa) Diagram**")
            st.info("""
            **What is it?** A structured brainstorming tool used by lab teams to visually organize potential causes of an assay problem (e.g., "Low Sequencing Yield"). Causes are grouped into categories like Reagents, Instruments, Methods, Personnel, etc.
            - **Strength:** Promotes comprehensive, systematic thinking, ensuring no stone is unturned.
            - **Caveat:** Generates *hypotheses* (e.g., "Lot #123 of Polymerase is bad") which must then be proven with data.
            """)
            st.graphviz_chart(plot_fishbone_diagram())
        with col2:
            st.markdown("##### **Classical Tool: Pareto Chart**")
            st.info("""
            **What is it?** A bar chart that identifies the most frequent failure modes recorded in LIMS or deviation logs, visualizing the "80/20 Rule."
            - **Strength:** Helps the team focus its limited resources on the "vital few" failure modes (e.g., "Library Prep Failure") that cause the majority of problems.
            - **Caveat:** Tells you *what* is failing most often, but not *why*.
            """)
            st.plotly_chart(plot_pareto_chart(), use_container_width=True)

    # --- Tool 2: Comparing Groups ---
    with st.container(border=True):
        st.subheader("2. Proving the Difference: Comparing Reagents, Lots, or Operators")
        st.markdown("Once you hypothesize that a factor is a problem (e.g., 'Reagent Lot B is worse than Lot A'), you need statistical proof.")

        st.sidebar.header("Reagent Lot Simulator")
        st.sidebar.markdown("Adjust the mean Signal-to-Noise of three reagent lots to see if the difference is statistically significant.")
        mean1 = st.sidebar.slider("Reagent Lot A Mean S/N", 8.0, 12.0, 9.5, 0.1, key='a1')
        mean2 = st.sidebar.slider("Reagent Lot B Mean S/N", 8.0, 12.0, 10.0, 0.1, key='a2')
        mean3 = st.sidebar.slider("Reagent Lot C Mean S/N", 8.0, 12.0, 10.5, 0.1, key='a3')
        anova_data = generate_reagent_lot_data(means=[mean1, mean2, mean3], stds=[0.5, 0.5, 0.5], n=50)

        tab1, tab2 = st.tabs(["🔬 Classical: ANOVA", "💻 ML: Permutation Testing"])
        with tab1:
            st.markdown("##### **Classical: ANOVA**")
            st.info("""
            **What is it?** A statistical test to determine if a significant difference exists between the mean performance of two or more groups (e.g., different reagent lots, different operators, different instruments).
            - **Strength:** The standard, rigorous method for such comparisons, universally accepted by regulatory bodies.
            - **Caveat:** Assumes data from each group is normally distributed with equal variances.
            """)
            fig_anova, p_val = plot_anova_groups(anova_data)
            st.plotly_chart(fig_anova, use_container_width=True)
            if p_val < 0.05: st.error(f"P-value is {p_val:.4f}. Reject the null hypothesis: a significant difference exists between lots.", icon="🚨")
            else: st.success(f"P-value is {p_val:.4f}. Fail to reject the null hypothesis: no significant difference detected.", icon="✅")
        with tab2:
            st.markdown("##### **ML Counterpart: Permutation Testing**")
            st.info("""
            **What is it?** A distribution-free computational method. It shuffles the group labels (e.g., which result came from which lot) thousands of times to see how often a difference as large as the one observed occurs by pure chance.
            - **Strength:** Makes no assumptions about data distribution, making it more robust for complex biological data that may not be normal.
            - **Caveat:** Can be computationally intensive.
            """)
            st.plotly_chart(plot_permutation_test(anova_data), use_container_width=True)

    # --- Tool 3: Finding Relationships ---
    with st.container(border=True):
        st.subheader("3. Finding Drivers: Modeling Assay Input-Output Relationships")
        st.markdown("This is the core of assay optimization: finding the specific protocol parameters (X's) that drive performance (Y).")
        df_reg = generate_pcr_optimization_data()
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### **Classical: Linear Regression**")
            st.info("Models the *linear* relationship between protocol parameters (e.g., temperature, time) and an assay output (e.g., fluorescence). It is simple and interpretable but often fails to capture complex biological interactions.")
            fig_reg, _, _ = plot_regression_comparison_pro(df_reg)
            st.plotly_chart(fig_reg, use_container_width=True)
        with col4:
            st.markdown("##### **ML: Ensemble Models & Explainability (SHAP)**")
            st.info("Ensemble models like **Random Forest** excel at capturing complex, non-linear biological relationships (e.g., an optimal temperature window). **SHAP** is then used to explain *which* parameters the accurate 'black box' model found most important.")
            _, model, X_reg = plot_regression_comparison_pro(df_reg)
            fig_shap = plot_shap_summary(model, X_reg)
            st.plotly_chart(fig_shap, use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy for the Analyze Phase:**
    1.  **Structure with Fishbone & Pareto:** Brainstorm potential causes for assay failure using a Fishbone Diagram. Use a Pareto chart of LIMS deviation logs to prioritize which failure modes to investigate first.
    2.  **Verify with ANOVA (or Permutation):** To test hypotheses like "Reagent Lot B is bad," use ANOVA as the standard. If the data is not normally distributed, a permutation test is a more robust alternative.
    3.  **Model with Both, Trust the Best:** Fit both a Linear Regression and a Random Forest model to your experimental data. If the Random Forest model is significantly more accurate (higher R²), trust its feature importances (from SHAP) to identify the true drivers of assay performance.
    """)


# ==============================================================================
# PAGE 4: IMPROVE PHASE
# ==============================================================================
def show_improve_phase():
    st.title("⚙️ Improve Phase: Optimizing Protocols and Mitigating Risks")
    st.markdown("""
    **Objective:** To identify, test, and implement solutions that address the root causes of poor assay performance. This involves finding the optimal settings for critical protocol steps (X's) and proactively mitigating future failure modes.
    """)
    st.markdown("---")
    
    # --- Tool 1: Finding Optimal Settings ---
    with st.container(border=True):
        st.subheader("1. Finding Optimal Assay/Protocol Settings")
        st.markdown("Once we know which parameters (e.g., temperature, concentrations, times) are critical, we need to find their optimal settings to maximize performance.")

        tab1, tab2 = st.tabs(["🧪 Classical: Design of Experiments (DOE)", "🤖 ML: Bayesian Optimization"])
        with tab1:
            st.markdown("##### **Classical: Design of Experiments (DOE)**")
            st.info("""
            **What is it?** A structured statistical method for efficiently exploring a parameter space. It's the gold standard for optimizing wet-lab protocols, like finding the best combination of annealing temperature, enzyme concentration, and incubation time.
            - **Strength:** Statistically rigorous, can separate main effects from interaction effects, and is the most reliable way to establish causality in a physical experiment.
            - **Caveat:** The number of required experiments grows rapidly with the number of factors, making it challenging for optimizing more than ~5-7 parameters at once.
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
            **What is it?** An intelligent search algorithm ideal for optimizing processes that are very expensive to test, like tuning a complex bioinformatics pipeline or a multi-day cell culture protocol.
            - **Strength:** Extremely sample-efficient. It uses a model to intelligently decide the next-best experiment to run, reaching the optimum with far fewer runs than a DOE.
            - **Caveat:** Less standardized than DOE. Can be sensitive to initial parameters and may struggle with highly stochastic (random) biological systems.
            """)
            st.sidebar.header("Bayesian Opt. Simulator")
            st.sidebar.markdown("Click the button to let the algorithm intelligently choose the next best protocol setting to test.")
            @st.cache_data 
            def true_func(x): return (np.sin(x * 0.8) * 15) + (np.cos(x * 2.5)) * 5 - (x/10)**3
            x_range = np.linspace(0, 20, 200)
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
            **What is it?** Failure Mode and Effects Analysis is a structured, team-based risk assessment for a lab protocol. The team brainstorms potential failures (e.g., "PCR Contamination"), their effects ("False Positive Result"), and causes ("Improper Pipette Technique"), then ranks them by a **Risk Priority Number (RPN)**.
            - **Strength:** A powerful, systematic way to force a team to think about what *could* go wrong and prioritize preventative actions (e.g., dedicated pipette sets, improved training).
            - **Caveat:** The RPN scores are subjective and based on team consensus, not always on hard data.
            """)
            st.plotly_chart(plot_fmea_table(), use_container_width=True)
        with col4:
            st.markdown("##### **ML: Prognostics & Health Management (PHM)**")
            st.info("""
            **What is it?** A data-driven approach where ML models are trained on instrument sensor data (e.g., laser power, temperature logs, fluidics pressure) to predict degradation and estimate the **Remaining Useful Life (RUL)** of a component before it fails and ruins a run.
            - **Strength:** Moves instrument maintenance from a fixed schedule to a predictive, condition-based schedule, preventing catastrophic failures.
            - **Caveat:** Requires high-quality, high-frequency sensor data from instruments, including run-to-failure examples, which can be difficult to obtain.
            """)
            st.plotly_chart(plot_rul_prediction(generate_sensor_degradation_data()), use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy for the Improve Phase:**
    1.  **Optimize with the Right Tool:** For wet-lab protocols with few variables (e.g., optimizing a 3-parameter PCR), use **DOE**. For highly complex, expensive processes (e.g., tuning a 10+ parameter bioinformatics pipeline), use **Bayesian Optimization**.
    2.  **Mitigate with FMEA, Predict with PHM:** Use a classical FMEA to identify the highest-risk failure modes in your protocol. For the top risks related to instrumentation, investigate if sensor data is available to build a PHM/RUL model for predictive maintenance.
    3.  **The Ultimate Hybrid (In-Silico Optimization):** Use DOE data to train an accurate ML model of your assay (a "digital twin"). Then, use Bayesian Optimization on this digital twin to explore thousands of parameter settings *in-silico* before a final, small confirmation experiment in the wet lab.
    """)

# ==============================================================================
# PAGE 5: CONTROL PHASE
# ==============================================================================
def show_control_phase():
    st.title("📡 Control Phase: Sustaining and Monitoring Assay Performance")
    st.markdown("""
    **Objective:** To implement a robust system to monitor the improved assay, ensuring it remains stable and that performance gains are sustained. This involves creating a QC and Control Plan to move from reactive troubleshooting to proactive process management.
    """)
    st.markdown("---")

    # --- Tool 1: Control Charts ---
    with st.container(border=True):
        st.subheader("1. Monitoring for Stability: Statistical Process Control (SPC) for QC")
        st.markdown("Control charts are the primary tool for monitoring the stability of an assay over time using daily controls, distinguishing between 'common cause' (expected assay noise) and 'special cause' (e.g., instrument drift) variation.")
        
        st.sidebar.header("QC Simulator")
        st.sidebar.markdown("Introduce a small, sustained shift in the positive control and see which chart detects it faster.")
        shift_mag = st.sidebar.slider("Magnitude of Shift (in Std Devs)", 0.2, 3.0, 1.0, 0.1, key="ctrl_shift_mag")
        ewma_lambda = st.sidebar.slider("EWMA Lambda (λ)", 0.1, 0.5, 0.2, 0.05, help="Higher λ reacts faster but is more sensitive to noise.")
        chart_data = generate_qc_control_data(shift_point=75, shift_magnitude=shift_mag)

        tab1, tab2, tab3 = st.tabs(["📊 Classical: Levey-Jennings Chart", "📈 Advanced Classical: EWMA/CUSUM", "🤖 ML: Multivariate Control"])
        with tab1:
            st.markdown("##### **Classical: Levey-Jennings Chart (Shewhart Chart)**")
            st.info("""
            **What is it?** The standard QC chart in clinical labs. It plots daily QC results over time with control limits at ±2 and ±3 standard deviations.
            - **Strength:** Simple, robust, and excellent for detecting large shifts in assay performance. A regulatory standard.
            - **Limitation:** Slow to detect small, sustained drifts (e.g., a slowly degrading laser), as it has no 'memory'.
            """)
            st.plotly_chart(plot_shewhart_chart(chart_data), use_container_width=True)
        with tab2:
            st.markdown("##### **Advanced Classical: EWMA & CUSUM Charts**")
            st.info("""
            **What are they?** These charts have 'memory', making them highly effective at detecting small, sustained drifts in assay performance before they breach standard QC limits.
            - **Strength:** Can provide early warning of instrument calibration drift or slow reagent degradation.
            - **Limitation:** More complex to set up; parameters are a trade-off between sensitivity and false alarms.
            """)
            st.plotly_chart(plot_ewma_chart(chart_data, lambda_val=ewma_lambda), use_container_width=True)
            st.plotly_chart(plot_cusum_chart(chart_data), use_container_width=True)
        with tab3:
            st.markdown("##### **ML: Multivariate Control**")
            st.info("""
            **What is it?** While classical SPC monitors one QC metric at a time, ML models (like **Hotelling's T²**) can monitor the health of the entire system at once by considering correlations between multiple QC parameters (e.g., signal intensity, background noise, peak shape).
            - **Strength:** Detects subtle, correlated drifts across multiple QC metrics that individual charts would miss.
            - **Caveat:** Can be a 'black box'. When an anomaly is flagged, it requires further analysis to identify the root cause parameter.
            """)
            st.plotly_chart(plot_hotelling_t2_chart(), use_container_width=True)

    # --- Tool 2: The Control Plan ---
    with st.container(border=True):
        st.subheader("2. Formalizing the Gains: The Control Plan")
        st.info("""
        **What is it?** A living document detailing the QC methods, responsibilities, and reaction plan for the assay. It's the "Standard Operating Procedure" (SOP) that ensures the improved performance is sustained.
        - **Strength:** Provides a clear, actionable plan for the clinical lab team, required for regulatory compliance (CLIA/CAP/FDA).
        - **Caveat:** Must be a living document, updated whenever a significant process change is validated.
        """)
        st.plotly_chart(plot_control_plan(), use_container_width=True)

    st.success("""
    **🏆 Verdict & Hybrid Strategy for the Control Phase:**
    1.  **Monitor Primary CTQs with Levey-Jennings:** Maintain standard Levey-Jennings charts for primary, reportable QC metrics for compliance and simple monitoring.
    2.  **Monitor Critical Parameters with Advanced SPC:** For critical underlying parameters (e.g., background fluorescence, sequencing cluster density), use more sensitive **EWMA** or **CUSUM** charts to get early warnings of drift.
    3.  **Create an Early Warning System with ML:** For high-throughput, automated systems, deploy a multivariate ML model to monitor the entire system's 'health signature'. This can predict failures hours or days in advance.
    4.  **Codify Everything in a Control Plan:** The plan must document which charts are used, QC materials, frequencies, control limits, and the exact out-of-control action plan (OOCAP).
    """)


# ==============================================================================
# PAGE 6: COMPARISON MATRIX (VISUALLY ENHANCED)
# ==============================================================================
def show_comparison_matrix():
    st.title("⚔️ Head-to-Head: Classical Stats vs. Machine Learning")
    st.markdown("A visual comparison of the core philosophies and practical strengths of the two approaches in a biotech context.")
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
        This chart visualizes which approach is generally superior for specific, common tasks in assay development and process improvement. 
        The direction and color of the bar indicate the winner, providing a clear, decisive verdict for each use case.
        """)
        st.plotly_chart(plot_verdict_barchart(), use_container_width=True)

# ==============================================================================
# PAGE 7: HYBRID STRATEGY (VISUALLY ENHANCED)
# ==============================================================================
def show_hybrid_strategy():
    st.title("🤝 The Hybrid Manifesto: The Future of Biotech Process Excellence")
    st.markdown("The most competitive biotech organizations do not choose one over the other; they build an **AI-Augmented Quality** program that fuses the statistical rigor required for validation with machine learning's predictive power for discovery and monitoring.")
    st.markdown("---")

    # --- Synergy Diagram ---
    with st.container(border=True):
        st.subheader("The Philosophy of Synergy")
        st.markdown("""
        Neither methodology is a silver bullet. The true power lies in their integration. Classical statistics provides the **rigor for inference and regulatory validation**, while machine learning provides the **power for high-dimensional discovery and proactive monitoring**.
        """)
        st.plotly_chart(plot_synergy_diagram(), use_container_width=True)
    
    # --- Interactive Recommender ---
    with st.container(border=True):
        st.subheader("Interactive Solution Recommender")
        st.info("💡 Select a common biotech R&D or operational scenario to see the recommended approach and rationale.")
        
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
        st.subheader("A Unified, Modern DMAIC Workflow for Biotech")
        st.markdown("This workflow demonstrates how to embed ML augmentation at each step of the traditional DMAIC cycle for assay development.")
        st.markdown(get_workflow_css(), unsafe_allow_html=True)
        st.markdown('<div class="workflow-container">', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="🌀 1. Define",
            phase_class="step-define",
            classical_tools=["Project Charter", "SIPOC", "Kano Model", "CTQ Tree"],
            ml_tools=["NLP for Lab Deviations", "Causal Discovery", "Literature Mining"]
        ), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="🔬 2. Measure",
            phase_class="step-measure",
            classical_tools=["Gage R&R (MSA)", "Assay Capability (Cpk)", "Value Stream Mapping"],
            ml_tools=["Process Mining from LIMS", "Kernel Density Estimation", "Image Analysis (QC)"]
        ), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="📈 3. Analyze",
            phase_class="step-analyze",
            classical_tools=["Hypothesis Testing (ANOVA)", "Pareto Analysis", "Fishbone Diagram", "Regression"],
            ml_tools=["Feature Importance (SHAP)", "Ensemble Models", "Permutation Testing"]
        ), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="⚙️ 4. Improve",
            phase_class="step-improve",
            classical_tools=["Design of Experiments (DOE)", "FMEA", "Pilot Validation"],
            ml_tools=["Bayesian Optimization", "Prognostics (PHM)", "In-Silico Simulation"]
        ), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="📡 5. Control",
            phase_class="step-control",
            classical_tools=["Levey-Jennings Charts", "Control Plan (SOP)", "QC Trending"],
            ml_tools=["Multivariate Anomaly Detection", "Predictive QC", "Automated Alerting"]
        ), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
