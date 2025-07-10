# app_pages.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Import all necessary helper functions from the single helper file.
from app_helpers import *

# ==============================================================================
# PAGE 0: WELCOME & FRAMEWORK
# ==============================================================================
def show_welcome_page():
    st.title("🧬 Welcome to the Bio-AI Excellence Framework")
    st.markdown("##### An interactive playbook for developing and optimizing robust genomic assays and devices.")
    st.markdown("---")

    st.info("""
    **This application is designed for a technically proficient audience** (e.g., R&D Scientists, Bioinformaticians, Lab Directors).
    It moves beyond introductory concepts to demonstrate a powerful, unified framework that fuses the **inferential rigor of classical Design of Experiments (DOE)** with the **predictive power of modern Machine Learning and Bioinformatics**.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classical Assay Development")
        st.markdown("""
        The gold standard for life sciences R&D, built on a foundation of statistical inference, hypothesis testing, and controlled experiments.
        - **Core Strength:** Establishing causality, understanding main effects and interactions, and ensuring statistical rigor for regulatory submissions (e.g., FDA).
        - **Primary Focus:** Analytical validation, optimizing key parameters, and robust characterization of assay performance.
        - **Best Suited For:** Problems with structured data, a limited number of well-understood factors, and where interpretability is paramount.
        """)

    with col2:
        st.subheader("ML & Bioinformatics Augmentation")
        st.markdown("""
        A suite of computational techniques that excel at finding patterns in high-dimensional biological data (e.g., genomics, proteomics), making predictions, and automating complex analysis pipelines.
        - **Core Strength:** Prediction, biomarker discovery, handling complexity, and extracting signals from noisy, high-dimensional data.
        - **Primary Focus:** Optimizing multi-parameter protocols, discovering novel signatures, and enabling proactive quality control.
        - **Best Suited For:** Problems with high-dimensional data (`p >> n`), non-linear interactions (epistasis), and where predictive accuracy is the key objective.
        """)

    st.subheader("The Hybrid Lab Philosophy: Augmentation, Not Replacement")
    st.markdown("""
    The most effective path to developing breakthrough diagnostics and therapies lies in the **synergistic integration** of these two disciplines.
    
    Use the navigation panel on the left to explore the R&D lifecycle (framed as **DMAIC**). Each phase will present:
    1.  **Classical Tools:** The trusted, foundational methods for life sciences.
    2.  **ML/Bio-AI Counterparts:** The modern techniques that augment and scale the classical approach.
    3.  **Hybrid Strategy:** A prescriptive guide on how to combine them for superior results in the lab.
    """)
    st.success("Click on a phase in the sidebar to begin your exploration.")


# ==============================================================================
# PAGE 1: DEFINE PHASE - CLINICAL NEED & ASSAY GOALS
# ==============================================================================
def show_define_phase():
    st.title("🌀 Define: Clinical Need & Assay Goals")
    st.markdown("""
    **Objective:** To clearly articulate the clinical problem (e.g., early cancer detection), establish the project's goals, define the scope of the assay, and translate the 'Voice of the Clinician' into quantifiable, 'Critical to Quality' (CTQ) assay performance characteristics.
    """)
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. The Mandate: Assay Design & Development Plan")
        st.markdown("The Design Plan is the foundational document, equivalent to a Project Charter. It formally defines the assay's intended use, target patient population, required performance specifications, and the overall development roadmap, aligning the scientific team with business and clinical goals.")
        st.plotly_chart(plot_project_charter_visual(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. The Landscape: Mapping the Assay Workflow & Hypotheses")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical Tool: SIPOC for Lab Workflows**")
            st.info("""
            A high-level map of the entire assay workflow, from sample receipt to data analysis.
            - **Function:** A qualitative, expert-driven tool for defining the boundaries of the process (e.g., sample prep vs. sequencing vs. bioinformatics) and fostering team alignment on all inputs and outputs.
            - **Limitation:** Cannot discover unknown sources of variability; it visualizes existing domain knowledge.
            """)
            st.plotly_chart(plot_sipoc_visual(), use_container_width=True)
        with col2:
            st.markdown("##### **ML Augmentation: Causal Discovery from Pilot Data**")
            st.info("""
            Algorithms that analyze pilot or historical experimental data to infer a graph of probable cause-and-effect relationships between protocol parameters and assay outcomes.
            - **Function:** Objectively generates data-driven hypotheses about which steps or reagents are most likely driving variability, guiding more focused experiments in later phases.
            - **Limitation:** Requires sufficient, high-quality data. It outputs a *hypothesis graph* for validation, not proven causality.
            """)
            st.graphviz_chart(plot_causal_discovery_visual())

    with st.container(border=True):
        st.subheader("3. The Target: Translating Clinical Needs into Assay Specs (CTQs)")
        st.markdown("This step translates a clinician's need (e.g., 'detect tumors earlier') into measurable assay performance characteristics.")
        tab1, tab2, tab3 = st.tabs(["📊 CTQ Tree for Assay Performance", "💖 Kano for Diagnostic Features", "🤖 NLP for Literature Review"])
        with tab1:
            st.markdown("##### **Classical Tool: CTQ Tree**")
            st.info("A decomposition tool to break down a broad clinical need (e.g., reliable liquid biopsy) into specific, measurable performance metrics like **Analytical Sensitivity (LOD), Specificity, and Precision (CV%)**.")
            st.graphviz_chart(plot_ctq_tree_visual())
        with tab2:
            st.markdown("##### **Classical Tool: Kano Model**")
            st.info("A framework for prioritizing diagnostic features. **Basic** needs might be detecting the presence of a known cancer mutation. **Performance** could be the accurate quantification of its allele frequency. An **Excitement** feature could be the unexpected discovery of a novel, actionable co-mutation.")
            st.plotly_chart(plot_kano_visual(), use_container_width=True)
        with tab3:
            st.markdown("##### **ML Augmentation: NLP for Scientific Literature**")
            st.info("Using algorithms (e.g., SciBERT, BioBERT) to analyze thousands of publications to automatically extract prevalent biomarkers, competing methodologies, and reported performance benchmarks. This massively accelerates the literature review process to inform the target product profile.")
            st.plotly_chart(plot_voc_treemap(), use_container_width=True)

    st.success("""
    **🏆 Hybrid Strategy for the Define Phase:**
    1.  **Mandate & Scope (Classical):** Begin with a formal **Assay Design Plan** and a team-based **SIPOC** of the lab workflow to establish clear boundaries and alignment.
    2.  **Discover at Scale (ML):** Deploy **NLP Topic Modeling** on scientific literature and competitor documentation to generate a data-driven list of critical biomarkers and performance benchmarks.
    3.  **Translate & Prioritize (Hybrid):** Use the outputs from the NLP analysis to build a data-grounded **CTQ Tree**, ensuring it reflects the current scientific landscape. Use the CTQ tree to define the **Target Product Profile (TPP)** with specific metrics for sensitivity, specificity, etc.
    """)


# ==============================================================================
# PAGE 2: MEASURE PHASE - ASSAY & SYSTEM VALIDATION
# ==============================================================================
def show_measure_phase():
    st.title("🔬 Measure: Assay & System Validation")
    st.markdown("""
    **Objective:** To validate the reliability of all measurement systems (pipettes, sequencers, software), collect data, and establish a robust, data-driven baseline of the assay's current performance (precision, accuracy, etc.).
    """)
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Prerequisite: Measurement System Analysis (MSA)")
        st.warning("**You cannot trust your assay data until you trust your instruments and operators.** MSA is a non-negotiable step to ensure observed variability comes from the biology, not the lab process.")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("##### **Classical Tool: Gage R&R**")
            st.info("""
            A designed experiment to partition measurement system variance into its components: **Repeatability** (e.g., variation from one sequencer on repeat runs) and **Reproducibility** (e.g., variation between different lab technicians running the same sample).
            - **Function:** Standardized method for qualifying instruments, operators, and protocols.
            """)
        with col2:
            st.plotly_chart(plot_gage_rr_variance_components(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Understanding the End-to-End Workflow")
        tab1, tab2 = st.tabs(["🗺️ Value Stream Mapping (VSM) of Lab Process", "🤖 Process Mining of LIMS Data"])
        with tab1:
            st.markdown("##### **Classical Tool: Value Stream Mapping (VSM)**")
            st.info("""
            A manual, observational flowchart of the entire lab process from sample accessioning to final report, capturing hands-on time and wait time.
            - **Function:** Excellent for identifying physical bottlenecks, sources of contamination risk, and opportunities to streamline the physical workflow.
            """)
            st.plotly_chart(plot_vsm(), use_container_width=True)
        with tab2:
            st.markdown("##### **ML Augmentation: Process Mining on LIMS Data**")
            st.info("""
            Algorithms that automatically discover the real lab workflow by analyzing event logs from a **Laboratory Information Management System (LIMS)**.
            - **Function:** Discovers the process as it *actually* happens, including all sample re-tests, QC failures, and true instrument bottlenecks that are often hidden in manual process maps.
            - **Limitation:** Requires well-structured LIMS data with a **Sample ID**, an **Activity Name**, and a **Timestamp**.
            """)
            st.graphviz_chart(plot_process_mining_graph())

    with st.container(border=True):
        st.subheader("3. Establishing Baseline Assay Capability")
        st.markdown("Capability analysis answers: **Can our assay reliably meet the required performance specifications (e.g., Limit of Detection)?**")
        st.sidebar.header("Assay Capability Simulator")
        st.sidebar.markdown("Adjust assay performance and spec limits to see the impact on capability.")
        lsl = st.sidebar.slider("Lower Spec Limit (LSL)", 0.5, 2.0, 0.8, key="m_lsl", help="e.g., Minimum required signal-to-noise")
        usl = st.sidebar.slider("Upper Spec Limit (USL)", 8.0, 10.0, 9.0, key="m_usl", help="e.g., Maximum tolerable background")
        process_mean = st.sidebar.slider("Assay Mean (μ)", 2.0, 8.0, 4.0, key="m_mean")
        process_std = st.sidebar.slider("Assay Std Dev (σ)", 0.2, 2.0, 0.5, key="m_std")

        data = generate_process_data(process_mean, process_std, 2000, lsl, usl)
        fig_cap, cp, cpk = plot_capability_analysis_pro(data, lsl, usl)

        col3, col4 = st.columns([1, 2])
        with col3:
            st.markdown("##### **Classical Indices: Cp & Cpk**")
            st.info("Industry-standard indices that summarize capability, assuming normality. A Cpk ≥ 1.33 is often a target for a robust process.")
            st.metric("Process Potential (Cp)", f"{cp:.2f}")
            st.metric("Process Capability (Cpk)", f"{cpk:.2f}")
            if cpk < 1.33: st.error("Assay is not capable.", icon="🚨")
            else: st.success("Assay is capable.", icon="✅")
        with col4:
            st.markdown("##### **ML Augmentation: Distributional View**")
            st.info("Cpk can be misleading. **Kernel Density Estimation (KDE)** visualizes the *true* shape of the assay's output distribution, revealing issues like skew or bimodality (e.g., from batch effects) that single-point indices hide.")
            st.plotly_chart(fig_cap, use_container_width=True)

    st.success("""
    **🏆 Hybrid Strategy for the Measure Phase:**
    1.  **Validate (Classical):** Always perform a **Gage R&R** on critical instruments (e.g., qPCR machine, sequencer) and assess inter-operator variability before baselining performance. This is a non-negotiable prerequisite for valid data.
    2.  **Discover (ML):** Begin by running **Process Mining** on LIMS event logs. This provides an objective map of the lab workflow, immediately highlighting QC failure loops, re-test rates, and true instrument/personnel bottlenecks.
    3.  **Detail (Classical):** Use the insights from process mining to guide a targeted, physical **VSM** exercise, focusing on areas with high wait times or rework to understand the physical-world causes.
    4.  **Baseline & Diagnose (Hybrid):** Report the official **Cpk** baseline against the TPP specifications. Internally, use the **KDE plot** to diagnose the *reason* for poor capability (e.g., a shifted mean, excessive noise, or batch effects causing bimodality).
    """)

# ==============================================================================
# PAGE 3: ANALYZE PHASE - ROOT CAUSE OF ASSAY VARIABILITY
# ==============================================================================
def show_analyze_phase():
    st.title("📈 Analyze: Root Cause of Assay Variability")
    st.markdown("""
    **Objective:** To analyze data to identify, validate, and quantify the root cause(s) of poor assay performance (e.g., low sensitivity, high CV%). This moves from *what* is failing to *why* it is failing.
    """)
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Qualitative Root Cause Analysis & Prioritization")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical Tool: Fishbone Diagram**")
            st.info("A structured brainstorming tool to organize potential causes of an assay problem (e.g., 'Low Library Yield') into categories like 'Reagents', 'Equipment', 'Technician', 'Method', 'Sample', etc.")
            st.graphviz_chart(plot_fishbone_diagram())
        with col2:
            st.markdown("##### **Classical Tool: Pareto Chart**")
            st.info("A chart to identify the 'vital few' failure modes. For an NGS assay, this could be 'Adapter-dimer formation', 'Low PCR efficiency', 'Failed QC metric', etc. This focuses effort on the highest-frequency problems.")
            st.plotly_chart(plot_pareto_chart(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Proving the Difference: Comparing Experimental Groups")
        st.markdown("Once hypotheses are formed (e.g., 'Reagent Lot B is causing lower library concentrations'), statistical proof is required.")
        st.sidebar.header("Group Comparison Simulator")
        st.sidebar.markdown("Adjust reagent lot means to see if the difference in library yield becomes statistically significant.")
        mean1 = st.sidebar.slider("Lot A Mean Yield (ng/µL)", 18.0, 22.0, 19.5, 0.1, key='a1')
        mean2 = st.sidebar.slider("Lot B Mean Yield (ng/µL)", 18.0, 22.0, 20.0, 0.1, key='a2')
        mean3 = st.sidebar.slider("Lot C Mean Yield (ng/µL)", 18.0, 22.0, 20.5, 0.1, key='a3')
        anova_data = generate_anova_data(means=[mean1, mean2, mean3], stds=[0.8, 0.8, 0.8], n=20)

        tab1, tab2 = st.tabs(["🔬 Classical: ANOVA", "💻 ML Augmentation: Permutation Testing"])
        with tab1:
            st.markdown("##### **Classical: Analysis of Variance (ANOVA)**")
            st.info("A statistical test to determine if significant differences exist between the mean yields of different reagent lots, enzyme suppliers, or technicians. Assumes normality and equal variances.")
            fig_anova, p_val = plot_anova_groups(anova_data)
            st.plotly_chart(fig_anova, use_container_width=True)
            if p_val < 0.05: st.error(f"P-value is {p_val:.4f}. Reject the null hypothesis: A statistically significant difference exists.", icon="🚨")
            else: st.success(f"P-value is {p_val:.4f}. Fail to reject null: No significant difference detected.", icon="✅")
        with tab2:
            st.markdown("##### **ML Augmentation: Permutation Testing**")
            st.info("A non-parametric, computational method that makes no assumptions about the data's distribution. It's more robust for the often non-normal, small-sample data common in biotech R&D.")
            st.plotly_chart(plot_permutation_test(anova_data), use_container_width=True)

    with st.container(border=True):
        st.subheader("3. Finding the Drivers: Modeling Assay Performance (Y = f(x))")
        st.markdown("This is the core of root cause analysis: finding which protocol parameters (X's like `Annealing Temp`, `PCR Cycles`, `Enzyme Conc.`) mathematically drive the key output (Y, e.g., `Mapping Quality` or `On-Target Rate`).")
        df_reg = generate_nonlinear_data()
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### **Classical: Multiple Linear Regression**")
            st.info("Models the linear relationship between parameters and output. Simple and interpretable, but often fails to capture the complex, non-linear biology of an assay.")
            fig_reg, _, _ = plot_regression_comparison_pro(df_reg)
            st.plotly_chart(fig_reg, use_container_width=True)
        with col4:
            st.markdown("##### **ML Augmentation: Ensemble Models & XAI (SHAP)**")
            st.info("Ensemble models (**Random Forest, Gradient Boosting**) capture complex, non-linear relationships. We then use **eXplainable AI (XAI)** tools like **SHAP** to understand the model's logic and rank parameter importance.")
            _, model, X_reg = plot_regression_comparison_pro(df_reg)
            fig_shap = plot_shap_summary(model, X_reg)
            st.plotly_chart(fig_shap, use_container_width=True)

    st.success("""
    **🏆 Hybrid Strategy for the Analyze Phase:**
    1.  **Structure & Prioritize (Classical):** Use a **Fishbone** diagram to brainstorm all potential causes for poor assay performance. Use a **Pareto** chart on QC failure data to identify which failure modes to investigate first.
    2.  **Verify Group Differences (Hybrid):** For comparing reagent lots or technicians, **ANOVA** is a good first step. However, always validate its assumptions (e.g., using a Shapiro-Wilk test). Given the small sample sizes in R&D, defaulting to a more robust **Permutation Test** is often the superior choice.
    3.  **Model Relationships (Hybrid):** Fit both a **Linear Regression** and an **Ensemble ML model**. If the ML model is significantly more accurate (check R²), its feature importance rankings from **SHAP** are a more reliable guide to the true root causes (e.g., a non-linear effect of enzyme concentration) than the coefficients from a poorly-fitting linear model.
    """)

# ==============================================================================
# PAGE 4: IMPROVE PHASE - ASSAY & WORKFLOW OPTIMIZATION
# ==============================================================================
def show_improve_phase():
    st.title("⚙️ Improve: Assay & Workflow Optimization")
    st.markdown("""
    **Objective:** To identify, test, and implement solutions that address the validated root causes. For assays, this almost always involves finding the optimal settings for critical protocol parameters to maximize performance (e.g., yield, sensitivity).
    """)
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Finding Optimal Protocol Settings")
        st.markdown("Once we know which parameters are critical (e.g., `Annealing Temp`, `Ligation Time`), we need to find their optimal settings to maximize our output (e.g., `On-Target Reads`).")
        tab1, tab2 = st.tabs(["🧪 Classical: Design of Experiments (DOE)", "🤖 ML Augmentation: Bayesian Optimization"])
        with tab1:
            st.markdown("##### **Classical: Design of Experiments (DOE)**")
            st.info("""
            A structured method for efficiently changing multiple parameters simultaneously to determine their main and interaction effects.
            - **Function:** The gold standard for physical lab experimentation, especially Response Surface Methodology (RSM) for optimization. Establishes causality with statistical rigor.
            - **Limitation:** The number of runs required grows exponentially with the number of factors, making it impractical for optimizing a high-dimensional protocol (>5-7 parameters).
            """)
            doe_data = generate_doe_data()
            fig_doe_main, fig_doe_interaction = plot_doe_effects(doe_data)
            col1, col2 = st.columns(2);
            with col1: st.plotly_chart(plot_doe_cube(doe_data), use_container_width=True)
            with col2: st.plotly_chart(fig_doe_main, use_container_width=True); st.plotly_chart(fig_doe_interaction, use_container_width=True)
        with tab2:
            st.markdown("##### **ML Augmentation: Bayesian Optimization**")
            st.info("""
            An intelligent search algorithm for finding the global optimum of an expensive-to-evaluate function (e.g., a multi-day cell culture experiment, or a full NGS run). It builds a probabilistic model of the assay's performance and uses it to intelligently select the most informative next experiment to run.
            - **Function:** Extremely sample-efficient. Ideal for optimizing high-dimensional protocols where a full factorial DOE is impossible.
            - **Limitation:** Can be sensitive to initial parameters and may struggle with very 'spiky', discontinuous response surfaces.
            """)
            st.sidebar.header("Bayesian Opt. Simulator"); st.sidebar.markdown("Let the algorithm intelligently choose the next experiment to run to find the maximum on-target rate.")
            @st.cache_data 
            def true_func(x): return (np.sin(x * 0.8) * 15) + (np.cos(x * 2.5)) * 5 - (x/10)**3
            x_range = np.linspace(0, 20, 400)
            if 'sampled_points' not in st.session_state: st.session_state.sampled_points = {'x': [2.0, 18.0], 'y': [true_func(2.0), true_func(18.0)]}
            if st.sidebar.button("Run Next Smart Experiment", key='bo_sample'): 
                _, next_point = plot_bayesian_optimization_interactive(true_func, x_range, st.session_state.sampled_points)
                st.session_state.sampled_points['x'].append(next_point); st.session_state.sampled_points['y'].append(true_func(next_point))
            if st.sidebar.button("Reset Simulation", key='bo_reset'): st.session_state.sampled_points = {'x': [2.0, 18.0], 'y': [true_func(2.0), true_func(18.0)]}
            fig_bo, _ = plot_bayesian_optimization_interactive(true_func, x_range, st.session_state.sampled_points)
            st.plotly_chart(fig_bo, use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Proactively Mitigating Risks")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### **Classical: FMEA**")
            st.info("""
            Failure Mode and Effects Analysis is a systematic, team-based risk assessment of the assay protocol. It brainstorms failure modes (e.g., 'Reagent Contamination', 'PCR Inhibition') and ranks them by a **Risk Priority Number (RPN)** to prioritize mitigation efforts.
            """)
            st.plotly_chart(plot_fmea_table(), use_container_width=True)
        with col4:
            st.markdown("##### **ML Augmentation: Predictive Instrument Maintenance**")
            st.info("""
            Using ML models trained on sensor and performance data from lab equipment (e.g., sequencer laser power, pump pressure, thermal cycler logs) to predict degradation and estimate **Remaining Useful Life (RUL)** before a failure occurs that could compromise an entire batch of expensive samples.
            """)
            st.plotly_chart(plot_rul_prediction(generate_sensor_degradation_data()), use_container_width=True)

    st.success("""
    **🏆 Hybrid Strategy for the Improve Phase:**
    1.  **Optimize with the Right Tool:** For optimizing a few (<5) well-understood parameters, **DOE (specifically Response Surface Methodology)** is the gold standard for its rigor. For optimizing a high-dimensional protocol with many interacting parameters, use **Bayesian Optimization** for its superior sample efficiency.
    2.  **Mitigate Risks (Hybrid):** Use a classical **FMEA** to identify the highest-risk failure modes in the protocol. For the top risks related to equipment, investigate if sensor data is available to build a **predictive maintenance (RUL) model**.
    3.  **The Ultimate Hybrid ("Digital Twin" of the Assay):** Use data from a space-filling **DOE** to train a highly accurate ML model of your assay (a "surrogate model"). Then, use **Bayesian Optimization** on this fast, cheap digital twin to find the global optimum *in silico* before performing one final confirmation experiment in the lab.
    """)

# ==============================================================================
# PAGE 5: CONTROL PHASE - LAB OPERATIONS & QC
# ==============================================================================
def show_control_phase():
    st.title("📡 Control: Lab Operations & QC")
    st.markdown("""
    **Objective:** To implement a robust Quality Control (QC) system to monitor the optimized assay in routine use, ensuring performance remains stable and that improvements are sustained. This involves creating a formal Control Plan and SOPs.
    """)
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Monitoring for Stability: Statistical Process Control (SPC) for QC")
        st.markdown("Control charts are used to monitor positive and negative controls over time, distinguishing between natural 'common cause' variation and 'special cause' variation that signals a problem.")
        st.sidebar.header("QC Simulator"); st.sidebar.markdown("Introduce a shift in a positive control standard and see which chart detects it faster.")
        shift_mag = st.sidebar.slider("Magnitude of Shift (in Std Devs)", 0.2, 3.0, 0.8, 0.1, key="ctrl_shift_mag")
        ewma_lambda = st.sidebar.slider("EWMA Lambda (λ)", 0.1, 0.5, 0.2, 0.05, help="Higher λ reacts faster to shifts.")
        chart_data = generate_control_chart_data(shift_point=75, shift_magnitude=shift_mag)

        tab1, tab2, tab3 = st.tabs(["📊 Classical: Levey-Jennings Chart", "📈 Advanced Classical: EWMA/CUSUM", "🤖 ML: Multivariate QC"])
        with tab1:
            st.markdown("##### **Classical: Levey-Jennings Chart (Shewhart)**")
            st.info("The standard QC chart in clinical labs. It plots QC measurements over time with control limits at ±2σ and ±3σ. It's excellent for detecting large, sudden shifts in assay performance.")
            st.plotly_chart(plot_shewhart_chart(chart_data), use_container_width=True)
        with tab2:
            st.markdown("##### **Advanced Classical: EWMA & CUSUM Charts**")
            st.info("These charts have 'memory', making them highly effective at detecting small, sustained drifts (e.g., slow reagent degradation) that Levey-Jennings charts would miss. EWMA is generally preferred for this.")
            st.plotly_chart(plot_ewma_chart(chart_data, lambda_val=ewma_lambda), use_container_width=True)
        with tab3:
            st.markdown("##### **ML Augmentation: Multivariate QC**")
            st.info("""
            An NGS assay has many correlated QC metrics (e.g., `% Mapped Reads`, `% Duplication`, `Insert Size`). ML can monitor the 'health' of the entire QC profile at once.
            - **Hotelling's T² Chart:** Monitors two or more correlated QC metrics, flagging any sample whose overall QC profile is abnormal, even if each individual metric is within its own limits.
            """)
            st.plotly_chart(plot_hotelling_t2_chart(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Formalizing the Gains: The Control Plan & SOPs")
        st.info("The Control Plan is a living document that details the QC methods, responsibilities, and reaction plan (e.g., 're-calibrate pipette', 'order new reagent lot') for any out-of-control signal. It is codified in the lab's Standard Operating Procedures (SOPs).")
        st.plotly_chart(plot_control_plan(), use_container_width=True)

    st.success("""
    **🏆 Hybrid Strategy for the Control Phase:**
    1.  **Monitor Key QC Metrics with Levey-Jennings:** Use a classical **Levey-Jennings (Shewhart) chart** for your primary positive and negative controls. It's simple, universally understood, and excellent for regulatory compliance.
    2.  **Detect Drifts with Advanced SPC:** For critical secondary metrics (e.g., reagent blanks, specific performance controls), use a more sensitive **EWMA chart** to detect small, slow drifts indicative of reagent or instrument degradation *before* it causes an out-of-spec failure.
    3.  **Holistic Sample QC with ML:** For each sample, run a **multivariate QC model** (like Hotelling's T²) on the full profile of NGS QC metrics. This provides a single, holistic quality score that can flag subtle sample-specific issues that univariate charts would miss.
    4.  **Codify Everything:** The **Control Plan** and **SOPs** must document which charts are used, their limits, measurement frequency, and the exact reaction plan for any out-of-control signal.
    """)


# ==============================================================================
# PAGE 6: METHODOLOGY COMPARISON
# ==============================================================================
def show_comparison_matrix():
    st.title("⚔️ Head-to-Head: Classical DOE vs. ML/Bioinformatics")
    st.markdown("A visual comparison of the core philosophies and practical strengths of the two approaches, tailored for biotech R&D tasks.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("Strengths Profile: A Multi-Dimensional View")
        st.markdown("This radar chart compares the two methodologies across key attributes relevant to assay development. The different 'shapes' of their capabilities highlight their complementary nature.")
        st.plotly_chart(plot_comparison_radar(), use_container_width=True)

    with st.container(border=True):
        st.subheader("The Verdict: Which Approach Excels for Which Task?")
        st.markdown("This chart provides a clear, decisive verdict for common use cases in a biotech R&D setting.")
        st.plotly_chart(plot_verdict_barchart(), use_container_width=True)

# ==============================================================================
# PAGE 7: THE HYBRID LAB MANIFESTO
# ==============================================================================
def show_hybrid_strategy():
    st.title("🤝 The Hybrid Lab Manifesto: The Future of Assay Development")
    st.markdown("The most competitive biotech organizations do not choose one over the other; they build a **Bio-AI framework** that fuses statistical rigor with machine learning's predictive power.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("The Philosophy of Synergy: Inference + Prediction")
        st.markdown("Neither methodology is a silver bullet. True power lies in their integration. Classical statistics provides the **rigor for inference and causality**, while machine learning provides the **power for prediction and scale**.")
        st.plotly_chart(plot_synergy_diagram(), use_container_width=True)

    with st.container(border=True):
        st.subheader("Interactive Solution Recommender")
        st.info("💡 Select a common R&D scenario to see the recommended hybrid approach and expert rationale.")
        guidance_data = get_guidance_data()
        scenarios = list(guidance_data.keys())
        selected_scenario = st.selectbox("Choose your scenario:", scenarios, label_visibility="collapsed")
        if selected_scenario:
            st.markdown(f"##### Recommended Approach: {guidance_data[selected_scenario]['approach']}")
            st.markdown(f"**Rationale:** {guidance_data[selected_scenario]['rationale']}")

    with st.container(border=True):
        st.subheader("A Unified, Modern R&D Workflow")
        st.markdown("This workflow demonstrates how to embed ML augmentation at each step of the traditional assay development cycle.")
        st.markdown(get_workflow_css(), unsafe_allow_html=True)
        st.markdown('<div class="workflow-container">', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="🌀 1. Define", phase_class="step-define",
            classical_tools=["Assay Design Plan", "SIPOC of Workflow", "Kano Model", "CTQ Tree (TPP)"],
            ml_tools=["NLP for Literature Review", "Causal Discovery from Pilot Data", "Patient Stratification"]), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="🔬 2. Measure", phase_class="step-measure",
            classical_tools=["Gage R&R (MSA)", "Process Capability (Cpk)", "VSM of Lab Process"],
            ml_tools=["Process Mining on LIMS", "Kernel Density Estimation (KDE)", "Assay Drift Detection"]), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="📈 3. Analyze", phase_class="step-analyze",
            classical_tools=["Hypothesis Testing (ANOVA)", "Pareto Analysis", "Fishbone Diagram", "Linear Regression"],
            ml_tools=["Biomarker Feature Importance (SHAP)", "Ensemble Models", "Permutation Testing"]), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="⚙️ 4. Improve", phase_class="step-improve",
            classical_tools=["Design of Experiments (DOE)", "FMEA", "Pilot Validation"],
            ml_tools=["Bayesian Optimization", "Predictive Maintenance (RUL)", "In Silico Surrogate Models"]), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(
            phase_name="📡 5. Control", phase_class="step-control",
            classical_tools=["Levey-Jennings Charts (SPC)", "Control Plan & SOPs"],
            ml_tools=["Multivariate QC (Hotelling's T²)", "Real-time Anomaly Detection", "Automated Batch Release"]), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
