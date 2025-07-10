# app_pages.py

import streamlit as st
import numpy as np

# Import all necessary helper functions from the single, corrected helper file.
from app_helpers import *


# ==============================================================================
# PAGE 0: WELCOME & FRAMEWORK
# ==============================================================================
def show_welcome_page():
    st.title("Welcome to the Bio-AI Excellence Framework")
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
        """)
    with col2:
        st.subheader("ML & Bioinformatics Augmentation")
        st.markdown("""
        A suite of computational techniques that excel at finding patterns in high-dimensional biological data, making predictions, and automating complex analysis pipelines.
        - **Core Strength:** Prediction, biomarker discovery, handling complexity, and extracting signals from noisy, high-dimensional data.
        - **Primary Focus:** Optimizing multi-parameter protocols, discovering novel signatures, and enabling proactive quality control.
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
    st.title("Define: Clinical Need & Assay Goals")
    st.markdown("**Objective:** To clearly articulate the clinical problem, establish the project's goals, define the scope of the assay, and translate the 'Voice of the Clinician' into quantifiable, 'Critical to Quality' (CTQ) assay performance characteristics.")
    st.markdown("---")
    
    with st.container(border=True):
        st.subheader("1. The Mandate: Assay Design & Development Plan")
        with st.expander("Learn more about the Design Plan"):
            st.markdown("The Design Plan is the foundational document, formally defining the assay's intended use, target population, required performance specifications, and development roadmap, aligning the scientific team with business and clinical goals.")
        st.plotly_chart(plot_project_charter_visual(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. The Landscape: Mapping the Assay Workflow & Hypotheses")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical Tool: SIPOC for Lab Workflows**")
            with st.expander("What is SIPOC?"):
                st.info("A high-level map of the entire assay workflow. It's a qualitative, expert-driven tool for defining process boundaries and fostering team alignment.")
            st.plotly_chart(plot_sipoc_visual(), use_container_width=True)
        with col2:
            st.markdown("##### **ML Augmentation: Causal Discovery**")
            with st.expander("What is Causal Discovery?"):
                st.info("Algorithms that analyze pilot data to infer a graph of probable cause-and-effect relationships between protocol parameters and assay outcomes, generating data-driven hypotheses for later validation.")
            st.graphviz_chart(plot_causal_discovery_visual())

    with st.container(border=True):
        st.subheader("3. The Target: Translating Clinical Needs into Assay Specs (CTQs)")
        st.markdown("This step translates a clinician's need (e.g., 'detect tumors earlier') into measurable assay performance characteristics.")
        tab1, tab2, tab3 = st.tabs(["📊 CTQ Tree", "💖 Kano Model", "🤖 NLP Review"])
        with tab1:
            st.markdown("##### **Classical Tool: CTQ Tree**")
            st.info("A decomposition tool to break down a broad clinical need (e.g., reliable liquid biopsy) into specific, measurable performance metrics like **Analytical Sensitivity (LOD), Specificity, and Precision (CV%)**.")
            st.graphviz_chart(plot_ctq_tree_visual())
        with tab2:
            st.markdown("##### **Classical Tool: Kano Model**")
            st.info("A framework for prioritizing diagnostic features. **Basic** needs might be detecting a known cancer mutation. **Performance** could be accurate VAF quantification. An **Excitement** feature could be discovering a novel, actionable co-mutation.")
            st.plotly_chart(plot_kano_visual(), use_container_width=True)
        with tab3:
            st.markdown("##### **ML Augmentation: NLP on Scientific Literature**")
            st.info("Using algorithms (e.g., SciBERT) to analyze thousands of publications to automatically extract prevalent biomarkers, competing methodologies, and reported performance benchmarks, massively accelerating the literature review process.")
            st.plotly_chart(plot_voc_treemap(), use_container_width=True)

    st.success("""**🏆 Hybrid Strategy for the Define Phase:**\n1. **Mandate & Scope (Classical):** Begin with a formal **Assay Design Plan** and a team-based **SIPOC** of the lab workflow to establish clear boundaries and alignment.\n2. **Discover at Scale (ML):** Deploy **NLP Topic Modeling** on scientific literature to generate a data-driven list of critical biomarkers and performance benchmarks.\n3. **Translate & Prioritize (Hybrid):** Use the NLP outputs to build a data-grounded **CTQ Tree**, ensuring it reflects the current scientific landscape and defines the **Target Product Profile (TPP)**.""")

# ==============================================================================
# PAGE 2: MEASURE PHASE - ASSAY & SYSTEM VALIDATION
# ==============================================================================
def show_measure_phase():
    st.title("Measure: Assay & System Validation")
    st.markdown("**Objective:** To validate the reliability of all measurement systems (pipettes, sequencers), collect data, and establish a robust, data-driven baseline of the assay's current performance.")
    st.markdown("---")
    
    # UX IMPROVEMENT: Local helper for clear, contextual interpretation of capability.
    def get_capability_interpretation(cp, cpk, target=1.33):
        cp_status, cpk_status = "", ""
        if cp < 1.0: cp_status = (f"🚨 **Inadequate Potential (Cp < 1.0):** The process variation is wider than the specification limits. Improvement is required.", "error")
        elif cp < target: cp_status = (f"⚠️ **Needs Improvement (1.0 ≤ Cp < {target}):** The process could potentially fit, but has little room for error or shift.", "warning")
        else: cp_status = (f"✅ **Adequate Potential (Cp ≥ {target}):** The process has low enough variation to meet specs.", "success")

        if cpk < 1.0: cpk_status = (f"🚨 **Not Capable (Cpk < 1.0):** The process is producing significant defects.", "error")
        elif cpk < target: cpk_status = (f"⚠️ **Needs Centering (1.0 ≤ Cpk < {target}):** The process is not well-centered within the specification limits.", "warning")
        else: cpk_status = (f"✅ **Capable & Centered (Cpk ≥ {target}):** The process is performing well within specifications.", "success")
        return cp_status, cpk_status

    with st.container(border=True):
        st.subheader("1. Prerequisite: Measurement System Analysis (MSA)")
        st.warning("**You cannot trust your assay data until you trust your instruments and operators.** MSA is a non-negotiable step to ensure observed variability comes from the biology, not the lab process.")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("##### **Classical Tool: Gage R&R**")
            st.info("A designed experiment to partition measurement variance into **Repeatability** (equipment) and **Reproducibility** (operator).")
        with col2:
            st.plotly_chart(plot_gage_rr_variance_components(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Understanding the End-to-End Workflow")
        tab1, tab2 = st.tabs(["🗺️ Value Stream Mapping (VSM)", "🤖 Process Mining"])
        with tab1:
            st.markdown("##### **Classical Tool: Value Stream Mapping (VSM)**")
            st.info("A manual flowchart of the lab process, capturing hands-on and wait times to identify physical bottlenecks.")
            st.plotly_chart(plot_vsm(), use_container_width=True)
        with tab2:
            st.markdown("##### **ML Augmentation: Process Mining on LIMS Data**")
            st.info("Algorithms that automatically discover the real lab workflow by analyzing event logs from a **LIMS**, revealing hidden bottlenecks like QC failures and re-tests.")
            st.graphviz_chart(plot_process_mining_graph())

    with st.container(border=True):
        st.subheader("3. Establishing Baseline Assay Capability")
        st.markdown("Capability analysis answers: **Can our assay reliably meet the required performance specifications (e.g., Limit of Detection)?**")
        
        st.sidebar.header("🔬 Simulators")
        st.sidebar.markdown("---")
        st.sidebar.subheader("Assay Capability")
        st.sidebar.markdown("Adjust assay performance and spec limits to see the impact on capability.")
        lsl = st.sidebar.slider("Lower Spec Limit (LSL)", 0.5, 2.0, 0.8, key="m_lsl", help="e.g., Minimum required signal-to-noise")
        usl = st.sidebar.slider("Upper Spec Limit (USL)", 8.0, 10.0, 9.0, key="m_usl", help="e.g., Maximum tolerable background")
        process_mean = st.sidebar.slider("Assay Mean (μ)", 2.0, 8.0, 4.0, key="m_mean")
        process_std = st.sidebar.slider("Assay Std Dev (σ)", 0.2, 2.0, 0.5, key="m_std")
        
        data = generate_process_data(process_mean, process_std, 2000)
        fig_cap_hist, cp, cpk = plot_capability_analysis_pro(data, lsl, usl)
        fig_cp_gauge, fig_cpk_gauge = plot_capability_metrics(cp, cpk)
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("##### **Assay Distribution vs. Specifications**")
            st.plotly_chart(fig_cap_hist, use_container_width=True)
        with col2:
            st.markdown("##### **Capability Indices**")
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.markdown("<p style='text-align: center;'><b>Process Potential (Cp)</b></p>", unsafe_allow_html=True)
                st.plotly_chart(fig_cp_gauge, use_container_width=True)
            with sub_col2:
                st.markdown("<p style='text-align: center;'><b>Process Capability (Cpk)</b></p>", unsafe_allow_html=True)
                st.plotly_chart(fig_cpk_gauge, use_container_width=True)
            
            # UX Improvement: Provide direct, contextual interpretation.
            cp_status, cpk_status = get_capability_interpretation(cp, cpk)
            getattr(st, cp_status[1])(cp_status[0])
            getattr(st, cpk_status[1])(cpk_status[0])

    st.success("""**🏆 Hybrid Strategy for the Measure Phase:**\n1. **Validate (Classical):** Always perform a **Gage R&R** on critical instruments and operators before baselining performance.\n2. **Discover (ML):** Run **Process Mining** on LIMS event logs to get an objective map of the real lab workflow and its bottlenecks.\n3. **Detail (Classical):** Use insights from process mining to guide a targeted, physical **VSM** exercise.\n4. **Baseline & Diagnose (Hybrid):** Report the official **Cpk** baseline. Internally, use the **KDE plot** and **Gauge Visuals** to diagnose the *reason* for poor capability.""")

# ==============================================================================
# PAGE 3: ANALYZE PHASE - ROOT CAUSE OF ASSAY VARIABILITY
# ==============================================================================
def show_analyze_phase():
    st.title("Analyze: Root Cause of Assay Variability")
    st.markdown("**Objective:** To analyze data to identify, validate, and quantify the root cause(s) of poor assay performance (e.g., low sensitivity, high CV%). This moves from *what* is failing to *why* it is failing.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Qualitative Root Cause Analysis & Prioritization")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical Tool: Fishbone Diagram**")
            st.info("A structured brainstorming tool to organize potential causes of a problem.")
            st.graphviz_chart(plot_fishbone_diagram())
        with col2:
            st.markdown("##### **Classical Tool: Pareto Chart**")
            st.info("A chart to identify the 'vital few' failure modes to focus improvement efforts.")
            st.plotly_chart(plot_pareto_chart(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Proving the Difference: Comparing Experimental Groups")
        with st.expander("Learn about group comparison methods"):
            st.markdown("Once hypotheses are formed (e.g., 'Reagent Lot B is causing lower library concentrations'), statistical proof is required to confirm a significant difference.")
        
        st.sidebar.header("🔬 Simulators")
        st.sidebar.markdown("---")
        st.sidebar.subheader("Group Comparison")
        st.sidebar.markdown("Adjust reagent lot means to see if the difference becomes statistically significant.")
        mean1 = st.sidebar.slider("Lot A Mean Yield (ng/µL)", 18.0, 22.0, 19.5, 0.1, key='a1')
        mean2 = st.sidebar.slider("Lot B Mean Yield (ng/µL)", 18.0, 22.0, 20.0, 0.1, key='a2')
        mean3 = st.sidebar.slider("Lot C Mean Yield (ng/µL)", 18.0, 22.0, 20.5, 0.1, key='a3')
        anova_data = generate_anova_data(means=[mean1, mean2, mean3], stds=[0.8, 0.8, 0.8], n=20)
        
        tab1, tab2 = st.tabs(["🔬 ANOVA", "💻 Permutation Testing"])
        with tab1:
            st.markdown("##### **Classical: Analysis of Variance (ANOVA)**")
            st.info("Tests if significant differences exist between group means. Assumes normality and equal variances.")
            fig_anova, p_val = plot_anova_groups(anova_data)
            st.plotly_chart(fig_anova, use_container_width=True)
            if p_val < 0.05: st.error(f"P-value is {p_val:.4f}. A statistically significant difference exists.", icon="🚨")
            else: st.success(f"P-value is {p_val:.4f}. No significant difference detected.", icon="✅")
        with tab2:
            st.markdown("##### **ML Augmentation: Permutation Testing**")
            st.info("A non-parametric method that makes no assumptions about data distribution. More robust for non-normal, small-sample biotech data.")
            st.plotly_chart(plot_permutation_test(anova_data), use_container_width=True)

    with st.container(border=True):
        st.subheader("3. Finding the Drivers: Modeling Assay Performance (Y = f(x))")
        st.markdown("This step finds which protocol parameters (X's like `Annealing Temp`) mathematically drive the key output (Y, e.g., `On-Target Rate`).")
        df_reg = generate_nonlinear_data()
        fig_reg, model, X_reg = train_and_plot_regression_models(df_reg)
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### **Classical vs. ML Model Fit**")
            st.info("A linear model often fails to capture complex biology. An ML model (Random Forest) can fit non-linear relationships, indicated by a higher R².")
            st.plotly_chart(fig_reg, use_container_width=True)
        with col4:
            st.markdown("##### **ML Augmentation: XAI to Find Root Cause**")
            st.info("If the ML model is better, we trust its interpretation. **SHAP** explains the model, revealing the most important drivers of performance.")
            st.plotly_chart(plot_shap_summary(model, X_reg), use_container_width=True)

    st.success("""**🏆 Hybrid Strategy for the Analyze Phase:**\n1. **Structure & Prioritize (Classical):** Use a **Fishbone** diagram to brainstorm causes and a **Pareto** chart on QC failures to identify which modes to investigate first.\n2. **Verify Group Differences (Hybrid):** Use **ANOVA** as a first step but default to a more robust **Permutation Test** given the small sample sizes and potential for non-normal data in R&D.\n3. **Model Relationships (Hybrid):** Fit both **Linear Regression** and an **Ensemble ML model**. If the ML model is more accurate (check R²), its **SHAP** rankings are a more reliable guide to the true root causes.""")

# ==============================================================================
# PAGE 4: IMPROVE PHASE - ASSAY & WORKFLOW OPTIMIZATION
# ==============================================================================
@st.cache_data
def get_true_bo_func(x):
    return (np.sin(x*0.8)*15) + (np.cos(x*2.5)*5) - (x/10)**3

def show_improve_phase():
    st.title("Improve: Assay & Workflow Optimization")
    st.markdown("**Objective:** To identify, test, and implement solutions that address validated root causes. For assays, this involves finding the optimal settings for critical protocol parameters to maximize performance.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Finding Optimal Protocol Settings")
        st.markdown("Once we know which parameters are critical, we need to find their optimal settings to maximize our output (e.g., `On-Target Reads`).")
        
        tab1, tab2 = st.tabs(["🧪 Design of Experiments (DOE)", "🤖 Bayesian Optimization"])
        with tab1:
            st.markdown("##### **Classical: Design of Experiments (DOE)**")
            st.info("A structured method for efficiently changing multiple parameters simultaneously to determine their main and interaction effects. Gold standard for physical lab work.")
            doe_data = generate_doe_data()
            fig_doe_main, fig_doe_interaction = plot_doe_effects(doe_data)
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(plot_doe_cube(doe_data), use_container_width=True)
            with col2: st.plotly_chart(fig_doe_main, use_container_width=True); st.plotly_chart(fig_doe_interaction, use_container_width=True)
        with tab2:
            st.markdown("##### **ML Augmentation: Bayesian Optimization**")
            st.info("An intelligent search algorithm for finding the global optimum of an expensive-to-evaluate function (e.g., a full NGS run). It intelligently selects the most informative next experiment to run.")
            
            st.sidebar.header("🔬 Simulators")
            st.sidebar.markdown("---")
            st.sidebar.subheader("Bayesian Optimization")
            st.sidebar.markdown("Let the algorithm choose the next experiment to find the maximum on-target rate.")
            
            x_range = np.linspace(0, 20, 400)
            true_func = get_true_bo_func

            if 'sampled_points' not in st.session_state:
                st.session_state.sampled_points = {'x': [2.0, 18.0], 'y': [true_func(2.0), true_func(18.0)]}
            
            if st.sidebar.button("Run Next Smart Experiment", key='bo_sample'):
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
            st.info("Failure Mode and Effects Analysis is a systematic, team-based risk assessment of the assay protocol, ranking risks by a Risk Priority Number (RPN).")
            st.plotly_chart(plot_fmea_table(), use_container_width=True)
        with col4:
            st.markdown("##### **ML Augmentation: Predictive Maintenance**")
            st.info("Using ML models trained on sensor data (e.g., sequencer laser power) to predict degradation and estimate Remaining Useful Life (RUL) before a failure occurs.")
            st.plotly_chart(plot_rul_prediction(generate_sensor_degradation_data()), use_container_width=True)

    st.success("""**🏆 Hybrid Strategy for the Improve Phase:**\n1. **Optimize with the Right Tool:** For optimizing a few (<5) parameters, **DOE** is the gold standard. For high-dimensional protocols, use **Bayesian Optimization** for its superior sample efficiency.\n2. **Mitigate Risks (Hybrid):** Use a classical **FMEA** to identify the highest-risk failure modes. For top risks related to equipment, build a **predictive maintenance (RUL) model**.\n3. **The Ultimate Hybrid ("Digital Twin"):** Use data from a space-filling **DOE** to train an ML model of your assay. Then, use **Bayesian Optimization** on this *in silico* twin to find the global optimum before one final confirmation experiment.""")

# ==============================================================================
# PAGE 5: CONTROL PHASE - LAB OPERATIONS & QC
# ==============================================================================
def show_control_phase():
    st.title("Control: Lab Operations & QC")
    st.markdown("**Objective:** To implement a robust Quality Control (QC) system to monitor the optimized assay in routine use, ensuring performance remains stable and that improvements are sustained.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Monitoring for Stability: Statistical Process Control (SPC)")
        st.markdown("Control charts monitor positive/negative controls over time, distinguishing 'common cause' variation from 'special cause' variation that signals a problem.")
        
        st.sidebar.header("🔬 Simulators")
        st.sidebar.markdown("---")
        st.sidebar.subheader("QC Simulator")
        st.sidebar.markdown("Introduce a shift in a positive control standard and see which chart detects it faster.")
        shift_mag = st.sidebar.slider("Magnitude of Shift (in Std Devs)", 0.2, 3.0, 0.8, 0.1, key="ctrl_shift_mag")
        ewma_lambda = st.sidebar.slider("EWMA Lambda (λ)", 0.1, 0.5, 0.2, 0.05, help="Higher λ reacts faster to shifts.")
        
        chart_data = generate_control_chart_data(shift_magnitude=shift_mag)
        
        tab1, tab2, tab3 = st.tabs(["📊 Levey-Jennings", "📈 EWMA/CUSUM", "🤖 Multivariate QC"])
        with tab1:
            st.markdown("##### **Classical: Levey-Jennings Chart (Shewhart)**")
            st.info("The standard QC chart in clinical labs. Excellent for detecting large, sudden shifts.")
            st.plotly_chart(plot_shewhart_chart(chart_data), use_container_width=True)
        with tab2:
            st.markdown("##### **Advanced Classical: EWMA & CUSUM Charts**")
            st.info("These charts have 'memory,' making them highly effective at detecting small, sustained drifts (e.g., slow reagent degradation) that Levey-Jennings charts would miss.")
            st.plotly_chart(plot_ewma_chart(chart_data, lambda_val=ewma_lambda), use_container_width=True)
            st.plotly_chart(plot_cusum_chart(chart_data), use_container_width=True)
        with tab3:
            st.markdown("##### **ML Augmentation: Multivariate QC**")
            st.info("Monitors the 'health' of the entire QC profile at once, flagging abnormalities even if each individual metric is within its own limits.")
            st.plotly_chart(plot_hotelling_t2_chart(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Formalizing the Gains: The Control Plan & SOPs")
        st.info("The Control Plan is a living document that details the QC methods, responsibilities, and reaction plan for any out-of-control signal. It is codified in the lab's Standard Operating Procedures (SOPs).")
        st.plotly_chart(plot_control_plan(), use_container_width=True)

    st.success("""**🏆 Hybrid Strategy for the Control Phase:**\n1. **Monitor with Levey-Jennings:** Use a classical **Levey-Jennings chart** for primary positive/negative controls for simplicity and regulatory compliance.\n2. **Detect Drifts with EWMA:** For critical secondary metrics, use a more sensitive **EWMA chart** to detect slow reagent or instrument degradation.\n3. **Holistic QC with ML:** For each sample, run a **multivariate QC model** (like Hotelling's T²) on the full profile of NGS QC metrics to flag subtle issues.\n4. **Codify Everything:** The **Control Plan** and **SOPs** must document all charts, limits, and reaction plans.""")

# ==============================================================================
# PAGE 6: METHODOLOGY COMPARISON
# ==============================================================================
def show_comparison_matrix():
    st.title("Head-to-Head: Classical DOE vs. ML/Bioinformatics")
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
# PAGE 7: HYBRID STRATEGY
# ==============================================================================
def show_hybrid_strategy():
    st.title("The Hybrid Lab Manifesto: The Future of Assay Development")
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
        st.markdown(render_workflow_step(phase_name="🌀 1. Define", phase_class="step-define", classical_tools=["Assay Design Plan", "SIPOC of Workflow", "Kano Model", "CTQ Tree (TPP)"], ml_tools=["NLP for Literature Review", "Causal Discovery from Pilot Data", "Patient Stratification"]), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(phase_name="🔬 2. Measure", phase_class="step-measure", classical_tools=["Gage R&R (MSA)", "Process Capability (Cpk)", "VSM of Lab Process"], ml_tools=["Process Mining on LIMS", "Kernel Density Estimation (KDE)", "Assay Drift Detection"]), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(phase_name="📈 3. Analyze", phase_class="step-analyze", classical_tools=["Hypothesis Testing (ANOVA)", "Pareto Analysis", "Fishbone Diagram", "Linear Regression"], ml_tools=["Biomarker Feature Importance (SHAP)", "Ensemble Models", "Permutation Testing"]), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(phase_name="⚙️ 4. Improve", phase_class="step-improve", classical_tools=["Design of Experiments (DOE)", "FMEA", "Pilot Validation"], ml_tools=["Bayesian Optimization", "Predictive Maintenance (RUL)", "In Silico Surrogate Models"]), unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">⬇️</div>', unsafe_allow_html=True)
        st.markdown(render_workflow_step(phase_name="📡 5. Control", phase_class="step-control", classical_tools=["Levey-Jennings Charts (SPC)", "Control Plan & SOPs"], ml_tools=["Multivariate QC (Hotelling's T²)", "Real-time Anomaly Detection", "Automated Batch Release"]), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
