# app_pages.py

import streamlit as st
import numpy as np

# Import all necessary helper functions from the single, definitive helper file.
from app_helpers import *


# ==============================================================================
# PAGE 0: WELCOME & FRAMEWORK
# ==============================================================================
def show_welcome_page():
    st.title("Welcome to the Bio-AI Excellence Framework")
    st.markdown("##### An interactive playbook for developing and optimizing robust genomic assays and devices.")
    st.markdown("---")
    st.info("""**This application is designed for a technically proficient audience** (e.g., R&D Scientists, Bioinformaticians, Lab Directors). It moves beyond introductory concepts to demonstrate a powerful, unified framework that fuses the **inferential rigor of classical statistics** with the **predictive power of modern Machine Learning**.""")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classical Assay Development")
        st.markdown("- **Core Strength:** Establishing causality, understanding main effects and interactions, and ensuring statistical rigor for regulatory submissions (e.g., FDA).")
    with col2:
        st.subheader("ML & Bioinformatics Augmentation")
        st.markdown("- **Core Strength:** Prediction, biomarker discovery, handling complexity, and extracting signals from noisy, high-dimensional data.")
    st.subheader("The Hybrid Lab Philosophy: Augmentation, Not Replacement")
    st.markdown("The most effective path to developing breakthrough diagnostics lies in the **synergistic integration** of these two disciplines. Use the navigation panel to explore the R&D lifecycle (framed as **DMAIC**). Each phase presents classical tools alongside their AI-augmented counterparts.")
    st.success("Click on a phase in the sidebar to begin your exploration.")

# ==============================================================================
# PAGE 1: DEFINE PHASE - CLINICAL NEED & ASSAY GOALS
# ==============================================================================
def show_define_phase():
    st.title("Define: Clinical Need & Assay Goals")
    st.markdown("**Objective:** To clearly articulate the clinical problem, establish project goals, define the assay scope, and translate clinical needs into quantifiable 'Critical to Quality' (CTQ) assay performance characteristics.")
    st.markdown("---")
    
    with st.container(border=True):
        st.subheader("1. The Mandate: Assay Design & Development Plan")
        st.plotly_chart(plot_project_charter_visual(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. The Landscape: Mapping the Workflow & Hypotheses")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical Tool: SIPOC**")
            st.plotly_chart(plot_sipoc_visual(), use_container_width=True)
        with col2:
            st.markdown("##### **ML Augmentation: Causal Discovery**")
            st.plotly_chart(plot_causal_discovery_plotly(), use_container_width=True)

    with st.container(border=True):
        st.subheader("3. The Target: Translating Needs into Assay Specs (CTQs)")
        tab1, tab2, tab3 = st.tabs(["📊 CTQ Tree", "💖 Kano Model", "🤖 NLP Review"])
        with tab1:
            st.markdown("##### **Classical Tool: CTQ Tree**")
            st.plotly_chart(plot_ctq_tree_plotly(), use_container_width=True)
        with tab2:
            st.markdown("##### **Classical Tool: Kano Model**")
            st.plotly_chart(plot_kano_visual(), use_container_width=True)
        with tab3:
            st.markdown("##### **ML Augmentation: NLP on Scientific Literature**")
            st.plotly_chart(plot_voc_bubble_chart(), use_container_width=True)
            with st.expander("📊 How to Interpret This Chart"):
                st.markdown("""
                - **What it is:** This bubble chart visualizes topics extracted from scientific literature.
                - **Y-Axis (Sentiment):** Higher values indicate more positive language associated with a topic.
                - **X-Axis (Topic):** The specific biomarker or methodology identified.
                - **Bubble Size (Count):** Larger bubbles mean the topic was mentioned more frequently in publications.
                - **Takeaway:** We can quickly see that `LOD <0.1%` is a highly desirable and frequently mentioned performance characteristic. `ddPCR` and `Shallow WGS` are mentioned less and have negative sentiment, suggesting they may be viewed as less favorable or have known issues.
                """)

    st.success("""**🏆 Hybrid Strategy for the Define Phase:**\n1. **Mandate & Scope (Classical):** Begin with a formal **Assay Design Plan** and a team-based **SIPOC** of the lab workflow to establish clear boundaries and alignment.\n2. **Discover at Scale (ML):** Deploy **NLP Topic Modeling** on scientific literature to generate a data-driven list of critical biomarkers and performance benchmarks.\n3. **Translate & Prioritize (Hybrid):** Use the NLP outputs to build a data-grounded **CTQ Tree**, ensuring it reflects the current scientific landscape and defines the **Target Product Profile (TPP)**.""")

# ==============================================================================
# PAGE 2: MEASURE PHASE - ASSAY & SYSTEM VALIDATION
# ==============================================================================
def show_measure_phase():
    st.title("Measure: Assay & System Validation")
    st.markdown("**Objective:** To validate measurement systems, collect data, and establish a robust, data-driven baseline of the assay's current performance.")
    st.markdown("---")
    
    def get_capability_interpretation(cp, cpk, target=1.33):
        cp_color = "success" if cp >= target else ("warning" if cp >= 1.0 else "error")
        cpk_color = "success" if cpk >= target else ("warning" if cpk >= 1.0 else "error")
        cp_help = "Measures process potential: Is the variation low enough to fit within the spec limits? (Target: > 1.33)"
        cpk_help = "Measures process performance: Is the process actually capable and centered? (Target: > 1.33)"
        return (cp_color, cpk_color, cp_help, cpk_help)

    with st.container(border=True):
        st.subheader("1. Prerequisite: Measurement System Analysis (MSA)")
        st.warning("**You cannot trust your assay data until you trust your measurement system.** MSA ensures observed variability comes from the biology, not the lab process.")
        st.plotly_chart(plot_gage_rr_pareto(), use_container_width=True)
        with st.expander("📊 How to Interpret This Chart"):
            st.markdown("""
            - **What it is:** A Pareto chart of the sources of variation in your measurement system.
            - **Bars (Contribution):** Shows the percentage of total measurement error attributable to each source.
            - **Line (Cumulative %):** Shows the cumulative contribution, highlighting the "vital few" sources.
            - **Takeaway:** In this example, the vast majority (92%) of variation is from the assay itself, which is ideal. The measurement system (Repeatability and Reproducibility) contributes very little error, meaning we can trust our measurements. If `Reproducibility (Operator)` were high, it would signal a need for better training.
            """)

    with st.container(border=True):
        st.subheader("2. Understanding the End-to-End Workflow")
        tab1, tab2 = st.tabs(["🗺️ Value Stream Mapping (VSM)", "🤖 Process Mining"])
        with tab1:
            st.markdown("##### **Classical Tool: Value Stream Mapping (VSM)**")
            st.plotly_chart(plot_vsm(), use_container_width=True)
        with tab2:
            st.markdown("##### **ML Augmentation: Process Mining on LIMS Data**")
            st.plotly_chart(plot_process_mining_plotly(), use_container_width=True)

    with st.container(border=True):
        st.subheader("3. Establishing Baseline Assay Capability")
        st.sidebar.header("🔬 Simulators"); st.sidebar.markdown("---"); st.sidebar.subheader("Assay Capability")
        lsl = st.sidebar.slider("Lower Spec Limit (LSL)", 0.5, 2.0, 0.8, key="m_lsl")
        usl = st.sidebar.slider("Upper Spec Limit (USL)", 8.0, 10.0, 9.0, key="m_usl")
        process_mean = st.sidebar.slider("Assay Mean (μ)", 2.0, 8.0, 4.0, key="m_mean")
        process_std = st.sidebar.slider("Assay Std Dev (σ)", 0.2, 2.0, 0.5, key="m_std")
        
        data = generate_process_data(process_mean, process_std, 2000)
        fig_cap_hist, cp, cpk = plot_capability_analysis_pro(data, lsl, usl)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(fig_cap_hist, use_container_width=True)
        with col2:
            st.markdown("##### **Capability Indices**")
            cp_color, cpk_color, cp_help, cpk_help = get_capability_interpretation(cp, cpk)
            st.metric(label="Process Potential (Cp)", value=f"{cp:.2f}", help=cp_help)
            st.markdown(f'<hr style="margin-top:0; margin-bottom:0.5rem; border-color:{COLORS[cp_color]}">', unsafe_allow_html=True)
            st.metric(label="Process Capability (Cpk)", value=f"{cpk:.2f}", help=cpk_help)
            st.markdown(f'<hr style="margin-top:0; margin-bottom:0.5rem; border-color:{COLORS[cpk_color]}">', unsafe_allow_html=True)
            
        with st.expander("📊 How to Interpret This Chart"):
            st.markdown("""
            - **What it is:** A histogram of your assay's output compared to its required specification limits (LSL/USL).
            - **Cp (Potential):** Asks, "Is the process variation *narrow enough* to fit within the spec window?" A value > 1.33 is good.
            - **Cpk (Capability):** Asks, "Is the process *actually centered* and performing within the spec window?" A value > 1.33 is good. Cpk is the true measure of performance.
            - **Takeaway:** You can have a good Cp (low variation) but a bad Cpk if your process mean is not centered between the spec limits. This chart helps diagnose whether you need to reduce variation, shift the mean, or both.
            """)

    st.success("""**🏆 Hybrid Strategy for the Measure Phase:**\n1. **Validate (Classical):** Always perform a **Gage R&R** on critical instruments and operators before baselining performance.\n2. **Discover (ML):** Run **Process Mining** on LIMS event logs to get an objective map of the real lab workflow and its bottlenecks.\n3. **Detail (Classical):** Use insights from process mining to guide a targeted, physical **VSM** exercise.\n4. **Baseline & Diagnose (Hybrid):** Report the official **Cpk** baseline. Internally, use the **KDE plot** and **Metric Visuals** to diagnose the *reason* for poor capability.""")

# ==============================================================================
# PAGE 3: ANALYZE PHASE - ROOT CAUSE OF ASSAY VARIABILITY
# ==============================================================================
def show_analyze_phase():
    st.title("Analyze: Root Cause of Assay Variability")
    st.markdown("**Objective:** To analyze data to identify, validate, and quantify the root cause(s) of poor assay performance, moving from *what* is failing to *why*.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Qualitative Root Cause Analysis & Prioritization")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical Tool: Fishbone Diagram**")
            st.plotly_chart(plot_fishbone_plotly(), use_container_width=True)
        with col2:
            st.markdown("##### **Classical Tool: Pareto Chart**")
            st.plotly_chart(plot_pareto_chart(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Proving the Difference: Comparing Experimental Groups")
        st.sidebar.header("🔬 Simulators"); st.sidebar.markdown("---"); st.sidebar.subheader("Group Comparison")
        mean1 = st.sidebar.slider("Lot A Mean Yield (ng/µL)", 18.0, 22.0, 19.5, 0.1, key='a1')
        mean2 = st.sidebar.slider("Lot B Mean Yield (ng/µL)", 18.0, 22.0, 20.0, 0.1, key='a2')
        mean3 = st.sidebar.slider("Lot C Mean Yield (ng/µL)", 18.0, 22.0, 20.5, 0.1, key='a3')
        anova_data = generate_anova_data(means=[mean1, mean2, mean3], stds=[0.8, 0.8, 0.8], n=20)
        
        tab1, tab2 = st.tabs(["🔬 ANOVA", "💻 Permutation Testing"])
        with tab1:
            st.markdown("##### **Classical: Analysis of Variance (ANOVA)**")
            fig_anova, p_val = plot_anova_groups(anova_data)
            st.plotly_chart(fig_anova, use_container_width=True)
            if p_val < 0.05: st.error(f"P-value is {p_val:.4f}. A statistically significant difference exists.", icon="🚨")
            else: st.success(f"P-value is {p_val:.4f}. No significant difference detected.", icon="✅")
        with tab2:
            st.markdown("##### **ML Augmentation: Permutation Testing**")
            st.plotly_chart(plot_permutation_test(anova_data), use_container_width=True)

    with st.container(border=True):
        st.subheader("3. Finding the Drivers: Modeling Assay Performance (Y = f(x))")
        df_reg = generate_nonlinear_data(); fig_reg, model, X_reg = train_and_plot_regression_models(df_reg)
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### **Classical vs. ML Model Fit**")
            st.plotly_chart(fig_reg, use_container_width=True)
        with col4:
            st.markdown("##### **ML Augmentation: XAI to Find Root Cause**")
            st.plotly_chart(plot_shap_summary(model, X_reg), use_container_width=True)

    st.success("""**🏆 Hybrid Strategy for the Analyze Phase:**\n1. **Structure & Prioritize (Classical):** Use a **Fishbone** diagram to brainstorm causes and a **Pareto** chart on QC failures to identify which modes to investigate first.\n2. **Verify Group Differences (Hybrid):** Use **ANOVA** as a first step but default to a more robust **Permutation Test** given the small sample sizes and potential for non-normal data in R&D.\n3. **Model Relationships (Hybrid):** Fit both **Linear Regression** and an **Ensemble ML model**. If the ML model is more accurate (check R²), its **SHAP** rankings are a more reliable guide to the true root causes.""")

# ==============================================================================
# PAGE 4: IMPROVE PHASE - ASSAY & WORKFLOW OPTIMIZATION
# ==============================================================================
@st.cache_data
def get_true_bo_func(x): return (np.sin(x*0.8)*15) + (np.cos(x*2.5)*5) - (x/10)**3

def show_improve_phase():
    st.title("Improve: Assay & Workflow Optimization")
    st.markdown("**Objective:** To identify, test, and implement solutions that address validated root causes, finding optimal settings for critical parameters.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Finding Optimal Protocol Settings")
        tab1, tab2 = st.tabs(["🧪 Design of Experiments (DOE)", "🤖 Bayesian Optimization"])
        with tab1:
            st.markdown("##### **Classical: Design of Experiments (DOE)**")
            doe_data = generate_doe_data(); fig_doe_main, fig_doe_interaction = plot_doe_effects(doe_data)
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(plot_doe_cube(doe_data), use_container_width=True)
            with col2: st.plotly_chart(fig_doe_main, use_container_width=True); st.plotly_chart(fig_doe_interaction, use_container_width=True)
            with st.expander("📊 How to Interpret These Charts"):
                st.markdown("""
                - **3D Cube Plot:** Visualizes the design space. Each corner is an experimental run. The color shows the measured `Library Yield`.
                - **Main Effects Plot:** Shows the average effect of each factor. A large bar (positive or negative) indicates a strong influence. Here, `Anneal Temp` has the largest (negative) effect.
                - **Interaction Plot:** Checks if the effect of one factor depends on the level of another. **Non-parallel lines indicate an interaction.** Here, the lines are not parallel, showing a strong interaction between `Anneal Temp` and `PCR Cycles`.
                - **Takeaway:** To maximize yield, we need to set `Anneal Temp` to Low (-1) and `PCR Cycles` to High (+1), and also consider their interaction.
                """)
        with tab2:
            st.markdown("##### **ML Augmentation: Bayesian Optimization**")
            st.sidebar.header("🔬 Simulators"); st.sidebar.markdown("---"); st.sidebar.subheader("Bayesian Optimization")
            x_range = np.linspace(0, 20, 400); true_func = get_true_bo_func
            if 'sampled_points' not in st.session_state: st.session_state.sampled_points = {'x': [2.0, 18.0], 'y': [true_func(2.0), true_func(18.0)]}
            if st.sidebar.button("Run Next Smart Experiment", key='bo_sample'):
                _, next_point = plot_bayesian_optimization_interactive(true_func, x_range, st.session_state.sampled_points)
                st.session_state.sampled_points['x'].append(next_point); st.session_state.sampled_points['y'].append(true_func(next_point))
            if st.sidebar.button("Reset Simulation", key='bo_reset'): st.session_state.sampled_points = {'x': [2.0, 18.0], 'y': [true_func(2.0), true_func(18.0)]}
            fig_bo, _ = plot_bayesian_optimization_interactive(true_func, x_range, st.session_state.sampled_points)
            st.plotly_chart(fig_bo, use_container_width=True)
            with st.expander("📊 How to Interpret This Chart"):
                st.markdown("""
                - **What it is:** A simulation of an intelligent search for an optimal parameter setting.
                - **True Performance Curve (dashed grey):** The "real" assay performance, which is unknown to the algorithm.
                - **Experiments Run (red 'x'):** The actual experiments performed.
                - **GP Model of Assay (blue line):** The algorithm's current belief about how the assay behaves, based on the experiments.
                - **Acquisition Fn (green line):** The algorithm's strategy for picking the next experiment. It balances exploring uncertain regions (wide confidence interval) and exploiting known high-performing regions.
                - **Takeaway:** The algorithm avoids wasting experiments on regions it knows are poor. It intelligently chooses points to quickly find the peak of the hidden performance curve, saving time and resources compared to a brute-force search.
                """)

    with st.container(border=True):
        st.subheader("2. Proactively Mitigating Risks")
        col3, col4 = st.columns(2)
        with col3: st.markdown("##### **Classical: FMEA**"); st.plotly_chart(plot_fmea_table(), use_container_width=True)
        with col4: st.markdown("##### **ML Augmentation: Predictive Maintenance**"); st.plotly_chart(plot_rul_prediction(generate_sensor_degradation_data()), use_container_width=True)

    st.success("""**🏆 Hybrid Strategy for the Improve Phase:**\n1. **Optimize with the Right Tool:** For optimizing a few (<5) parameters, **DOE** is the gold standard. For high-dimensional protocols, use **Bayesian Optimization** for its superior sample efficiency.\n2. **Mitigate Risks (Hybrid):** Use a classical **FMEA** to identify the highest-risk failure modes. For top risks related to equipment, build a **predictive maintenance (RUL) model**.\n3. **The Ultimate Hybrid ("Digital Twin"):** Use data from a space-filling **DOE** to train an ML model of your assay. Then, use **Bayesian Optimization** on this *in silico* twin to find the global optimum before one final confirmation experiment.""")

# ==============================================================================
# PAGE 5: CONTROL PHASE - LAB OPERATIONS & QC
# ==============================================================================
def show_control_phase():
    st.title("Control: Lab Operations & QC")
    st.markdown("**Objective:** To implement a robust Quality Control (QC) system to monitor the optimized assay in routine use, ensuring performance remains stable.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Monitoring for Stability: Statistical Process Control (SPC)")
        st.sidebar.header("🔬 Simulators"); st.sidebar.markdown("---"); st.sidebar.subheader("QC Simulator")
        shift_mag = st.sidebar.slider("Magnitude of Shift (in Std Devs)", 0.2, 3.0, 0.8, 0.1, key="ctrl_shift_mag")
        ewma_lambda = st.sidebar.slider("EWMA Lambda (λ)", 0.1, 0.5, 0.2, 0.05, help="Higher λ reacts faster to shifts.")
        chart_data = generate_control_chart_data(shift_magnitude=shift_mag)
        
        tab1, tab2, tab3 = st.tabs(["📊 Levey-Jennings", "📈 EWMA", "📉 CUSUM"])
        with tab1:
            st.markdown("##### **Classical: Levey-Jennings Chart (Shewhart)**")
            st.plotly_chart(plot_shewhart_chart(chart_data), use_container_width=True)
            with st.expander("📊 How to Interpret This Chart"):
                st.markdown("- **What it is:** The standard QC chart. It plots individual QC measurements against control limits set at ±3 standard deviations from the mean. \n- **Best for:** Detecting large, sudden process shifts. \n- **Limitation:** It is not sensitive to small, gradual drifts.")
        with tab2:
            st.markdown("##### **Advanced: EWMA Chart**")
            st.plotly_chart(plot_ewma_chart(chart_data, lambda_val=ewma_lambda), use_container_width=True)
            with st.expander("📊 How to Interpret This Chart"):
                st.markdown("- **What it is:** An Exponentially Weighted Moving Average chart. It gives more weight to recent data points, essentially having a 'memory'. \n- **Best for:** Detecting small, sustained drifts in the process mean (e.g., slow reagent degradation). \n- **Takeaway:** The EWMA chart (green line) will often detect a small shift much earlier than a standard Levey-Jennings chart.")
        with tab3:
            st.markdown("##### **Advanced: CUSUM Chart**")
            st.plotly_chart(plot_cusum_chart(chart_data), use_container_width=True)
            with st.expander("📊 How to Interpret This Chart"):
                st.markdown("- **What it is:** A Cumulative Sum chart. It plots the cumulative sum of deviations from a target value. \n- **Best for:** Detecting very small, persistent shifts in the process mean. It is the most sensitive of the three charts to tiny drifts. \n- **Takeaway:** When a process has a very small shift, the CUSUM value will steadily increase (or decrease) until it crosses the control limit (H), signaling a problem.")

    with st.container(border=True):
        st.subheader("2. Formalizing the Gains: The Control Plan & SOPs")
        st.plotly_chart(plot_control_plan(), use_container_width=True)

    st.success("""**🏆 Hybrid Strategy for the Control Phase:**\n1. **Monitor with Levey-Jennings:** Use a classical **Levey-Jennings chart** for primary positive/negative controls for simplicity and regulatory compliance.\n2. **Detect Drifts with EWMA:** For critical secondary metrics, use a more sensitive **EWMA chart** to detect slow reagent or instrument degradation.\n3. **Holistic QC with ML:** For each sample, run a **multivariate QC model** (like Hotelling's T²) on the full profile of NGS QC metrics to flag subtle issues.\n4. **Codify Everything:** The **Control Plan** and **SOPs** must document all charts, limits, and reaction plans.""")

# ==============================================================================
# PAGE 6 & 7: COMPARISON & MANIFESTO (UNCHANGED)
# ==============================================================================
def show_comparison_matrix():
    st.title("Head-to-Head: Classical DOE vs. ML/Bioinformatics")
    st.markdown("A visual comparison of the core philosophies and practical strengths of the two approaches.")
    st.markdown("---")
    with st.container(border=True): st.subheader("Strengths Profile: A Multi-Dimensional View"); st.plotly_chart(plot_comparison_radar(), use_container_width=True)
    with st.container(border=True): st.subheader("The Verdict: Which Approach Excels for Which Task?"); st.plotly_chart(plot_verdict_barchart(), use_container_width=True)

def show_hybrid_strategy():
    st.title("The Hybrid Lab Manifesto: The Future of Assay Development")
    st.markdown("The most competitive biotech organizations do not choose one over the other; they build a **Bio-AI framework** that fuses statistical rigor with machine learning's predictive power.")
    st.markdown("---")
    with st.container(border=True): st.subheader("The Philosophy of Synergy: Inference + Prediction"); st.plotly_chart(plot_synergy_diagram(), use_container_width=True)
    with st.container(border=True):
        st.subheader("Interactive Solution Recommender")
        guidance_data = get_guidance_data(); scenarios = list(guidance_data.keys())
        selected_scenario = st.selectbox("Choose your R&D scenario:", scenarios)
        if selected_scenario:
            st.markdown(f"##### Recommended Approach: {guidance_data[selected_scenario]['approach']}")
            st.markdown(f"**Rationale:** {guidance_data[selected_scenario]['rationale']}")
    with st.container(border=True):
        st.subheader("A Unified, Modern R&D Workflow")
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
