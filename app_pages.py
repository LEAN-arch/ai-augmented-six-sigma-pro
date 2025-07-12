# app_pages.py

import streamlit as st
import numpy as np
import pandas as pd

# Import all necessary helper functions from the single, definitive helper file.
from app_helpers import *


# ==============================================================================
# PAGE 0: WELCOME & FRAMEWORK
# ==============================================================================
def show_welcome_page():
    st.title("Welcome to the Bio-AI Excellence Framework")
    st.markdown("##### An interactive playbook for developing and optimizing robust genomic assays and devices.")
    st.markdown("---")
    st.info("""**This application is designed for a technically proficient audience** (e.g., R&D Scientists, Bioinformaticians, Lab Directors, QA/RA Professionals). It moves beyond introductory concepts to demonstrate a powerful, unified framework that fuses the **inferential rigor of classical statistics** with the **predictive power of modern Machine Learning**.""")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classical Six Sigma & Quality Tools")
        st.markdown("- **Core Strength:** Establishing causality, ensuring statistical rigor for regulatory submissions (e.g., FDA, ICH), and providing a structured, auditable trail (e.g., QFD, FMEA, DOE).")
    with col2:
        st.subheader("ML & Bioinformatics Augmentation")
        st.markdown("- **Core Strength:** Prediction, biomarker discovery, handling complexity, and extracting signals from noisy, high-dimensional data (e.g., NLP, Predictive Modeling, Clustering).")
    st.subheader("The Hybrid Lab Philosophy: Augmentation, Not Replacement")
    st.markdown("The most effective path to developing breakthrough diagnostics lies in the **synergistic integration** of these two disciplines. Use the navigation panel to explore the R&D lifecycle (framed as **DMAIC**). Each phase presents classical tools alongside their AI-augmented counterparts, enriched with their relevance to regulatory milestones.")
    st.success("Click on a phase in the sidebar to begin your exploration.")

# ==============================================================================
# PAGE 1: DEFINE PHASE
# ==============================================================================
def show_define_phase():
    st.title("🌀 Define: Clinical Need & Product Design")
    st.markdown("**Objective:** To clearly articulate the clinical problem, establish project goals, define the assay scope, and translate clinical needs into quantifiable 'Critical to Quality' (CTQ) assay performance characteristics.")
    st.markdown("> **Applicable Regulatory Stages:** FDA Design Controls (21 CFR 820.30), ICH Q8 (Pharmaceutical Development)")
    st.markdown("---")
    
    with st.container(border=True):
        st.subheader("1. The Mandate: Project Charter & High-Level Scope")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical Tool: Project Charter**")
            st.plotly_chart(plot_project_charter_visual(), use_container_width=True)
        with col2:
            st.markdown("##### **Classical Tool: SIPOC Diagram**")
            st.plotly_chart(plot_sipoc_visual(), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Requirements Translation & Design Input Prioritization")
        st.markdown("Translating the **Voice of the Customer (VOC)** into the **Voice of the Engineer**.")
        
        tab1, tab2 = st.tabs(["🏛️ Classical Tools", "🤖 ML Augmentation"])
        with tab1:
            st.markdown("##### **Tool: Critical to Quality (CTQ) Tree**")
            st.plotly_chart(plot_ctq_tree_plotly(), use_container_width=True)
            st.markdown("---")
            st.markdown("##### **Tool: Quality Function Deployment (QFD)**")
            st.plotly_chart(plot_qfd_house_of_quality(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance (QFD)"):
                st.markdown("""
                **Methodology:** QFD is a structured method for translating customer requirements into design specifications. The 'House of Quality' matrix maps customer needs to technical characteristics, creating a traceable link.
                
                **Regulatory Significance:** This creates a **documented, traceable link** between user needs and design inputs, a core requirement of FDA Design Controls. It provides objective evidence for *why* certain technical specifications were prioritized for the **Design History File (DHF)**.
                """)
            st.markdown("---")
            st.markdown("##### **Tool: Kano Model**")
            st.plotly_chart(plot_kano_visual(), use_container_width=True)
            
        with tab2:
            st.markdown("##### **Tool: NLP on Scientific Literature (VOC Analysis)**")
            st.plotly_chart(plot_voc_bubble_chart(), use_container_width=True)
            st.markdown("---")
            st.markdown("##### **Tool: Dimensionality Reduction (PCA) & Feature Importance**")
            df_reg = generate_nonlinear_data()
            _, model, X_reg = train_and_plot_regression_models(df_reg)
            st.plotly_chart(plot_shap_summary(model, X_reg), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** After collecting pilot data, an ML model (e.g., Random Forest) can be trained to predict a key outcome. **Principal Component Analysis (PCA)** can be used first to reduce the dimensionality of complex data, while eXplainable AI (XAI) tools like **SHAP** then determine which input features have the most impact on the outcome.
                
                **Regulatory Significance:** While not a replacement for QFD, this provides powerful, data-driven evidence to **support or challenge** the assumptions made in the QFD. If SHAP reveals a parameter thought to be unimportant is actually critical, this is a vital finding for de-risking the project and must be documented in the DHF as evidence of a data-driven design process.
                """)
    
    with st.container(border=True):
        st.subheader("3. Early Risk Assessment")
        st.markdown("Proactively identifying potential failures and emergent risks *before* they are locked into the product design.")
        tab3, tab4 = st.tabs(["🏛️ Classical: DFMEA & Fishbone", "🤖 ML Augmentation: Unsupervised Risk Clustering"])
        with tab3:
            st.markdown("##### **Tool: Design FMEA (DFMEA)**")
            st.plotly_chart(plot_dfmea_table(), use_container_width=True)
        with tab4:
            st.markdown("##### **Tool: Unsupervised Clustering for Risk Signal Grouping**")
            st.plotly_chart(plot_risk_signal_clusters(), use_container_width=True)

# ==============================================================================
# PAGE 2: MEASURE PHASE
# ==============================================================================
def show_measure_phase():
    st.title("🔬 Measure: Baseline & System Validation")
    st.markdown("**Objective:** To validate measurement systems, collect data, and establish a robust, data-driven baseline of the current process performance and capability.")
    st.markdown("> **Applicable Regulatory Stages:** FDA Process Validation (Stage 1), ICH Q8/Q11")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Prerequisite: Measurement System Analysis (MSA)")
        st.warning("**You cannot trust your process data until you trust your measurement system.** MSA ensures observed variability comes from the biology, not the lab process.")
        st.plotly_chart(plot_gage_rr_pareto(), use_container_width=True)
        with st.expander("Methodology & Regulatory Significance"):
            st.markdown("""
            **Methodology:** A Gage R&R (Repeatability & Reproducibility) study is a designed experiment to quantify the amount of variation in a measurement system. It partitions variance into its components: the part, the appraiser (operator), and the equipment.
            
            **Regulatory Significance:** Before any process characterization (ICH Q11) or validation (PPQ), the analytical methods used must themselves be validated. A Gage R&R study is the standard way to provide evidence that the measurement system is reliable and its variation is acceptable (typically <10% of total process tolerance). This is a prerequisite for all subsequent stages.
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
        fig_cap, cp, cpk = plot_capability_analysis_pro(data, lsl, usl)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(fig_cap, use_container_width=True)
        with col2:
            st.markdown("##### **Capability Indices**")
            cp_color = "success" if cp >= 1.33 else ("warning" if cp >= 1.0 else "error")
            cpk_color = "success" if cpk >= 1.33 else ("warning" if cp >= 1.0 else "error")
            st.metric(label="Process Potential (Cp)", value=f"{cp:.2f}", help="Measures potential. Target: > 1.33")
            st.markdown(f'<hr style="margin-top:0; margin-bottom:0.5rem; border-color:{COLORS[cp_color]}">', unsafe_allow_html=True)
            st.metric(label="Process Capability (Cpk)", value=f"{cpk:.2f}", help="Measures performance. Target: > 1.33")
            st.markdown(f'<hr style="margin-top:0; margin-bottom:0.5rem; border-color:{COLORS[cpk_color]}">', unsafe_allow_html=True)
            
        with st.expander("Methodology & Regulatory Significance"):
            st.markdown("""
            **Methodology:** Process Capability analysis uses metrics like **Cp and Cpk** to measure how well a process can meet its specification limits. Cpk is the key metric, as it accounts for both process variation and centering.
            
            **Regulatory Significance:** Establishing a baseline Cpk is a critical part of the 'Measure' phase. It quantifies the problem. Later, during **Process Validation (PPQ)**, a Cpk value ≥ 1.33 is the widely accepted industry standard to demonstrate that a process is robust and in a state of control.
            """)
            
    with st.container(border=True):
        st.subheader("4. Validating Supporting Models")
        st.markdown("Rigorously assessing the performance of any statistical or ML models used in the process.")
        st.plotly_chart(plot_model_validation_ci(), use_container_width=True)
        with st.expander("Methodology & Regulatory Significance"):
            st.markdown("""
            **Methodology:** Instead of reporting a single performance metric, a more rigorous approach is to calculate a **Confidence Interval (CI)**. Bootstrapping is a powerful method to do this by resampling the validation dataset thousands of times to create a distribution of the metric. **Cross-validation (CV)** is another key technique where the model is trained and tested on different subsets of the data to ensure it generalizes well.
            
            **Regulatory Significance:** Reporting a CI (e.g., "The model accuracy is 95% with a 95% CI of [93.5%, 96.5%]") is far more transparent to a regulator than a single point estimate. It demonstrates a deep understanding of the model's uncertainty and stability. This is especially critical when validating an ML model as part of a device or process.
            """)

# ==============================================================================
# PAGE 3: ANALYZE PHASE
# ==============================================================================
def show_analyze_phase():
    st.title("📈 Analyze: Root Cause & Failure Modes")
    st.markdown("**Objective:** To analyze data to identify, validate, and quantify the root cause(s) of poor performance, moving from *what* is failing to *why*.")
    st.markdown("> **Applicable Regulatory Stages:** CAPA (21 CFR 820.100), Quality Risk Management (ISO 14971, ICH Q9)")
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
        st.subheader("2. Statistical Root Cause Analysis: Comparing Groups")
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
        st.subheader("3. Finding the Drivers: Modeling Assay Performance")
        df_reg = generate_nonlinear_data()
        fig_reg, model, X_reg = train_and_plot_regression_models(df_reg)
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### **Classical vs. ML Model Fit**")
            st.plotly_chart(fig_reg, use_container_width=True)
        with col4:
            st.markdown("##### **ML Augmentation: XAI to Find Root Cause**")
            st.plotly_chart(plot_shap_summary(model, X_reg), use_container_width=True)
            
    with st.container(border=True):
        st.subheader("4. Process Failure Analysis")
        tab3, tab4 = st.tabs(["🏛️ Classical: FMEA & FTA", "🤖 ML Augmentation: NLP & Clustering on Logs"])
        with tab3:
            st.markdown("##### **Tool: Fault Tree Analysis (FTA)**")
            st.plotly_chart(plot_fault_tree_plotly(), use_container_width=True)
            st.markdown("##### **Tool: 5 Whys Analysis**")
            st.plotly_chart(plot_5whys_diagram(), use_container_width=True)
        with tab4:
            st.markdown("##### **Tool: NLP and Clustering on CAPA/Deviation Logs**")
            st.plotly_chart(plot_nlp_on_capa_logs(), use_container_width=True)

# ==============================================================================
# PAGE 4: IMPROVE PHASE
# ==============================================================================
def show_improve_phase():
    st.title("⚙️ Improve: Optimization & Robustness")
    st.markdown("**Objective:** To identify, test, and implement solutions that address validated root causes, finding optimal settings for critical parameters to create a robust process.")
    st.markdown("> **Applicable Regulatory Stages:** ICH Q8 (Design Space), FDA Process Validation (Stage 1)")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Finding Optimal Protocol Settings")
        tab1, tab2 = st.tabs(["🧪 Design of Experiments (DOE) & RSM", "🤖 Bayesian Optimization"])
        with tab1:
            st.markdown("##### **Classical: Design of Experiments (DOE)**")
            doe_data = generate_doe_data(); fig_doe_main, fig_doe_interaction = plot_doe_effects(doe_data)
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(plot_doe_cube(doe_data), use_container_width=True)
            with col2: st.plotly_chart(fig_doe_main, use_container_width=True); st.plotly_chart(fig_doe_interaction, use_container_width=True)
            st.markdown("---")
            st.markdown("##### **Classical: Response Surface Methodology (RSM)**")
            st.plotly_chart(plot_rsm_contour(generate_rsm_data()), use_container_width=True)
        with tab2:
            st.markdown("##### **ML Augmentation: Bayesian Optimization**")
            st.sidebar.header("🔬 Simulators"); st.sidebar.markdown("---"); st.sidebar.subheader("Bayesian Optimization")
            x_range = np.linspace(0, 20, 400); true_func = lambda x: (np.sin(x*0.8)*15) + (np.cos(x*2.5)*5) - (x/10)**3
            if 'sampled_points' not in st.session_state: st.session_state.sampled_points = {'x': [2.0, 18.0], 'y': [true_func(2.0), true_func(18.0)]}
            if st.sidebar.button("Run Next Smart Experiment", key='bo_sample'):
                _, next_point = plot_bayesian_optimization_interactive(true_func, x_range, st.session_state.sampled_points)
                st.session_state.sampled_points['x'].append(next_point); st.session_state.sampled_points['y'].append(true_func(next_point))
            if st.sidebar.button("Reset Simulation", key='bo_reset'): st.session_state.sampled_points = {'x': [2.0, 18.0], 'y': [true_func(2.0), true_func(18.0)]}
            fig_bo, _ = plot_bayesian_optimization_interactive(true_func, x_range, st.session_state.sampled_points)
            st.plotly_chart(fig_bo, use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Proactively Mitigating Risks in the Improved Process")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### **Classical: FMEA**")
            st.plotly_chart(plot_fmea_table(), use_container_width=True)
        with col4:
            st.markdown("##### **ML Augmentation: Predictive Maintenance**")
            st.plotly_chart(plot_rul_prediction(generate_sensor_degradation_data()), use_container_width=True)
            
# ==============================================================================
# PAGE 5: CONTROL PHASE
# ==============================================================================
def show_control_phase():
    st.title("📡 Control: Lab Operations & Post-Market Surveillance")
    st.markdown("**Objective:** To implement a robust Quality Control (QC) system to monitor the optimized process, ensuring performance remains stable and compliant over time, and to actively monitor post-market data.")
    st.markdown("> **Applicable Regulatory Stages:** Continued Process Verification (CPV, FDA Stage 3), Post-Market Surveillance (PMS)")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Monitoring for Stability: Statistical Process Control (SPC)")
        st.sidebar.header("🔬 Simulators"); st.sidebar.markdown("---"); st.sidebar.subheader("QC Simulator")
        shift_mag = st.sidebar.slider("Magnitude of Shift (in Std Devs)", 0.2, 3.0, 0.8, 0.1, key="ctrl_shift_mag")
        ewma_lambda = st.sidebar.slider("EWMA Lambda (λ)", 0.1, 0.5, 0.2, 0.05, help="Higher λ reacts faster to shifts.")
        chart_data = generate_control_chart_data(shift_magnitude=shift_mag)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Levey-Jennings", "📈 EWMA", "📉 CUSUM", "🤖 Multivariate QC"])
        with tab1:
            st.markdown("##### **Classical: Levey-Jennings Chart (Shewhart)**")
            st.plotly_chart(plot_shewhart_chart(chart_data), use_container_width=True)
        with tab2:
            st.markdown("##### **Advanced: EWMA Chart**")
            st.plotly_chart(plot_ewma_chart(chart_data, lambda_val=ewma_lambda), use_container_width=True)
        with tab3:
            st.markdown("##### **Advanced: CUSUM Chart**")
            st.plotly_chart(plot_cusum_chart(chart_data), use_container_width=True)
        with tab4:
            st.markdown("##### **ML Augmentation: Multivariate QC**")
            st.plotly_chart(plot_hotelling_t2_chart(), use_container_width=True)
            
    with st.container(border=True):
        st.subheader("2. Formalizing the Gains & Post-Market Activities")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical: The Control Plan**")
            st.plotly_chart(plot_control_plan(), use_container_width=True)
        with col2:
            st.markdown("##### **ML Augmentation: PMS Signal Detection**")
            st.plotly_chart(plot_adverse_event_clusters(), use_container_width=True)

    with st.container(border=True):
        st.subheader("3. Digital Health / SaMD Control Plan")
        st.markdown("Managing the lifecycle of an AI/ML medical device.")
        st.plotly_chart(plot_pccp_monitoring(), use_container_width=True)

# ==============================================================================
# PAGE 6 & 7: COMPARISON & MANIFESTO
# ==============================================================================
def show_comparison_matrix():
    st.title("Head-to-Head: Classical DOE vs. ML/Bioinformatics")
    st.markdown("A visual comparison of the core philosophies and practical strengths of the two approaches.")
    st.markdown("---")
    with st.container(border=True): st.subheader("Strengths Profile: A Multi-Dimensional View"); st.plotly_chart(plot_comparison_radar(), use_container_width=True)
    with st.container(border=True): st.subheader("The Verdict: Which Approach Excels for Which Task?"); st.plotly_chart(plot_verdict_barchart(), use_container_width=True)

def show_hybrid_manifesto():
    st.title("🤝 The Hybrid Manifesto & GxP Compliance")
    st.markdown("The most competitive biotech organizations do not choose one methodology over the other; they build a **Bio-AI framework** that leverages the unique strengths of each to achieve superior outcomes while maintaining impeccable compliance.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("The Philosophy of Synergy: Inference + Prediction")
        st.plotly_chart(plot_synergy_diagram(), use_container_width=True)
    
    with st.container(border=True):
        st.subheader("The Regulatory Reality: Navigating FDA & ICH Compliance")
        st.markdown("Understanding *why* and *how* to use these tools is critical for regulatory success. The choice is not arbitrary; it is governed by decades of policy and practice.")
        
        with st.expander("Expand: Why Classical Statistics is the Bedrock of Regulatory Submissions"):
            st.markdown("""
            Regulatory bodies like the FDA require that validation evidence be:
            - **Transparent and Interpretable:** Reviewers must be able to understand how a conclusion was reached.
            - **Reproducible:** The analysis must yield the same result given the same data.
            - **Based on Well-Established Principles:** Methods should have a long history of statistical validation.
            - **Hypothesis-Driven:** Claims must be tested against a pre-specified null hypothesis with a defined error rate (alpha).

            **Classical tools (DOE, ANOVA, SPC, t-tests) were designed to meet these criteria.** Their outputs (p-values, confidence intervals, Cpk) are the accepted currency of regulatory communication.
            """)

        with st.expander("Expand: How Machine Learning Can Be Used (with Critical Caveats)"):
            st.markdown("""
            ML is increasingly acceptable in **supporting roles**, provided its limitations are addressed.
            - **Exploratory Analysis:** Use ML to discover patterns and generate hypotheses, which are then confirmed with classical methods.
            - **Process Optimization:** Use ML to find optimal settings (e.g., Bayesian Optimization), which are then locked and validated in a formal PPQ.
            - **Post-Market Surveillance:** Use ML (especially NLP) to find signals in large, unstructured real-world data, a use case the FDA actively encourages under its Real-World Evidence (RWE) program.
            - **As the Device Itself (SaMD):** When the ML algorithm *is* the medical device, it falls under specific guidance like GMLP and PCCP.
            """)
        
        with st.expander("Expand: Key Regulatory Challenges for ML & The Path Forward (GMLP & PCCP)"):
            st.error("""
            **Key Challenges that Must be Addressed in a Submission:**
            - **Lack of Interpretability (The "Black Box" problem):** Must be mitigated with XAI tools like SHAP/LIME.
            - **Bias and Overfitting:** Must be addressed with rigorous validation on independent test sets and fairness audits.
            - **Model Drift & Lifecycle Management:** The plan for monitoring and updating a model post-deployment must be prospectively defined in a **Predetermined Change Control Plan (PCCP)**.
            
            The FDA's **Good Machine Learning Practice (GMLP)** principles provide a clear roadmap for developing trustworthy AI/ML medical devices.
            """)

    with st.container(border=True):
        st.subheader("Strategic Guidance: Tool-by-Tool Comparison")
        tool_comp_df = pd.DataFrame({
            'Use Case / Stage': ['Design & Development', 'Process Validation (PPQ)', 'Risk Management', 'CAPA / RCA', 'Ongoing Monitoring (CPV)', 'Post-Market Surveillance'],
            'Six Sigma Tools': ['QFD, DOE, DFMEA', 'Gage R&R, Cpk, SPC', 'FMEA, FTA, Fishbone', '5 Whys, Pareto', 'Control Charts, Cpk Trend', 'Trend Charts'],
            'Machine Learning Tools': ['Feature Selection, Clustering', 'Anomaly Detection, Time Series', 'Risk Scoring, Bayesian Nets', 'NLP on logs, Decision Trees', 'Predictive Maint., Drift Detect', 'NLP on complaints, RWE Mining'],
            'Verdict & Regulatory Fit': ['✅ Six Sigma is core for traceability. ⚠️ ML supports discovery but must be documented.', '✅ Six Sigma metrics are FDA standard. ✅ ML enhances detection if validated & locked.', '✅ FMEA/FTA are standard. ⚠️ ML risk models must be explainable.', '✅ Six Sigma tools are auditable. ⚠️ ML discovers patterns but needs traceability.', '✅ Six Sigma is standard for CPV. ✅ ML adds predictive power but doesn’t replace SPC.', '⚠️ Six Sigma limited. ✅ ML excels here and is encouraged by FDA for RWE.']
        })
        st.dataframe(tool_comp_df, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.subheader("Interactive Solution Recommender")
        guidance_data = get_guidance_data(); scenarios = list(guidance_data.keys())
        selected_scenario = st.selectbox("Choose your R&D scenario:", scenarios)
        if selected_scenario:
            st.markdown(f"##### Recommended Approach: {guidance_data[selected_scenario]['approach']}")
            st.markdown(f"**Rationale:** {guidance_data[selected_scenario]['rationale']}")
