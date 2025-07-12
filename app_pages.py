"""
app_pages.py

This module contains the rendering logic for each main page of the Bio-AI
Excellence Framework application.

Each 'show_*_page' function is responsible for orchestrating the layout and
content of a single page. These functions follow a consistent pattern:
1.  Set the page title and introduction.
2.  Call helper functions from the `helpers` package to:
    a. Generate or load data (which will be cached).
    b. Train models or perform computations (which will be cached).
    c. Create Plotly visualizations.
3.  Arrange the content on the page using Streamlit layout primitives
    (e.g., st.columns, st.tabs, st.expander).

This modular approach ensures that the page-rendering logic is clean,
declarative, and decoupled from the underlying data and business logic.

Author: AI Engineering SME
Version: 23.1 (Commercial Grade Refactor)
Date: 2023-10-26
"""

import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# 1. EXPLICIT IMPORTS FROM REFACTORED HELPER MODULES
# ==============================================================================
# By importing explicitly, we improve code readability, prevent namespace
# pollution, and allow for better static analysis.
from helpers.styling import COLORS
from helpers.content import (
    get_qfd_expander_content, get_kano_expander_content,
    get_msa_expander_content, get_pccp_expander_content,
    get_guidance_data, render_workflow_step, get_workflow_css
)
from helpers.data_generators import (
    generate_nonlinear_data, generate_doe_data, generate_rsm_data,
    generate_fmea_data, generate_dfmea_data, generate_qfd_data,
    generate_kano_data, generate_pareto_data, generate_vsm_data,
    generate_risk_signal_data, generate_adverse_event_data,
    generate_pccp_data, generate_validation_data, generate_sensor_degradation_data,
    generate_control_chart_data, generate_anova_data, generate_hotelling_data
)
from helpers.ml_models import (
    train_regression_models, get_shap_explanation,
    perform_risk_signal_clustering, perform_text_clustering,
    perform_topic_modeling_on_capa
)
from helpers.visualizations import (
    plot_project_charter_visual, plot_sipoc_visual, plot_ctq_tree_plotly,
    plot_qfd_house_of_quality, plot_kano_visual, plot_voc_bubble_chart,
    plot_dfmea_table, plot_risk_signal_clusters, plot_gage_rr_pareto,
    plot_vsm, plot_process_mining_plotly, plot_capability_analysis_pro,
    plot_model_validation_ci, plot_fishbone_plotly, plot_pareto_chart,
    plot_anova_groups, plot_permutation_test, plot_regression_comparison,
    plot_shap_summary, plot_fault_tree_plotly, plot_5whys_diagram,
    plot_nlp_on_capa_logs, plot_doe_effects, plot_doe_cube, plot_rsm_contour,
    plot_bayesian_optimization_interactive, plot_rul_prediction,
    plot_shewhart_chart, plot_ewma_chart, plot_cusum_chart,

    plot_hotelling_t2_chart, plot_control_plan, plot_adverse_event_clusters,
    plot_pccp_monitoring, plot_comparison_radar, plot_verdict_barchart,
    plot_synergy_diagram
)

# ==============================================================================
# 2. PAGE-SPECIFIC HELPER FUNCTIONS (DRY Principle)
# ==============================================================================
# These helpers reduce code duplication within the page rendering functions.

def _display_chart_with_expander(
    chart_func,
    chart_args: dict,
    expander_title: str,
    expander_content_func,
    container_width: bool = True
):
    """
    Renders a chart and an expander with methodology details.

    :param chart_func: The plotting function to call.
    :param chart_args: A dictionary of arguments to pass to the plotting function.
    :param expander_title: The title for the st.expander.
    :param expander_content_func: A function that returns the markdown content.
    :param container_width: The use_container_width setting for the chart.
    """
    st.plotly_chart(chart_func(**chart_args), use_container_width=container_width)
    with st.expander(expander_title):
        st.markdown(expander_content_func())


# ==============================================================================
# PAGE 0: WELCOME & FRAMEWORK
# ==============================================================================
def show_welcome_page():
    """Renders the main landing page of the application."""
    st.title("Welcome to the Bio-AI Excellence Framework")
    st.markdown("##### An interactive playbook for developing and optimizing robust genomic assays and devices.")
    st.markdown("---")
    st.info("""
    **This application is designed for a technically proficient audience** (e.g., R&D Scientists, Bioinformaticians, Lab Directors, QA/RA Professionals). It moves beyond introductory concepts to demonstrate a powerful, unified framework that fuses the **inferential rigor of classical statistics** with the **predictive power of modern Machine Learning**.
    """)

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
    """Renders the content for the 'Define' phase of DMAIC."""
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
            weights, rel_df = generate_qfd_data() # Data is simple, no need to cache
            _display_chart_with_expander(
                plot_qfd_house_of_quality,
                {'weights': weights, 'rel_df': rel_df},
                "Methodology & Regulatory Significance (QFD)",
                get_qfd_expander_content
            )
            st.markdown("---")

            st.markdown("##### **Tool: Kano Model**")
            df_kano = generate_kano_data() # Cached in a real app if data is large
            _display_chart_with_expander(
                plot_kano_visual,
                {'df_kano': df_kano},
                "Methodology & Interpretation (Kano)",
                get_kano_expander_content
            )

        with tab2:
            st.markdown("##### **Tool: NLP on Scientific Literature (VOC Analysis)**")
            st.info("Demonstration using static data. In a real application, this would involve scraping PubMed or analyzing internal documents.")
            df_voc_data = pd.DataFrame({
                'Category': ['Biomarkers', 'Biomarkers', 'Methodology', 'Methodology', 'Performance', 'Performance'],
                'Topic': ['EGFR Variants', 'KRAS Hotspots', 'ddPCR', 'Shallow WGS', 'LOD <0.1%', 'Specificity >99%'],
                'Count': [180, 150, 90, 60, 250, 210],
                'Sentiment': [0.5, 0.4, -0.2, -0.4, 0.8, 0.7]
            })
            st.plotly_chart(plot_voc_bubble_chart(df_voc_data), use_container_width=True)
            st.markdown("---")

            st.markdown("##### **Tool: Data-Driven Feature Importance**")
            st.markdown("Using ML to support or challenge initial assumptions about critical parameters.")
            df_reg = generate_nonlinear_data()
            model_results = train_regression_models(df_reg, target_column='On_Target_Rate')
            shap_values = get_shap_explanation(model_results['rf_model'], model_results['X'])
            st.plotly_chart(plot_shap_summary(shap_values, model_results['X']), use_container_width=True)


    with st.container(border=True):
        st.subheader("3. Early Risk Assessment")
        st.markdown("Proactively identifying potential failures *before* they are locked into the product design.")
        tab3, tab4 = st.tabs(["🏛️ Classical: DFMEA", "🤖 ML Augmentation"])
        with tab3:
            st.markdown("##### **Tool: Design FMEA (DFMEA)**")
            df_dfmea = generate_dfmea_data()
            st.plotly_chart(plot_dfmea_table(df_dfmea), use_container_width=True)
        with tab4:
            st.markdown("##### **Tool: Unsupervised Clustering for Risk Signal Grouping**")
            df_risk_signals = generate_risk_signal_data()
            df_clustered = perform_risk_signal_clustering(df_risk_signals)
            st.plotly_chart(plot_risk_signal_clusters(df_clustered), use_container_width=True)


# ==============================================================================
# PAGE 2: MEASURE PHASE
# ==============================================================================
def show_measure_phase():
    """Renders the content for the 'Measure' phase of DMAIC."""
    st.title("🔬 Measure: Baseline & System Validation")
    st.markdown("**Objective:** To validate measurement systems, collect data, and establish a robust, data-driven baseline of the current process performance and capability.")
    st.markdown("> **Applicable Regulatory Stages:** FDA Process Validation (Stage 1), ICH Q8/Q11, Analytical Method Validation")
    st.markdown("---")

    # Initialize session state for interactive widgets ON THIS PAGE
    # This prevents state loss if widgets are placed in tabs.
    if 'measure_lsl' not in st.session_state:
        st.session_state.measure_lsl = 0.8
        st.session_state.measure_usl = 9.0
        st.session_state.measure_mean = 4.0
        st.session_state.measure_std = 0.5

    with st.container(border=True):
        st.subheader("1. Prerequisite: Measurement System Analysis (MSA)")
        st.warning("**You cannot trust your process data until you trust your measurement system.** MSA ensures observed variability comes from the biology, not the lab process.")
        df_gage = pd.DataFrame({
            'Source of Variation': ['Assay Variation (Biology)', 'Repeatability (Sequencer)', 'Reproducibility (Operator)'],
            'Contribution (%)': [92, 5, 3]
        })
        _display_chart_with_expander(
            plot_gage_rr_pareto,
            {'df_gage': df_gage},
            "Methodology & Regulatory Significance (Gage R&R)",
            get_msa_expander_content
        )

    with st.container(border=True):
        st.subheader("2. Establishing Baseline Assay Capability")
        
        # --- Sidebar for Interactive Controls ---
        # NOTE: In a REAL app, these would be in main_app.py to persist across pages.
        # For this self-contained example, we place them here and manage with session_state.
        with st.sidebar:
            st.header("🔬 Simulators")
            st.markdown("---")
            st.subheader("Assay Capability")
            st.slider("Lower Spec Limit (LSL)", 0.5, 2.0, key="measure_lsl")
            st.slider("Upper Spec Limit (USL)", 8.0, 10.0, key="measure_usl")
            st.slider("Assay Mean (μ)", 2.0, 8.0, key="measure_mean")
            st.slider("Assay Std Dev (σ)", 0.2, 2.0, key="measure_std")
        
        data = generate_process_data(
            st.session_state.measure_mean,
            st.session_state.measure_std,
            2000
        )
        fig_cap, cp, cpk = plot_capability_analysis_pro(
            data, st.session_state.measure_lsl, st.session_state.measure_usl
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(fig_cap, use_container_width=True)
        with col2:
            st.markdown("##### **Capability Indices**")
            cp_color = "success" if cp >= 1.33 else ("warning" if cp >= 1.0 else "error")
            cpk_color = "success" if cpk >= 1.33 else ("warning" if cpk >= 1.0 else "error")
            st.metric(label="Process Potential (Cp)", value=f"{cp:.2f}", help="Measures potential. Target: > 1.33")
            st.markdown(f'<hr style="margin-top:0; margin-bottom:0.5rem; border-color:{COLORS[cp_color]}">', unsafe_allow_html=True)
            st.metric(label="Process Capability (Cpk)", value=f"{cpk:.2f}", help="Measures performance. Target: > 1.33")
            st.markdown(f'<hr style="margin-top:0; margin-bottom:0.5rem; border-color:{COLORS[cpk_color]}">', unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: ANALYZE PHASE
# ==============================================================================
def show_analyze_phase():
    """Renders the content for the 'Analyze' phase of DMAIC."""
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
            df_pareto = generate_pareto_data()
            st.plotly_chart(plot_pareto_chart(df_pareto), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Finding the Drivers: Modeling Assay Performance")
        df_reg = generate_nonlinear_data()
        model_results = train_regression_models(df_reg, 'On_Target_Rate')
        shap_values = get_shap_explanation(model_results['rf_model'], model_results['X'])
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### **Classical vs. ML Model Fit**")
            st.plotly_chart(plot_regression_comparison(model_results), use_container_width=True)
        with col4:
            st.markdown("##### **ML Augmentation: XAI to Find Root Cause**")
            st.plotly_chart(plot_shap_summary(shap_values, model_results['X']), use_container_width=True)

    with st.container(border=True):
        st.subheader("3. Process Failure Analysis (CAPA & Deviations)")
        tab3, tab4 = st.tabs(["🏛️ Classical: FMEA, FTA, 5 Whys", "🤖 ML Augmentation: NLP on Logs"])
        with tab3:
            st.markdown("##### **Tool: Fault Tree Analysis (FTA)**")
            st.plotly_chart(plot_fault_tree_plotly(), use_container_width=True)
            st.markdown("---")
            st.markdown("##### **Tool: 5 Whys Analysis**")
            st.plotly_chart(plot_5whys_diagram(), use_container_width=True)
        with tab4:
            st.markdown("##### **Tool: NLP and Clustering on CAPA/Deviation Logs**")
            df_capa = generate_dfmea_data()
            df_topics = perform_topic_modeling_on_capa(df_capa, 'Description')
            st.plotly_chart(plot_nlp_on_capa_logs(df_topics), use_container_width=True)


# ==============================================================================
# PAGE 4: IMPROVE PHASE
# ==============================================================================
def show_improve_phase():
    """Renders the content for the 'Improve' phase of DMAIC."""
    st.title("⚙️ Improve: Optimization & Robustness")
    st.markdown("**Objective:** To identify, test, and implement solutions that address validated root causes, finding optimal settings for critical parameters to create a robust process.")
    st.markdown("> **Applicable Regulatory Stages:** ICH Q8 (Design Space), FDA Process Validation (Stage 1)")
    st.markdown("---")

    if 'bo_sampled_points' not in st.session_state:
        st.session_state.bo_sampled_points = {'x': [2.0, 18.0], 'y': [(np.sin(2.0*0.8)*15) + (np.cos(2.0*2.5)*5) - (2.0/10)**3, (np.sin(18.0*0.8)*15) + (np.cos(18.0*2.5)*5) - (18.0/10)**3]}

    with st.container(border=True):
        st.subheader("1. Design Space & Process Optimization")
        tab1, tab2 = st.tabs(["🧪 Design of Experiments (DOE) & RSM", "🤖 Bayesian Optimization"])
        with tab1:
            st.markdown("##### **Classical: Design of Experiments (DOE)**")
            doe_data = generate_doe_data()
            fig_doe_main, fig_doe_interaction = plot_doe_effects(doe_data)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_doe_cube(doe_data), use_container_width=True)
            with col2:
                st.plotly_chart(fig_doe_main, use_container_width=True)
                st.plotly_chart(fig_doe_interaction, use_container_width=True)
            st.markdown("---")
            st.markdown("##### **Classical: Response Surface Methodology (RSM)**")
            st.plotly_chart(plot_rsm_contour(generate_rsm_data()), use_container_width=True)

        with tab2:
            st.markdown("##### **ML Augmentation: Bayesian Optimization**")
            with st.sidebar:
                st.header("🔬 Simulators")
                st.markdown("---")
                st.subheader("Bayesian Optimization")
                if st.button("Run Next Smart Experiment", key='bo_sample'):
                    true_func_bo = lambda x: (np.sin(x*0.8)*15) + (np.cos(x*2.5)*5) - (x/10)**3
                    x_range_bo = np.linspace(0, 20, 400)
                    _, next_point = plot_bayesian_optimization_interactive(true_func_bo, x_range_bo, st.session_state.bo_sampled_points)
                    st.session_state.bo_sampled_points['x'].append(next_point)
                    st.session_state.bo_sampled_points['y'].append(true_func_bo(next_point))

                if st.button("Reset Simulation", key='bo_reset'):
                    st.session_state.bo_sampled_points = {'x': [2.0, 18.0], 'y': [(np.sin(2.0*0.8)*15) + (np.cos(2.0*2.5)*5) - (2.0/10)**3, (np.sin(18.0*0.8)*15) + (np.cos(18.0*2.5)*5) - (18.0/10)**3]}

            true_func_bo = lambda x: (np.sin(x*0.8)*15) + (np.cos(x*2.5)*5) - (x/10)**3
            x_range_bo = np.linspace(0, 20, 400)
            fig_bo, _ = plot_bayesian_optimization_interactive(true_func_bo, x_range_bo, st.session_state.bo_sampled_points)
            st.plotly_chart(fig_bo, use_container_width=True)

# ==============================================================================
# PAGE 5: CONTROL PHASE
# ==============================================================================
def show_control_phase():
    """Renders the content for the 'Control' phase of DMAIC."""
    st.title("📡 Control: Lab Operations & Post-Market Surveillance")
    st.markdown("**Objective:** To implement a robust Quality Control (QC) system to monitor the optimized process, ensuring performance remains stable and compliant over time, and to actively monitor post-market data.")
    st.markdown("> **Applicable Regulatory Stages:** Continued Process Verification (CPV, FDA Stage 3), Post-Market Surveillance (PMS)")
    st.markdown("---")

    with st.sidebar:
        st.header("🔬 Simulators")
        st.markdown("---")
        st.subheader("QC Simulator")
        shift_mag = st.slider("Magnitude of Shift (in Std Devs)", 0.2, 3.0, 0.8, 0.1, key="ctrl_shift_mag")
        ewma_lambda = st.slider("EWMA Lambda (λ)", 0.1, 0.5, 0.2, 0.05, help="Higher λ reacts faster to shifts.")
    
    chart_data = generate_control_chart_data(shift_magnitude=shift_mag)

    with st.container(border=True):
        st.subheader("1. Monitoring for Stability: Statistical Process Control (SPC)")
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
            df_hotelling = generate_hotelling_data()
            st.plotly_chart(plot_hotelling_t2_chart(df_hotelling), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Formalizing Gains & Post-Market Activities")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical: The Control Plan**")
            st.plotly_chart(plot_control_plan(), use_container_width=True)
        with col2:
            st.markdown("##### **ML Augmentation: PMS Signal Detection**")
            df_ae = generate_adverse_event_data()
            df_ae_clustered = perform_text_clustering(df_ae, 'description')
            st.plotly_chart(plot_adverse_event_clusters(df_ae_clustered), use_container_width=True)

    with st.container(border=True):
        st.subheader("3. Digital Health / SaMD Control Plan")
        df_pccp = generate_pccp_data()
        _display_chart_with_expander(
            plot_pccp_monitoring,
            {'df_pccp': df_pccp},
            "Methodology & Regulatory Significance (PCCP)",
            get_pccp_expander_content
        )

# ==============================================================================
# PAGE 6 & 7: COMPARISON & MANIFESTO
# ==============================================================================
def show_comparison_matrix():
    """Renders the comparison between classical stats and ML."""
    st.title("Head-to-Head: Classical DOE vs. ML/Bioinformatics")
    st.markdown("A visual comparison of the core philosophies and practical strengths of the two approaches.")
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Strengths Profile: A Multi-Dimensional View")
        st.plotly_chart(plot_comparison_radar(), use_container_width=True)
    with st.container(border=True):
        st.subheader("The Verdict: Which Approach Excels for Which Task?")
        st.plotly_chart(plot_verdict_barchart(), use_container_width=True)

def show_hybrid_manifesto():
    """Renders the final manifesto page, tying all concepts together."""
    st.title("🤝 The Hybrid Manifesto & GxP Compliance")
    st.markdown("The most competitive biotech organizations do not choose one methodology over the other; they build a **Bio-AI framework** that leverages the unique strengths of each to achieve superior outcomes while maintaining impeccable compliance.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("The Philosophy of Synergy: Inference + Prediction")
        st.plotly_chart(plot_synergy_diagram(), use_container_width=True)

    with st.container(border=True):
        st.subheader("Interactive Solution Recommender")
        guidance_data = get_guidance_data()
        scenarios = list(guidance_data.keys())
        selected_scenario = st.selectbox("Choose your R&D scenario:", scenarios)
        if selected_scenario:
            st.markdown(f"##### Recommended Approach: {guidance_data[selected_scenario]['approach']}")
            st.markdown(f"**Rationale:** {guidance_data[selected_scenario]['rationale']}")
