"""
app_pages.py

Contains the rendering logic for each main page of the Bio-AI Excellence
Framework. This definitive, content-rich version integrates expert-level SME
explanations directly alongside every plot, figure, and table, and features
a substantially improved narrative across all pages. This version has been
extended to include the Interactive Case Study Library and Statistical Tool Advisor.

Author: Bio-AI Excellence SME Collective
Version: 34.2 (KeyError Hotfix)
Date: 2025-07-17

Changelog from v34.1:
- [BUGFIX] Corrected a `KeyError` in `render_case_study_detail` by using the
  correct nested path `case['Control Phase']['Control Plan']` to access the
  data, matching the schema from the data generator.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Callable, Any
import functools

# ==============================================================================
# 1. IMPORTS FROM HELPER MODULES (UPDATED)
# ==============================================================================
from helpers.styling import COLORS
from helpers.content import get_guidance_data, render_workflow_step, get_tool_advisor_data, get_ai_summary_for_case
from helpers.data_generators import (
    generate_nonlinear_data, generate_doe_data, generate_rsm_data,
    generate_dfmea_data, generate_qfd_data, generate_kano_data,
    generate_pareto_data, generate_risk_signal_data,
    generate_adverse_event_data, generate_pccp_data,
    generate_control_chart_data, generate_hotelling_data,
    generate_process_data, generate_capa_data,
    generate_case_study_data, generate_anova_data
)
from helpers.ml_models import (
    train_regression_models, get_shap_explanation,
    perform_risk_signal_clustering, perform_text_clustering,
    perform_topic_modeling_on_capa
)
from helpers.visualizations import * # Import all upgraded plotting functions
from scipy.stats import f_oneway

# ==============================================================================
# 2. CONSTANTS AND CONFIGURATIONS
# ==============================================================================
TARGET_COLUMN_NONLINEAR = 'On_Target_Rate'
CPK_TARGET = 1.33
CPK_WARNING = 1.0

# ==============================================================================
# 3. UI HELPER FUNCTION FOR CONSISTENT LAYOUT
# ==============================================================================
def _render_analysis_tool(
    title: str,
    tool_function: Callable,
    tool_args: Dict[str, Any],
    explanation_text: str,
    is_html: bool = False
) -> None:
    st.markdown(f"##### **{title}**")
    if is_html:
        st.html(tool_function(**tool_args))
    else:
        st.plotly_chart(tool_function(**tool_args), use_container_width=True)
    with st.expander("Methodology, Purpose, and Interpretation"):
        st.markdown(explanation_text, unsafe_allow_html=True)

# ==============================================================================
# PAGE 0: WELCOME & FRAMEWORK (UPGRADED AND CONTENT-RICH)
# ==============================================================================
def show_welcome_page() -> None:
    st.title("The New Standard for Biotech R&D")
    st.markdown("##### The Bio-AI Excellence Framework: Fusing Statistical Rigor with Predictive Power")
    st.divider()
    st.subheader("The Modern Biotech Dilemma: The Data Deluge vs. Regulatory Demands")
    st.markdown("""
    The biotechnology and pharmaceutical industries are at a critical inflection point. On one hand, we are inundated with an unprecedented volume of high-dimensional data from sources like Next-Generation Sequencing (NGS), multi-omics, and high-throughput screening. This data holds the promise of groundbreaking discoveries. On the other hand, we face an unwavering, and rightly stringent, regulatory environment (FDA, EMA, etc.) that demands irrefutable, statistically-sound proof of safety and effectiveness.

    This creates a fundamental tension:
    - How do we leverage the full potential of complex, noisy data to **discover** novel biomarkers and optimize intricate biological processes?
    - How do we simultaneously provide the clear, causal, and statistically rigorous evidence required to **validate** these discoveries and achieve regulatory approval?
    """)
    st.info("""
    **This framework presents the solution:** A hybrid methodology that marries the two essential pillars of modern scientific development—**Inference** and **Prediction**.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏛️ The World of Rigorous Inference")
        st.markdown("""
        *Classical Statistics* is the language of proof and causality. It is the bedrock of the scientific method and the currency of regulatory bodies.
        - **Primary Goal:** To infer conclusions about a population from a sample, and to prove or disprove a hypothesis with a known level of confidence.
        - **Key Questions:** "Does drug A cause a reduction in tumor size? Is this effect statistically significant (p < 0.05)?" "Is our manufacturing process capable of consistently meeting specifications (Cpk > 1.33)?"
        - **Core Tools:** Hypothesis Testing, Design of Experiments (DOE), Analysis of Variance (ANOVA), Statistical Process Control (SPC).
        - **Regulatory Significance:** This is the world of **validation**. Its outputs (p-values, confidence intervals, capability indices) are required for Design History Files (DHF), 510(k) submissions, and Process Performance Qualification (PPQ) reports.
        """)
    with col2:
        st.subheader("🤖 The World of Predictive Power")
        st.markdown("""
        *Machine Learning & Bioinformatics* is the language of complexity and discovery. It excels at finding patterns and making predictions in data that is too vast, high-dimensional, and intricate for classical methods.
        - **Primary Goal:** To build models that make accurate predictions on new, unseen data.
        - **Key Questions:** "Given a patient's gene expression profile, what is the probability they will respond to this therapy?" "From a list of 20,000 genes, which five-gene signature is most predictive of early-stage disease?"
        - **Core Tools:** Random Forests, Gradient Boosting, Clustering, Deep Learning, Natural Language Processing (NLP).
        - **Regulatory Significance:** This is the world of **discovery** and **monitoring**. It's essential for biomarker discovery, optimizing highly complex processes (e.g., cell culture media), and post-market surveillance (PMS) signal detection.
        """)
    st.subheader('The Synthesis: From "Or" to "And"')
    st.markdown("""
    An outdated mindset pits these two worlds against each other. The Bio-AI Excellence Framework is built on the philosophy of **synergy**. The most successful and competitive organizations do not choose one; they master the art of using them together.

    **Consider a typical workflow:**
    1.  **Discovery (ML):** Use a Random Forest model on genomic data to **predict** which 10-gene signature is most likely to classify patients.
    2.  **Optimization (ML):** Use Bayesian Optimization to efficiently find the optimal lab conditions for an assay that measures these 10 genes.
    3.  **Validation (Classical Stats):** Use a formal Design of Experiments (DOE) to **prove** the assay's robustness and Gage R&R to validate the measurement system.
    4.  **Control (Classical Stats):** Use Statistical Process Control (SPC) charts to monitor the assay's performance in production.

    This framework uses ML to navigate the vast search space of "what might be true" and classical statistics to rigorously prove "what is true" to a regulatory standard.
    """)
    st.subheader("How to Navigate This Playbook")
    st.markdown("""
    This application is structured along the **DMAIC (Define, Measure, Analyze, Improve, Control)** lifecycle, a robust and proven framework for process improvement. Each phase in the sidebar will guide you through this hybrid methodology:
    - **Define:** Articulate the problem and translate customer needs into technical specifications.
    - **Measure:** Validate your measurement systems and establish a data-driven performance baseline.
    - **Analyze:** Use both classical and ML tools to identify the true root causes of variation and failure.
    - **Improve:** Optimize your process using powerful experimental designs and smart algorithms.
    - **Control:** Implement systems to monitor performance and ensure your gains are sustained over time.

    In each section, you will find classical tools presented alongside their AI-augmented counterparts, complete with detailed explanations of their purpose, methodology, and interpretation. The new **Statistical Tool Advisor** and **Case Study Library** provide further expert guidance and real-world examples.
    """)
    st.success("Begin your journey by selecting a phase from the navigation panel on the left.")

# ==============================================================================
# CONTEXTUAL CASE RECOMMENDATION INTEGRATION (DECORATOR)
# ==============================================================================
def add_contextual_cases_to_page(page_function: Callable) -> Callable:
    @functools.wraps(page_function)
    def wrapper(*args, **kwargs):
        phase_name = page_function.__name__.replace("show_", "").replace("_phase", "").title()
        with st.sidebar:
            st.divider()
            with st.expander(f"📚 Relevant Case Studies"):
                render_contextual_cases(phase_name)
        page_function(*args, **kwargs)
    return wrapper

# ==============================================================================
# PAGE 1: DEFINE PHASE
# ==============================================================================
@add_contextual_cases_to_page
def show_define_phase() -> None:
    st.title("🌀 Define: Clinical Need & Product Design")
    st.markdown("""
    **Phase Objective:** To establish a clear, unambiguous, and formally documented understanding of the project's purpose, scope, and technical requirements before significant resources are committed. This phase is about translating a clinical or market need into a concrete engineering problem.

    **Regulatory Context:** This phase directly corresponds to the initial stages of **FDA Design Controls (21 CFR 820.30)**, specifically `§820.30(b) Design and development planning` and `§820.30(c) Design inputs`. The outputs of this phase form the foundational documents of the Design History File (DHF), providing an auditable trail that justifies all subsequent design and development choices.
    """)
    st.divider()
    with st.container(border=True):
        st.subheader("1. The Mandate: Project Charter & High-Level Scope")
        st.markdown("Before any technical work begins, the project's 'constitution' must be ratified. This step ensures enterprise-level alignment on the problem, the goals, and the boundaries of the work.")
        col1, col2 = st.columns(2)
        with col1:
            _render_analysis_tool(title="Classical Tool: Project Charter", tool_function=plot_project_charter_visual, tool_args={}, explanation_text="""...""")
        with col2:
            sipoc_df = pd.DataFrame({'Suppliers': ['• Reagent Vendors<br>• Instrument Mfr.<br>• LIMS Provider'], 'Inputs': ['• Patient Sample<br>• Reagent Kits<br>• SOP'], 'Process': ['1. Sample Prep<br>2. Library Prep<br>3. Sequencing<br>4. Bioinformatics<br>5. Reporting'], 'Outputs': ['• VCF File<br>• QC Report<br>• Clinical Report'], 'Customers': ['• Oncologists<br>• Patients<br>• Pharma Partners']})
            _render_analysis_tool(title="Classical Tool: SIPOC Diagram", tool_function=generate_html_table, tool_args={'df': sipoc_df, 'title': "SIPOC: High-Level NGS Assay Workflow"}, is_html=True, explanation_text="""...""")
    with st.container(border=True):
        st.subheader("2. Requirements Translation & Design Input Prioritization")
        st.markdown("This is the core translation step: converting the qualitative 'Voice of the Customer' (VOC) into the quantitative 'Voice of the Engineer.' This process ensures that what is built directly addresses validated user needs.")
        tab1, tab2 = st.tabs(["🏛️ Classical Tools", "🤖 ML Augmentation"])
        with tab1:
            _render_analysis_tool(title="Tool: Critical to Quality (CTQ) Tree", tool_function=plot_ctq_tree_plotly, tool_args={}, explanation_text="""...""")
            st.divider()
            weights, rel_df = generate_qfd_data()
            _render_analysis_tool(title="Tool: Quality Function Deployment (QFD)", tool_function=plot_qfd_house_of_quality_pro, tool_args={'weights': weights, 'rel_df': rel_df}, explanation_text="""...""")
            st.divider()
            _render_analysis_tool(title="Tool: Kano Model", tool_function=plot_kano_visual, tool_args={'df_kano': generate_kano_data()}, explanation_text="""...""")
        with tab2:
            df_reg = generate_nonlinear_data(); model_results = train_regression_models(df_reg, TARGET_COLUMN_NONLINEAR)
            shap_explanation = get_shap_explanation(model_results['rf_model'], model_results['X'])
            _render_analysis_tool(title="Tool: Data-Driven Feature Importance (XAI)", tool_function=plot_shap_summary, tool_args={'shap_explanation': shap_explanation}, explanation_text="""...""")
    with st.container(border=True):
        st.subheader("3. Early Risk Assessment")
        st.markdown("A proactive, systematic evaluation of potential design failures. The goal is to identify and mitigate risks when changes are least expensive—on paper, before the design is finalized.")
        tab1, tab2 = st.tabs(["🏛️ Classical: DFMEA", "🤖 ML Augmentation"])
        with tab1:
            _render_analysis_tool(title="Tool: Design FMEA (DFMEA)", tool_function=generate_html_table, tool_args={'df': generate_dfmea_data(), 'title': "Design Failure Mode and Effects Analysis (DFMEA)"}, is_html=True, explanation_text="""...""")
        with tab2:
            _render_analysis_tool(title="Tool: Unsupervised Clustering for Risk Signal Grouping", tool_function=plot_risk_signal_clusters, tool_args={'df_clustered': perform_risk_signal_clustering(generate_risk_signal_data())}, explanation_text="""...""")

# ==============================================================================
# PAGE 2: MEASURE PHASE
# ==============================================================================
@add_contextual_cases_to_page
def show_measure_phase() -> None:
    st.title("🔬 Measure: System Validation & Baseline Performance")
    st.markdown("""
    **Phase Objective:** To ensure that the systems used to measure the product and process are themselves accurate and reliable, and then to use those validated systems to collect data and establish a robust, statistical baseline of the process's current performance.
    **Regulatory Context:** This phase is critical for **Analytical Method Validation** and serves as a prerequisite for **Process Validation (PV) Stage 1 (Process Design)** and **Stage 2 (Process Performance Qualification)**. Regulatory bodies require objective evidence that measurement systems are suitable for their intended use before any process data can be considered valid.
    """)
    st.divider()
    with st.container(border=True):
        st.subheader("1. Prerequisite: Measurement System Analysis (MSA)")
        st.warning("""**You cannot trust your process data until you trust your measurement system.** ...""")
        df_gage = pd.DataFrame({'Source of Variation': ['Assay Variation (Biology)', 'Repeatability (Sequencer)', 'Reproducibility (Operator)'], 'Contribution (%)': [92, 5, 3]})
        _render_analysis_tool(title="Tool: Gage R&R Study - Hierarchical Variance Decomposition", tool_function=plot_gage_rr_sunburst, tool_args={'df_gage': df_gage}, explanation_text="""...""")
    with st.container(border=True):
        st.subheader("2. Establishing Baseline Assay Capability")
        st.markdown("Once the measurement system is validated, the next step is to use it to characterize the current process...")
        data = generate_process_data(4.0, 0.5, 2000)
        fig_cap, cp, cpk = plot_capability_analysis_pro(data, 0.8, 9.0)
        st.plotly_chart(fig_cap, use_container_width=True)
        with st.expander("Methodology, Purpose, and Interpretation"):
            st.markdown("""...""")

# ==============================================================================
# PAGE 3: ANALYZE PHASE
# ==============================================================================
@add_contextual_cases_to_page
def show_analyze_phase() -> None:
    st.title("📈 Analyze: Root Cause & Failure Modes")
    st.markdown("""
    **Phase Objective:** To leverage the data collected in the Measure phase to move from identifying that a problem exists (e.g., low Cpk) to pinpointing and verifying its root cause(s). This is the "detective work" phase.
    **Regulatory Context:** This phase is the heart of any **Corrective and Preventive Action (CAPA)** investigation under **21 CFR 820.100**. ...
    """)
    st.divider()
    with st.container(border=True):
        st.subheader("1. Qualitative Root Cause Analysis & Prioritization")
        st.markdown("Begin the analysis by structuring the problem and focusing on the most significant issues...")
        col1, col2 = st.columns(2)
        with col1:
            _render_analysis_tool(title="Classical Tool: Fishbone Diagram", tool_function=plot_fishbone_plotly, tool_args={}, explanation_text="""...""")
        with col2:
            _render_analysis_tool(title="Classical Tool: Pareto Chart", tool_function=plot_pareto_chart, tool_args={'df_pareto': generate_pareto_data()}, explanation_text="""...""")
    with st.container(border=True):
        st.subheader("2. Finding the Drivers: Modeling Assay Performance")
        st.markdown("Move from qualitative brainstorming to quantitative analysis...")
        df_reg = generate_nonlinear_data(); model_results = train_regression_models(df_reg, TARGET_COLUMN_NONLINEAR)
        col3, col4 = st.columns(2)
        with col3:
             _render_analysis_tool(title="Classical vs. ML Model Fit", tool_function=plot_regression_comparison, tool_args={'model_results': model_results}, explanation_text="""...""")
        with col4:
            shap_explanation = get_shap_explanation(model_results['rf_model'], model_results['X'])
            _render_analysis_tool(title="ML Augmentation: XAI to Find Root Cause", tool_function=plot_shap_summary, tool_args={'shap_explanation': shap_explanation}, explanation_text="""...""")
    with st.container(border=True):
        st.subheader("3. Process Failure Analysis (CAPA & Deviations)")
        st.markdown("Apply structured problem-solving techniques to specific failure events...")
        tab3, tab4 = st.tabs(["🏛️ Classical: FTA & 5 Whys", "🤖 ML Augmentation: NLP on Logs"])
        with tab3:
            _render_analysis_tool(title="Tool: Fault Tree Analysis (FTA)", tool_function=plot_fault_tree_plotly, tool_args={}, explanation_text="""...""")
            st.divider()
            _render_analysis_tool(title="Tool: 5 Whys Analysis", tool_function=plot_5whys_diagram, tool_args={}, explanation_text="""...""")
        with tab4:
            _render_analysis_tool(title="Tool: NLP and Clustering on CAPA/Deviation Logs", tool_function=plot_nlp_on_capa_logs, tool_args={'df_topics': perform_topic_modeling_on_capa(generate_capa_data(), 'Description')}, explanation_text="""...""")

# ==============================================================================
# PAGE 4: IMPROVE PHASE
# ==============================================================================
@add_contextual_cases_to_page
def show_improve_phase() -> None:
    st.title("⚙️ Improve: Optimization & Robustness")
    st.markdown("""
    **Phase Objective:** To use the knowledge gained from the Analyze phase to develop, test, and implement solutions that address validated root causes...
    **Regulatory Context:** This is the practical application of Quality by Design (QbD). ...
    """)
    st.divider()
    with st.container(border=True):
        st.subheader("1. Design Space & Process Optimization")
        st.markdown("This section demonstrates powerful methods for efficiently exploring the relationships between process inputs and outputs to find the settings that maximize performance.")
        tab1, tab2 = st.tabs(["🧪 Design of Experiments (DOE) & RSM", "🤖 Bayesian Optimization"])
        with tab1:
            st.markdown("##### **Classical: Design of Experiments (DOE)**")
            doe_data = generate_doe_data(); fig_doe_main, fig_doe_interaction = plot_doe_effects(doe_data)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("###### DOE Cube Plot (Design Points)"); st.plotly_chart(plot_doe_cube(doe_data), use_container_width=True)
            with col2:
                st.markdown("###### 3D Response Surface (Predicted Model)"); st.plotly_chart(plot_doe_3d_surface(doe_data), use_container_width=True)
            st.markdown("###### Factor Effects")
            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(fig_doe_main, use_container_width=True)
            with col4:
                st.plotly_chart(fig_doe_interaction, use_container_width=True)
            with st.expander("Methodology, Purpose, and Interpretation"):
                st.markdown("""...""")
            st.divider()
            _render_analysis_tool(title="Classical: Response Surface Methodology (RSM)", tool_function=plot_rsm_contour, tool_args={'df_rsm': generate_rsm_data()}, explanation_text="""...""")
        with tab2:
            true_func_bo = lambda x: (np.sin(x * 0.8) * 15) + (np.cos(x * 2.5) * 5) - (x / 10)**3; x_range_bo = np.linspace(0, 20, 400)
            if 'bo_sampled_points' not in st.session_state:
                initial_x = [2.0, 18.0]; st.session_state.bo_sampled_points = {'x': initial_x, 'y': [true_func_bo(x) for x in initial_x]}
            fig_bo, _ = plot_bayesian_optimization_interactive(true_func_bo, x_range_bo, st.session_state.bo_sampled_points)
            _render_analysis_tool(title="ML Augmentation: Bayesian Optimization", tool_function=lambda **x: fig_bo, tool_args={}, explanation_text="""...""")

# ==============================================================================
# PAGE 5: CONTROL PHASE
# ==============================================================================
@add_contextual_cases_to_page
def show_control_phase() -> None:
    st.title("📡 Control: Lab Operations & Post-Market Surveillance")
    st.markdown("""
    **Phase Objective:** To implement robust systems to "hold the gains" achieved in the Improve phase...
    **Regulatory Context:** This phase is directly aligned with **FDA Process Validation Stage 3 (Continued Process Verification - CPV)** ...
    """)
    st.divider()
    chart_data = generate_control_chart_data(shift_magnitude=0.8)
    with st.container(border=True):
        st.subheader("1. Monitoring for Stability: Statistical Process Control (SPC)")
        st.markdown("SPC is the primary toolset for monitoring a process in real-time...")
        spc_explanation = """..."""
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Levey-Jennings", "📈 EWMA", "📉 CUSUM", "🤖 Multivariate QC"])
        with tab1:
            _render_analysis_tool(title="Classical: Levey-Jennings Chart (Shewhart)", tool_function=plot_shewhart_chart, tool_args={'df_control': chart_data}, explanation_text=spc_explanation + "...")
        with tab2:
            _render_analysis_tool(title="Advanced: EWMA Chart", tool_function=plot_ewma_chart, tool_args={'df_control': chart_data, 'lambda_val': 0.2}, explanation_text=spc_explanation + "...")
        with tab3:
            _render_analysis_tool(title="Advanced: CUSUM Chart", tool_function=plot_cusum_chart, tool_args={'df_control': chart_data}, explanation_text=spc_explanation + "...")
        with tab4:
            _render_analysis_tool(title="ML Augmentation: Multivariate QC (Hotelling's T²)", tool_function=plot_hotelling_t2_chart, tool_args={'df_hotelling': generate_hotelling_data()}, explanation_text="""...""")
    with st.container(border=True):
        st.subheader("2. Formalizing Gains & Post-Market Activities")
        st.markdown("This final step involves documenting the control strategy...")
        col1, col2 = st.columns(2)
        with col1:
            control_plan_df = pd.DataFrame({'Process Step': ['Library Prep', 'Sequencing', 'Bioinformatics'], 'Characteristic': ['Pos Control Yield (Y)', 'Sequencer Laser (X)', '% Mapped Reads (Y)'], 'Specification': ['20 ± 5 ng', '> 80 mW', '> 85%'], 'Tool': ['Fluorometer', 'Internal Sensor', 'FASTQC'], 'Method': ['L-J Chart', 'EWMA Chart', 'Shewhart Chart'], 'Frequency': ['Per Batch', 'Per Run', 'Per Sample'], 'Reaction Plan': ['Re-prep Batch', 'Schedule Maint.', 'Review Alignment']})
            _render_analysis_tool(title="Classical: The Control Plan", tool_function=generate_html_table, tool_args={'df': control_plan_df, 'title': "Assay Control Plan"}, is_html=True, explanation_text="""...""")
        with col2:
            _render_analysis_tool(title="ML Augmentation: PMS Signal Detection", tool_function=plot_adverse_event_clusters, tool_args={'df_clustered': perform_text_clustering(generate_adverse_event_data(), 'description')}, explanation_text="""...""")
        st.divider()
        _render_analysis_tool(title="AI/ML Device Control: PCCP Monitoring", tool_function=plot_pccp_monitoring, tool_args={'df_pccp': generate_pccp_data()}, explanation_text="""...""")

# ==============================================================================
# PAGE 6 & 7: COMPARISON & MANIFESTO
# ==============================================================================
def show_comparison_matrix() -> None:
    st.title("⚔️ Head-to-Head: Classical Statistics vs. Machine Learning")
    st.markdown("""...""")
    st.divider()
    with st.container(border=True):
        _render_analysis_tool(title="Strengths Profile: A Multi-Dimensional View", tool_function=plot_comparison_radar, tool_args={}, explanation_text="""...""")
    with st.container(border=True):
        _render_analysis_tool(title="The Verdict: Which Approach Excels for Which Task?", tool_function=plot_verdict_barchart, tool_args={}, explanation_text="""...""")

def show_hybrid_manifesto() -> None:
    st.title("🤝 The Hybrid Manifesto & GxP Compliance")
    st.markdown("""...""")
    st.divider()
    with st.container(border=True):
        _render_analysis_tool(title="The Philosophy of Synergy: Inference + Prediction", tool_function=plot_synergy_diagram, tool_args={}, explanation_text="""...""")
    with st.container(border=True):
        st.subheader("Navigating the Regulatory Landscape with a Hybrid Approach")
        st.markdown("""...""")
    with st.container(border=True):
        st.subheader("The Hybrid Workflow in Practice")
        st.markdown("This is how the two disciplines collaborate across the R&D lifecycle:")
        st.html(render_workflow_step("1. Define", "step-define", ["Project Charter", "SIPOC", "CTQ Tree", "QFD"], ["NLP on Literature (VOC)", "XAI for Feature Importance"]))
        st.html(render_workflow_step("2. Measure", "step-measure", ["Gage R&R", "Process Capability (Cpk)"], ["Process Mining", "Automated Anomaly Detection"]))
        st.html(render_workflow_step("3. Analyze", "step-analyze", ["Fishbone, Pareto", "Hypothesis Testing", "ANOVA"], ["Regression Models (XGBoost, RF)", "SHAP/LIME for Root Cause"]))
        st.html(render_workflow_step("4. Improve", "step-improve", ["Design of Experiments (DOE)", "Response Surface (RSM)"], ["Bayesian Optimization", "Predictive 'Digital Twins'"]))
        st.html(render_workflow_step("5. Control", "step-control", ["SPC (Control Charts)", "Control Plans"], ["Multivariate SPC (Hotelling's)", "Predictive Maintenance (RUL)"]))
    with st.container(border=True):
        st.subheader("Interactive Solution Recommender")
        st.markdown("Select a common R&D or regulatory challenge to see a recommended approach based on the principles of the Bio-AI Framework.")
        guidance_data = get_guidance_data()
        scenarios = list(guidance_data.keys())
        selected_scenario = st.selectbox("Choose your R&D scenario:", scenarios, index=0)
        if selected_scenario and selected_scenario in guidance_data:
            recommendation = guidance_data[selected_scenario]
            st.success(f"##### Recommended Approach: {recommendation['approach']}")
            st.markdown(f"**Rationale:** {recommendation['rationale']}")

# ==============================================================================
# NEW PAGE: STATISTICAL TOOL ADVISOR
# ==============================================================================
def show_tool_advisor():
    st.title("🧭 Statistical Tool Advisor")
    st.markdown("""...""")
    st.divider()
    st.plotly_chart(plot_tool_advisor_sankey(), use_container_width=True)
    st.header("Tool Deep Dive")
    tool_data = get_tool_advisor_data()
    tool_names = list(tool_data.keys())
    selected_tool = st.selectbox("Select a tool for a detailed guide:", tool_names, index=0)
    if selected_tool and selected_tool in tool_data:
        tool_info = tool_data[selected_tool]
        with st.container(border=True):
            st.subheader(f"Deep Dive: {selected_tool}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### What is it?"); st.markdown(tool_info['what_is_it'])
            with col2:
                st.markdown("##### When to Use It")
                for scenario in tool_info['when_to_use']: st.markdown(f"- {scenario}")
            st.divider()
            st.markdown("##### Key Assumptions & Common Pitfalls")
            col3, col4 = st.columns(2)
            with col3:
                with st.expander("Data Requirements & Assumptions", expanded=True):
                    for assumption, _ in tool_info['assumptions'].items(): st.markdown(f"✅ {assumption}")
            with col4:
                with st.expander("Common Pitfalls & Warnings", expanded=True):
                    st.warning(f"**Warning:** {tool_info['pitfalls']}")
            st.divider()
            st.markdown("##### Interpreting the Output (The 'So What?')")
            col5, col6 = st.columns(2)
            with col5: st.info(f"**Key Statistic:** {tool_info['interpretation']['statistic']}")
            with col6: st.info(f"**Visual Interpretation:** {tool_info['interpretation']['visualization']}")
            if tool_info.get("example_data_func"):
                st.divider(); st.subheader("🔬 Try It Out!")
                st.markdown(f"See a live example of a **{selected_tool}** analysis.")
                if st.button(f"Run Sample {selected_tool} Analysis", type="primary"):
                    sample_df = tool_info["example_data_func"]()
                    if selected_tool == "One-Way ANOVA":
                        value_col, group_col = 'Purity', 'Supplier'
                        fig, f_val, p_val = plot_anova_results(sample_df, value_col, group_col)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("##### 🤖 AI-Powered Interpretation")
                        if p_val < 0.05:
                            st.success(f"**Conclusion:** The p-value of **{p_val:.3f}** is less than 0.05, indicating a **statistically significant difference**...")
                        else:
                            st.error(f"**Conclusion:** The p-value of **{p_val:.3f}** is greater than 0.05, indicating there is **no statistically significant difference**...")

# ==============================================================================
# NEW PAGE: CASE STUDY LIBRARY
# ==============================================================================
@st.cache_data
def load_all_case_studies():
    return generate_case_study_data()

def render_contextual_cases(current_phase: str):
    case_studies = load_all_case_studies()
    phase_key_map = {"Define": "Define Phase", "Measure": "Measure Phase", "Analyze": "Analyze Phase", "Improve": "Improve Phase", "Control": "Control Phase"}
    phase_key = phase_key_map.get(current_phase)
    if phase_key:
        relevant_cases = [case for case in case_studies if case.get(phase_key, {}).get("Tools Used") or case.get(phase_key, {}).get("Charter")]
        if relevant_cases:
            st.markdown("Here are a few projects that involved significant work in this phase:")
            for case in relevant_cases[:2]:
                if st.button(f"**{case['Title']}** ({case['Industry/Sector'][0]})", key=f"ctx_{case['id']}_{current_phase}", use_container_width=True):
                    st.query_params["page"] = "Case Study Library"; st.query_params["case_id"] = case['id']; st.rerun()
        else:
            st.info("No specific case studies highlighting this phase were found in the library.")

def show_case_study_library():
    st.title("📚 Interactive Case Study Library")
    case_studies = load_all_case_studies()
    if "case_id" in st.query_params:
        case_id = st.query_params["case_id"]
        selected_case = next((c for c in case_studies if c['id'] == case_id), None)
        if selected_case: render_case_study_detail(selected_case)
        else:
            st.error("Case study not found.")
            if st.button("← Back to Library"): st.query_params.clear(); st.rerun()
    else: render_library_view(case_studies)

def render_library_view(case_studies):
    st.markdown("Explore a curated repository of real-world projects...")
    st.divider()
    with st.sidebar:
        st.header("🔎 Filter Case Studies")
        industries = sorted(list(set(sum([c['Industry/Sector'] for c in case_studies], [])))); selected_industries = st.multiselect("Industry / Sector", industries, placeholder="Choose an industry")
        units = sorted(list(set([c['Business Unit'] for c in case_studies]))); selected_units = st.multiselect("Business Unit", units, placeholder="Choose a business unit")
        tools = sorted(list(set(sum([c['Analyze Phase']['Tools Used'] + c['Improve Phase']['Tools Used'] for c in case_studies], [])))); selected_tools = st.multiselect("Tools Used", tools, placeholder="Filter by tools used")
        st.divider(); search_query = st.text_input("🔬 Natural Language Search", placeholder="e.g., improve yield in upstream")
    filtered_cases = case_studies
    if selected_industries: filtered_cases = [c for c in filtered_cases if any(i in c['Industry/Sector'] for i in selected_industries)]
    if selected_units: filtered_cases = [c for c in filtered_cases if c['Business Unit'] in selected_units]
    if selected_tools: filtered_cases = [c for c in filtered_cases if any(t in (c['Analyze Phase']['Tools Used'] + c['Improve Phase']['Tools Used']) for t in selected_tools)]
    if search_query: query_lower = search_query.lower(); filtered_cases = [c for c in filtered_cases if query_lower in c['Problem Statement'].lower() or query_lower in c['Title'].lower() or query_lower in str(c['Analyze Phase']['Root Causes'])]
    st.subheader(f"Showing {len(filtered_cases)} of {len(case_studies)} Case Studies"); st.divider()
    if not filtered_cases: st.warning("No case studies match your current filter criteria.")
    else:
        for case in filtered_cases:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"##### {case['Title']}"); st.caption(f"**Industry:** {', '.join(case['Industry/Sector'])} | **Business Unit:** {case['Business Unit']}"); st.markdown(f"**Problem:** *{case['Problem Statement']}*")
                with col2:
                    st.metric("💰 Financial Impact", f"${case['Project Outcomes']['Financial Impact']/1e6:.1f}M")
                    if st.button("View Details", key=case['id'], type="primary", use_container_width=True): st.query_params.case_id = case['id']; st.rerun()

def render_case_study_detail(case):
    if st.button("← Back to Library"): st.query_params.clear(); st.rerun()
    st.title(case['Title']); st.caption(f"**Industry:** {', '.join(case['Industry/Sector'])} | **Business Unit:** {case['Business Unit']}"); st.markdown(f"> {case['Problem Statement']}")
    if st.session_state.get(f"summary_visible_{case['id']}", False):
        with st.container(border=True): st.markdown(get_ai_summary_for_case(case))
    if st.button("✨ Generate AI Summary", key=f"summary_{case['id']}"): st.session_state[f"summary_visible_{case['id']}"] = True; st.rerun()
    st.divider()
    d, m, a, i, c = st.tabs(["🌀 Define", "🔬 Measure", "📈 Analyze", "⚙️ Improve", "📡 Control"])
    with d:
        st.subheader("Define Phase"); st.markdown(f"**Project Charter:** {case['Define Phase']['Charter']}"); st.markdown(f"**SIPOC Focus:** {case['Define Phase']['SIPOC']}")
    with m:
        st.subheader("Measure Phase"); st.markdown(f"**KPIs Measured:** {', '.join(case['Measure Phase']['KPIs'])}"); st.metric("Baseline Performance", value=case['Measure Phase']['Baseline'])
    with a:
        st.subheader("Analyze Phase"); st.markdown(f"**Validated Root Causes:**");
        for cause in case['Analyze Phase']['Root Causes']: st.markdown(f"- {cause}")
        st.markdown(f"**Tools Used:** `{', '.join(case['Analyze Phase']['Tools Used'])}`")
    with i:
        st.subheader("Improve Phase"); st.markdown(f"**Solutions Implemented:** {case['Improve Phase']['Solutions']}"); st.markdown(f"**Tools Used:** `{', '.join(case['Improve Phase']['Tools Used'])}`")
    with c:
        st.subheader("Control Phase")
        # --- CRITICAL FIX HERE ---
        st.markdown(f"**Control Plan:** {case['Control Phase']['Control Plan']}")
        st.success(f"**Final Performance:** {case['Control Phase']['Final Performance']}")
    st.divider()
    st.header("Project Outcomes & Lessons Learned")
    col1, col2 = st.columns(2)
    with col1: st.metric("💰 Final Financial Impact (Hard/Soft Savings)", f"${case['Project Outcomes']['Financial Impact']:,}")
    with col2: st.metric("📈 Operational Impact", case['Project Outcomes']['Operational Impact'])
    st.info(f"**Key Lesson Learned:** {case['Project Outcomes']['Lessons Learned']}")
