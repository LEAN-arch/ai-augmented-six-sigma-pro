"""
app_pages.py

Contains the rendering logic for each main page of the Bio-AI Excellence
Framework. This definitive, content-rich version integrates expert-level SME
explanations directly alongside every plot, figure, and table, and features
a substantially improved narrative across all pages.

Author: Bio-AI Excellence SME Collective
Version: 33.1 (Elite Visualization Build)
Date: 2025-07-16

Changelog from v33.0:
- [CRITICAL-FIX] Corrected a SyntaxError on the welcome page.
- [LAYOUT] Updated the 'Improve' phase layout to accommodate the new 3D
  Response Surface plot for DOE analysis.
- [INTEGRATION] Confirmed seamless integration with all new and upgraded
  elite-tier visualizations from `helpers/visualizations.py`, including the
  new QFD, SHAP, Gage R&R, and NLP plots.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Callable, Any

# ==============================================================================
# 1. IMPORTS FROM HELPER MODULES
# ==============================================================================
from helpers.styling import COLORS
from helpers.content import get_guidance_data, render_workflow_step
from helpers.data_generators import (
    generate_nonlinear_data, generate_doe_data, generate_rsm_data,
    generate_dfmea_data, generate_qfd_data, generate_kano_data,
    generate_pareto_data, generate_risk_signal_data,
    generate_adverse_event_data, generate_pccp_data,
    generate_control_chart_data, generate_hotelling_data,
    generate_process_data, generate_capa_data
)
from helpers.ml_models import (
    train_regression_models, get_shap_explanation,
    perform_risk_signal_clustering, perform_text_clustering,
    perform_topic_modeling_on_capa
)
from helpers.visualizations import * # Import all upgraded plotting functions

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
    """
    A standardized function to render a visual tool (plot or table)
    with its detailed, expert-level explanation in an expander.
    """
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
    """
    Renders the main landing page of the application with substantially
    improved context and a compelling narrative.
    """
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

    In each section, you will find classical tools presented alongside their AI-augmented counterparts, complete with detailed explanations of their purpose, methodology, and interpretation.
    """)
    st.success("Begin your journey by selecting a phase from the navigation panel on the left.")

# ==============================================================================
# PAGE 1: DEFINE PHASE
# ==============================================================================
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
            _render_analysis_tool(
                title="Classical Tool: Project Charter",
                tool_function=plot_project_charter_visual,
                tool_args={},
                explanation_text="""
                - **What is it?** A formal, high-level document that acts as the project's "constitution." It captures the problem statement, project goals, key deliverables, and metrics for success (KPIs) in a concise, dashboard-style format.
                - **Why do it?** To secure stakeholder alignment and provide a formal mandate for the project. It serves as a constant reference point to prevent "scope creep" and ensures the entire team is working towards the same, clearly defined objectives.
                - **How is it done?** A cross-functional team articulates the business case (Problem Statement) and the desired outcome (Goal Statement). They then define specific, measurable, achievable, relevant, and time-bound (SMART) KPIs that will objectively define project success.
                - **Purpose & Meaning of Results:** The charter is a foundational strategic document. Its "result" is clarity and alignment. The KPIs shown (e.g., "LOD < 0.1% VAF") become the high-level specifications that will be translated into detailed technical requirements.
                """
            )
        with col2:
            sipoc_df = pd.DataFrame({'Suppliers': ['• Reagent Vendors<br>• Instrument Mfr.<br>• LIMS Provider'], 'Inputs': ['• Patient Sample<br>• Reagent Kits<br>• SOP'], 'Process': ['1. Sample Prep<br>2. Library Prep<br>3. Sequencing<br>4. Bioinformatics<br>5. Reporting'], 'Outputs': ['• VCF File<br>• QC Report<br>• Clinical Report'], 'Customers': ['• Oncologists<br>• Patients<br>• Pharma Partners']})
            _render_analysis_tool(
                title="Classical Tool: SIPOC Diagram",
                tool_function=generate_html_table,
                tool_args={'df': sipoc_df, 'title': "SIPOC: High-Level NGS Assay Workflow"},
                is_html=True,
                explanation_text="""
                - **What is it?** A high-level process map that summarizes the inputs and outputs of a process. SIPOC stands for **S**uppliers, **I**nputs, **P**rocess, **O**utputs, and **C**ustomers.
                - **Why do it?** To establish the boundaries and scope of the process being analyzed. Before delving into details, the SIPOC provides a "50,000-foot view," ensuring no critical stakeholders or process steps are overlooked.
                - **How is it done?** The team brainstorms and populates each of the five columns, typically working from the central "Process" column outwards.
                - **Purpose & Meaning of Results:** The resulting table provides a definitive scope for the project. It clarifies who provides what is necessary for the process and who receives the results, which is critical for risk management and understanding the complete process ecosystem.
                """
            )

    with st.container(border=True):
        st.subheader("2. Requirements Translation & Design Input Prioritization")
        st.markdown("This is the core translation step: converting the qualitative 'Voice of the Customer' (VOC) into the quantitative 'Voice of the Engineer.' This process ensures that what is built directly addresses validated user needs.")
        tab1, tab2 = st.tabs(["🏛️ Classical Tools", "🤖 ML Augmentation"])
        
        with tab1:
            _render_analysis_tool(
                title="Tool: Critical to Quality (CTQ) Tree",
                tool_function=plot_ctq_tree_plotly,
                tool_args={},
                explanation_text="""
                - **What is it?** A hierarchical diagram that deconstructs broad customer needs into specific, measurable, and actionable technical performance requirements.
                - **Why do it?** To rigorously translate the qualitative "Voice of the Customer" (e.g., "I need a fast test") into the quantitative "Voice of the Engineer" (e.g., "Sample-to-report time must be < 5 days"). This translation is fundamental to good product design.
                - **How is it done?** The process starts with a primary customer **Need**. This is broken down into rational **Drivers** of quality. Each driver is then further broken down into one or more specific, quantifiable **Critical-to-Quality** characteristics (CTQs).
                - **Purpose & Meaning of Results:** The CTQ tree produces the technical backbone of the project's Design Inputs (per 21 CFR 820.30). These CTQs (e.g., LOD, Specificity) are the very characteristics that will be formally verified and validated in later phases.
                """
            )
            st.divider()
            weights, rel_df = generate_qfd_data()
            _render_analysis_tool(
                title="Tool: Quality Function Deployment (QFD)",
                tool_function=plot_qfd_house_of_quality_pro,
                tool_args={'weights': weights, 'rel_df': rel_df},
                explanation_text="""
                - **What is it?** A structured methodology for translating customer requirements into product specifications. The "House of Quality" is its primary tool—a matrix that mathematically links customer needs to technical design features. This elite version integrates all components into a single, actionable dashboard.

                - **Why do it?** To prioritize development efforts on the technical features that will have the greatest impact on customer satisfaction, preventing misallocation of resources. It provides a rigorous, data-driven, and auditable trail from customer needs to final design specifications.

                - **How to Read This Plot:**
                  1.  **Left (Blue Bars):** Start with the **Customer Needs** and their importance ratings. Higher bars mean more important needs.
                  2.  **Center (Heatmap):** The central matrix shows the **Relationship Strength** between customer needs and the technical characteristics your team can control. Darker blue indicates a stronger link. Hover over any cell for details.
                  3.  **Roof (Triangle Heatmap):** The 'roof' shows **Correlations** between technical characteristics. Green ('✚') indicates a synergistic relationship (improving one helps the other), while red ('—') indicates a trade-off (improving one hurts the other).
                  4.  **Bottom (Green Bars):** This is the **actionable output**. It shows the calculated **Technical Priority** scores.

                - **Purpose & Meaning of Results:** The bottom bar chart provides the final, data-driven answer. The tallest bars represent the most critical technical characteristics to focus on to maximize customer satisfaction. In this example, 'LOD (VAF %)' and 'Reagent Cost ($)' are the highest priorities and should receive the most engineering attention and resources.
                """
            )
            st.divider()
            _render_analysis_tool(
                title="Tool: Kano Model",
                tool_function=plot_kano_visual,
                tool_args={'df_kano': generate_kano_data()},
                explanation_text="""
                - **What is it?** A model of customer satisfaction that classifies product features into three categories: Basic (Must-be), Performance, and Excitement (Delighters).
                - **Why do it?** To achieve a more sophisticated understanding of a feature's value than a simple "high/low importance" scale. It helps a team decide not only what to build, but also *how* to prioritize features to create a competitive advantage.
                - **How is it done?** Features are categorized via a survey that asks users how they feel if a feature is present (functional) and absent (dysfunctional).
                - **Purpose & Meaning of Results:** The visualization guides strategic prioritization. All **Basic** needs must be met. The company must be competitive on **Performance** attributes. Investing in a few key **Excitement** features can create a "wow" factor that differentiates the product.
                """
            )
        
        with tab2:
            df_reg = generate_nonlinear_data(); model_results = train_regression_models(df_reg, TARGET_COLUMN_NONLINEAR)
            shap_explanation = get_shap_explanation(model_results['rf_model'], model_results['X'])
            _render_analysis_tool(
                title="Tool: Data-Driven Feature Importance (XAI)",
                tool_function=plot_shap_summary,
                tool_args={'shap_explanation': shap_explanation},
                explanation_text="""
                - **What is it?** The application of Explainable AI (XAI) techniques, like SHAP (SHapley Additive exPlanations), to an early predictive model to understand which input parameters have the most significant impact on a key outcome.
                - **Why do it?** To use early experimental or historical data to support or challenge the team's assumptions about which process parameters are truly critical. It can uncover unexpected drivers of performance.
                - **How is it done?** A machine learning model (e.g., Random Forest) is trained on the data. The SHAP algorithm then computes the contribution of each feature to each individual prediction, 'explaining' the model's logic.
                - **Purpose & Meaning of Results:** This sophisticated plot shows not just the rank of important features, but the distribution of their impact. The position of each box shows its typical impact, while the width shows the range of impacts. This provides a data-driven, prioritized list of factors that should be considered critical design inputs, complementing the qualitative insights from tools like QFD.
                """
            )

    with st.container(border=True):
        st.subheader("3. Early Risk Assessment")
        st.markdown("A proactive, systematic evaluation of potential design failures. The goal is to identify and mitigate risks when changes are least expensive—on paper, before the design is finalized.")
        tab1, tab2 = st.tabs(["🏛️ Classical: DFMEA", "🤖 ML Augmentation"])
        
        with tab1:
            _render_analysis_tool(
                title="Tool: Design FMEA (DFMEA)",
                tool_function=generate_html_table,
                tool_args={'df': generate_dfmea_data(), 'title': "Design Failure Mode and Effects Analysis (DFMEA)"},
                is_html=True,
                explanation_text="""
                - **What is it?** A Design Failure Mode and Effects Analysis is a systematic activity to proactively identify, evaluate, and mitigate potential failures in a product design.
                - **Why do it?** To improve product reliability and safety by addressing potential problems during the design phase. It is a cornerstone of risk management required by **ISO 14971** and FDA regulations.
                - **How is it done?** A team rates potential failures on **S**everity (S), **O**ccurrence (O), and **D**etection (D) on a 1-10 scale, then calculates the **Risk Priority Number (RPN) = S × O × D**.
                - **Purpose & Meaning of Results:** The FMEA table, sorted by RPN, creates a prioritized list of risks. High-RPN items demand immediate mitigation. The color-coded RPN column provides an instant visual cue for the highest-risk items.
                """
            )
        with tab2:
            _render_analysis_tool(
                title="Tool: Unsupervised Clustering for Risk Signal Grouping",
                tool_function=plot_risk_signal_clusters,
                tool_args={'df_clustered': perform_risk_signal_clustering(generate_risk_signal_data())},
                explanation_text="""
                - **What is it?** The application of unsupervised machine learning (specifically, a clustering algorithm like DBSCAN) to early process or manufacturing data.
                - **Why do it?** To augment the human-driven FMEA by discovering data-driven, unexpected patterns or anomalies that may represent an unknown source of risk. It can identify failure modes the team has not yet conceived of.
                - **How is it done?** Multi-parameter data (e.g., temperature and pressure) is plotted. DBSCAN is used because it does not require a pre-specified number of clusters and can identify noise points as "outliers."
                - **Purpose & Meaning of Results:** The resulting scatter plot shows the natural groupings in the data. Clusters might represent different stable operating states. The most critical results are the **outliers**, highlighted in red. Each outlier is an anomalous event that warrants immediate investigation.
                """
            )

# ==============================================================================
# PAGE 2: MEASURE PHASE
# ==============================================================================
def show_measure_phase() -> None:
    st.title("🔬 Measure: System Validation & Baseline Performance")
    st.markdown("""
    **Phase Objective:** To ensure that the systems used to measure the product and process are themselves accurate and reliable, and then to use those validated systems to collect data and establish a robust, statistical baseline of the process's current performance.

    **Regulatory Context:** This phase is critical for **Analytical Method Validation** and serves as a prerequisite for **Process Validation (PV) Stage 1 (Process Design)** and **Stage 2 (Process Performance Qualification)**. Regulatory bodies require objective evidence that measurement systems are suitable for their intended use before any process data can be considered valid.
    """)
    st.divider()
    if 'measure_lsl' not in st.session_state: st.session_state.update({"measure_lsl": 0.8, "measure_usl": 9.0, "measure_mean": 4.0, "measure_std": 0.5})
    with st.sidebar: st.header("🔬 Simulators"); st.divider(); st.subheader("Assay Capability"); st.slider("Lower Spec Limit (LSL)", 0.5, 2.0, key="measure_lsl"); st.slider("Upper Spec Limit (USL)", 8.0, 10.0, key="measure_usl"); st.slider("Assay Mean (μ)", 2.0, 8.0, key="measure_mean"); st.slider("Assay Std Dev (σ)", 0.2, 2.0, key="measure_std")
    
    with st.container(border=True):
        st.subheader("1. Prerequisite: Measurement System Analysis (MSA)")
        st.warning("""
        **You cannot trust your process data until you trust your measurement system.** This is the foundational principle of the Measure phase. MSA quantifies the 'noise' from your measurement system to ensure that the variation you later observe is from the process or biology, not the lab tools.
        """)
        df_gage = pd.DataFrame({'Source of Variation': ['Assay Variation (Biology)', 'Repeatability (Sequencer)', 'Reproducibility (Operator)'], 'Contribution (%)': [92, 5, 3]})
        _render_analysis_tool(
            title="Tool: Gage R&R Study",
            tool_function=plot_gage_rr_pareto,
            tool_args={'df_gage': df_gage},
            explanation_text="""
            - **What is it?** A statistical study designed to quantify the amount of variation in your data that comes from the measurement system itself.
            - **Why do it?** This is a non-negotiable prerequisite for any data-based decision-making. If your measurement system is noisy, you cannot trust your process data.
            - **How is it done?** A structured experiment is performed where multiple operators measure multiple samples multiple times. The resulting data is analyzed using ANOVA to partition the total observed variance into its components: the true Part-to-Part variation and the measurement error (Gage R&R).
            - **Purpose & Meaning of Results:** This dashboard visualizes the variance components. The stacked bar on the left shows the total variation, with the red 'Gage R&R' portion representing measurement error. The donut chart on the right breaks this error down into its sources. The key guideline is that the total Gage R&R should contribute **less than 10%** of the total variation for the system to be deemed 'Acceptable'.
            """
        )

    with st.container(border=True):
        st.subheader("2. Establishing Baseline Assay Capability")
        st.markdown("Once the measurement system is validated, the next step is to use it to characterize the current process. This involves assessing the process's ability to meet the Critical-to-Quality (CTQ) specifications defined in the previous phase.")
        data = generate_process_data(st.session_state.measure_mean, st.session_state.measure_std, 2000)
        fig_cap, cp, cpk = plot_capability_analysis_pro(data, st.session_state.measure_lsl, st.session_state.measure_usl)
        
        st.plotly_chart(fig_cap, use_container_width=True)
        with st.expander("Methodology, Purpose, and Interpretation"):
            st.markdown("""
            - **What is it?** A comprehensive statistical report that quantifies how well a process can produce output within its specification limits (the CTQs defined earlier).
            - **Why do it?** To move beyond a simple "pass/fail" on individual samples and get a holistic, predictive assessment of the process's long-term performance. It answers: "Is our process robust enough to *consistently* meet customer requirements?"
            - **How is it done?** Data is plotted as a histogram against the Lower (LSL) and Upper (USL) Specification Limits. A normal distribution curve is fitted to model the process, and a Q-Q plot tests for normality. Key indices are calculated:
                - **Cp (Potential):** Measures the spread of the process relative to the spread of the specifications. It ignores centering.
                - **Cpk (Actual):** Measures the spread *and* centering of the process relative to the nearest specification limit. This is the more important, real-world metric.
            - **Purpose & Meaning of Results:** The plot provides a complete diagnostic. The **Cpk value** is the ultimate report card: a value **> 1.33** is considered capable. The visualization instantly shows *why* a process might be failing—if Cp is high but Cpk is low, the process is precise but off-center. If both are low, the process has too much variation. This is a key output for FDA Process Validation.
            """)

# ==============================================================================
# PAGE 3: ANALYZE PHASE
# ==============================================================================
def show_analyze_phase() -> None:
    st.title("📈 Analyze: Root Cause & Failure Modes")
    st.markdown("""
    **Phase Objective:** To leverage the data collected in the Measure phase to move from identifying that a problem exists (e.g., low Cpk) to pinpointing and verifying its root cause(s). This is the "detective work" phase.

    **Regulatory Context:** This phase is the heart of any **Corrective and Preventive Action (CAPA)** investigation under **21 CFR 820.100**. The tools used here provide the objective evidence required to justify the "Investigation" and "Analysis" sections of a CAPA record. It also directly informs **Quality Risk Management (ISO 14971, ICH Q9)** by identifying and quantifying the drivers of failure.
    """)
    st.divider()
    with st.container(border=True):
        st.subheader("1. Qualitative Root Cause Analysis & Prioritization")
        st.markdown("Begin the analysis by structuring the problem and focusing on the most significant issues. These tools help organize team brainstorming and apply the 80/20 rule to focus efforts.")
        col1, col2 = st.columns(2)
        with col1:
            _render_analysis_tool(
                title="Classical Tool: Fishbone Diagram",
                tool_function=plot_fishbone_plotly,
                tool_args={},
                explanation_text="""
                - **What is it?** A structured brainstorming tool used to systematically explore and categorize the potential causes of a specific problem or effect.
                - **Why do it?** To organize a team's thinking during a root cause analysis session, ensuring all potential causal areas are considered (e.g., Man, Method, Machine, Material, Measurement, Environment).
                - **How is it done?** The "effect" or problem (e.g., "Low Library Yield") is the "head" of the fish. The main "bones" are the causal categories. The team then brainstorms specific causes within each category.
                - **Purpose & Meaning of Results:** The diagram is a comprehensive map of all *potential* causes. It does not prove causality but serves as the input for gathering data to test which of these potential causes are the true root causes.
                """
            )
        with col2:
            _render_analysis_tool(
                title="Classical Tool: Pareto Chart",
                tool_function=plot_pareto_chart,
                tool_args={'df_pareto': generate_pareto_data()},
                explanation_text="""
                - **What is it?** A combination bar and line chart that displays failure modes in descending order of frequency, and the cumulative percentage of total failures.
                - **Why do it?** To apply the **Pareto Principle (80/20 rule)** to problem-solving. It helps a team focus its limited resources on the "vital few" causes that are responsible for the majority of the problems.
                - **How is it done?** Failure data is tallied by category. The categories are plotted as bars in descending order of frequency. A line graph is overlaid showing the cumulative percentage.
                - **Purpose & Meaning of Results:** The chart instantly reveals the biggest contributors to a problem. The team should focus its efforts on the first few bars that account for roughly 80% of the total failures.
                """
            )
    with st.container(border=True):
        st.subheader("2. Finding the Drivers: Modeling Assay Performance")
        st.markdown("Move from qualitative brainstorming to quantitative analysis. This step uses statistical and machine learning models to identify which input variables have a significant, measurable impact on the process output.")
        df_reg = generate_nonlinear_data(); model_results = train_regression_models(df_reg, TARGET_COLUMN_NONLINEAR)
        col3, col4 = st.columns(2)
        with col3:
             _render_analysis_tool(
                title="Classical vs. ML Model Fit",
                tool_function=plot_regression_comparison,
                tool_args={'model_results': model_results},
                explanation_text="""
                - **What is it?** A plot that overlays the predictions from a simple statistical model (Linear Regression) and a complex machine learning model (Random Forest) on top of the actual data.
                - **Why do it?** To quickly diagnose the nature of the relationships in the process. If they are complex and non-linear, a more powerful ML model is required to understand them.
                - **How is it done?** Both models are trained to predict an outcome from process parameters. The plot shows the actual data points and the fitted lines for both models. R-squared quantifies the goodness-of-fit.
                - **Purpose & Meaning of Results:** If the Random Forest line fits the data much more closely than the Linear Regression line, it is strong evidence that the process is governed by non-linearities or interactions, justifying the use of XAI tools like SHAP for root cause analysis.
                """
            )
        with col4:
            shap_explanation = get_shap_explanation(model_results['rf_model'], model_results['X'])
            _render_analysis_tool(
                title="ML Augmentation: XAI to Find Root Cause",
                tool_function=plot_shap_summary,
                tool_args={'shap_explanation': shap_explanation},
                explanation_text="""
                - **What is it?** A visualization from Explainable AI (XAI) that summarizes the output of a SHAP analysis. It shows which features are most important to a machine learning model and what their impact is.
                - **Why do it?** To open the "black box" of a complex model. After establishing that the ML model is accurate, SHAP is used to understand *why* it is accurate, turning the model into a powerful root cause analysis tool.
                - **How is it done?** This advanced 'Lollipop & Box Plot' shows the distribution of SHAP values for each feature. The lollipop shows the mean absolute importance, while the box shows the range and direction of impacts.
                - **Purpose & Meaning of Results:** The plot ranks features by importance (top is most important). The box plot for each feature reveals if it typically has a positive (right of zero) or negative (left of zero) impact on the outcome, providing a rich, data-driven hypothesis for root cause.
                """
            )
    with st.container(border=True):
        st.subheader("3. Process Failure Analysis (CAPA & Deviations)")
        st.markdown("Apply structured problem-solving techniques to specific failure events, and use NLP to find systemic trends in historical failure data.")
        tab3, tab4 = st.tabs(["🏛️ Classical: FTA & 5 Whys", "🤖 ML Augmentation: NLP on Logs"])
        with tab3:
            _render_analysis_tool(
                title="Tool: Fault Tree Analysis (FTA)",
                tool_function=plot_fault_tree_plotly,
                tool_args={},
                explanation_text="""
                - **What is it?** A top-down, deductive risk analysis technique that starts with a single, undesirable top-level event and traces it down to all the potential lower-level events that could cause it.
                - **Why do it?** To provide a logical and quantitative framework for understanding the pathways to a critical system failure.
                - **How is it done?** The tree is constructed using logic gates (OR, AND). If probabilities are known for the basic events, they can be propagated up the tree to calculate the probability of the top event.
                - **Purpose & Meaning of Results:** The diagram provides a clear visual map of all failure pathways. It helps identify single points of failure and areas where redundancy is needed. It is a formal, auditable artifact for a CAPA or risk file.
                """
            )
            st.divider()
            _render_analysis_tool(
                title="Tool: 5 Whys Analysis",
                tool_function=plot_5whys_diagram,
                tool_args={},
                explanation_text="""
                - **What is it?** An iterative, interrogative technique used to explore the cause-and-effect relationships underlying a problem by repeatedly asking "Why?".
                - **Why do it?** To prevent teams from implementing "band-aid" solutions that only address immediate symptoms, instead forcing a deeper analysis to uncover the systemic issue.
                - **How is it done?** The team starts with a problem statement and asks "Why did this happen?". They take the answer and ask "Why did *that* happen?" This is repeated approximately five times.
                - **Purpose & Meaning of Results:** The diagram shows the logical chain from the surface-level problem to the actionable root cause. The final answer is often related to a process or system, leading to a much more effective corrective action.
                """
            )
        with tab4:
            _render_analysis_tool(
                title="Tool: NLP and Clustering on CAPA/Deviation Logs",
                tool_function=plot_nlp_on_capa_logs,
                tool_args={'df_topics': perform_topic_modeling_on_capa(generate_capa_data(), 'Description')},
                explanation_text="""
                - **What is it?** The application of Natural Language Processing (NLP), specifically topic modeling, to a large body of unstructured text from CAPA or deviation logs.
                - **Why do it?** To overcome the limitations of manual review by analyzing thousands of reports to identify systemic, recurring themes and trends that would otherwise be invisible.
                - **How is it done?** The text from all logs is categorized into topics. The frequency of each topic is then visualized using a Treemap, where the area of each rectangle is proportional to its frequency.
                - **Purpose & Meaning of Results:** The Treemap makes it instantly clear which failure themes are most prevalent. This allows management to focus quality improvement efforts on the biggest systemic problems (e.g., the largest rectangles) rather than just reacting to individual incidents.
                """
            )

# ==============================================================================
# PAGE 4: IMPROVE PHASE
# ==============================================================================
def show_improve_phase() -> None:
    st.title("⚙️ Improve: Optimization & Robustness")
    st.markdown("""
    **Phase Objective:** To use the knowledge gained from the Analyze phase to develop, test, and implement solutions that address validated root causes. The goal is not just to fix the problem, but to find the optimal settings for critical parameters to create a high-performing and robust process.

    **Regulatory Context:** This is the practical application of Quality by Design (QbD). The outputs of this phase directly define the **Design Space** (ICH Q8) and the proven acceptable ranges for critical process parameters. This work provides the core data for **Process Validation Stage 1 (Process Design)**, demonstrating that the proposed process is capable of consistently delivering a quality product.
    """)
    st.divider()
    true_func_bo = lambda x: (np.sin(x * 0.8) * 15) + (np.cos(x * 2.5) * 5) - (x / 10)**3; x_range_bo = np.linspace(0, 20, 400)
    if 'bo_sampled_points' not in st.session_state: initial_x = [2.0, 18.0]; st.session_state.bo_sampled_points = {'x': initial_x, 'y': [true_func_bo(x) for x in initial_x]}
    with st.sidebar:
        st.header("🔬 Simulators"); st.divider(); st.subheader("Bayesian Optimization")
        if st.button("Run Next Smart Experiment", key='bo_sample'): _, next_point = plot_bayesian_optimization_interactive(true_func_bo, x_range_bo, st.session_state.bo_sampled_points); st.session_state.bo_sampled_points['x'].append(next_point); st.session_state.bo_sampled_points['y'].append(true_func_bo(next_point)); st.rerun()
        if st.button("Reset Simulation", key='bo_reset'): initial_x = [2.0, 18.0]; st.session_state.bo_sampled_points = {'x': initial_x, 'y': [true_func_bo(x) for x in initial_x]}; st.rerun()
    with st.container(border=True):
        st.subheader("1. Design Space & Process Optimization")
        st.markdown("This section demonstrates powerful methods for efficiently exploring the relationships between process inputs and outputs to find the settings that maximize performance.")
        tab1, tab2 = st.tabs(["🧪 Design of Experiments (DOE) & RSM", "🤖 Bayesian Optimization"])
        with tab1:
            st.markdown("##### **Classical: Design of Experiments (DOE)**")
            doe_data = generate_doe_data()
            fig_doe_main, fig_doe_interaction = plot_doe_effects(doe_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("###### DOE Cube Plot (Design Points)")
                st.plotly_chart(plot_doe_cube(doe_data), use_container_width=True)
            with col2:
                st.markdown("###### 3D Response Surface (Predicted Model)")
                st.plotly_chart(plot_doe_3d_surface(doe_data), use_container_width=True)

            st.markdown("###### Factor Effects")
            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(fig_doe_main, use_container_width=True)
            with col4:
                st.plotly_chart(fig_doe_interaction, use_container_width=True)

            with st.expander("Methodology, Purpose, and Interpretation"):
                st.markdown("""
                - **What is it?** A structured and statistically powerful approach to experimentation where multiple factors are varied simultaneously to understand their individual and interactive effects on a process output.
                - **Why do it?** It is vastly more efficient and informative than the traditional "One-Factor-at-a-Time" (OFAT) approach. DOE can reveal interactions between factors which OFAT can never detect.
                - **How is it done?** A structured experimental plan is created. The experiments are run, and a statistical model calculates the **Effect** of each factor and interaction.
                - **Purpose & Meaning of Results:**
                  - **Effects Plots:** Rank factors by their effect magnitude, showing the most powerful levers for control.
                  - **Cube Plot:** Visualizes the response at each corner of the design space.
                  - **3D Response Surface:** Visualizes the predicted process model over the entire design space, providing an intuitive topographical map of performance.
                """)
            st.divider()
            _render_analysis_tool(
                title="Classical: Response Surface Methodology (RSM)",
                tool_function=plot_rsm_contour,
                tool_args={'df_rsm': generate_rsm_data()},
                explanation_text="""
                - **What is it?** A specialized set of DOE techniques used to find the optimal settings for a process and to characterize the "design space" around that optimum.
                - **Why do it?** After DOE has identified the "vital few" significant factors, RSM is used to fine-tune those factors to achieve a peak or valley in the response (e.g., to maximize yield).
                - **How is it done?** RSM uses experimental designs that allow the fitting of a more complex quadratic (curved) statistical model. The results are visualized as a contour plot.
                - **Purpose & Meaning of Results:** The **Contour Plot** acts like a topographical map of the process response. The colors represent the output value, and the "star" marks the predicted optimal settings. This plot is the direct visualization of the **Design Space** as required by ICH Q8.
                """
            )
        with tab2:
            fig_bo, _ = plot_bayesian_optimization_interactive(true_func_bo, x_range_bo, st.session_state.bo_sampled_points)
            _render_analysis_tool(
                title="ML Augmentation: Bayesian Optimization",
                tool_function=lambda **x: fig_bo, # Pass the pre-computed figure
                tool_args={},
                explanation_text="""
                - **What is it?** A modern, machine-learning-driven approach to optimization. It is a "smart" sequential search algorithm that uses the results of past experiments to decide the most informative next experiment to run.
                - **Why do it?** It is extremely efficient for optimizing processes where experiments are very expensive, time-consuming, or have many parameters, as it aims to find the optimum in the minimum number of runs.
                - **How is it done?** It fits a probabilistic surrogate model (a Gaussian Process) to the data. It then uses an "acquisition function" that balances **exploitation** (sampling near the current best point) and **exploration** (sampling in areas of high uncertainty) to choose the next experiment.
                - **Purpose & Meaning of Results:** The plot visualizes this intelligent search. It shows the model's current belief (the GP mean and confidence interval), the points already sampled, and the acquisition function guiding the next choice. You can see the model progressively converging on the true optimum.
                """
            )

# ==============================================================================
# PAGE 5: CONTROL PHASE
# ==============================================================================
def show_control_phase() -> None:
    st.title("📡 Control: Lab Operations & Post-Market Surveillance")
    st.markdown("""
    **Phase Objective:** To implement robust systems to "hold the gains" achieved in the Improve phase. This involves creating a monitoring strategy to ensure the optimized process remains stable and in control over time, and to actively monitor product performance in the real world.

    **Regulatory Context:** This phase is directly aligned with **FDA Process Validation Stage 3 (Continued Process Verification - CPV)** and **Post-Market Surveillance (PMS)** requirements under **21 CFR 820.200**. The tools in this section provide the framework for ongoing monitoring, data trending, and signal detection necessary for maintaining a validated state.
    """)
    st.divider()
    if 'ctrl_shift_mag' not in st.session_state: st.session_state.update({"ctrl_shift_mag": 0.8, "ewma_lambda": 0.2})
    with st.sidebar: st.header("🔬 Simulators"); st.divider(); st.subheader("QC Simulator"); st.slider("Magnitude of Shift (in Std Devs)", 0.2, 3.0, 0.8, 0.1, key="ctrl_shift_mag"); st.slider("EWMA Lambda (λ)", 0.1, 0.5, 0.2, 0.05, key="ewma_lambda", help="Higher λ reacts faster to shifts.")
    chart_data = generate_control_chart_data(shift_magnitude=st.session_state.ctrl_shift_mag)
    
    with st.container(border=True):
        st.subheader("1. Monitoring for Stability: Statistical Process Control (SPC)")
        st.markdown("SPC is the primary toolset for monitoring a process in real-time to ensure it remains stable and predictable. The key is to distinguish between normal process variation ('common cause') and a true signal that something has changed ('special cause').")
        spc_explanation = """
        - **What are they?** A family of time-series graphs used to monitor a process to determine if it is in a state of "statistical control" (i.e., stable and predictable).
        - **Why do it?** To provide a real-time, objective method for process monitoring. They help distinguish between **common cause variation** (natural noise) and **special cause variation** (a real problem). Reacting to common cause variation makes a process worse; reacting to special cause variation is essential.
        - **How are they done?** A key process metric is plotted over time. Control limits (typically ±3σ) are added. Different charts are sensitive to different types of problems.
        - **Purpose & Meaning of Results:** A process is "in control" if points are randomly distributed within the limits. Any point falling outside the limits or a non-random pattern (highlighted as a violation) is a signal of a special cause that mandates investigation. This is the foundation of Continued Process Verification (CPV).
        """
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Levey-Jennings", "📈 EWMA", "📉 CUSUM", "🤖 Multivariate QC"])
        with tab1:
            _render_analysis_tool(
                title="Classical: Levey-Jennings Chart (Shewhart)",
                tool_function=plot_shewhart_chart,
                tool_args={'df_control': chart_data},
                explanation_text=spc_explanation + "\n\n- **Shewhart Specifics:** This is the most common SPC chart. It is excellent at detecting large, sudden shifts in the process."
            )
        with tab2:
            _render_analysis_tool(
                title="Advanced: EWMA Chart",
                tool_function=plot_ewma_chart,
                tool_args={'df_control': chart_data, 'lambda_val': st.session_state.ewma_lambda},
                explanation_text=spc_explanation + "\n\n- **EWMA Specifics:** The Exponentially Weighted Moving Average chart gives more weight to recent data. It is more sensitive than a Shewhart chart for detecting small, sustained shifts."
            )
        with tab3:
            _render_analysis_tool(
                title="Advanced: CUSUM Chart",
                tool_function=plot_cusum_chart,
                tool_args={'df_control': chart_data},
                explanation_text=spc_explanation + "\n\n- **CUSUM Specifics:** The Cumulative Sum chart plots the cumulative sum of deviations from the target. It is the most sensitive chart for detecting very small, persistent shifts in the process mean."
            )
        with tab4:
            _render_analysis_tool(
                title="ML Augmentation: Multivariate QC (Hotelling's T²)",
                tool_function=plot_hotelling_t2_chart,
                tool_args={'df_hotelling': generate_hotelling_data()},
                explanation_text="""
                - **What is it?** A multivariate extension of the Shewhart chart. Instead of monitoring one variable, it monitors multiple correlated variables simultaneously by collapsing them into a single T² statistic.
                - **Why do it?** To detect out-of-control conditions that would not be apparent on any single univariate control chart (e.g., a small, correlated shift in two parameters).
                - **How is it done?** The T² statistic, which measures the statistical distance of each multi-variable data point from the process center, is plotted over time against a single upper control limit (UCL). This is linked to a scatter plot of the original variables.
                - **Purpose & Meaning of Results:** The **top chart** tells you *when* the process went out of control. The **bottom chart** tells you *how* it went out of control by showing the out-of-control point's position relative to the in-control data cloud.
                """
            )

    with st.container(border=True):
        st.subheader("2. Formalizing Gains & Post-Market Activities")
        st.markdown("This final step involves documenting the control strategy and extending monitoring activities into the post-market phase to ensure long-term product safety and effectiveness.")
        col1, col2 = st.columns(2)
        with col1:
            control_plan_df = pd.DataFrame({'Process Step': ['Library Prep', 'Sequencing', 'Bioinformatics'], 'Characteristic': ['Pos Control Yield (Y)', 'Sequencer Laser (X)', '% Mapped Reads (Y)'], 'Specification': ['20 ± 5 ng', '> 80 mW', '> 85%'], 'Tool': ['Fluorometer', 'Internal Sensor', 'FASTQC'], 'Method': ['L-J Chart', 'EWMA Chart', 'Shewhart Chart'], 'Frequency': ['Per Batch', 'Per Run', 'Per Sample'], 'Reaction Plan': ['Re-prep Batch', 'Schedule Maint.', 'Review Alignment']})
            _render_analysis_tool(
                title="Classical: The Control Plan",
                tool_function=generate_html_table,
                tool_args={'df': control_plan_df, 'title': "Assay Control Plan"},
                is_html=True,
                explanation_text="""
                - **What is it?** A formal table that summarizes the entire monitoring strategy for a process.
                - **Why do it?** To provide a clear, single-source-of-truth document for lab operations, outlining exactly how the process will be controlled to maintain its validated state.
                - **How is it done?** The table lists each critical process step, the characteristic being measured, its specification, the monitoring method, the frequency of measurement, and the "Reaction Plan" to be followed if a problem is detected.
                - **Purpose & Meaning of Results:** The Control Plan operationalizes the entire DMAIC project. It is a living document that serves as a direct instruction manual for technicians and a key auditable record for quality assurance.
                """
            )
        with col2:
            _render_analysis_tool(
                title="ML Augmentation: PMS Signal Detection",
                tool_function=plot_adverse_event_clusters,
                tool_args={'df_clustered': perform_text_clustering(generate_adverse_event_data(), 'description')},
                explanation_text="""
                - **What is it?** The application of unsupervised machine learning (NLP and clustering) to post-market surveillance (PMS) data, such as adverse event narratives or customer complaints.
                - **Why do it?** To proactively detect new or emerging safety signals from real-world product use by automating the search for patterns in vast amounts of unstructured text data.
                - **How is it done?** Text narratives are converted into numerical vectors. Dimensionality reduction (PCA) projects the data into 2D space. A clustering algorithm then groups similar reports together.
                - **Purpose & Meaning of Results:** Each cluster on the plot represents a distinct type of adverse event. The emergence of a new, well-defined cluster over time is a powerful signal of a previously unknown side effect or failure mode, which would trigger a formal safety investigation.
                """
            )
        st.divider()
        _render_analysis_tool(
            title="AI/ML Device Control: PCCP Monitoring",
            tool_function=plot_pccp_monitoring,
            tool_args={'df_pccp': generate_pccp_data()},
            explanation_text="""
            - **What is it?** A control chart specifically for monitoring the performance of a deployed AI/ML model, as part of a Pre-determined Change Control Plan (PCCP).
            - **Why do it?** AI/ML models can degrade over time due to "data drift." Regulatory bodies (e.g., FDA) require that manufacturers of AI/ML-based medical devices have a plan to monitor this degradation and manage model updates.
            - **How is it done?** A key model performance metric (e.g., AUC) is calculated periodically on new data and plotted over time against a pre-defined performance threshold.
            - **Purpose & Meaning of Results:** The chart provides objective evidence of the model's continued performance. If the metric drops below the threshold, the PCCP is triggered, which dictates the pre-defined plan for safely retraining, re-validating, and re-deploying the model.
            """
        )

# ==============================================================================
# PAGE 6 & 7: COMPARISON & MANIFESTO
# ==============================================================================
def show_comparison_matrix() -> None:
    st.title("Head-to-Head: Classical Statistics vs. Machine Learning")
    st.markdown("""
    This section provides a direct comparison of the two core disciplines that power the Bio-AI Excellence Framework. Understanding their relative strengths, weaknesses, and core philosophies is key to knowing which tool to use for a given R&D challenge.
    """)
    st.divider()
    with st.container(border=True):
        _render_analysis_tool(
            title="Strengths Profile: A Multi-Dimensional View",
            tool_function=plot_comparison_radar,
            tool_args={},
            explanation_text="""
            - **What is it?** A radar chart that provides a multi-dimensional comparison of the strengths of Classical Statistics and Machine Learning.
            - **Why do it?** To visually summarize the relative advantages of each methodology across a range of important attributes for biotech R&D.
            - **How is it done?** Key attributes (e.g., Interpretability, Data Volume Needs, Regulatory Ease) are defined as axes. Each methodology is scored on a 1-5 scale for each attribute, and the resulting shapes are plotted.
            - **Purpose & Meaning of Results:** The chart shows that each approach has a distinct 'shape' of capabilities. Classical Stats excels in regulatory ease and interpretability, while ML excels in handling complexity and scalability. This highlights their complementary nature.
            """
        )
    with st.container(border=True):
        _render_analysis_tool(
            title="The Verdict: Which Approach Excels for Which Task?",
            tool_function=plot_verdict_barchart,
            tool_args={},
            explanation_text="""
            - **What is it?** A diverging bar chart that gives a direct, opinionated recommendation of which approach is typically better suited for specific R&D tasks.
            - **Why do it?** To provide clear, actionable guidance for practitioners trying to decide which tool to use for a given problem.
            - **How is it done?** A list of common R&D tasks is created. Each task is assigned a 'winner' (Classical or ML), and a bar is plotted to the left or right of a central axis accordingly.
            - **Purpose & Meaning of Results:** This chart serves as a quick reference guide. It shows that for tasks requiring regulatory validation and causality (like Analytical Validation), classical stats are the winner. For tasks involving discovery and high-dimensionality (like Biomarker Discovery), ML is the winner.
            """
        )

def show_hybrid_manifesto() -> None:
    st.title("🤝 The Hybrid Manifesto & GxP Compliance")
    st.markdown("""
    The most competitive biotech organizations do not choose one methodology over the other; they build a **Bio-AI framework** that leverages the unique strengths of each to achieve superior outcomes while maintaining impeccable compliance. This is the core philosophy of this entire playbook.
    """)
    st.divider()
    with st.container(border=True):
        _render_analysis_tool(
            title="The Philosophy of Synergy: Inference + Prediction",
            tool_function=plot_synergy_diagram,
            tool_args={},
            explanation_text="""
            - **What is it?** A Venn diagram used as a metaphor to illustrate the core philosophy of the Bio-AI Excellence Framework.
            - **Why do it?** To communicate the central thesis of the entire application in a single, memorable visual: the greatest value lies at the intersection of the two disciplines.
            - **How is it done?** Two overlapping circles represent Classical Statistics and Machine Learning, with their core strengths listed. The overlapping area represents the "Bio-AI Excellence" zone.
            - **Purpose & Meaning of Results:** The diagram reinforces the message that the optimal strategy is not a choice *between* the two fields, but a synergistic **integration** of both. Use classical stats for rigor, validation, and causality. Use machine learning for scale, discovery, and complexity.
            """
        )
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
        st.markdown("Select a common R&D challenge to see a recommended approach based on the principles of the Bio-AI Framework.")
        guidance_data = get_guidance_data()
        scenarios = list(guidance_data.keys())
        selected_scenario = st.selectbox("Choose your R&D scenario:", scenarios, index=0)
        if selected_scenario and selected_scenario in guidance_data:
            recommendation = guidance_data[selected_scenario]
            st.success(f"##### Recommended Approach: {recommendation['approach']}")
            st.markdown(f"**Rationale:** {recommendation['rationale']}")
