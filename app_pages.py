# app_pages.py

import streamlit as st
import numpy as np
import pandas as pd

# Import all necessary helper functions from the single, definitive helper file.
from app_helpers import *

# ==============================================================================
# PAGE 0: WELCOME & FRAMEWORK OVERVIEW
# ==============================================================================
def show_welcome_page():
    st.title("Welcome to the Bio-AI Excellence Framework")
    st.markdown("##### An interactive, expert-level playbook for **compliant** and **accelerated** development of drugs, devices, and diagnostics.")
    st.markdown("---")

    st.subheader("The Core Philosophy: DMAIC as the Engine for Regulated Industries")
    st.markdown("""
    In the life sciences, process improvement is not just about efficiency; it's about **quality, safety, and compliance**. The **DMAIC (Define, Measure, Analyze, Improve, Control)** framework provides a robust, structured, and auditable methodology to drive this improvement. 
    
    This application demonstrates how to supercharge the traditional DMAIC cycle by integrating modern Machine Learning and AI tools alongside the classical, regulatory-accepted Six Sigma methods. Each phase of the DMAIC lifecycle is presented with tools relevant to key GxP milestones, from initial **Design & Development** to **Post-Market Surveillance**.
    """)
    
    st.info("""
    - **🏛️ Classical Tools:** The established, FDA-accepted methods that form the backbone of any QMS (e.g., DOE, FMEA, SPC).
    - **🤖 ML Augmentations:** The advanced data science techniques that provide unprecedented speed, predictive power, and insight (e.g., NLP, Predictive Modeling, Clustering).
    """)
    st.success("Use the sidebar to navigate through the DMAIC lifecycle. Each page is a deep dive into the tools and strategies that drive excellence in the modern biotech organization.")

# ==============================================================================
# PAGE 1: DEFINE PHASE
# ==============================================================================
def show_define_phase():
    st.title("🌀 Define: Project Mandate & Product Design")
    st.markdown("**Objective:** To clearly articulate the project goals, translate user needs into quantifiable design specifications, and establish the initial risk management framework.")
    st.markdown("> **Applicable Regulatory Stages:** FDA Design Controls (21 CFR 820.30), ICH Q8 (Pharmaceutical Development)")
    st.markdown("---")
    
    with st.container(border=True):
        st.subheader("1. Requirements Translation & Design Input Prioritization")
        st.markdown("Translating the **Voice of the Customer (VOC)** into the **Voice of the Engineer**.")
        
        tab1, tab2 = st.tabs(["🏛️ Classical: QFD & Kano Model", "🤖 ML Augmentation: Data-Driven Feature Importance"])
        with tab1:
            st.markdown("##### **Quality Function Deployment (QFD) - House of Quality**")
            st.plotly_chart(plot_qfd_house_of_quality(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance (QFD)"):
                st.markdown("""
                **Methodology:** QFD is a structured method for translating customer requirements into design specifications. The 'House of Quality' matrix maps customer needs (the 'Whats') to technical characteristics (the 'Hows'), creating a traceable link.
                
                **Regulatory Significance:** This creates a **documented, traceable link** between user needs and design inputs, a core requirement of FDA Design Controls. It provides objective evidence for *why* certain technical specifications were prioritized for the **Design History File (DHF)**.
                """)
            
            st.markdown("---")
            st.markdown("##### **Kano Model for Feature Prioritization**")
            st.plotly_chart(plot_kano_visual(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance (Kano)"):
                st.markdown("""
                **Methodology:** The Kano model categorizes features based on how they impact customer satisfaction, helping prioritize development.
                - **Basic (Must-be):** Taken for granted. Absence causes dissatisfaction.
                - **Performance:** More is better. Satisfaction is proportional to implementation.
                - **Excitement (Delighter):** Unexpected features that create high satisfaction.
                
                **Regulatory Significance:** The Kano model provides a strategic framework to justify which features are critical for the product's intended use and which are secondary. This helps focus validation and verification efforts on what matters most to the end-user (e.g., the clinician), a key principle of user-centric design praised by regulators.
                """)
        with tab2:
            st.markdown("##### **Dimensionality Reduction (PCA) & Feature Importance**")
            df_reg = generate_nonlinear_data()
            _, model, X_reg = train_and_plot_regression_models(df_reg)
            st.plotly_chart(plot_shap_summary(model, X_reg), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** After collecting pilot data, an ML model (e.g., Random Forest) can be trained to predict a key outcome. **Principal Component Analysis (PCA)** can be used first to reduce the dimensionality of complex data, while eXplainable AI (XAI) tools like **SHAP** then determine which input features have the most impact on the outcome.
                
                **Regulatory Significance:** While not a replacement for QFD, this provides powerful, data-driven evidence to **support or challenge** the assumptions made in the QFD. If SHAP reveals a parameter thought to be unimportant is actually critical, this is a vital finding for de-risking the project and must be documented in the DHF as evidence of a data-driven design process.
                """)

    with st.container(border=True):
        st.subheader("2. Early Risk Assessment")
        st.markdown("Proactively identifying potential failures and emergent risks *before* they are locked into the product design.")
        tab3, tab4 = st.tabs(["🏛️ Classical: DFMEA & Fishbone", "🤖 ML Augmentation: Unsupervised Risk Clustering"])
        with tab3:
            st.markdown("##### **Design Failure Mode & Effects Analysis (DFMEA)**")
            st.plotly_chart(plot_dfmea_table(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance (DFMEA)"):
                st.markdown("""
                **Methodology:** A systematic, bottom-up analysis of potential failure modes in a product's design. Each failure is scored for Severity (S), Occurrence (O), and Detection (D) to calculate a Risk Priority Number (RPN = S x O x D).
                
                **Regulatory Significance:** DFMEA is a cornerstone of risk management under **ISO 14971** and a required part of the Design History File. It demonstrates that risks have been considered and prioritized for mitigation. A Preliminary Hazard Analysis (PHA) often precedes this for a broader, system-level view.
                """)
        with tab4:
            st.markdown("##### **Unsupervised Clustering for Risk Signal Grouping**")
            st.plotly_chart(plot_risk_signal_clusters(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** Unsupervised ML algorithms (like DBSCAN) are applied to multi-parameter process data from early development runs. The algorithm identifies 'clusters' of data that behave similarly and flags anomalies, without any prior labels.
                
                **Regulatory Significance:** This can uncover **hidden failure modes or process interactions** not conceived of during the DFMEA. Discovering an underperforming cluster from 'Line A' would trigger an investigation and an update to the FMEA, demonstrating a data-driven and proactive approach to risk management that regulators value.
                """)

# ==============================================================================
# PAGE 2: MEASURE PHASE
# ==============================================================================
def show_measure_phase():
    st.title("🔬 Measure: Baseline Performance & Validation")
    st.markdown("**Objective:** To validate measurement systems, collect data, and establish a robust, data-driven baseline of the current process performance and capability.")
    st.markdown("> **Applicable Regulatory Stages:** FDA Process Validation (Stage 1), ICH Q8/Q11")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Measurement System Analysis (MSA)")
        st.markdown("You cannot trust your process data until you trust your measurement system. **Gage R&R** is the foundational tool to ensure observed variability comes from the process, not the measurement tool.")
        st.plotly_chart(plot_gage_rr_pareto(), use_container_width=True)
        with st.expander("Methodology & Regulatory Significance"):
            st.markdown("""
            **Methodology:** A Gage R&R (Repeatability & Reproducibility) study is a designed experiment to quantify the amount of variation in a measurement system. It partitions variance into its components: the part, the appraiser (operator), and the equipment.
            
            **Regulatory Significance:** Before any process characterization (ICH Q11) or validation (PPQ), the analytical methods used must themselves be validated. A Gage R&R study is the standard way to provide evidence that the measurement system is reliable and its variation is acceptable (typically <10% of total process tolerance). This is a prerequisite for all subsequent stages.
            """)

    with st.container(border=True):
        st.subheader("2. Establishing Baseline Process Capability")
        st.markdown("Comparing the 'Voice of the Process' to the 'Voice of the Customer' (specifications).")
        st.sidebar.header("🔬 Simulators"); st.sidebar.markdown("---"); st.sidebar.subheader("Capability Simulator")
        lsl = st.sidebar.slider("Lower Spec Limit (LSL)", 80.0, 88.0, 85.0, key="m_lsl")
        usl = st.sidebar.slider("Upper Spec Limit (USL)", 110.0, 120.0, 115.0, key="m_usl")
        process_mean = st.sidebar.slider("Process Mean (μ)", 95.0, 105.0, 100.0, key="m_mean")
        process_std = st.sidebar.slider("Process Std Dev (σ)", 1.0, 5.0, 2.0, key="m_std")

        data = generate_process_data(process_mean, process_std, 300)
        fig_cap, cp, cpk = plot_capability_analysis_pro(data, lsl, usl)
        st.plotly_chart(fig_cap, use_container_width=True)
        with st.expander("Methodology & Regulatory Significance"):
            st.markdown("""
            **Methodology:** Process Capability analysis uses metrics like **Cp and Cpk** to measure how well a process can meet its specification limits. Cpk is the key metric, as it accounts for both process variation and centering.
            
            **Regulatory Significance:** Establishing a baseline Cpk is a critical part of the 'Measure' phase. It quantifies the problem. Later, during **Process Validation (PPQ)**, a Cpk value ≥ 1.33 is the widely accepted industry standard to demonstrate that a process is robust and in a state of control.
            """)

    with st.container(border=True):
        st.subheader("3. Validating Supporting Models")
        st.markdown("Rigorously assessing the performance of any statistical or ML models used in the process.")
        st.plotly_chart(plot_model_validation_ci(), use_container_width=True)
        with st.expander("Methodology & Regulatory Significance"):
            st.markdown("""
            **Methodology:** Instead of reporting a single performance metric (e.g., Accuracy = 95%), a more rigorous approach is to calculate a **Confidence Interval (CI)**. Bootstrapping is a powerful method to do this by resampling the validation dataset thousands of times to create a distribution of the metric. **Cross-validation (CV)** is another key technique where the model is trained and tested on different subsets of the data to ensure it generalizes well.
            
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
        st.subheader("1. Qualitative Root Cause Brainstorming & Prioritization")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Fishbone (Ishikawa) Diagram**")
            st.plotly_chart(plot_fishbone_plotly(), use_container_width=True)
            with st.expander("Methodology"):
                st.markdown("A structured brainstorming tool used to explore all potential causes of a problem, grouped into categories (e.g., Man, Machine, Method).")
        with col2:
            st.markdown("##### **Pareto Chart**")
            st.plotly_chart(plot_pareto_chart(), use_container_width=True)
            with st.expander("Methodology"):
                st.markdown("The 80/20 rule in action. This chart identifies the 'vital few' causes that contribute to the majority of problems, allowing teams to focus their efforts.")

    with st.container(border=True):
        st.subheader("2. Statistical Root Cause Analysis")
        st.markdown("Using data to prove or disprove the hypotheses generated during brainstorming.")
        st.sidebar.header("🔬 Simulators"); st.sidebar.markdown("---"); st.sidebar.subheader("Group Comparison")
        mean1 = st.sidebar.slider("Lot A Mean", 18.0, 22.0, 19.5, 0.1, key='a1')
        mean2 = st.sidebar.slider("Lot B Mean", 18.0, 22.0, 20.0, 0.1, key='a2')
        mean3 = st.sidebar.slider("Lot C Mean", 18.0, 22.0, 20.5, 0.1, key='a3')
        anova_data = generate_anova_data(means=[mean1, mean2, mean3], stds=[0.8, 0.8, 0.8], n=20)
        
        tab1, tab2 = st.tabs(["🏛️ Classical: Hypothesis Testing", "🤖 ML Augmentation: Advanced Modeling"])
        with tab1:
            st.markdown("##### **ANOVA & t-Tests**")
            fig_anova, p_val = plot_anova_groups(anova_data)
            st.plotly_chart(fig_anova, use_container_width=True)
            if p_val < 0.05: st.error(f"ANOVA P-value is {p_val:.4f}. A statistically significant difference exists between lots.", icon="🚨")
            else: st.success(f"ANOVA P-value is {p_val:.4f}. No significant difference detected between lots.", icon="✅")
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** Hypothesis tests like **ANOVA** (for 3+ groups) or **t-tests** (for 2 groups) are used to determine if observed differences between group means are statistically significant or just due to random chance.
                
                **Regulatory Significance:** This is the standard, auditable method for providing statistical evidence during a **CAPA investigation** or process characterization study. A low p-value provides the justification for implementing a corrective action (e.g., disqualifying a raw material lot).
                """)
        with tab2:
            st.markdown("##### **Regression & SHAP for CPP Identification**")
            df_reg = generate_nonlinear_data()
            fig_reg, model, X_reg = train_and_plot_regression_models(df_reg)
            st.plotly_chart(fig_reg, use_container_width=True)
            st.plotly_chart(plot_shap_summary(model, X_reg), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** While standard **Regression analysis** is a classical tool, it often fails with complex, non-linear biological data. Fitting a more powerful ML model (like a Random Forest) and using **SHAP** provides a much more accurate view of which parameters are truly driving the outcome. Regularized regression like **Lasso/ElasticNet** can also be used for automated feature selection.
                
                **Regulatory Significance:** This provides a far more robust method for identifying **Critical Process Parameters (CPPs)** as required by ICH Q11. If the ML model is significantly more predictive than a linear model, its feature importance rankings provide stronger, more reliable evidence for classifying parameters as critical.
                """)

    with st.container(border=True):
        st.subheader("3. Process Failure Analysis")
        tab3, tab4 = st.tabs(["🏛️ Classical: FMEA & FTA", "🤖 ML Augmentation: NLP & Clustering on Logs"])
        with tab3:
            st.markdown("##### **Fault Tree Analysis (FTA)**")
            st.plotly_chart(plot_fault_tree_plotly(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("FTA is a top-down, deductive failure analysis that is invaluable for understanding complex failure pathways. It's a key tool in **Quality Risk Management (ICH Q9)**.")
        with tab4:
            st.markdown("##### **NLP and Clustering on CAPA/Deviation Logs**")
            st.plotly_chart(plot_nlp_on_capa_logs(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** Uses **NLP Topic Modeling** to automatically read and categorize thousands of unstructured text entries from deviation logs. **Clustering** (e.g., k-means, DBSCAN) can then group similar failure events together.
                
                **Regulatory Significance:** This demonstrates a mature, proactive quality system. While a **5 Whys** analysis is sufficient for a single CAPA, an NLP analysis provides evidence that the organization is monitoring for *systemic* issues, a key tenet of **21 CFR 820.100**.
                """)
                
# ==============================================================================
# PAGE 4: IMPROVE PHASE
# ==============================================================================
def show_improve_phase():
    st.title("⚙️ Improve: Optimization & Robustness")
    st.markdown("**Objective:** To identify, test, and implement solutions that address validated root causes, finding optimal settings for critical parameters to create a robust process.")
    st.markdown("> **Applicable Regulatory Stages:** ICH Q8 (Design Space), FDA Process Validation (Stage 1)")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Design Space & Process Optimization")
        st.markdown("Finding the combination of process parameters that reliably delivers a quality product.")
        tab1, tab2 = st.tabs(["🏛️ Classical: DOE & Response Surface", "🤖 ML Augmentation: Surrogate Modeling"])
        with tab1:
            st.markdown("##### **Design of Experiments (DOE) & Response Surface Methodology (RSM)**")
            doe_data = generate_doe_data()
            fig_main, fig_int = plot_doe_effects(doe_data)
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(fig_main, use_container_width=True)
            with col2: st.plotly_chart(fig_int, use_container_width=True)
            st.markdown("---")
            st.plotly_chart(plot_rsm_contour(generate_rsm_data()), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** DOE and RSM are the gold standard for efficiently exploring process parameters and their interactions to find an optimal operating window.
                
                **Regulatory Significance:** This is the primary method for defining and justifying the **Design Space** under ICH Q8. A well-defined Design Space provides operational flexibility and is a sign of a deep process understanding, which is highly valued by regulators.
                """)
        with tab2:
            st.markdown("##### **Surrogate Modeling for Accelerated Optimization**")
            st.info("Here, an ML model like a **Gaussian Process Regression (GPR)** acts as a 'digital twin' of the process, learning from limited experimental data to predict the outcome for any combination of parameters.")
            st.warning("Live GPR Surrogate Model visualization is pending implementation.")
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** A GPR model is trained on the results of an initial, smaller DOE. This allows for intelligent subsequent experimentation (see Bayesian Optimization) to find the true optimum faster.
                
                **Regulatory Significance:** Surrogate models can drastically reduce the number of physical experiments needed to map a complex design space. The model is used for *exploration*; the final proposed Design Space must still be **confirmed with physical validation runs**. The model development and its use in guiding experiments is documented in the DHF, showcasing a state-of-the-art, efficient development process.
                """)

# ==============================================================================
# PAGE 5: CONTROL PHASE
# ==============================================================================
def show_control_phase():
    st.title("📡 Control: Monitoring & Surveillance")
    st.markdown("**Objective:** To implement a robust Quality Control (QC) system to monitor the optimized process, ensuring performance remains stable and compliant over time.")
    st.markdown("> **Applicable Regulatory Stages:** Continued Process Verification (CPV, FDA Stage 3), Post-Market Surveillance (PMS)")
    st.markdown("---")
    
    with st.container(border=True):
        st.subheader("1. Ongoing Process Monitoring (CPV)")
        st.markdown("Using Statistical Process Control (SPC) to ensure the process remains in its validated state.")
        st.sidebar.header("🔬 Simulators"); st.sidebar.markdown("---"); st.sidebar.subheader("SPC Simulator")
        shift_mag = st.sidebar.slider("Shift Magnitude (Std Devs)", 0.2, 3.0, 0.8, 0.1, key="ctrl_shift")
        chart_data = generate_control_chart_data(shift_magnitude=shift_mag)

        tab1, tab2 = st.tabs(["🏛️ Classical: SPC Control Charts", "🤖 ML Augmentation: Predictive Monitoring"])
        with tab1:
            st.markdown("##### **Levey-Jennings & Trend Charts**")
            st.plotly_chart(plot_shewhart_chart(chart_data), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** **Control Charts** (like the Levey-Jennings chart shown) are the fundamental tool of SPC. They plot process data over time against statistically derived control limits. **Trend Charts** and **Cpk Trending** are also used to monitor performance over longer periods.
                
                **Regulatory Significance:** A robust CPV program using SPC charts is a regulatory expectation under FDA's Stage 3 Validation guidance. It provides ongoing assurance that the process remains in the validated state.
                """)
        with tab2:
            st.markdown("##### **Predictive Maintenance & Drift Detection**")
            st.info("ML models can be used to predict equipment failure (Predictive Maintenance) or detect subtle, multivariate process drifts that individual SPC charts might miss. This includes **Time Series Models (ARIMA, LSTM)** and **Ensemble Models (XGBoost, RF)** for early warning classification.")
            st.warning("Predictive maintenance visualization is pending implementation.")

    with st.container(border=True):
        st.subheader("2. Post-Market Surveillance (PMS)")
        st.markdown("Actively monitoring real-world data to ensure continued safety and effectiveness.")
        tab3, tab4 = st.tabs(["🏛️ Classical: Manual Review & Trending", "🤖 ML Augmentation: Automated Signal Detection"])
        with tab3:
            st.markdown("##### **Manual Complaint Categorization & Trend Charts**")
            st.info("The classical approach involves manually reading and categorizing complaints/adverse events, then using **Trend Charts** or **Stratified Analysis** to look for patterns. This is slow and struggles with unstructured data.")
        with tab4:
            st.markdown("##### **NLP on Adverse Event Reports**")
            st.plotly_chart(plot_adverse_event_clusters(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** **NLP** and **Topic Modeling** process thousands of unstructured text reports to automatically group them into clusters of similar events. This can also be used for **sentiment analysis** on patient feedback.
                
                **Regulatory Significance:** This is highly aligned with the FDA's **Real-World Evidence (RWE)** program. It enables much earlier detection of safety signals (e.g., unexpected combinations of side effects) than manual review, demonstrating a state-of-the-art PMS system.
                """)

    with st.container(border=True):
        st.subheader("3. Digital Health / SaMD Control Plan")
        st.markdown("Managing the lifecycle of an AI/ML medical device.")
        st.plotly_chart(plot_pccp_monitoring(), use_container_width=True)
        with st.expander("Methodology & Regulatory Significance"):
            st.markdown("""
            **Methodology:** A **Pre-determined Change Control Plan (PCCP)** is a comprehensive plan submitted to regulators that prospectively defines how an AI/ML model will be monitored and updated.
            
            **Regulatory Significance:** This demonstrates a controlled, auditable process for managing a learning algorithm, satisfying FDA's need for safety and effectiveness while allowing for state-of-the-art model maintenance under **Good Machine Learning Practice (GMLP)**. The **Control Plan** includes performance monitoring, re-validation plans, and robust **MLOps pipelines**.
            """)

# ==============================================================================
# PAGE 6: STRATEGIC & REGULATORY SYNTHESIS
# ==============================================================================
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

    st.success("""**Executive Summary: The Hybrid Mandate**
    - **Classical Statistics is the language of compliance.** It is used to generate the definitive, auditable evidence for your Design History File and regulatory submissions.
    - **Machine Learning is the engine of discovery and efficiency.** It is used to find patterns, predict outcomes, and automate analysis at a scale and speed that classical methods cannot match.
    
    An integrated, hybrid approach is not optional; it is the new standard for excellence and competitiveness in the regulated life sciences.
    """)
