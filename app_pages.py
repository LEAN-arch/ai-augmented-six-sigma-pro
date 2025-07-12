# app_pages.py

import streamlit as st
import numpy as np
import pandas as pd

# Import all necessary helper functions from the single, definitive helper file.
from app_helpers import *

# ==============================================================================
# PAGE 0: FRAMEWORK OVERVIEW
# ==============================================================================
def show_framework_overview():
    st.title("The Bio-AI Excellence Framework")
    st.markdown("##### An integrated playbook for **compliant** and **accelerated** development of drugs, devices, and diagnostics.")
    st.markdown("---")

    st.subheader("The Core Philosophy: Two Frameworks, One Goal")
    st.markdown("""
    In the regulated life sciences, success is not just about innovation; it's about **compliant innovation**. This application demonstrates how to achieve this by weaving two powerful frameworks together:
    - **The Regulatory & QMS Framework (The 'What' & 'Why'):** The non-negotiable lifecycle stages and quality systems mandated by bodies like the FDA and defined by standards like ICH and ISO. This provides the structure and defines the required evidence.
    - **The Data Science & Process Improvement Toolkit (The 'How'):** The collection of Six Sigma and Machine Learning tools used to generate that evidence efficiently, solve complex problems, and accelerate development *within* the regulatory guardrails.
    """)
    st.plotly_chart(plot_framework_diagram(), use_container_width=True)
    st.info("Use the sidebar to navigate through the GxP Product Lifecycle. Each page will demonstrate how classical and AI-augmented tools are applied at that specific regulatory stage.")

# ==============================================================================
# GXP PRODUCT LIFECYCLE PAGES
# ==============================================================================

def show_design_and_development():
    st.title("1. Design & Development")
    st.markdown("> **Regulatory Context:** FDA Design Controls (21 CFR 820.30), ICH Q8 (Pharmaceutical Development), ISO 13485")
    st.markdown("**Objective:** To translate user needs into a robust, well-defined product design, and to proactively identify and mitigate design-related risks.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("Use Case: Requirements Translation & Design Input Prioritization")
        st.markdown("The challenge is to translate vague user needs (e.g., 'an accurate test') into specific, quantifiable engineering specifications (Design Inputs) and to provide objective evidence for their prioritization.")
        
        tab1, tab2 = st.tabs(["🏛️ Classical: Quality Function Deployment (QFD)", "🤖 ML Augmentation: Data-Driven Feature Importance"])
        with tab1:
            st.markdown("##### **Quality Function Deployment (QFD) - House of Quality**")
            st.plotly_chart(plot_qfd_house_of_quality(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** QFD is a structured, team-based method for translating the 'Voice of the Customer' (VOC) into the 'Voice of the Engineer'. The 'House of Quality' matrix maps customer needs to technical characteristics, scoring the strength of their relationship. The bar chart at the bottom calculates a weighted 'Technical Importance Score', highlighting the most critical engineering parameters.
                
                **Regulatory Significance:** This creates a **traceable, documented link** between user needs and design inputs, a core requirement of FDA Design Controls (21 CFR 820.30). It provides objective evidence for *why* certain technical specifications were prioritized, which is essential for design reviews and regulatory submissions. Here, the chart clearly justifies focusing engineering effort on `LOD (VAF %)` and `Specificity (%)`.
                """)
        with tab2:
            st.markdown("##### **ML-Driven Feature Importance from Pilot Data**")
            st.warning("This approach requires preliminary or pilot data where both technical parameters and a key performance outcome have been measured.")
            df_reg = generate_nonlinear_data()
            _, model, X_reg = train_and_plot_regression_models(df_reg)
            st.plotly_chart(plot_shap_summary(model, X_reg), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** An ML model (e.g., Random Forest) is trained on pilot data to predict a key performance outcome (e.g., 'On-Target Rate'). eXplainable AI (XAI) tools like SHAP then quantify the impact of each input parameter on the model's prediction, revealing the true drivers of performance.
                
                **Regulatory Significance:** While not a replacement for QFD in initial documentation, SHAP provides powerful, data-driven evidence to **support or challenge** the assumptions made in the QFD. If SHAP reveals a parameter thought to be unimportant is actually critical (or vice-versa), this is a vital finding for de-risking the project. This analysis should be documented in the Design History File (DHF) as evidence of a data-driven design process.
                """)

    with st.container(border=True):
        st.subheader("Use Case: Early Risk Assessment")
        st.markdown("How do we proactively identify potential design failures and emergent risks *before* they are locked into the product design?")
        tab3, tab4 = st.tabs(["🏛️ Classical: Design FMEA (DFMEA)", "🤖 ML Augmentation: Unsupervised Risk Clustering"])
        with tab3:
            st.markdown("##### **Design Failure Mode & Effects Analysis (DFMEA)**")
            st.plotly_chart(plot_dfmea_table(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** A systematic, bottom-up analysis of potential failure modes in a product's design. Each failure is scored for Severity (S), Occurrence (O), and Detection (D) to calculate a Risk Priority Number (RPN = S x O x D), which ranks risks for mitigation.
                
                **Regulatory Significance:** DFMEA is a cornerstone of risk management under ISO 14971 and a required part of the Design History File. It demonstrates to auditors that risks have been considered and prioritized for mitigation. High RPN items become mandatory inputs for design improvements or additional controls.
                """)
        with tab4:
            st.markdown("##### **Unsupervised Clustering for Risk Signal Grouping**")
            st.plotly_chart(plot_risk_signal_clusters(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** Unsupervised ML algorithms (like DBSCAN, used here) are applied to multi-parameter process data from early development runs. The algorithm identifies 'clusters' of data points that behave similarly and flags points that don't belong to any cluster as anomalies, without any prior labels.
                
                **Regulatory Significance:** This technique can uncover **hidden failure modes or process interactions** that were not conceived of during the team-based DFMEA. For instance, the plot reveals a distinct underperforming cluster of data from 'Line A' and a set of dangerous 'Anomalous Events'. This discovery would trigger an investigation and an update to the DFMEA, demonstrating a data-driven and proactive approach to risk management that regulators value.
                """)

    st.success("""**🏆 Hybrid Strategy for Design & Development:**
    1.  **Mandate with QFD/DFMEA:** Use **QFD** and **DFMEA** as the formal, auditable foundation for documenting design inputs and risks, satisfying 21 CFR 820.30 and ISO 14971.
    2.  **Discover with ML:** Leverage **ML Feature Importance** on pilot data to validate QFD assumptions and use **ML Clustering** to find unanticipated risk signals, feeding these data-driven insights back into the formal Risk Management File and Design History File.
    """)

def show_process_characterization():
    st.title("2. Process Characterization")
    st.markdown("> **Regulatory Context:** ICH Q8/Q11, FDA Process Validation Guidance (Stage 1)")
    st.markdown("**Objective:** To identify and understand the sources of process variability, defining Critical Process Parameters (CPPs) and their relationship to Critical Quality Attributes (CQAs) to establish a Design Space.")
    st.markdown("---")
    
    with st.container(border=True):
        st.subheader("Use Case: Design Space Exploration & Robustness Studies")
        st.markdown("How do we efficiently explore the relationships between process parameters and product quality to define a robust operating region?")
        tab1, tab2 = st.tabs(["🏛️ Classical: DOE & Response Surface Methodology", "🤖 ML Augmentation: Surrogate Modeling"])
        with tab1:
            st.markdown("##### **Design of Experiments (DOE) & Response Surface Methodology (RSM)**")
            col1, col2 = st.columns(2)
            with col1:
                doe_data = generate_doe_data()
                fig_main, fig_int = plot_doe_effects(doe_data)
                st.plotly_chart(fig_main, use_container_width=True)
            with col2:
                st.plotly_chart(fig_int, use_container_width=True)
            
            st.markdown("---")
            st.plotly_chart(plot_rsm_contour(generate_rsm_data()), use_container_width=True)

            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:**
                - **DOE:** A structured approach to experimentation that varies multiple factors simultaneously to identify main effects and interactions with maximum statistical power and minimal runs.
                - **RSM:** A more advanced form of DOE used to find the optimal settings. It fits a model to the experimental data to create a 'map' (contour plot) of the response surface, visualizing the optimal operating conditions.
                
                **Regulatory Significance:** DOE and RSM are the gold standard for process characterization under ICH Q8. They provide the statistical evidence needed to classify parameters as critical (CPPs) and to define and justify the **Design Space**—the multidimensional combination of input variables that has been demonstrated to provide assurance of quality. Operating within this validated space is not considered a change and provides operational flexibility.
                """)
        with tab2:
            st.markdown("##### **ML-based Surrogate Modeling (Gaussian Process)**")
            st.info("Here, an ML model like a Gaussian Process Regression (GPR) acts as a 'digital twin' of the process. It learns from limited experimental data and can then predict the outcome for any combination of parameters, including uncertainty estimates.")
            st.warning("Live GPR Surrogate Model visualization is pending implementation.")
            with st.expander("Methodology & Regulatory Significance"):
                 st.markdown("""
                **Methodology:** A GPR model is trained on the results of an initial, smaller DOE. This model doesn't just predict the outcome; it also predicts its own uncertainty. This allows for intelligent subsequent experimentation (see Bayesian Optimization).
                
                **Regulatory Significance:** Surrogate models are powerful for **accelerating development**. They can drastically reduce the number of physical experiments needed to map a complex design space. In a regulatory context, the surrogate model is used for *exploration*; the final proposed Design Space must still be confirmed with a small number of physical validation runs at the proposed boundaries. The development of the model is documented in the DHF.
                """)
    st.success("""**🏆 Hybrid Strategy for Process Characterization:**
    1.  **Screen with DOE:** Use a classical **Fractional Factorial DOE** to efficiently screen many parameters and identify the likely main drivers.
    2.  **Model with RSM/GPR:** For the critical few parameters, use **RSM** for simpler processes. For complex, high-dimensional, or expensive processes, build a **GPR surrogate model** to map the design space *in silico*.
    3.  **Confirm & File:** Use the model's predictions to define the optimal Design Space, then run a final set of **confirmatory experiments** at the edges of this space to provide the definitive validation data for the regulatory submission.
    """)

def show_process_validation_ppq():
    st.title("3. Process Validation (PPQ)")
    st.markdown("> **Regulatory Context:** FDA 2011 Process Validation Guidance (Stage 2), ICH Q8–Q10")
    st.markdown("**Objective:** To confirm the Design Space and demonstrate that the manufacturing process, operated within established parameters, consistently produces product meeting all predetermined specifications.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("Use Case: Demonstrating Batch-to-Batch Consistency")
        st.markdown("The core of Process Performance Qualification (PPQ) is providing objective, statistical evidence of consistency across a series of production-scale batches (typically at least 3).")
        st.sidebar.header("🔬 Simulators"); st.sidebar.markdown("---"); st.sidebar.subheader("PPQ Simulator")
        lsl = st.sidebar.slider("Lower Spec Limit (LSL)", 80.0, 88.0, 85.0, key="ppq_lsl")
        usl = st.sidebar.slider("Upper Spec Limit (USL)", 110.0, 120.0, 115.0, key="ppq_usl")
        process_mean = st.sidebar.slider("Process Mean (μ)", 95.0, 105.0, 100.0, key="ppq_mean")
        process_std = st.sidebar.slider("Process Std Dev (σ)", 1.0, 5.0, 2.0, key="ppq_std")

        data = generate_process_data(process_mean, process_std, 300)
        fig_cap, cp, cpk = plot_capability_analysis_pro(data, lsl, usl)

        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown("##### **Classical: Process Capability Analysis (Cp, Cpk)**")
            st.plotly_chart(fig_cap, use_container_width=True)
        with col2:
            st.markdown("##### **Capability Indices**")
            cp_color = "success" if cp >= 1.33 else ("warning" if cp >= 1.0 else "error")
            cpk_color = "success" if cpk >= 1.33 else ("warning" if cpk >= 1.0 else "error")
            st.metric(label="Process Potential (Cp)", value=f"{cp:.2f}", help="Measures process potential. Target: > 1.33")
            st.markdown(f'<hr style="margin-top:0; margin-bottom:0.5rem; border-color:{COLORS[cp_color]}">', unsafe_allow_html=True)
            st.metric(label="Process Capability (Cpk)", value=f"{cpk:.2f}", help="Measures actual process performance. Target: > 1.33")
            st.markdown(f'<hr style="margin-top:0; margin-bottom:0.5rem; border-color:{COLORS[cpk_color]}">', unsafe_allow_html=True)
        
        with st.expander("Methodology & Regulatory Significance"):
            st.markdown("""
            **Methodology:** Process Capability analysis compares the 'Voice of the Process' (the actual distribution of your data, 6σ) to the 'Voice of the Customer' (the required specification limits). Cpk is the key metric, as it accounts for both process variation and centering.
            
            **Regulatory Significance:** This is the **definitive statistical evidence** required in a PPQ report. A Cpk value ≥ 1.33 is the widely accepted industry standard to demonstrate that a process is robust, capable, and in a state of control. This data forms the basis for release of the PPQ batches and approval of the manufacturing process.
            """)

    with st.container(border=True):
        st.subheader("Use Case: Validation of Analytical & ML Models")
        st.markdown("How do we rigorously validate the performance of the models (statistical or ML) used to support our process or product?")
        st.plotly_chart(plot_model_validation_ci(), use_container_width=True)
        with st.expander("Methodology & Regulatory Significance"):
            st.markdown("""
            **Methodology:** Instead of reporting a single performance metric (e.g., Accuracy = 95%), a much more rigorous approach is to calculate a confidence interval for that metric. **Bootstrapping** is a powerful resampling method to do this:
            1. From your validation dataset, draw a random sample *with replacement*.
            2. Calculate the performance metric on this new sample.
            3. Repeat this thousands of times to create a distribution of the metric.
            4. The 95% confidence interval is the range covering the central 95% of this distribution.

            **Regulatory Significance:** Reporting a confidence interval (e.g., "The model accuracy is 95% with a 95% CI of [93.5%, 96.5%]") is far more powerful and transparent to a regulator than a single point estimate. It demonstrates a deep understanding of the model's uncertainty and stability. This is especially critical when validating an ML model, as it provides a robust estimate of its performance on unseen data.
            """)

    st.success("""**🏆 Hybrid Strategy for PPQ & Validation:**
    1.  **Anchor with Cpk:** Use **Process Capability (Cpk)** as the non-negotiable, classical standard for demonstrating manufacturing process consistency in the PPQ report.
    2.  **Validate Models with CIs:** For any supporting analytical or ML model, go beyond point estimates. Use **bootstrapping to calculate and report confidence intervals** on key performance metrics (Accuracy, R², etc.) to provide a robust and defensible validation package.
    """)

def show_cpv_and_monitoring():
    st.title("4. Continued Process Verification (CPV)")
    st.markdown("> **Regulatory Context:** FDA Process Validation Guidance (Stage 3), ICH Q10 (Pharmaceutical Quality System)")
    st.markdown("**Objective:** To implement an ongoing program to collect and analyze process data, ensuring the process remains in a state of control throughout commercial manufacturing.")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("Use Case: Real-time Monitoring & Process Deviation Prediction")
        st.markdown("How do we move from retrospective batch review to proactive, real-time monitoring that can detect and even predict deviations before they occur?")
        st.sidebar.header("🔬 Simulators"); st.sidebar.markdown("---"); st.sidebar.subheader("QC Simulator")
        shift_mag = st.sidebar.slider("Magnitude of Shift (Std Devs)", 0.2, 3.0, 0.8, 0.1, key="cpv_shift")
        ewma_lambda = st.sidebar.slider("EWMA Lambda (λ)", 0.1, 0.5, 0.2, 0.05, help="Higher λ reacts faster to shifts.")
        chart_data = generate_control_chart_data(shift_magnitude=shift_mag)

        tab1, tab2 = st.tabs(["🏛️ Classical: SPC Charts for Detection", "🤖 ML Augmentation: Early Warning"])
        with tab1:
            st.markdown("##### **Standard & Advanced Statistical Process Control (SPC)**")
            st.plotly_chart(plot_shewhart_chart(chart_data), use_container_width=True)
            st.plotly_chart(plot_ewma_chart(chart_data, lambda_val=ewma_lambda), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:**
                - **Levey-Jennings (Shewhart) Chart:** The industry standard. Plots individual QC points and is best for detecting large, sudden shifts.
                - **EWMA Chart:** A more advanced chart that weights recent data more heavily. It's far more sensitive to small, gradual process drifts (e.g., reagent degradation, instrument wear).
                
                **Regulatory Significance:** A robust CPV program utilizing SPC charts is a regulatory expectation. It provides ongoing assurance that the process remains in the validated state. While Levey-Jennings is the baseline, using more sensitive charts like EWMA demonstrates a more mature and proactive quality system.
                """)
        with tab2:
            st.markdown("##### **Predictive Classification for Early Warning**")
            st.info("Here, an ML classifier (e.g., XGBoost) could be trained on historical in-process data (sensor readings, etc.) to predict whether a batch will fail its final QC release specs *hours or days before* the batch is complete. This allows for early intervention, potentially saving the batch or preventing a costly investigation.")
            st.warning("Early warning classification model visualization is pending implementation.")

    st.success("""**🏆 Hybrid Strategy for CPV:**
    1.  **Monitor with SPC:** Use a suite of **SPC charts** (Shewhart for large shifts, EWMA for small drifts) as the foundation of the CPV program for regulatory compliance.
    2.  **Predict with ML:** For high-value or high-risk processes, develop an **ML classifier** that uses in-process data to provide an early warning of potential batch failure. This shifts the quality paradigm from 'detecting' failure to 'preventing' it, the ultimate goal of ICH Q10.
    """)

# ==============================================================================
# KEY QMS MODULES
# ==============================================================================
def show_qrm():
    st.title("Quality Risk Management (QRM)")
    st.markdown("> **Regulatory Context:** ISO 14971 (Medical Devices), ICH Q9 (Quality Risk Management)")
    st.markdown("**Objective:** To implement a systematic, lifecycle-long process for the assessment, control, communication, and review of risks to product quality and patient safety.")
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Use Case: Risk Analysis & Mitigation Modeling")
        st.markdown("How do we model the pathways to failure and quantify risk to make informed decisions?")
        tab1, tab2 = st.tabs(["🏛️ Classical: FMEA & FTA", "🤖 ML Augmentation: Probabilistic Risk Models"])
        with tab1:
            st.markdown("##### **Top-Down: Fault Tree Analysis (FTA)**")
            st.plotly_chart(plot_fault_tree_plotly(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                 st.markdown("""
                **Methodology:** FTA is a top-down, deductive failure analysis. It begins with an undesirable top-level event (e.g., a critical device failure) and traces it backwards to all the potential root causes, connected by logic gates (AND, OR).
                
                **Regulatory Significance:** FTA is invaluable for understanding complex failure pathways and justifying the focus of mitigation efforts. It provides a clear visual model for auditors, showing that the relationships between lower-level failures and a critical hazard have been thoroughly considered. It complements the bottom-up FMEA by focusing on system-level interactions.
                """)
        with tab2:
            st.markdown("##### **Bayesian Networks for Probabilistic Risk**")
            st.info("A Bayesian Network could be visualized here, showing nodes for each failure mode and directed edges representing conditional probabilities (e.g., P(Assay_Failure | Low_DNA_Input)). The network can be used to calculate the exact probability of the top event given evidence about lower-level events. This transforms a static risk file into a dynamic, learning model that can be updated with new data.")
            st.warning("Visualization of a dynamic Bayesian Network is pending implementation.")

    st.success("""**🏆 Hybrid Strategy for QRM:**
    1.  **Document with FMEA/FTA:** Use **FMEA** and **FTA** as the standard, auditable tools to build the Risk Management File. This satisfies the core requirements of ISO 14971 and ICH Q9.
    2.  **Quantify with Bayesian Nets:** For the highest-risk pathways identified in the FTA, build a **Bayesian Network**. This elevates the analysis from qualitative (High/Medium/Low) to quantitative (e.g., "a 3.5% probability of failure given current conditions"), enabling more sophisticated, data-driven, and defensible risk-based decision making.
    """)

def show_capa_rca():
    st.title("CAPA & Root Cause Analysis (RCA)")
    st.markdown("> **Regulatory Context:** FDA 21 CFR 820.100 (Corrective and Preventive Action)")
    st.markdown("**Objective:** To implement a systematic process for investigating, correcting, and—most importantly—preventing recurrences of non-conformities.")
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Use Case: Root Cause Identification")
        st.markdown("How do we move beyond the initial symptom to find the true underlying cause of a deviation, and how do we spot systemic trends across many deviations?")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Classical: 5 Whys & Fishbone**")
            st.plotly_chart(plot_5whys_diagram(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** The 5 Whys is an iterative interrogative technique used to explore the cause-and-effect relationships underlying a problem. The primary goal is to determine the root cause by repeatedly asking "Why?".
                
                **Regulatory Significance:** This provides a simple, structured, and auditable record of a single root cause investigation. It demonstrates to an auditor that the investigation was thorough and didn't stop at a superficial cause.
                """)
        with col2:
            st.markdown("##### **ML Augmentation: NLP on Deviation Logs**")
            st.plotly_chart(plot_nlp_on_capa_logs(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** Uses Natural Language Processing (NLP) to automatically read and categorize hundreds or thousands of unstructured text entries from deviation logs (e.g., CAPAs, OOS reports). By identifying recurring themes (topics), it can rapidly highlight systemic problems that are invisible when looking at events one by one.
                
                **Regulatory Significance:** This is a powerful tool for demonstrating a mature quality system. While a 5 Whys analysis is sufficient for a single CAPA, an NLP analysis provides evidence that the organization is proactively monitoring for *systemic* issues. Finding that 'Hardware Failure' is a major recurring theme would justify a large-scale preventative action, like overhauling the preventative maintenance program—the ultimate goal of the CAPA system.
                """)
    st.success("""**🏆 Hybrid Strategy for CAPA:**
    1.  **Investigate with 5 Whys:** For any individual high-severity deviation, use the **5 Whys** or a **Fishbone diagram** to guide and document a focused team investigation.
    2.  **Find Systemic Trends with NLP:** On a quarterly basis, run an **NLP topic model** across all deviation logs. This provides a macro, data-driven view of systemic issues, allowing the quality unit to focus preventive actions on the most impactful problems.
    """)

def show_post_market_surveillance():
    st.title("Post-Market Surveillance (PMS)")
    st.markdown("> **Regulatory Context:** FDA Post-Market Surveillance, FDA Real-World Evidence (RWE) Program, ICH Q10")
    st.markdown("**Objective:** To actively monitor the safety, effectiveness, and performance of a device or drug after it has been released to the market, using real-world data.")
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Use Case: Signal Detection from Real-World Data")
        st.markdown("How do we analyze diverse, high-volume, and unstructured data sources (e.g., complaints, adverse event reports, social media) to identify emerging safety signals or performance issues?")
        tab1, tab2 = st.tabs(["🏛️ Classical: Manual Categorization", "🤖 ML Augmentation: NLP & Clustering"])
        with tab1:
            st.markdown("##### **Manual Categorization & Trend Charts**")
            df_pareto = generate_pareto_data().rename(columns={'QC_Failure_Mode': 'Complaint_Category'})
            st.plotly_chart(plot_pareto_chart(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("The classical approach involves quality personnel manually reading reports, categorizing them based on a pre-defined schema, and then plotting the frequency of each category (e.g., with a Pareto chart). This is labor-intensive, slow, and can miss novel or un-categorized issues.")
        with tab2:
            st.markdown("##### **Unsupervised Clustering of Adverse Event Narratives**")
            st.plotly_chart(plot_adverse_event_clusters(), use_container_width=True)
            with st.expander("Methodology & Regulatory Significance"):
                st.markdown("""
                **Methodology:** This advanced approach uses ML to process unstructured text from thousands of reports. It involves vectorization (TF-IDF), dimensionality reduction (PCA), and clustering (K-Means) to automatically group similar reports together.
                
                **Regulatory Significance:** This method is highly aligned with the FDA's push for leveraging **Real-World Evidence (RWE)**. It can automatically discover novel groupings of adverse events (e.g., a specific combination of symptoms not previously associated with the product) that would be impossible to find with manual categorization. This enables much earlier detection of safety signals, leading to faster patient safety interventions and demonstrating a state-of-the-art PMS system to regulators.
                """)
    st.success("""**🏆 Hybrid Strategy for PMS:**
    1.  **Report with Trends:** Continue to use **Pareto charts** of manually coded complaint categories for high-level, standardized reporting.
    2.  **Discover with NLP:** Implement an **NLP clustering pipeline** to run continuously in the background on all incoming real-world data. Regularly review the discovered clusters to find emerging, non-obvious safety signals that require deeper investigation and potential reporting.
    """)

# ==============================================================================
# ADVANCED & STRATEGIC TOPICS
# ==============================================================================
def show_digital_health_samd():
    st.title("Digital Health / SaMD")
    st.markdown("> **Regulatory Context:** FDA GMLP, FDA PCCP Guidance, ISO 13485")
    st.markdown("**Objective:** To address the unique challenges of validating and maintaining AI/ML-based Software as a Medical Device (SaMD) throughout its lifecycle.")
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Use Case: Continuous Learning & Algorithm Updates")
        st.markdown("How do we manage the lifecycle of an ML model that may need to be updated after deployment, in a controlled, compliant manner?")
        st.plotly_chart(plot_pccp_monitoring(), use_container_width=True)
        with st.expander("Methodology & Regulatory Significance"):
            st.markdown("""
            **Methodology:** A **Pre-determined Change Control Plan (PCCP)** is a comprehensive plan submitted to regulators *before* marketing. It prospectively defines the "what, why, how, and where" of future model changes. A key component is continuous performance monitoring against a pre-defined threshold.
            
            **Interpretation & Regulatory Significance:** The chart shows the model's performance (AUC) over time. It remains stable until a drift in the input data (or the underlying disease) causes performance to degrade. When it crosses the pre-defined threshold (red dashed line), the PCCP is **triggered**, initiating a pre-planned, pre-approved process of retraining, re-validating, and deploying a new version of the model. This demonstrates a controlled, auditable process for managing a learning algorithm, satisfying the FDA's need for safety and effectiveness while allowing for state-of-the-art model maintenance.
            """)
    st.success("""**🏆 Hybrid Strategy for SaMD:**
    1.  **Validate with Rigor:** Use classical statistical tests and **bootstrapped confidence intervals** on a locked, independent test set to prove the **final model's performance** for the initial 510(k) or De Novo submission.
    2.  **Manage with MLOps & PCCP:** Implement a robust **MLOps pipeline** and a detailed **PCCP** to govern the entire lifecycle of the model post-deployment, from real-time performance monitoring to controlled, auditable updates. This is the new standard for compliant AI/ML in medicine.
    """)

def show_hybrid_manifesto():
    st.title("The Hybrid Manifesto & GxP Compliance")
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
