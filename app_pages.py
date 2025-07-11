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
            with st.expander("📊 Methodology & Interpretation"):
                st.markdown("""
                **Methodology:** The Kano model is a framework for categorizing product or service features based on how they impact customer satisfaction. It helps prioritize development efforts.
                - **Basic (Must-be):** Features that are taken for granted. Their absence causes dissatisfaction, but their presence doesn't increase satisfaction (e.g., a car having brakes).
                - **Performance:** Features for which more is better. Satisfaction is proportional to how well they are implemented (e.g., a car's fuel efficiency).
                - **Excitement (Delighter):** Unexpected features that create high satisfaction when present, but their absence does not cause dissatisfaction (e.g., the first time a car had a built-in GPS).
                
                **Interpretation:** The chart shows the relationship between feature implementation (x-axis) and customer satisfaction (y-axis) for each category. This model is crucial for defining the Target Product Profile (TPP) and ensuring resources are focused on features that deliver the most value to the user (e.g., the clinician).
                """)
        with tab3:
            st.markdown("##### **ML Augmentation: NLP on Scientific Literature**")
            st.plotly_chart(plot_voc_bubble_chart(), use_container_width=True)
            with st.expander("📊 Methodology & Interpretation"):
                st.markdown("""
                **Methodology:** Natural Language Processing (NLP) models, like transformers (e.g., SciBERT), are used to parse thousands of scientific publications. They perform two key tasks:
                1.  **Topic Modeling:** Identify and group recurring themes, such as biomarkers (`EGFR`, `KRAS`) or lab methods (`ddPCR`).
                2.  **Sentiment Analysis:** Score the context around each topic to determine if it's discussed positively (e.g., "highly effective") or negatively (e.g., "prone to artifacts").

                **Interpretation:**
                - **Y-Axis (Sentiment):** Higher values indicate more positive language.
                - **Bubble Size (Count):** Larger bubbles mean the topic was mentioned more frequently.
                - **Takeaway:** This allows a rapid, data-driven survey of the scientific landscape. Here, we see that `LOD <0.1%` is a highly desirable and frequently mentioned performance goal. `ddPCR` and `Shallow WGS` are mentioned less and have negative sentiment, suggesting they may be viewed as less favorable or have known issues in the literature.
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
        with st.expander("📊 Methodology & Interpretation"):
            st.markdown("""
                **Methodology:** A Gage R&R (Repeatability & Reproducibility) study is a designed experiment to quantify the amount of variation in a measurement system. It partitions the total observed variance into its components. The goal is for the vast majority of variation to come from the parts being measured (the 'Assay Variation'), not the measurement system itself.
                
                **Interpretation:**
                - **Bars (Contribution):** Shows the percentage of total measurement error attributable to each source.
                - **Line (Cumulative %):** Shows the cumulative contribution, highlighting the "vital few" sources according to the 80/20 rule.
                - **Takeaway:** Here, 92% of variation is from the assay itself (good). The measurement system (`Repeatability` from the sequencer and `Reproducibility` between operators) contributes very little error. This gives us confidence to proceed. If `Reproducibility` were high, it would signal a need for better operator training.
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
            
        with st.expander("📊 Methodology & Interpretation"):
            st.subheader("Methodology")
            st.markdown("Capability analysis measures how well a process can meet its specification limits. It compares the 'Voice of the Process' (the actual distribution of your data) to the 'Voice of the Customer' (the required specification limits).")
            st.subheader("The Math Behind It")
            st.markdown("Two key metrics are calculated:")
            st.latex(r'''
            C_p = \frac{\text{Specification Width}}{\text{Process Width}} = \frac{USL - LSL}{6\sigma}
            ''')
            st.markdown("- **Cp (Process Potential):** Measures if the process variation is *narrow enough* to fit. It ignores the process average.")
            st.latex(r'''
            C_{pk} = \min\left(\frac{USL - \mu}{3\sigma}, \frac{\mu - LSL}{3\sigma}\right)
            ''')
            st.markdown("- **Cpk (Process Capability):** Measures if the process is *actually capable and centered*. This is the true measure of performance. It accounts for both variation and centering.")
            st.markdown("Where: $USL$ = Upper Spec Limit, $LSL$ = Lower Spec Limit, $\mu$ = Process Mean, $\sigma$ = Process Standard Deviation.")
            st.subheader("Interpretation")
            st.markdown("- A value > 1.33 is generally considered capable for most processes. \n- A high Cp but a low Cpk indicates that your process has low variation but is off-center. You need to shift the mean. \n- A low Cp and low Cpk indicates that your process has too much variation. You need to investigate root causes to reduce it.")
            
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
            with st.expander("📊 Methodology & Interpretation"):
                st.subheader("Methodology")
                st.markdown("ANOVA is a statistical test used to determine whether there are any statistically significant differences between the means of two or more independent groups. It works by partitioning the total variance into variance *between* groups and variance *within* groups.")
                st.subheader("The Math Behind It")
                st.markdown("The core of ANOVA is the F-statistic:")
                st.latex(r''' F = \frac{\text{Mean Square Between Groups (MSB)}}{\text{Mean Square Within Groups (MSW)}} = \frac{\text{Variance between groups}}{\text{Variance within groups}} ''')
                st.markdown("- A large F-statistic suggests that the variation between the groups is greater than the variation within the groups, implying a real difference between the group means.")
                st.subheader("Interpretation")
                st.markdown("- **Null Hypothesis ($H_0$):** All group means are equal. \n- **P-value:** The probability of observing an F-statistic as large as the one calculated, assuming the null hypothesis is true. \n- **Conclusion:** If the p-value is small (typically < 0.05), we reject the null hypothesis and conclude that at least one group mean is different from the others.")
        with tab2:
            st.markdown("##### **ML Augmentation: Permutation Testing**")
            st.plotly_chart(plot_permutation_test(anova_data), use_container_width=True)
            with st.expander("📊 Methodology & Interpretation"):
                st.subheader("Methodology")
                st.markdown("A permutation test is a non-parametric method that does not rely on assumptions about the data's distribution (like normality). It directly simulates the null hypothesis.")
                st.markdown("1. Calculate the observed difference between the means of two groups. \n2. Pool all the data together. \n3. Repeatedly (e.g., 1000 times) shuffle the pooled data and randomly re-assign it to the two groups. \n4. Calculate the difference in means for each shuffled permutation to create a distribution of differences under the null hypothesis. \n5. The p-value is the proportion of permuted differences that are at least as extreme as the observed difference.")
                st.subheader("Interpretation")
                st.markdown("- The histogram shows the distribution of mean differences that could occur by chance alone. \n- The dashed red line shows the actual difference you observed in your experiment. \n- **Takeaway:** If your observed difference is far out in the tails of the histogram, it's unlikely to have occurred by chance, resulting in a small p-value.")

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
            with st.expander("📊 Methodology & Interpretation"):
                st.subheader("Methodology")
                st.markdown("Design of Experiments (DOE) is a systematic method to determine the relationship between factors affecting a process and the output of that process. It is far more efficient than changing one factor at a time.")
                st.subheader("The Math Behind It")
                st.markdown("- **Main Effect:** The average change in the output (response) when a factor is changed from its low level to its high level.")
                st.latex(r''' \text{Effect}_A = \bar{y}_{A, \text{high}} - \bar{y}_{A, \text{low}} ''')
                st.markdown("- **Interaction Effect:** Occurs when the effect of one factor depends on the level of another factor.")
                st.latex(r''' \text{Interaction}_{AB} = \frac{\text{Effect}_A(\text{at } B_{high}) - \text{Effect}_A(\text{at } B_{low})}{2} ''')
                st.subheader("Interpretation")
                st.markdown("- **3D Cube Plot:** Visualizes the design space. Each corner is an experimental run. The color shows the measured `Library Yield`. \n- **Main Effects Plot:** Shows the average effect of each factor. A large bar (positive or negative) indicates a strong influence. Here, `Anneal Temp` has the largest (negative) effect. \n- **Interaction Plot:** Checks for interactions. **Non-parallel lines indicate a significant interaction.** Here, the lines are not parallel, showing a strong interaction between `Anneal Temp` and `PCR Cycles`. \n- **Takeaway:** To maximize yield, we should set `Anneal Temp` to Low (-1) and `PCR Cycles` to High (+1), a conclusion that would be difficult to reach by testing one factor at a time.")
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
            with st.expander("📊 Methodology & Interpretation"):
                st.subheader("Methodology")
                st.markdown("Bayesian Optimization is an intelligent search algorithm for finding the optimum of an expensive-to-evaluate function. It works in a loop:")
                st.markdown("1. **Build a surrogate model:** Fit a probabilistic model (like a Gaussian Process) to all the data collected so far. \n2. **Define an acquisition function:** Use the model's predictions and uncertainty to decide where to sample next. \n3. **Run the experiment:** Perform the experiment at the new point and add the result to the dataset.")
                st.subheader("The Math Behind It")
                st.markdown("A common acquisition function is Upper Confidence Bound (UCB):")
                st.latex(r''' \text{UCB}(x) = \mu(x) + \kappa\sigma(x) ''')
                st.markdown("Where $\mu(x)$ is the model's mean prediction at point $x$, $\sigma(x)$ is the standard deviation (uncertainty), and $\kappa$ is a parameter that balances **exploitation** (sampling where the mean is high) and **exploration** (sampling where uncertainty is high).")
                st.subheader("Interpretation")
                st.markdown("- **Takeaway:** The algorithm avoids wasting experiments on regions it knows are poor. It intelligently chooses points to quickly find the peak of the hidden performance curve, saving time and resources compared to a brute-force search or a full DOE.")

    with st.container(border=True):
        st.subheader("2. Proactively Mitigating Risks")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### **Classical: FMEA**")
            st.plotly_chart(plot_fmea_table(), use_container_width=True)
            with st.expander("📊 Methodology & Interpretation"):
                st.subheader("Methodology")
                st.markdown("Failure Mode and Effects Analysis (FMEA) is a systematic, team-based activity to identify and prevent potential process failures. Each potential failure mode is scored on three criteria:")
                st.markdown("- **Severity (S):** How severe is the effect of the failure? (1-10) \n- **Occurrence (O):** How frequently is the failure likely to occur? (1-10) \n- **Detection (D):** How likely are we to detect the failure before it reaches the customer? (1-10, where 1 is very likely to detect)")
                st.subheader("The Math Behind It")
                st.latex(r''' \text{Risk Priority Number (RPN)} = S \times O \times D ''')
                st.markdown("- The RPN is used to rank and prioritize the failure modes for corrective action.")
        with col4:
            st.markdown("##### **ML Augmentation: Predictive Maintenance**")
            st.plotly_chart(plot_rul_prediction(generate_sensor_degradation_data()), use_container_width=True)
            with st.expander("📊 Methodology & Interpretation"):
                st.subheader("Methodology")
                st.markdown("Predictive maintenance uses sensor data from equipment to predict when a component is likely to fail. In this case, we model the degradation of a sequencer's laser power over time.")
                st.subheader("The Math Behind It")
                st.markdown("The degradation is modeled as an exponential decay process:")
                st.latex(r''' P(t) = P_0 e^{-kt} ''')
                st.markdown("Where $P(t)$ is the power at time $t$, $P_0$ is the initial power, and $k$ is the decay constant. We fit this model to historical data to predict the Remaining Useful Life (RUL) before the power drops below a failure threshold.")

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
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Levey-Jennings", "📈 EWMA", "📉 CUSUM", "🤖 Multivariate QC"])
        with tab1:
            st.markdown("##### **Classical: Levey-Jennings Chart (Shewhart)**")
            st.plotly_chart(plot_shewhart_chart(chart_data), use_container_width=True)
            with st.expander("📊 Methodology & Interpretation"):
                st.subheader("Methodology"); st.markdown("The most common SPC chart. It plots individual QC measurements against control limits derived from the process's historical mean and standard deviation.")
                st.subheader("The Math Behind It"); st.latex(r''' \text{UCL/LCL} = \mu \pm 3\sigma '''); st.markdown("Where $\mu$ is the historical mean and $\sigma$ is the historical standard deviation.")
                st.subheader("Interpretation"); st.markdown("- **Best for:** Detecting large, sudden process shifts. \n- **Limitation:** It is not sensitive to small, gradual drifts, as it has no 'memory' of past data points.")
        with tab2:
            st.markdown("##### **Advanced: EWMA Chart**")
            st.plotly_chart(plot_ewma_chart(chart_data, lambda_val=ewma_lambda), use_container_width=True)
            with st.expander("📊 Methodology & Interpretation"):
                st.subheader("Methodology"); st.markdown("An Exponentially Weighted Moving Average chart gives more weight to recent data points, essentially having a 'memory'.")
                st.subheader("The Math Behind It"); st.latex(r''' E_t = \lambda y_t + (1-\lambda)E_{t-1} '''); st.markdown("Where $E_t$ is the EWMA value at time $t$, $y_t$ is the current observation, and $\lambda$ (lambda) is a smoothing constant (0 < $\lambda$ ≤ 1).")
                st.subheader("Interpretation"); st.markdown("- **Best for:** Detecting small, sustained drifts in the process mean (e.g., slow reagent degradation). \n- **Takeaway:** The EWMA chart (green line) will often detect a small shift much earlier than a standard Levey-Jennings chart. A smaller $\lambda$ is more sensitive to smaller shifts.")
        with tab3:
            st.markdown("##### **Advanced: CUSUM Chart**")
            st.plotly_chart(plot_cusum_chart(chart_data), use_container_width=True)
            with st.expander("📊 Methodology & Interpretation"):
                st.subheader("Methodology"); st.markdown("A Cumulative Sum chart plots the cumulative sum of deviations from a target value. It is extremely sensitive to small, persistent shifts.")
                st.subheader("The Math Behind It"); st.latex(r''' S_H(i) = \max(0, S_H(i-1) + (y_i - \mu_0) - k) '''); st.latex(r''' S_L(i) = \min(0, S_L(i-1) + (y_i - \mu_0) + k) '''); st.markdown("Where $S_H$ and $S_L$ are the upper and lower CUSUMs, $k$ is an allowance or 'slack' value, and an out-of-control signal occurs if $S_H$ or $S_L$ cross a control limit $H$.")
                st.subheader("Interpretation"); st.markdown("- **Best for:** Detecting very small, persistent shifts in the process mean. \n- **Takeaway:** When a process has a very small shift, the CUSUM value will steadily increase (or decrease) until it crosses the control limit, signaling a problem often invisible to other charts.")
        with tab4:
            st.markdown("##### **ML Augmentation: Multivariate QC**")
            st.plotly_chart(plot_hotelling_t2_chart(), use_container_width=True)
            with st.expander("📊 Methodology & Interpretation"):
                st.subheader("Methodology"); st.markdown("Monitors multiple correlated variables at once. Instead of separate charts for `% Mapped` and `% Duplication`, it combines them into a single statistic that represents the overall 'health' of the QC profile.")
                st.subheader("The Math Behind It"); st.markdown("Hotelling's T² statistic measures the statistical distance of a data point from the center of a multivariate distribution, accounting for the covariance between variables."); st.latex(r''' T^2 = (\mathbf{x} - \bar{\mathbf{x}})^T \mathbf{S}^{-1} (\mathbf{x} - \bar{\mathbf{x}}) '''); st.markdown("Where $\mathbf{x}$ is the vector of current observations, $\bar{\mathbf{x}}$ is the vector of historical means, and $\mathbf{S}^{-1}$ is the inverse of the covariance matrix.")
                st.subheader("Interpretation"); st.markdown("- **Takeaway:** A single point on this chart can be out of control even if all individual metrics are within their own limits. This is powerful for detecting subtle, correlated shifts that would otherwise be missed.")

    with st.container(border=True):
        st.subheader("2. Formalizing the Gains: The Control Plan & SOPs")
        st.plotly_chart(plot_control_plan(), use_container_width=True)

    st.success("""**🏆 Hybrid Strategy for the Control Phase:**\n1. **Monitor with Levey-Jennings:** Use a classical **Levey-Jennings chart** for primary positive/negative controls for simplicity and regulatory compliance.\n2. **Detect Drifts with EWMA:** For critical secondary metrics, use a more sensitive **EWMA chart** to detect slow reagent or instrument degradation.\n3. **Holistic QC with ML:** For each sample, run a **multivariate QC model** (like Hotelling's T²) on the full profile of NGS QC metrics to flag subtle issues.\n4. **Codify Everything:** The **Control Plan** and **SOPs** must document all charts, limits, and reaction plans.""")

# ==============================================================================
# PAGE 6 & 7: COMPARISON & MANIFESTO
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
    
    with st.container(border=True):
        st.subheader("The Philosophy of Synergy: Inference + Prediction")
        st.plotly_chart(plot_synergy_diagram(), use_container_width=True)

    with st.container(border=True):
        st.subheader("A Granular, Head-to-Head Comparison")
        
        st.markdown("##### Attribute Comparison Matrix")
        st.plotly_chart(plot_attribute_matrix(), use_container_width=True)
        
        st.markdown("##### The Verdict: Strengths at a Glance")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("<h5 style='text-align: center; color: #0072B2;'>🏆 Classical Statistics' Core Strengths</h5>", unsafe_allow_html=True)
                st.markdown("""
                - **Interpretability:** Models and outputs are easier to explain and defend without extra tools.
                - **Ease of Implementation:** Tools (like Excel/Minitab) and training are widely available and simpler for basic analyses.
                - **Auditability & Compliance:** Methods are standardized and preferred in regulated industries (e.g., FDA, CLIA).
                """)
        with col2:
            with st.container(border=True):
                st.markdown("<h5 style='text-align: center; color: #009E73;'>🏆 Machine Learning's Core Strengths</h5>", unsafe_allow_html=True)
                st.markdown("""
                - **Scalability:** Natively handles larger, messier, and higher-dimensional datasets.
                - **Accuracy in Complex Systems:** Effectively captures nonlinear patterns and complex interactions.
                - **Proactive Detection:** Designed to predict future outcomes, not just describe past events.
                """)

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
