"""
helpers/content.py

This module centralizes all static text, markdown, and HTML content used
throughout the application.

Author: Bio-AI Excellence SME Collective
Version: 32.3 (Typing Hotfix)
Date: 2025-07-17

Changelog from v32.2:
- [BUGFIX] Corrected a `NameError` by importing the `Any` type from the
  `typing` module, which is used in the new case study functions.
"""

from typing import List, Dict, Any
from .styling import COLORS
from .data_generators import generate_anova_data

# ==============================================================================
# 1. UI DATA STRUCTURES
# ==============================================================================

def get_guidance_data() -> Dict[str, Dict[str, str]]:
    """
    Returns a complete and expanded dictionary of scenarios and recommended
    approaches for the interactive solution recommender, including all
    specified classical, ML, and complex regulatory cases.
    """
    return {
        # --- NEW, ADVANCED REGULATORY SCENARIOS ---
        "Validating a Class III device for a PMA submission": {
            "approach": "🏆 **Maximum Rigor Classical Stats** + **Explainable AI (XAI) for Risk Mitigation**",
            "rationale": "A Premarket Approval (PMA) is the FDA's most stringent pathway, requiring a standalone demonstration of safety and effectiveness. An unimpeachable statistical master plan is non-negotiable, typically involving large-scale clinical trials and meticulous validation (DOE, Gage R&R, Cpk). **ML/AI can supplement, but not replace, this.** Use XAI on clinical data to proactively identify patient subpopulations at higher risk, which strengthens your risk analysis (as per **ISO 14971**) and demonstrates a deep, data-driven understanding of your device's benefit-risk profile to regulators."
        },
        "Validating a Combination Product (21 CFR Part 4)": {
            "approach": "🏆 **Dual-Track Hybrid Approach**",
            "rationale": "Combination products (e.g., a pre-filled drug syringe, drug-eluting stent) are regulated by multiple FDA centers under **21 CFR Part 4**. Your evidence must satisfy both the device (CDRH) and drug (CDER) requirements. This necessitates a dual approach: **Classical DOE and SPC** for the device manufacturing process (as per the Quality System Regulation, **21 CFR 820**) and **rigorous clinical trial statistics and CMC (Chemistry, Manufacturing, and Controls)** for the drug component (as per cGMP, **21 CFR 210/211**). ML can be used for discovery and process optimization in both tracks, but the final validation packages must be distinct and meet the standards of each respective center."
        },
        "A high-risk device undergoing ISO 14971 Risk Management": {
            "approach": "🏆 **Classical FMEA/FTA** + **ML for Unk-Unk Detection**",
            "rationale": "**ISO 14971** requires a comprehensive, lifecycle-based risk management process. Use classical, structured methods like **DFMEA and Fault Tree Analysis (FTA)** to identify and mitigate *known* and *foreseeable* risks. Augment this by using **unsupervised ML models** (e.g., anomaly detection, clustering) on manufacturing and post-market data. The purpose of ML here is to uncover 'unknown-unknowns' (unk-unks)—unexpected patterns or failure modes that your human-led FMEA team did not anticipate, thus making your risk file far more robust and demonstrating proactive post-market surveillance."
        },
        "Ensuring QMS compliance under ISO 13485": {
            "approach": "🏆 **Process-Centric Classical Tools** (SPC, Cpk, MSA)",
            "rationale": "**ISO 13485** is a process-based standard for Quality Management Systems (QMS). Compliance hinges on demonstrating that your processes are defined, controlled, and effective. **Statistical Process Control (SPC)** provides objective evidence of process stability. **Process Capability (Cpk)** proves the process meets its defined requirements. **Measurement System Analysis (MSA)** ensures your process monitoring is reliable. These classical tools provide the auditable records that form the backbone of a compliant QMS."
        },
        "A device under the EU's IVDR (In-Vitro Diagnostic Regulation)": {
            "approach": "🏆 **Performance Evaluation-Focused Classical Stats**",
            "rationale": "The EU's **IVDR (Regulation (EU) 2017/746)** places immense emphasis on the **Performance Evaluation Report (PER)**. This requires rigorous data on scientific validity, analytical performance, and clinical performance. You must use established classical methods (following CLSI guidelines where applicable) for **analytical performance studies** (precision, LoD/LoQ, linearity, interference, stability). Clinical performance must be demonstrated with robust statistical analysis from patient samples. The evidence standard is extremely high and demands formal inferential statistics."
        },
        "DSCSA Compliance for a Genomic Medical Device": {
            "approach": "🏆 **Process Control & Digital Twin Hybrid**",
            "rationale": "While the **Drug Supply Chain Security Act (DSCSA)** is primarily for pharmaceuticals, its principles of interoperable, electronic, package-level tracing are becoming a best practice for high-value diagnostics. Compliance benefits from a flawlessly controlled process. Use **classical SPC** on your kitting and labeling lines to ensure zero errors. Augment this with a **predictive 'digital twin' (ML model)** of your supply chain to forecast potential bottlenecks or shipping excursions that could compromise product integrity and traceability, allowing for proactive intervention."
        },
        "Validating a 'Breakthrough Device Designation' product": {
            "approach": "🏆 **Agile Hybrid:** Bayesian Stats & Adaptive Trials",
            "rationale": "The **FDA's Breakthrough Devices Program** is designed to expedite patient access to novel devices for serious conditions. The FDA encourages frequent communication and often allows for more flexible clinical evidence generation. This is an ideal scenario for a **hybrid agile approach**. Use **Bayesian statistical methods** within an **adaptive clinical trial design**. This allows the trial to be modified at interim points based on accumulating evidence (e.g., sample size re-estimation), which is far more efficient than a traditional fixed design for demonstrating benefit-risk in an expedited fashion."
        },
        "Validating an Investigational Device Exemption (IDE) study": {
            "approach": "🏆 **Early-Stage Classical Stats & Risk-Based Monitoring**",
            "rationale": "An **IDE (21 CFR 812)** allows an unapproved device to be used in a clinical study to collect safety and effectiveness data. The primary goal is to ensure subject safety while gathering initial performance data. The focus is on **rigorous safety monitoring and data integrity**. Use **classical statistical analysis** to define the study endpoints and sample size. Augment this with **ML-driven risk-based monitoring**. Instead of auditing 100% of data, use an ML model to identify sites or data patterns that are anomalous, focusing your limited auditing resources on the highest-risk areas, thereby improving both efficiency and data quality."
        },
        # --- ORIGINAL SCENARIOS (PRESERVED) ---
        "Validating an assay for FDA 510(k) submission": {
            "approach": "🏆 **Classical Stats** (DOE, LoD/LoB studies, Gage R&R, Cpk)",
            "rationale": "For a 510(k), demonstrating **Substantial Equivalence** is key. This requires rigorous, classical statistical methods that follow established guidelines (e.g., CLSI). The outputs—p-values, confidence intervals, Cpk—are the accepted currency for proving your device is as safe and effective as a predicate device. This is a pure validation and inference task."
        },
        "Discovering a new gene signature from RNA-Seq data": {
            "approach": "🏆 **Machine Learning** (Elastic Net, Random Forest with SHAP)",
            "rationale": "This is a classic 'p >> n' problem (many more features than samples). ML excels at feature selection from high-dimensional data where classical regression would fail. ML models can identify a minimal, predictive set of genes from thousands of candidates, generating a novel hypothesis for later validation."
        },
        "Optimizing a 12-parameter cell culture media": {
            "approach": "🏆 **Hybrid:** ML Model + Bayesian Optimization",
            "rationale": "A full factorial DOE is impossible (2^12 = 4096 runs). The hybrid approach is far more efficient. Run a small, space-filling DOE to gather initial data, train a Gaussian Process model to create a 'digital twin' of the culture, and then use Bayesian Optimization to intelligently navigate the parameter space *in silico* to find the optimum, which is then confirmed with a few targeted wet-lab experiments."
        },
        "Monitoring daily QC for a clinical diagnostic lab": {
            "approach": "🏆 **Hybrid:** Levey-Jennings + EWMA + Multivariate Control",
            "rationale": "This tiered approach provides comprehensive control. Use standard **Levey-Jennings charts** for regulatory compliance and simple rule interpretation ('what' is out). Augment this with more sensitive **EWMA or CUSUM charts** to detect slow reagent drift earlier ('when' it started). Finally, apply a **Hotelling's T² chart** on the full QC profile to catch subtle, correlated shifts that individual charts would miss ('how' it's failing)."
        },
        "Identifying sources of contamination in a clean room from microbiome data": {
            "approach": "🏆 **Bioinformatics & ML** (PCA, Clustering, Source Tracking)",
            "rationale": "These are high-dimensional, complex datasets unsuited for simple analysis. Unsupervised learning (PCA, UMAP) is required to visualize sample relationships. **Clustering** can group samples by microbial signature, identify outlier profiles, and trace them back to potential environmental or personnel sources using specialized algorithms like **FEAST or SourceTracker**."
        }
    }

# ==============================================================================
# 2. DYNAMIC HTML GENERATORS FOR MANIFESTO PAGE
# ==============================================================================

def render_workflow_step(
    phase_name: str,
    phase_class: str,
    classical_tools: List[str],
    ml_tools: List[str]
) -> str:
    """
    Generates a self-contained, stable HTML block for a DMAIC workflow step.
    """
    phase_color = {"step-define": COLORS['primary'], "step-measure": COLORS['secondary'], "step-analyze": COLORS['accent'], "step-improve": COLORS['neutral_yellow'], "step-control": COLORS['neutral_pink']}.get(phase_class, COLORS['dark_gray'])
    classical_list_html = "".join(f"<li>{tool}</li>" for tool in classical_tools)
    ml_list_html = "".join(f"<li>{tool}</li>" for tool in ml_tools)

    return f"""
    <div style="margin-bottom: 25px;">
        <h3 style="color: {phase_color}; border-bottom: 2px solid {phase_color}; padding-bottom: 5px; margin-bottom: 15px;">{phase_name}</h3>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            <div style="flex: 1; min-width: 300px; background-color: #f9f9f9; border: 1px solid {COLORS['light_gray']}; border-radius: 8px; padding: 15px;">
                <h5 style="margin-top: 0; color: {COLORS['primary']};">🏛️ Classical Tools (Rigor & Validation)</h5>
                <ul style="padding-left: 20px; margin: 0;">{classical_list_html}</ul>
            </div>
            <div style="flex: 1; min-width: 300px; background-color: #f9f9f9; border: 1px solid {COLORS['light_gray']}; border-radius: 8px; padding: 15px;">
                <h5 style="margin-top: 0; color: {COLORS['secondary']};">🤖 ML Augmentation (Scale & Discovery)</h5>
                <ul style="padding-left: 20px; margin: 0;">{ml_list_html}</ul>
            </div>
        </div>
    </div>
    """

# ==============================================================================
# 3. AI & ADVISOR CONTENT (NEW)
# ==============================================================================

def get_ai_summary_for_case(case: Dict[str, Any]) -> str:
    """Generates a concise, AI-style executive summary for a case study."""
    return f"""
    ### ✨ AI-Generated Executive Summary

    - **Problem:** {case['Problem Statement']}
    - **Key Root Cause(s) Identified:** {", ".join(case['Analyze Phase']['Root Causes'])}.
    - **Solution Implemented:** {case['Improve Phase']['Solutions']}
    - **Primary Tools for Success:** {", ".join(set(case['Analyze Phase']['Tools Used'] + case['Improve Phase']['Tools Used']))}.
    - **Outcome:** Achieved a financial impact of **${case['Project Outcomes']['Financial Impact']:,}** and {case['Project Outcomes']['Operational Impact'].lower()}
    """

def get_tool_advisor_data() -> Dict[str, Dict[str, Any]]:
    """
    Returns a structured dictionary containing deep-dive information for
    various statistical and quality tools.
    """
    return {
        "One-Way ANOVA": {
            "objective": "Compare Means",
            "what_is_it": "A statistical test used to determine whether there are any statistically significant differences between the means of three or more independent groups.",
            "when_to_use": [
                "Comparing the average yield from three different manufacturing lines.",
                "Testing if different suppliers provide raw material with the same average purity.",
                "Evaluating if various sterilization methods affect the average bioburden.",
            ],
            "assumptions": {
                "Data must be continuous": True,
                "Groups must be independent": True,
                "Data should be normally distributed (within each group)": True,
                "Variances should be equal across groups (homoscedasticity)": True,
            },
            "interpretation": {
                "statistic": "The **p-value** is the most critical output. A p-value less than your significance level (typically 0.05) indicates that you can reject the null hypothesis; it means at least one group mean is different from the others.",
                "visualization": "Use a **Box Plot** to visualize the differences. Look for non-overlapping 'notches' or medians at different levels, which visually support a significant finding.",
            },
            "pitfalls": "ANOVA tells you *that* a difference exists, but not *which* specific groups are different. After a significant ANOVA result, you must run a **post-hoc test** (like Tukey's HSD) to find out which pairs of groups are different.",
            "example_data_func": lambda: generate_anova_data(means=[102, 105, 98], stds=[2, 2.5, 2.2], n=30)
        },
        "Process Capability (Cpk)": {
            "objective": "Analyze Process Capability",
            "what_is_it": "A statistical measure that quantifies how well a process can produce output within its specification limits. It measures both the process spread and its centering relative to the specifications.",
            "when_to_use": [
                "Validating a process to ensure it can consistently meet customer requirements.",
                "Providing objective evidence of process performance for regulatory submissions (e.g., PPQ).",
                "Determining if a process improvement project has been successful.",
            ],
            "assumptions": {
                "Data must be continuous": True,
                "Data should be from a stable, in-control process": True,
                "Data should be approximately normally distributed": True,
            },
            "interpretation": {
                "statistic": "The **Cpk value** is the key metric. A Cpk of **1.33 or higher** is generally considered capable for most industries. A Cpk of **1.67 or higher** is a common goal for safety-critical processes.",
                "visualization": "The **Capability Histogram** shows the 'voice of the process' (your data's distribution) against the 'voice of the customer' (the specification limits - LSL/USL). This plot instantly reveals if your process is off-center or has too much variation.",
            },
            "pitfalls": "A Cpk value is meaningless if the process is not stable and in statistical control. Always confirm process stability with a control chart *before* running a capability analysis.",
            "example_data_func": None
        },
        "Design of Experiments (DOE)": {
            "objective": "Look for Relationships",
            "what_is_it": "A structured and efficient method for experimentation where multiple input factors are varied simultaneously to understand their individual and interactive effects on a process output.",
            "when_to_use": [
                "Optimizing a process with multiple parameters (e.g., finding the best temperature, pressure, and time).",
                "Identifying the 'vital few' factors that have the biggest impact on performance.",
                "Characterizing a design space for regulatory filings (QbD).",
            ],
            "assumptions": {
                "Factors can be controlled": True,
                "Response can be measured quantitatively": True,
            },
            "interpretation": {
                "statistic": "The **Main Effects Plot** shows the impact of each individual factor. The **Interaction Plot** is crucial and reveals if the effect of one factor depends on the level of another—an insight impossible with one-factor-at-a-time testing.",
                "visualization": "A **Pareto Plot of Effects** ranks the factors and interactions by importance. A **Cube Plot** or **Contour Plot** helps visualize the relationship between inputs and the output.",
            },
            "pitfalls": "Failing to include center points in the experiment, which prevents you from detecting curvature in the process response. Also, failing to randomize the experimental runs can lead to incorrect conclusions due to time-based effects.",
            "example_data_func": None
        }
    }
