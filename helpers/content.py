"""
helpers/content.py

This module centralizes all static text, markdown, and HTML content used
throughout the application.

By separating content from the application logic and layout (in app_pages.py),
we make the app easier to maintain, update, and potentially translate. This
module provides functions that return formatted strings or data structures
for rendering in the UI.

Author: AI Engineering SME
Version: 23.1 (Commercial Grade Refactor)
Date: 2023-10-26
"""

from typing import List, Dict
from .styling import COLORS # Import color palette for consistent styling

# ==============================================================================
# 1. MARKDOWN CONTENT FOR EXPANDERS
# ==============================================================================
# Storing long markdown blocks here keeps the page-rendering scripts clean.

def get_qfd_expander_content() -> str:
    """Returns the markdown content for the QFD expander."""
    return """
    **Methodology:** Quality Function Deployment (QFD) is a structured method for
    translating customer requirements (the "Voice of the Customer") into design
    specifications (the "Voice of the Engineer"). The 'House of Quality' matrix
    is the primary tool used to map customer needs to technical characteristics,
    creating a traceable and justifiable link.

    **Regulatory Significance:** This creates a **documented, traceable link**
    between user needs and design inputs, which is a core requirement of
    **FDA Design Controls (21 CFR 820.30)**. It provides objective evidence
    for *why* certain technical specifications were prioritized for the
    **Design History File (DHF)**, demonstrating a systematic approach to design.
    """

def get_kano_expander_content() -> str:
    """Returns the markdown content for the Kano Model expander."""
    return """
    **Methodology:** The Kano model is a framework for categorizing product or
    service features based on how they impact customer satisfaction. It helps
    prioritize development efforts by distinguishing between different types
    of customer needs.
    - **Basic (Must-be):** Features that are taken for granted. Their absence
      causes dissatisfaction, but their presence doesn't increase satisfaction
      (e.g., a car having brakes). In diagnostics, this could be detecting a
      well-known, common mutation.
    - **Performance:** Features for which "more is better." Satisfaction is
      proportional to how well they are implemented (e.g., a car's fuel
      efficiency). In diagnostics, this could be assay sensitivity (lower LoD).
    - **Excitement (Delighter):** Unexpected features that create high
      satisfaction when present, but whose absence does not cause
      dissatisfaction (e.g., the first time a car had a built-in GPS). In
      diagnostics, this could be identifying a novel, clinically actionable biomarker.

    **Interpretation:** This model provides a strategic framework to justify
    which features are critical for the product's intended use and which are
    secondary. This helps focus Verification & Validation (V&V) efforts on what
    matters most to the end-user (e.g., the clinician), a key principle of
    user-centric design that is highly regarded by regulators.
    """

# ... This pattern would be repeated for every expander in the application.
# For brevity, I will show a few more key examples.

def get_msa_expander_content() -> str:
    """Returns markdown content for the Measurement System Analysis expander."""
    return """
    **Methodology:** A Gage R&R (Repeatability & Reproducibility) study is a
    designed experiment to quantify the amount of variation in a measurement
    system itself. It partitions the total observed variance into its components:
    - **Part-to-Part Variation:** The true biological variation we want to measure.
    - **Repeatability (Equipment Variation):** Variation from repeated measurements
      by the same operator with the same equipment.
    - **Reproducibility (Appraiser Variation):** Variation between different
      operators using the same equipment.

    **Regulatory Significance:** You cannot trust your process data until you trust
    your measurement system. Before any process characterization (ICH Q11) or
    validation (Process Performance Qualification - PPQ), the analytical methods
    used must themselves be validated. A Gage R&R study provides the standard
    evidence that the measurement system is reliable and its contribution to
    total variation is acceptable (typically <10% of total process tolerance,
    or %-Contribution < 30%). This is a non-negotiable prerequisite for all
    subsequent stages and a key part of Analytical Validation.
    """

def get_pccp_expander_content() -> str:
    """Returns markdown content for the PCCP expander."""
    return """
    **Methodology:** A **Pre-determined Change Control Plan (PCCP)** is a
    comprehensive "blueprint" submitted to regulators that prospectively defines
    how a learning AI/ML model will be monitored, updated, and re-validated
    after deployment. It includes:
    - **Performance Monitoring:** The specific metrics that will be tracked (e.g., AUC, F1-score).
    - **Retraining Triggers:** The objective criteria that will trigger a model update
      (e.g., performance drops below 0.90 AUC for 3 consecutive days).
    - **Modification Protocol:** The specific, pre-defined changes that can be made
      (e.g., re-training on new data, but not changing the model architecture).
    - **Re-validation Protocol:** The plan for validating the updated model before it
      is deployed.

    **Regulatory Significance:** This is a cornerstone of **Good Machine Learning
    Practice (GMLP)** and is essential for any AI/ML-based Software as a
    Medical Device (SaMD) that is expected to learn over time. It demonstrates
    a controlled, auditable, and safe process for managing a "locked" but
    updatable algorithm, satisfying the FDA's need for safety and effectiveness
    while allowing for state-of-the-art model maintenance.
    """

# ==============================================================================
# 2. UI DATA STRUCTURES
# ==============================================================================

def get_guidance_data() -> Dict[str, Dict[str, str]]:
    """
    Returns a dictionary of scenarios and recommended approaches for the
    interactive solution recommender widget.

    :return: A dictionary where keys are scenarios and values are dicts
             containing the recommended approach and rationale.
    """
    return {
        "Validating an assay for FDA 510(k) submission": {
            "approach": "🏆 **Classical Stats** (DOE, LoD/LoB studies, Gage R&R)",
            "rationale": "Methods are traceable, validated, and follow CLSI/FDA guidelines, which is paramount for regulatory bodies. The focus is on rigorous inference and establishing performance characteristics beyond reproach. Their outputs (p-values, Cpk, confidence intervals) are the accepted currency of regulatory submissions."
        },
        "Discovering a new gene signature from RNA-Seq data": {
            "approach": "🏆 **Machine Learning** (Elastic Net, Random Forest with SHAP)",
            "rationale": "ML excels at feature selection from high-dimensional data where the number of features vastly exceeds the number of samples (p >> n). It can identify a minimal, predictive set of genes from thousands of candidates, a task that is statistically challenging or impossible for classical regression."
        },
        "Optimizing a 12-parameter cell culture media": {
            "approach": "🏆 **Hybrid:** ML Model + Bayesian Optimization",
            "rationale": "A full factorial DOE is impossible (2^12 = 4096 runs). Instead, run a small space-filling DOE (e.g., Latin Hypercube) to train a Gaussian Process model (the 'digital twin' of the culture). Then, use Bayesian Optimization to intelligently navigate the parameter space and find the optimal media composition *in silico* before final wet-lab confirmation."
        },
        "Monitoring daily QC for a clinical diagnostic lab": {
            "approach": "🏆 **Hybrid:** Levey-Jennings + EWMA + Multivariate Control",
            "rationale": "Use standard Levey-Jennings charts for regulatory compliance and ease of interpretation (the 'what'). Use more sensitive EWMA or CUSUM charts to detect slow reagent drift earlier (the 'when'). Use a Hotelling's T² chart on the full QC profile to catch subtle, correlated shifts that individual charts would miss (the 'how')."
        },
        "Identifying sources of contamination in a clean room from microbiome data": {
            "approach": "🏆 **Bioinformatics & ML** (PCA, Clustering, Source Tracking)",
            "rationale": "These are high-dimensional, complex datasets. Unsupervised learning methods are required to cluster samples by microbial signature, identify outlier profiles, and trace them back to potential environmental or personnel sources using algorithms like FEAST or SourceTracker."
        }
    }

# ==============================================================================
# 3. DYNAMIC HTML & CSS GENERATORS
# ==============================================================================

def get_workflow_css() -> str:
    """Generates CSS for the Hybrid Manifesto workflow diagram."""
    return f"""
    <style>
        .workflow-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
        }}
        .workflow-step {{
            background-color: #FFFFFF;
            border: 1px solid {COLORS['light_gray']};
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            width: 95%;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 5px solid;
            transition: all 0.2s ease-in-out;
        }}
        .workflow-step:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }}
        .workflow-arrow {{
            font-size: 28px;
            color: {COLORS['dark_gray']};
            margin-bottom: 20px;
            font-weight: bold;
        }}
        .step-define    {{ border-left-color: {COLORS['primary']}; }}
        .step-measure   {{ border-left-color: {COLORS['secondary']}; }}
        .step-analyze   {{ border-left-color: {COLORS['accent']}; }}
        .step-improve   {{ border-left-color: {COLORS['neutral_yellow']}; }}
        .step-control   {{ border-left-color: {COLORS['neutral_pink']}; }}

        .workflow-step h4 {{
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 1.5em;
            color: #333333;
        }}
        .workflow-step .tool-col {{ padding: 0 15px; }}
        .workflow-step .tool-col h5 {{
            color: #555555;
            border-bottom: 2px solid #EEEEEE;
            padding-bottom: 5px;
            margin-bottom: 10px;
        }}
        .workflow-step .tool-col ul {{
            padding-left: 20px;
            margin: 0;
        }}
        .workflow-step .tool-col li {{ margin-bottom: 5px; }}
        .tool-col-classical h5 {{ color: {COLORS['primary']}; }}
        .tool-col-ml h5 {{ color: {COLORS['secondary']}; }}
    </style>
    """

def render_workflow_step(phase_name: str, phase_class: str, classical_tools: List[str], ml_tools: List[str]) -> str:
    """
    Generates an HTML block for a single step in the DMAIC workflow diagram.

    :param phase_name: The name of the DMAIC phase (e.g., "Define: ...").
    :param phase_class: The CSS class for the phase color (e.g., "step-define").
    :param classical_tools: A list of classical tools for this phase.
    :param ml_tools: A list of ML/AI tools for this phase.
    :return: An HTML string representing the workflow step.
    """
    classical_list = "".join([f"<li>{tool}</li>" for tool in classical_tools])
    ml_list = "".join([f"<li>{tool}</li>" for tool in ml_tools])

    return f"""
    <div class="workflow-step {phase_class}">
        <h4>{phase_name}</h4>
        <div style="display: flex; justify-content: space-between;">
            <div style="flex: 1; margin-right: 10px;" class="tool-col tool-col-classical">
                <h5>Classical Tools (Rigor & Validation)</h5>
                <ul>{classical_list}</ul>
            </div>
            <div style="flex: 1; margin-left: 10px;" class="tool-col tool-col-ml">
                <h5>ML/Bio-AI Augmentation (Scale & Discovery)</h5>
                <ul>{ml_list}</ul>
            </div>
        </div>
    </div>
    """
