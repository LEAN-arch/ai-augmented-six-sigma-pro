"""
helpers/content.py

This module centralizes all static text, markdown, and HTML content used
throughout the application.

By separating content from the application logic (in app_pages.py), we make
the app easier to maintain, update, and potentially translate.

Author: AI Engineering SME
Version: 25.1 (Commercial Grade Build)
Date: 2025-07-12

Changelog from v24.1:
- [CRITICAL-CONTENT] Implemented a robust `render_workflow_step` function. This
  function now generates self-contained, modern HTML with inline CSS (Flexbox)
  to create the stable and visually appealing workflow diagram for the enhanced
  "Hybrid Manifesto" page.
- [ENHANCEMENT] Polished all markdown content within the `get_*_expander_content`
  functions to improve readability, using consistent headers and formatting for
  a professional user experience.
- [ENHANCEMENT] Improved the "guidance_data" content with clearer, more
  compelling rationales for each recommended approach.
- [MAINTAINABILITY] The HTML generation in `render_workflow_step` is structured
  within a single f-string, making it easy to modify the layout and styles in one
  place.
- [STYLE] Added type hints to all function signatures for improved code quality
  and static analysis.
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
**Methodology:**
Quality Function Deployment (QFD) is a structured method for translating customer requirements (the "Voice of the Customer") into design specifications (the "Voice of the Engineer"). The 'House of Quality' matrix is the primary tool used to map customer needs to technical characteristics, creating a traceable and justifiable link.

**Regulatory Significance:**
This creates a **documented, traceable link** between user needs and design inputs, which is a core requirement of **FDA Design Controls (21 CFR 820.30)**. It provides objective evidence for *why* certain technical specifications were prioritized for the **Design History File (DHF)**, demonstrating a systematic approach to design.
"""

def get_kano_expander_content() -> str:
    """Returns the markdown content for the Kano Model expander."""
    return """
**Methodology:**
The Kano model is a framework for categorizing product or service features based on how they impact customer satisfaction. It helps prioritize development efforts by distinguishing between different types of customer needs:
- **Basic (Must-be):** Features that are taken for granted. Their absence causes dissatisfaction, but their presence doesn't increase satisfaction (e.g., a car having brakes). In diagnostics, this could be detecting a well-known, common mutation.
- **Performance:** Features for which "more is better." Satisfaction is proportional to how well they are implemented (e.g., a car's fuel efficiency). In diagnostics, this could be assay sensitivity (lower LoD).
- **Excitement (Delighter):** Unexpected features that create high satisfaction when present, but whose absence does not cause dissatisfaction (e.g., the first time a car had a built-in GPS). In diagnostics, this could be identifying a novel, clinically actionable biomarker.

**Interpretation:**
This model provides a strategic framework to justify which features are critical for the product's intended use and which are secondary. This helps focus Verification & Validation (V&V) efforts on what matters most to the end-user (e.g., the clinician), a key principle of user-centric design that is highly regarded by regulators.
"""

def get_msa_expander_content() -> str:
    """Returns markdown content for the Measurement System Analysis expander."""
    return """
**Methodology:**
A Gage R&R (Repeatability & Reproducibility) study is a designed experiment to quantify the amount of variation in a measurement system itself. It partitions the total observed variance into its components:
- **Part-to-Part Variation:** The true biological variation we want to measure.
- **Repeatability (Equipment Variation):** Variation from repeated measurements by the same operator with the same equipment.
- **Reproducibility (Appraiser Variation):** Variation between different operators using the same equipment.

**Regulatory Significance:**
You cannot trust your process data until you trust your measurement system. Before any process characterization (ICH Q11) or validation (Process Performance Qualification - PPQ), the analytical methods used must themselves be validated. A Gage R&R study provides the standard evidence that the measurement system is reliable and its contribution to total variation is acceptable (typically <10% of total process tolerance, or %-Contribution < 30%). This is a non-negotiable prerequisite for all subsequent stages and a key part of Analytical Validation.
"""

def get_pccp_expander_content() -> str:
    """Returns markdown content for the PCCP expander."""
    return """
**Methodology:**
A **Pre-determined Change Control Plan (PCCP)** is a comprehensive "blueprint" submitted to regulators that prospectively defines how a learning AI/ML model will be monitored, updated, and re-validated after deployment. It includes:
- **Performance Monitoring:** The specific metrics that will be tracked (e.g., AUC, F1-score).
- **Retraining Triggers:** The objective criteria that will trigger a model update (e.g., performance drops below 0.90 AUC for 3 consecutive days).
- **Modification Protocol:** The specific, pre-defined changes that can be made (e.g., re-training on new data, but not changing the model architecture).
- **Re-validation Protocol:** The plan for validating the updated model before it is deployed.

**Regulatory Significance:**
This is a cornerstone of **Good Machine Learning Practice (GMLP)** and is essential for any AI/ML-based Software as a Medical Device (SaMD) that is expected to learn over time. It demonstrates a controlled, auditable, and safe process for managing a "locked" but updatable algorithm, satisfying the FDA's need for safety and effectiveness while allowing for state-of-the-art model maintenance.
"""


# ==============================================================================
# 2. UI DATA STRUCTURES
# ==============================================================================

def get_guidance_data() -> Dict[str, Dict[str, str]]:
    """
    Returns a dictionary of scenarios and recommended approaches for the
    interactive solution recommender widget.

    Returns:
        A dictionary where keys are scenarios and values are dicts
        containing the recommended approach and rationale.
    """
    return {
        "Validating an assay for FDA 510(k) submission": {
            "approach": "🏆 **Classical Stats** (DOE, LoD/LoB studies, Gage R&R)",
            "rationale": "Regulatory bodies demand traceable, validated methods that follow established guidelines (e.g., CLSI, FDA). The outputs of classical statistics—p-values, Cpk, confidence intervals—are the accepted currency for demonstrating safety and effectiveness. This approach focuses on rigorous inference and establishing performance characteristics beyond reproach."
        },
        "Discovering a new gene signature from RNA-Seq data": {
            "approach": "🏆 **Machine Learning** (Elastic Net, Random Forest with SHAP)",
            "rationale": "ML excels at feature selection from high-dimensional data where the number of features vastly exceeds samples (p >> n). It can identify a minimal, predictive set of genes from thousands of candidates, a task where classical regression struggles due to multicollinearity and overfitting."
        },
        "Optimizing a 12-parameter cell culture media": {
            "approach": "🏆 **Hybrid:** ML Model + Bayesian Optimization",
            "rationale": "A full factorial DOE is impossible (2^12 = 4096 runs). The hybrid approach is far more efficient. First, run a small, space-filling DOE (e.g., Latin Hypercube) to gather initial data. Then, train a Gaussian Process model to create a 'digital twin' of the culture. Finally, use Bayesian Optimization to intelligently navigate the parameter space *in silico* to find the optimum, which is then confirmed with a few targeted wet-lab experiments."
        },
        "Monitoring daily QC for a clinical diagnostic lab": {
            "approach": "🏆 **Hybrid:** Levey-Jennings + EWMA + Multivariate Control",
            "rationale": "This tiered approach provides comprehensive control. Use standard Levey-Jennings charts for regulatory compliance and simple rule interpretation ('what' is out). Augment this with more sensitive EWMA or CUSUM charts to detect slow reagent drift earlier ('when' it started). Finally, apply a Hotelling's T² chart on the full QC profile to catch subtle, correlated shifts that individual charts would miss ('how' it's failing)."
        },
        "Identifying sources of contamination in a clean room from microbiome data": {
            "approach": "🏆 **Bioinformatics & ML** (PCA, Clustering, Source Tracking)",
            "rationale": "These are high-dimensional, complex datasets unsuited for simple analysis. Unsupervised learning (PCA, UMAP) is required to visualize sample relationships. Clustering can group samples by microbial signature, identify outlier profiles, and trace them back to potential environmental or personnel sources using specialized algorithms like FEAST or SourceTracker."
        }
    }


# ==============================================================================
# 3. DYNAMIC HTML GENERATORS FOR MANIFESTO PAGE
# ==============================================================================

def render_workflow_step(
    phase_name: str,
    phase_class: str,
    classical_tools: List[str],
    ml_tools: List[str]
) -> str:
    """
    Generates a self-contained, stable HTML block for a DMAIC workflow step.
    This function uses inline CSS with Flexbox to ensure perfect layout
    containment and visual stability.

    Args:
        phase_name: The name of the DMAIC phase (e.g., "1. Define").
        phase_class: The CSS class identifier for the phase color.
        classical_tools: A list of classical tools for this phase.
        ml_tools: A list of ML/AI tools for this phase.

    Returns:
        An HTML string representing the styled workflow step.
    """
    phase_color = {
        "step-define": COLORS['primary'],
        "step-measure": COLORS['secondary'],
        "step-analyze": COLORS['accent'],
        "step-improve": COLORS['neutral_yellow'],
        "step-control": COLORS['neutral_pink']
    }.get(phase_class, COLORS['dark_gray'])

    # Generate bulleted lists for each toolset
    classical_list_html = "".join(f"<li>{tool}</li>" for tool in classical_tools)
    ml_list_html = "".join(f"<li>{tool}</li>" for tool in ml_tools)

    # Return a single, formatted HTML string. Using inline styles in this manner
    # makes the component self-contained and avoids CSS conflicts.
    return f"""
    <div style="margin-bottom: 25px;">
        <!-- Phase Header with Color Bar -->
        <h3 style="color: {phase_color}; border-bottom: 2px solid {phase_color}; padding-bottom: 5px; margin-bottom: 15px;">
            {phase_name}
        </h3>
        <!-- Flexbox container for side-by-side columns -->
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            <!-- Classical Tools Column -->
            <div style="flex: 1; min-width: 300px; background-color: #f9f9f9; border: 1px solid {COLORS['light_gray']}; border-radius: 8px; padding: 15px;">
                <h5 style="margin-top: 0; color: {COLORS['primary']};">🏛️ Classical Tools (Rigor & Validation)</h5>
                <ul style="padding-left: 20px; margin: 0;">{classical_list_html}</ul>
            </div>
            <!-- ML Augmentation Column -->
            <div style="flex: 1; min-width: 300px; background-color: #f9f9f9; border: 1px solid {COLORS['light_gray']}; border-radius: 8px; padding: 15px;">
                <h5 style="margin-top: 0; color: {COLORS['secondary']};">🤖 ML Augmentation (Scale & Discovery)</h5>
                <ul style="padding-left: 20px; margin: 0;">{ml_list_html}</ul>
            </div>
        </div>
    </div>
    """
