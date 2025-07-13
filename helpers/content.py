"""
helpers/content.py

This module centralizes all static text, markdown, and HTML content used
throughout the application.

Author: Bio-AI Excellence SME Collective
Version: 31.0 (Content-Rich Gold Build)
Date: 2025-07-15

Changelog from v30.1:
- [REFACTOR] Removed obsolete `get_*_expander_content_pro` functions.
  All detailed explanations have been moved directly into `app_pages.py` to
  be co-located with their respective visualizations, improving maintainability.
"""

from typing import List, Dict
from .styling import COLORS

# ==============================================================================
# 1. UI DATA STRUCTURES
# ==============================================================================

def get_guidance_data() -> Dict[str, Dict[str, str]]:
    """
    Returns a dictionary of scenarios and recommended approaches for the
    interactive solution recommender widget.
    """
    return {
        "Validating an assay for FDA 510(k) submission": {
            "approach": "🏆 **Classical Stats** (DOE, LoD/LoB studies, Gage R&R)",
            "rationale": "Regulatory bodies demand traceable, validated methods that follow established guidelines (e.g., CLSI, FDA). The outputs of classical statistics—p-values, Cpk, confidence intervals—are the accepted currency for demonstrating safety and effectiveness. This approach focuses on **rigorous inference** and establishing performance characteristics beyond reproach."
        },
        "Discovering a new gene signature from RNA-Seq data": {
            "approach": "🏆 **Machine Learning** (Elastic Net, Random Forest with SHAP)",
            "rationale": "ML excels at feature selection from **high-dimensional data** where the number of features vastly exceeds samples (p >> n). It can identify a minimal, predictive set of genes from thousands of candidates, a task where classical regression struggles due to multicollinearity and overfitting."
        },
        "Optimizing a 12-parameter cell culture media": {
            "approach": "🏆 **Hybrid:** ML Model + Bayesian Optimization",
            "rationale": "A full factorial DOE is impossible (2^12 = 4096 runs). The hybrid approach is far more efficient. First, run a small, **space-filling DOE** (e.g., Latin Hypercube) to gather initial data. Then, train a Gaussian Process model to create a 'digital twin' of the culture. Finally, use **Bayesian Optimization** to intelligently navigate the parameter space *in silico* to find the optimum, which is then confirmed with a few targeted wet-lab experiments."
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
