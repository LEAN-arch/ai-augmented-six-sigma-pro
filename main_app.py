# main_app.py

import streamlit as st

# Import helper functions and all page definitions from their respective modules.
from app_helpers import get_custom_css
from app_pages import (
    show_welcome_page,
    show_define_phase,
    show_measure_phase,
    show_analyze_phase,
    show_improve_phase,
    show_control_phase,
    show_comparison_matrix,
    show_hybrid_strategy
)

# ==============================================================================
# 1. GLOBAL PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Genomic Assay Development & Optimization Framework",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/your-project',
        'Report a bug': "https://github.com/your-repo/your-project/issues",
        'About': """
        ## 🧬 The Bio-AI Excellence Framework
        **An interactive playbook for developing and optimizing robust genomic assays and devices.**
        This application demonstrates a unified framework that fuses the statistical rigor of
        classical **Design of Experiments (DOE)** with the predictive power of **Machine Learning and Bioinformatics**.
        """
    }
)

st.markdown(get_custom_css(), unsafe_allow_html=True)

# ==============================================================================
# 2. APPLICATION NAVIGATION & STATE MANAGEMENT
# ==============================================================================
PAGES = {
    "🏠 Welcome & Framework": show_welcome_page,
    "---": None,
    "🌀 Define: Clinical Need & Assay Goals": show_define_phase,
    "🔬 Measure: Assay & System Validation": show_measure_phase,
    "📈 Analyze: Root Cause of Assay Variability": show_analyze_phase,
    "⚙️ Improve: Assay & Workflow Optimization": show_improve_phase,
    "📡 Control: Lab Operations & QC": show_control_phase,
    "---": None,
    "⚔️ Methodology Comparison": show_comparison_matrix,
    "🤝 The Hybrid Lab Manifesto": show_hybrid_strategy
}

st.sidebar.title("🧬 Bio-AI Framework")
st.sidebar.markdown("### Assay Development Playbook")
st.sidebar.markdown("Navigate the R&D lifecycle below.")

# --- FIX APPLIED for KeyError: More robust page selection ---
# If the page in session state is no longer valid, default to the first page.
if 'current_page' not in st.session_state or st.session_state.current_page not in PAGES:
    st.session_state.current_page = list(PAGES.keys())[0]

for page_name, page_function in PAGES.items():
    if page_name == "---":
        st.sidebar.divider()
    else:
        is_active = (st.session_state.current_page == page_name)
        button_type = "primary" if is_active else "secondary"
        if st.sidebar.button(page_name, use_container_width=True, type=button_type):
            st.session_state.current_page = page_name
            st.rerun()

st.sidebar.divider()
st.sidebar.info(
    "This app demonstrates a framework for integrating Machine Learning into the "
    "biotech R&D lifecycle to achieve superior assay performance and reliability."
)
st.sidebar.markdown(
    "**[View Source on GitHub](https://github.com/your-repo/your-project)**"
)

# ==============================================================================
# 3. PAGE RENDERING LOGIC
# ==============================================================================
page_to_render = PAGES[st.session_state.current_page]
if page_to_render:
    page_to_render()
