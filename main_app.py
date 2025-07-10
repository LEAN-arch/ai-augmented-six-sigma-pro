# main_app.py

import streamlit as st

# Import helper functions and all page definitions from their respective modules.
# This architecture keeps the main app file clean and focused on navigation.
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
# Configured for a professional, wide-screen experience suitable for complex scientific data.
st.set_page_config(
    page_title="Genomic Assay Development & Optimization Framework",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/your-project', # Placeholder
        'Report a bug': "https://github.com/your-repo/your-project/issues", # Placeholder
        'About': """
        ## 🧬 The Bio-AI Excellence Framework
        
        **An interactive playbook for developing and optimizing robust genomic assays and devices.**

        This application demonstrates a unified framework that fuses the statistical rigor of
        classical **Design of Experiments (DOE)** with the predictive power of **Machine Learning and Bioinformatics**.
        
        Navigate through the R&D lifecycle (framed as DMAIC) to see classical methods presented 
        alongside their AI-augmented counterparts for the modern biotech lab.
        
        **Version:** 5.0 (Biotech/Genomics Professional Release)
        """
    }
)

# Apply the custom CSS theme for a consistent, polished look.
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ==============================================================================
# 2. APPLICATION NAVIGATION & STATE MANAGEMENT
# ==============================================================================

# The PAGES dictionary maps user-friendly names to their rendering functions.
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

# --- Sidebar Rendering ---
st.sidebar.title("🧬 Bio-AI Framework")
st.sidebar.markdown("### Assay Development Playbook")
st.sidebar.markdown("Navigate the R&D lifecycle below.")

# Initialize session state for navigation.
if 'current_page' not in st.session_state:
    st.session_state.current_page = list(PAGES.keys())[0]

# --- Custom Navigation Menu ---
for page_name, page_function in PAGES.items():
    if page_name == "---":
        st.sidebar.divider()
    else:
        is_active = (st.session_state.current_page == page_name)
        button_type = "primary" if is_active else "secondary"
        if st.sidebar.button(page_name, use_container_width=True, type=button_type):
            st.session_state.current_page = page_name
            st.rerun()

# --- Sidebar Footer ---
st.sidebar.divider()
st.sidebar.info(
    "This app demonstrates a framework for integrating Machine Learning into the "
    "biotech R&D lifecycle to achieve superior assay performance and reliability."
)
st.sidebar.markdown(
    "**[View Source on GitHub](https://github.com/your-repo/your-project)**" # Placeholder
)


# ==============================================================================
# 3. PAGE RENDERING LOGIC
# ==============================================================================
# Retrieve and execute the function for the currently selected page.
page_to_render = PAGES[st.session_state.current_page]
if page_to_render:
    page_to_render()
