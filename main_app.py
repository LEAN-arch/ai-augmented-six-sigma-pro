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
# This must be the first Streamlit command in the script.
st.set_page_config(
    page_title="Bio-AI Excellence Framework",
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
        
        **Version:** 6.0 (Modern Navigation Release)
        """
    }
)

# Apply the custom CSS theme for a consistent, polished look.
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ==============================================================================
# 2. APPLICATION NAVIGATION (Modern Streamlit Approach)
# ==============================================================================
# UX/DX Enhancement: Replaced the manual button-based navigation with st.navigation.
# This provides a more robust, stateful, and idiomatic multi-page app experience
# with features like automatic URL routing for each page.

# Define the pages of the application
PAGES = [
    st.Page(show_welcome_page, title="🏠 Welcome & Framework", icon="🏠"),
    st.Page(show_define_phase, title="🌀 Define: Clinical Need", icon="🌀"),
    st.Page(show_measure_phase, title="🔬 Measure: System Validation", icon="🔬"),
    st.Page(show_analyze_phase, title="📈 Analyze: Root Cause", icon="📈"),
    st.Page(show_improve_phase, title="⚙️ Improve: Optimization", icon="⚙️"),
    st.Page(show_control_phase, title="📡 Control: Lab Operations & QC", icon="📡"),
    st.Page(show_comparison_matrix, title="⚔️ Methodology Comparison", icon="⚔️"),
    st.Page(show_hybrid_strategy, title="🤝 The Hybrid Lab Manifesto", icon="🤝")
]

# --- Sidebar Rendering ---
st.sidebar.title("🧬 Bio-AI Framework")
st.sidebar.markdown("##### Assay Development Playbook")
st.sidebar.markdown("Navigate the R&D lifecycle below.")

# Create the navigation menu
pg = st.navigation(PAGES)

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
# The st.navigation object handles the rendering of the selected page.
# The following line executes the function associated with the current page.
pg.run()
