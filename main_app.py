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
    show_hybrid_manifesto
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
        classical **Design of Experiments (DOE) and Six Sigma** with the predictive power of **Machine Learning and Bioinformatics**.
        
        Navigate through the R&D lifecycle (framed as DMAIC) to see foundational methods presented 
        alongside their AI-augmented and regulatory-compliant counterparts for the modern biotech lab. This version integrates both the original educational examples and the advanced, expert-level QMS/regulatory tools.
        
        **Version:** 21.0 (Definitive, Unabridged, Error-Free Release)
        """
    }
)

# Apply the custom CSS theme for a consistent, polished look.
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ==============================================================================
# 2. APPLICATION NAVIGATION (Modern Streamlit Approach)
# ==============================================================================
# Uses st.navigation for robust, stateful, and idiomatic multi-page app experience.
# The page definitions are clean, with icons handled correctly by the `icon` parameter.
PAGES = [
    st.Page(show_welcome_page, title="Welcome & Framework", icon="🏠"),
    st.Page(show_define_phase, title="Define: Clinical Need & Design", icon="🌀"),
    st.Page(show_measure_phase, title="Measure: System Validation", icon="🔬"),
    st.Page(show_analyze_phase, title="Analyze: Root Cause & Failure", icon="📈"),
    st.Page(show_improve_phase, title="Improve: Optimization & Robustness", icon="⚙️"),
    st.Page(show_control_phase, title="Control: Lab Operations & PMS", icon="📡"),
    st.Page(show_comparison_matrix, title="Methodology Comparison", icon="⚔️"),
    st.Page(show_hybrid_manifesto, title="The Hybrid Manifesto & GxP", icon="🤝")
]

# --- Sidebar Rendering ---
st.sidebar.title("🧬 Bio-AI Framework")
st.sidebar.markdown("##### Assay Development Playbook")
st.sidebar.markdown("Navigate the R&D lifecycle below.")

# Create the navigation menu from the list of pages
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
