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
        
        **An interactive, expert-level playbook for compliant and accelerated development of drugs, devices, and diagnostics.**

        This application demonstrates a unified framework that fuses the **regulatory-mandated rigor of classical Six Sigma** 
        with the **predictive and discovery power of modern Machine Learning**.
        
        Navigate through the **DMAIC (Define, Measure, Analyze, Improve, Control)** process improvement lifecycle. Each phase has been substantially enriched with a comprehensive suite of tools and examples relevant to the key regulatory milestones in the Bio/Pharma/Medtech industries, providing a complete roadmap for the modern, compliant, and competitive organization.
        
        **Version:** 17.0 (Definitive All-Inclusive DMAIC Integration)
        """
    }
)

# Apply the custom CSS theme for a consistent, polished look.
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ==============================================================================
# 2. APPLICATION NAVIGATION (Final, Verified DMAIC Structure)
# ==============================================================================
# The navigation is structured around the DMAIC framework, with each page
# now containing a comprehensive set of tools relevant to that phase.
PAGES = {
    "Framework Overview": [
        st.Page(show_welcome_page, title="Welcome & Introduction", icon="🏠"),
    ],
    "The DMAIC Lifecycle": [
        st.Page(show_define_phase, title="Define: Project Mandate & Design", icon="🌀"),
        st.Page(show_measure_phase, title="Measure: Baseline & Validation", icon="🔬"),
        st.Page(show_analyze_phase, title="Analyze: Root Cause & Failure Modes", icon="📈"),
        st.Page(show_improve_phase, title="Improve: Optimization & Robustness", icon="⚙️"),
        st.Page(show_control_phase, title="Control: Monitoring & Surveillance", icon="📡"),
    ],
    "Strategic & Regulatory Synthesis": [
        st.Page(show_hybrid_manifesto, title="The Hybrid Manifesto & GxP", icon="🤝")
    ]
}

# --- Sidebar Rendering ---
st.sidebar.title("🧬 Bio-AI Framework")
st.sidebar.markdown("##### DMAIC Development Playbook")
st.sidebar.markdown("Navigate the lifecycle below.")

# Create the navigation menu from the dictionary of pages
pg = st.navigation(PAGES)

# --- Sidebar Footer ---
st.sidebar.divider()
st.sidebar.info(
    "This app demonstrates a framework for integrating Machine Learning into the "
    "biotech lifecycle to achieve superior performance while maintaining full regulatory compliance."
)
st.sidebar.markdown(
    "**[View Source on GitHub](https://github.com/your-repo/your-project)**" # Placeholder
)


# ==============================================================================
# 3. PAGE RENDERING LOGIC
# ==============================================================================
# The st.navigation object handles the rendering of the selected page.
# This line executes the function associated with the current page.
pg.run()
