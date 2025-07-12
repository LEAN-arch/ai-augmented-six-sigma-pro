# main_app.py

import streamlit as st

# Import helper functions and all page definitions from their respective modules.
# This architecture keeps the main app file clean and focused on navigation.
from app_helpers import get_custom_css
from app_pages import (
    show_framework_overview,
    show_design_and_development,
    show_process_characterization,
    show_process_validation_ppq,
    show_qrm,
    show_capa_rca,
    show_cpv_and_monitoring,
    show_post_market_surveillance,
    show_digital_health_samd,
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
        
        **An interactive playbook for compliant and accelerated development of drugs, devices, and diagnostics.**

        This application demonstrates a unified framework that fuses the **regulatory-mandated rigor of classical statistics and Six Sigma** 
        with the **predictive and discovery power of modern Machine Learning**.
        
        Navigate through the GxP product lifecycle to see classical quality tools presented alongside their AI-augmented counterparts, 
        providing a roadmap for the modern, compliant, and competitive biotech organization.
        
        This tool is designed for a technically proficient audience (e.g., R&D Scientists, Process Engineers, Bioinformaticians, Quality & Regulatory Affairs Professionals).
        
        **Version:** 15.0 (SME Final Integration & Lifecycle Overhaul)
        """
    }
)

# Apply the custom CSS theme for a consistent, polished look.
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ==============================================================================
# 2. APPLICATION NAVIGATION (Final Regulatory Lifecycle Structure)
# ==============================================================================
# The navigation is organized by regulatory stage, providing an intuitive journey
# for professionals in the Bio / Pharma / Medtech space.
PAGES = {
    "Framework Overview": [
        st.Page(show_framework_overview, title="The Integrated Framework", icon="🏠"),
    ],
    "The GxP Product Lifecycle": [
        st.Page(show_design_and_development, title="1. Design & Development", icon="📐"),
        st.Page(show_process_characterization, title="2. Process Characterization", icon="🧪"),
        st.Page(show_process_validation_ppq, title="3. Process Validation (PPQ)", icon="✔️"),
        st.Page(show_cpv_and_monitoring, title="4. Continued Verification", icon="🔄"),
    ],
    "Key Quality System (QMS) Modules": [
        st.Page(show_qrm, title="Quality Risk Management", icon="⚠️"),
        st.Page(show_capa_rca, title="CAPA & Root Cause Analysis", icon="🔍"),
        st.Page(show_post_market_surveillance, title="Post-Market Surveillance", icon="🌐"),
    ],
    "Advanced & Strategic Topics": [
        st.Page(show_digital_health_samd, title="Digital Health / SaMD", icon="🤖"),
        st.Page(show_hybrid_manifesto, title="The Hybrid Manifesto & GxP", icon="🤝")
    ]
}

# --- Sidebar Rendering ---
st.sidebar.title("🧬 Bio-AI Framework")
st.sidebar.markdown("##### Compliant Development Playbook")
st.sidebar.markdown("Navigate the product lifecycle and QMS modules below.")

# Create the navigation menu from the dictionary of pages
pg = st.navigation(PAGES)

# --- Sidebar Footer ---
st.sidebar.divider()
st.sidebar.info(
    "This application demonstrates a framework for integrating Machine Learning into the "
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
