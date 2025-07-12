"""
main_app.py

Serves as the primary entry point and navigation controller for the Bio-AI
Excellence Framework application.

This script is responsible for:
1.  Initializing global configurations (logging, page settings).
2.  Loading external configurations and secrets with robust fallbacks.
3.  Dynamically and safely importing page-rendering functions.
4.  Constructing the application's navigation structure using Streamlit's
    modern st.navigation API.
5.  Rendering the main application layout, including the sidebar.
6.  Executing the selected page's rendering logic.

Author: AI Engineering SME
Version: 23.6 (Final Verification Build)
Date: 2023-10-26
"""

import streamlit as st
import logging
import sys

# ==============================================================================
# 0. APPLICATION BOOTSTRAP & INITIALIZATION
# ==============================================================================
def main():
    """
    Main function to configure and run the Streamlit application.
    """
    # --- VERIFICATION STEP ---
    # If you see balloons, you are running the correct version of this file.
    # If the app crashes without showing balloons, the file was not updated.
    st.balloons()
    
    # --- 0.1. Logging Configuration ---
    log_config = st.secrets.get("logging", {})
    log_level_str = log_config.get("level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level_str, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)
    logger.info("Application starting up. Verification build is running.")

    # --- 0.2. Dynamic & Resilient Module Imports ---
    try:
        from helpers.styling import get_custom_css
        logger.debug("Successfully imported 'helpers' modules.")
    except ImportError as e:
        logger.error(f"Fatal error: Failed to import critical helper modules. {e}")
        st.error(
            "Application startup failed: A critical component could not be loaded. "
            "Please check the logs and file structure. The 'helpers' package may be missing."
        )
        st.stop()

    try:
        from app_pages import (
            show_welcome_page, show_define_phase, show_measure_phase,
            show_analyze_phase, show_improve_phase, show_control_phase,
            show_comparison_matrix, show_hybrid_manifesto
        )
        page_modules = [
            ("Welcome & Framework", show_welcome_page, "🏠"),
            ("Define: Clinical Need & Design", show_define_phase, "🌀"),
            ("Measure: System Validation", show_measure_phase, "🔬"),
            ("Analyze: Root Cause & Failure", show_analyze_phase, "📈"),
            ("Improve: Optimization & Robustness", show_improve_phase, "⚙️"),
            ("Control: Lab Operations & PMS", show_control_phase, "📡"),
            ("Methodology Comparison", show_comparison_matrix, "⚔️"),
            ("The Hybrid Manifesto & GxP", show_hybrid_manifesto, "🤝")
        ]
        logger.debug("Successfully imported all page modules.")
    except ImportError as e:
        logger.error(f"Error importing page modules: {e}. The app may be unstable.")
        st.error(f"Failed to load page definitions from app_pages.py. Error: {e}")
        st.stop()

    # ==============================================================================
    # 1. GLOBAL PAGE CONFIGURATION (WITH GRACEFUL DEGRADATION)
    # ==============================================================================
    app_meta = st.secrets.get("app_meta", {})
    app_version = app_meta.get("version", "N/A")

    # --- DEFINITIVE FIX: Dynamically build the menu_items dictionary ---
    menu_items = {
        'About': f"""
        ## 🧬 The Bio-AI Excellence Framework
        **An interactive playbook for optimizing genomic assays and devices.**
        This application demonstrates a unified framework that fuses the statistical rigor of
        **Six Sigma** with the predictive power of **Machine Learning**.
        **Version:** {app_version}
        """
    }
    url_config = st.secrets.get("urls", {})
    help_url = url_config.get("help")
    bug_report_url = url_config.get("bug_report")
    source_code_url = url_config.get("source_code")

    if help_url:
        menu_items['Get Help'] = help_url
    if bug_report_url:
        menu_items['Report a bug'] = bug_report_url

    st.set_page_config(
        page_title="Bio-AI Excellence Framework",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=menu_items
    )

    # --- 1.1. Custom Styling ---
    try:
        st.markdown(get_custom_css(), unsafe_allow_html=True)
    except Exception as e:
        logger.warning(f"Failed to apply custom CSS. Error: {e}")
        st.toast("Could not load custom theme.", icon="🎨")

    # ==============================================================================
    # 2. APPLICATION NAVIGATION & SIDEBAR
    # ==============================================================================
    PAGES = [
        st.Page(page_func, title=title, icon=icon)
        for title, page_func, icon in page_modules
    ]

    with st.sidebar:
        st.title("🧬 Bio-AI Framework")
        st.markdown("##### Assay Development Playbook")
        st.markdown("Navigate the R&D lifecycle below.")
        pg = st.navigation(PAGES)
        st.divider()
        st.info(
            "This app demonstrates integrating ML into the biotech R&D lifecycle "
            "for superior performance and reliability."
        )
        if source_code_url:
            st.markdown(f"**[View Source on GitHub]({source_code_url})**")
        st.caption(f"Version: {app_version}")

    # ==============================================================================
    # 3. PAGE RENDERING LOGIC WITH ERROR BOUNDARY
    # ==============================================================================
    logger.info(f"Running page: '{pg.title}'")
    try:
        pg.run()
        logger.info(f"Finished rendering page: '{pg.title}'")
    except Exception as e:
        logger.error(f"An error occurred while rendering page '{pg.title}': {e}", exc_info=True)
        st.error(f"An unexpected error occurred on this page. Please check the logs or contact support.")
        st.exception(e)

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()
