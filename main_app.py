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
Version: 23.8 (Definitive Final Build)
Date: 2023-10-26
"""

import streamlit as st
import logging
import sys

def main():
    """
    Main function to configure and run the Streamlit application.
    """
    # This must be the very first Streamlit command.
    st.set_page_config(
        page_title="Bio-AI Excellence Framework",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Logging Configuration ---
    log_config = st.secrets.get("logging", {})
    log_level_str = log_config.get("level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level_str, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)
    logger.info("Application starting up.")

    # --- Resilient Module Imports ---
    try:
        from helpers.styling import get_custom_css
        from app_pages import (
            show_welcome_page, show_define_phase, show_measure_phase,
            show_analyze_phase, show_improve_phase, show_control_phase,
            show_comparison_matrix, show_hybrid_manifesto
        )
    except ImportError as e:
        logger.error(f"Fatal error during import: {e}", exc_info=True)
        st.error(
            "Application startup failed: A critical component could not be loaded. "
            "Please check that 'app_pages.py' and the 'helpers/' package exist and are correct."
        )
        st.stop()

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

    # --- Apply Custom Styling ---
    try:
        st.markdown(get_custom_css(), unsafe_allow_html=True)
    except Exception as e:
        logger.warning(f"Could not apply custom CSS. Error: {e}")

    # --- Sidebar and Navigation ---
    app_meta = st.secrets.get("app_meta", {})
    app_version = app_meta.get("version", "N/A")
    url_config = st.secrets.get("urls", {})
    source_code_url = url_config.get("source_code")

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
        st.info("A hybrid framework for superior biotech R&D.")
        if source_code_url:
            st.markdown(f"**[View Source on GitHub]({source_code_url})**")
        st.caption(f"Version: {app_version}")

    # --- Page Rendering ---
    logger.info(f"Running page: '{pg.title}'")
    try:
        pg.run()
    except Exception as e:
        logger.error(f"Error rendering page '{pg.title}'", exc_info=True)
        st.error("An unexpected error occurred on this page.")
        st.exception(e)

if __name__ == "__main__":
    main()
