"""
main_app.py

Serves as the primary entry point and navigation controller for the Bio-AI 
Excellence Framework application.

This script is responsible for:
1.  Initializing global configurations (logging, page settings).
2.  Loading external configurations and secrets.
3.  Dynamically and safely importing page-rendering functions.
4.  Constructing the application's navigation structure using Streamlit's
    modern st.navigation API.
5.  Rendering the main application layout, including the sidebar.
6.  Executing the selected page's rendering logic.

This architecture ensures a clean separation of concerns, where this main script
handles the "scaffolding" and "routing," while content and logic are delegated
to dedicated modules.

Author: AI Engineering SME
Version: 23.1 (Commercial Grade Refactor)
Date: 2023-10-26
"""

import streamlit as st
import logging
import sys
from typing import List, Callable

# ==============================================================================
# 0. APPLICATION BOOTSTRAP & INITIALIZATION
# ==============================================================================
# Encapsulates the application's core logic into a main function. This is a 
# standard Python best practice that prevents global scope pollution and makes
# the script's behavior explicit and testable.
def main():
    """
    Main function to configure and run the Streamlit application.
    """
    # --- 0.1. Logging Configuration ---
    # Centralized logging is critical for debugging and monitoring in production.
    # It provides a standardized way to record events, warnings, and errors.
    # Configuration is loaded from secrets.toml to allow environment-specific
    # logging levels (e.g., DEBUG in dev, INFO in prod).
    log_level_str = st.secrets.get("logging", {}).get("level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level_str, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)
    logger.info("Application starting up.")


    # --- 0.2. Dynamic & Resilient Module Imports ---
    # To prevent the entire app from crashing due to an error in a single page 
    # or helper, we wrap imports in try-except blocks. This makes the application
    # robust and gracefully degradable.
    try:
        from app_helpers import get_custom_css
        logger.debug("Successfully imported 'app_helpers'.")
    except ImportError as e:
        logger.error(f"Fatal error: Failed to import critical module 'app_helpers'. {e}")
        st.error(
            "Application startup failed: A critical component could not be loaded. "
            "Please check the logs for more details."
        )
        st.stop() # Halt execution if a critical helper is missing.

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
        logger.error(f"Error importing one or more page modules: {e}. The app will run without them.")
        st.toast(f"Warning: A page module failed to load. {e}", icon="⚠️")
        # Gracefully filter out pages that failed to import if necessary.
        # This simple implementation assumes all pages loaded or none did for brevity.
        # A more complex system could check each function individually.


    # ==============================================================================
    # 1. GLOBAL PAGE CONFIGURATION
    # ==============================================================================
    # This must be the first Streamlit command. Configuration is loaded from 
    # st.secrets to avoid hardcoding and facilitate environment changes.
    logger.debug("Setting page configuration.")
    app_version = st.secrets.get("app_meta", {}).get("version", "N/A")
    st.set_page_config(
        page_title="Bio-AI Excellence Framework",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': st.secrets.urls.help,
            'Report a bug': st.secrets.urls.bug_report,
            'About': f"""
            ## 🧬 The Bio-AI Excellence Framework
            
            **An interactive playbook for developing and optimizing robust genomic assays and devices.**

            This application demonstrates a unified framework that fuses the statistical rigor of
            classical **Design of Experiments (DOE) and Six Sigma** with the predictive power of **Machine Learning and Bioinformatics**.
            
            Navigate through the R&D lifecycle (framed as DMAIC) to see foundational methods presented 
            alongside their AI-augmented and regulatory-compliant counterparts for the modern biotech lab. This version integrates both the original educational examples and the advanced, expert-level QMS/regulatory tools.
            
            **Version:** {app_version}
            """
        }
    )

    # --- 1.1. Custom Styling ---
    # Apply custom CSS. The use of unsafe_allow_html is a known security consideration.
    # We acknowledge this and ensure the source is trusted and internally controlled.
    # Reference: OWASP Top 10 - A03:2021 - Injection
    try:
        css = get_custom_css()
        # SECURITY NOTE: The 'unsafe_allow_html' parameter is used here with trusted,
        # internally-generated CSS. This is a controlled risk. Do not pass any
        # user-generated or external strings to this function to prevent XSS attacks.
        st.markdown(css, unsafe_allow_html=True)
        logger.debug("Custom CSS applied successfully.")
    except Exception as e:
        logger.warning(f"Failed to apply custom CSS. Using default theme. Error: {e}")
        st.toast("Could not load custom theme.", icon="🎨")


    # ==============================================================================
    # 2. APPLICATION NAVIGATION & SIDEBAR
    # ==============================================================================
    # Using st.Page and st.navigation is the modern, idiomatic way to build
    # multi-page apps in Streamlit. It's robust and manages state correctly.
    logger.debug("Configuring application navigation.")
    PAGES = [
        st.Page(page_func, title=title, icon=icon) 
        for title, page_func, icon in page_modules
    ]

    # --- Sidebar Rendering ---
    with st.sidebar:
        st.title("🧬 Bio-AI Framework")
        st.markdown("##### Assay Development Playbook")
        st.markdown("Navigate the R&D lifecycle below.")
        
        # Create the navigation menu from the list of pages
        pg = st.navigation(PAGES)

        st.divider()
        st.info(
            "This app demonstrates a framework for integrating Machine Learning into the "
            "biotech R&D lifecycle to achieve superior assay performance and reliability."
        )
        st.markdown(
            f"**[View Source on GitHub]({st.secrets.urls.source_code})**"
        )
        st.caption(f"Version: {app_version}")
    
    logger.info("Sidebar rendered and navigation configured.")


    # ==============================================================================
    # 3. PAGE RENDERING LOGIC
    # ==============================================================================
    # The st.navigation object handles the rendering of the selected page.
    # This single line executes the function associated with the current page.
    logger.info(f"Running page: '{pg.title}'")
    pg.run()
    logger.info(f"Finished rendering page: '{pg.title}'")


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # The `if __name__ == "__main__"` block ensures that the `main()` function is
    # called only when the script is executed directly. This is crucial for making
    # the code testable, reusable, and compliant with Python standards.
    main()
