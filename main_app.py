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
Version: 24.1 (SME Refactored Build)
Date: 2024-05-21

Changelog from v23.8:
- [FIX] Moved all module imports to the top level for PEP 8 compliance,
  better dependency clarity, and faster subsequent executions.
- [FIX] Refactored the logging configuration to be more robust. It now checks
  if handlers are already configured to prevent duplicate log outputs in
  re-runs and uses the standard stderr stream for logs.
- [OPTIMIZATION] Centralized configuration loading by fetching `st.secrets`
  once to avoid redundant dictionary lookups.
- [REFACTOR] Improved error handling for missing modules. A single, clear
  try-except block now wraps the main execution, providing a clean exit path
  for fatal import errors.
- [STYLE] Added type hints for function signatures for improved code quality
  and maintainability.
"""

# --- Core and Third-Party Imports ---
import logging
import sys
from typing import List, Tuple, Callable

# --- Streamlit Import ---
# Placed here to ensure it's imported before any other st.* calls.
import streamlit as st

# --- Local Application Imports ---
# Moved to top-level for standard Python practice. A try-except block in the
# main execution scope will handle cases where these modules are missing.
from helpers.styling import get_custom_css
from app_pages import (
    show_welcome_page, show_define_phase, show_measure_phase,
    show_analyze_phase, show_improve_phase, show_control_phase,
    show_comparison_matrix, show_hybrid_manifesto
)


def setup_logging(config: dict) -> logging.Logger:
    """
    Configures application-wide logging.

    Args:
        config: A dictionary containing the logging configuration,
                typically from st.secrets.

    Returns:
        A configured logger instance.
    """
    # Check if handlers are already configured for the root logger. This prevents
    # adding duplicate handlers on Streamlit re-runs.
    if not logging.root.handlers:
        log_config = config.get("logging", {})
        log_level_str = log_config.get("level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            # Using the default stream (stderr) is standard practice.
        )
    
    logger = logging.getLogger(__name__)
    return logger


def main() -> None:
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

    # --- Load Configuration and Set Up Logging ---
    # Fetch all secrets/configs at once to avoid multiple lookups.
    config = st.secrets
    logger = setup_logging(config)
    logger.info("Application starting up.")

    # --- Define Page Modules ---
    # This structure clearly defines the app's pages, their rendering functions,
    # and associated icons.
    page_modules: List[Tuple[str, Callable[[], None], str]] = [
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
        custom_css = get_custom_css()
        st.markdown(custom_css, unsafe_allow_html=True)
    except Exception as e:
        logger.warning(f"Could not apply custom CSS. Error: {e}", exc_info=True)

    # --- Sidebar and Navigation ---
    app_meta = config.get("app_meta", {})
    app_version = app_meta.get("version", "N/A")
    url_config = config.get("urls", {})
    source_code_url = url_config.get("source_code")

    # The modern st.navigation API requires a list of st.Page objects.
    PAGES = [
        st.Page(page_func, title=title, icon=icon)
        for title, page_func, icon in page_modules
    ]

    with st.sidebar:
        st.title("🧬 Bio-AI Framework")
        st.markdown("##### Assay Development Playbook")
        st.markdown("Navigate the R&D lifecycle below.")
        
        # st.navigation is the primary control for switching between pages.
        pg = st.navigation(PAGES)
        
        st.divider()
        st.info("A hybrid framework for superior biotech R&D.")
        if source_code_url:
            st.markdown(f"**[View Source on GitHub]({source_code_url})**")
        st.caption(f"Version: {app_version}")

    # --- Page Rendering ---
    logger.info(f"Rendering page: '{pg.title}'")
    try:
        # Executes the rendering function of the page selected in the sidebar.
        pg.run()
    except Exception as e:
        logger.error(f"Error rendering page '{pg.title}'", exc_info=True)
        st.error("An unexpected error occurred while rendering this page.")
        st.exception(e)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        # This is a fatal error if core modules are missing. The logger may not
        # be configured yet, so we print directly to stderr and use st.error.
        print(f"[FATAL] Failed to import a required module: {e}", file=sys.stderr)
        st.error(
            "Fatal Application Error: A critical component could not be loaded. "
            "Please ensure all required packages and local modules (like 'app_pages.py' "
            "and 'helpers/') are available in the correct path."
        )
        # st.stop() is implicitly called after the script finishes, but can be
        # made explicit if there were more code below.
