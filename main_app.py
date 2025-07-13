"""
main_app.py

Serves as the primary entry point and navigation controller for the Bio-AI
Excellence Framework application.

This script is responsible for:
1.  Initializing global configurations (logging, page settings).
2.  Loading external configurations and secrets with robust fallbacks.
3.  Dynamically and safely importing all required page-rendering functions.
4.  Constructing the application's navigation structure using Streamlit's
    modern st.navigation API.
5.  Rendering the main application layout, including the sidebar and custom CSS.
6.  Executing the selected page's rendering logic with robust error handling.

Author: AI Engineering SME
Version: 26.1 (Commercial Grade Content Overhaul)
Date: 2025-07-13

Changelog from v25.3:
- [ROBUSTNESS] Enhanced the fatal ImportError handler to explicitly instruct
  the user to install dependencies via `requirements.txt`, directly addressing
  potential setup issues.
- [MAINTAINABILITY] Refined the `BioAIApp` class structure. Encapsulating the
  app logic in a class improves state management, organizes methods logically,
  and makes the application's structure more scalable and testable.
- [DOC] Upgraded all docstrings and comments to a commercial-grade standard,
  explaining the rationale behind key architectural decisions and reflecting
  the context of the application's new, enhanced content.
"""

# --- Core and Third-Party Imports ---
import logging
import sys
from typing import List, Tuple, Callable, Dict, Any

# --- Streamlit Import ---
# Must be imported before any other st.* calls.
import streamlit as st

# --- Local Application Imports ---
# A single try/except block in the main execution scope handles cases where
# modules are missing, guiding the developer to install dependencies.
try:
    from helpers.styling import get_custom_css
    from app_pages import (
        show_welcome_page, show_define_phase, show_measure_phase,
        show_analyze_phase, show_improve_phase, show_control_phase,
        show_comparison_matrix, show_hybrid_manifesto
    )
except ImportError as e:
    # This is a fatal, pre-emptive error. We cannot proceed.
    error_msg = (
        f"A critical module is missing: {e}. "
        "This is likely due to an incomplete setup.\n\n"
        "Please install all required packages by running the following "
        "command in your terminal from the project's root directory:\n"
        "pip install -r requirements.txt"
    )
    print(f"[FATAL ERROR] {error_msg}", file=sys.stderr)
    st.error(f"""
    **Application Startup Failed**

    A critical module could not be found: `{e}`

    Please ensure your environment is set up correctly. In your terminal, run:
    ```
    pip install -r requirements.txt
    ```
    """)
    st.stop()


class BioAIApp:
    """
    An object-oriented class to encapsulate the Bio-AI Framework application.

    This class structure organizes the application's state and logic,
    promoting maintainability and scalability.
    """
    def __init__(self):
        """Initializes the application's components."""
        self.logger = self._setup_logging()
        self.config: Dict[str, Any] = {}
        self.pages: List[st.Page] = []
        self.selected_page: st.Page = None

    def _setup_logging(self) -> logging.Logger:
        """Configures application-wide logging safely."""
        # Configure logging only if the root logger has no handlers to prevent
        # duplicate log entries on Streamlit re-runs.
        if not logging.root.handlers:
            log_config = st.secrets.get("logging", {})
            log_level_str = log_config.get("level", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)

            logging.basicConfig(
                level=log_level,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                stream=sys.stderr,
            )
        return logging.getLogger(__name__)

    def _load_config(self) -> None:
        """Loads all necessary configurations from st.secrets."""
        self.config = st.secrets
        self.logger.info("Configuration loaded.")

    def _apply_styling(self) -> None:
        """Applies custom CSS to the application."""
        try:
            custom_css = get_custom_css()
            st.markdown(custom_css, unsafe_allow_html=True)
        except Exception as e:
            self.logger.warning(f"Could not apply custom CSS. Error: {e}", exc_info=True)

    def _build_sidebar(self) -> None:
        """Constructs and renders the sidebar navigation."""
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

        self.pages = [
            st.Page(page_func, title=title, icon=icon)
            for title, page_func, icon in page_modules
        ]

        with st.sidebar:
            st.title("🧬 Bio-AI Framework")
            st.markdown("##### Assay Development Playbook")
            st.markdown("Navigate the R&D lifecycle below.")

            self.selected_page = st.navigation(self.pages)

            st.divider()
            st.info("A commercial-grade hybrid framework for superior biotech R&D.")

            app_meta = self.config.get("app_meta", {})
            app_version = app_meta.get("version", "26.1")
            url_config = self.config.get("urls", {})
            source_code_url = url_config.get("source_code")

            if source_code_url:
                st.markdown(f"**[View Source on GitHub]({source_code_url})**")
            st.caption(f"Version: {app_version}")

    def _render_page(self) -> None:
        """Executes the rendering logic for the selected page."""
        page_title = self.selected_page.title
        self.logger.info(f"Rendering page: '{page_title}'")
        try:
            self.selected_page.run()
        except Exception as e:
            self.logger.error(f"Error rendering page '{page_title}'", exc_info=True)
            st.error(f"An unexpected error occurred on the '{page_title}' page.")
            st.exception(e)

    def run(self) -> None:
        """The main execution method for the application."""
        # This must be the very first Streamlit command.
        st.set_page_config(
            page_title="Bio-AI Excellence Framework",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.logger.info("Application starting up.")

        self._load_config()
        self._apply_styling()
        self._build_sidebar()
        self._render_page()
        self.logger.info("Page rendering complete.")

if __name__ == "__main__":
    app = BioAIApp()
    app.run()
