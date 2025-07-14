"""
main_app.py

Serves as the primary entry point and navigation controller for the Bio-AI
Excellence Framework application. This version is updated to incorporate the
new Case Study Library and Statistical Tool Advisor pages.

Author: AI Engineering SME
Version: 30.1 (Navigation Hotfix)
Date: 2025-07-17

Changelog from v30.0:
- [BUGFIX] Corrected a `StreamlitAPIException` related to non-unique URL
  pathnames in `st.navigation`. Removed the explicit `url_path` generation,
  allowing Streamlit to handle path creation automatically and robustly.
- [ENHANCEMENT] Simplified the page selection logic to rely solely on the
  `st.navigation` component, which improves maintainability.
"""
import logging
import sys
from typing import List, Tuple, Callable, Dict, Any
import streamlit as st

try:
    from helpers.styling import get_custom_css
    from app_pages import (
        show_welcome_page, show_define_phase, show_measure_phase,
        show_analyze_phase, show_improve_phase, show_control_phase,
        show_comparison_matrix, show_hybrid_manifesto,
        # --- New Page Imports ---
        show_case_study_library, show_tool_advisor
    )
except ImportError as e:
    error_msg = (
        f"A critical module is missing: {e}. "
        "This is likely due to an incomplete setup or a syntax error in an imported file.\n\n"
        "1. First, ensure all required packages are installed by running the following "
        "command in your terminal from the project's root directory:\n"
        "   pip install -r requirements.txt\n\n"
        "2. If the error persists, check the file mentioned in the error message for syntax errors."
    )
    print(f"[FATAL ERROR] {error_msg}", file=sys.stderr)
    st.error(f"""
    **Application Startup Failed**

    A critical module could not be loaded: `{e}`

    Please follow these steps:
    1.  **Install all dependencies:**
        ```
        pip install -r requirements.txt
        ```
    2.  **Check for syntax errors** in the file mentioned in the error traceback if the problem continues.
    """)
    st.stop()

class BioAIApp:
    def __init__(self):
        self.logger = self._setup_logging()
        self.config: Dict[str, Any] = {}
        self.pages: List[st.Page] = []
        self.selected_page: st.Page = None

    def _setup_logging(self) -> logging.Logger:
        if not logging.root.handlers:
            log_config = st.secrets.get("logging", {})
            log_level_str = log_config.get("level", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stderr)
        return logging.getLogger(__name__)

    def _load_config(self) -> None:
        self.config = st.secrets
        self.logger.info("Configuration loaded.")

    def _apply_styling(self) -> None:
        try:
            st.markdown(get_custom_css(), unsafe_allow_html=True)
        except Exception as e:
            self.logger.warning(f"Could not apply custom CSS. Error: {e}", exc_info=True)

    def _build_sidebar(self) -> None:
        page_modules: List[Tuple[str, Callable[[], None], str]] = [
            ("Welcome & Framework", show_welcome_page, "🏠"),
            ("Define: Clinical Need & Design", show_define_phase, "🌀"),
            ("Measure: System Validation", show_measure_phase, "🔬"),
            ("Analyze: Root Cause & Failure", show_analyze_phase, "📈"),
            ("Improve: Optimization & Robustness", show_improve_phase, "⚙️"),
            ("Control: Lab Operations & PMS", show_control_phase, "📡"),
            # --- New Pages Added to Navigation ---
            ("Statistical Tool Advisor", show_tool_advisor, "🧭"),
            ("Case Study Library", show_case_study_library, "📚"),
            # ---
            ("Methodology Comparison", show_comparison_matrix, "⚔️"),
            ("The Hybrid Manifesto & GxP", show_hybrid_manifesto, "🤝")
        ]
        
        # Create st.Page objects for st.navigation.
        # REMOVED the problematic url_path parameter to let Streamlit handle it.
        self.pages = [st.Page(func, title=title, icon=icon) for title, func, icon in page_modules]
        
        with st.sidebar:
            st.title("🧬 Bio-AI Framework")
            st.markdown("##### Assay Development Playbook")
            st.markdown("Navigate the R&D lifecycle below.")
            
            # st.navigation handles page selection and updates query params robustly.
            self.selected_page = st.navigation(self.pages)
            
            st.divider()
            st.info("A commercial-grade hybrid framework for superior biotech R&D.")
            app_version = self.config.get("app_meta", {}).get("version", "30.0") # Version bumped for new release
            source_code_url = self.config.get("urls", {}).get("source_code")
            if source_code_url: st.markdown(f"**[View Source on GitHub]({source_code_url})**")
            st.caption(f"Version: {app_version}")

    def _render_page(self) -> None:
        page_title = self.selected_page.title
        self.logger.info(f"Rendering page: '{page_title}'")
        try:
            # st.navigation takes care of running the correct page function
            self.selected_page.run()
        except Exception as e:
            self.logger.error(f"Error rendering page '{page_title}'", exc_info=True)
            st.error(f"An unexpected error occurred on the '{page_title}' page.")
            st.exception(e)

    def run(self) -> None:
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
