# main_app.py

import streamlit as st

# Import helper functions for styling and all page definitions from their respective modules.
# This architecture keeps the main app file clean and focused on navigation and structure.
from app_helpers import get_custom_css
from app_pages import (
    show_welcome_page,  # New: A dedicated home page for user orientation
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
# Configured for a professional, wide-screen experience suitable for detailed dashboards.
st.set_page_config(
    page_title="AI-Augmented Process Excellence | A Hybrid Framework",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/your-project', # Placeholder
        'Report a bug': "https://github.com/your-repo/your-project/issues", # Placeholder
        'About': """
        ## AI-Augmented Process Excellence Framework
        
        **An interactive playbook for the modern Process Improvement professional.**

        This application demonstrates a unified framework that fuses the statistical rigor of
        traditional Six Sigma with the predictive and scaling power of modern Machine Learning.
        
        Navigate through the DMAIC cycle to see classical tools presented alongside their
        AI-augmented counterparts, complete with interactive simulations and expert rationale.
        
        **Version:** 4.0 (UX/DX Professional Release)
        """
    }
)

# Apply the custom CSS theme from the central helper file for a consistent, polished look.
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ==============================================================================
# 2. APPLICATION NAVIGATION & STATE MANAGEMENT
# ==============================================================================

# The PAGES dictionary maps user-friendly names to their rendering functions.
# This "routing" approach is clean and scalable. The "---" key is a special case
# handled by the navigation logic to render a visual divider in the sidebar.
PAGES = {
    "🏠 Welcome & Overview": show_welcome_page,
    "---": None,  # This will be rendered as a visual separator.
    "🌀 Define Phase": show_define_phase,
    "🔬 Measure Phase": show_measure_phase,
    "📈 Analyze Phase": show_analyze_phase,
    "⚙️ Improve Phase": show_improve_phase,
    "📡 Control Phase": show_control_phase,
    "---": None,
    "⚔️ Tool Comparison Matrix": show_comparison_matrix,
    "🤝 The Hybrid Manifesto": show_hybrid_strategy
}

# --- Sidebar Rendering ---
st.sidebar.title("🚀 AI-Augmented Excellence")
st.sidebar.markdown("### The Modern DMAIC Playbook")
st.sidebar.markdown("Navigate through the unified framework below.")

# Initialize session state to keep track of the current page.
# The "Welcome" page is the default on first load.
if 'current_page' not in st.session_state:
    st.session_state.current_page = list(PAGES.keys())[0]

# --- Custom Navigation Menu using Buttons for a modern feel ---
# Iterate through the pages dictionary to create the navigation buttons and dividers.
for page_name, page_function in PAGES.items():
    if page_name == "---":
        st.sidebar.divider()  # Render a non-selectable horizontal line.
    else:
        # Determine if the button corresponds to the currently active page.
        is_active = (st.session_state.current_page == page_name)
        
        # Use Streamlit's "primary" type for the active page to visually highlight it.
        button_type = "primary" if is_active else "secondary"
        
        # When a button is clicked, update the session state and rerun the script.
        if st.sidebar.button(page_name, use_container_width=True, type=button_type):
            st.session_state.current_page = page_name
            st.rerun()

# --- Sidebar Footer ---
st.sidebar.divider()
st.sidebar.info(
    "This app demonstrates a framework for integrating Machine Learning into the "
    "Six Sigma DMAIC cycle to achieve superior, data-driven process outcomes."
)
st.sidebar.markdown(
    "**[View Source on GitHub](https://github.com/your-repo/your-project)**" # Placeholder
)


# ==============================================================================
# 3. PAGE RENDERING LOGIC
# ==============================================================================

# Retrieve the function associated with the currently selected page.
page_to_render = PAGES[st.session_state.current_page]

# Execute the function to render the main content area of the application.
# This ensures a clean separation of concerns: navigation logic in the sidebar,
# content rendering in the main panel.
if page_to_render:
    page_to_render()
