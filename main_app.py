# main_app.py

import streamlit as st

# Import helper functions for styling and page definitions from their respective modules.
# This keeps the main app file clean and focused on navigation and structure.
from app_helpers import get_custom_css
from app_pages import (
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
st.set_page_config(
    page_title="AI-Augmented Process Excellence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': """
        ## AI-Augmented Process Excellence Dashboard
        
        This interactive dashboard demonstrates how to fuse the rigor of traditional Six Sigma 
        with the predictive power of modern Machine Learning.
        
        **Developed by:** A DX/UX Expert AI
        **Version:** 3.0 (Define Phase Enhanced)
        """
    }
)

# Apply custom CSS from the central helper file to ensure a consistent theme.
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ==============================================================================
# 2. PAGE NAVIGATION DICTIONARY
# ==============================================================================
# This dictionary maps the user-friendly page names to the functions that render them.
# The "---" key is a special case handled by the navigation logic to render a divider.
PAGES = {
    "🌀 Define Phase": show_define_phase,
    "🔬 Measure Phase": show_measure_phase,
    "📈 Analyze Phase": show_analyze_phase,
    "⚙️ Improve Phase": show_improve_phase,
    "📡 Control Phase": show_control_phase,
    "---": None,  # This will be rendered as a visual separator.
    "⚔️ Tool Comparison": show_comparison_matrix,
    "🤝 The Hybrid Manifesto": show_hybrid_strategy
}

# ==============================================================================
# 3. SIDEBAR NAVIGATION & STATE MANAGEMENT
# ==============================================================================
st.sidebar.title("📊 Process Excellence Hub")
st.sidebar.markdown("Navigate through the AI-augmented DMAIC framework.")

# Initialize session state to keep track of the current page.
# The first page in the dictionary is the default.
if 'current_page' not in st.session_state:
    st.session_state.current_page = list(PAGES.keys())[0]

# --- Custom Navigation Menu ---
# Iterate through the pages dictionary to create the navigation buttons and dividers.
for page_name, page_function in PAGES.items():
    if page_name == "---":
        st.sidebar.divider()  # Render a non-selectable horizontal line.
    else:
        # Determine if the button is for the currently active page.
        is_active = (st.session_state.current_page == page_name)
        
        # Set the button type to "primary" for the active page, "secondary" for others.
        button_type = "primary" if is_active else "secondary"
        
        # When a button is clicked, update the session state with the new page name.
        if st.sidebar.button(page_name, use_container_width=True, type=button_type):
            st.session_state.current_page = page_name
            # Rerun the script to immediately render the new page.
            st.rerun()

# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.info(
    "This app is a demonstration of integrating Machine Learning into the Six Sigma DMAIC framework to achieve superior process outcomes."
)
st.sidebar.markdown(
    "**Built with [Streamlit](https://streamlit.io)**"
)


# ==============================================================================
# 4. PAGE RENDERING LOGIC
# ==============================================================================
# Get the function for the currently selected page from the dictionary.
page_to_render = PAGES[st.session_state.current_page]

# Execute the function to render the page content.
if page_to_render:
    page_to_render()
