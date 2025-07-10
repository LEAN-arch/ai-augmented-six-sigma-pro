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
    page_icon="🚀",  # A more professional and relevant icon
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': """
        ## AI-Augmented Process Excellence Dashboard
        
        This interactive dashboard demonstrates how to fuse the rigor of traditional Six Sigma 
        with the predictive power of modern Machine Learning.
        
        **Developed by:** [Your Name/Company]
        **Version:** 1.0
        """
    }
)

# Apply custom CSS from the central helper file to ensure a consistent theme.
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ==============================================================================
# 2. PAGE NAVIGATION DICTIONARY
# ==============================================================================
# This dictionary maps the user-friendly page names in the sidebar to the 
# specific functions that render each page. The order here dictates the order 
# in the sidebar, making it easy to manage the app's structure.
PAGES = {
    "🌀 Define Phase": show_define_phase,
    "🔬 Measure Phase": show_measure_phase,
    "📈 Analyze Phase": show_analyze_phase,
    "⚙️ Improve Phase": show_improve_phase,
    "📡 Control Phase": show_control_phase,
    "---": None,  # A visual separator in the sidebar
    "⚔️ Tool Comparison": show_comparison_matrix,
    "🤝 The Hybrid Manifesto": show_hybrid_strategy
}

# ==============================================================================
# 3. SIDEBAR NAVIGATION & FOOTER
# ==============================================================================
st.sidebar.title("📊 Process Excellence Hub")
st.sidebar.markdown("Navigate through the AI-augmented DMAIC framework.")

# Create the radio button navigation.
selection = st.sidebar.radio(
    "Go to page:", 
    list(PAGES.keys()),
    label_visibility="collapsed" # Hides the "Go to page:" label for a cleaner look
)

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
# This block executes the function corresponding to the user's selection.
# It handles the special case of the separator.
if selection == "---":
    # If the user clicks the separator, we can just re-run to do nothing,
    # or you could have it default to the first page.
    # For now, we do nothing and let Streamlit handle it.
    pass
else:
    page_function = PAGES[selection]
    if page_function:
        page_function()
