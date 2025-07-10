# main_app.py
import streamlit as st

# We import the functions that define our pages and helpers
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

# --- Global Page Configuration ---
st.set_page_config(
    page_title="AI-Augmented Six Sigma | Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


# --- Page Navigation Dictionary ---
# This dictionary maps the desired sidebar names to the functions that render each page.
# The order here dictates the order in the sidebar. This is our manual control.
PAGES = {
    "🌀 Define Phase": show_define_phase,
    "🔬 Measure Phase": show_measure_phase,
    "📈 Analyze Phase": show_analyze_phase,
    "⚙️ Improve Phase": show_improve_phase,
    "📡 Control Phase": show_control_phase,
    "⚔️ Comparison Matrix": show_comparison_matrix,
    "🧠 Hybrid Strategy": show_hybrid_strategy
}

# --- Sidebar Navigation ---
st.sidebar.title("Dashboard Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# --- Page Rendering ---
# Based on the user's selection, we call the corresponding function from app_pages.py
page_function = PAGES[selection]
page_function()

# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.info(
    "This is an interactive dashboard demonstrating the integration of Machine Learning into the Six Sigma DMAIC framework."
)
