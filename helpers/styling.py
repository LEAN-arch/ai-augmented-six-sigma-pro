"""
helpers/styling.py

This module centralizes all visual styling and configuration for the application.
It defines the color palette and generates the main CSS stylesheet.

By isolating styling concerns, we ensure a consistent look and feel across the
entire application and make global style changes easy to implement.

Author: AI Engineering SME
Version: 28.1 (Definitive Final Build)
Date: 2025-07-13

Changelog from v26.1:
- [ENHANCEMENT] Added subtle `transition` and `hover` effects to bordered
  containers (`st.container(border=True)`) to give the UI a more dynamic,
  professional feel.
- [MAINTAINABILITY] CSS selectors remain modern and specific (e.g.,
  `[data-testid="stAppViewContainer"]`) for maximum stability against future
  Streamlit updates. The CSS rules have been reviewed for clarity and purpose.
- [DOC] Finalized all docstrings and comments to a commercial-grade standard,
  explaining the purpose of each styling rule.
"""

from typing import Dict


# ==============================================================================
# 1. APPLICATION COLOR PALETTE
# ==============================================================================
# A centralized, colorblind-safe palette for visual consistency. This serves as
# the Single Source of Truth for colors in Python plotting and CSS.
# Reference: https://davidmathlogic.com/colorblind/
COLORS: Dict[str, str] = {
    "primary": "#0072B2",      # Blue
    "secondary": "#009E73",    # Green
    "accent": "#D55E00",       # Vermillion
    "neutral_yellow": "#F0E442",
    "neutral_pink": "#CC79A7",
    "background": "#F8F9FA",
    "text": "#212529",
    "light_gray": "#DEE2E6",
    "dark_gray": "#495057",
    "success": "#28A745",
    "warning": "#FFC107",
    "danger": "#DC3545"
}


# ==============================================================================
# 2. GLOBAL CSS STYLESHEET
# ==============================================================================
def get_custom_css() -> str:
    """
    Generates the global CSS for the Streamlit application.

    This function injects custom CSS to override default styles, ensuring a
    polished, professional look that aligns with the defined color palette. It
    uses modern `data-testid` selectors for stability.

    Returns:
        A string containing the full CSS stylesheet within <style> tags.
    """
    return f"""
    <style>
        /* --- Global App & Typography --- */
        /* Set main background and default text color for the entire app view */
        div[data-testid="stAppViewContainer"] > main {{
            background-color: {COLORS['background']};
        }}
        .stApp {{
             color: {COLORS['text']};
        }}
        h1, h2 {{
            border-bottom: 2px solid {COLORS['light_gray']};
            padding-bottom: 10px;
        }}
        h3 {{ color: {COLORS['primary']}; }}

        /* --- Bordered Containers (st.container(border=True)) --- */
        /* Targets the specific container wrapper used when border=True is set. */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: white;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            border: 1px solid {COLORS['light_gray']};
            transition: all 0.2s ease-in-out; /* Add smooth transition for hover */
        }}
        /* Add a subtle lift effect on hover for interactivity */
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transform: translateY(-2px);
        }}

        /* --- Widget Styling: Buttons and Tabs --- */
        .stButton > button {{
            border-radius: 0.5rem;
            font-weight: 600;
        }}

        /* Style the tabs for a cleaner, more modern look */
        .stTabs [data-baseweb="tab-list"] button {{
            background-color: transparent;
            border-bottom: 2px solid transparent !important;
            transition: all 0.2s ease-in-out;
            color: {COLORS['dark_gray']};
        }}
        .stTabs [data-baseweb="tab-list"] button:hover {{
             background-color: {hex_to_rgba(COLORS['primary'], 0.05)};
             border-bottom-color: {hex_to_rgba(COLORS['primary'], 0.5)} !important;
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            border-bottom: 2px solid {COLORS['primary']} !important;
            color: {COLORS['primary']};
            font-weight: 600;
        }}
    </style>
    """


# ==============================================================================
# 3. UTILITY FUNCTIONS
# ==============================================================================
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """
    Converts a HEX color string to an RGBA color string.

    This utility is essential for creating semi-transparent colors in Plotly,
    useful for overlays, confidence intervals, and highlighting.

    Args:
        hex_color: The HEX color string (e.g., "#RRGGBB" or "RRGGBB").
        alpha: The alpha (transparency) value, from 0.0 to 1.0.

    Returns:
        The RGBA color string (e.g., "rgba(r, g, b, a)").

    Raises:
        ValueError: If alpha is out of bounds or hex_color has an invalid format.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha value must be between 0.0 and 1.0.")

    hex_color_clean = hex_color.lstrip('#')
    if len(hex_color_clean) != 6:
        raise ValueError(f"Invalid HEX color format '{hex_color}'. Must be 6 hex characters.")

    try:
        r = int(hex_color_clean[0:2], 16)
        g = int(hex_color_clean[2:4], 16)
        b = int(hex_color_clean[4:6], 16)
    except ValueError as e:
        raise ValueError(f"Invalid character in HEX string '{hex_color}'.") from e

    return f"rgba({r}, {g}, {b}, {alpha})"
