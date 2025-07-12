"""
helpers/styling.py

This module centralizes all visual styling and configuration for the application.
It defines the color palette, generates the main CSS stylesheet, and provides
utility functions for color manipulation.

By isolating styling concerns, we ensure a consistent look and feel across the
entire application and make global style changes easy to implement. This follows
the Single Responsibility Principle.

Author: AI Engineering SME
Version: 23.1 (Commercial Grade Refactor)
Date: 2023-10-26
"""
from typing import Dict

# ==============================================================================
# 1. APPLICATION COLOR PALETTE
# ==============================================================================
# A centralized color palette ensures visual consistency. This dictionary serves
# as the Single Source of Truth for all colors used in Python-based plotting
# and CSS. The chosen colors are from a colorblind-safe palette.
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
    Generates the global CSS stylesheet for the Streamlit application.

    This function injects custom CSS to override default Streamlit styles,
    ensuring a polished, professional, and consistent visual identity that
    aligns with the defined color palette.

    :return: A string containing the full CSS stylesheet, wrapped in <style> tags.
    :rtype: str
    """
    return f"""
    <style>
        /* --- Global App Styling --- */
        div[data-testid="stAppViewContainer"] > main {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
        }}

        /* --- Typography --- */
        h1, h2 {{
            color: {COLORS['dark_gray']};
            border-bottom: 2px solid {COLORS['light_gray']};
            padding-bottom: 10px;
        }}
        h3 {{ color: {COLORS['primary']}; }}
        h4, h5 {{ color: {COLORS['dark_gray']}; }}

        /* --- Containers and Borders --- */
        /* Targets custom containers for a card-like effect */
        div[data-testid="stBlock"],
        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {{
            border: 1px solid {COLORS['light_gray']};
            border-radius: 0.5rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
            padding: 1rem 1rem 0 1rem; /* Add padding to bordered containers */
        }}

        /* --- Widgets Styling --- */
        button[data-testid="stButton"] > button {{
            border-radius: 0.5rem;
        }}

        .stTabs [data-baseweb="tab-list"] button {{
            background-color: transparent;
            border-bottom: 2px solid transparent;
            transition: all 0.3s ease-in-out;
        }}

        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            border-bottom: 2px solid {COLORS['primary']};
            color: {COLORS['primary']};
            font-weight: 600; /* Add emphasis to selected tab */
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
    which is useful for overlays, confidence intervals, and highlighting.

    :param hex_color: The HEX color string (e.g., "#RRGGBB").
    :type hex_color: str
    :param alpha: The alpha (transparency) value, from 0.0 (fully transparent)
                  to 1.0 (fully opaque).
    :type alpha: float
    :raises ValueError: If alpha is out of bounds or hex_color has an invalid format.
    :return: The RGBA color string (e.g., "rgba(r, g, b, a)").
    :rtype: str
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha value must be between 0.0 and 1.0.")

    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid HEX color format '{hex_color}'. Must be 6 characters long.")

    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError as e:
        raise ValueError(f"Invalid character in HEX string '{hex_color}'.") from e

    return f"rgba({r}, {g}, {b}, {alpha})"
