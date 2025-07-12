"""
helpers/styling.py

This module centralizes all visual styling and configuration for the application.
It defines the color palette, generates the main CSS stylesheet, and provides
utility functions for color manipulation.

By isolating styling concerns, we ensure a consistent look and feel across the
entire application and make global style changes easy to implement. This follows
the Single Responsibility Principle.

Author: AI Engineering SME
Version: 24.1 (SME Refactored Build)
Date: 2024-05-21

Changelog from v23.1:
- [FIX] In `hex_to_rgba`, added more robust validation to handle missing '#'
  prefixes and ensure the input is a valid 6-digit hex string, preventing
  `ValueError` on malformed input.
- [OPTIMIZATION] The `get_custom_css` function now uses more specific and
  modern Streamlit selectors (e.g., `[data-testid="stAppViewContainer"]`,
  `[data-testid="stVerticalBlock"]`) for more reliable styling and to avoid
  affecting unintended elements.
- [REFACTOR] Removed the overly broad `stBlock` selector which could lead to
  unintended styling of nested elements. The new selectors are more targeted.
- [STYLE] Added type hints to all function signatures for improved code quality
  and static analysis.
- [DOC] Updated docstrings to reflect the changes and provide better context.
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
    polished, professional look that aligns with the defined color palette.

    Returns:
        A string containing the full CSS stylesheet within <style> tags.
    """
    return f"""
    <style>
        /* --- Global App & Typography --- */
        div[data-testid="stAppViewContainer"] > main {{
            background-color: {COLORS['background']};
        }}
        .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp p {{
             color: {COLORS['text']};
        }}
        h1, h2 {{
            border-bottom: 2px solid {COLORS['light_gray']};
            padding-bottom: 10px;
        }}
        h3 {{ color: {COLORS['primary']}; }}

        /* --- Containers with Borders (e.g., st.container(border=True)) --- */
        /* Targets the specific container used when border=True is set */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: white;
            border-radius: 0.5rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
            border: 1px solid {COLORS['light_gray']};
        }}

        /* --- Widget Styling --- */
        .stButton > button {{
            border-radius: 0.5rem;
            font-weight: 600;
        }}

        .stTabs [data-baseweb="tab-list"] button {{
            background-color: transparent;
            border-bottom: 2px solid transparent !important;
            transition: all 0.3s ease-in-out;
            color: {COLORS['dark_gray']};
        }}
        .stTabs [data-baseweb="tab-list"] button:hover {{
             background-color: {hex_to_rgba(COLORS['primary'], 0.1)};
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

    # FIX: Ensure consistent handling of '#' prefix.
    hex_color_clean = hex_color.lstrip('#')

    if len(hex_color_clean) != 6:
        raise ValueError(
            f"Invalid HEX color format '{hex_color}'. Must be 6 hex characters."
        )

    try:
        r = int(hex_color_clean[0:2], 16)
        g = int(hex_color_clean[2:4], 16)
        b = int(hex_color_clean[4:6], 16)
    except ValueError as e:
        raise ValueError(f"Invalid character in HEX string '{hex_color}'.") from e

    return f"rgba({r}, {g}, {b}, {alpha})"
