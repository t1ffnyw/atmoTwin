import streamlit as st

_LIFE_CSS = """
<style>
/* Sidebar border accent */
section[data-testid="stSidebar"] {
    border-right: 2px solid #00b4d8;
}

/* Teal accent for buttons */
div.stButton > button {
    border: 1px solid #00b4d8;
    color: #e2e8f0;
}
div.stButton > button:hover {
    background-color: #00b4d8;
    color: #0B1120;
}

/* Divider color */
hr {
    border-color: rgba(0,180,216,0.3) !important;
}

/* Slider accent */
div[data-testid="stSlider"] [role="slider"] {
    background-color: #00b4d8 !important;
}

/* Tab styling */
button[data-baseweb="tab"] {
    color: #94a3b8 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00b4d8 !important;
}
</style>
"""


def configure_page() -> None:
    """Apply a consistent page configuration and LIFE color theme CSS."""
    st.set_page_config(
        page_title="AtmoTwin",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_LIFE_CSS, unsafe_allow_html=True)

