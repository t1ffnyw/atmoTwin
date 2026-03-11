import streamlit as st


def configure_page() -> None:
    """Apply a consistent page configuration for all Streamlit pages."""
    st.set_page_config(
        page_title="AtmoTwin",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

