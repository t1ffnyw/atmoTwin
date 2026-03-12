import streamlit as st

from state import init_state
from ui import configure_page
from home_content import HERO_CONTENT, WORKFLOW_STEPS, RESOURCES


configure_page()
init_state()


def render_hero() -> None:
    """Hero section with problem statement and configurable summary block."""
    st.title(HERO_CONTENT["title"])
    st.caption(HERO_CONTENT["tagline"])

    st.divider()

    col_left, col_right = st.columns([2, 2])

    with col_left:
        st.subheader("Why AtmoTwin?")
        st.markdown(HERO_CONTENT["problem_statement"])

    with col_right:
        with st.container(border=True):
            st.markdown("**Scenario summary**")
            st.markdown(
                f"- **Baseline**: {HERO_CONTENT['baseline_scenario']}\n"
                f"- **Wavelength range**: {HERO_CONTENT['wavelength_range']}\n"
                f"- **Primary application**: {HERO_CONTENT['primary_application']}"
            )
            


def render_workflow() -> None:
    """Step-by-step workflow guide (Builder → Results → Explore)."""
    st.subheader("How to use AtmoTwin")
    st.markdown(
        "Follow this suggested path to go from a physical scenario to interpretable spectra "
        "and scenario comparison. You should be able to scan this in under 30 seconds."
    )

    cols = st.columns(len(WORKFLOW_STEPS))
    for col, step in zip(cols, WORKFLOW_STEPS):
        with col:
            with st.container(border=True):
                st.markdown(f"**{step['step_number']}. {step['label']}**")
                st.caption(step["headline"])
                st.write(step["summary"])

                if step.get("page_path"):
                    st.page_link(
                        step["page_path"],
                        label=step.get("cta_label", f"Go to {step['label']}"),
                    )


def render_resources() -> None:
    """Footer-style resources and external links."""
    st.subheader("Resources and links")

    cols = st.columns(len(RESOURCES))
    for col, item in zip(cols, RESOURCES):
        with col:
            with st.container(border=True):
                icon = f"{item['emoji']} " if item.get("emoji") else ""
                st.markdown(f"**{icon}{item['label']}**")
                st.caption(item["description"])
                st.markdown(f"[Open link]({item['url']})")

    st.info(
        "Use the sidebar to switch between pages at any time."
    )


def main() -> None:
    render_hero()
    render_workflow()
    render_resources()


if __name__ == "__main__":
    main()