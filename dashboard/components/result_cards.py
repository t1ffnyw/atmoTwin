import streamlit as st


def render_classification_card(result: dict) -> None:
    """
    result: {
        "label": "disequilibrium" | "equilibrium",
        "confidence": 0.0-1.0,
        "key_features": [("O2/CH4 ratio", 0.85), ...],
    }
    """
    label = result["label"]
    conf = result["confidence"]

    if label == "disequilibrium":
        emoji, color = "🟢", "green"
        subtitle = (
            "Atmospheric chemistry is far from equilibrium — consistent with biological activity"
        )
    else:
        emoji, color = "🔴", "red"
        subtitle = (
            "Atmospheric chemistry is near equilibrium — no strong biosignature signal"
        )

    st.markdown(
        f"""
    ### {emoji} Classification: **:{color}[{label.upper()}]**
    *{subtitle}*
    """
    )

    # confidence bar
    st.progress(conf, text=f"Confidence: {conf:.0%}")

    # top contributing features
    if result.get("key_features"):
        st.markdown("**Key contributing features:**")
        for feat_name, importance in result["key_features"][:5]:
            st.markdown(f"- `{feat_name}` — importance {importance:.2f}")


def render_false_positive_warnings(flags: list[dict]) -> None:
    """
    flags: [{"type": "high_CO", "message": "...", "severity": "warning"|"critical"}, ...]
    """
    if not flags:
        st.success("No false positive indicators detected.")
        return

    for flag in flags:
        if flag["severity"] == "critical":
            st.error(f"⚠️ **{flag['type']}**: {flag['message']}")
        else:
            st.warning(f"⚡ **{flag['type']}**: {flag['message']}")

