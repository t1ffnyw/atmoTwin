import plotly.graph_objects as go
import streamlit as st

# ── upload classification display ──────────────────────────────────────────────

_CLASS_DISPLAY = {
    "modern_earth": (
        "Inhabited (Modern)",
        "#22c55e",  # green
        "Oxygenic photosynthesis biosignature — O\u2083+CH\u2084 coexistence",
    ),
    "archean_earth": (
        "Inhabited (Archean)",
        "#86efac",  # light green
        "Methanogenic biosphere — high CH\u2084, no O\u2082",
    ),
    "lifeless": (
        "Lifeless",
        "#ef4444",  # red
        "Chemical equilibrium — no strong biosignature signal",
    ),
    "false_positive": (
        "False Positive",
        "#f97316",  # orange
        "Abiotic O\u2082 — waterworld photolysis, not life",
    ),
}

_FEATURE_LABELS = {
    "o3_depth": "O\u2083 band depth (9.6 \u00b5m)",
    "ch4_depth": "CH\u2084 band depth (7.7 \u00b5m)",
    "n2o_depth": "N\u2082O band depth (7.8 \u00b5m)",
    "co_depth": "CO band depth (4.67 \u00b5m)",
    "h2o_depth": "H\u2082O band depth (6.3 \u00b5m)",
    "co2_depth": "CO\u2082 band depth (15.0 \u00b5m)",
    "o3_ch4_ratio": "O\u2083/CH\u2084 ratio",
    "co_o3_ratio": "CO/O\u2083 ratio",
    "ch4_co2_ratio": "CH\u2084/CO\u2082 ratio",
}

_RATIO_EXPLANATIONS = {
    "o3_ch4_ratio": "Higher when O₃ and CH₄ coexist (classic disequilibrium biosignature).",
    "co_o3_ratio": "Higher can indicate abiotic O₂/O₃ production (false-positive risk) when CO is also high.",
    "ch4_co2_ratio": "Higher can indicate methane-rich atmospheres (often Archean-like) relative to CO₂.",
}


def render_upload_classification(result: dict) -> None:
    """Display 4-class RF classification results for an uploaded spectrum.

    result keys: class_names, probabilities, predicted_class, confidence,
                 is_inhabited, key_features, diagnostics (optional)
    """
    predicted = result["predicted_class"]
    label, color, subtitle = _CLASS_DISPLAY.get(
        predicted, (predicted, "#94a3b8", "")
    )

    life_emoji = "\U0001f7e2" if result["is_inhabited"] else "\U0001f534"
    st.markdown(f"### {life_emoji} Classification: **{label}**")
    st.caption(subtitle)

    st.divider()

    # ── headline metrics ───────────────────────────────────────────────────────
    m1, m2 = st.columns([1, 1])
    with m1:
        st.metric("Predicted class", label)
    with m2:
        st.metric("Confidence", f"{result['confidence']:.1%}")

    # ── probability bar chart ──────────────────────────────────────────────────
    class_names = result["class_names"]
    probs = result["probabilities"]

    display_labels = [
        _CLASS_DISPLAY.get(c, (c, "#94a3b8", ""))[0] for c in class_names
    ]
    bar_colors = [
        _CLASS_DISPLAY.get(c, (c, "#94a3b8", ""))[1] for c in class_names
    ]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=display_labels,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1.15], tickformat=".0%", title="Probability"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#0B1120",
        paper_bgcolor="#0B1120",
        font_color="#e2e8f0",
        margin=dict(l=10, r=60, t=10, b=40),
        height=220,
    )
    st.plotly_chart(fig, width="stretch")

    st.progress(result["confidence"], text=f"Confidence: {result['confidence']:.1%}")

    # ── diagnostic engineered feature values (instance-level) ──────────────────
    diagnostics = result.get("diagnostics") or {}
    if diagnostics:
        st.divider()
        st.markdown("**Diagnostic molecular features (from this spectrum):**")

        cols = st.columns(3)
        for i, k in enumerate(
            ["o3_depth", "ch4_depth", "co2_depth", "h2o_depth", "n2o_depth", "co_depth"]
        ):
            if k in diagnostics:
                with cols[i % 3]:
                    st.metric(_FEATURE_LABELS.get(k, k), f"{diagnostics[k]:.4g}")

        st.markdown("**Disequilibrium ratios:**")
        for rk in ["o3_ch4_ratio", "co_o3_ratio", "ch4_co2_ratio"]:
            if rk in diagnostics:
                st.markdown(
                    f"- `{_FEATURE_LABELS.get(rk, rk)}` = **{diagnostics[rk]:.4g}**  \n"
                    f"  { _RATIO_EXPLANATIONS.get(rk, '') }"
                )

    # ── top contributing features ──────────────────────────────────────────────
    if result.get("key_features"):
        st.markdown("**Top contributing features:**")
        for feat_name, importance in result["key_features"]:
            friendly = _FEATURE_LABELS.get(feat_name, feat_name)
            st.markdown(f"- `{friendly}` — importance {importance:.3f}")


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

