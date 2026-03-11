import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config import BIOSIG_BANDS


def make_spectrum_figure(spectrum: dict, show_bands: bool = True) -> go.Figure:
    """
    Build a Plotly figure of the transmission spectrum with
    optional biosignature band overlays.

    spectrum: {"wavelength": np.array (μm), "depth": np.array (ppm)}
    """
    wl = np.array(spectrum["wavelength"])
    depth = np.array(spectrum["depth"])

    fig = go.Figure()

    # main spectrum trace
    fig.add_trace(
        go.Scatter(
            x=wl,
            y=depth,
            mode="lines",
            line=dict(color="#e2e8f0", width=1.5),
            name="Transmission Spectrum",
        )
    )

    # biosignature band highlights
    if show_bands:
        for band_name, band in BIOSIG_BANDS.items():
            gas_id = band["gas"]
            gas_mixing = st.session_state.gases.get(gas_id, -8.0)
            # only show band if gas is present at meaningful level
            if gas_mixing > -7.5:
                lo = band["center"] - band["width"] / 2
                hi = band["center"] + band["width"] / 2
                fig.add_vrect(
                    x0=lo,
                    x1=hi,
                    fillcolor=band["color"],
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                    annotation_text=band_name,
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color=band["color"],
                )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Wavelength (μm)",
        yaxis_title="Transit Depth (ppm)",
        xaxis=dict(range=[0.5, 20], type="log"),
        height=500,
        margin=dict(t=40, b=60),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig

