import numpy as np
import plotly.graph_objects as go

# Molecular markers for thermal emission spectrum: (wavelength μm, label, hex color)
# Matches the reference style: vertical dashed lines with colored labels
THERMAL_EMISSION_MARKERS = [
    (4.3, "CO2", "#ef4444"),
    (6.3, "H2O", "#3b82f6"),
    (7.7, "CH4", "#f59e0b"),
    (8.5, "N2O", "#22c55e"),
    (9.6, "O3", "#8b5cf6"),
    (15.0, "CO2", "#ef4444"),
]


def make_spectrum_figure(spectrum: dict, show_bands: bool = True) -> go.Figure:
    """
    Build a Plotly figure of the spectrum (thermal emission or transmission) with
    optional molecular band markers as vertical dashed lines and labels.

    spectrum: {"wavelength": np.array (μm), "depth": np.array}
    """
    wl = np.array(spectrum["wavelength"])
    depth = np.array(spectrum["depth"])

    fig = go.Figure()

    # Main spectrum trace — blue line
    fig.add_trace(
        go.Scatter(
            x=wl,
            y=depth,
            mode="lines",
            line=dict(color="#2563eb", width=2),
            name="Spectrum",
        )
    )

    # Molecular markers: vertical dashed lines with labels (reference style)
    if show_bands:
        for wl_marker, label, color in THERMAL_EMISSION_MARKERS:
            if wl.min() <= wl_marker <= wl.max():
                fig.add_vline(
                    x=wl_marker,
                    line=dict(dash="dash", color=color, width=1.5),
                    opacity=0.9,
                )
                fig.add_annotation(
                    x=wl_marker,
                    y=1.0,
                    yref="paper",
                    yanchor="bottom",
                    yshift=4,
                    text=label,
                    showarrow=False,
                    font=dict(size=11, color=color),
                )

    fig.update_layout(
        title=dict(
            text="Thermal Emission Spectrum",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=18, color="#e2e8f0"),
        ),
        template="plotly_dark",
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#e2e8f0", size=12),
        xaxis=dict(
            title="Wavelength (μm)",
            range=[max(3.5, wl.min()), min(18.5, wl.max())],
            type="linear",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.25)",
            zeroline=False,
            title_font=dict(color="#e2e8f0"),
            tickfont=dict(color="#e2e8f0"),
        ),
        yaxis=dict(
            title="Contrast (Planet/Star)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.25)",
            zeroline=False,
            showticklabels=True,
            exponentformat="e",
            tickformat=".2e",
            title_font=dict(color="#e2e8f0"),
            tickfont=dict(color="#e2e8f0", size=11),
            tickangle=0,
        ),
        height=520,
        margin=dict(t=90, b=60, l=85, r=40),
        showlegend=False,
    )

    return fig
