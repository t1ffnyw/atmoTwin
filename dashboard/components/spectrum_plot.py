from typing import Dict

import numpy as np
import plotly.graph_objects as go

# Molecular markers for thermal emission spectrum: (wavelength μm, label, hex color)
THERMAL_EMISSION_MARKERS = [
    (4.3, "CO2", "#ef4444"),
    (6.3, "H2O", "#3b82f6"),
    (7.7, "CH4", "#f59e0b"),
    (8.5, "N2O", "#22c55e"),
    (9.6, "O3", "#8b5cf6"),
    (15.0, "CO2", "#ef4444"),
]

# Consistent molecule → color mapping for the contribution plot
MOLECULE_COLORS = {
    "O3": "#8b5cf6",
    "CH4": "#f59e0b",
    "CO2": "#ef4444",
    "H2O": "#3b82f6",
    "N2O": "#22c55e",
    "CO": "#92400e",
    "O2": "#ec4899",
}


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


def make_contributions_figure(
    wavelength: np.ndarray,
    contributions: Dict[str, np.ndarray],
) -> go.Figure:
    """
    Stacked semi-transparent area chart showing each molecule's contribution
    to the spectrum (baseline − baseline_without_molecule).
    """
    fig = go.Figure()

    for mol_name, contrib in contributions.items():
        if np.nanmax(contrib) == 0:
            continue
        color = MOLECULE_COLORS.get(mol_name, "#888888")
        fig.add_trace(
            go.Scatter(
                x=wavelength,
                y=contrib,
                mode="lines",
                fill="tozeroy",
                line=dict(width=0.5, color=color),
                fillcolor=_hex_to_rgba(color, 0.5),
                name=mol_name,
            )
        )

    fig.update_layout(
        title=dict(
            text="Per-molecule contributions to the spectrum",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=16, color="#e2e8f0"),
        ),
        template="plotly_dark",
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#e2e8f0", size=12),
        xaxis=dict(
            title="Wavelength (μm)",
            range=[max(3.5, wavelength.min()), min(18.5, wavelength.max())],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.25)",
            zeroline=False,
            title_font=dict(color="#e2e8f0"),
            tickfont=dict(color="#e2e8f0"),
        ),
        yaxis=dict(
            title="Contribution (baseline − no molecule)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.25)",
            zeroline=False,
            showticklabels=True,
            exponentformat="e",
            tickformat=".1e",
            title_font=dict(color="#e2e8f0"),
            tickfont=dict(color="#e2e8f0", size=11),
        ),
        height=400,
        margin=dict(t=80, b=60, l=85, r=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#e2e8f0", size=11),
        ),
        showlegend=True,
    )

    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
