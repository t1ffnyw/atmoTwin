from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import BIOSIG_BANDS

# ── LIFE mission color palette ──
_BG = "#0B1120"
_GRID = "rgba(0,180,216,0.15)"
_TEXT = "#e2e8f0"
_ACCENT = "#00b4d8"

# Molecular markers for thermal emission spectrum: (wavelength μm, label, hex color)
THERMAL_EMISSION_MARKERS = [
    (4.3, "CO₂", "#ff6b6b"),
    (6.3, "H₂O", "#38bdf8"),
    (7.7, "CH₄", "#fbbf24"),
    (8.5, "N₂O", "#4ade80"),
    (9.6, "O₃", "#a78bfa"),
    (15.0, "CO₂", "#ff6b6b"),
]

# Consistent molecule → color mapping for the contribution plot
MOLECULE_COLORS = {
    "O3": "#a78bfa",
    "CH4": "#fbbf24",
    "CO2": "#ff6b6b",
    "H2O": "#38bdf8",
    "N2O": "#4ade80",
    "CO": "#c084fc",
    "O2": "#f472b6",
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

    fig.add_trace(
        go.Scatter(
            x=wl,
            y=depth,
            mode="lines",
            line=dict(color=_ACCENT, width=2),
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
            font=dict(size=18, color=_TEXT),
        ),
        template="plotly_dark",
        plot_bgcolor=_BG,
        paper_bgcolor=_BG,
        font=dict(color=_TEXT, size=12),
        xaxis=dict(
            title="Wavelength (μm)",
            range=[max(3.5, wl.min()), min(18.5, wl.max())],
            type="linear",
            showgrid=True,
            gridcolor=_GRID,
            zeroline=False,
            title_font=dict(color=_TEXT),
            tickfont=dict(color=_TEXT),
        ),
        yaxis=dict(
            title="Contrast (Planet/Star)",
            showgrid=True,
            gridcolor=_GRID,
            zeroline=False,
            showticklabels=True,
            exponentformat="e",
            tickformat=".2e",
            title_font=dict(color=_TEXT),
            tickfont=dict(color=_TEXT, size=11),
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
            font=dict(size=16, color=_TEXT),
        ),
        template="plotly_dark",
        plot_bgcolor=_BG,
        paper_bgcolor=_BG,
        font=dict(color=_TEXT, size=12),
        xaxis=dict(
            title="Wavelength (μm)",
            range=[max(3.5, wavelength.min()), min(18.5, wavelength.max())],
            showgrid=True,
            gridcolor=_GRID,
            zeroline=False,
            title_font=dict(color=_TEXT),
            tickfont=dict(color=_TEXT),
        ),
        yaxis=dict(
            title="Contribution (baseline − no molecule)",
            showgrid=True,
            gridcolor=_GRID,
            zeroline=False,
            showticklabels=True,
            exponentformat="e",
            tickformat=".1e",
            title_font=dict(color=_TEXT),
            tickfont=dict(color=_TEXT, size=11),
        ),
        height=400,
        margin=dict(t=80, b=60, l=85, r=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color=_TEXT, size=11),
        ),
        showlegend=True,
    )

    return fig


_COMPARISON_COLORS = ["#00b4d8", "#f472b6", "#4ade80", "#fbbf24", "#a78bfa"]


def make_comparison_figure(
    wavelength: np.ndarray,
    fluxes: List[np.ndarray],
    labels: List[str],
    show_biosig_bands: bool = True,
) -> go.Figure:
    """
    Two-panel Plotly figure: overlaid spectra on top, deviation (A − B) on bottom.
    Supports exactly two scenarios.
    """
    color_a = _COMPARISON_COLORS[0]
    color_b = _COMPARISON_COLORS[1]

    deviation = fluxes[0] - fluxes[1]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("", f"{labels[0]}  −  {labels[1]}"),
    )

    fig.add_trace(
        go.Scatter(
            x=wavelength,
            y=fluxes[0],
            mode="lines",
            line=dict(color=color_a, width=2),
            name=labels[0],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=wavelength,
            y=fluxes[1],
            mode="lines",
            line=dict(color=color_b, width=2),
            name=labels[1],
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=wavelength,
            y=deviation,
            mode="lines",
            line=dict(color="#e2e8f0", width=1.5),
            name="Deviation",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color="gray", width=0.8, dash="dot"), row=2, col=1)

    wl_lo = max(3.5, float(wavelength.min()))
    wl_hi = min(18.5, float(wavelength.max()))

    shared_axis = dict(
        showgrid=True,
        gridcolor=_GRID,
        zeroline=False,
        title_font=dict(color=_TEXT),
        tickfont=dict(color=_TEXT, size=11),
    )
    y_fmt = dict(
        exponentformat="e",
        tickformat=".2e",
        showticklabels=True,
    )

    fig.update_xaxes(range=[wl_lo, wl_hi], **shared_axis, row=1, col=1)
    fig.update_xaxes(
        range=[wl_lo, wl_hi],
        title_text="Wavelength (μm)",
        **shared_axis,
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Contrast (Planet/Star)", **shared_axis, **y_fmt, row=1, col=1)
    fig.update_yaxes(title_text="Deviation", **shared_axis, **y_fmt, row=2, col=1)

    if show_biosig_bands:
        visible_bands = []
        for label, band in BIOSIG_BANDS.items():
            center = float(band["center"])
            width = float(band["width"])
            color = band.get("color", "#ffffff")

            x0 = center - width / 2.0
            x1 = center + width / 2.0

            if x1 < wl_lo or x0 > wl_hi:
                continue

            visible_bands.append((label, center, width, color))

            x0_clipped = max(x0, wl_lo)
            x1_clipped = min(x1, wl_hi)

            fig.add_vrect(
                x0=x0_clipped,
                x1=x1_clipped,
                line_width=0,
                fillcolor=color,
                opacity=0.18,
                row=1,
                col=1,
                layer="below",
            )

        visible_bands.sort(key=lambda b: b[1])
        min_gap = (wl_hi - wl_lo) * 0.06
        y_offsets = [0, 14, 28]
        prev_centers: list[tuple[float, int]] = []

        for label, center, _width, color in visible_bands:
            tier = 0
            for prev_c, prev_tier in prev_centers:
                if abs(center - prev_c) < min_gap and prev_tier == tier:
                    tier += 1
            tier = tier % len(y_offsets)
            prev_centers.append((center, tier))

            fig.add_annotation(
                x=center,
                y=1.0,
                xref="x1",
                yref="y domain",
                yanchor="bottom",
                yshift=4 + y_offsets[tier],
                text=label,
                showarrow=False,
                font=dict(size=10, color=color),
            )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=_BG,
        paper_bgcolor=_BG,
        font=dict(color=_TEXT, size=12),
        height=750,
        margin=dict(t=120, b=60, l=85, r=40),
        title=dict(
            text="Spectra Overlay",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=18, color=_TEXT),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.09,
            xanchor="center",
            x=0.5,
            font=dict(color=_TEXT, size=12),
        ),
        showlegend=True,
    )

    for ann in fig.layout.annotations:
        ann.font = dict(color=_TEXT, size=14)

    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
