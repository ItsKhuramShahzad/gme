#!/usr/bin/env python3
"""
render_surface_html.py
----------------------
Turn a cached (t/U, T) thermal-GME surface PKL into an INTERACTIVE, rotatable
3D plot saved as ONE self-contained HTML file. Open it in any web browser
(double-click, or `explorer.exe outputs/<file>.html` from WSL) and:

    * drag        -> rotate  (each panel rotates independently in grid mode)
    * scroll      -> zoom
    * hover       -> read the exact (t/U, T, C_GME) value

Layouts:
    --layout grid      (default) all body-orders n shown at once, 2x3 scenes
    --layout dropdown  one large panel + a dropdown to switch n

Every panel, n=2 included, shows the convex roof C_GME exactly as stored,
matching the PDF/heatmap figures. (For n=2 the convex roof equals the Wootters
concurrence; the Werner form sqrt(1-sqrt(1-C^2)) is NOT a valid GME and is not
plotted.) No recomputation — it reads the cached PKL.

Usage:
    python render_surface_html.py <tTsurface_*.pkl> [-o out.html] [--layout grid|dropdown]
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def _load(pkl_path):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    r = payload["result"]
    si = r.get("system_info", {})
    L, U = si.get("L"), si.get("U")
    mode = si.get("mode", "uniform")
    sp = r.get("spin_preserving", True)
    sector = "Sz=0 sector" if sp else "full Hamiltonian (all Sz)"
    tu = np.asarray(r["t_values"], float) / U          # x  (len n_t)
    T = np.asarray(r["T_values"], float)               # y  (len n_T)
    n_values = list(r["n_values"])
    Cs = r["C_surface"]
    return L, U, mode, sector, tu, T, n_values, Cs


def _label(n):
    return f"n={n} (bipartite)" if n == 2 else f"n={n}"


def _Z(Cs, n):
    # Stored convex roof, untransformed for every n (n=2 included).
    Z = np.asarray(Cs[n], float)                       # (n_t, n_T)
    return Z.T                                         # plotly wants (n_T, n_t)


_HOVER = "t/U=%{x:.3f}<br>T=%{y:.3f}<br>C_GME=%{z:.4f}<extra></extra>"


def render_grid(pkl_path, out_html=None):
    """All n as a 2x3 grid of independently-rotatable 3D scenes."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    L, U, mode, sector, tu, T, n_values, Cs = _load(pkl_path)
    ncols = 3
    nrows = int(np.ceil(len(n_values) / ncols))
    specs = [[{"type": "surface"} for _ in range(ncols)] for _ in range(nrows)]
    fig = make_subplots(
        rows=nrows, cols=ncols, specs=specs,
        subplot_titles=[_label(n) for n in n_values],
        horizontal_spacing=0.02, vertical_spacing=0.06,
    )
    for i, n in enumerate(n_values):
        row, col = i // ncols + 1, i % ncols + 1
        fig.add_trace(
            go.Surface(x=tu, y=T, z=_Z(Cs, n), colorscale="Viridis",
                       showscale=False, hovertemplate=_HOVER),
            row=row, col=col,
        )
    # label every scene's axes (z-axis carries the C_GME scale per panel)
    for i in range(len(n_values)):
        key = "scene" if i == 0 else f"scene{i + 1}"
        fig.layout[key].update(
            xaxis_title="t/U", yaxis_title="T", zaxis_title="C_GME",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
        )
    fig.update_layout(
        title=f"Thermal GME surfaces  C_GME(t/U, T)   "
              f"[L={L}, U={U}, {mode}, {sector}]  —  drag any panel to rotate",
        width=1500, height=520 * nrows, margin=dict(l=0, r=0, t=70, b=0),
    )
    return _write(fig, pkl_path, out_html, "_interactive_grid")


def render_dropdown(pkl_path, out_html=None):
    """One large panel with a dropdown to switch n."""
    import plotly.graph_objects as go

    L, U, mode, sector, tu, T, n_values, Cs = _load(pkl_path)

    def _title(n):
        return (f"Thermal GME surface  C_GME(t/U, T)  —  {_label(n)}"
                f"   [L={L}, U={U}, {mode}, {sector}]")

    traces = [
        go.Surface(x=tu, y=T, z=_Z(Cs, n), colorscale="Viridis",
                   visible=(i == 0), showscale=True,
                   colorbar=dict(title="C_GME"), hovertemplate=_HOVER)
        for i, n in enumerate(n_values)
    ]
    buttons = [
        dict(label=_label(n), method="update",
             args=[{"visible": [j == i for j in range(len(n_values))]},
                   {"title": _title(n)}])
        for i, n in enumerate(n_values)
    ]
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=_title(n_values[0]),
        scene=dict(xaxis_title="t/U", yaxis_title="Temperature T",
                   zaxis_title="C_GME"),
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                          x=0.02, y=0.98, xanchor="left", yanchor="top")],
        width=1000, height=760, margin=dict(l=0, r=0, t=70, b=0),
    )
    return _write(fig, pkl_path, out_html, "_interactive")


def _write(fig, pkl_path, out_html, suffix):
    if out_html is None:
        base = os.path.splitext(os.path.basename(pkl_path))[0]
        out_html = os.path.join("outputs", f"{base}{suffix}.html")
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    return out_html


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Interactive rotatable 3D HTML from a surface PKL.")
    ap.add_argument("pkl", help="path to a tTsurface_*.pkl")
    ap.add_argument("-o", "--output", default=None, help="output .html path")
    ap.add_argument("--layout", choices=["grid", "dropdown"], default="grid",
                    help="grid = all n at once (default); dropdown = one panel")
    a = ap.parse_args()
    fn = render_grid if a.layout == "grid" else render_dropdown
    out = fn(a.pkl, a.output)
    print(f"wrote interactive 3D ({a.layout})  ->  {out}")
    print("Open in a browser to rotate.  From WSL:  explorer.exe " + out)
