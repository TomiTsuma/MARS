"""
Figure production for Phase 2.

One module so every figure shares a visual identity and so the "how to read it"
guidance in the specification maps onto a single named function.

Figure catalogue:
    A   layer x position sensitivity heatmap        (steering/select.plot_heatmap)
    B1  dose-response with Spearman rho             fig_b1_dose
    B2  Pareto frontier  -- THE headline verdict    fig_b2_pareto
    C1  off-target specificity matrix               fig_c1_offtarget
    C2  effective dimensionality (PCA scree)        fig_c2_scree
    D1  layer x timestep efficacy surface           fig_d1_surface
    D2  schedule ablation curves                    fig_d2_schedules
    D3  property-timing correspondence              fig_d3_timing
    E   validity vs chemical sanity divergence      fig_e_trap
    F   cost comparison                             fig_f_cost
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

NAVY = "#1F3864"; ACC = "#2E75B6"; GRN = "#548235"
AMB = "#BF8F00"; RED = "#C55A11"; GREY = "#7F7F7F"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": "#BFBFBF", "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlecolor": NAVY})
    cmap = LinearSegmentedColormap.from_list(
        "tp", ["#FFFFFF", "#DEEBF7", "#2E75B6", "#1F3864"])
    return plt, cmap


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


# ------------------------------------------------------------------- B1 / B2
def fig_b1_dose(alphas, values, ci=None, rho=None, prop="property",
                out_path="figB1.png"):
    plt, _ = _mpl()
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    a = np.asarray(alphas, float); v = np.asarray(values, float)
    ax.axhline(0, color="#CCCCCC", lw=.9); ax.axvline(0, color="#CCCCCC", lw=.9)
    lin = np.polyfit(a, v, 1)
    ax.plot(a, np.polyval(lin, a), "--", color="#AAAAAA", lw=1.3, label="linear fit")
    ax.plot(a, v, "o-", color=ACC, lw=2, ms=5, label=f"observed Δ{prop}")
    if ci is not None:
        ci = np.asarray(ci, float)
        ax.fill_between(a, v - ci, v + ci, color=ACC, alpha=.15)
    ax.set_xlabel("steering coefficient α  (units of residual-stream σ)")
    ax.set_ylabel(f"Δ {prop} vs unconditional")
    ax.set_title("Figure B1 · Dose–response")
    if rho is not None:
        ax.text(.03, .05, f"Spearman ρ = {rho:+.3f}"
                          f"{'  (Tier 1 PASS)' if abs(rho) > .7 else '  (Tier 1 FAIL)'}",
                transform=ax.transAxes, fontsize=9, color=NAVY, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out_path)


def fig_b2_pareto(df, x="delta_property", y="fidelity_cost", alpha_col="alpha",
                  baselines: Optional[List[Dict]] = None, out_path="figB2.png"):
    """The headline artefact. A method is useful if its frontier lies below and
    to the right of the baselines."""
    plt, _ = _mpl()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    d = df.dropna(subset=[x, y]).sort_values(x)
    ax.plot(d[x].abs(), d[y], "o-", color=ACC, lw=2, ms=5, label="activation steering")
    for _, r in d.iterrows():
        ax.annotate(f"α={r[alpha_col]:g}", (abs(r[x]), r[y]),
                    textcoords="offset points", xytext=(5, -9),
                    fontsize=7, color=GREY)
    markers = ["^", "s", "D", "P", "X"]
    for i, b in enumerate(baselines or []):
        ax.scatter([abs(b["x"])], [b["y"]], marker=markers[i % len(markers)],
                   s=90, color=b.get("color", RED), zorder=5, label=b["label"])
    ax.set_xlabel("|Δ property| achieved")
    ax.set_ylabel("fidelity cost  (1 − chemical sanity)")
    ax.set_title("Figure B2 · Pareto frontier")
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    return _save(fig, out_path)


# ------------------------------------------------------------------- C1 / C2
def fig_c1_offtarget(M, out_path="figC1.png", heavy=None):
    """Diagonal dominance means disentangled directions. A dense matrix means a
    shared latent factor — usually molecular size — under several names."""
    plt, cmap = _mpl()
    names = list(M.index)
    cols = [c for c in M.columns if c in names]
    A = M[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    im = ax.imshow(np.abs(A), cmap=cmap, vmin=0,
                   vmax=max(1e-6, np.nanmax(np.abs(A))))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("measured property"); ax.set_ylabel("steered property")
    ax.set_title("Figure C1 · Off-target specificity matrix")
    for i in range(len(names)):
        for j in range(len(cols)):
            if np.isfinite(A[i, j]):
                ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(A[i, j]) > .55 * np.nanmax(np.abs(A)) else "#333",
                        fontweight="bold" if names[i] == cols[j] else "normal")
    fig.colorbar(im, ax=ax, fraction=.04, pad=.02).set_label("normalised |Δ| (σ units)")
    if heavy is not None:
        txt = "  ".join(f"{n}:{h:+.2f}" for n, h in zip(names, heavy))
        ax.text(0, -0.28, f"heavy-atom shift per row — {txt}",
                transform=ax.transAxes, fontsize=7.4, color=RED)
    return _save(fig, out_path)


def fig_c2_scree(series: Dict[str, Sequence[float]], out_path="figC2.png"):
    """Tests the approximate one-dimensionality claim rather than assuming it."""
    plt, _ = _mpl()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    colors = [GRN, ACC, AMB, RED]
    n = max(len(v) for v in series.values())
    w = .8 / max(1, len(series))
    for i, (name, ve) in enumerate(series.items()):
        pcs = np.arange(1, len(ve) + 1)
        ax.bar(pcs + (i - len(series) / 2) * w + w / 2, ve, w,
               color=colors[i % len(colors)], label=name)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel("principal component of per-sample difference vectors")
    ax.set_ylabel("variance explained")
    ax.set_title("Figure C2 · Effective dimensionality")
    ax.axhline(.5, ls=":", color=GREY, lw=1)
    ax.text(n * .55, .52, "PC1 > 0.5 ⇒ approximately 1-D", fontsize=7.6, color=GREY)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out_path)


# ------------------------------------------------------------------- D1..D3
def fig_d1_surface(Z, layer_labels=None, step_labels=None, out_path="figD1.png",
                   title="Figure D1 · Layer × timestep efficacy surface"):
    """The axis with no autoregressive analogue. Read for the JOINT hotspot,
    not the marginals."""
    plt, cmap = _mpl()
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    im = ax.imshow(np.abs(Z), aspect="auto", cmap=cmap, origin="lower")
    ax.set_ylabel("layer ℓ"); ax.set_xlabel("reverse-diffusion step window")
    if layer_labels is not None:
        ax.set_yticks(range(len(layer_labels))); ax.set_yticklabels(layer_labels)
    if step_labels is not None:
        ax.set_xticks(range(len(step_labels))); ax.set_xticklabels(step_labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=.028, pad=.012).set_label("|Δ property|")
    return _save(fig, out_path)


def fig_d2_schedules(curves: Dict[str, Dict], full_value=None,
                     out_path="figD2.png", normalise_by_mass=False):
    """A steep FIRST-κ rise with a flat LAST-κ is the signature of early-step
    dominance. Must be read alongside the fixed-unmasking control."""
    plt, _ = _mpl()
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    style = {"first_k": ("-o", ACC), "last_k": ("-s", RED),
             "every_k": ("-^", AMB), "window": ("-d", GRN)}
    for name, d in curves.items():
        st, col = style.get(name, ("-o", GREY))
        y = np.asarray(d["values"], float)
        if normalise_by_mass and d.get("mass"):
            m = np.asarray(d["mass"], float)
            y = y / np.maximum(m / m.max(), 1e-8)
        ax.plot(d["kappa"], y, st, ms=3.5, color=col, lw=1.8,
                label=name.replace("_", "-").upper())
    if full_value is not None:
        ax.axhline(full_value, ls=":", color=GREY, lw=1)
        ax.text(.02, .93, "full-schedule steering", transform=ax.transAxes,
                fontsize=7.6, color=GREY)
    ax.set_xlabel("intervention steps κ")
    ax.set_ylabel("|Δ property|" + (" per unit mass" if normalise_by_mass else ""))
    ax.set_title("Figure D2 · Schedule ablation"
                 + (" (mass-normalised)" if normalise_by_mass else ""))
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out_path)


def fig_d3_timing(rows: List[Dict], out_path="figD3.png"):
    """Property-timing correspondence. Pre-registered prediction: window
    position tracks the structural level the property depends on."""
    plt, _ = _mpl()
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    rows = sorted(rows, key=lambda r: r["centre"])
    y = np.arange(len(rows))
    colors = [ACC, ACC, GRN, GRN, AMB, RED]
    for i, r in enumerate(rows):
        ax.barh(i, r["width"], left=r["centre"] - r["width"] / 2, height=.55,
                color=colors[i % len(colors)], alpha=.85)
        ax.plot(r["centre"], i, "|", ms=14, color=NAVY, mew=2)
    ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("normalised denoising progress  (0 = start → 1 = complete)")
    ax.set_title("Figure D3 · Property–timing correspondence")
    return _save(fig, out_path)


# ----------------------------------------------------------------------- E, F
def fig_e_trap(alphas, validity, sanity, fcd=None, out_path="figE.png"):
    """Validity flat while chemistry degrades. A study reporting only the green
    curve would conclude, incorrectly, that steering is harmless."""
    plt, _ = _mpl()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    a = np.asarray(alphas, float)
    ax.plot(a, 100 * np.asarray(validity, float), "-o", ms=5, color=GRN, lw=2.2,
            label="SELFIES validity (%)")
    ax.plot(a, 100 * np.asarray(sanity, float), "-s", ms=5, color=RED, lw=2.2,
            label="chemical-sanity pass rate (%)")
    ax.set_xlabel("steering coefficient α"); ax.set_ylabel("percentage")
    ax.set_ylim(0, 108)
    ax.set_title("Figure E · Validity vs chemical sanity")
    if fcd is not None:
        ax2 = ax.twinx()
        ax2.plot(a, fcd, "-^", ms=5, color=ACC, lw=2.2, label="FCD")
        ax2.set_ylabel("FCD", color=ACC); ax2.tick_params(axis="y", colors=ACC)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower left")
    else:
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    return _save(fig, out_path)


def fig_f_cost(rows: List[Dict], out_path="figF.png"):
    """Cost is reported with equal prominence to control: the central practical
    claim is cheapness, and a claim of that kind must be measured."""
    plt, _ = _mpl()
    rows = sorted(rows, key=lambda r: r["setup_seconds"])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(rows))
    ax.barh(y - .2, [max(r["setup_seconds"], 1e-2) for r in rows], .38,
            color=RED, label="setup (training a surrogate / conditional model)")
    ax.barh(y + .2, [max(r["inference_seconds_per_1k"], 1e-2) for r in rows], .38,
            color=ACC, label="inference (per 1k molecules)")
    ax.set_yticks(y); ax.set_yticklabels([r["method"] for r in rows], fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("seconds (log scale)")
    ax.set_title("Figure F · Cost comparison")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out_path)