"""
d42_anderson_robustness.py — Robustness of the Anderson signal to
the graph-construction hyperparameters (k, σ).

The d38 phenomenology was computed at (k, σ) = (6, 300 m).  At this
choice, seven of nine networks already sit in the Poisson regime at
W = 0 (no disorder).  A hostile reviewer can argue that we are
observing topological (Lifshitz-like / sparse-connectivity) localization
rather than Anderson (potential-driven) localization.

The contre-test is to find a (k, σ) regime where the BARE Laplacian
W = 0 sits in the metallic (GOE) regime on most cities, and to then
demonstrate that turning on the IMD potential W = 1 drives the same
networks into the Poisson regime.

This separates:
  • Topological localization : present at W = 0, dominant on sparse graphs
  • Anderson localization    : induced by W > 0, dominant on dense graphs

We sweep:
  • k ∈ {6, 10, 15, 20}
  • σ ∈ {300, 500, 800, 1200} metres
  on all nine cities, at W ∈ {0, 1}.

For each combination we report ⟨r⟩, ⟨IPR⟩, mean degree, and the
phase classification.  We tag combinations that exhibit a CLEAN
Anderson transition (W = 0 in GOE, W = 1 in Poisson, |Δ⟨r⟩| > 0.05).

Output:
  outputs/d42_robustness_sweep.csv
  outputs/d42_robustness_summary.json
  figures/fig_anderson_robustness.{pdf,png}
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
IMD_INTL_DIR = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CITIES = [
    ("boston_bluebikes",    "boston_bluebikes",         "Bluebikes Boston"),
    ("dc_capitalbikeshare", "dc_capitalbikeshare",      "Capital Bikeshare DC"),
    ("chicago_divvy",       "chicago_divvy",            "Divvy Chicago"),
    ("sf_baywheels",        "sf_baywheels",             "Bay Wheels SF"),
    ("london_tfl",          "london_tfl",               "Santander Cycles London"),
    ("montreal_bixi",       "world_ca_bixi_montr_al",   "BIXI Montréal"),
    ("tier2_paris",         "world_fr_v_lib_metropole", "Vélib Paris"),
    ("tier2_lyon",          "world_fr_v_lo_v",          "Vélo'v Lyon"),
    ("tier2_toulouse",      "world_fr_v_l_toulouse",    "VéLÔ Toulouse"),
]

K_GRID = [6, 10, 15, 20]
SIGMA_GRID = [300.0, 500.0, 800.0, 1200.0]
W_LEVELS = [0.0, 1.0]
EARTH_R = 6_371_000.0
FEATS_IMD = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
             "elevation_m", "topography_roughness_index"]

R_GOE = 0.5295
R_POISSON = 0.3863
R_MIDPOINT = 0.5 * (R_GOE + R_POISSON)

# Tolerance for "clean Anderson transition" classification
DELTA_R_THRESHOLD = 0.05


def haversine_matrix(lat, lng):
    lat_r = np.deg2rad(lat); lng_r = np.deg2rad(lng)
    dphi = lat_r[:, None] - lat_r[None, :]
    dlam = lng_r[:, None] - lng_r[None, :]
    a = np.sin(dphi/2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlam/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_L_sym(lat, lng, k, sigma):
    N = len(lat)
    D = haversine_matrix(lat, lng); np.fill_diagonal(D, np.inf)
    k_eff = min(k, N - 1)
    knn = np.argpartition(D, k_eff, axis=1)[:, :k_eff]
    W = np.zeros((N, N))
    for i in range(N):
        for j in knn[i]:
            w = np.exp(-D[i, j]**2 / (2*sigma**2))
            W[i, j] = max(W[i, j], w); W[j, i] = W[i, j]
    deg = W.sum(axis=1); deg_safe = np.maximum(deg, 1e-12)
    Dinv2 = 1.0/np.sqrt(deg_safe)
    L_sym = np.eye(N) - (W * Dinv2[:, None]) * Dinv2[None, :]
    return W, deg, L_sym


def composite_potential(imd_df):
    avail = [f for f in FEATS_IMD if f in imd_df.columns]
    if not avail: return np.zeros(len(imd_df))
    X = imd_df[avail].astype(float).values
    X = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-12)
    X = np.nan_to_num(X, nan=0.0)
    eps = X.mean(axis=1)
    return (eps - eps.mean()) / (eps.std() + 1e-12)


def ipr_cols(psi):
    psi2 = psi**2
    psi2 = psi2 / (psi2.sum(axis=0, keepdims=True) + 1e-30)
    return (psi2**2).sum(axis=0)


def gap_ratio(eigvals):
    s = np.diff(np.sort(eigvals))
    return np.minimum(s[:-1], s[1:]) / (np.maximum(s[:-1], s[1:]) + 1e-30)


def phase_label(r_mean):
    if r_mean > R_GOE - 0.05: return "GOE"
    if r_mean < R_POISSON + 0.05: return "Poisson"
    return "intermediate"


def main():
    t0 = time.time()
    rows = []

    print(f"Sweep: {len(CITIES)} cities × {len(K_GRID)} k × {len(SIGMA_GRID)} σ × "
          f"{len(W_LEVELS)} W = {len(CITIES)*len(K_GRID)*len(SIGMA_GRID)*len(W_LEVELS)} "
          f"diagonalisations.\n")

    for slug, stem, pretty in CITIES:
        imd_path = IMD_INTL_DIR / f"{stem}.parquet"
        if not imd_path.exists():
            print(f"  ✗ {pretty}: no IMD"); continue
        imd = pd.read_parquet(imd_path).dropna(subset=["lat","lng"]).reset_index(drop=True)
        N = len(imd)
        if N < 50: continue
        eps = composite_potential(imd)
        lat = imd["lat"].values; lng = imd["lng"].values

        for k in K_GRID:
            for sigma in SIGMA_GRID:
                _, deg, L_sym = build_L_sym(lat, lng, k, sigma)
                for W in W_LEVELS:
                    H = L_sym + W * np.diag(eps) if W > 0 else L_sym
                    eigvals, eigvecs = np.linalg.eigh(H)
                    ipr_vals = ipr_cols(eigvecs)
                    r_vals = gap_ratio(eigvals)
                    rows.append(dict(
                        city=pretty, slug=slug, N=N, k=k, sigma=sigma, W=W,
                        mean_degree=float(deg.mean()),
                        median_degree=float(np.median(deg)),
                        r_mean=float(r_vals.mean()),
                        ipr_mean=float(ipr_vals.mean()),
                        ipr_max=float(ipr_vals.max()),
                        frac_localized=float((ipr_vals > 5.0/N).mean()),
                        phase=phase_label(float(r_vals.mean())),
                    ))
                print(f"  {pretty:24s}  k={k:3d}  σ={sigma:5.0f}m  "
                      f"deg={deg.mean():4.1f}  "
                      f"<r>(W=0)={rows[-2]['r_mean']:.3f} ({rows[-2]['phase']:12s})  "
                      f"<r>(W=1)={rows[-1]['r_mean']:.3f} ({rows[-1]['phase']:12s})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d42_robustness_sweep.csv", index=False)

    # ── Identify "clean Anderson transition" combinations ────────────────
    # For each (city, k, σ): need r(W=0) > midpoint (GOE-side) AND
    # r(W=1) < midpoint − δ (Poisson-side), with delta = DELTA_R_THRESHOLD.
    pivoted = df.pivot_table(
        index=["city", "slug", "N", "k", "sigma", "mean_degree"],
        columns="W", values=["r_mean", "ipr_mean", "frac_localized"]).reset_index()
    pivoted.columns = ["_".join([str(c) for c in col]).strip("_")
                       for col in pivoted.columns]
    pivoted["delta_r"] = pivoted["r_mean_0.0"] - pivoted["r_mean_1.0"]
    pivoted["w0_metallic"] = pivoted["r_mean_0.0"] > R_MIDPOINT
    pivoted["w1_insulating"] = pivoted["r_mean_1.0"] < R_MIDPOINT - DELTA_R_THRESHOLD/2
    pivoted["clean_anderson"] = pivoted["w0_metallic"] & pivoted["w1_insulating"] & \
                                (pivoted["delta_r"] > DELTA_R_THRESHOLD)
    pivoted.to_csv(OUT / "d42_robustness_pivoted.csv", index=False)

    print("\n=== Clean-Anderson transitions (W=0 in GOE, W=1 in Poisson, Δ⟨r⟩>0.05) ===")
    clean = pivoted[pivoted["clean_anderson"]]
    if len(clean) > 0:
        print(clean[["city","k","sigma","mean_degree","r_mean_0.0","r_mean_1.0","delta_r"]]
              .sort_values("delta_r", ascending=False).to_string(index=False))
        print(f"\n  → {len(clean)} / {len(pivoted)} (k, σ, city) combinations exhibit a CLEAN Anderson transition")
        clean_per_city = clean.groupby("city").size().to_dict()
        print(f"  → Cities with at least one clean transition: {len(clean_per_city)}/9")
        for city, n in sorted(clean_per_city.items(), key=lambda x: -x[1]):
            print(f"     {city}: {n} (k,σ) combinations")
    else:
        print("  NO clean Anderson transitions found.  The phenomenon is NOT separable")
        print("  from topological localization in any tested (k, σ) regime.")

    # ── Figure: heatmap of W=0 phase per city × (k, σ) ──────────────────
    fig, axes = plt.subplots(2, len(CITIES)//3+1, figsize=(20, 8),
                             constrained_layout=True)
    axes = axes.flatten()
    for ax_idx, (slug, stem, pretty) in enumerate(CITIES):
        if ax_idx >= len(axes): break
        ax = axes[ax_idx]
        sub_pivot = pivoted[pivoted["slug"] == slug]
        if len(sub_pivot) == 0:
            ax.set_visible(False); continue
        # Build a heatmap of <r>(W=0) on (k, σ) grid
        heat_w0 = sub_pivot.pivot(index="k", columns="sigma", values="r_mean_0.0")
        heat_w1 = sub_pivot.pivot(index="k", columns="sigma", values="r_mean_1.0")
        # Show W=0 with overlay markers where clean transition exists
        im = ax.imshow(heat_w0.values, cmap="RdYlGn", vmin=0.35, vmax=0.55,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(SIGMA_GRID)))
        ax.set_xticklabels([f"{s:.0f}m" for s in SIGMA_GRID], fontsize=8)
        ax.set_yticks(range(len(K_GRID)))
        ax.set_yticklabels([f"k={k}" for k in K_GRID], fontsize=8)
        ax.set_title(f"{pretty}\n$\\langle r\\rangle(W=0)$", fontsize=9)
        # Annotate cells with (W=0, W=1) values
        for i, k in enumerate(K_GRID):
            for j, sigma in enumerate(SIGMA_GRID):
                v0 = heat_w0.values[i, j]
                v1 = heat_w1.values[i, j]
                clean = sub_pivot[(sub_pivot["k"]==k) & (sub_pivot["sigma"]==sigma)]["clean_anderson"]
                marker = "★" if (len(clean) > 0 and clean.iloc[0]) else ""
                ax.text(j, i, f"{v0:.2f}\n{v1:.2f}{marker}",
                        ha="center", va="center", fontsize=7,
                        color="black")
    # Drop unused axes
    for ax in axes[len(CITIES):]:
        ax.set_visible(False)
    fig.suptitle("(k, σ)-robustness sweep: $\\langle r\\rangle$ at $W=0$ (background colour)\n"
                 "Each cell: top = $\\langle r\\rangle(W=0)$, bottom = $\\langle r\\rangle(W=1)$.  "
                 "★ marks clean Anderson transitions (W=0 GOE-side, W=1 Poisson-side, $\\Delta\\langle r\\rangle$>0.05)",
                 fontsize=10)
    fig.savefig(FIG / "fig_anderson_robustness.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_anderson_robustness.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"\n✓ {FIG/'fig_anderson_robustness.pdf'}")

    # ── Summary JSON ────────────────────────────────────────────────────
    with open(OUT / "d42_robustness_summary.json", "w") as f:
        json.dump({
            "config": dict(K_GRID=K_GRID, SIGMA_GRID=SIGMA_GRID,
                           W_LEVELS=W_LEVELS, FEATS_IMD=FEATS_IMD,
                           R_GOE=R_GOE, R_POISSON=R_POISSON,
                           DELTA_R_THRESHOLD=DELTA_R_THRESHOLD),
            "n_clean_anderson": int(len(clean)),
            "n_total_combos": int(len(pivoted)),
            "cities_with_clean_anderson": list(clean["city"].unique()) if len(clean) else [],
            "wall_time_s": round(time.time()-t0, 1),
        }, f, indent=2)
    print(f"  Total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
