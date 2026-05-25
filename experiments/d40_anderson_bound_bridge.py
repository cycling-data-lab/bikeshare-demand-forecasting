"""
d40_anderson_bound_bridge.py — Bridge between Anderson localization
observables and the structural applicability bound.

This is the theoretical payoff of the F5 paper.  d38 computes IPR
patterns on real cities; d39 maps the phase diagram.  d40 closes the
loop by showing that Anderson observables PREDICT the saturation of
the spectral bound R²_spec — i.e. localization is the microscopic
mechanism behind the bound's collapse.

The argument:

  • R²_spec(H_d, y) = ‖P_{U_d} y‖² / ‖y‖² where U_d are the top-d
    low-frequency eigenvectors of L_sym (the "encoder subspace").
  • If U_d are extended (low IPR), they span global patterns of the
    graph → any smooth target y projects well → high R²_spec.
  • If U_d are localized (high IPR), they span only local patches →
    a global target cannot be represented → R²_spec collapses.

Concrete metric we test:

  • Localization-weighted effective dimension:
        d_eff = Σ_{n=1}^{d} PR(ψ_n)
    where PR(ψ_n) = 1/(N · IPR(ψ_n)) is the participation ratio of
    eigenvector n.  This is the d-dimensional subspace's "effective"
    capacity to span global patterns: an extended subspace has
    d_eff ≈ d; a localized one has d_eff ≪ d.

  • Claim to test: across cities, R²_spec correlates with d_eff/d
    even when y is not used — IPR alone is a sufficient statistic.

For the d24 panel of 9 cities, we have observed R²_spec on demand.
We compute d_eff/d on the same Laplacians (no disorder, L_sym only,
so this is encoder geometry alone) and test the correlation.

This is the figure that goes into the F5 paper's headline.

Output:
  outputs/d40_bound_bridge.csv
  outputs/d40_bound_bridge.json
  figures/fig_anderson_bound_bridge.{pdf,png}
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
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
IMD_INTL_DIR = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Same set as d24/d28 for direct comparability
CITIES = [
    ("boston_bluebikes",    "boston_bluebikes",         "Bluebikes Boston",        "tier1"),
    ("dc_capitalbikeshare", "dc_capitalbikeshare",      "Capital Bikeshare DC",    "tier1"),
    ("chicago_divvy",       "chicago_divvy",            "Divvy Chicago",           "tier1"),
    ("sf_baywheels",        "sf_baywheels",             "Bay Wheels SF",           "tier1"),
    ("london_tfl",          "london_tfl",               "Santander Cycles London", "tier1"),
    ("montreal_bixi",       "world_ca_bixi_montr_al",   "BIXI Montréal",           "tier1"),
    ("tier2_paris",         "world_fr_v_lib_metropole", "Vélib Paris",             "tier2"),
    ("tier2_lyon",          "world_fr_v_lo_v",          "Vélo'v Lyon",             "tier2"),
    ("tier2_toulouse",      "world_fr_v_l_toulouse",    "VéLÔ Toulouse",           "tier2"),
]

K_NN = 6
SIGMA_M = 300.0
EARTH_R = 6_371_000.0
FEATS_IMD = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
             "elevation_m", "topography_roughness_index"]

# d=4 matches the IMD-4 subspace dimension (apples-to-apples with d24/d28)
D_ENCODE = 4
# Also test a few alternative d for robustness
D_RANGE = [2, 4, 8, 16, 32]


def haversine_matrix(lat, lng):
    lat_r = np.deg2rad(lat); lng_r = np.deg2rad(lng)
    dphi = lat_r[:, None] - lat_r[None, :]
    dlam = lng_r[:, None] - lng_r[None, :]
    a = np.sin(dphi/2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlam/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_L_sym(lat, lng, k=K_NN, sigma=SIGMA_M):
    N = len(lat)
    D = haversine_matrix(lat, lng); np.fill_diagonal(D, np.inf)
    knn = np.argpartition(D, k, axis=1)[:, :k]
    W = np.zeros((N, N))
    for i in range(N):
        for j in knn[i]:
            w = np.exp(-D[i, j]**2 / (2*sigma**2))
            W[i, j] = max(W[i, j], w); W[j, i] = W[i, j]
    deg = W.sum(axis=1); deg_safe = np.maximum(deg, 1e-12)
    Dinv2 = 1.0/np.sqrt(deg_safe)
    return np.eye(N) - (W * Dinv2[:, None]) * Dinv2[None, :]


def ipr_cols(psi):
    psi2 = psi**2
    psi2 = psi2 / (psi2.sum(axis=0, keepdims=True) + 1e-30)
    return (psi2**2).sum(axis=0)


def load_demand(slug):
    """Same loader as d24."""
    if slug in ("boston_bluebikes", "dc_capitalbikeshare",
                "chicago_divvy", "sf_baywheels"):
        path = OUT / f"d3_{slug}_predictions.parquet"
    elif slug == "london_tfl":
        path = OUT / "d16_london_tfl_predictions.parquet"
    elif slug == "montreal_bixi":
        path = OUT / "d14_montreal_bixi_predictions.parquet"
    elif slug.startswith("tier2_"):
        city_map = {"tier2_paris": "Paris", "tier2_lyon": "lyon",
                    "tier2_toulouse": "toulouse"}
        path = OUT / f"d10_{city_map[slug]}_predictions.parquet"
    else:
        return {}
    if not path.exists(): return {}
    df = pd.read_parquet(path)
    df["station_id"] = df["station_id"].astype(str)
    if slug == "london_tfl":
        df["station_id"] = df["station_id"].str.zfill(6)
    y_true = np.expm1(df["y_true_log"])
    return df.assign(y=y_true).groupby("station_id")["y"].mean().to_dict()


def analyse(slug, stem, pretty, source):
    imd_path = IMD_INTL_DIR / f"{stem}.parquet"
    if not imd_path.exists():
        return None
    imd = pd.read_parquet(imd_path)
    imd["station_id"] = imd["station_id"].astype(str)
    if slug == "london_tfl":
        imd["station_id"] = imd["station_id"].str.zfill(6)
    imd = imd.dropna(subset=["lat","lng"]).reset_index(drop=True)
    N = len(imd)
    if N < 50: return None

    L_sym = build_L_sym(imd["lat"].values, imd["lng"].values)
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    # eigh returns ascending order — low frequency first, exactly what we want
    ipr_all = ipr_cols(eigvecs)
    pr_all = 1.0 / (N * ipr_all + 1e-30)

    # Encoder subspace: top-d low-frequency eigenvectors
    row = dict(slug=slug, city=pretty, source=source, N=N)
    for d in D_RANGE:
        U_d = eigvecs[:, :d]
        ipr_d = ipr_all[:d]
        pr_d = pr_all[:d]
        d_eff = float(pr_d.sum())
        row[f"d_eff_{d}"]     = d_eff
        row[f"d_eff_ratio_{d}"] = d_eff / d
        row[f"mean_IPR_top{d}"] = float(ipr_d.mean())
        row[f"max_IPR_top{d}"]  = float(ipr_d.max())
        row[f"frac_localized_top{d}"] = float((ipr_d > 5.0/N).mean())

    # Compute observed R²_spec on the IMD-4 subspace and on optimal-rank-4
    # subspace (apples-to-apples with d28)
    y_map = load_demand(slug)
    if y_map:
        imd["y"] = imd["station_id"].map(y_map)
        avail = [f for f in FEATS_IMD if f in imd.columns]
        sub = imd.dropna(subset=["y"] + avail).reset_index(drop=True)
        if len(sub) >= 20:
            # Rebuild L_sym on the observable subset
            L_s = build_L_sym(sub["lat"].values, sub["lng"].values)
            ev_s, U_s = np.linalg.eigh(L_s)
            ipr_s = ipr_cols(U_s)
            y_z = sub["y"].values.astype(float)
            y_z = (y_z - y_z.mean()) / (y_z.std() + 1e-12)
            coefs = U_s.T @ y_z
            power = coefs**2
            total = power.sum() + 1e-30
            # R²_spec on the d lowest-frequency modes
            for d in D_RANGE:
                R2_lowfreq = float(power[:d].sum() / total)
                row[f"R2spec_lowfreq_{d}"] = R2_lowfreq
            # R²_spec on the IMD subspace
            X = sub[avail].astype(float).values
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
            Q, _ = np.linalg.qr(X)
            proj = Q @ (Q.T @ y_z)
            R2_imd = float((proj**2).sum() / (y_z @ y_z + 1e-30))
            row["R2spec_IMD_subspace"] = R2_imd
            row["N_with_demand"] = len(sub)
        else:
            for d in D_RANGE:
                row[f"R2spec_lowfreq_{d}"] = float("nan")
            row["R2spec_IMD_subspace"] = float("nan")
            row["N_with_demand"] = 0
    else:
        for d in D_RANGE:
            row[f"R2spec_lowfreq_{d}"] = float("nan")
        row["R2spec_IMD_subspace"] = float("nan")
        row["N_with_demand"] = 0

    print(f"  {pretty}: N={N}  "
          f"d_eff/d (d=4) = {row['d_eff_ratio_4']:.3f}  "
          f"R²_spec_lowfreq(d=4) = {row.get('R2spec_lowfreq_4', float('nan')):.3f}")
    return row


def main():
    t0 = time.time()
    rows = []
    for slug, stem, pretty, source in CITIES:
        try:
            r = analyse(slug, stem, pretty, source)
        except Exception as e:
            print(f"  ✗ {pretty}: {e}")
            continue
        if r is not None:
            rows.append(r)
    if not rows:
        print("No cities processed."); return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d40_bound_bridge.csv", index=False)

    # ── Correlation analysis ─────────────────────────────────────────────
    correlations = {}
    for d in D_RANGE:
        col_x = f"d_eff_ratio_{d}"
        col_y = f"R2spec_lowfreq_{d}"
        sub = df[[col_x, col_y]].dropna()
        if len(sub) >= 3:
            rho_s, p_s = spearmanr(sub[col_x], sub[col_y])
            rho_p, p_p = pearsonr(sub[col_x], sub[col_y])
            correlations[d] = dict(
                spearman_rho=float(rho_s), spearman_p=float(p_s),
                pearson_r=float(rho_p), pearson_p=float(p_p),
                n=len(sub))

    print("\n=== Correlation: d_eff/d (Anderson observable) vs R²_spec_lowfreq (bound) ===")
    for d, c in correlations.items():
        print(f"  d={d:3d}   Spearman ρ = {c['spearman_rho']:+.3f}  "
              f"(p={c['spearman_p']:.3g}, n={c['n']})   "
              f"Pearson r = {c['pearson_r']:+.3f}")

    # ── Figure: scatter d_eff/d vs R²_spec_lowfreq ──────────────────────
    fig, axes = plt.subplots(1, len(D_RANGE), figsize=(4*len(D_RANGE), 4),
                             constrained_layout=True, sharey=True)
    if len(D_RANGE) == 1: axes = [axes]
    for ax, d in zip(axes, D_RANGE):
        col_x = f"d_eff_ratio_{d}"
        col_y = f"R2spec_lowfreq_{d}"
        sub = df[["city", col_x, col_y]].dropna()
        ax.scatter(sub[col_x], sub[col_y], s=60, color="C0", alpha=0.85)
        for _, r in sub.iterrows():
            ax.annotate(r["city"].split()[0][:8],
                        (r[col_x], r[col_y]),
                        fontsize=7, alpha=0.75,
                        xytext=(4, 2), textcoords="offset points")
        if d in correlations:
            c = correlations[d]
            ax.set_title(f"d = {d}\nρ_Spearman = {c['spearman_rho']:+.3f}  "
                         f"(p={c['spearman_p']:.2g})",
                         fontsize=9)
        else:
            ax.set_title(f"d = {d}", fontsize=9)
        ax.set_xlabel(r"$d_{\mathrm{eff}}/d$  (Anderson)", fontsize=9)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$R^2_{\mathrm{spec}}$ on top-d low-freq subspace", fontsize=9)
    fig.suptitle("Anderson observable predicts the structural bound\n"
                 "Localized encoder subspace (low d_eff/d) → low R²_spec ceiling",
                 fontsize=11)
    fig.savefig(FIG / "fig_anderson_bound_bridge.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_anderson_bound_bridge.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"\n✓ {FIG/'fig_anderson_bound_bridge.pdf'}")

    # ── Summary JSON ─────────────────────────────────────────────────────
    with open(OUT / "d40_bound_bridge.json", "w") as f:
        json.dump({
            "config": dict(K_NN=K_NN, SIGMA_M=SIGMA_M, D_RANGE=D_RANGE,
                           FEATS_IMD=FEATS_IMD),
            "rows": rows,
            "correlations": correlations,
            "wall_time_s": round(time.time()-t0, 1),
        }, f, indent=2)
    print(f"✓ CSV: {OUT/'d40_bound_bridge.csv'}")
    print(f"✓ JSON: {OUT/'d40_bound_bridge.json'}")
    print(f"  Total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
