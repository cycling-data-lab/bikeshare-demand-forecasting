"""
d41_anderson_placebo.py — Placebo / null-model tests for the d40b
mode-by-mode bridge.

The headline result of d40b is ρ(IPR, c_n) = −0.30 at p = 2.3·10⁻¹⁴⁴
on n = 6 923 (city, mode) pairs.  Two distinct critiques threaten
the Anderson interpretation:

  (A) Centrality artefact.  Central nodes have higher local density →
      tend to be in extended modes (low IPR).  Central nodes also
      tend to concentrate demand.  If IPR and demand both proxy
      "distance to centre", the correlation is a centrality artefact
      not an Anderson signal.  Test: permute demand among nodes of
      similar degree (bin by degree quantile), recompute ρ.  If the
      permuted ρ ≈ 0, the real signal is informative; if the permuted
      ρ ≈ −0.30, the signal is centrality.

  (B) Disorder-vs-topology critique.  d40b uses the BARE Laplacian
      L_sym (W = 0), so the headline does NOT actually test the
      Anderson Hamiltonian — it tests pure graph topology.  If adding
      IMD on-site potential (W = 1) does not change ρ, then the
      Anderson framing is misleading: the bound's collapse is
      explained by topology alone, not by the disorder field.  Test:
      compute ρ for eigenvectors of H(W = 0) and H(W = 1) on the same
      cities.  Compare.

We pool 9 cities × 2 disorder levels × 4 null types = 72 conditions
and report the matrix.

Output:
  outputs/d41_placebo.csv             (one row per condition)
  outputs/d41_placebo.json            (summary + verdict)
  figures/fig_anderson_placebo.{pdf,png}
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
from scipy.stats import spearmanr

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

K_NN = 6
SIGMA_M = 300.0
EARTH_R = 6_371_000.0
FEATS_IMD = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
             "elevation_m", "topography_roughness_index"]

# Disorder levels to compare
W_LEVELS = [0.0, 1.0]

# Number of null-permutation reps
N_PERMS_GLOBAL = 100   # uniform random permutation null
N_PERMS_DEG    = 100   # degree-preserving permutation null
N_DEG_BINS     = 5     # bins of degree quantiles for degree-preserving null

SEED = 42


def haversine_matrix(lat, lng):
    lat_r = np.deg2rad(lat); lng_r = np.deg2rad(lng)
    dphi = lat_r[:, None] - lat_r[None, :]
    dlam = lng_r[:, None] - lng_r[None, :]
    a = np.sin(dphi/2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlam/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_graph(lat, lng, k=K_NN, sigma=SIGMA_M):
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


def load_demand(slug):
    if slug in ("boston_bluebikes","dc_capitalbikeshare","chicago_divvy","sf_baywheels"):
        path = OUT / f"d3_{slug}_predictions.parquet"
    elif slug == "london_tfl":
        path = OUT / "d16_london_tfl_predictions.parquet"
    elif slug == "montreal_bixi":
        path = OUT / "d14_montreal_bixi_predictions.parquet"
    elif slug.startswith("tier2_"):
        city_map = {"tier2_paris":"Paris","tier2_lyon":"lyon","tier2_toulouse":"toulouse"}
        path = OUT / f"d10_{city_map[slug]}_predictions.parquet"
    else:
        return {}
    if not path.exists(): return {}
    df = pd.read_parquet(path)
    df["station_id"] = df["station_id"].astype(str)
    if slug == "london_tfl":
        df["station_id"] = df["station_id"].str.zfill(6)
    return df.assign(y=np.expm1(df["y_true_log"])).groupby("station_id")["y"].mean().to_dict()


def degree_preserving_permutation(y, deg, n_bins, rng):
    """Permute y within bins of degree-quantile.  Preserves the
    marginal distribution of demand within each connectivity stratum.
    Returns the permuted y."""
    edges = np.quantile(deg, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bin_idx = np.digitize(deg, edges[1:-1], right=False)
    y_perm = y.copy()
    for b in range(n_bins):
        mask = (bin_idx == b)
        if mask.sum() <= 1: continue
        y_perm[mask] = rng.permutation(y[mask])
    return y_perm


def compute_rho_pooled(eigvecs_per_city, y_per_city):
    """Compute pooled Spearman ρ across all (city, mode) pairs."""
    all_ipr = []
    all_cn = []
    for slug in eigvecs_per_city:
        evecs = eigvecs_per_city[slug]
        y = y_per_city[slug]
        N = evecs.shape[0]
        if y is None: continue
        y_z = (y - y.mean()) / (y.std() + 1e-12)
        ipr_vals = ipr_cols(evecs)
        coefs = evecs.T @ y_z
        c_n = (coefs**2) / (y_z @ y_z + 1e-30)
        all_ipr.append(ipr_vals)
        all_cn.append(c_n)
    if not all_ipr:
        return float("nan"), float("nan"), 0
    ipr_arr = np.concatenate(all_ipr)
    cn_arr = np.concatenate(all_cn)
    rho, p = spearmanr(ipr_arr, cn_arr)
    return float(rho), float(p), len(ipr_arr)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # ── (1) Build graphs and load demand for each city ──────────────────
    print("=== Building 9-city panel ===")
    panel = {}  # slug -> dict(L_sym, eps, deg, y)
    for slug, stem, pretty in CITIES:
        imd_path = IMD_INTL_DIR / f"{stem}.parquet"
        if not imd_path.exists():
            print(f"  ✗ {pretty}: no IMD"); continue
        imd = pd.read_parquet(imd_path)
        imd["station_id"] = imd["station_id"].astype(str)
        if slug == "london_tfl":
            imd["station_id"] = imd["station_id"].str.zfill(6)
        imd = imd.dropna(subset=["lat","lng"]).reset_index(drop=True)

        y_map = load_demand(slug)
        if not y_map:
            print(f"  ✗ {pretty}: no demand"); continue
        imd["y"] = imd["station_id"].map(y_map)
        sub = imd.dropna(subset=["y"]).reset_index(drop=True)
        N = len(sub)
        if N < 50: continue

        _, deg, L_sym = build_graph(sub["lat"].values, sub["lng"].values)
        eps = composite_potential(sub)

        panel[slug] = dict(
            pretty=pretty, N=N, L_sym=L_sym, eps=eps, deg=deg,
            y=sub["y"].values.astype(float),
        )
        print(f"  ✓ {pretty}: N={N}, mean_deg={deg.mean():.2f}")

    # ── (2) For each W in {0, 1}, diagonalise and store eigenvectors ────
    print("\n=== Diagonalising for W = 0 and W = 1 ===")
    eigvecs_per_W = {0.0: {}, 1.0: {}}
    for slug, d in panel.items():
        L = d["L_sym"]
        eps = d["eps"]
        for W in W_LEVELS:
            H = L + W * np.diag(eps) if W > 0 else L
            _, evecs = np.linalg.eigh(H)
            eigvecs_per_W[W][slug] = evecs

    # ── (3) Compute reference (REAL demand) ρ at each W ─────────────────
    print("\n=== REAL demand correlations ===")
    real_results = {}
    for W in W_LEVELS:
        y_per_city = {slug: panel[slug]["y"] for slug in panel}
        rho, p, n = compute_rho_pooled(eigvecs_per_W[W], y_per_city)
        real_results[W] = dict(rho=rho, p=p, n=n)
        print(f"  W = {W}:  ρ = {rho:+.4f}   p = {p:.3g}   n = {n}")

    # ── (4) NULL models on demand: uniform permutation ──────────────────
    print(f"\n=== NULL: uniform random permutation ({N_PERMS_GLOBAL} reps) ===")
    null_global = {0.0: [], 1.0: []}
    for rep in range(N_PERMS_GLOBAL):
        y_perm = {slug: rng.permutation(panel[slug]["y"]) for slug in panel}
        for W in W_LEVELS:
            rho, _, _ = compute_rho_pooled(eigvecs_per_W[W], y_perm)
            null_global[W].append(rho)
    for W in W_LEVELS:
        arr = np.array(null_global[W])
        print(f"  W = {W}:  null ρ mean = {arr.mean():+.4f}   std = {arr.std():.4f}   "
              f"95% CI = [{np.percentile(arr,2.5):+.4f}, {np.percentile(arr,97.5):+.4f}]")

    # ── (5) NULL models: degree-preserving permutation ──────────────────
    print(f"\n=== NULL: degree-preserving permutation "
          f"({N_PERMS_DEG} reps, {N_DEG_BINS} bins) ===")
    null_deg = {0.0: [], 1.0: []}
    for rep in range(N_PERMS_DEG):
        y_perm = {slug: degree_preserving_permutation(
                       panel[slug]["y"], panel[slug]["deg"], N_DEG_BINS, rng)
                  for slug in panel}
        for W in W_LEVELS:
            rho, _, _ = compute_rho_pooled(eigvecs_per_W[W], y_perm)
            null_deg[W].append(rho)
    for W in W_LEVELS:
        arr = np.array(null_deg[W])
        print(f"  W = {W}:  null ρ mean = {arr.mean():+.4f}   std = {arr.std():.4f}   "
              f"95% CI = [{np.percentile(arr,2.5):+.4f}, {np.percentile(arr,97.5):+.4f}]")

    # ── (6) Synthesise rows for CSV ─────────────────────────────────────
    rows = []
    for W in W_LEVELS:
        rows.append(dict(condition="REAL_demand", W=W,
                         rho=real_results[W]["rho"],
                         p=real_results[W]["p"],
                         n_modes=real_results[W]["n"]))
        for label, arr in [("NULL_uniform", null_global[W]),
                           ("NULL_degree_preserving", null_deg[W])]:
            a = np.array(arr)
            rows.append(dict(
                condition=label, W=W,
                rho=float(a.mean()), p=float("nan"),
                rho_std=float(a.std()),
                rho_q025=float(np.percentile(a, 2.5)),
                rho_q975=float(np.percentile(a, 97.5)),
                n_reps=len(a)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d41_placebo.csv", index=False)

    # ── (7) Verdicts ────────────────────────────────────────────────────
    print("\n=== VERDICT ===")
    verdict = {}
    for W in W_LEVELS:
        real_rho = real_results[W]["rho"]
        null_dp = np.array(null_deg[W])
        null_un = np.array(null_global[W])
        # One-sided test: is real_rho more negative than the null?
        p_emp_dp = float((null_dp <= real_rho).mean())
        p_emp_un = float((null_un <= real_rho).mean())
        survives_uniform = real_rho < np.percentile(null_un, 2.5)
        survives_degree = real_rho < np.percentile(null_dp, 2.5)
        verdict[W] = dict(
            real_rho=real_rho,
            p_empirical_vs_uniform=p_emp_un,
            p_empirical_vs_degree=p_emp_dp,
            survives_uniform_null=bool(survives_uniform),
            survives_degree_null=bool(survives_degree),
        )
        print(f"\n  W = {W}:")
        print(f"    Real ρ = {real_rho:+.4f}")
        print(f"    vs uniform null  ({null_un.mean():+.4f} ± {null_un.std():.4f}): "
              f"empirical p = {p_emp_un:.3g}  → {'SURVIVES' if survives_uniform else 'FAILS'}")
        print(f"    vs degree-preserving null  ({null_dp.mean():+.4f} ± {null_dp.std():.4f}): "
              f"empirical p = {p_emp_dp:.3g}  → {'SURVIVES' if survives_degree else 'FAILS'}")

    # ── (8) Disorder check ──────────────────────────────────────────────
    delta = real_results[1.0]["rho"] - real_results[0.0]["rho"]
    print(f"\n  Δρ = ρ(W=1) − ρ(W=0) = {delta:+.4f}")
    if delta < -0.05:
        disorder_verdict = "IMD on-site disorder STRENGTHENS the signal (Anderson framing supported)"
    elif delta > +0.05:
        disorder_verdict = "IMD on-site disorder WEAKENS the signal (Anderson framing weakened)"
    else:
        disorder_verdict = "IMD on-site disorder has negligible effect on ρ (signal is topological, NOT Anderson per se)"
    print(f"  → {disorder_verdict}")
    verdict["delta_rho_W1_minus_W0"] = float(delta)
    verdict["disorder_verdict"] = disorder_verdict

    # ── (9) Figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax_idx, W in enumerate(W_LEVELS):
        ax = axes[ax_idx]
        null_un = np.array(null_global[W])
        null_dp = np.array(null_deg[W])
        ax.hist(null_un, bins=25, color="lightgrey", alpha=0.75,
                label=f"Uniform null  ({N_PERMS_GLOBAL} reps)", density=True)
        ax.hist(null_dp, bins=25, color="C2", alpha=0.55,
                label=f"Degree-preserving null  ({N_PERMS_DEG} reps, {N_DEG_BINS} bins)",
                density=True)
        ax.axvline(real_results[W]["rho"], color="red", linewidth=2.0,
                   label=f"Real ρ = {real_results[W]['rho']:+.3f}")
        ax.set_xlabel(r"$\rho$  (Spearman, pooled (city, mode) pairs)")
        ax.set_ylabel("density")
        ax.set_title(f"W = {W}  ({'bare Laplacian' if W==0 else 'IMD on-site disorder'})")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
        ax.grid(True, alpha=0.3)
    fig.suptitle("d41 placebo test: real ρ vs degree-preserving and uniform nulls\n"
                 "Red line outside the null distributions ⇒ signal is informative, not a centrality artefact",
                 fontsize=11)
    fig.savefig(FIG / "fig_anderson_placebo.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_anderson_placebo.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"\n✓ {FIG/'fig_anderson_placebo.pdf'}")

    # ── (10) Save JSON ──────────────────────────────────────────────────
    with open(OUT / "d41_placebo.json", "w") as f:
        json.dump({
            "config": dict(K_NN=K_NN, SIGMA_M=SIGMA_M, FEATS_IMD=FEATS_IMD,
                           W_LEVELS=W_LEVELS, N_PERMS_GLOBAL=N_PERMS_GLOBAL,
                           N_PERMS_DEG=N_PERMS_DEG, N_DEG_BINS=N_DEG_BINS,
                           SEED=SEED),
            "real_results": {str(W): real_results[W] for W in W_LEVELS},
            "null_global_stats": {str(W): dict(
                mean=float(np.mean(null_global[W])),
                std=float(np.std(null_global[W])),
                q025=float(np.percentile(null_global[W], 2.5)),
                q975=float(np.percentile(null_global[W], 97.5))) for W in W_LEVELS},
            "null_degree_stats": {str(W): dict(
                mean=float(np.mean(null_deg[W])),
                std=float(np.std(null_deg[W])),
                q025=float(np.percentile(null_deg[W], 2.5)),
                q975=float(np.percentile(null_deg[W], 97.5))) for W in W_LEVELS},
            "verdict": verdict,
            "wall_time_s": round(time.time()-t0, 1),
        }, f, indent=2)
    print(f"  Total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
