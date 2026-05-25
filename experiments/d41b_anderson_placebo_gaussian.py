"""
d41b_anderson_placebo_gaussian.py — Refinement of d41 with synthetic
Gaussian disorder.

d41 showed that adding the IMD on-site potential at W=1 DESTROYS the
mode-by-mode bridge:  ρ drops from −0.30 (W=0) to −0.07 (W=1 with IMD),
which is statistically indistinguishable from the degree-preserving
null.

Diagnosis: the IMD composite ε_IMD is highly correlated with the
demand signal y (it is precisely calibrated to predict demand
elsewhere in the project).  Injecting ε_IMD into the Hamiltonian as
"disorder" violates the Anderson assumption that ε be independent of
y.  The IMD potential is not disorder; it is signal travesti.

The clean test is to compare:
  • W=1 with ε_IMD (correlated with y) — already in d41
  • W=1 with ε_Gauss (independent Gaussian noise) — this script

If the Anderson framing is correct for genuinely uncorrelated disorder,
then ρ(W=1, Gauss) should match or exceed ρ(W=0).  If both ε_IMD and
ε_Gauss destroy ρ, then Anderson localization plays no role in
demand prediction; the result is purely about bare-Laplacian topology.

Output:
  outputs/d41b_placebo_gaussian.csv
  outputs/d41b_placebo_gaussian.json
  figures/fig_anderson_placebo_gaussian.{pdf,png}
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

# Conditions to test
CONDITIONS = [
    ("W0_bare",         0.0, "none"),       # baseline: bare Laplacian (no potential)
    ("W1_IMD",          1.0, "IMD"),        # IMD-correlated potential (= d41)
    ("W1_Gauss_seed0",  1.0, "Gauss_0"),    # uncorrelated Gaussian, draw #0
    ("W1_Gauss_seed1",  1.0, "Gauss_1"),    # uncorrelated Gaussian, draw #1
    ("W1_Gauss_seed2",  1.0, "Gauss_2"),    # uncorrelated Gaussian, draw #2
    ("W3_Gauss_seed0",  3.0, "Gauss_0"),    # stronger Gaussian disorder
    ("W3_IMD",          3.0, "IMD"),        # stronger IMD disorder
]

SEED = 42


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


def composite_potential(imd_df):
    avail = [f for f in FEATS_IMD if f in imd_df.columns]
    if not avail: return np.zeros(len(imd_df))
    X = imd_df[avail].astype(float).values
    X = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-12)
    X = np.nan_to_num(X, nan=0.0)
    eps = X.mean(axis=1)
    return (eps - eps.mean()) / (eps.std() + 1e-12)


def gaussian_potential(N, rng):
    """Draw standard Gaussian, centred and unit-variance normalised."""
    eps = rng.standard_normal(N)
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


def diagonalise_hamiltonian(L_sym, eps, W):
    H = L_sym + W * np.diag(eps) if W > 0 else L_sym
    _, eigvecs = np.linalg.eigh(H)
    return eigvecs


def pooled_rho(eigvecs_per_city, y_per_city):
    """Pooled Spearman ρ across (city, mode) pairs."""
    all_ipr = []; all_cn = []
    for slug, evecs in eigvecs_per_city.items():
        y = y_per_city[slug]; N = evecs.shape[0]
        y_z = (y - y.mean()) / (y.std() + 1e-12)
        ipr_vals = ipr_cols(evecs)
        coefs = evecs.T @ y_z
        c_n = (coefs**2) / (y_z @ y_z + 1e-30)
        all_ipr.append(ipr_vals); all_cn.append(c_n)
    ipr_arr = np.concatenate(all_ipr); cn_arr = np.concatenate(all_cn)
    rho, p = spearmanr(ipr_arr, cn_arr)
    return float(rho), float(p), len(ipr_arr)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # ── Build panel ──────────────────────────────────────────────────────
    print("=== Building 9-city panel ===")
    panel = {}
    for slug, stem, pretty in CITIES:
        imd_path = IMD_INTL_DIR / f"{stem}.parquet"
        if not imd_path.exists():
            print(f"  ✗ {pretty}"); continue
        imd = pd.read_parquet(imd_path)
        imd["station_id"] = imd["station_id"].astype(str)
        if slug == "london_tfl":
            imd["station_id"] = imd["station_id"].str.zfill(6)
        imd = imd.dropna(subset=["lat","lng"]).reset_index(drop=True)
        y_map = load_demand(slug)
        if not y_map: print(f"  ✗ {pretty}: no demand"); continue
        imd["y"] = imd["station_id"].map(y_map)
        sub = imd.dropna(subset=["y"]).reset_index(drop=True)
        if len(sub) < 50: continue
        L_sym = build_L_sym(sub["lat"].values, sub["lng"].values)
        eps_imd = composite_potential(sub)
        eps_gauss = {i: gaussian_potential(len(sub), np.random.default_rng(SEED + 1000*i + hash(slug) % 100000))
                     for i in range(3)}
        panel[slug] = dict(pretty=pretty, N=len(sub),
                           L_sym=L_sym, eps_imd=eps_imd,
                           eps_gauss=eps_gauss,
                           y=sub["y"].values.astype(float))
        print(f"  ✓ {pretty}: N={len(sub)}")

    # ── Diagonalise + correlate per condition ────────────────────────────
    print("\n=== Real ρ per condition ===")
    rows = []
    y_per_city = {slug: panel[slug]["y"] for slug in panel}
    for label, W, pot_type in CONDITIONS:
        eigvecs_per_city = {}
        for slug, d in panel.items():
            if pot_type == "none":
                eps = np.zeros(d["N"])
            elif pot_type == "IMD":
                eps = d["eps_imd"]
            elif pot_type.startswith("Gauss_"):
                seed_idx = int(pot_type.split("_")[1])
                eps = d["eps_gauss"][seed_idx]
            else:
                continue
            eigvecs_per_city[slug] = diagonalise_hamiltonian(d["L_sym"], eps, W)
        rho, p, n = pooled_rho(eigvecs_per_city, y_per_city)
        rows.append(dict(condition=label, W=W, potential=pot_type,
                         rho=rho, p=p, n_modes=n))
        print(f"  {label:20s}  ρ = {rho:+.4f}   p = {p:.3g}   n = {n}")

    # ── Aggregate: mean Gaussian ρ at W=1 and W=3 ────────────────────────
    g1 = np.array([r["rho"] for r in rows
                   if r["potential"].startswith("Gauss_") and r["W"] == 1.0])
    g3 = np.array([r["rho"] for r in rows
                   if r["potential"].startswith("Gauss_") and r["W"] == 3.0])
    print(f"\n=== Gaussian aggregation ===")
    print(f"  W=1 Gaussian:  ρ mean = {g1.mean():+.4f}  range [{g1.min():+.4f}, {g1.max():+.4f}]  "
          f"({len(g1)} draws)")
    print(f"  W=3 Gaussian:  ρ mean = {g3.mean():+.4f}  range [{g3.min():+.4f}, {g3.max():+.4f}]  "
          f"({len(g3)} draws)")

    # ── Verdict ──────────────────────────────────────────────────────────
    rho_W0 = next(r["rho"] for r in rows if r["condition"] == "W0_bare")
    rho_W1_imd = next(r["rho"] for r in rows if r["condition"] == "W1_IMD")
    rho_W1_gauss_mean = g1.mean()

    print("\n=== VERDICT ===")
    print(f"  ρ(W=0, bare)            = {rho_W0:+.4f}")
    print(f"  ρ(W=1, IMD-correlated)  = {rho_W1_imd:+.4f}   Δ vs bare = {rho_W1_imd-rho_W0:+.4f}")
    print(f"  ρ(W=1, Gauss mean)      = {rho_W1_gauss_mean:+.4f}   Δ vs bare = {rho_W1_gauss_mean-rho_W0:+.4f}")
    print(f"  ρ(W=3, Gauss mean)      = {g3.mean():+.4f}   Δ vs bare = {g3.mean()-rho_W0:+.4f}")

    preservation_gauss = abs(rho_W1_gauss_mean - rho_W0)
    destruction_imd = abs(rho_W1_imd - rho_W0)
    if abs(rho_W1_gauss_mean) > 0.9 * abs(rho_W0):
        verdict = ("Gaussian disorder PRESERVES the signal whereas IMD DESTROYS it.\n"
                   "  → Anderson framing is recoverable for genuinely uncorrelated disorder.\n"
                   "  → The d41 destruction was an artefact of IMD-demand correlation.")
    elif abs(rho_W1_gauss_mean) < 0.3 * abs(rho_W0):
        verdict = ("Gaussian disorder ALSO destroys the signal.\n"
                   "  → Anderson framing is empirically FALSE.\n"
                   "  → The bare-Laplacian result is purely topological; reframe paper.")
    else:
        verdict = ("Gaussian disorder PARTIALLY preserves the signal.\n"
                   "  → Mixed outcome; further analysis required.")
    print(f"\n  {verdict}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d41b_placebo_gaussian.csv", index=False)

    # ── Figure: barplot of ρ per condition ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    labels = [r["condition"] for r in rows]
    rhos = [r["rho"] for r in rows]
    colors = ["C0" if r["potential"] == "none"
              else "C3" if r["potential"] == "IMD"
              else "C2" for r in rows]
    bars = ax.barh(range(len(labels)), rhos, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="black", lw=0.6)
    ax.axvline(rho_W0, color="C0", ls="--", lw=1.0, alpha=0.6, label="W=0 baseline")
    for b, r in zip(bars, rhos):
        ax.text(r - 0.005 if r < 0 else r + 0.005, b.get_y() + b.get_height()/2,
                f"{r:+.3f}", va="center", ha="right" if r < 0 else "left",
                fontsize=9, color="black")
    ax.set_xlabel(r"Pooled Spearman $\rho(\mathrm{IPR}, c_n)$  on $n=6\,923$ city-mode pairs")
    ax.set_title("d41b: pooled ρ per disorder condition.\n"
                 "Anderson predicts: Gaussian (green) should preserve ρ; IMD (red) is signal-correlated.")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.92)
    fig.savefig(FIG / "fig_anderson_placebo_gaussian.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_anderson_placebo_gaussian.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"\n✓ {FIG/'fig_anderson_placebo_gaussian.pdf'}")

    with open(OUT / "d41b_placebo_gaussian.json", "w") as f:
        json.dump({
            "config": dict(K_NN=K_NN, SIGMA_M=SIGMA_M, FEATS_IMD=FEATS_IMD,
                           CONDITIONS=[(c[0], c[1], c[2]) for c in CONDITIONS],
                           SEED=SEED),
            "rows": rows,
            "gauss_W1_mean": float(g1.mean()),
            "gauss_W1_range": [float(g1.min()), float(g1.max())],
            "gauss_W3_mean": float(g3.mean()),
            "verdict": verdict,
            "wall_time_s": round(time.time()-t0, 1),
        }, f, indent=2)
    print(f"  Total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
