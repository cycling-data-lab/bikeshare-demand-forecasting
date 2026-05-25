"""
d40b_anderson_mode_bridge.py — Mode-by-mode test of the Anderson↔bound
bridge.

The city-level test in d40 gave ρ ≈ +0.4 with p=0.29 at n=9 — suggestive
but not significant.  This script gives the proper statistical test by
pooling all (city, eigenmode) pairs.

For each city and each eigenvector ψ_n of L_sym we record:
  • frequency rank          n / N  (≈ 0 lowest, ≈ 1 highest)
  • IPR(ψ_n)               Σ_i |ψ_n(i)|^4
  • PR(ψ_n) = 1/(N · IPR)
  • normalised demand coefficient   c_n = |<ψ_n, y_z>|² / ||y_z||²

The Anderson hypothesis says: HOLDING n / N FIXED, IPR should be
negatively correlated with c_n — extended modes capture demand
variance, localized modes do not.

We test this two ways:
  (1) Spearman correlation between IPR and c_n within each frequency
      band [n/N ∈ b, b+Δ], then aggregated across bands.
  (2) Conditional means: at each frequency rank percentile, plot
      <c_n>_extended vs <c_n>_localized, where extended = bottom-quartile
      IPR and localized = top-quartile IPR.

If the Anderson bridge holds: extended modes capture more demand
variance than localized modes at the same frequency, across the
spectrum.  This is the microscopic version of the d40 claim.

Output:
  outputs/d40b_mode_bridge.csv     (all (city, mode) pairs)
  outputs/d40b_mode_bridge.json    (summary statistics)
  figures/fig_anderson_mode_bridge.{pdf,png}
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
    if slug in ("boston_bluebikes","dc_capitalbikeshare",
                "chicago_divvy","sf_baywheels"):
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


def analyse(slug, stem, pretty):
    imd_path = IMD_INTL_DIR / f"{stem}.parquet"
    if not imd_path.exists(): return None
    imd = pd.read_parquet(imd_path)
    imd["station_id"] = imd["station_id"].astype(str)
    if slug == "london_tfl":
        imd["station_id"] = imd["station_id"].str.zfill(6)
    imd = imd.dropna(subset=["lat","lng"]).reset_index(drop=True)

    y_map = load_demand(slug)
    if not y_map: return None
    imd["y"] = imd["station_id"].map(y_map)
    sub = imd.dropna(subset=["y"]).reset_index(drop=True)
    N = len(sub)
    if N < 50: return None

    L_sym = build_L_sym(sub["lat"].values, sub["lng"].values)
    eigvals, eigvecs = np.linalg.eigh(L_sym)   # ascending
    ipr_n = ipr_cols(eigvecs)
    pr_n = 1.0 / (N * ipr_n + 1e-30)

    y = sub["y"].values.astype(float)
    y_z = (y - y.mean()) / (y.std() + 1e-12)
    coefs = eigvecs.T @ y_z
    c_n = (coefs**2) / (y_z @ y_z + 1e-30)

    df = pd.DataFrame({
        "city": pretty, "slug": slug, "N": N,
        "mode_idx": np.arange(N),
        "freq_rank": np.arange(N) / N,
        "eigval": eigvals,
        "ipr": ipr_n,
        "pr": pr_n,
        "c_n": c_n,
    })
    print(f"  {pretty}: N={N}  modes  range(IPR)=[{ipr_n.min():.4f}, {ipr_n.max():.4f}]  "
          f"max(c_n)={c_n.max():.4f}")
    return df


def banded_correlation(all_modes, n_bands=10, col_x="ipr", col_y="c_n"):
    """Within-frequency-band Spearman correlation between IPR and c_n.
    Pools modes across cities but stratifies by freq_rank band."""
    all_modes = all_modes.copy()
    all_modes["band"] = pd.cut(all_modes["freq_rank"],
                                bins=np.linspace(0, 1, n_bands+1),
                                labels=False, include_lowest=True)
    rows = []
    for b in range(n_bands):
        sub = all_modes[all_modes["band"] == b]
        if len(sub) < 30: continue
        rho, p = spearmanr(sub[col_x], sub[col_y])
        rows.append(dict(band=b,
                         freq_lo=b/n_bands, freq_hi=(b+1)/n_bands,
                         n=len(sub),
                         spearman_rho=float(rho), spearman_p=float(p),
                         ipr_median=float(sub[col_x].median()),
                         cn_median=float(sub[col_y].median())))
    return pd.DataFrame(rows)


def quartile_contrast(all_modes, n_bands=10):
    """Within each band, contrast mean c_n for bottom-IPR-quartile
    (extended) vs top-IPR-quartile (localized) modes."""
    all_modes = all_modes.copy()
    all_modes["band"] = pd.cut(all_modes["freq_rank"],
                                bins=np.linspace(0, 1, n_bands+1),
                                labels=False, include_lowest=True)
    rows = []
    for b in range(n_bands):
        sub = all_modes[all_modes["band"] == b]
        if len(sub) < 30: continue
        q_lo, q_hi = sub["ipr"].quantile([0.25, 0.75])
        ext = sub[sub["ipr"] <= q_lo]
        loc = sub[sub["ipr"] >= q_hi]
        rows.append(dict(
            band=b, freq_lo=b/n_bands, freq_hi=(b+1)/n_bands,
            n_ext=len(ext), n_loc=len(loc),
            cn_extended_mean=float(ext["c_n"].mean()),
            cn_localized_mean=float(loc["c_n"].mean()),
            cn_extended_median=float(ext["c_n"].median()),
            cn_localized_median=float(loc["c_n"].median()),
            ratio_ext_over_loc=float(ext["c_n"].mean() / (loc["c_n"].mean() + 1e-30)),
        ))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    all_dfs = []
    for slug, stem, pretty in CITIES:
        try:
            df = analyse(slug, stem, pretty)
        except Exception as e:
            print(f"  ✗ {pretty}: {e}"); continue
        if df is not None:
            all_dfs.append(df)
    if not all_dfs:
        print("No cities processed."); return

    all_modes = pd.concat(all_dfs, ignore_index=True)
    all_modes.to_csv(OUT / "d40b_mode_bridge.csv", index=False)
    print(f"\n=== Pooled (city, mode) pairs: {len(all_modes)} ===")

    # ── (1) Banded Spearman ──────────────────────────────────────────────
    print("\n--- Within-frequency-band correlation (IPR vs c_n) ---")
    print("Hypothesis: extended modes (low IPR) capture more demand variance.")
    print("Expected sign: NEGATIVE ρ (low IPR → high c_n).")
    band_corr = banded_correlation(all_modes, n_bands=10)
    print(band_corr[["band","freq_lo","freq_hi","n","spearman_rho","spearman_p"]].to_string(index=False))
    # Overall pooled correlation (no band)
    rho_all, p_all = spearmanr(all_modes["ipr"], all_modes["c_n"])
    print(f"\n  Pooled ρ(IPR, c_n)  = {rho_all:+.4f}   p = {p_all:.3g}   "
          f"n = {len(all_modes)}")
    # Pooled correlation restricted to low-frequency half
    lowfreq = all_modes[all_modes["freq_rank"] <= 0.5]
    rho_lf, p_lf = spearmanr(lowfreq["ipr"], lowfreq["c_n"])
    print(f"  Low-freq half ρ      = {rho_lf:+.4f}   p = {p_lf:.3g}   "
          f"n = {len(lowfreq)}")

    # ── (2) Quartile contrast ────────────────────────────────────────────
    print("\n--- Extended (Q1 IPR) vs Localized (Q4 IPR) within each band ---")
    qc = quartile_contrast(all_modes, n_bands=10)
    print(qc[["band","freq_lo","cn_extended_mean","cn_localized_mean",
              "ratio_ext_over_loc"]].to_string(index=False))

    # ── Figures ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ax = axes[0]
    # Scatter: all modes pooled, color by frequency rank
    sc = ax.scatter(all_modes["ipr"], all_modes["c_n"] + 1e-8,
                    c=all_modes["freq_rank"], s=4, alpha=0.40,
                    cmap="viridis")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("IPR (localization)")
    ax.set_ylabel(r"$|\langle \psi_n, y \rangle|^2 / \|y\|^2$")
    ax.set_title(f"Mode-by-mode bridge: IPR vs demand-projection\n"
                 f"pooled n={len(all_modes)}  "
                 f"ρ={rho_all:+.3f} (p={p_all:.2g})  "
                 f"low-freq ρ={rho_lf:+.3f} (p={p_lf:.2g})",
                 fontsize=9)
    plt.colorbar(sc, ax=ax, label="frequency rank n/N", shrink=0.85)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    # Banded quartile contrast
    ax.semilogy((qc["freq_lo"] + qc["freq_hi"])/2, qc["cn_extended_mean"],
                "o-", color="C2", lw=1.5, markersize=6, label="extended (Q1 IPR)")
    ax.semilogy((qc["freq_lo"] + qc["freq_hi"])/2, qc["cn_localized_mean"],
                "s-", color="C3", lw=1.5, markersize=6, label="localized (Q4 IPR)")
    ax.set_xlabel("frequency rank (band centre)")
    ax.set_ylabel(r"mean $|\langle \psi_n, y \rangle|^2 / \|y\|^2$")
    ax.set_title("Extended vs localized modes (within frequency bands)\n"
                 "Anderson hypothesis: green > red across all bands")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Mode-by-mode test of the Anderson ↔ R²_spec bridge",
                 fontsize=11)
    fig.savefig(FIG / "fig_anderson_mode_bridge.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_anderson_mode_bridge.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"\n✓ {FIG/'fig_anderson_mode_bridge.pdf'}")

    # ── Verdict ──────────────────────────────────────────────────────────
    print("\n=== Verdict ===")
    bands_in_anderson_direction = (qc["ratio_ext_over_loc"] > 1.0).sum()
    print(f"  Bands where extended > localized (mean c_n): "
          f"{bands_in_anderson_direction}/{len(qc)}")
    if rho_all < -0.05 and p_all < 0.01:
        print("  → STRONG pooled signal in the Anderson direction.")
    elif rho_lf < -0.05 and p_lf < 0.01:
        print("  → SIGNAL in the low-frequency half (the regime the bound cares about).")
    else:
        print("  → Pooled signal weak.  Inspect banded results.")

    with open(OUT / "d40b_mode_bridge.json", "w") as f:
        json.dump({
            "n_modes_total": int(len(all_modes)),
            "n_modes_lowfreq": int(len(lowfreq)),
            "rho_pooled": float(rho_all),
            "p_pooled": float(p_all),
            "rho_lowfreq_half": float(rho_lf),
            "p_lowfreq_half": float(p_lf),
            "banded": band_corr.to_dict("records"),
            "quartile_contrast": qc.to_dict("records"),
            "wall_time_s": round(time.time()-t0, 1),
        }, f, indent=2)
    print(f"  Total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
