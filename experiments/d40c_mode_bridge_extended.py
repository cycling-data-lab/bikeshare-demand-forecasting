"""
d40c_mode_bridge_extended.py — Replication of the d40b mode-by-mode
bridge on an EXTENDED city panel.

The original d40b headline pools 6 923 (city, eigenmode) pairs across
9 networks (6 Tier 1 + 3 Tier 2): pooled Spearman ρ(IPR, c_n) =
−0.300, p = 2.3·10⁻¹⁴⁴.

This script extends the panel to the full set of GBFS-polled French
Tier 2 networks for which both an IMD parquet and a demand-prediction
parquet are available — 16 additional networks beyond Paris, Lyon,
Toulouse already in d40b.  Total panel: 6 Tier 1 + 19 Tier 2 = 25
networks.

Two outputs:

  (1) pooled ρ on the extended panel
  (2) per-tier breakdown:
      - Tier 1 alone (6 networks, n ≈ 6 000 pairs)
      - Tier 2 (19 networks, n ≈ 10 000 pairs)
      - Combined (25 networks, n ≈ 16 000 pairs)
  (3) banded analysis on the extended pool

If the topological-localization mechanism is genuine, the headline ρ
should hold across the extended panel.  If it weakens substantially,
the result is specific to Tier-1-scale dense networks.

Output:
  outputs/d40c_mode_bridge_extended.csv     (one row per (city, mode))
  outputs/d40c_mode_bridge_extended.json    (summary statistics)
  figures/fig_mode_bridge_extended.{pdf,png}
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

# Extended panel: 6 Tier 1 + 19 Tier 2 French GBFS-polled
CITIES = [
    # (slug, imd parquet stem, pretty name, tier)
    ("boston_bluebikes",    "boston_bluebikes",          "Bluebikes Boston",        "tier1"),
    ("dc_capitalbikeshare", "dc_capitalbikeshare",       "Capital Bikeshare DC",    "tier1"),
    ("chicago_divvy",       "chicago_divvy",             "Divvy Chicago",           "tier1"),
    ("sf_baywheels",        "sf_baywheels",              "Bay Wheels SF",           "tier1"),
    ("london_tfl",          "london_tfl",                "Santander Cycles London", "tier1"),
    ("montreal_bixi",       "world_ca_bixi_montr_al",    "BIXI Montréal",           "tier1"),
    # 19 Tier 2 French GBFS-polled (the 3 used in d40b are first; the new 16 follow)
    ("tier2_paris",                    "world_fr_v_lib_metropole",      "Vélib Paris",                       "tier2"),
    ("tier2_lyon",                     "world_fr_v_lo_v",                "Vélo'v Lyon",                       "tier2"),
    ("tier2_toulouse",                 "world_fr_v_l_toulouse",          "VéLÔ Toulouse",                     "tier2"),
    ("tier2_velo-tbm-bordeaux",        "world_fr_le_v_lo_par_tbm",       "Le VélO Bordeaux",                  "tier2"),
    ("tier2_levelo_inurba_marseille",  "world_fr_lev_lo_marseille",      "Le Vélo Marseille",                 "tier2"),
    ("tier2_velivert_saint_etienne",   "world_fr_v_livert",              "Vélivert Saint-Étienne",            "tier2"),
    ("tier2_nantes",                   "world_fr_naolib",                "Naolib Nantes",                     "tier2"),
    ("tier2_twisto_velolib_caen",      "world_fr_twisto_v_lolib",        "Twisto VéloLib Caen",               "tier2"),
    ("tier2_inurba-rouen",             "world_fr_lovelo_libre_service",  "LoveLo Rouen",                      "tier2"),
    ("tier2_velonecy60minutes_annecy", "world_fr_v_lonecy",              "Vélonecy Annecy",                   "tier2"),
    ("tier2_vilvolt_epinal",           "world_fr_vilvolt",               "VilVolt Épinal",                    "tier2"),
    ("tier2_velozef",                  "world_fr_v_lozef",               "VéloZef",                           "tier2"),
    ("tier2_velopop",                  "world_fr_v_lopop",               "VéloPop",                           "tier2"),
    ("tier2_le_velo_star",             "world_fr_le_v_lo_star",          "Le Vélo Star",                      "tier2"),
    ("tier2_tanlib",                   "world_fr_v_lo_tanlib",           "TanLib",                            "tier2"),
    ("tier2_capcotentin",              "world_fr_capcotentin",           "Cap Cotentin",                      "tier2"),
    ("tier2_zebullo",                  "world_fr_zebullo",               "Zebullo",                           "tier2"),
    ("tier2_nancy",                    "world_fr_v_lostan_lib",          "VéloStan Nancy",                    "tier2"),
    ("tier2_amiens",                   "world_fr_velam",                 "Vélam Amiens",                      "tier2"),
]

K_NN = 6
SIGMA_M = 300.0
EARTH_R = 6_371_000.0

# d10 city slug -> demand prediction parquet (city short slug)
TIER2_CITY_MAP = {
    "tier2_paris":                    "Paris",
    "tier2_lyon":                     "lyon",
    "tier2_toulouse":                 "toulouse",
    "tier2_velo-tbm-bordeaux":        "velo-tbm-bordeaux",
    "tier2_levelo_inurba_marseille":  "levelo_inurba_marseille",
    "tier2_velivert_saint_etienne":   "velivert_saint_etienne",
    "tier2_nantes":                   "nantes",
    "tier2_twisto_velolib_caen":      "twisto_velolib_caen",
    "tier2_inurba-rouen":             "inurba-rouen",
    "tier2_velonecy60minutes_annecy": "velonecy60minutes_annecy",
    "tier2_vilvolt_epinal":           "vilvolt_epinal",
    "tier2_velozef":                  "velozef",
    "tier2_velopop":                  "velopop",
    "tier2_le_velo_star":             "le_velo_star",
    "tier2_tanlib":                   "tanlib",
    "tier2_capcotentin":              "capcotentin",
    "tier2_zebullo":                  "zebullo",
    "tier2_nancy":                    "nancy",
    "tier2_amiens":                   "amiens",
}


def haversine_matrix(lat, lng):
    lat_r = np.deg2rad(lat); lng_r = np.deg2rad(lng)
    dphi = lat_r[:, None] - lat_r[None, :]
    dlam = lng_r[:, None] - lng_r[None, :]
    a = np.sin(dphi/2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlam/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_L_sym(lat, lng, k=K_NN, sigma=SIGMA_M):
    N = len(lat)
    D = haversine_matrix(lat, lng); np.fill_diagonal(D, np.inf)
    knn = np.argpartition(D, min(k, N-1), axis=1)[:, :min(k, N-1)]
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
    if slug in ("boston_bluebikes","dc_capitalbikeshare","chicago_divvy","sf_baywheels"):
        path = OUT / f"d3_{slug}_predictions.parquet"
    elif slug == "london_tfl":
        path = OUT / "d16_london_tfl_predictions.parquet"
    elif slug == "montreal_bixi":
        path = OUT / "d14_montreal_bixi_predictions.parquet"
    elif slug.startswith("tier2_"):
        short = TIER2_CITY_MAP.get(slug)
        if short is None: return {}
        path = OUT / f"d10_{short}_predictions.parquet"
    else:
        return {}
    if not path.exists(): return {}
    df = pd.read_parquet(path)
    df["station_id"] = df["station_id"].astype(str)
    if slug == "london_tfl":
        df["station_id"] = df["station_id"].str.zfill(6)
    return df.assign(y=np.expm1(df["y_true_log"])).groupby("station_id")["y"].mean().to_dict()


def analyse(slug, stem, pretty, tier):
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
    if N < 30: return None   # relaxed from 50 for small Tier-2 networks

    L_sym = build_L_sym(sub["lat"].values, sub["lng"].values)
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    ipr_n = ipr_cols(eigvecs)
    pr_n = 1.0 / (N * ipr_n + 1e-30)

    y = sub["y"].values.astype(float)
    y_z = (y - y.mean()) / (y.std() + 1e-12)
    coefs = eigvecs.T @ y_z
    c_n = (coefs**2) / (y_z @ y_z + 1e-30)

    return pd.DataFrame({
        "city": pretty, "slug": slug, "tier": tier, "N": N,
        "mode_idx": np.arange(N), "freq_rank": np.arange(N) / N,
        "eigval": eigvals, "ipr": ipr_n, "pr": pr_n, "c_n": c_n,
    })


def banded_correlation(all_modes, n_bands=10):
    all_modes = all_modes.copy()
    all_modes["band"] = pd.cut(all_modes["freq_rank"],
                                bins=np.linspace(0, 1, n_bands+1),
                                labels=False, include_lowest=True)
    rows = []
    for b in range(n_bands):
        sub = all_modes[all_modes["band"] == b]
        if len(sub) < 30: continue
        rho, p = spearmanr(sub["ipr"], sub["c_n"])
        rows.append(dict(band=b,
                         freq_lo=b/n_bands, freq_hi=(b+1)/n_bands,
                         n=len(sub),
                         spearman_rho=float(rho), spearman_p=float(p)))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    all_dfs = []
    per_city_summary = []
    for slug, stem, pretty, tier in CITIES:
        try:
            df = analyse(slug, stem, pretty, tier)
        except Exception as e:
            print(f"  ✗ {pretty}: {e}"); continue
        if df is None:
            print(f"  ✗ {pretty}: no demand or N<30"); continue
        # Per-city rho
        rho_city, p_city = spearmanr(df["ipr"], df["c_n"])
        per_city_summary.append(dict(
            city=pretty, slug=slug, tier=tier, N=int(df["N"].iloc[0]),
            n_modes=len(df), rho=float(rho_city), p=float(p_city)))
        print(f"  ✓ {pretty:32s} ({tier})  N={int(df['N'].iloc[0]):4d}  "
              f"ρ_city={rho_city:+.3f}  p={p_city:.2e}")
        all_dfs.append(df)
    if not all_dfs:
        print("No cities processed."); return

    all_modes = pd.concat(all_dfs, ignore_index=True)
    all_modes.to_csv(OUT / "d40c_mode_bridge_extended.csv", index=False)

    # Per-city z-scoring to remove between-city heterogeneity
    # (naive pooling suffers from Simpson's paradox when networks have
    #  different size and demand-concentration regimes).
    all_modes["ipr_z_city"] = all_modes.groupby("slug")["ipr"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-12))
    all_modes["cn_z_city"] = all_modes.groupby("slug")["c_n"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-12))

    # Per-tier and combined statistics
    print("\n=== Pooled statistics (NAIVE pooling — exposed to Simpson's paradox) ===")
    pooled = {}
    for tier_label, mask in [
        ("Tier 1 (6 networks)",  all_modes["tier"] == "tier1"),
        ("Tier 2 (19 networks)", all_modes["tier"] == "tier2"),
        ("Combined (25 networks)", np.ones(len(all_modes), bool)),
    ]:
        sub = all_modes[mask]
        rho_n, p_n = spearmanr(sub["ipr"], sub["c_n"])
        lf = sub[sub["freq_rank"] <= 0.5]
        rho_lf, p_lf = spearmanr(lf["ipr"], lf["c_n"])
        # Within-city z-scored pooled (Simpson's-paradox-resistant)
        rho_z, p_z = spearmanr(sub["ipr_z_city"], sub["cn_z_city"])
        pooled[tier_label] = dict(
            n_modes=len(sub),
            n_cities=int(sub["slug"].nunique()),
            rho_naive_pooled=float(rho_n), p_naive_pooled=float(p_n),
            rho_zscored_within_city=float(rho_z), p_zscored_within_city=float(p_z),
            rho_lowfreq_half_naive=float(rho_lf), p_lowfreq_half=float(p_lf))
        print(f"  {tier_label:30s}  n={len(sub):5d}/{sub['slug'].nunique():2d}  "
              f"ρ_naive={rho_n:+.4f}  (p={p_n:.2e})   "
              f"ρ_z(within-city)={rho_z:+.4f} (p={p_z:.2e})")

    # Per-city distribution
    print("\n=== Per-city Spearman ρ distribution ===")
    per_df = pd.DataFrame(per_city_summary)
    for tier in ["tier1", "tier2"]:
        sub = per_df[per_df["tier"] == tier]
        print(f"  {tier}: n={len(sub):2d} cities  "
              f"median ρ = {sub['rho'].median():+.3f}  "
              f"mean ρ = {sub['rho'].mean():+.3f}  "
              f"fraction ρ<0: {(sub['rho']<0).mean()*100:.0f}%  "
              f"fraction with p<0.05: {(sub['p']<0.05).mean()*100:.0f}%")
    # Fisher's combined p-value across all per-city tests (one-tailed in topological direction)
    from scipy.stats import combine_pvalues
    one_tail_pvals = [(r["p"]/2 if r["rho"] < 0 else 1 - r["p"]/2) for r in per_city_summary]
    fisher_stat, fisher_p = combine_pvalues(one_tail_pvals, method="fisher")
    print(f"  Fisher combined one-tailed p across all 24 cities: p_fisher = {fisher_p:.3e}")

    # Banded analysis on combined panel
    print("\n=== Banded analysis (combined 25-city panel) ===")
    band_corr = banded_correlation(all_modes, n_bands=10)
    print(band_corr[["band","freq_lo","freq_hi","n","spearman_rho","spearman_p"]].to_string(index=False))
    n_anderson_dir = (band_corr["spearman_rho"] < 0).sum()
    print(f"  Bands in topological direction (ρ<0): {n_anderson_dir}/{len(band_corr)}")

    # Compare to d40b headline (9-city)
    print(f"\n=== Comparison to d40b (9-city headline) ===")
    print(f"  d40b headline:                     ρ = -0.300  (p = 2.3e-144, n = 6,923)")
    rho_combined = pooled["Combined (25 networks)"]["rho_naive_pooled"]
    p_combined = pooled["Combined (25 networks)"]["p_naive_pooled"]
    rho_combined_z = pooled["Combined (25 networks)"]["rho_zscored_within_city"]
    p_combined_z = pooled["Combined (25 networks)"]["p_zscored_within_city"]
    n_combined = pooled["Combined (25 networks)"]["n_modes"]
    print(f"  d40c naive (25-city):              ρ = {rho_combined:+.4f}  (p = {p_combined:.2e}, n = {n_combined})")
    print(f"  d40c within-city z-scored:         ρ = {rho_combined_z:+.4f}  (p = {p_combined_z:.2e})")
    print(f"  d40c Fisher combined (per-city):   p = {fisher_p:.2e}  across 24 cities")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(all_modes["ipr"], all_modes["c_n"] + 1e-8,
                    c=all_modes["freq_rank"], s=3, alpha=0.35, cmap="viridis")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("IPR (localization)")
    ax.set_ylabel(r"$|\langle \psi_n, y \rangle|^2 / \|y\|^2$")
    ax.set_title(f"Extended 25-city panel  (n = {n_combined} pairs)\n"
                 f"naive pooled ρ = {rho_combined:+.3f};  within-city z-scored ρ = {rho_combined_z:+.3f}\n"
                 f"Fisher combined p = {fisher_p:.2e}", fontsize=9)
    plt.colorbar(sc, ax=ax, label="frequency rank n/N", shrink=0.85)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    per_df = pd.DataFrame(per_city_summary).sort_values("rho")
    colors = ["C0" if t == "tier1" else "C1" for t in per_df["tier"]]
    ax.barh(range(len(per_df)), per_df["rho"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(per_df)))
    ax.set_yticklabels(per_df["city"], fontsize=7)
    ax.axvline(0, color="black", lw=0.5)
    ax.axvline(rho_combined, color="red", ls="--", lw=0.8,
               label=f"pooled = {rho_combined:+.3f}")
    ax.set_xlabel(r"per-city Spearman $\rho(\mathrm{IPR}, c_n)$")
    ax.set_title("Per-city correlations\nblue = Tier 1 trip-log, orange = Tier 2 GBFS-polled", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Topological localization bridge — replication on extended 25-city panel",
                 fontsize=11)
    fig.savefig(FIG / "fig_mode_bridge_extended.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_mode_bridge_extended.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"\n✓ {FIG/'fig_mode_bridge_extended.pdf'}")

    # Fisher combined per-tier
    from scipy.stats import combine_pvalues
    fisher_per_tier = {}
    for tier in ["tier1", "tier2"]:
        subset = [r for r in per_city_summary if r["tier"] == tier]
        if subset:
            pvals = [(r["p"]/2 if r["rho"] < 0 else 1 - r["p"]/2) for r in subset]
            stat, pf = combine_pvalues(pvals, method="fisher")
            fisher_per_tier[tier] = dict(p_fisher=float(pf), n_cities=len(subset))

    with open(OUT / "d40c_mode_bridge_extended.json", "w") as f:
        json.dump({
            "config": dict(K_NN=K_NN, SIGMA_M=SIGMA_M),
            "n_cities_attempted": len(CITIES),
            "n_cities_processed": len(per_city_summary),
            "per_city": per_city_summary,
            "pooled_per_tier": pooled,
            "fisher_combined": dict(p_fisher_all=float(fisher_p), n_cities=len(per_city_summary)),
            "fisher_per_tier": fisher_per_tier,
            "banded": band_corr.to_dict("records"),
            "d40b_headline_reference": dict(rho_naive=-0.300, p="2.3e-144", n=6923, n_cities=9),
            "wall_time_s": round(time.time()-t0, 1),
        }, f, indent=2)
    print(f"  Total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
