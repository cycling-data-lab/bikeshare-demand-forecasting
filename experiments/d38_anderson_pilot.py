"""
d38_anderson_pilot.py — Anderson localization diagnostic on urban
mobility graphs.

Hypothesis (F5): the structural applicability bound's collapse on
certain cities is the spectral signature of an Anderson localization
transition.  We test this by:

  1. Building the station-proximity Laplacian L_sym (same recipe as d24).
  2. Constructing the Anderson Hamiltonian
        H = L_sym + W · diag(ε_i)
     where ε_i are on-site potentials built from the standardised IMD
     features (a composite of M, I, T, D — the "disorder" landscape).
  3. Diagonalising H and computing for each eigenvector ψ_n:
       • IPR(ψ_n)   = Σ_i |ψ_n(i)|^4              (≈1/N extended, O(1) localized)
       • PR(ψ_n)    = 1 / (N · IPR(ψ_n))           ∈ (0,1]
       • normalised mass on top-5% sites          (localization concentration)
  4. Computing level statistics:
       • adjacent gap ratio r_n = min(s,s')/max(s,s')
         where s = λ_{n+1}-λ_n, s' = λ_{n+2}-λ_{n+1}
       • <r> ≈ 0.5295 for Gaussian Orthogonal Ensemble (extended/metallic)
       • <r> ≈ 0.3863 for Poisson statistics       (localized/insulating)
  5. Locating the mobility edge E_c: the eigenvalue separating extended
     and localized regimes (if a separatrix exists).
  6. Mapping the spatial signature of the most localized eigenvectors.

This is a PILOT — the question it answers is binary:
  • If localization is present in real cities → F5 is empirically viable,
    proceed to the full paper with the disorder-sweep validation (d39).
  • If IPR is uniform and <r> hovers at GOE for all cities → F5 dies
    here, pivot to F1 (percolation) instead.

Output:
  outputs/d38_anderson_per_city.csv
  outputs/d38_anderson_eigenmodes.npz   (eigenvalues, IPR, top-mode vectors)
  outputs/d38_anderson_summary.json
  figures/fig_anderson_ipr_vs_energy.{pdf,png}
  figures/fig_anderson_level_statistics.{pdf,png}
  figures/fig_anderson_geographic_modes.{pdf,png}
  figures/fig_anderson_pr_distribution.{pdf,png}
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
from matplotlib.colors import LogNorm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
IMD_INTL_DIR = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

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

# Same graph parameters as d24/d28 for direct comparability
K_NN = 6
SIGMA_M = 300.0
EARTH_R = 6_371_000.0

# Anderson Hamiltonian: H = L_sym + W * diag(ε)
# W is the disorder strength.  W=1.0 gives diagonal of the same magnitude
# as the standardised Laplacian spectrum.  d39 varies W; d38 fixes W=1.
W_DISORDER = 1.0

# Composite IMD potential: simple unweighted average of the
# standardised available features.  d28/d24 already shows these axes are
# quasi-independent, so the mean is a reasonable composite.
FEATS_IMD = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
             "elevation_m", "topography_roughness_index"]

# Reference benchmarks from Random Matrix Theory:
R_GOE = 0.5295        # extended, metallic phase
R_POISSON = 0.3863    # localized, insulating phase


# ─────────────────────────────────────────────────────────────────────────
# Graph construction (reuses d24 conventions)
# ─────────────────────────────────────────────────────────────────────────
def haversine_matrix(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    lat_r = np.deg2rad(lat); lng_r = np.deg2rad(lng)
    dphi = lat_r[:, None] - lat_r[None, :]
    dlam = lng_r[:, None] - lng_r[None, :]
    a = np.sin(dphi/2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlam/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_knn_graph(lat, lng, k=K_NN, sigma=SIGMA_M):
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


def composite_potential(imd_df: pd.DataFrame, feats: list) -> np.ndarray:
    """Aggregate IMD features into a scalar on-site potential ε_i.
    Each feature is standardised (mean 0, std 1), then averaged."""
    avail = [f for f in feats if f in imd_df.columns]
    if not avail:
        return np.zeros(len(imd_df))
    X = imd_df[avail].astype(float).values
    X = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-12)
    X = np.nan_to_num(X, nan=0.0)
    eps = X.mean(axis=1)
    # Center the composite (the diagonal shift is absorbed in spectrum origin)
    eps = eps - eps.mean()
    # Renormalise so that std(ε) = 1 ; W controls the absolute scale
    std_eps = eps.std() + 1e-12
    return eps / std_eps


# ─────────────────────────────────────────────────────────────────────────
# Anderson observables
# ─────────────────────────────────────────────────────────────────────────
def ipr(psi: np.ndarray) -> np.ndarray:
    """IPR(ψ_n) = Σ_i |ψ_n(i)|^4   for each column of psi.
    psi is shape (N, N) — columns are eigenvectors."""
    # Normalise columns (eigh already returns normalised, but be safe)
    psi2 = psi**2
    psi2 = psi2 / (psi2.sum(axis=0, keepdims=True) + 1e-30)
    return (psi2**2).sum(axis=0)


def participation_ratio(ipr_vals: np.ndarray, N: int) -> np.ndarray:
    """PR = 1/(N · IPR) ∈ (0,1].  PR ≈ 1 → fully extended; PR ≈ 1/N → localized."""
    return 1.0 / (N * ipr_vals + 1e-30)


def top_mass(psi_col: np.ndarray, frac: float = 0.05) -> float:
    """Fraction of |ψ|² mass concentrated on the top `frac` of nodes."""
    p = psi_col**2
    p = p / (p.sum() + 1e-30)
    p_sorted = np.sort(p)[::-1]
    k = max(1, int(np.ceil(frac * len(p))))
    return float(p_sorted[:k].sum())


def gap_ratio(eigvals: np.ndarray) -> np.ndarray:
    """Adjacent gap ratios r_n = min(s,s')/max(s,s'),
    s = λ_{n+1}-λ_n, s' = λ_{n+2}-λ_{n+1}.  Returns length N-2."""
    s = np.diff(np.sort(eigvals))   # length N-1
    r = np.minimum(s[:-1], s[1:]) / (np.maximum(s[:-1], s[1:]) + 1e-30)
    return r


def localization_phase(r_mean: float, tol: float = 0.05) -> str:
    """Coarse classifier from the mean gap ratio."""
    if r_mean > R_GOE - tol:
        return "extended_GOE"
    if r_mean < R_POISSON + tol:
        return "localized_Poisson"
    return "intermediate"


def estimate_mobility_edge(eigvals: np.ndarray, ipr_vals: np.ndarray,
                           ipr_threshold: float = None) -> float:
    """Mobility edge E_c: the eigenvalue at which IPR rises above
    a threshold (states above E_c are 'localized' relative to states
    below).  If no clear separatrix exists, return NaN."""
    if ipr_threshold is None:
        # Use median IPR as a robust threshold
        ipr_threshold = float(np.median(ipr_vals)) * 2.0
    order = np.argsort(eigvals)
    e_sorted = eigvals[order]
    ipr_sorted = ipr_vals[order]
    # Smooth IPR with a small box-average to denoise
    win = max(5, len(ipr_sorted) // 50)
    ipr_smooth = np.convolve(ipr_sorted, np.ones(win)/win, mode="same")
    above = ipr_smooth > ipr_threshold
    if not above.any():
        return float("nan")
    # First sustained crossing (10+ consecutive above) — robust separatrix
    for i in range(len(above) - 10):
        if above[i:i+10].all():
            return float(e_sorted[i])
    return float("nan")


# ─────────────────────────────────────────────────────────────────────────
# Per-city analysis
# ─────────────────────────────────────────────────────────────────────────
def analyse_city(slug, imd_stem, pretty, source):
    imd_path = IMD_INTL_DIR / f"{imd_stem}.parquet"
    if not imd_path.exists():
        return None, None
    imd = pd.read_parquet(imd_path)
    imd["station_id"] = imd["station_id"].astype(str)
    imd = imd.dropna(subset=["lat", "lng"]).reset_index(drop=True)
    N = len(imd)
    if N < 50:
        return None, None

    print(f"\n=== {pretty} ({slug}) ===")
    print(f"  {N} stations  building k-NN graph (k={K_NN}, σ={SIGMA_M}m)...")

    _, deg, L_sym = build_knn_graph(imd["lat"].values, imd["lng"].values)
    eps = composite_potential(imd, FEATS_IMD)
    H = L_sym + W_DISORDER * np.diag(eps)
    print(f"  Hamiltonian H = L_sym + W·diag(ε)   W={W_DISORDER},  std(ε)={eps.std():.3f}")

    eigvals_L, eigvecs_L = np.linalg.eigh(L_sym)
    eigvals_H, eigvecs_H = np.linalg.eigh(H)
    print(f"  Spectra: λ(L) ∈ [{eigvals_L[0]:.4f}, {eigvals_L[-1]:.4f}]   "
          f"λ(H) ∈ [{eigvals_H[0]:.4f}, {eigvals_H[-1]:.4f}]")

    # Anderson observables on H
    ipr_H = ipr(eigvecs_H)
    pr_H = participation_ratio(ipr_H, N)
    r_H = gap_ratio(eigvals_H)
    r_mean_H = float(r_H.mean())
    phase_H = localization_phase(r_mean_H)

    # Baseline on L_sym alone (no disorder)
    ipr_L = ipr(eigvecs_L)
    pr_L = participation_ratio(ipr_L, N)
    r_L = gap_ratio(eigvals_L)
    r_mean_L = float(r_L.mean())
    phase_L = localization_phase(r_mean_L)

    # Mobility edge on H
    E_c = estimate_mobility_edge(eigvals_H, ipr_H)

    # The 5 most-localized eigenvectors of H (highest IPR)
    top_loc_idx = np.argsort(-ipr_H)[:5]
    top_modes = eigvecs_H[:, top_loc_idx]   # shape (N, 5)
    top_mass_5pct = np.array([top_mass(top_modes[:, k]) for k in range(5)])

    # Fraction of modes that are "localized" by IPR > 5/N criterion
    # (extended states have IPR ≈ 1/N, so 5/N is a soft threshold)
    frac_localized = float((ipr_H > 5.0/N).mean())

    # Localization length proxy: 1/sqrt(IPR · N) for the most-localized mode
    xi_top = float(1.0 / np.sqrt(ipr_H[top_loc_idx[0]] * N))

    row = dict(
        slug=slug, city=pretty, source=source, N=N,
        mean_degree=float(deg.mean()),
        # Disorder strength applied
        W_disorder=W_DISORDER,
        std_potential=float(eps.std()),
        # IPR & PR statistics on H (Anderson Hamiltonian)
        ipr_H_mean=float(ipr_H.mean()),
        ipr_H_median=float(np.median(ipr_H)),
        ipr_H_max=float(ipr_H.max()),
        pr_H_mean=float(pr_H.mean()),
        pr_H_min=float(pr_H.min()),
        # IPR & PR on L (no disorder, reference)
        ipr_L_mean=float(ipr_L.mean()),
        pr_L_mean=float(pr_L.mean()),
        # Level statistics
        r_mean_H=r_mean_H,
        r_mean_L=r_mean_L,
        phase_H=phase_H,
        phase_L=phase_L,
        # GOE/Poisson distance
        dist_to_GOE_H=abs(r_mean_H - R_GOE),
        dist_to_Poisson_H=abs(r_mean_H - R_POISSON),
        # Mobility edge
        mobility_edge_E_c=E_c,
        # Fraction of localized states (IPR > 5/N)
        frac_localized=frac_localized,
        # Top-mode characterisation
        top_mode_mass_5pct=float(top_mass_5pct[0]),
        top_mode_xi_proxy=xi_top,
        lambda_min_H=float(eigvals_H[0]),
        lambda_max_H=float(eigvals_H[-1]),
        lambda_min_L=float(eigvals_L[0]),
        lambda_max_L=float(eigvals_L[-1]),
    )

    print(f"  IPR(H)  mean={ipr_H.mean():.4f}  median={np.median(ipr_H):.4f}  max={ipr_H.max():.4f}")
    print(f"  PR(H)   mean={pr_H.mean():.3f}  min={pr_H.min():.4f}")
    print(f"  <r>(H)  = {r_mean_H:.4f}   ({phase_H})   "
          f"[GOE={R_GOE}, Poisson={R_POISSON}]")
    print(f"  <r>(L)  = {r_mean_L:.4f}   ({phase_L})   (reference, no disorder)")
    print(f"  Mobility edge E_c ≈ {E_c}")
    print(f"  Fraction localized states (IPR>5/N): {frac_localized*100:.1f}%")
    print(f"  Top-1 mode: mass on top-5% nodes = {top_mass_5pct[0]:.3f}   ξ_proxy = {xi_top:.4f}")

    # Bundle eigenmode artifacts for downstream plots
    artifacts = dict(
        eigvals_H=eigvals_H, ipr_H=ipr_H,
        eigvals_L=eigvals_L, ipr_L=ipr_L,
        r_H=r_H, r_L=r_L,
        top_modes=top_modes, top_loc_idx=top_loc_idx,
        lat=imd["lat"].values, lng=imd["lng"].values,
        eps=eps,
        pretty=pretty,
    )
    return row, artifacts


# ─────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────
def fig_ipr_vs_energy(artifacts_by_city, path):
    """IPR(eigenvalue) for each city, with the mobility edge marked."""
    n = len(artifacts_by_city)
    cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.0*rows),
                             constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, (slug, art) in zip(axes, artifacts_by_city.items()):
        N = len(art["eigvals_H"])
        order = np.argsort(art["eigvals_H"])
        e = art["eigvals_H"][order]
        i = art["ipr_H"][order]
        ax.semilogy(e, i, ".", markersize=2, alpha=0.55, color="C0", label="H = L+Wε")
        ax.semilogy(np.sort(art["eigvals_L"]),
                    art["ipr_L"][np.argsort(art["eigvals_L"])],
                    ".", markersize=2, alpha=0.30, color="grey", label="L (no disorder)")
        ax.axhline(1.0/N, color="green", ls="--", lw=0.7, alpha=0.7,
                   label="extended (1/N)")
        ax.axhline(5.0/N, color="red", ls=":", lw=0.7, alpha=0.7,
                   label="localized threshold (5/N)")
        ax.set_title(art["pretty"], fontsize=9)
        ax.set_xlabel(r"$\lambda$", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("IPR", fontsize=9)
    axes[0].legend(fontsize=7, loc="lower right", framealpha=0.9)
    for ax in axes[len(artifacts_by_city):]:
        ax.set_visible(False)
    fig.suptitle("Anderson IPR vs energy on urban mobility graphs\n"
                 "Above the 5/N red line: localized.  At the 1/N green line: extended.",
                 fontsize=10)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def fig_level_statistics(artifacts_by_city, path):
    """Distribution of adjacent gap ratios r, with GOE and Poisson refs."""
    n = len(artifacts_by_city)
    cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.0*rows),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    x = np.linspace(0, 1, 200)
    # Wigner surmise for GOE: P(r) = (27/8) (r+r²) / (1+r+r²)^(5/2)
    P_GOE = (27/8) * (x + x**2) / (1 + x + x**2)**(2.5)
    # Poisson: P(r) = 2/(1+r)²
    P_POISSON = 2.0 / (1 + x)**2
    for ax, (slug, art) in zip(axes, artifacts_by_city.items()):
        ax.hist(art["r_H"], bins=40, range=(0,1), density=True,
                color="C0", alpha=0.55, label="H (with disorder)")
        ax.hist(art["r_L"], bins=40, range=(0,1), density=True,
                color="grey", alpha=0.30, label="L (no disorder)")
        ax.plot(x, P_GOE, "-", color="green", lw=1.2, label=f"GOE (metallic, <r>={R_GOE})")
        ax.plot(x, P_POISSON, "--", color="red", lw=1.2, label=f"Poisson (localized, <r>={R_POISSON})")
        ax.axvline(art["r_H"].mean(), color="C0", lw=1.0, alpha=0.9)
        ax.set_title(f"{art['pretty']}  <r>={art['r_H'].mean():.3f}",
                     fontsize=9)
        ax.set_xlabel("r (adjacent gap ratio)", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 2.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("P(r)", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper right", framealpha=0.9)
    for ax in axes[len(artifacts_by_city):]:
        ax.set_visible(False)
    fig.suptitle("Level-spacing statistics: extended (GOE) vs localized (Poisson)",
                 fontsize=10)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def fig_geographic_modes(artifacts_by_city, path):
    """Spatial map of the most localized eigenmode for each city."""
    n = len(artifacts_by_city)
    cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 3.6*rows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, (slug, art) in zip(axes, artifacts_by_city.items()):
        psi = art["top_modes"][:, 0]
        psi2 = psi**2
        psi2 = psi2 / (psi2.sum() + 1e-30)
        # Sort by intensity so localized nodes are plotted on top
        order = np.argsort(psi2)
        sc = ax.scatter(art["lng"][order], art["lat"][order],
                        c=psi2[order], s=12, cmap="inferno",
                        norm=LogNorm(vmin=max(psi2[psi2>0].min(), 1e-8),
                                     vmax=psi2.max()),
                        edgecolors="none")
        ax.set_title(f"{art['pretty']}  most-localized mode\n"
                     f"IPR={art['ipr_H'][art['top_loc_idx'][0]]:.4f}",
                     fontsize=9)
        ax.set_xlabel("lng", fontsize=8); ax.set_ylabel("lat", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")
        plt.colorbar(sc, ax=ax, label=r"$|\psi|^2$", shrink=0.75)
    for ax in axes[len(artifacts_by_city):]:
        ax.set_visible(False)
    fig.suptitle("Geographic footprint of the most-localized Anderson eigenmode",
                 fontsize=11)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def fig_pr_distribution(artifacts_by_city, path):
    """Distribution of participation ratios — extended vs localized mass."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(artifacts_by_city)))
    for (slug, art), c in zip(artifacts_by_city.items(), colors):
        N = len(art["eigvals_H"])
        pr_vals = 1.0 / (N * art["ipr_H"] + 1e-30)
        # CCDF (1 - empirical CDF) on log-x
        sorted_pr = np.sort(pr_vals)
        ccdf = 1.0 - np.arange(len(sorted_pr)) / len(sorted_pr)
        ax.semilogx(sorted_pr, ccdf, "-", color=c, lw=1.2, label=art["pretty"])
    ax.axvline(1.0, color="green", ls="--", lw=0.7, label="fully extended (PR=1)")
    ax.set_xlabel("Participation ratio  PR = 1/(N·IPR)")
    ax.set_ylabel("CCDF: fraction of modes with PR ≥ x")
    ax.set_title("PR distributions — left mass = localized modes")
    ax.set_xlim(1e-3, 1.5)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    rows = []
    artifacts_by_city = {}
    npz_bundle = {}

    for slug, stem, pretty, source in CITIES:
        try:
            row, art = analyse_city(slug, stem, pretty, source)
        except Exception as e:
            print(f"  ✗ {pretty}: {e}")
            continue
        if row is None:
            continue
        rows.append(row)
        artifacts_by_city[slug] = art
        # Pack into NPZ
        npz_bundle[f"{slug}__eigvals_H"] = art["eigvals_H"]
        npz_bundle[f"{slug}__eigvals_L"] = art["eigvals_L"]
        npz_bundle[f"{slug}__ipr_H"]     = art["ipr_H"]
        npz_bundle[f"{slug}__ipr_L"]     = art["ipr_L"]
        npz_bundle[f"{slug}__top_modes"] = art["top_modes"]
        npz_bundle[f"{slug}__lat"]       = art["lat"]
        npz_bundle[f"{slug}__lng"]       = art["lng"]
        npz_bundle[f"{slug}__eps"]       = art["eps"]

    if not rows:
        print("No cities processed.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d38_anderson_per_city.csv", index=False)
    print("\n=== Per-city summary ===")
    cols_print = ["city", "N", "ipr_H_mean", "r_mean_H", "phase_H",
                  "frac_localized", "top_mode_mass_5pct", "mobility_edge_E_c"]
    print(df[cols_print].to_string(index=False))

    # ── Verdict logic ──────────────────────────────────────────────────
    print("\n=== Pilot verdict ===")
    n_localized_or_intermediate = (df["phase_H"] != "extended_GOE").sum()
    n_total = len(df)
    mean_dist_to_GOE = df["dist_to_GOE_H"].mean()
    print(f"  Cities NOT in GOE (i.e. showing localization signature): "
          f"{n_localized_or_intermediate}/{n_total}")
    print(f"  Mean distance to GOE benchmark: {mean_dist_to_GOE:.3f}  "
          f"(>0.05 = clear signal)")
    if n_localized_or_intermediate >= max(2, n_total // 3) or mean_dist_to_GOE > 0.05:
        print("  → SIGNAL PRESENT.  F5 (Anderson localization) is empirically viable.")
        print("    Next: run d39_anderson_disorder_sweep.py to map the phase diagram.")
    else:
        print("  → SIGNAL ABSENT or weak.  F5 may be marginal; consider pivot to F1.")

    # ── Figures ────────────────────────────────────────────────────────
    print("\n=== Generating figures ===")
    fig_ipr_vs_energy(artifacts_by_city, FIG / "fig_anderson_ipr_vs_energy")
    print(f"  ✓ {FIG/'fig_anderson_ipr_vs_energy.pdf'}")
    fig_level_statistics(artifacts_by_city, FIG / "fig_anderson_level_statistics")
    print(f"  ✓ {FIG/'fig_anderson_level_statistics.pdf'}")
    fig_geographic_modes(artifacts_by_city, FIG / "fig_anderson_geographic_modes")
    print(f"  ✓ {FIG/'fig_anderson_geographic_modes.pdf'}")
    fig_pr_distribution(artifacts_by_city, FIG / "fig_anderson_pr_distribution")
    print(f"  ✓ {FIG/'fig_anderson_pr_distribution.pdf'}")

    # ── Save eigenmode bundle ──────────────────────────────────────────
    np.savez_compressed(OUT / "d38_anderson_eigenmodes.npz", **npz_bundle)
    with open(OUT / "d38_anderson_summary.json", "w") as f:
        json.dump({
            "config": dict(K_NN=K_NN, SIGMA_M=SIGMA_M, W_DISORDER=W_DISORDER,
                           FEATS_IMD=FEATS_IMD,
                           R_GOE=R_GOE, R_POISSON=R_POISSON),
            "summary": rows,
            "wall_time_s": round(time.time()-t0, 1),
        }, f, indent=2)
    print(f"\n✓ Total wall time: {time.time()-t0:.1f}s")
    print(f"✓ CSV:  {OUT/'d38_anderson_per_city.csv'}")
    print(f"✓ NPZ:  {OUT/'d38_anderson_eigenmodes.npz'}")
    print(f"✓ JSON: {OUT/'d38_anderson_summary.json'}")


if __name__ == "__main__":
    main()
