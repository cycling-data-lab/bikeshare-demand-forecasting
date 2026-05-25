"""
d39_anderson_disorder_sweep.py — Phase transition validation.

If d38 confirms a localization signature at the natural disorder
strength (W=1), d39 maps the phase diagram by sweeping W ∈ [0, W_max]
on a fixed urban graph.  The expected behaviour for a genuine Anderson
transition (2D effective):

  • W = 0      : <r> → GOE (0.5295), all states extended (IPR ≈ 1/N)
  • W small    : band-edge states begin localizing (IPR rises at edges)
  • W ≈ W_c    : mobility edge sweeps to band center, <r> transitions
  • W >> W_c   : all states localized, <r> → Poisson (0.3863)

We sweep W on three subjects:

  1. A clean reference: the bare city graph with synthetic Gaussian
     disorder (no IMD features). This gives the "textbook" Anderson
     curve for the given graph topology — establishes that the graph
     CAN host a transition in principle.

  2. The IMD-driven disorder: each W rescales the natural IMD potential.
     The question is whether the natural urban heterogeneity sits
     in the extended, transition, or localized regime.

  3. Finite-size scaling on graph sub-samples (50%, 75%, 100% of
     stations) to detect whether the transition is sharp (true phase
     transition) or a crossover (size-dependent).

Output:
  outputs/d39_anderson_sweep.csv
  outputs/d39_anderson_sweep.json
  figures/fig_anderson_phase_diagram.{pdf,png}
  figures/fig_anderson_finite_size_scaling.{pdf,png}
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

# ── Subjects ────────────────────────────────────────────────────────────
# Sweep on Paris: largest GOE-starting city (mean degree 2.81, <r>(L)=0.49).
# Boston (the first attempt) was already past the localization transition
# at W=0 due to its sparse graph (mean degree 1.62), so no clean phase
# transition could be observed.  Paris gives the cleanest GOE→Poisson curve.
SUBJECT_CITY = ("tier2_paris", "world_fr_v_lib_metropole", "Vélib Paris")

# Sweep grid: log-spaced from very weak to strongly disordered
W_GRID = np.concatenate([
    np.array([0.0]),
    np.logspace(np.log10(0.05), np.log10(10.0), 24),
])

# Finite-size sub-samples (Paris N=1511, so diag is O(N³) ≈ 5-10s each)
FSS_FRACTIONS = [0.40, 0.70, 1.00]
N_FSS_REPS = 3

# Graph parameters identical to d38
K_NN = 6
SIGMA_M = 300.0
EARTH_R = 6_371_000.0
FEATS_IMD = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
             "elevation_m", "topography_roughness_index"]

R_GOE = 0.5295
R_POISSON = 0.3863


# ── Graph & observables (mirror d38) ────────────────────────────────────
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


def composite_potential(imd_df: pd.DataFrame) -> np.ndarray:
    avail = [f for f in FEATS_IMD if f in imd_df.columns]
    if not avail: return np.zeros(len(imd_df))
    X = imd_df[avail].astype(float).values
    X = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-12)
    X = np.nan_to_num(X, nan=0.0)
    eps = X.mean(axis=1)
    eps = eps - eps.mean()
    return eps / (eps.std() + 1e-12)


def ipr_cols(psi):
    psi2 = psi**2
    psi2 = psi2 / (psi2.sum(axis=0, keepdims=True) + 1e-30)
    return (psi2**2).sum(axis=0)


def gap_ratio(eigvals):
    s = np.diff(np.sort(eigvals))
    return np.minimum(s[:-1], s[1:]) / (np.maximum(s[:-1], s[1:]) + 1e-30)


# ── Sweep core ──────────────────────────────────────────────────────────
def sweep_one(L_sym, potential, W_grid):
    """Diagonalise H(W) = L_sym + W·diag(potential) for each W.
    Returns a DataFrame with one row per W."""
    N = L_sym.shape[0]
    out = []
    for W in W_grid:
        H = L_sym + W * np.diag(potential) if W > 0 else L_sym.copy()
        eigvals, eigvecs = np.linalg.eigh(H)
        ipr_vals = ipr_cols(eigvecs)
        r = gap_ratio(eigvals)
        out.append(dict(
            W=float(W),
            ipr_mean=float(ipr_vals.mean()),
            ipr_median=float(np.median(ipr_vals)),
            ipr_max=float(ipr_vals.max()),
            frac_localized=float((ipr_vals > 5.0/N).mean()),
            r_mean=float(r.mean()),
            r_GOE_dist=abs(r.mean() - R_GOE),
            r_Poisson_dist=abs(r.mean() - R_POISSON),
            lambda_min=float(eigvals[0]),
            lambda_max=float(eigvals[-1]),
            spectral_width=float(eigvals[-1] - eigvals[0]),
        ))
    return pd.DataFrame(out)


def find_W_critical(df, target=0.5*(R_GOE + R_POISSON)):
    """Critical W where <r> crosses the midpoint between GOE and Poisson.
    Uses a centred rolling mean with min_periods=1 (no zero-padding
    artefact at the edges)."""
    rs_raw = df["r_mean"].values
    Ws = df["W"].values
    win = min(5, len(rs_raw))
    # Pandas rolling avoids the zero-padding edge artefact of np.convolve
    rs = pd.Series(rs_raw).rolling(window=win, min_periods=1, center=True).mean().values
    # Require the crossing to be SUSTAINED: the next 2 points should also
    # be below target.  This rejects single-point dips at low W.
    below = rs < target
    if not below.any(): return float("nan"), float("nan")
    sustained = np.array([below[i] and (i+1 >= len(below) or below[i+1])
                          for i in range(len(below))])
    if not sustained.any(): return float("nan"), float("nan")
    idx = int(np.argmax(sustained))
    if idx == 0:
        # Even the first W is sustainedly below — system starts past transition
        return float(Ws[0]), float(rs[0])
    W1, W2 = Ws[idx-1], Ws[idx]
    r1, r2 = rs[idx-1], rs[idx]
    if r2 == r1: return float(W2), float(r2)
    W_c = W1 + (W2 - W1) * (target - r1) / (r2 - r1)
    return float(W_c), float(target)


def find_W_critical_by_ipr(df, target_frac=0.5):
    """Critical W where the fraction of localized states (IPR > 5/N)
    crosses `target_frac`.  This is a cleaner observable than <r> on
    sparse urban graphs."""
    fl = df["frac_localized"].values
    Ws = df["W"].values
    above = fl > target_frac
    if not above.any(): return float("nan")
    idx = int(np.argmax(above))
    if idx == 0: return float(Ws[0])
    W1, W2 = Ws[idx-1], Ws[idx]
    f1, f2 = fl[idx-1], fl[idx]
    if f2 == f1: return float(W2)
    return float(W1 + (W2 - W1) * (target_frac - f1) / (f2 - f1))


# ── Subject loaders ─────────────────────────────────────────────────────
def load_subject_city(slug, stem):
    imd_path = IMD_INTL_DIR / f"{stem}.parquet"
    if not imd_path.exists():
        raise FileNotFoundError(f"No IMD at {imd_path}")
    imd = pd.read_parquet(imd_path).dropna(subset=["lat","lng"]).reset_index(drop=True)
    return imd


# ── Main ────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    slug, stem, pretty = SUBJECT_CITY
    print(f"=== Disorder sweep on {pretty} ===")
    imd = load_subject_city(slug, stem)
    N = len(imd)
    print(f"  N = {N} stations")
    print(f"  W grid: {len(W_GRID)} values from 0 to {W_GRID.max():.2f}")

    # Build base Laplacian
    L_sym = build_L_sym(imd["lat"].values, imd["lng"].values)
    eps_imd = composite_potential(imd)

    # Synthetic Gaussian disorder (clean reference)
    rng = np.random.default_rng(42)
    eps_synth = rng.standard_normal(N)
    eps_synth = (eps_synth - eps_synth.mean()) / (eps_synth.std() + 1e-12)

    print("\n--- (1) Sweep with IMD-driven disorder ---")
    df_imd = sweep_one(L_sym, eps_imd, W_GRID)
    df_imd["disorder_type"] = "IMD_composite"
    W_c_imd, r_at_c_imd = find_W_critical(df_imd)
    W_c_imd_fl = find_W_critical_by_ipr(df_imd)
    print(f"  W_c (IMD, <r> midpoint, smoothed) ≈ {W_c_imd:.3f}")
    print(f"  W_c (IMD, frac_localized=0.5)     ≈ {W_c_imd_fl:.3f}")

    print("\n--- (2) Sweep with synthetic Gaussian disorder ---")
    df_synth = sweep_one(L_sym, eps_synth, W_GRID)
    df_synth["disorder_type"] = "synthetic_Gaussian"
    W_c_synth, r_at_c_synth = find_W_critical(df_synth)
    W_c_synth_fl = find_W_critical_by_ipr(df_synth)
    print(f"  W_c (Gaussian, <r> midpoint, smoothed) ≈ {W_c_synth:.3f}")
    print(f"  W_c (Gaussian, frac_localized=0.5)     ≈ {W_c_synth_fl:.3f}")

    # Finite-size scaling on synthetic disorder (cleaner signal)
    print("\n--- (3) Finite-size scaling (synthetic disorder) ---")
    fss_rows = []
    for frac in FSS_FRACTIONS:
        for rep in range(N_FSS_REPS):
            idx = rng.choice(N, size=int(frac*N), replace=False)
            sub = imd.iloc[idx].reset_index(drop=True)
            L_sub = build_L_sym(sub["lat"].values, sub["lng"].values)
            eps_sub = rng.standard_normal(len(sub))
            eps_sub = (eps_sub - eps_sub.mean()) / (eps_sub.std() + 1e-12)
            df_sub = sweep_one(L_sub, eps_sub, W_GRID)
            df_sub["fraction"] = frac
            df_sub["rep"] = rep
            df_sub["N_sub"] = len(sub)
            fss_rows.append(df_sub)
            print(f"  frac={frac:.2f} rep={rep} N_sub={len(sub)}  "
                  f"<r>(W=max)={df_sub['r_mean'].iloc[-1]:.3f}")
    df_fss = pd.concat(fss_rows, ignore_index=True)

    # ── Save ─────────────────────────────────────────────────────────────
    df_all = pd.concat([df_imd, df_synth], ignore_index=True)
    df_all.to_csv(OUT / "d39_anderson_sweep.csv", index=False)
    df_fss.to_csv(OUT / "d39_anderson_sweep_fss.csv", index=False)

    # ── Phase diagram figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ax = axes[0]
    ax.semilogx(df_imd["W"], df_imd["r_mean"], "o-", color="C0",
                label="IMD-driven disorder", markersize=4, lw=1.2)
    ax.semilogx(df_synth["W"], df_synth["r_mean"], "s-", color="C1",
                label="Synthetic Gaussian", markersize=4, lw=1.2)
    ax.axhline(R_GOE, color="green", ls="--", lw=0.8, label=f"GOE = {R_GOE}")
    ax.axhline(R_POISSON, color="red", ls="--", lw=0.8, label=f"Poisson = {R_POISSON}")
    ax.axvline(W_c_imd, color="C0", ls=":", lw=0.7, alpha=0.7)
    ax.axvline(W_c_synth, color="C1", ls=":", lw=0.7, alpha=0.7)
    ax.set_xlabel("Disorder strength W")
    ax.set_ylabel("<r> (mean adjacent gap ratio)")
    ax.set_title(f"{pretty}: level-statistics transition\n"
                 f"$W_c^{{IMD}}$≈{W_c_imd:.2f}, $W_c^{{synth}}$≈{W_c_synth:.2f}")
    ax.set_xlim(W_GRID[1]/2, W_GRID[-1]*1.2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    ax = axes[1]
    ax.semilogx(df_imd["W"], df_imd["frac_localized"], "o-", color="C0",
                label="IMD-driven", markersize=4, lw=1.2)
    ax.semilogx(df_synth["W"], df_synth["frac_localized"], "s-", color="C1",
                label="Gaussian", markersize=4, lw=1.2)
    ax.set_xlabel("Disorder strength W")
    ax.set_ylabel("Fraction localized states (IPR > 5/N)")
    ax.set_title("Localized fraction vs disorder")
    ax.set_xlim(W_GRID[1]/2, W_GRID[-1]*1.2); ax.set_ylim(0, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    fig.suptitle("Anderson phase diagram on a real urban mobility graph",
                 fontsize=11)
    fig.savefig(FIG / "fig_anderson_phase_diagram.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_anderson_phase_diagram.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  ✓ {FIG/'fig_anderson_phase_diagram.pdf'}")

    # ── Finite-size scaling figure ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    grp = df_fss.groupby(["fraction", "W"])["r_mean"].agg(["mean", "std"]).reset_index()
    cmap = plt.cm.viridis
    fracs = sorted(grp["fraction"].unique())
    for i, frac in enumerate(fracs):
        g = grp[grp["fraction"] == frac]
        col = cmap(i / max(1, len(fracs)-1))
        ax.semilogx(g["W"], g["mean"], "o-", color=col, lw=1.2, markersize=4,
                    label=f"frac={frac:.2f}")
        ax.fill_between(g["W"], g["mean"]-g["std"], g["mean"]+g["std"],
                        color=col, alpha=0.20)
    ax.axhline(R_GOE, color="green", ls="--", lw=0.7)
    ax.axhline(R_POISSON, color="red", ls="--", lw=0.7)
    ax.set_xlabel("Disorder strength W")
    ax.set_ylabel("<r> (mean of N_FSS_REPS reps)")
    ax.set_title(f"Finite-size scaling: {pretty}\n"
                 "Curve sharpening with N suggests a true transition; size-dependence suggests crossover.")
    ax.set_xlim(W_GRID[1]/2, W_GRID[-1]*1.2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.savefig(FIG / "fig_anderson_finite_size_scaling.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_anderson_finite_size_scaling.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  ✓ {FIG/'fig_anderson_finite_size_scaling.pdf'}")

    # ── Summary JSON ─────────────────────────────────────────────────────
    summary = dict(
        config=dict(K_NN=K_NN, SIGMA_M=SIGMA_M, W_GRID=W_GRID.tolist(),
                    FEATS_IMD=FEATS_IMD, FSS_FRACTIONS=FSS_FRACTIONS,
                    N_FSS_REPS=N_FSS_REPS, R_GOE=R_GOE, R_POISSON=R_POISSON),
        subject=dict(slug=slug, pretty=pretty, N=N),
        W_critical=dict(
            IMD_composite=W_c_imd, IMD_r_at_W_c=r_at_c_imd,
            IMD_by_frac_localized=W_c_imd_fl,
            synthetic_Gaussian=W_c_synth, synthetic_r_at_W_c=r_at_c_synth,
            synthetic_by_frac_localized=W_c_synth_fl),
        wall_time_s=round(time.time()-t0, 1),
    )
    with open(OUT / "d39_anderson_sweep.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Total wall time: {time.time()-t0:.1f}s")
    print(f"✓ CSV: {OUT/'d39_anderson_sweep.csv'}")
    print(f"✓ FSS: {OUT/'d39_anderson_sweep_fss.csv'}")
    print(f"✓ JSON: {OUT/'d39_anderson_sweep.json'}")


if __name__ == "__main__":
    main()
