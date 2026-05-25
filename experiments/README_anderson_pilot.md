# F5 — Anderson localization pilot

Three-script pipeline to test whether urban mobility graphs exhibit
Anderson-localization phenomenology, which would mechanistically
explain the collapse of `R²_spec` on certain cities (the structural
applicability bound).

## Pipeline

| Script | What it does | Wall time (typical) |
|---|---|---|
| `d38_anderson_pilot.py` | Builds H = L_sym + W·diag(ε_IMD) on 9 cities, computes IPR, level statistics, geographic maps. Outputs a binary verdict: SIGNAL PRESENT / ABSENT. | 1–3 min |
| `d39_anderson_disorder_sweep.py` | Sweeps disorder W ∈ [0, 10] on Boston, maps the level-statistics transition (GOE → Poisson), runs finite-size scaling. Identifies the critical W_c. | 3–8 min |
| `d40_anderson_bound_bridge.py` | Tests the theoretical bridge: does the Anderson observable `d_eff/d` correlate with the observed `R²_spec` ceiling? This is the headline claim of the paper. | 1–3 min |

## Running

```bash
cd /Users/rfosse/cesi-research/bikeshare-demand-forecasting/experiments
python d38_anderson_pilot.py        # → verdict
python d39_anderson_disorder_sweep.py
python d40_anderson_bound_bridge.py
```

All three are self-contained and reuse the existing graph conventions
of `d24_gsp_real_cities.py` and `d28_spectral_bottleneck.py`
(k-NN = 6, σ = 300 m, symmetric-normalised Laplacian, IMD-4 features).

## Decision tree after running

| d38 verdict | d39 result | d40 result | Action |
|---|---|---|---|
| SIGNAL PRESENT | clear transition near W_c | ρ(d_eff, R²_spec) ≥ +0.6 (p<0.05) | **Go.** Bootstrap F5 paper from `paper-template/` |
| SIGNAL PRESENT | crossover only (no sharp transition) | ρ ≥ +0.5 | Soft go — vend l'angle "Anderson-like crossover", venue PRE plutôt que Nature Physics |
| SIGNAL ABSENT | — | — | **Pivot to F1** (percolation) using `d28_spectral_bottleneck` extended |
| Mixed | — | ρ low or negative | The bridge to R²_spec fails — F5 abandoned, keep observations as a methodological note |

## Key outputs to inspect first

After running:

- `figures/fig_anderson_ipr_vs_energy.pdf` — the headline plot. If you
  see IPR rising at band edges or above the 5/N line, localization is
  present.
- `figures/fig_anderson_level_statistics.pdf` — if the histograms lean
  toward Poisson (red curve) for some cities and GOE (green curve) for
  others, the phase is heterogeneous across cities — perfect material
  for the paper.
- `figures/fig_anderson_phase_diagram.pdf` — should show <r> dropping
  from GOE to Poisson as W increases. If the drop is sharp and
  size-independent (in the FSS figure), it's a true transition.
- `figures/fig_anderson_bound_bridge.pdf` — the theoretical payoff. If
  d_eff/d predicts R²_spec, the Anderson framing is fully validated.

## Next step if positive

```bash
cp -r /Users/rfosse/cesi-research/paper-template \
      /Users/rfosse/cesi-research/anderson-localization-mobility
```

Then port the figures and write the paper around the three-figure
narrative:

1. *Phenomenology* — IPR maps on real cities (from d38)
2. *Phase diagram* — disorder-induced transition (from d39)
3. *Bridge to applicability bound* — Anderson predicts predictability (from d40)
