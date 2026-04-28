# Chaotic Dynamics of the Damped, Driven Pendulum

Numerical exploration of nonlinear chaos in the driven, damped pendulum,
implemented in Python with a custom 4th-order Runge–Kutta integrator (RK4).
Includes phase-space portraits, Poincaré sections at multiple resolutions,
a bifurcation diagram, topological braid analysis, and a benchmarking
study of RK4 against the analytic mass-on-spring solution.

The accompanying paper (PDF) is in `docs/`.

## Repo layout

```
sections/
  01_mass_on_spring_benchmark.py    Benchmark RK4 against the SHM analytic solution
  02_energy_conservation.py         Energy conservation check
  03_phase_space_harmonic.py        Linear pendulum phase portrait
  04_pendulum_basics.py             Nonlinear pendulum
  05_pendulum_phase_space.py        Full phase space
  06_damped_pendulum_wide.py        Damped pendulum, wide initial conditions
  07_damped_pendulum_fine.py        Damped pendulum, fine grid
  08_driven_damped_pendulum_phase.py    Driven damped phase portrait
  09_poincare_section.py            Poincaré section (baseline)
  10_topological_braid.py           Topological braid / skeleton analysis
  11_bifurcation_diagram.py         Bifurcation diagram across drive amplitudes
  13_poincare_low_resolution.py     Poincaré (low-res, fast)
  14_poincare_super_resolution.py   Poincaré (high-res, slow)
  utils_and_functions.py            Shared RK4 + helpers
  performance_analysis.py           RK4 timing/accuracy analysis
  analyze_poincare_fidelity.py      Compare Poincaré sections across resolutions
  COMPLETE_PROJECT_REFERENCE.py     ← Canonical entry point. Full project code.
  COMPREHENSIVE_ANALYSIS_PUBLICATION.py    Publication-quality analysis

# Generated figures (kept in repo)
Benchmarking_Rk4.png
Benchmarking_Rk4_Error.png
Bifurcation_diagram.png
Phaseplot (reg pend).png
Phaseplot_damped_wide.png
Phaseplot_driven_damped.png
Topological_folding.png
Topological_skeleton.png
phase_portraits.gif
poincare_plot.png
performance_tradeoff.png

poincare_super_analysis.pdf       High-resolution Poincaré analysis
poincare_super_resolution.pdf
poincare_plot_low_def.pdf
poincare_analysis_results.txt     Numerical results

performance_*.json                 Per-script timing data
performance_all_data.csv           Aggregated timing data

docs/
  RK4_final.pdf                    Project writeup
```

## How to run

```
pip install numpy scipy matplotlib
cd sections
python COMPLETE_PROJECT_REFERENCE.py      # full project, end to end
```

To run individual analyses:

```
python 01_mass_on_spring_benchmark.py     # RK4 sanity check
python 09_poincare_section.py             # headline Poincaré section
python 11_bifurcation_diagram.py          # bifurcation diagram
```

## Notes

- The `*_resolution` Poincaré scripts trade off speed vs. fidelity; see
  `analyze_poincare_fidelity.py` for the comparison.
- All physics + numerics lives in `utils_and_functions.py`.
