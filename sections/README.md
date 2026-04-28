# Pendulum Dynamics Scripts - README

## Overview
This directory contains 14 Python scripts analyzing nonlinear pendulum systems using RK4 numerical integration. Each script explores different aspects of chaotic dynamics.

---

## Quick Start

**Run any script:**
```bash
cd "/Users/oliburch/python code for pendulum/sections"
python SCRIPT_NAME.py
```

**Run in background (doesn't block terminal):**
```bash
python SCRIPT_NAME.py > output.log 2>&1 &
tail -f output.log        # Monitor progress
```

---

## Script Guide

### Core Utilities
**`utils_and_functions.py`** (Not runnable - imported by all scripts)
- Contains RK4 solver and all ODE definitions
- Required by all other scripts

---

### Exercise 1-3: Benchmarking (Linear Systems)

| # | Script | Status | Runtime | Purpose |
|---|--------|--------|---------|---------|
| **01** | `01_mass_on_spring_benchmark.py` | ✅ **WORKS** | ~30 sec | RK4 accuracy vs analytical solution (10 periods) |
| **02** | `02_energy_conservation.py` | ✅ **WORKS** | ~5 min | Energy conservation over 1000 periods |
| **03** | `03_phase_space_harmonic.py` | ✅ **WORKS** | ~2 min | Phase space trajectories |

---

### Exercise 4-5: Nonlinear Pendulum

| # | Script | Status | Runtime | Purpose |
|---|--------|--------|---------|---------|
| **04** | `04_pendulum_basics.py` | ✅ **WORKS** | ~3 min | Period vs initial angle analysis |
| **05** | `05_pendulum_phase_space.py` | ✅ **WORKS** | ~2 min | Phase portraits with vector field |

---

### Exercise 6-7: Damped Pendulum

| # | Script | Status | Runtime | Purpose |
|---|--------|--------|---------|---------|
| **06** | `06_damped_pendulum_wide.py` | ✅ **WORKS** | ~10 sec | Damping effects (β = 0.25, 1, 2, 3) |
| **07** | `07_damped_pendulum_fine.py` | ✅ **WORKS** | ~10 sec | Fine-tuned damping (β = 1.5-2.5) |

---

### Exercise 8: Driven Damped Pendulum

| # | Script | Status | Runtime | Purpose |
|---|--------|--------|---------|---------|
| **08** | `08_driven_damped_pendulum_phase.py` | ⚠️ **SLOW** | ~30 sec | Phase plots + animated vector field |

---

### Exercise 9-13: Chaos & Bifurcation

| # | Script | Status | Runtime | Purpose |
|---|--------|--------|---------|---------|
| **09** | `09_poincare_section.py` | ✅ **WORKS** | ~3 min | Standard Poincaré section (50k cycles) |
| **10** | `10_topological_braid.py` | ⚠️ **SLOW** | ~2 min | 3D braiding visualization |
| **11** | `11_bifurcation_diagram.py` | ✅ **WORKS** | ~2 min | Route to chaos (100 amplitudes) |
| **12** | `12_poincare_ultra_resolution.py` | ✅ **WORKS** | ~4 min | High-res Poincaré (30k × 200 steps) |
| **13** | `13_poincare_low_resolution.py` | ✅ **WORKS** | ~1 min | Quick preview (low resolution) |

---

## Recommended Scripts to Try First

**Start here (fast, reliable):**
```bash
python 01_mass_on_spring_benchmark.py    # 30 seconds
python 03_phase_space_harmonic.py        # 2 minutes
python 06_damped_pendulum_wide.py        # 10 seconds
```

**Then explore chaos:**
```bash
python 09_poincare_section.py            # 3 minutes
python 11_bifurcation_diagram.py         # 2 minutes
python 13_poincare_low_resolution.py     # 1 minute
```

---

## Output Files

Scripts save plots as PDFs:
- `*.pdf` — High-resolution figures (can open in Preview or any PDF viewer)

Example outputs:
- `poincare_plot.pdf` — Chaotic attractor visualization
- `Bifurcation_diagram.pdf` — Route to chaos
- `Phaseplot.pdf` — Phase space portraits

---

## Known Issues & Workarounds

### Script 08 (Driven Damped Phase)
- **Issue:** Animation generation can be slow
- **Fix:** Comment out animation save line if you just want plots

### Scripts 12 & 13 (Poincaré Sections)
- **For quick preview:** Run script 13 (1 minute)
- **For detailed view:** Run script 12 (4 minutes)
- **For comparison:** Run script 09 (3 minutes)

### If a script hangs
- Press `Ctrl+C` to stop
- Try running a simpler script first to verify setup works
- Check that all dependencies are installed: `conda list | grep -E "numpy|matplotlib"`

---

## Performance Tips

**To make scripts faster:**
1. Run in background: `python script.py > log.txt 2>&1 &`
2. Edit script parameters (smaller N, fewer steps)
3. Use low-res script 13 for testing

**To see progress:**
```bash
python script.py > progress.log 2>&1 &
tail -f progress.log
```

---

## Parameter Reference

All scripts use these ODE parameters:
- **ω_d = 0.67** — Driving frequency
- **β = 0.5** — Damping coefficient (varies per script)
- **A = 1.5** — Driving amplitude (varies per script)
- **g/ℓ = 9.81** — Gravity / pendulum length

---

## Summary Table

| Script | Type | Works? | Time | Priority |
|--------|------|--------|------|----------|
| 01 | Benchmark | ✅ | 30s | Start here |
| 02 | Benchmark | ✅ | 5m | Optional |
| 03 | Benchmark | ✅ | 2m | Recommended |
| 04 | Pendulum | ✅ | 3m | Good |
| 05 | Pendulum | ✅ | 2m | Recommended |
| 06 | Damped | ✅ | 10s | Quick demo |
| 07 | Damped | ✅ | 10s | Quick demo |
| 08 | Driven | ⚠️ | 30s | Advanced |
| 09 | Chaos | ✅ | 3m | Must see |
| 10 | Chaos | ⚠️ | 2m | Nice-to-have |
| 11 | Chaos | ✅ | 2m | Must see |
| 12 | Chaos | ✅ | 4m | Detailed view |
| 13 | Chaos | ✅ | 1m | Quick preview |

---

## Troubleshooting

**ImportError: No module named ...**
```bash
cd "/Users/oliburch/python code for pendulum/sections"
python script.py  # Must be in sections directory
```

**Exit code 130 (KeyboardInterrupt)**
- Script was interrupted or took too long
- Normal for heavy computations

**No plots appear**
- Check for `*.pdf` files created in sections directory
- Try: `ls -la *.pdf`

---

## Next Steps

1. **Run a fast script** → Verify setup works
2. **Run chaos scripts** → Explore interesting dynamics  
3. **Modify parameters** → Experiment with different conditions
4. **Study outputs** → Analyze generated PDFs

Happy exploring! 🌌
