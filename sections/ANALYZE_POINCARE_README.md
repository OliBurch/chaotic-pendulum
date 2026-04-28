# Poincaré Section Topological Fidelity Analysis

## Overview

`analyze_poincare_fidelity.py` is a comprehensive analysis tool that compares three different resolution levels of Poincaré section plots for the driven-damped pendulum system. It quantifies **topological fidelity** to determine which resolution provides the most reliable and accurate representation of the system's chaotic attractor.

---

## What Problem Does It Solve?

When simulating chaotic systems numerically, a critical question arises:
- **How many data points do I need to accurately capture the system's topology?**
- **Are my observations converging to the true attractor structure?**
- **How confident can I be in my analytical conclusions?**

This script answers these questions by comparing three independently-computed Poincaré sections and evaluating their mathematical agreement:

| Script | Time Resolution | Points Generated | Use Case |
|--------|-----------------|------------------|----------|
| **09** | 100 steps/period | ~50k points | Baseline/reference resolution |
| **12** | 200 steps/period | ~30k points | High-detail focused view |
| **14** | 250 steps/period | ~100k points | Publication-quality gold standard |

---

## System Configuration

All three scripts analyze the same physical system:
```
Driven-Damped Pendulum ODE:
  d²θ/dt² = -(g/ℓ)sin(θ) - β(dθ/dt) + A·cos(ω_d·t)

Parameters:
  Driving amplitude:    A = 1.5
  Driving frequency:    ω_d = 0.67 rad/s
  Damping coefficient:  β = 0.5
  
Poincaré Section:       Stroboscopic sampling at driving period T_d = 2π/ω_d
```

The system exhibits **chaotic behavior** with a strange attractor in phase space (θ, ω).

---

## Core Analysis Pipeline

### 1. **Data Extraction** → `extract_poincare_points()`

For each script, the analyzer:
- Numerically integrates the ODE using 4th-order Runge-Kutta (RK4)
- Runs for N driving periods with `steps` subdivisions per period
- Skips first `skip` periods to remove transient dynamics
- **Samples at the driving frequency** (Poincaré section definition)
- Wraps angle measurement to [-π, π] for consistency

**Output:** Two arrays containing ~thousands to ~100k points sampled at regular driving period intervals

---

### 2. **Spatial Analysis**

#### 2a. **Focus Region Counting** → `count_points_in_region()`
Counts points in the chaotic region where the attractor is densest:
- **Focus box:** θ ∈ [1.3, 1.8] rad, ω ∈ [-0.4, 0.3] rad/s
- Measures how well each resolution captures the main chaos zone
- **Metric:** Points and percentage of total points in region

#### 2b. **Phase Space Coverage** → `analyze_coverage()`
Computes what fraction of phase space is occupied:
- Divides phase space into 20×20 grid cells
- Counts occupied vs empty cells
- **Metric:** % of grid occupied
- Indicates how much of state space trajectory explores

---

### 3. **Topological Properties**

#### 3a. **Correlation Dimension** → `estimate_correlation_dimension()`

Measures the fractal dimension D₂ of the attractor using the correlation integral method:

$$D_2 = \lim_{r \to 0} \frac{\log C(r)}{\log r}$$

Where C(r) = fraction of point pairs separated by distance < r

**Algorithm:**
- Sample up to 5000 points randomly
- Compute pairwise distances between all points
- Count pairs at different radius thresholds
- Fit log-log slope to get D₂

**Interpretation:**
- D₂ ≈ 0.8 indicates a **thin, folded 1D-like structure** (expected for this system)
- **Low-dimensional chaos** (not random, not fully 2D)
- **Convergence check:** All three scripts should yield very similar D₂ values
  - If they differ by <0.02, measurements are trustworthy
  - If they diverge, resolution is insufficient

**Physics:** The chaotic attractor forms a Cantor set-like structure with self-similar folding.

---

#### 3b. **Lyapunov Exponent** → `compute_lyapunov_exponent_estimate()`

Estimates the largest Lyapunov exponent λ (chaos indicator):

$$\lambda = \lim_{t \to \infty} \frac{1}{t} \log \frac{|\delta x(t)|}{|\delta x(0)|}$$

Measures exponential divergence of nearby trajectories.

**Algorithm:**
- For each point, find nearby neighbors
- Compute log-distances to those neighbors
- Average divergence rate across all pairs
- **Fast approximation** (not full Lyapunov computation)

**Interpretation:**
- λ > 0 ⟹ **Chaotic dynamics** (trajectories diverge exponentially)
- λ < 0 ⟹ Regular/stable dynamics (dissipative, attract to fixed point/cycle)
- λ ≈ 0.5 ⟹ Strong chaos for this system

**Consistency Check:** All three resolutions should agree on sign (all positive = chaotic).

---

#### 3c. **Clustering Analysis** → `detect_clusters()`

Identifies distinct clusters of points using connected components algorithm:
- Groups points within radius r=0.15 as "neighbors"
- Builds clusters via breadth-first search
- Reports number of clusters and size distribution

**Why it matters:**
- Chaotic attractors often have **multiple branches** or **layers**
- These structures should be consistent across resolutions
- Clustering pattern reveals saddle point topology

**Expected result:** 2 large clusters (two main bifurcation branches of the attractor)

---

## Fidelity Scoring System

The script generates a **composite fidelity score** (0-100) based on four criteria:

### Criterion 1: Point Density Improvement (max +30 pts)
```
Ratio = Total_Points(Script14) / Total_Points(Script09)

✓✓ Excellent (30 pts):  ratio ≥ 1.8
✓  Good      (25 pts):  ratio ≥ 1.5
```
**Goal:** Script 14 should capture ~2× more points than baseline.

### Criterion 2: Dimension Convergence (max +30 pts)
```
Agreement = 1 - |D₂(Script14) - D₂(Script09)| / |D₂(Script09)|

✓✓ Excellent (30 pts):  agreement > 0.95 (difference < 5%)
✓  Good      (20 pts):  agreement > 0.85 (difference < 15%)
```
**Goal:** All resolutions measure the same fractal dimension (≤0.02 difference).

### Criterion 3: Phase Space Coverage (max +25 pts)
```
Coverage_Improvement = Coverage(Script14) - Coverage(Script09)

✓✓ Excellent (25 pts):  improvement ≥ 10%
✓  Good      (15 pts):  improvement ≥ 5%
⚠  Fair      (5 pts):   improvement < 5%
```
**Goal:** Higher resolution explores more of the attractor.

### Criterion 4: Chaos Characterization (max +15 pts)
```
Consistency = (λ₉ > 0) AND (λ₁₂ > 0) AND (λ₁₄ > 0)

✓✓ Excellent (15 pts):  All positive (all chaotic)
✓  Good      (10 pts):  Consistent sign across scripts
```
**Goal:** All resolutions confirm chaotic behavior.

---

## Interpretation Guide

### Score ≥ 90/100: **EXCELLENT** ✓✓✓
- **Verdict:** Script 14 captures attractor topology with high fidelity
- **Use for:** Publication-quality analysis, PhD theses, peer review
- **Confidence:** Very high — results reproducible at higher resolution

### Score 80-89/100: **VERY GOOD** ✓✓
- **Verdict:** Strong convergence across all resolutions
- **Use for:** Research papers, rigorous dynamical systems analysis
- **Confidence:** High — dimension and chaos confirmed

### Score 70-79/100: **GOOD** ✓
- **Verdict:** Convergence observed; consistent with baseline
- **Use for:** Technical reports, presentations, conference papers
- **Confidence:** Moderate — recommend comparing Script 14 and 09 carefully

### Score < 70/100: **FAIR/LOW** ⚠
- **Verdict:** Results sensitive to resolution choice
- **Use for:** Exploratory analysis, undergraduate work
- **Confidence:** Low — consider increasing resolution further

---

## Example Output Interpretation

```
┌─────────────┬──────────┬──────────┬──────────┐
│ Property    │ Script09 │ Script12 │ Script14 │
├─────────────┼──────────┼──────────┼──────────┤
│ Total Pts   │    49900 │    29950 │    99950 │
│ D_2         │    0.818 │    0.800 │    0.807 │  ← Excellent agreement!
│ λ estimate  │   0.4613 │   0.4400 │   0.5096 │  ← All positive (chaotic)
│ Coverage %  │     22.2 │     22.2 │     22.2 │  ← Same regions explored
└─────────────┴──────────┴──────────┴──────────┘

Key observations:
  • D_2 values differ by only 0.018 (98% agreement)
  • All λ values > 0 (consistently chaotic)
  • Script 14 has 2× the points of Script 09
  → VERDICT: High topological fidelity (score ≈ 80/100)
```

---

## How to Run

### Quick Run (foreground):
```bash
cd "/Users/oliburch/python code for pendulum/sections"
python analyze_poincare_fidelity.py
```

### Background Run (recommended for long jobs):
```bash
cd "/Users/oliburch/python code for pendulum/sections"
nohup python analyze_poincare_fidelity.py > fidelity_results.txt 2>&1 &
```

### Watch Progress:
```bash
tail -f fidelity_results.txt
```

---

## Runtime Expectations

| Phase | Duration | Activity |
|-------|----------|----------|
| Script 09 extraction | 2-3 min | Integrates 5M timesteps |
| Script 12 extraction | 2-3 min | Integrates 6M timesteps |
| Script 14 extraction | 8-10 min | Integrates 25M timesteps |
| Dimension estimation | 3-5 min | Distance calculations |
| Other analysis | 1-2 min | Clustering, coverage |
| **Total** | **18-25 min** | All analysis complete |

**Typical machine:** ~20 minutes on Apple Silicon M1/M2

---

## Clinical Notes for Troubleshooting

### Problem: "Dimension values differ by >0.05"
- **Cause:** Insufficient transient skip; attractor not fully settled
- **Fix:** Increase `skip` parameter in Scripts 09/12/14 to 100+ periods

### Problem: "Coverage is <20%; many empty grid cells"
- **Cause:** Attractor is thin/sparse in phase space (normal!)
- **Fix:** Don't increase coverage expectation — this is correct behavior

### Problem: "Lyapunov exponent is negative!"
- **Cause:** Parameters don't produce chaos (try A=1.5 exactly)
- **Fix:** Verify physical parameters match literature values

### Problem: "Script 14 takes >30 minutes"
- **Cause:** Step count or period count too high
- **Fix:** Reduce N or steps (be aware this reduces resolution)

---

## Mathematical References

This analysis uses standard tools from nonlinear dynamics:

1. **Poincaré Sections** - Stroboscopic sampling method for periodic forcing
2. **Correlation Dimension (D₂)** - Grassberger & Procaccia (1983)
3. **Lyapunov Exponents** - Benettin et al. (1980)
4. **Strange Attractors** - Ruelle & Takens (1971)

Key textbook: *Nonlinear Dynamics and Chaos* by Steven Strogatz

---

## Output Files Generated

The script prints detailed output to terminal and optionally to file:

- **fidelity_results.txt** — Complete analysis report with all metrics
- **Summary statistics** — Point counts, dimensions, coverage percentages
- **Comparison tables** — Side-by-side metrics for all three scripts

To save output:
```bash
python analyze_poincare_fidelity.py > analysis_report.txt 2>&1
```

---

## Key Takeaway

**This tool answers:** *"Can I trust my Poincaré section plots to capture the true attractor topology?"*

By running THREE different resolutions and comparing them, we gain confidence that the observed structures are **robust mathematical features** rather than **numerical artifacts**. A high fidelity score means you can write physics papers with confidence!

---

**Last Updated:** March 2026  
**Script Version:** v1.0 (3-script comparison)  
**Status:** Production-ready
