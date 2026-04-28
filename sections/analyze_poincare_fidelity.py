"""
Poincaré Section Topological Fidelity Analysis
Counts points and evaluates the resolution/coverage of both plots
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from utils_and_functions import RK4, DrivenDampedPendulum
from scipy.spatial.distance import cdist
from scipy.special import comb

# ============================================================================
# DUAL LOGGER SETUP
# ============================================================================
class DualLogger(object):
    """
    Custom logger that writes to both the terminal (stdout) and a text file.
    """
    def __init__(self, filename="poincare_analysis_results.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility
        # and to handle the `flush=True` in your print statements
        self.terminal.flush()
        self.log.flush()

# ============================================================================
# POINCARE POINT EXTRACTION
# ============================================================================

def extract_poincare_points(N, steps, skip, A_val, omega_d, beta_val):
    """
    Generate Poincaré section points for given parameters
    Returns: (theta_points, omega_points, num_periods)
    """
    dt = 2 * np.pi / (omega_d * steps)
    time = np.arange(0, N * 2 * np.pi / omega_d, dt)
    nsteps = len(time)
    
    omega = np.zeros(nsteps)
    theta = np.zeros(nsteps)
    omega[0] = 0.0
    theta[0] = 0.0
    
    print(f"  Integrating {nsteps} steps (~{nsteps*dt/(2*np.pi):.0f} periods)...", end='', flush=True)
    
    for j in range(nsteps - 1):
        x_v_current = np.array([theta[j], omega[j]])
        x_v_next = RK4(time[j], x_v_current,
                       lambda t, y: DrivenDampedPendulum(t, y, beta_val, A_val),
                       dt)
        theta[j+1] = x_v_next[0]
        omega[j+1] = x_v_next[1]
    
    # Remove transient
    theta_clean = theta[skip*steps:]
    omega_clean = omega[skip*steps:]
    
    # Extract Poincaré points (stroboscopic sampling)
    offset = steps // 4
    theta_p = theta_clean[offset :: steps]
    omega_p = omega_clean[offset :: steps]
    
    # Wrap theta to [-π, π]
    theta_p_wrapped = (theta_p + np.pi) % (2 * np.pi) - np.pi
    
    num_periods = len(theta_p)
    
    print(f" Done!")
    return theta_p_wrapped, omega_p, num_periods


# ============================================================================
# POINT COUNTING & DENSITY ANALYSIS
# ============================================================================

def count_points_in_region(theta, omega, theta_range, omega_range):
    """Count points within specified θ and ω ranges"""
    mask = (theta >= theta_range[0]) & (theta <= theta_range[1]) & \
           (omega >= omega_range[0]) & (omega <= omega_range[1])
    return np.sum(mask)


def compute_point_density(theta, omega, grid_size=20):
    """
    Divide space into grid and compute density
    Returns grid data for visualization
    """
    theta_edges = np.linspace(-np.pi, np.pi, grid_size + 1)
    omega_min, omega_max = omega.min(), omega.max()
    omega_edges = np.linspace(omega_min, omega_max, grid_size + 1)
    
    density = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            count = count_points_in_region(
                theta, omega,
                (theta_edges[i], theta_edges[i+1]),
                (omega_edges[j], omega_edges[j+1])
            )
            density[i, j] = count
    
    return density, theta_edges, omega_edges


# ============================================================================
# TOPOLOGICAL PROPERTIES
# ============================================================================

def estimate_correlation_dimension(theta, omega, max_samples=5000):
    """
    Estimate correlation dimension using correlation integral
    D_2 ≈ log(C(r)) / log(r) for small r
    """
    if len(theta) > max_samples:
        idx = np.random.choice(len(theta), max_samples, replace=False)
        theta_sample = theta[idx]
        omega_sample = omega[idx]
    else:
        theta_sample = theta
        omega_sample = omega
    
    # Stack points
    points = np.column_stack([theta_sample, omega_sample])
    
    # Compute distances
    distances = cdist(points, points)
    
    # Test different radius values
    r_values = np.logspace(-2, 0, 20)
    C_r = []
    
    for r in r_values:
        # Count pairs within radius r (excluding self-pairs)
        count = np.sum(distances < r) - len(points)
        C_r.append(max(count, 1))  # Avoid log(0)
    
    C_r = np.array(C_r)
    
    # Linear regression on log-log plot
    valid_idx = C_r > 0
    log_r = np.log(r_values[valid_idx])
    log_C = np.log(C_r[valid_idx])
    
    coeffs = np.polyfit(log_r, log_C, 1)
    D_2 = coeffs[0]
    
    return D_2, r_values, C_r


def detect_clusters(theta, omega, radius=0.1):
    """
    Simple clustering: count isolated groups of points
    Two points are in same cluster if distance < radius
    """
    points = np.column_stack([theta, omega])
    n = len(points)
    
    if n == 0:
        return 0, []
    
    visited = np.zeros(n, dtype=bool)
    clusters = []
    
    for i in range(n):
        if visited[i]:
            continue
        
        cluster = [i]
        visited[i] = True
        queue = [i]
        
        while queue:
            current = queue.pop(0)
            # Find neighbors
            distances = np.linalg.norm(points - points[current], axis=1)
            neighbors = np.where((distances < radius) & ~visited)[0]
            
            for j in neighbors:
                visited[j] = True
                cluster.append(j)
                queue.append(j)
        
        clusters.append(cluster)
    
    return len(clusters), clusters


def compute_lyapunov_exponent_estimate(theta, omega, max_samples=2000):
    """
    Rough estimate based on nearby point divergence (optimized)
    """
    if len(theta) > max_samples:
        idx = np.random.choice(len(theta), max_samples, replace=False)
        theta_sample = theta[idx]
        omega_sample = omega[idx]
    else:
        theta_sample = theta
        omega_sample = omega
    
    points = np.column_stack([theta_sample, omega_sample])
    
    # Estimate divergence from time evolution
    divergences = []
    for i in range(min(100, len(points) - 10)):
        for j in range(i+1, min(i+20, len(points))):
            dist_now = np.linalg.norm(points[i] - points[j])
            if dist_now > 1e-6:
                divergences.append(np.log(dist_now))
    
    lyapunov_est = np.mean(divergences) if divergences else 0.0
    return lyapunov_est


# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

def analyze_coverage(theta, omega, grid_size=20):
    """Compute what fraction of phase space is covered"""
    density, _, _ = compute_point_density(theta, omega, grid_size)
    non_empty_cells = np.sum(density > 0)
    total_cells = grid_size * grid_size
    coverage = 100 * non_empty_cells / total_cells
    return coverage, density


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    # Redirect stdout to our DualLogger
    sys.stdout = DualLogger("poincare_analysis_results.txt")

    print("\n" + "="*70)
    print("POINCARÉ SECTION TOPOLOGICAL FIDELITY ANALYSIS")
    print("="*70)
    
    # Parameters
    A_val = 1.5
    omega_d = 0.67
    beta_val = 0.5
    
    # Script 13 parameters
    print("\n[SCRIPT 13] Low Resolution Poincaré Section")
    print("-" * 70)
    N_09 = 25000
    steps_09 = 25
    skip_09 = 100
    
    theta_09, omega_09, periods_09 = extract_poincare_points(
        N_09, steps_09, skip_09, A_val, omega_d, beta_val
    )
    
    print(f"✓ Total Points Extracted: {len(theta_09)}")
    print(f"✓ Driving Periods After Transient: {periods_09}")
    print(f"✓ Points per Period: {len(theta_09) / periods_09:.2f}")
    
    # Script 09 parameters
    print("\n[SCRIPT 09] Regular Resolution Poincaré Section")
    print("-" * 70)
    N_12 = 50000
    steps_12 = 100
    skip_12 = 100
    
    theta_12, omega_12, periods_12 = extract_poincare_points(
        N_12, steps_12, skip_12, A_val, omega_d, beta_val
    )
    
    print(f"✓ Total Points Extracted: {len(theta_12)}")
    print(f"✓ Driving Periods After Transient: {periods_12}")
    print(f"✓ Points per Period: {len(theta_12) / periods_12:.2f}")
    
    # Script 14 parameters
    print("\n[SCRIPT 14] Super-High Resolution Poincaré Section")
    print("-" * 70)
    N_14 = 100000
    steps_14 = 250
    skip_14 = 100
    
    theta_14, omega_14, periods_14 = extract_poincare_points(
        N_14, steps_14, skip_14, A_val, omega_d, beta_val
    )
    
    print(f"✓ Total Points Extracted: {len(theta_14)}")
    print(f"✓ Driving Periods After Transient: {periods_14}")
    print(f"✓ Points per Period: {len(theta_14) / periods_14:.2f}")
    
    # =====================================================================
    # DETAILED ANALYSIS
    # =====================================================================
    
    print("\n" + "="*70)
    print("SPATIAL ANALYSIS")
    print("="*70)
    
    # Focus region analysis (where chaos concentrates)
    focus_region = {"theta": (1.3, 1.8), "omega": (-0.4, 0.3)}
    
    count_09_focus = count_points_in_region(
        theta_09, omega_09, focus_region["theta"], focus_region["omega"]
    )
    count_12_focus = count_points_in_region(
        theta_12, omega_12, focus_region["theta"], focus_region["omega"]
    )
    count_14_focus = count_points_in_region(
        theta_14, omega_14, focus_region["theta"], focus_region["omega"]
    )
    
    print(f"\nFocus Region θ∈{focus_region['theta']}, ω∈{focus_region['omega']}:")
    print(f"  Script 13: {count_09_focus} points ({100*count_09_focus/len(theta_09):.1f}%)")
    print(f"  Script 09: {count_12_focus} points ({100*count_12_focus/len(theta_12):.1f}%)")
    print(f"  Script 14: {count_14_focus} points ({100*count_14_focus/len(theta_14):.1f}%)")
    print(f"  Density ratio (14/13): {count_14_focus/max(count_09_focus, 1):.2f}×")
    
    # Phase space coverage
    coverage_09, _ = analyze_coverage(theta_09, omega_09)
    coverage_12, _ = analyze_coverage(theta_12, omega_12)
    coverage_14, _ = analyze_coverage(theta_14, omega_14)
    
    print(f"\nPhase Space Coverage (20×20 grid):")
    print(f"  Script 13: {coverage_09:.1f}%")
    print(f"  Script 09: {coverage_12:.1f}%")
    print(f"  Script 14: {coverage_14:.1f}%")
    
    # =====================================================================
    # TOPOLOGICAL PROPERTIES
    # =====================================================================
    
    print("\n" + "="*70)
    print("TOPOLOGICAL PROPERTIES")
    print("="*70)
    
    # Correlation dimension
    print("\nEstimating Correlation Dimension (D_2)...")
    D2_09, r_vals_09, C_vals_09 = estimate_correlation_dimension(theta_09, omega_09)
    D2_12, r_vals_12, C_vals_12 = estimate_correlation_dimension(theta_12, omega_12)
    D2_14, r_vals_14, C_vals_14 = estimate_correlation_dimension(theta_14, omega_14)
    
    print(f"  Script 13: D_2 ≈ {D2_09:.3f}")
    print(f"  Script 09: D_2 ≈ {D2_12:.3f}")
    print(f"  Script 14: D_2 ≈ {D2_14:.3f}")
    print(f"  Convergence (13→14): {abs(D2_14 - D2_09):.3f} (lower = better convergence)")
    
    # Lyapunov exponent estimate
    print("\nEstimating Lyapunov Exponent (chaos indicator)...")
    lyap_09 = compute_lyapunov_exponent_estimate(theta_09, omega_09)
    lyap_12 = compute_lyapunov_exponent_estimate(theta_12, omega_12)
    lyap_14 = compute_lyapunov_exponent_estimate(theta_14, omega_14)
    
    print(f"  Script 13: λ ≈ {lyap_09:.4f} {'(chaotic)' if lyap_09 > 0 else '(regular)'}")
    print(f"  Script 09: λ ≈ {lyap_12:.4f} {'(chaotic)' if lyap_12 > 0 else '(regular)'}")
    print(f"  Script 14: λ ≈ {lyap_14:.4f} {'(chaotic)' if lyap_14 > 0 else '(regular)'}")
    
    # Clustering analysis
    print("\nClustering Analysis (r=0.15)...")
    n_clusters_09, clusters_09 = detect_clusters(theta_09, omega_09, radius=0.15)
    n_clusters_12, clusters_12 = detect_clusters(theta_12, omega_12, radius=0.15)
    n_clusters_14, clusters_14 = detect_clusters(theta_14, omega_14, radius=0.15)
    
    cluster_sizes_09 = sorted([len(c) for c in clusters_09], reverse=True)
    cluster_sizes_12 = sorted([len(c) for c in clusters_12], reverse=True)
    cluster_sizes_14 = sorted([len(c) for c in clusters_14], reverse=True)
    
    print(f"  Script 13: {n_clusters_09} clusters found")
    print(f"    Top 3 sizes: {cluster_sizes_09[:3]}")
    print(f"    Average cluster size: {np.mean(cluster_sizes_09):.1f}")
    
    print(f"  Script 09: {n_clusters_12} clusters found")
    print(f"    Top 3 sizes: {cluster_sizes_12[:3]}")
    print(f"    Average cluster size: {np.mean(cluster_sizes_12):.1f}")
    
    print(f"  Script 14: {n_clusters_14} clusters found")
    print(f"    Top 3 sizes: {cluster_sizes_14[:3]}")
    print(f"    Average cluster size: {np.mean(cluster_sizes_14):.1f}")
    
    # =====================================================================
    # FIDELITY SUMMARY
    # =====================================================================
    
    print("\n" + "="*70)
    print("TOPOLOGICAL FIDELITY ASSESSMENT (09 vs 12 vs 14)")
    print("="*70)
    
    # Create comparison table
    print("\n┌─────────────┬──────────┬──────────┬──────────┐")
    print("│ Property    │ Low │ Regular │ High│")
    print("├─────────────┼──────────┼──────────┼──────────┤")
    print(f"│ Total Pts   │ {len(theta_09):>8} │ {len(theta_12):>8} │ {len(theta_14):>8} │")
    print(f"│ D_2         │ {D2_09:>8.3f} │ {D2_12:>8.3f} │ {D2_14:>8.3f} │")
    print(f"│ λ estimate  │ {lyap_12:>8.4f} │ {lyap_09:>8.4f} │ {lyap_14:>8.4f} │")
    print(f"│ Coverage %  │ {coverage_09:>8.1f} │ {coverage_12:>8.1f} │ {coverage_14:>8.1f} │")
    print("└─────────────┴──────────┴──────────┴──────────┘")
    
    # Fidelity scoring
    fidelity_score_14 = 0
    
    # Point density improvement
    density_ratio_12_09 = len(theta_12) / len(theta_09)
    density_ratio_14_09 = len(theta_14) / len(theta_09)
    print(f"\n1. Point Density Improvement:")
    print(f"   13/09 ratio: {density_ratio_12_09:.2f}×")
    print(f"   14/09 ratio: {density_ratio_14_09:.2f}×")
    if density_ratio_14_09 >= 1.8:
        fidelity_score_14 += 30
        print("   ✓✓ Excellent density in Script 14")
    elif density_ratio_14_09 >= 1.5:
        fidelity_score_14 += 25
        print("   ✓ Good density improvement")
    
    # Correlation dimension convergence
    dim_agreement_14_09 = 1 - min(abs(D2_14 - D2_09) / max(abs(D2_09), 1), 1)
    dim_agreement_14_12 = 1 - min(abs(D2_14 - D2_12) / max(abs(D2_12), 1), 1)
    print(f"\n2. Dimension Convergence:")
    print(f"   (14-13) agreement: {dim_agreement_14_09:.2%}")
    print(f"   (14-09) agreement: {dim_agreement_14_12:.2%}")
    if dim_agreement_14_09 > 0.95 and dim_agreement_14_12 > 0.95:
        fidelity_score_14 += 30
        print("   ✓✓ Excellent convergence - dimension stable across all resolutions")
    elif dim_agreement_14_09 > 0.85:
        fidelity_score_14 += 20
        print("   ✓ Good convergence")
    
    # Coverage
    coverage_improvement = coverage_14 - coverage_09
    print(f"\n3. Phase Space Coverage:")
    print(f"   09: {coverage_09:.1f}% → 14: {coverage_14:.1f}% (+{coverage_improvement:.1f}%)")
    if coverage_improvement >= 10:
        fidelity_score_14 += 25
        print("   ✓✓ Script 14 explores significantly more regions")
    elif coverage_improvement >= 5:
        fidelity_score_14 += 15
        print("   ✓ Improved phase space exploration")
    else:
        fidelity_score_14 += 5
        print("   ⚠ Coverage similar to Script 13")
    
    # Chaos detection consistency
    chaos_consistent = (lyap_09 > 0) == (lyap_12 > 0) == (lyap_14 > 0)
    print(f"\n4. Chaos Characterization Consistency:")
    all_chaotic = (lyap_09 > 0) and (lyap_12 > 0) and (lyap_14 > 0)
    print(f"   All positive Lyapunov: {all_chaotic}")
    if chaos_consistent and all_chaotic:
        fidelity_score_14 += 15
        print("   ✓✓ All resolutions confirm chaotic behavior")
    elif chaos_consistent:
        fidelity_score_14 += 10
        print("   ✓ Results consistent")
    
    print(f"\n{'='*70}")
    print(f"TOPOLOGICAL FIDELITY SCORE: {fidelity_score_14}/100")
    print(f"{'='*70}")
    
    if fidelity_score_14 >= 90:
        print("✓✓✓ EXCELLENT: Script 14 captures attractor topology with high fidelity")
        print("    Recommended for publication-quality analysis")
    elif fidelity_score_14 >= 80:
        print("✓✓ VERY GOOD: Strong convergence across all resolutions")
        print("   Script 14 suitable for rigorous dynamical systems analysis")
    elif fidelity_score_14 >= 70:
        print("✓ GOOD: Convergence observed; all scripts consistent")
        print("  Compare carefully; Script 14 provides added confidence")
    else:
        print("⚠ FAIR: Consider further optimization")
    
    print("\n" + "="*70 + "\n")