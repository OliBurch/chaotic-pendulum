"""
PERFORMANCE ANALYSIS - Computational Cost Assessment
=====================================================
Analyzes runtime, memory usage, and efficiency across all Poincaré simulations
Reads performance data from individual scripts and creates summary tables + plots
"""

import json
import csv
import matplotlib.pyplot as plt
from pathlib import Path

print("="*70)
print("PERFORMANCE ANALYSIS - Pendulum Simulations")
print("="*70)

# ============================================================================
# LOAD PERFORMANCE DATA (NOW SEARCHES ALL SUBFOLDERS)
# ============================================================================

performance_data = []

# This will automatically search the current directory AND the 'sections' folder
for json_file in Path('.').rglob('performance_*.json'):
    with open(json_file, 'r') as f:
        try:
            data = json.load(f)
            # Handle if the JSON is a list (like performance_complete.json)
            if isinstance(data, list):
                for entry in data:
                    name = entry.get('script', entry.get('name', 'Unknown'))
                    if '12' not in name and 'Ultra' not in name:
                        performance_data.append(entry)
            # Handle if the JSON is a single dictionary (like individual files)
            else:
                name = data.get('script', data.get('name', 'Unknown'))
                if '12' not in name and 'Ultra' not in name:
                    performance_data.append(data)
            print(f"✓ Loaded: {json_file}")
        except Exception as e:
            print(f"⚠ Could not load {json_file}: {e}")

if not performance_data:
    print("\n⚠ WARNING: No performance data files found!")
    exit(0)

# ============================================================================
# EXTRACT DATA FOR PLOTTING
# ============================================================================

# Use a dictionary to automatically remove duplicates
extracted_dict = {}

for entry in performance_data:
    name = entry.get('script', entry.get('name', 'Unknown'))
    
    # Map labels, order, and colors together
    if '13' in name or 'Low' in name:
        label = 'Low resolution'
        order = 0
        color = '#2ca02c' # Green
    elif '09' in name or 'Regular' in name:
        label = 'Middle ground'
        order = 1
        color = '#1f77b4' # Blue
    elif '14' in name or 'Super' in name:
        label = 'High resolution'
        order = 2
        color = '#d62728' # Red
    else:
        continue # Skip any unexpected scripts
    
    time_val = entry.get('integration_time_sec', entry.get('time_sec', 0))
    mem_val = entry.get('peak_memory_mb', entry.get('memory_mb', 0))
    points = entry.get('points_extracted', entry.get('points', 0))
    
    # Store in dict using 'label' as key. This overwrites duplicates.
    extracted_dict[label] = (order, label, time_val, mem_val, points, color)

# Convert back to list and sort
extracted_data = list(extracted_dict.values())
extracted_data.sort(key=lambda x: x[0])

scripts = [item[1] for item in extracted_data]
times = [item[2] for item in extracted_data]
memories = [item[3] for item in extracted_data]
points_list = [item[4] for item in extracted_data]
colors_list = [item[5] for item in extracted_data]

# ============================================================================
# CREATE VISUALIZATIONS (ONLY TRADEOFF PLOT)
# ============================================================================

if scripts:
    print("\n" + "="*70)
    print("DATA READY FOR PLOTTING:")
    for label, t, p in zip(scripts, times, points_list):
        print(f" -> {label}: {t:.1f} seconds, {p} points")
    print("="*70)

    print("\nGenerating visualization...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sizes = [mem * 3 for mem in memories]  # Size proportional to memory
    
    for time_val, points, color, label, size in zip(times, points_list, colors_list, scripts, sizes):
        ax.scatter(time_val, points, s=size, alpha=0.6, color=color, edgecolors='black', linewidth=2, label=label)
    
    ax.set_xlabel('Runtime (seconds)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Points Extracted from Poincaré Section', fontsize=11, fontweight='bold')
    
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('performance_tradeoff.png', dpi=700)
    plt.close()
    print("✓ Saved: performance_tradeoff.png")