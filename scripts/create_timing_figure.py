#!/usr/bin/env python3
"""
Create time vs compression figure with actual timing data
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['text.usetex'] = False

fig, ax = plt.subplots(figsize=(6, 3))

# Actual data from timed experiments
methods = ['Basic', 'Clean-GS\n(neighbor)']
times = [20.7, 36.8]  # seconds
compression = [60.4, 62.4]  # percent

x = np.arange(len(methods))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, times, width, label='Processing Time (s)',
               color='#3498db', alpha=0.85, edgecolor='black', linewidth=1)
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, compression, width, label='Compression (%)',
                color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1)

# Labels
ax.set_ylabel('Processing Time (seconds)', fontweight='bold', fontsize=9)
ax2.set_ylabel('Compression (%)', fontweight='bold', fontsize=9)
ax.set_xlabel('Method', fontweight='bold', fontsize=9)
ax.set_title('Time vs. Compression Trade-off (Temple Dataset)', fontweight='bold', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=8)
ax.set_ylim([0, 50])
ax2.set_ylim([0, 70])

# Add value labels
for bar, t in zip(bars1, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{t:.1f}s', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

for bar, c in zip(bars2, compression):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'{c:.1f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# Add delta annotation
ax.annotate('', xy=(1, 36.8), xytext=(0, 20.7),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax.text(0.5, 29, '+16.1s\n(+78%)', ha='center', va='center',
        fontsize=7, color='red', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', linewidth=1))

ax2.annotate('', xy=(1.15, 62.4), xytext=(1.15, 60.4),
             arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
ax2.text(1.35, 61.4, '+2%', ha='left', va='center',
         fontsize=7, color='green', fontweight='bold')

# Grid
ax.grid(axis='y', alpha=0.25, linestyle='--')

# Legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('paper/figures/time_vs_compression.pdf', bbox_inches='tight', dpi=300)
plt.close()
print("✓ Created time_vs_compression.pdf")
