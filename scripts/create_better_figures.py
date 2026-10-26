#!/usr/bin/env python3
"""
Create improved academic-quality figures
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['text.usetex'] = False

# Figure: Improved Pipeline
fig = plt.figure(figsize=(10, 2.5))

boxes = [
    {'x': 0.05, 'text': 'Input\n3DGS Model\n(N Gaussians)', 'color': '#ffcccc'},
    {'x': 0.25, 'text': 'Stage 1:\nWhitelist Filter\n(Project to masks)', 'color': '#cce5ff'},
    {'x': 0.50, 'text': 'Stage 2:\nColor Validation\n(Depth buffer)', 'color': '#ccf2cc'},
    {'x': 0.75, 'text': 'Stage 3:\nOutlier Removal\n(k-NN filter)', 'color': '#fff4cc'},
]

ax = fig.add_subplot(111)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Draw boxes
for i, box in enumerate(boxes):
    rect = FancyBboxPatch((box['x'], 0.3), 0.15, 0.4,
                          boxstyle="round,pad=0.01", 
                          facecolor=box['color'],
                          edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(box['x'] + 0.075, 0.5, box['text'],
            ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw arrows between boxes
    if i < len(boxes) - 1:
        arrow = FancyArrowPatch((box['x'] + 0.16, 0.5), 
                               (boxes[i+1]['x'] - 0.01, 0.5),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='black')
        ax.add_patch(arrow)

# Add output
rect = FancyBboxPatch((0.92, 0.35), 0.07, 0.3,
                      boxstyle="round,pad=0.005",
                      facecolor='#ccffcc',
                      edgecolor='black', linewidth=2)
ax.add_patch(rect)
ax.text(0.955, 0.5, 'Clean\nModel',
        ha='center', va='center', fontsize=8, fontweight='bold')

# Reduction annotations
ax.text(0.325, 0.15, '-31%', ha='center', fontsize=7, style='italic')
ax.text(0.575, 0.15, '-29%', ha='center', fontsize=7, style='italic')
ax.text(0.825, 0.15, '-2%', ha='center', fontsize=7, style='italic')

plt.tight_layout()
plt.savefig('paper/figures/pipeline.pdf', bbox_inches='tight', dpi=300)
plt.close()
print("Created improved pipeline.pdf")

# Figure: Results comparison (more academic style)
fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

# Temple
methods = ['Original', 'Basic', 'Clean-GS']
gaussians = [525.7, 208.0, 197.6]
colors = ['#e74c3c', '#3498db', '#2ecc71']

ax = axes[0]
bars = ax.bar(methods, gaussians, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
ax.set_ylabel('Gaussians (thousands)', fontweight='bold', fontsize=9)
ax.set_title('Temple Dataset', fontweight='bold', fontsize=10)
ax.set_ylim([0, 600])
ax.grid(axis='y', alpha=0.25, linestyle='--')

for bar, g in zip(bars, gaussians):
    reduction = 100 * (1 - g/gaussians[0]) if g != gaussians[0] else 0
    label = f'{g:.0f}K'
    if reduction > 0:
        label += f'\n$-${reduction:.1f}\%'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, label,
            ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# Isha
methods_i = ['Original', 'Basic', 'Clean-GS']
gaussians_i = [1112.6, 233.1, 221.5]

ax = axes[1]
bars = ax.bar(methods_i, gaussians_i, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
ax.set_ylabel('Gaussians (thousands)', fontweight='bold', fontsize=9)
ax.set_title('Isha Dataset', fontweight='bold', fontsize=10)
ax.set_ylim([0, 1300])
ax.grid(axis='y', alpha=0.25, linestyle='--')

for bar, g in zip(bars, gaussians_i):
    reduction = 100 * (1 - g/gaussians_i[0]) if g != gaussians_i[0] else 0
    label = f'{g:.0f}K'
    if reduction > 0:
        label += f'\n$-${reduction:.1f}\%'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40, label,
            ha='center', va='bottom', fontsize=7.5, fontweight='bold')

plt.tight_layout()
plt.savefig('paper/figures/results_comparison.pdf', bbox_inches='tight', dpi=300)
plt.close()
print("Created improved results_comparison.pdf")

# Figure: Ablation (cleaner)
fig, ax = plt.subplots(figsize=(6, 2.5))

stages = ['Original', '+ Whitelist', '+ Color', '+ Neighbor']
gaussians = [525.7, 363.1, 208.0, 197.6]
colors_ab = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

x = np.arange(len(stages))
bars = ax.bar(x, gaussians, color=colors_ab, alpha=0.85, edgecolor='black', linewidth=1, width=0.65)

ax.set_ylabel('Gaussians (thousands)', fontweight='bold', fontsize=9)
ax.set_xlabel('Pipeline Stage', fontweight='bold', fontsize=9)
ax.set_title('Stage Contribution (Temple Dataset)', fontweight='bold', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=8)
ax.set_ylim([0, 600])
ax.grid(axis='y', alpha=0.25, linestyle='--')

# Annotations
for i, (bar, g) in enumerate(zip(bars, gaussians)):
    reduction = 100 * (1 - g/gaussians[0]) if i > 0 else 0
    label = f'{g:.0f}K'
    if i > 0:
        label += f'\n$-${reduction:.1f}\%'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, label,
            ha='center', va='bottom', fontsize=7.5, fontweight='bold')

plt.tight_layout()
plt.savefig('paper/figures/ablation_stages.pdf', bbox_inches='tight', dpi=300)
plt.close()
print("Created improved ablation_stages.pdf")

print("\n✅ All improved figures created")
