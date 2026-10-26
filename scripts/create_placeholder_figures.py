#!/usr/bin/env python3
"""
Create placeholder figures for the paper
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Figure 1: Teaser
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].text(0.5, 0.5, 'Original Scene\n(125 MB)\n525K Gaussians', 
             ha='center', va='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat'))
axes[0].set_title('Input: 3DGS + Environment')
axes[0].axis('off')

axes[1].text(0.5, 0.5, 'Semantic Masks\n(3 views)', 
             ha='center', va='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='lightblue'))
axes[1].set_title('Sparse Masks (1% of views)')
axes[1].axis('off')

axes[2].text(0.5, 0.5, 'Clean Object\n(47 MB)\n198K Gaussians\n62% reduction', 
             ha='center', va='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgreen'))
axes[2].set_title('Output: Isolated Object')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('figures/teaser.pdf', bbox_inches='tight', dpi=300)
plt.close()
print("Created teaser.pdf")

# Figure 2: Pipeline
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Stage 1
axes[0].text(0.5, 0.7, 'Stage 1: Whitelist', ha='center', fontsize=16, weight='bold')
axes[0].text(0.5, 0.5, 'Project to masked views\nKeep if in object region\n\n525K → 363K Gaussians', 
             ha='center', va='center', fontsize=12)
axes[0].arrow(0.2, 0.3, 0.6, 0, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
axes[0].axis('off')

# Stage 2
axes[1].text(0.5, 0.7, 'Stage 2: Color Validation', ha='center', fontsize=16, weight='bold')
axes[1].text(0.5, 0.5, 'Depth buffering\nValidate colors\n\n363K → 208K Gaussians', 
             ha='center', va='center', fontsize=12)
axes[1].arrow(0.2, 0.3, 0.6, 0, head_width=0.05, head_length=0.1, fc='green', ec='green')
axes[1].axis('off')

# Stage 3
axes[2].text(0.5, 0.7, 'Stage 3: Outlier Removal', ha='center', fontsize=16, weight='bold')
axes[2].text(0.5, 0.5, 'K-NN analysis\nRemove isolated\n\n208K → 198K Gaussians', 
             ha='center', va='center', fontsize=12)
axes[2].arrow(0.2, 0.3, 0.6, 0, head_width=0.05, head_length=0.1, fc='red', ec='red')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('figures/pipeline.pdf', bbox_inches='tight', dpi=300)
plt.close()
print("Created pipeline.pdf")

# Figure 3: Sparse masks analysis
masks = [1, 3, 5, 10, 20, 50, 100]
compression = [58, 60, 61, 62, 62, 63, 63]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(masks, compression, 'o-', linewidth=2, markersize=8, color='steelblue')
ax.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Our setup (3 masks)')
ax.set_xlabel('Number of Masks', fontsize=14)
ax.set_ylabel('Compression Ratio (%)', fontsize=14)
ax.set_title('Sparse Mask Robustness', fontsize=16, weight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_ylim([55, 65])
plt.tight_layout()
plt.savefig('figures/sparse_masks.pdf', bbox_inches='tight', dpi=300)
plt.close()
print("Created sparse_masks.pdf")

# Figure 4: Qualitative results placeholder
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
labels = ['Original\n(with background)', 'Masks\n(3 views)', 'Clean-GS Result\n(isolated)', 'Closeup\nComparison']
colors = ['lightcoral', 'lightyellow', 'lightgreen', 'lightblue']

for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
    ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=14, 
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    ax.axis('off')

plt.suptitle('Qualitative Comparison (Temple Dataset)', fontsize=16, weight='bold', y=0.98)
plt.tight_layout()
plt.savefig('figures/qualitative.pdf', bbox_inches='tight', dpi=300)
plt.close()
print("Created qualitative.pdf")

print("\nAll placeholder figures created successfully!")
print("To add real renders, replace these PDFs with actual visualization images.")
