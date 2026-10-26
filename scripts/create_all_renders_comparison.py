#!/usr/bin/env python3
"""Create comprehensive qualitative comparison using all available renders."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from PIL import Image

def create_comprehensive_comparison():
    """Create comparison figure using all 6 views."""

    renders_dir = Path('figures/renders')
    output_path = Path('figures/qualitative_all_renders.pdf')

    # All available views (excluding total/overview - that's used in teaser)
    views = [
        ('front', 'Front View'),
        ('front2', 'Front View 2'),
        ('side', 'Side View'),
        ('back', 'Back View'),
        ('back2', 'Back View 2'),
    ]

    # Create figure with gridspec for tight control
    fig = plt.figure(figsize=(10, 1.65*len(views)))
    gs = gridspec.GridSpec(len(views), 2, figure=fig,
                          wspace=0.01, hspace=0.02,
                          left=0.01, right=0.99, top=0.96, bottom=0.02)

    for row_idx, (view_name, view_title) in enumerate(views):
        orig_path = renders_dir / f'original-{view_name}.png'
        clean_path = renders_dir / f'clean-{view_name}.png'

        # Load images
        orig_img = np.array(Image.open(orig_path))
        clean_img = np.array(Image.open(clean_path))

        # Original
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.imshow(orig_img)
        ax0.axis('off')
        if row_idx == 0:
            ax0.set_title('Original (526K Gaussians)', fontsize=12, fontweight='bold', pad=2)
        ax0.text(0.02, 0.98, view_title, transform=ax0.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='white', alpha=0.8))

        # Clean
        ax1 = fig.add_subplot(gs[row_idx, 1])
        ax1.imshow(clean_img)
        ax1.axis('off')
        if row_idx == 0:
            ax1.set_title('Clean-GS (198K Gaussians, 62% reduction)',
                         fontsize=12, fontweight='bold', pad=2)

    # Save to PDF
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=150, bbox_inches='tight', pad_inches=0.02)

    plt.close()

    print(f"Created comprehensive comparison: {output_path}")

    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")

if __name__ == '__main__':
    create_comprehensive_comparison()
