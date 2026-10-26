#!/usr/bin/env python3
"""Create teaser figure using actual overview renders."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from PIL import Image

def create_teaser():
    """Create teaser with overview renders."""

    renders_dir = Path('figures/renders')
    output_path = Path('figures/teaser_renders.pdf')

    # Load overview renders
    orig_img = np.array(Image.open(renders_dir / 'original-total.png'))
    clean_img = np.array(Image.open(renders_dir / 'clean-total.png'))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Original
    axes[0].imshow(orig_img)
    axes[0].axis('off')
    axes[0].set_title('Original 3DGS\n525K Gaussians, 125 MB', fontsize=11, fontweight='bold')

    # Clean
    axes[1].imshow(clean_img)
    axes[1].axis('off')
    axes[1].set_title('Clean-GS (Ours)\n198K Gaussians, 47 MB (62% reduction)', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=150)

    plt.close()

    print(f"Created teaser: {output_path}")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")

if __name__ == '__main__':
    create_teaser()
