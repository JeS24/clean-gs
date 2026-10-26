#!/usr/bin/env python3
"""Create figure showing all 3 temple masks."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from PIL import Image

def create_all_masks_figure():
    """Show all 3 temple masks with their corresponding images."""

    images_dir = Path('../data/datasets/temple/images')
    masks_dir = Path('../data/masks/temple')
    output_path = Path('figures/all_masks.pdf')

    # All 3 masks
    mask_files = [
        ('00002', '00002_bgrem.png'),
        ('00156', '00156_bgrem.png'),
        ('00262', '00262_bgrem.png'),
    ]

    # Create figure with 3 rows and 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(10, 11))

    for row_idx, (image_id, mask_file) in enumerate(mask_files):
        # Load image and mask
        img = np.array(Image.open(images_dir / f'{image_id}.jpg'))
        mask = np.array(Image.open(masks_dir / mask_file))

        # Original image
        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title('Training Image', fontsize=12, fontweight='bold')

        # Mask overlay
        # Create red overlay where mask is white
        overlay = img.copy()
        if len(mask.shape) == 2:
            mask_binary = mask > 128
        else:
            mask_binary = mask[:,:,0] > 128
        overlay[mask_binary] = overlay[mask_binary] * 0.5 + np.array([128, 0, 0]) * 0.5

        axes[row_idx, 1].imshow(overlay.astype(np.uint8))
        axes[row_idx, 1].axis('off')
        if row_idx == 0:
            axes[row_idx, 1].set_title('Semantic Mask Overlay', fontsize=12, fontweight='bold')

        # Add view label on the right
        axes[row_idx, 1].text(1.02, 0.5, f'View {image_id}', transform=axes[row_idx, 1].transAxes,
                             fontsize=10, verticalalignment='center', rotation=-90)

    plt.tight_layout()

    # Save
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=150)

    plt.close()

    print(f"Created all masks figure: {output_path}")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")

if __name__ == '__main__':
    create_all_masks_figure()
