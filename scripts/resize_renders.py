#!/usr/bin/env python3
"""Resize render images to reduce file size."""

from PIL import Image
from pathlib import Path

renders_dir = Path('figures/renders')
resized_dir = renders_dir / 'resized'
resized_dir.mkdir(exist_ok=True)

# Resize all renders to 50% with good quality
for img_path in renders_dir.glob('*.png'):
    if img_path.parent.name == 'resized':
        continue

    print(f"Resizing {img_path.name}...")
    img = Image.open(img_path)

    # Resize to 50%
    new_size = (img.width // 2, img.height // 2)
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

    # Save with optimization
    output_path = resized_dir / img_path.name
    img_resized.save(output_path, 'PNG', optimize=True)

    orig_size = img_path.stat().st_size / (1024 * 1024)
    new_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  {orig_size:.1f} MB -> {new_size_mb:.1f} MB")

print("\nDone!")
