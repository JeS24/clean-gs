# CleanGS: Pareto-Guided Post-Hoc Cleanup of 3D Gaussian Splatting Scenes

Jyotirmaya Shivottam, Om Kumar, Subhankar Mishra

ICVGIP 2026 · [Paper (arXiv)](https://arxiv.org/abs/2601.00913) · [Supplementary Material](docs/CleanGS_ICVGIP2026_supplementary.pdf)

<!-- TODO: add the proceedings DOI once published. -->

CleanGS removes clutter from a pretrained 3D Gaussian Splatting (3DGS) scene
without retraining. It takes calibrated images and binary masks of the target,
then selects three views by comparing four objectives on a Pareto front.
A Gaussian is kept if it is visible and its projected center falls inside the
target mask in at least two selected views. The retained Gaussians are exported
with their geometry, opacity and appearance unchanged.


## Citation

If you use CleanGS, please cite our paper:

```bibtex
@inproceedings{shivottam2026cleangs,
  title     = {{CleanGS}: Pareto-Guided Post-Hoc Cleanup of {3D} {Gaussian} Splatting Scenes},
  author    = {Shivottam, Jyotirmaya and Kumar, Om and Mishra, Subhankar},
  booktitle = {Proceedings of the Seventeenth Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP)},
  year      = {2026},
  publisher = {ACM}
}
```

---

## Contents

```text
.
├── pyproject.toml
├── uv.lock
├── src/cleangs/
│   ├── select_views.py         stage 1: Pareto-guided view selection    (cleangs-select)
│   ├── multiview_filter.py     stage 2: multiview filtering and export  (cleangs-filter)
│   ├── evaluate.py             stage 3: exported asset evaluation      (cleangs-evaluate)
│   ├── camera_utils.py         COLMAP camera loading and projection
│   ├── gaussians.py            gsplat checkpoint and PLY loading
│   └── _vendor/gsplat_colmap/  COLMAP parser from the gsplat examples (Apache-2.0)
├── CITATION.cff
└── LICENSE
```

## Setup

CleanGS requires Linux and an NVIDIA GPU with a driver that supports CUDA 12.8.
`gsplat` compiles its CUDA kernels on first use. So, the CUDA compiler (`nvcc`)
must be on `PATH` the first time you run stage 2 or 3.

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/smlab-niser/clean-gs.git
cd clean-gs
uv sync
source .venv/bin/activate
```

`uv sync` installs the dependency versions used for the paper, including
PyTorch 2.8.0 built for CUDA 12.8. You can also run the commands below with
`uv run` instead of activating the environment.

With pip, install PyTorch first and then the package:

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Check the installation:

```bash
python -c "import torch, gsplat; print(torch.cuda.is_available(), gsplat.__version__)"
cleangs-select --help
```

## Inputs

Each scene is a COLMAP-style directory:

```text
<scene>/
├── images/                 capture images
├── sparse/0/               COLMAP model (cameras.bin, images.bin, points3D.bin)
├── masks_train_no_val/     target masks for training views (selection candidates)
└── masks_latest/           target masks for all views (evaluation only)
```

You also need a pretrained Gaussian scene: a `gsplat` checkpoint
(for example `ckpt_29999_rank0.pt`) or a `.ply` file.

Masks are binary PNGs with the target in white and the same filename stem as
the corresponding image. The commands below use `masks_train_no_val/` for view
selection and filtering, and `masks_latest/` to evaluate the exported asset.
With the default `--test_every 8`, every eighth view in filename order is held
out for evaluation, starting with the first.

## Running CleanGS

The commands below reproduce the configuration reported in the paper. Set
`SCENE` to the scene directory, `CKPT` to the pretrained checkpoint, and `OUT`
to an output directory.

```bash
SCENE=/path/to/scene
NAME=$(basename $SCENE)
CKPT=/path/to/ckpt_29999_rank0.pt
OUT=./out/$NAME
```

### 1. View selection

```bash
cleangs-select \
  --dataset $SCENE --masks_dir $SCENE/masks_train_no_val \
  --k 3 --candidate_pool 0 \
  --coverage_aggregation min \
  --w_coverage 0.45 --w_baseline 0.30 --w_overlap 0.15 --w_angle 0.10 \
  --min_coverage 0.02 --max_coverage 0.60 \
  --target_coverage 0.30 --coverage_sigma 0.15 \
  --max_border_touch_sides 1 --min_bbox_margin 0.03 --max_bbox_frac 0.90 \
  --output_masks_dir masks_pareto_k3
```

The selector scores each mask and filters candidates by mask coverage and
containment. It also removes dominated candidates when at least three would
remain. It then scores every three-view subset on mask reliability `A(S)`,
camera baseline `B(S)`, dissimilarity of SfM points inside the masks `O(S)` and
diversity of viewing directions `T(S)`. From the resulting Pareto front, it picks
the subset with the smallest weighted Chebyshev distance to the ideal point.
View selection runs on the CPU.

**If too few views pass.** If the log reports fewer than three eligible views,
the selector stops without creating `$SCENE/masks_pareto_k3/`. Run it again with
the containment tests disabled and the candidate pool capped at 60 views:

```bash
cleangs-select \
  --dataset $SCENE --masks_dir $SCENE/masks_train_no_val \
  --k 3 --candidate_pool 60 --disable_heuristics \
  --coverage_aggregation min \
  --w_coverage 0.45 --w_baseline 0.30 --w_overlap 0.15 --w_angle 0.10 \
  --min_coverage 0.02 --max_coverage 0.60 \
  --target_coverage 0.30 --coverage_sigma 0.15 \
  --output_masks_dir masks_pareto_k3
```

Outputs, written into the scene directory:

```text
<scene>/
├── masks_pareto_k3/             the three selected masks (used by stage 2)
├── selected_views.txt           the selected three-view subset
├── selected_views_pareto.txt    all non-dominated subsets, with the selection marked
├── pareto_front.json            objective values of the front
└── pareto_view_selection.log    per-view and per-subset details
```

### 2. Multiview filtering and export

```bash
cleangs-filter \
  --ply $CKPT --data_dir $SCENE \
  --masked_images $SCENE/masks_pareto_k3 \
  --output_dir $OUT --scene_name $NAME \
  --steps 1 --step1_mode multiview --min_views 2 \
  --no_metrics
```

The filter renders the scene from each selected camera. A view supports a
Gaussian when the renderer marks it visible and its projected center lies
inside that view's mask. With `--min_views 2`, the filter keeps Gaussians supported
by at least two of the three views.

Output: `$OUT/1_<NAME>_cgspp.ply`, containing the retained Gaussians.
`step1_<NAME>_cgspp.ply` contains the same asset.

`--min_views 1` or `3` changes the vote threshold.

> [!NOTE]
> The released code skips an unused colour-conversion loop included in the paper's timings, so filtering is faster and the reported runtimes in the paper are conservative -- CleanGS is faster than reported : )

### 3. Evaluation

```bash
cleangs-evaluate \
  --ply $OUT/1_${NAME}_cgspp.ply \
  --data_dir $SCENE --masks $SCENE/masks_latest \
  --output_dir $OUT/eval \
  --zoom_factor 6.0 --zoom_mode faceon --test_every 8
```

The evaluator renders the exported asset at the original framing and with a
wider field of view at the same camera pose. In `faceon` mode, it divides the
focal length by `--zoom_factor`. It reports masked PSNR, SSIM, LPIPS and
silhouette leakage at both framings.

Output: `$OUT/eval/<NAME>_metrics_more.json`. Its `averages` field contains the
mean scores across evaluated views:

* `standard` and `zoomed`: PSNR, SSIM and LPIPS at the two framings;
* `leakage.l_leak` and `zoomed.l_leak_z`: mean rendered opacity outside the
  target at the two framings, as reported in the paper. `l_leak_tau` measures
  the fraction of pixels outside the target whose opacity exceeds the threshold.

Rendered views go to `$OUT/eval/renders/`; pass `--no_save_renders` to skip them.

**Other asset formats.** CleanGS exports assets in gsplat format, which need no
extra flags. For assets in the original 3DGS (INRIA) format, including the
FlashSplat and GaussianCut exports used in the paper and raw `point_cloud.ply`
files, add `--no_normalize_cameras --sh_layout 3dgs`.

Each stage can also be run as a module, for example
`python -m cleangs.select_views --help`.

## License

CleanGS is released under the [MIT License](LICENSE). The COLMAP parser in
`src/cleangs/_vendor/gsplat_colmap/` is copied from
[gsplat](https://github.com/nerfstudio-project/gsplat) and is under the
Apache License 2.0 (see the `LICENSE` file in that directory). We thank the 
authors of `gsplat` for making their code open-source.
