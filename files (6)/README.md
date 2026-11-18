# UICE-MIRNet (PyTorch)

Implementation of the image-enhancement network from:

> Sarkar, P., De, S., Gurung, S. & Dey, P. *"UICE-MIRNet guided image
> enhancement for underwater object detection."* Scientific Reports 14,
> 22448 (2024). https://doi.org/10.1038/s41598-024-73243-9

This covers **Section 3.1 of the paper** — the enhancement network itself
(UICE-MIRNet: RRB → UI-CEB → FSFB + SCAB). It does **not** include YOLOv4
(Section 3.2); use the enhanced output images from this network as your
YOLOv4 training/inference set, same as the paper's pipeline.

## Files

| File | Purpose |
|---|---|
| `model.py` | The UICE-MIRNet architecture: `SCAB`, `FSFB`, `UICEB`, `RRB`, `UICEMIRNet` |
| `losses.py` | Charbonnier + SSIM + color-consistency training loss |
| `dataset.py` | Paired dataset loader (`input/`, `target/`) + inference dataset |
| `generate_pseudo_targets.py` | Bootstraps training targets from raw images (see caveat below) |
| `split_by_sequence.py` | Creates a leakage-free train/validation/test split by capture sequence |
| `train.py` | Training loop |
| `inference.py` | Runs a trained model on a folder of images |
| `postprocessing.py` | Fixed gray-world -> gamma -> CLAHE inference stage |
| `compare_test_image.py` | Runs the cumulative MIRNet -> color -> gamma -> CLAHE chain on two held-out images and saves five-stage comparisons plus metrics JSON |
| `trash_icra_experiment.py` | Validates splits, trains, then launches the comparison |
| `quality_metrics.py` | Scores every saved stage with UIQM, UCIQE, entropy, RMS contrast, average gradient and Laplacian variance; exports JSON/CSV and baseline deltas |
| `export_enhanced_images.py` | Exports the entire held-out test split with epoch-25 weights into four stage folders; resumes completed exports using its manifest |

### Export all enhanced test images

The optional section 7 in `main.ipynb` exports every test image when
`RUN_BATCH_EXPORT=True`. Its default is False. Run the export separately with:

```powershell
& '.\.venv\Scripts\python.exe' -u 'files (6)/export_enhanced_images.py'
```

Outputs are full-resolution RGB PNGs under `results/epoch25_enhanced_test/`:
`01_mirnet/`, `02_white_balance/`, `03_gamma/`, and `04_final_clahe/`.
The final folder contains MIRNet followed by all three classical operations.
Source image stems are preserved. `export_manifest.json` records checkpoint
identity, settings, source files, and completion status. This is inference only;
it does not generate training targets or update model weights. Existing completed
outputs are reused on a matching rerun. Metrics and comparison figures are
handled separately by the notebook's evaluation cells.

### Image quality evaluation

`main.ipynb` displays the three requested image panels and a metric table for
all five stages. Training remains manual. After training, the experiment writes
`results/test_comparison_metrics.json` and `.csv`. Each row includes absolute
scores and changes from both the original image and the MIRNet output.

The notebook's **Metrics only** cell can run independently, using saved PNGs
without loading the model or starting training. Select `SAVED_REPORT` to choose
a run; the default uses the final report if present, otherwise the saved epoch-5
report. This writes separate `image_quality_metrics.json` and `.csv` files beside
that report and prints the source checkpoint.

The JSON records all numerical conventions, including UIQM's 10-pixel block
variant and the existing normalized-Lab UCIQE convention. Scores from other
implementations may have different scales. Entropy is grayscale Shannon entropy
in bits; contrast, gradient and Laplacian variance use grayscale in [0, 1].
These descriptive statistics may rise with noise or oversharpening. No clean
test references are available, so PSNR/SSIM are not reported.

To attribute quality changes to the learned model versus fixed post-processing,
score all exported stages across the complete test set:

```powershell
python score_export_stages.py --export-root ..\results\epoch25_enhanced_test
```

This does not replace detector evaluation. Compare original, classical,
MIRNet-only, and MIRNet-plus-post-processing images with identical detector
splits, seeds, and training settings.

Metric references: [UIQM](https://doi.org/10.1109/JOE.2015.2469915) and
[UCIQE](https://doi.org/10.1109/TIP.2015.2491020).

## ⚠️ Important: about "ground truth" for Brackish / Trash-ICRA19

Both datasets ship with **object-detection bounding-box labels**, not
enhanced reference images. Training an enhancement network needs
`(degraded, enhanced)` **pairs**. The paper doesn't state exactly how its
training targets were produced. You have two options:

1. **Bootstrap pseudo-targets** (what `generate_pseudo_targets.py` does):
   applies a strong classical fusion enhancement (white balance + dual
   exposure fusion + CLAHE + gamma) to your raw images to create plausible
   "enhanced" targets, then trains UICE-MIRNet to reproduce/improve on that.
   This is standard practice in underwater-enhancement papers that lack
   ground truth.
2. **Supply your own reference images** — e.g. outputs from an existing
   strong method (UDCP, fusion-based, etc.), and drop them directly into
   `target/`.

Either way, the network learns from whatever you put in `target/` — the
architecture itself is faithful to the paper regardless of which target
source you use.

## Architecture summary

```
I -> Conv -> X0 -> RRB x3 -> Conv -> R
Iout = R + I

RRB(F)   = Conv( UI-CEB( UI-CEB( Conv(F) ) ) ) + F
UI-CEB   = 4-scale (1x, 1/2, 1/4, 1/8) SCAB -> cross-scale FSFB -> SCAB -> final FSFB + Conv, + residual
FSFB     = fuse (sum) -> GAP -> descriptor -> per-stream softmax attention -> weighted sum
SCAB     = dual attention (spatial + "modified" channel attention using ReLU instead of sigmoid) + residual
```

Some specifics (base channel count = 32, RRB count = 3, SSIM/Charbonnier
loss weights, patch size) aren't pinned down exactly by the paper and are
configurable via CLI args — defaults are reasonable starting points.

## Usage

Run these commands from the directory containing the Python files.

### 0. Split Trash-ICRA19 by sequence (required)

The distributed frame splits are not safe for evaluation: frames sharing the
prefix before `_frame` are highly correlated and must remain in one split.
Create a separate split before creating pairs; the source dataset is never
modified:

```powershell
python split_by_sequence.py --source-root ..\trash_ICRA19 --output-root ..\trash_icra_sequence_split --hardlink
python generate_pseudo_targets.py --input ..\trash_icra_sequence_split --output ..\trash_icra_paired
```

Use `trash_icra_sequence_split` as `--dataset_root` for the experiment. The
experiment driver rejects sequence overlap rather than reporting a leaked test
score.

### 1. Prepare Trash-ICRA19 pseudo-target pairs
```powershell
python generate_pseudo_targets.py --input ..\trash_ICRA19 --output ..\trash_icra_paired
```

The script detects `trash_ICRA19/dataset/{train,val}` and creates:

```text
trash_icra_paired/
  train/input/  train/target/
  val/input/    val/target/
```

The test images remain held out for inference. The `val` split is used only
to select the best checkpoint; it is never used for optimizer updates.

Only the images are used by the enhancement model. Trash-ICRA's `.txt` and
`.xml` object-detection annotations stay in the original dataset; image
enhancement does not alter their bounding-box geometry.

### 2. Train
```powershell
python train.py --data_root ..\trash_icra_paired --epochs 100 --batch_size 2 --patch_size 256 --lr 2e-4 --amp --save_dir ..\checkpoints\trash_icra
```

`--amp` uses BF16 by default. This avoids the gradient overflow observed with
FP16 in the deep residual-attention blocks while retaining CUDA acceleration.

The trainer uses `train` for optimization and `val` for validation. It saves
`uice_mirnet_best.pth` whenever validation improves and still supports the
old flat `input/` + `target/` layout. Use `--resume CHECKPOINT` to continue
an interrupted run.

### 3. Compare one held-out test image
```powershell
python compare_test_image.py --checkpoint ..\checkpoints\trash_icra\uice_mirnet_best.pth --test_dir ..\trash_ICRA19\dataset\test --image bio0000_frame0000016.jpg --output ..\results\trash_icra_comparison.jpg
```

Omit `--image` to use the first test image alphabetically. The comparison
contains exactly: **original**, **trained UICE-MIRNet output**, and
**post-processing compounded on the UICE-MIRNet output**. The post-processing
order is gray-world white balance -> gamma correction -> CLAHE. The script
also writes a JSON file beside the comparison image with each UCIQE value and
its change from the original.

Trash-ICRA19 has no clean enhancement reference, so this is a visual
comparison rather than a reference-based PSNR/SSIM test. Each panel also
reports UCIQE, a no-reference underwater image-quality score.

### 4. Run the complete experiment with one command

First verify the dataset and all train/validation pairs. This command does
not train or run inference:

```powershell
python trash_icra_experiment.py --dataset_root ..\trash_ICRA19 --paired_root ..\trash_icra_paired --checkpoint_dir ..\checkpoints\trash_icra --results_dir ..\results --check_only
```

After the check succeeds, remove `--check_only` to train and automatically
compare the best validation checkpoint on a held-out test image:

```powershell
python trash_icra_experiment.py --dataset_root ..\trash_ICRA19 --paired_root ..\trash_icra_paired --checkpoint_dir ..\checkpoints\trash_icra --results_dir ..\results --epochs 100 --batch_size 2 --patch_size 256 --lr 2e-4 --amp
```

Add `--prepare_targets` only when `trash_icra_paired` has not already been
created. The driver intentionally prepares only `train` and `val`; it never
creates or trains on test targets.

To enhance the complete test split instead, use `inference.py`:

```powershell
python inference.py --checkpoint ..\checkpoints\trash_icra\uice_mirnet_best.pth --input ..\trash_ICRA19\dataset\test --output ..\trash_icra_enhanced\test
```

## Sanity check
```bash
python model.py
# prints input/output tensor shapes and parameter count
```

## Requirements
```bash
pip install torch opencv-python pillow numpy
```
(A CUDA-capable GPU is strongly recommended — the paper trains in Google
Colab with a GPU runtime.)
