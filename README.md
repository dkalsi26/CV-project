# ERG-UICE-MIRNet for Underwater Enhancement and YOLOv8 Detection

This repository evaluates whether underwater image enhancement improves object detection in two datasets: **Brackish** and **Trash-ICRA19**. It compares the original UICE-MIRNet enhancement network with the proposed **ERG-UICE-MIRNet** variant, then trains/evaluates YOLOv8 on matched original and enhanced image sets.

The experiment is designed as a controlled pipeline: preserve each image's geometry and detection labels, change only the image appearance, and use the same train/validation/test split, detector configuration, and seed for every comparison.

## Pipeline

```text
Raw underwater images + bounding-box labels
                 |
                 |-- Original images ----------------------|
                 |-- UICE-MIRNet (baseline) images --------|-- YOLOv8 object detection
                 `-- ERG-UICE-MIRNet (proposed) images ----|
                                                            |
                                              matched detection metrics
```

1. Split images by capture sequence to prevent adjacent frames from leaking across splits.
2. Train the baseline UICE-MIRNet and the proposed ERG-UICE-MIRNet on the same paired training data.
3. Enhance the held-out images using each trained enhancement model, without resizing or changing bounding boxes.
4. Export three YOLO datasets: original, baseline-enhanced, and ERG-enhanced.
5. Train YOLOv8 with identical settings on each dataset and compare the same checkpoint epoch and test split.

## Models

| Model | Role | Implementation |
|---|---|---|
| UICE-MIRNet | Baseline enhancement model | [`files (6)/model.py`](<files%20(6)/model.py>) |
| ERG-UICE-MIRNet | Proposed enhancement model, adding ERG feature/reliability guidance to UICE-MIRNet | [`files (6)/erg_uice_mirnet.py`](<files%20(6)/erg_uice_mirnet.py>) |
| YOLOv8n | Object detector trained independently on each image variant | Ultralytics YOLO |

The enhancement networks operate on RGB images only. Detection labels are copied unchanged because the enhancement stage retains the original spatial resolution.

## Datasets

### Brackish

The Brackish workflow uses a leakage-safe sequence split:

```text
brackish/extracted_frames/             # raw source frames
brackish_sequence_split/{train,val,test}/
brackish_paired_sequence/{train,val}/input/
brackish_paired_sequence/{train,val}/target/
```

The paired `target/` images are generated only for enhancement-model training. Detection labels are used later when creating the YOLO-format original, baseline, and ERG datasets.

Use [`files (6)/brackish_erg_training.ipynb`](<files%20(6)/brackish_erg_training.ipynb>) to:

- create/reuse the Brackish sequence split;
- prepare paired pseudo-targets;
- train UICE-MIRNet and ERG-UICE-MIRNet in matched five-epoch segments;
- export baseline and ERG enhanced test images; and
- calculate no-reference image-quality metrics.

The resulting YOLO-format data already used by the detector experiment are expected at:

```text
results/brackish_yolov8_epoch20/datasets/
├── original/
├── uice_mirnet/
└── erg_uice_mirnet/
```

Each folder must contain matching `images/{train,val,test}` and `labels/{train,val,test}` directories.

### Trash-ICRA19

Trash-ICRA19 follows the same principle: first split by sequence, then use the same baseline/proposed enhancement training and export process. The baseline preparation and validation driver is [`files (6)/trash_icra_experiment.py`](<files%20(6)/trash_icra_experiment.py>); the model implementations are shared with Brackish.

```text
trash_ICRA19/                         # raw images and detection annotations
trash_icra_sequence_split/            # sequence-disjoint split
trash_icra_paired/                    # paired enhancement training images
checkpoints/                          # baseline and ERG checkpoints (ignored by Git)
results/                              # enhanced outputs and evaluation artefacts (ignored by Git)
```

Do not train an enhancement model on a test split. Test images are only enhanced and passed to the final detector evaluation.

## YOLOv8 three-way comparison

The Brackish YOLOv8 notebook is [`files (6)/brackish_yolov8_three_way_comparison.ipynb`](<files%20(6)/brackish_yolov8_three_way_comparison.ipynb>). It uses the pre-exported Brackish datasets directly; it does **not** retrain an enhancement model or generate images.

The notebook:

- verifies that all three variants have identical image stems and labels;
- alternates detector training in the order **original → baseline UICE-MIRNet → proposed ERG-UICE-MIRNet**;
- trains five epochs per turn until each detector reaches 50 epochs;
- saves a resumable `segment_epoch_N.pt` after every five-epoch turn; and
- exports validation comparisons and final held-out test CSVs.

Before starting training, set the notebook configuration flag:

```python
RUN_TRAINING = True
```

If a run stops, rerun the training cell. Completed five-epoch segments are detected and skipped; the next turn resumes from the appropriate `segment_epoch_N.pt` checkpoint.

## Installation

Create a Python environment with CUDA-enabled PyTorch if a compatible NVIDIA GPU is available, then install the project dependencies:

```bash
pip install torch torchvision ultralytics opencv-python pillow numpy pandas scipy matplotlib pyyaml
```

Open the notebooks from either the repository root or `files (6)`. Paths in the notebooks resolve both locations.

## Outputs and evaluation

Generated checkpoints, datasets, images, and run artefacts are intentionally excluded from version control. Typical outputs are written below:

```text
checkpoints/<experiment>/
results/<experiment>/
final_yolo_results/
├── brackish_yolov8_three_way_50ep/
│   ├── configs/
│   ├── reports/
│   └── runs/
└── statistical_tables/
```

Image-quality evaluation includes UIQM, UCIQE, entropy, RMS contrast, average gradient, and Laplacian variance. These are no-reference quality proxies, not detection metrics. Detector performance is assessed separately with precision, recall, F1 score, mAP@0.50, and mAP@0.50:0.95 on matched splits.

## Reproducibility notes

- Keep the original, baseline-enhanced, and ERG-enhanced image sets aligned by filename and resolution.
- Copy the same YOLO labels to all three image variants.
- Use the same detector architecture, seed, image size, batch policy, and epochs for every variant.
- Compare models at the same completed epoch; do not compare a partially trained model against a fully trained one.
- Report whether statistics are calculated over images, classes, or independent training repetitions. Per-class summaries should not be interpreted as independent repeated training runs.

## Project files

The original UICE-MIRNet implementation README and its lower-level commands are available at [`files (6)/README.md`](<files%20(6)/README.md>). The root README focuses on the complete baseline-versus-ERG enhancement-to-detection experiment.

