"""
generate_pseudo_targets.py
===========================
Bootstraps a `target/` folder of pseudo-ground-truth enhanced images from a
folder of raw underwater images, since Brackish / Trash-ICRA19 don't ship
with reference enhanced images.

Uses a stronger classical fusion method than plain CLAHE+white balance+gamma:
    1. Gray-World white balance (color correction)
    2. Two exposure-compensated versions of the balanced image (fused via
       simple multi-exposure Laplacian-pyramid-style blending, approximated
       here with a weighted average using contrast/saturation/exposedness
       weight maps - this mirrors classic underwater fusion enhancement,
       e.g. Ancuti et al. style techniques referenced in the paper)
    3. CLAHE for local contrast
    4. Mild gamma correction

This is only meant to bootstrap training targets; if you already have
reference images (e.g. from another enhancement method or manual curation),
just populate `target/` directly and skip this script.

Usage:
    # A flat image folder
    python generate_pseudo_targets.py --input path/to/raw_images --output path/to/data_root

    # Trash-ICRA19 (auto-detects dataset/train and dataset/val)
    python generate_pseudo_targets.py --input ../trash_ICRA19 --output ../trash_icra_paired
"""

import os
import shutil
import argparse
import cv2
import numpy as np


def gray_world_white_balance(img):
    img = img.astype(np.float32)
    b, g, r = cv2.split(img)
    b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
    gray_mean = (b_mean + g_mean + r_mean) / 3.0
    b = np.clip(b * (gray_mean / (b_mean + 1e-6)), 0, 255)
    g = np.clip(g * (gray_mean / (g_mean + 1e-6)), 0, 255)
    r = np.clip(r * (gray_mean / (r_mean + 1e-6)), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def clahe_enhance(img, clip_limit=2.5, tile_grid=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def exposedness_weight(gray, sigma=0.25):
    gray_f = gray.astype(np.float32) / 255.0
    return np.exp(-((gray_f - 0.5) ** 2) / (2 * sigma ** 2))


def contrast_weight(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return np.abs(lap)


def saturation_weight(img):
    img_f = img.astype(np.float32) / 255.0
    mean_c = img_f.mean(axis=2)
    return np.sqrt(np.mean((img_f - mean_c[..., None]) ** 2, axis=2))


def fuse_two_exposures(under_exp, over_exp):
    """Simple per-pixel weighted fusion (fast approximation of Laplacian-pyramid fusion)."""
    def weights(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        w = (contrast_weight(gray) + 1e-3) * \
            (saturation_weight(img) + 1e-3) * \
            (exposedness_weight(gray) + 1e-3)
        return w

    w1 = weights(under_exp)
    w2 = weights(over_exp)
    total = w1 + w2 + 1e-6
    w1n, w2n = (w1 / total)[..., None], (w2 / total)[..., None]

    fused = under_exp.astype(np.float32) * w1n + over_exp.astype(np.float32) * w2n
    return np.clip(fused, 0, 255).astype(np.uint8)


def make_pseudo_target(img_bgr):
    balanced = gray_world_white_balance(img_bgr)

    under_exp = gamma_correction(balanced, gamma=0.7)  # darker / more contrast
    over_exp = clahe_enhance(balanced, clip_limit=3.0)  # brighter, local contrast boosted

    fused = fuse_two_exposures(under_exp, over_exp)
    final = gamma_correction(fused, gamma=1.1)  # slight brightness restraint
    return final


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TRASH_ICRA_SPLITS = ("train", "val", "test")
DEFAULT_TRASH_ICRA_SPLITS = ("train", "val")


def process(input_dir, output_root, overwrite=False):
    input_out = os.path.join(output_root, "input")
    target_out = os.path.join(output_root, "target")
    os.makedirs(input_out, exist_ok=True)
    os.makedirs(target_out, exist_ok=True)

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(VALID_EXTENSIONS))
    if not files:
        raise RuntimeError(f"No images found in {input_dir}")

    processed = 0
    skipped = 0
    for fname in files:
        src_path = os.path.join(input_dir, fname)
        input_path = os.path.join(input_out, fname)
        target_path = os.path.join(target_out, fname)
        if not overwrite and os.path.isfile(input_path) and os.path.isfile(target_path):
            skipped += 1
            continue

        img = cv2.imread(src_path)
        if img is None:
            print(f"Skipping unreadable file: {src_path}")
            continue

        pseudo_target = make_pseudo_target(img)

        shutil.copy2(src_path, input_path)
        if not cv2.imwrite(target_path, pseudo_target):
            raise OSError(f"Failed to write pseudo-target: {target_path}")
        processed += 1
        print(f"Processed: {fname}")

    print(f"Paired data ready at {output_root}: {processed} processed, {skipped} already present")
    return len(files)


def find_trash_icra_dataset(input_path):
    """Return the directory containing Trash-ICRA's train/val/test folders."""
    candidates = (input_path, os.path.join(input_path, "dataset"))
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "train")):
            return candidate
    return None


def process_trash_icra(dataset_root, output_root, splits, overwrite=False):
    total = 0
    for split in splits:
        split_input = os.path.join(dataset_root, split)
        if not os.path.isdir(split_input):
            raise FileNotFoundError(f"Trash-ICRA split not found: {split_input}")
        print(f"\nPreparing Trash-ICRA19 {split} split")
        total += process(
            split_input,
            os.path.join(output_root, split),
            overwrite=overwrite,
        )
    print(f"\nDone. {total} Trash-ICRA19 images are available under {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo ground-truth targets for UICE-MIRNet training")
    parser.add_argument(
        "--input",
        required=True,
        help="Raw image folder, Trash-ICRA19 root, or Trash-ICRA19 dataset folder",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Paired-data output root; Trash-ICRA split folders are preserved",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=TRASH_ICRA_SPLITS,
        default=list(DEFAULT_TRASH_ICRA_SPLITS),
        help="Trash-ICRA19 splits to prepare (default: train val)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate pairs that already exist")
    args = parser.parse_args()

    dataset_root = find_trash_icra_dataset(args.input)
    if dataset_root:
        process_trash_icra(dataset_root, args.output, args.splits, args.overwrite)
    else:
        process(args.input, args.output, args.overwrite)


if __name__ == "__main__":
    main()
