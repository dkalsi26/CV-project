"""Evaluate the cumulative UICE-MIRNet enhancement chain on test images.

Every held-out image follows the same inference-only sequence:

    original -> UICE-MIRNet -> gray-world colour correction
             -> gamma correction -> CLAHE

Intermediate images are saved so the effect of every cumulative stage can
be inspected. Trash-ICRA19 has no clean enhancement references, therefore the
report uses no-reference metrics (UIQM, UCIQE, entropy and detail/contrast
statistics) instead of PSNR/SSIM.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from dataset import IMG_EXTENSIONS
from inference import load_model
from quality_metrics import uciqe, stage_metrics, export_metrics, print_metric_table, METRIC_DEFINITIONS
from postprocessing import (
    clahe_luminance,
    gamma_correction,
    gray_world_white_balance,
)


def select_test_images(test_dir, image_names=None, count=2):
    """Return explicit images, or the first ``count`` test images."""
    test_path = Path(test_dir)
    if not test_path.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    candidates = sorted(
        path
        for path in test_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    )
    if not candidates:
        raise RuntimeError(f"No test images found in {test_path}")

    if image_names:
        selected = []
        for image_name in image_names:
            image_path = test_path / image_name
            if not image_path.is_file():
                raise FileNotFoundError(f"Test image not found: {image_path}")
            if image_path.suffix.lower() not in IMG_EXTENSIONS:
                raise ValueError(f"Unsupported image type: {image_path}")
            selected.append(image_path)
        return selected

    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    if len(candidates) < count:
        raise RuntimeError(
            f"Requested {count} test images, but only {len(candidates)} were found"
        )
    return candidates[:count]


def image_to_tensor(image, device):
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).div_(255.0)
    return tensor.unsqueeze(0).to(
        device,
        non_blocking=device.type == "cuda",
        memory_format=torch.channels_last,
    )


def enhance_image(model, image, device, use_amp=True, amp_dtype=torch.float16):
    """Run MIRNet at original resolution, padding only for its three scales."""
    tensor = image_to_tensor(image, device)
    _, _, height, width = tensor.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=use_amp and device.type == "cuda",
    ):
        output = model(tensor).clamp(0.0, 1.0)

    output = output[:, :, :height, :width]
    array = output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")


def apply_cumulative_classical_stages(mirnet_image, gamma, clip_limit, tile_grid_size):
    """Apply all classical algorithms together, in order, after MIRNet."""
    mirnet_rgb = np.asarray(mirnet_image.convert("RGB"), dtype=np.uint8)
    mirnet_bgr = cv2.cvtColor(mirnet_rgb, cv2.COLOR_RGB2BGR)

    colour_bgr = gray_world_white_balance(mirnet_bgr)
    gamma_bgr = gamma_correction(colour_bgr, gamma=gamma)
    clahe_bgr = clahe_luminance(
        gamma_bgr,
        clip_limit=clip_limit,
        tile_grid_size=tuple(tile_grid_size),
    )

    def to_pil(image_bgr):
        return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), mode="RGB")

    return to_pil(colour_bgr), to_pil(gamma_bgr), to_pil(clahe_bgr)


def make_comparison(panels):
    """Render the five cumulative stages as equally sized side-by-side panels."""
    header_height = 62
    gap = 8
    width, height = panels[0][1].size
    canvas = Image.new(
        "RGB",
        (len(panels) * width + (len(panels) - 1) * gap, height + header_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image, score) in enumerate(panels):
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        x = index * (width + gap)
        canvas.paste(image, (x, header_height))
        draw.text((x + 10, 8), f"{index + 1}. {label}", fill="black")
        draw.text((x + 10, 32), f"UCIQE: {score:.4f}", fill="black")
    return canvas


def save_stage_images(output_dir, named_images):
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = (
        "00_original.png",
        "01_uice_mirnet.png",
        "02_colour_corrected.png",
        "03_gamma_corrected.png",
        "04_clahe_final.png",
    )
    saved = {}
    for filename, (stage, image) in zip(filenames, named_images):
        path = output_dir / filename
        image.save(path)
        saved[stage] = str(path)
    return saved


def compare(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    selected_images = select_test_images(
        args.test_dir,
        image_names=args.images,
        count=args.num_images,
    )

    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    model = load_model(args.checkpoint, device).to(memory_format=torch.channels_last)
    results_root = Path(args.output_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    reports = []

    for image_path in selected_images:
        original = Image.open(image_path).convert("RGB")
        mirnet_result = enhance_image(model, original, device, True, amp_dtype)
        colour_result, gamma_result, clahe_result = apply_cumulative_classical_stages(
            mirnet_result,
            gamma=args.gamma,
            clip_limit=args.clahe_clip_limit,
            tile_grid_size=args.clahe_tile_grid,
        )

        named_images = [
            ("Original", original),
            ("UICE-MIRNet", mirnet_result),
            ("+ Color correction", colour_result),
            ("+ Gamma correction", gamma_result),
            ("+ CLAHE (final)", clahe_result),
        ]
        results = stage_metrics(named_images)
        panels = [(name, image, result["uciqe"])
                  for (name, image), result in zip(named_images, results)]
        image_output_dir = results_root / image_path.stem
        saved_stages = save_stage_images(image_output_dir, named_images)
        comparison_path = image_output_dir / "side_by_side_comparison.jpg"
        make_comparison(panels).save(comparison_path, quality=95)

        report = {
            "test_image": str(image_path.resolve()),
            "comparison_image": str(comparison_path),
            "stage_images": saved_stages,
            "results": results,
        }
        reports.append(report)

    combined_report = {
        "device": str(device),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "pipeline_order": [
            "UICE-MIRNet",
            "gray_world_white_balance",
            "gamma_correction",
            "clahe_luminance",
        ],
        "post_processing": {
            "gamma": args.gamma,
            "clahe_clip_limit": args.clahe_clip_limit,
            "clahe_tile_grid": list(args.clahe_tile_grid),
        },
        "metrics_note": METRIC_DEFINITIONS["interpretation"],
        "images": reports,
    }
    metrics_path = results_root / "test_comparison_metrics.json"
    export_metrics(combined_report, metrics_path)
    print_metric_table(combined_report)

    print(f"\nDevice: {device}")
    print(f"Processed {len(reports)} held-out test images")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Metrics CSV: {metrics_path.with_suffix('.csv')}")
    return combined_report


def get_args():
    parser = argparse.ArgumentParser(
        description="Run MIRNet then cumulative colour, gamma, and CLAHE stages"
    )
    parser.add_argument("--checkpoint", required=True, help="Trained UICE-MIRNet checkpoint")
    parser.add_argument("--test_dir", required=True, help="Held-out test image directory")
    parser.add_argument("--images", nargs="*", default=None, help="Specific test filenames")
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--output_dir", default="test_results")
    parser.add_argument("--gamma", type=float, default=1.2)
    parser.add_argument("--clahe_clip_limit", type=float, default=2.0)
    parser.add_argument(
        "--clahe_tile_grid",
        type=int,
        nargs=2,
        default=(8, 8),
        metavar=("W", "H"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    compare(get_args())
