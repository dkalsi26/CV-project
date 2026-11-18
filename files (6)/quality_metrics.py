"""No-reference metrics on full-resolution, 8-bit RGB stage images.

UIQM: Panetta, Gao & Agaian, doi:10.1109/JOE.2015.2469915.
UCIQE: Yang & Sowmya, doi:10.1109/TIP.2015.2491020.
Numerical conventions are recorded in METRIC_DEFINITIONS. UIQM/UCIQE
implementations differ in scaling and block handling; compare like with like.
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


METRIC_NAMES = (
    "uiqm", "uciqe", "entropy", "rms_contrast", "average_gradient",
    "laplacian_variance", "uicm", "uism", "uiconm",
)
METRIC_DEFINITIONS = {
    "version": "uice-no-reference-v1",
    "input": "Full-resolution uint8 RGB; no resizing; grayscale is OpenCV RGB2GRAY.",
    "uiqm": "0.0282*UICM + 0.2953*UISM + 3.5753*UIConM; block implementation described below.",
    "uicm": "RG=R-G, YB=(R+G)/2-B; 10% floor-trimmed means; full-population variance around trimmed means; -0.0268*mean_norm + 0.1586*std_norm.",
    "uism": "RGB Sobel magnitudes (3x3, reflected border), each normalized to peak 255 and multiplied by its channel; EME=2*mean(ln(max/min)); zero-minimum blocks contribute zero; RGB weights 0.299/0.587/0.114.",
    "uiconm": "RGB block extrema; -mean(r*ln(r)), r=(max-min)/(max+min); zero denominators/ratios contribute zero (alpha=1 block contrast variant).",
    "uiqm_blocks": "Non-overlapping 10x10 blocks; discard incomplete bottom/right blocks. For images smaller than 10 pixels, use available size on that axis.",
    "uciqe": "Existing pipeline convention retained: OpenCV uint8 Lab; L/255, (a-128)/255, (b-128)/255; 0.4680*std(chroma)+0.2745*(L_p99-L_p1)+0.2576*mean(chroma/sqrt(chroma^2+L^2+1e-12)). Normalized variant, not interchangeable with raw-Lab scores.",
    "entropy": "Shannon entropy in bits, 256-bin grayscale histogram (0 to 8).",
    "rms_contrast": "Population standard deviation of grayscale / 255.",
    "average_gradient": "Mean sqrt((dx^2+dy^2)/2) on shared forward-difference interior of grayscale / 255; zero for a singleton axis.",
    "laplacian_variance": "Population variance of OpenCV Laplacian on grayscale / 255, ksize=3, reflected border.",
    "interpretation": "UIQM/UCIQE are quality proxies. Entropy, contrast, gradient and Laplacian variance are descriptive; noise/oversharpening can increase them. These are not detection metrics or reference-based fidelity measures.",
    "sources": ["https://doi.org/10.1109/JOE.2015.2469915", "https://doi.org/10.1109/TIP.2015.2491020"],
}


def _rgb(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image; convert arrays explicitly to uint8 RGB.")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if not rgb.size:
        raise ValueError("Cannot score an empty image.")
    return rgb


def uciqe(image):
    """Preserve the pipeline's original normalized-Lab UCIQE convention."""
    lab = cv2.cvtColor(_rgb(image), cv2.COLOR_RGB2LAB).astype(np.float32)
    luminance = lab[..., 0] / 255.0
    channel_a = (lab[..., 1] - 128.0) / 255.0
    channel_b = (lab[..., 2] - 128.0) / 255.0
    chroma = np.sqrt(channel_a * channel_a + channel_b * channel_b)
    saturation = chroma / np.sqrt(chroma * chroma + luminance * luminance + 1e-12)
    contrast = np.percentile(luminance, 99) - np.percentile(luminance, 1)
    return float(0.4680 * np.std(chroma)  + 0.2745 * contrast + 0.2576 * np.mean(saturation))


def _extrema(array, block_size=10):
    h, w = array.shape[:2]
    bh, bw = min(h, block_size), min(w, block_size)
    nh, nw = h // bh, w // bw
    blocks = array[:nh * bh, :nw * bw].reshape(nh, bh, nw, bw, -1)
    return blocks.min(axis=(1, 3, 4)), blocks.max(axis=(1, 3, 4))


def _trimmed_stats(values):
    flat = np.sort(values.ravel())
    trim = int(0.1 * flat.size)
    mean = flat[trim:flat.size - trim].mean()
    return mean, np.mean((flat - mean) ** 2)


def uiqm_components(image):
    rgb = _rgb(image).astype(np.float64)
    r, g, b = np.moveaxis(rgb, -1, 0)
    mrg, vrg = _trimmed_stats(r - g)
    myb, vyb = _trimmed_stats((r + g) / 2 - b)
    uicm = -0.0268 * np.hypot(mrg, myb) + 0.1586 * np.sqrt(vrg + vyb)
    sharpness = []
    for channel in (r, g, b):
        dx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
        dy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
        edge = np.hypot(dx, dy)
        if edge.max() > 0:
            edge *= 255 / edge.max()
        low, high = _extrema(edge * channel)
        terms = np.zeros_like(low)
        valid = (low > 0) & (high > 0)
        terms[valid] = np.log(high[valid] / low[valid])
        sharpness.append(2 * terms.mean())
    uism = np.dot(sharpness, (0.299, 0.587, 0.114))
    low, high = _extrema(rgb)
    ratio = np.divide(high - low, high + low, out=np.zeros_like(low), where=(high + low) > 0)
    terms = np.zeros_like(ratio)
    valid = ratio > 0
    terms[valid] = -ratio[valid] * np.log(ratio[valid])
    uiconm = terms.mean()
    return {"uiqm": float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm),
            "uicm": float(uicm), "uism": float(uism), "uiconm": float(uiconm)}


def image_metrics(image):
    gray8 = cv2.cvtColor(_rgb(image), cv2.COLOR_RGB2GRAY)
    gray = gray8.astype(np.float64) / 255
    hist = np.bincount(gray8.ravel(), minlength=256)
    probabilities = hist[hist > 0] / gray8.size
    gradient = 0.0
    if min(gray.shape) > 1:
        dx = gray[:-1, 1:] - gray[:-1, :-1]
        dy = gray[1:, :-1] - gray[:-1, :-1]
        gradient = float(np.sqrt((dx * dx + dy * dy) / 2).mean())
    scores = uiqm_components(image)
    scores.update(uciqe=uciqe(image), entropy=float(-np.sum(probabilities * np.log2(probabilities))),
                  rms_contrast=float(gray.std()), average_gradient=gradient,
                  laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3,
                                                       borderType=cv2.BORDER_REFLECT).var()))
    return {key: scores[key] for key in METRIC_NAMES}


def stage_metrics(named_images):
    scored = [(name, image_metrics(image)) for name, image in named_images]
    by_name = dict(scored)
    original, mirnet = by_name["Original"], by_name["UICE-MIRNet"]
    return [dict(stage=name, **scores,
                 **{f"{key}_change_from_original": scores[key] - original[key] for key in METRIC_NAMES},
                 **{f"{key}_change_from_mirnet": scores[key] - mirnet[key] for key in METRIC_NAMES})
            for name, scores in scored]


def export_metrics(report, json_path):
    """Write detailed JSON and one CSV row per test image and pipeline stage."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report["metric_definitions"] = METRIC_DEFINITIONS
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    fields = ["test_image", "stage", *METRIC_NAMES,
              *(f"{key}_change_from_original" for key in METRIC_NAMES),
              *(f"{key}_change_from_mirnet" for key in METRIC_NAMES)]
    with json_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image_report in report["images"]:
            for result in image_report["results"]:
                writer.writerow(dict(test_image=image_report["test_image"], **result))


def score_saved_report(report_path, output_path=None):
    """Re-score saved PNG stages without loading a model or running training."""
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for image_report in report["images"]:
        named_images = []
        for stage, filename in image_report["stage_images"].items():
            path = Path(filename)
            if not path.is_absolute():
                path = report_path.parent / path
            with Image.open(path) as image:
                named_images.append((stage, image.convert("RGB")))
        if len({image.size for _, image in named_images}) != 1:
            raise ValueError("All stages must have the same resolution for comparison.")
        image_report["results"] = stage_metrics(named_images)
    report["metrics_note"] = METRIC_DEFINITIONS["interpretation"]
    report["source_report"] = str(report_path)
    output_path = Path(output_path) if output_path else report_path.with_name("image_quality_metrics.json")
    export_metrics(report, output_path)
    return report


def print_metric_table(report):
    shown = METRIC_NAMES[:6]
    print("Checkpoint:", report["checkpoint"])
    for image_report in report["images"]:
        print("\nTest image:", image_report["test_image"])
        print(f"{'Stage':<25}" + "".join(f"{key:>20}" for key in shown))
        for result in image_report["results"]:
            print(f"{result['stage']:<25}" + "".join(f"{result[key]:>20.5f}" for key in shown))
        print("Changes from MIRNet (post-processing stages):")
        for result in image_report["results"]:
            if result["stage"] not in ("Original", "UICE-MIRNet"):
                print(f"{result['stage']:<25}" + "".join(
                    f"{result[key + '_change_from_mirnet']:>+20.5f}" for key in shown))


def score_final_export(export_root):
    """Score every final PNG in a completed epoch-25 export, without inference.

    Produces per-image CSV, detailed JSON, and a summary CSV. Summary statistics
    weight each image equally and use population standard deviation (ddof=0).
    """
    export_root = Path(export_root).resolve()
    manifest_path = export_root / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["configuration"]
    if manifest["status"] != "complete" or config["completed_training_epochs"] != 25:
        raise ValueError("Expected a completed epoch-25 image export.")
    sources = config["sources"]
    expected = {row["output_filename"] for row in sources}
    source_names = {row["filename"] for row in sources}
    if not sources or len(expected) != len(sources) or set(manifest["completed"]) != source_names:
        raise ValueError("Export manifest has missing or duplicate image records.")
    final_dir = export_root / "04_final_clahe"
    actual = {path.name for path in final_dir.glob("*.png")}
    if actual != expected:
        raise ValueError(f"Final PNGs do not match the manifest: missing={len(expected - actual)}, "
                         f"extra={len(actual - expected)}")
    results = []
    for index, source in enumerate(sources, 1):
        path = final_dir / source["output_filename"]
        with Image.open(path) as image:
            scores = image_metrics(image)
            width, height = image.size
        if not all(np.isfinite(value) for value in scores.values()):
            raise ValueError(f"Non-finite metric for {path}")
        results.append(dict(source_filename=source["filename"], filename=path.name,
                            width=width, height=height,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **scores))
        if index == 1 or index % 100 == 0 or index == len(sources):
            print(f"Scored {index}/{len(sources)} final images", flush=True)
    summary = {}
    for name in METRIC_NAMES:
        values = np.asarray([row[name] for row in results], dtype=np.float64)
        summary[name] = dict(mean=float(values.mean()), std=float(values.std(ddof=0)),
                             median=float(np.median(values)), min=float(values.min()),
                             max=float(values.max()))
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "MIRNet epoch 25 + gray-world white balance + gamma + CLAHE",
        "image_directory": str(final_dir), "image_count": len(results),
        "export_manifest": str(manifest_path),
        "checkpoint": config["checkpoint"], "checkpoint_sha256": config["checkpoint_sha256"],
        "post_processing": config["post_processing"],
        "metric_definitions": METRIC_DEFINITIONS,
        "aggregation": "Equal weight per image; population standard deviation (ddof=0).",
        "summary": summary, "results": results,
    }
    json_path = export_root / "final_image_metrics.json"
    csv_path = export_root / "final_image_metrics.csv"
    summary_path = export_root / "final_metrics_summary.csv"
    # Write only after every expected image has been scored successfully.
    temporary = json_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(json_path)
    for path, rows, fields in (
        (csv_path, results, ["source_filename", "filename", "width", "height", "sha256", *METRIC_NAMES]),
        (summary_path, [dict(metric=name, **summary[name]) for name in METRIC_NAMES],
         ["metric", "mean", "std", "median", "min", "max"]),
    ):
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    print(f"\nFinal-image metrics: {len(results)} images")
    for name in METRIC_NAMES[:6]:
        print(f"{name:<22} mean={summary[name]['mean']:.6f}  std={summary[name]['std']:.6f}")
    print(f"Per-image CSV: {csv_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Detailed JSON: {json_path}")
    return report


def score_export_stages(export_root):
    """Score raw input and every exported stage for attribution, per image."""
    export_root = Path(export_root).resolve()
    manifest_path = export_root / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["configuration"]
    if manifest.get("status") != "complete":
        raise ValueError("Expected a completed image export.")
    stage_paths = {
        "UICE-MIRNet": export_root / "01_mirnet",
        "+ Color correction": export_root / "02_white_balance",
        "+ Gamma correction": export_root / "03_gamma",
        "+ CLAHE (final)": export_root / "04_final_clahe",
    }
    if not all(path.is_dir() for path in stage_paths.values()):
        raise ValueError("Expected all four exported stage directories.")
    source_dir = Path(config["test_dir"])
    image_reports = []
    for index, source in enumerate(config["sources"], 1):
        original_path = source_dir / source["filename"]
        if not original_path.is_file():
            raise FileNotFoundError(f"Source test image is missing: {original_path}")
        named_images = []
        with Image.open(original_path) as image:
            named_images.append(("Original", image.convert("RGB")))
        for stage, folder in stage_paths.items():
            path = folder / source["output_filename"]
            if not path.is_file():
                raise FileNotFoundError(f"Missing {stage} export: {path}")
            with Image.open(path) as image:
                named_images.append((stage, image.convert("RGB")))
        if len({image.size for _, image in named_images}) != 1:
            raise ValueError(f"Stage dimensions differ for {source['filename']}")
        image_reports.append({"test_image": str(original_path), "results": stage_metrics(named_images)})
        if index == 1 or index % 100 == 0 or index == len(config["sources"]):
            print(f"Scored {index}/{len(config['sources'])} images across all stages", flush=True)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": config["checkpoint"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "export_manifest": str(manifest_path),
        "post_processing": config["post_processing"],
        "metrics_note": METRIC_DEFINITIONS["interpretation"],
        "images": image_reports,
    }
    json_path = export_root / "all_stage_image_metrics.json"
    export_metrics(report, json_path)
    summaries = []
    for stage in ("Original", *stage_paths):
        stage_rows = [next(row for row in item["results"] if row["stage"] == stage) for item in image_reports]
        for metric in METRIC_NAMES:
            values = np.asarray([row[metric] for row in stage_rows], dtype=np.float64)
            summaries.append({"stage": stage, "metric": metric, "mean": float(values.mean()),
                              "std": float(values.std(ddof=0)), "median": float(np.median(values))})
    summary_path = export_root / "all_stage_metrics_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "metric", "mean", "std", "median"])
        writer.writeheader()
        writer.writerows(summaries)
    print(f"All-stage CSV: {json_path.with_suffix('.csv')}")
    print(f"All-stage summary: {summary_path}")
    return report
