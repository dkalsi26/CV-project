"""Export all held-out test images through the fixed epoch-25 pipeline.

Inference only. Existing completed images are reused when the manifest and
source files match. No training, metrics computation, or image resizing.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image

from compare_test_image import enhance_image, apply_cumulative_classical_stages
from dataset import IMG_EXTENSIONS
from inference import load_model


STAGES = ("01_mirnet", "02_white_balance", "03_gamma", "04_final_clahe")


def write_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def export_test_images(project_root, output_dir=None):
    root = Path(project_root).resolve()
    checkpoint = root / "checkpoints" / "trash_icra" / "uice_mirnet_epoch25.pth"
    dataset = root / "trash_ICRA19"
    if not (dataset / "test").is_dir():
        dataset = dataset / "dataset"
    test_dir = dataset / "test"
    sources = sorted(p for p in test_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)
    if not sources:
        raise ValueError(f"No test images in {test_dir}")
    names = [p.stem + ".png" for p in sources]
    if len(set(name.casefold() for name in names)) != len(names):
        raise ValueError("Test filenames would collide when converted to PNG.")
    output_dir = Path(output_dir or root / "results" / "epoch25_enhanced_test").resolve()
    if output_dir == test_dir.resolve() or test_dir.resolve() in output_dir.parents:
        raise ValueError("Export directory must be separate from the input test images.")
    configuration = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "completed_training_epochs": 25,
        "test_dir": str(test_dir),
        "post_processing": {"gamma": 1.2, "clahe_clip_limit": 2.0, "clahe_tile_grid": [8, 8]},
        "stage_folders": list(STAGES),
        "format": "RGB PNG, original resolution",
        "sources": [{"filename": p.name, "output_filename": name,
                     "bytes": p.stat().st_size, "modified_ns": p.stat().st_mtime_ns}
                    for p, name in zip(sources, names)],
    }
    manifest_path = output_dir / "export_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["configuration"] != configuration:
            raise ValueError("Export configuration or input files changed; use a new output directory.")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("Output folder is nonempty without a matching export manifest.")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"configuration": configuration, "status": "running", "completed": []}
    for stage in STAGES:
        (output_dir / stage).mkdir(exist_ok=True)

    # Validate metadata before loading a network or writing any stage images.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if int(saved["epoch"]) + 1 != 25:
        raise ValueError("Expected exactly 25 completed training epochs.")
    del saved
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    model = load_model(checkpoint, device).to(memory_format=torch.channels_last)
    model.requires_grad_(False)
    manifest["device"] = str(device)
    manifest["precision"] = str(amp_dtype) if device.type == "cuda" else "torch.float32"
    manifest["status"] = "running"
    write_manifest(manifest_path, manifest)
    completed = set(manifest["completed"])
    started = time.monotonic()
    print(f"Epoch-25 inference on {device}; {len(sources)} held-out test images", flush=True)
    print(f"Export folder: {output_dir}", flush=True)
    for index, (source, name) in enumerate(zip(sources, names), 1):
        destinations = [output_dir / stage / name for stage in STAGES]
        if source.name in completed and all(path.is_file() for path in destinations):
            continue
        with Image.open(source) as image:
            original = image.convert("RGB")
        mirnet = enhance_image(model, original, device, True, amp_dtype)
        white_balance, gamma, final = apply_cumulative_classical_stages(mirnet, 1.2, 2.0, (8, 8))
        for image, destination in zip((mirnet, white_balance, gamma, final), destinations):
            if image.size != original.size:
                raise RuntimeError(f"Unexpected size change for {source.name}")
            temporary = destination.with_suffix(".tmp")
            image.save(temporary, format="PNG")
            temporary.replace(destination)
        completed.add(source.name)
        manifest["completed"] = sorted(completed)
        write_manifest(manifest_path, manifest)
        if index == 1 or index % 25 == 0 or index == len(sources):
            print(f"Exported {len(completed)}/{len(sources)} images through all four stages "
                  f"({time.monotonic() - started:.1f}s)", flush=True)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != configuration["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during export.")
    manifest["status"] = "complete"
    manifest["total_source_images"] = len(sources)
    manifest["total_enhanced_images"] = len(sources) * len(STAGES)
    write_manifest(manifest_path, manifest)
    print(f"Complete: {len(sources) * len(STAGES)} enhanced PNGs saved.", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    export_test_images(args.project_root, args.output_dir)
