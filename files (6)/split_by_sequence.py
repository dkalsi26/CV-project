"""Create a leakage-free Trash-ICRA split without modifying the source dataset.

Frames sharing a ``<sequence>_frameNNNN`` prefix are kept together.  Image
files and all same-stem sidecars (for example YOLO ``.txt`` and Pascal VOC
``.xml`` annotations) are copied or hard-linked into a new dataset root.
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from dataset import IMG_EXTENSIONS


def sequence_id(path):
    marker = "_frame"
    if marker not in path.stem:
        raise ValueError(f"Cannot infer sequence from filename: {path.name}")
    return path.stem.split(marker, 1)[0]


def source_directory(path):
    path = Path(path).resolve()
    candidate = path / "dataset" if (path / "dataset").is_dir() else path
    if not all((candidate / split).is_dir() for split in ("train", "val", "test")):
        raise FileNotFoundError("Expected train, val, and test directories under --source-root.")
    return candidate


def collect_groups(root):
    groups = defaultdict(list)
    for split in ("train", "val", "test"):
        for image in (root / split).iterdir():
            if image.is_file() and image.suffix.lower() in IMG_EXTENSIONS:
                groups[sequence_id(image)].append(image)
    if not groups:
        raise RuntimeError("No supported images found.")
    return groups


def assign_groups(groups, ratios, seed):
    """Greedily fill split targets by frame count after deterministic shuffling."""
    names = list(groups)
    random.Random(seed).shuffle(names)
    total = sum(len(groups[name]) for name in names)
    targets = {name: total * ratio for name, ratio in ratios.items()}
    assigned = {name: [] for name in ratios}
    counts = {name: 0 for name in ratios}
    for group in sorted(names, key=lambda name: len(groups[name]), reverse=True):
        split = min(ratios, key=lambda name: (counts[name] - targets[name], counts[name]))
        assigned[split].append(group)
        counts[split] += len(groups[group])
    return assigned, counts


def copy_group(images, destination, hardlink):
    destination.mkdir(parents=True, exist_ok=True)
    for image in images:
        for source in image.parent.glob(image.stem + ".*"):
            target = destination / source.name
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite {target}")
            if hardlink:
                try:
                    os.link(source, target)
                    continue
                except OSError:
                    pass
            shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardlink", action="store_true", help="Hard-link files when possible; otherwise copy them.")
    args = parser.parse_args()
    if any(r <= 0 for r in args.ratios) or abs(sum(args.ratios) - 1.0) > 1e-8:
        raise ValueError("--ratios must be positive and sum to one.")
    source = source_directory(args.source_root)
    output = Path(args.output_root).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("--output-root must be a separate sibling directory, never the source dataset.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("--output-root must be empty or absent; existing data is never overwritten.")
    groups = collect_groups(source)
    ratios = dict(zip(("train", "val", "test"), args.ratios))
    assigned, counts = assign_groups(groups, ratios, args.seed)
    for split, names in assigned.items():
        for name in names:
            copy_group(groups[name], output / split, args.hardlink)
    manifest = {
        "source_root": str(source), "seed": args.seed, "ratios": ratios,
        "frames": counts, "sequences": {split: len(names) for split, names in assigned.items()},
        "groups": assigned,
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Created sequence-disjoint split:", ", ".join(f"{name}={counts[name]} frames/{len(assigned[name])} sequences" for name in ratios))


if __name__ == "__main__":
    main()
