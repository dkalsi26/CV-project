"""Train UICE-MIRNet, then run its cumulative enhancement chain on test images.

This is the end-to-end experiment driver. It deliberately keeps the roles
of the splits separate:

* train: optimizer updates
* val: checkpoint selection only
* test: final inference/comparison only

Use ``--check_only`` to validate paths and pairing without starting training.
"""

import argparse
from argparse import Namespace
from pathlib import Path

from compare_test_image import compare
from dataset import IMG_EXTENSIONS
from generate_pseudo_targets import process_trash_icra
from train import train


def image_names(folder):
    return {
        path.name
        for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    }


def sequence_id(filename):
    """Return the capture-sequence identifier used by Trash-ICRA filenames."""
    stem = Path(filename).stem
    marker = "_frame"
    if marker not in stem:
        raise ValueError(f"Cannot infer sequence from filename: {filename}")
    return stem.split(marker, 1)[0]


def validate_sequence_isolation(dataset_root):
    """Reject frame-wise splits that leak the same capture into evaluation."""
    sequences = {
        split: {sequence_id(name) for name in image_names(dataset_root / split)}
        for split in ("train", "val", "test")
    }
    overlaps = {
        "train/val": sequences["train"] & sequences["val"],
        "train/test": sequences["train"] & sequences["test"],
        "val/test": sequences["val"] & sequences["test"],
    }
    leaked = {name: sorted(values) for name, values in overlaps.items() if values}
    if leaked:
        summary = "; ".join(
            f"{name}={len(values)} (e.g. {values[:3]})" for name, values in leaked.items()
        )
        raise RuntimeError(
            "Sequence leakage detected: frames from the same capture occur in multiple splits. "
            f"{summary}. Create a new grouped dataset with split_by_sequence.py and use it here."
        )
    return {split: len(values) for split, values in sequences.items()}


def resolve_dataset_root(path):
    """Accept either trash_ICRA19/ or trash_ICRA19/dataset/."""
    root = Path(path).resolve()
    candidates = (root, root / "dataset")
    for candidate in candidates:
        if all((candidate / split).is_dir() for split in ("train", "val", "test")):
            return candidate
    raise FileNotFoundError(
        f"Expected train, val, and test folders under {root} or {root / 'dataset'}"
    )


def validate_pairing(dataset_root, paired_root):
    """Fail before training if pseudo-pairs are missing, stale, or leaked."""
    sequence_counts = validate_sequence_isolation(dataset_root)
    counts = {}
    for split in ("train", "val"):
        raw = image_names(dataset_root / split)
        paired_input_dir = paired_root / split / "input"
        paired_target_dir = paired_root / split / "target"
        if not paired_input_dir.is_dir() or not paired_target_dir.is_dir():
            raise FileNotFoundError(
                f"Missing paired {split} data under {paired_root}. "
                "Run again with --prepare_targets."
            )
        paired_input = image_names(paired_input_dir)
        paired_target = image_names(paired_target_dir)
        if raw != paired_input:
            raise RuntimeError(
                f"{split}/input does not exactly match the raw {split} split "
                f"({len(raw)} raw, {len(paired_input)} paired inputs)."
            )
        if paired_input != paired_target:
            missing = sorted(paired_input - paired_target)[:5]
            extra = sorted(paired_target - paired_input)[:5]
            raise RuntimeError(
                f"{split} input/target filenames do not match. "
                f"Missing targets: {missing}; extra targets: {extra}"
            )
        counts[split] = len(raw)

    counts["test"] = len(image_names(dataset_root / "test"))
    if any(value == 0 for value in counts.values()):
        raise RuntimeError(f"At least one Trash-ICRA19 split is empty: {counts}")
    counts["sequences"] = sequence_counts
    return counts


def run(args):
    dataset_root = resolve_dataset_root(args.dataset_root)
    paired_root = Path(args.paired_root).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    results_dir = Path(args.results_dir).resolve()

    if args.prepare_targets:
        # Never create test targets: test remains strictly held out.
        process_trash_icra(
            str(dataset_root),
            str(paired_root),
            splits=("train", "val"),
            overwrite=args.overwrite_targets,
        )

    counts = validate_pairing(dataset_root, paired_root)
    print(
        "Verified Trash-ICRA19 split isolation: "
        f"train={counts['train']}, val={counts['val']}, test={counts['test']}"
    )
    print("Sequence counts: " + ", ".join(f"{split}={count}" for split, count in counts["sequences"].items()))
    print(f"Paired training data: {paired_root}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Comparison outputs: {results_dir}")

    if args.check_only:
        print("Check-only mode: no training or inference was started.")
        return None

    # Defaults keep older notebooks compatible with the expanded pipeline.
    prefetch_factor = getattr(args, "prefetch_factor", 4)
    requested_images = getattr(args, "images", None)
    legacy_image = getattr(args, "image", None)
    if requested_images is None and legacy_image:
        requested_images = [legacy_image]
    num_test_images = getattr(args, "num_test_images", 2)

    training_args = Namespace(
        data_root=str(paired_root),
        train_split="train",
        val_split="val",
        no_val=False,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        lr=args.lr,
        channels=args.channels,
        num_rrb=args.num_rrb,
        num_workers=args.num_workers,
        prefetch_factor=prefetch_factor,
        save_dir=str(checkpoint_dir),
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        resume=args.resume,
        seed=args.seed,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
    )
    train(training_args)

    best_checkpoint = checkpoint_dir / "uice_mirnet_best.pth"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(
            f"Training ended without the expected validation checkpoint: {best_checkpoint}"
        )

    comparison_args = Namespace(
        checkpoint=str(best_checkpoint),
        test_dir=str(dataset_root / "test"),
        images=requested_images,
        num_images=num_test_images,
        output_dir=str(results_dir),
        gamma=args.gamma,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=tuple(args.clahe_tile_grid),
    )
    return compare(comparison_args)


def get_args():
    parser = argparse.ArgumentParser(
        description="Train UICE-MIRNet on Trash-ICRA19 and compare the held-out result"
    )
    parser.add_argument("--dataset_root", required=True, help="Trash-ICRA19 root or dataset folder")
    parser.add_argument("--paired_root", required=True, help="Paired pseudo-target data root")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument(
        "--prepare_targets",
        action="store_true",
        help="Generate train/val pseudo-target pairs before training",
    )
    parser.add_argument("--overwrite_targets", action="store_true")
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Validate all splits and pairs, then exit without training/inference",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--num_rrb", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Specific held-out test filenames (default: first two)",
    )
    parser.add_argument("--num_test_images", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1.2)
    parser.add_argument("--clahe_clip_limit", type=float, default=2.0)
    parser.add_argument("--clahe_tile_grid", type=int, nargs=2, default=(8, 8))
    return parser.parse_args()


if __name__ == "__main__":
    run(get_args())
