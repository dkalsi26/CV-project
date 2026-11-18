"""
train.py
========
Trains UICE-MIRNet on a paired (input, target) dataset.

Usage (after preparing Trash-ICRA19 with generate_pseudo_targets.py):
    python train.py --data_root path/to/trash_icra_paired --epochs 100 --batch_size 8 \
                     --patch_size 256 --lr 2e-4 --save_dir checkpoints

`data_root` may contain split folders such as `train/input`, `train/target`,
`val/input`, and `val/target`, or the original flat `input/target` layout.
See dataset.py and generate_pseudo_targets.py.
"""

import os
import argparse
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import UICEMIRNet
from losses import UICEMIRNetLoss
from dataset import PairedEnhancementDataset


def paired_split_exists(root, split):
    return all(os.path.isdir(os.path.join(root, split, name)) for name in ("input", "target"))


def validate(model, loader, criterion, device, use_amp, amp_dtype):
    model.eval()
    running_loss = 0.0
    out_of_range = 0.0
    with torch.no_grad():
        for inp, tgt, _ in loader:
            inp = inp.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            tgt = tgt.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                # Keep the prediction unclamped while computing the loss.
                # A hard clamp gives zero gradient to pixels outside [0, 1],
                # which can stall an under-trained residual model.
                pred = model(inp)
                loss, _ = criterion(pred, tgt)
            running_loss += loss.item()
            out_of_range += ((pred < 0) | (pred > 1)).float().mean().item()
    return running_loss / len(loader), out_of_range / len(loader)


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, args, best_val_loss):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }, path)


def train(args):
    # Let PyTorch CPU kernels use every logical core. Data loading is also
    # parallelized independently below with one requested worker per core.
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("This GPU does not support BF16; falling back to FP16 AMP")
        amp_dtype = torch.float16
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    prefetch_factor = getattr(args, "prefetch_factor", 4)
    print(f"CPU DataLoader workers: {args.num_workers}")
    if args.num_workers > 0:
        print(f"Prefetched batches per worker: {prefetch_factor}")
    print("Tensor memory format: channels_last")
    print(
        f"Automatic mixed precision: "
        f"{args.amp_dtype if use_amp else 'disabled'}"
    )

    split_layout = paired_split_exists(args.data_root, args.train_split)
    train_split = args.train_split if split_layout else None
    train_ds = PairedEnhancementDataset(
        args.data_root, patch_size=args.patch_size, train=True, split=train_split
    )
    if len(train_ds) < args.batch_size:
        raise ValueError(
            f"Training set has {len(train_ds)} images, fewer than batch size {args.batch_size}"
        )

    pin_memory = device.type == "cuda"
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        # Keep CPU workers alive and prepare the next batches while the GPU
        # works on the current batch.
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    print(f"Training samples: {len(train_ds)} ({train_split or 'flat'} layout)")

    val_loader = None
    if not args.no_val and paired_split_exists(args.data_root, args.val_split):
        val_ds = PairedEnhancementDataset(
            args.data_root, patch_size=args.patch_size, train=False, split=args.val_split
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_options,
        )
        print(f"Validation samples: {len(val_ds)}")
    elif not args.no_val:
        print(f"No paired '{args.val_split}' split found; training without validation")

    model = UICEMIRNet(in_channels=3, channels=args.channels, num_rrb=args.num_rrb).to(
        device,
        memory_format=torch.channels_last,
    )
    criterion = UICEMIRNetLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    # BF16 has FP32-like exponent range and does not need gradient scaling.
    # FP16 does; UICE-MIRNet's deep residual attention can otherwise overflow.
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp and amp_dtype == torch.float16,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume}: {start_epoch} epochs completed; "
              f"continuing at epoch {start_epoch + 1}/{args.epochs}")

    stop_after_epoch = getattr(args, "stop_after_epoch", None)
    if stop_after_epoch is not None:
        if not 1 <= stop_after_epoch <= args.epochs:
            raise ValueError(
                f"stop_after_epoch must be in [1, {args.epochs}], got {stop_after_epoch}"
            )
        stop_epoch = stop_after_epoch
    else:
        stop_epoch = args.epochs

    if start_epoch >= stop_epoch:
        print(f"Nothing to train: {start_epoch} epochs already completed (segment ends at {stop_epoch}).")
        return

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        out_of_range = 0.0
        running_components = {"charbonnier": 0.0, "ssim": 0.0, "color": 0.0}

        for step, (inp, tgt, _) in enumerate(train_loader):
            inp = inp.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            tgt = tgt.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred = model(inp)
                loss, components = criterion(pred, tgt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            out_of_range += ((pred.detach() < 0) | (pred.detach() > 1)).float().mean().item()
            for k in running_components:
                running_components[k] += components[k]

            if (step + 1) % args.log_interval == 0:
                avg_loss = running_loss / (step + 1)
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} "
                      f"(char={running_components['charbonnier']/(step+1):.4f}, "
                      f"ssim={running_components['ssim']/(step+1):.4f}, "
                      f"color={running_components['color']/(step+1):.4f})")

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = running_loss / len(train_loader)
        print(f"== Epoch {epoch+1} done in {epoch_time:.1f}s | avg loss: {avg_epoch_loss:.4f} "
              f"| out-of-range: {out_of_range / len(train_loader):.2%} "
              f"| lr: {scheduler.get_last_lr()[0]:.2e} ==")

        if val_loader is not None:
            val_loss, val_out_of_range = validate(model, val_loader, criterion, device, use_amp, amp_dtype)
            print(f"Validation loss: {val_loss:.4f} | out-of-range: {val_out_of_range:.2%}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.save_dir, "uice_mirnet_best.pth")
                save_checkpoint(
                    best_path, epoch, model, optimizer, scheduler, scaler, args, best_val_loss
                )
                print(f"Saved best checkpoint: {best_path}")

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == stop_epoch:
            ckpt_path = os.path.join(args.save_dir, f"uice_mirnet_epoch{epoch+1}.pth")
            save_checkpoint(
                ckpt_path, epoch, model, optimizer, scheduler, scaler, args, best_val_loss
            )
            print(f"Saved checkpoint: {ckpt_path}")

    print(f"Training segment complete: {stop_epoch}/{args.epochs} epochs.")


def get_args():
    parser = argparse.ArgumentParser(description="Train UICE-MIRNet")
    parser.add_argument(
        "--data_root",
        required=True,
        help="Paired root containing train/input + train/target (or flat input + target)",
    )
    parser.add_argument("--train_split", default="train", help="Training split folder name")
    parser.add_argument("--val_split", default="val", help="Validation split folder name")
    parser.add_argument("--no_val", action="store_true", help="Disable validation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--channels", type=int, default=32, help="Base feature channels in UICE-MIRNet")
    parser.add_argument("--num_rrb", type=int, default=3, help="Number of Recursive Residual Blocks")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Batches prefetched by each DataLoader worker",
    )
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--stop_after_epoch",
        type=int,
        default=None,
        help="Stop after this absolute epoch and save a resumable checkpoint (default: train through --epochs)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed-precision training")
    parser.add_argument(
        "--amp_dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="CUDA AMP datatype (default: bfloat16, recommended for numerical stability)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
