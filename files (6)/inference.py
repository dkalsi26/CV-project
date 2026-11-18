"""
inference.py
============
Runs a trained UICE-MIRNet checkpoint on a folder of raw underwater images
and saves the enhanced outputs. These enhanced images are what you would
then feed into YOLOv4 for training/evaluation, matching the paper's pipeline.

Usage:
    python inference.py --checkpoint checkpoints/uice_mirnet_epoch100.pth \
                         --input path/to/raw_images --output path/to/enhanced_images
"""

import os
import argparse

import numpy as np
import torch
from PIL import Image

from model import UICEMIRNet
from dataset import UnpairedInferenceDataset
from torch.utils.data import DataLoader


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    train_args = ckpt.get("args", {})
    channels = train_args.get("channels", 32)
    num_rrb = train_args.get("num_rrb", 3)

    model = UICEMIRNet(in_channels=3, channels=channels, num_rrb=num_rrb).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    dataset = UnpairedInferenceDataset(args.input)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    os.makedirs(args.output, exist_ok=True)

    with torch.no_grad():
        for img_tensor, fname, orig_size in loader:
            img_tensor = img_tensor.to(device)

            # Pad to a multiple of 8 (helps avoid shape mismatches from the
            # 3 levels of 2x down/up-sampling inside UI-CEB)
            _, _, h, w = img_tensor.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h or pad_w:
                img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")

            out = model(img_tensor)
            out = torch.clamp(out, 0.0, 1.0)

            if pad_h or pad_w:
                out = out[:, :, :h, :w]

            out_array = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_array = np.rint(out_array * 255.0).astype(np.uint8)
            out_img = Image.fromarray(out_array, mode="RGB")
            out_path = os.path.join(args.output, fname[0])
            out_img.save(out_path)
            print(f"Enhanced: {fname[0]}")

    print(f"\nDone. Enhanced images saved to: {args.output}")


def get_args():
    parser = argparse.ArgumentParser(description="Run UICE-MIRNet inference")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--input", required=True, help="Folder of raw underwater images")
    parser.add_argument("--output", required=True, help="Folder to save enhanced images")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_inference(args)
