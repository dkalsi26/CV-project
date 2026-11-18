"""
Dataset loader for UICE-MIRNet training.

IMPORTANT CAVEAT
----------------
The Brackish Underwater Dataset and Trash-ICRA19 Dataset ship with OBJECT
DETECTION annotations (bounding boxes), not "clean" enhanced ground-truth
images. To train an enhancement network you need (degraded, enhanced) PAIRS.

This loader accepts either a simple paired-folder layout:

    data_root/
        input/    <- raw underwater images (from Brackish / Trash-ICRA19)
        target/   <- corresponding enhanced/reference images, SAME filenames

or the split layout produced for Trash-ICRA19 by
``generate_pseudo_targets.py``:

    data_root/
        train/input/  train/target/
        val/input/    val/target/
        test/input/   test/target/g

If you don't have real reference images, use `generate_pseudo_targets.py`
(included in this project) to bootstrap a `target/` folder from `input/`
using a strong classical fusion-based enhancement method. This is the same
workaround commonly used in underwater-enhancement literature when paired
ground truth doesn't exist.
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class PairedEnhancementDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, train=True, split=None):
        """
        Args:
            root_dir: paired dataset root
            patch_size: random crop size used during training (None = full image, resized)
            train: whether to apply random augmentation (crop/flip)
            split: optional split name (for example ``train`` or ``val``)
        """
        pair_root = os.path.join(root_dir, split) if split else root_dir
        self.input_dir = os.path.join(pair_root, "input")
        self.target_dir = os.path.join(pair_root, "target")
        if not os.path.isdir(self.input_dir):
            raise FileNotFoundError(f"Missing input folder: {self.input_dir}")
        if not os.path.isdir(self.target_dir):
            raise FileNotFoundError(f"Missing target folder: {self.target_dir}")

        self.filenames = sorted([
            f for f in os.listdir(self.input_dir) if f.lower().endswith(IMG_EXTENSIONS)
        ])
        if not self.filenames:
            raise RuntimeError(f"No supported images found in {self.input_dir}")
        missing = [f for f in self.filenames if not os.path.exists(os.path.join(self.target_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} input files have no matching target file, e.g. {missing[:5]}"
            )

        self.patch_size = patch_size
        self.train = train

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        inp = Image.open(os.path.join(self.input_dir, fname)).convert("RGB")
        tgt = Image.open(os.path.join(self.target_dir, fname)).convert("RGB")

        if self.patch_size is not None:
            if self.train:
                inp, tgt = self._random_crop(inp, tgt, self.patch_size)
                inp, tgt = self._random_flip(inp, tgt)
            else:
                # Do not distort the 4:3 camera images into square validation
                # inputs.  The network is fully convolutional; resize only when
                # needed and reflect-pad both paired images to a common size.
                inp, tgt = self._resize_and_pad(inp, tgt, self.patch_size)

        inp_t = self._to_tensor(inp)
        tgt_t = self._to_tensor(tgt)
        return inp_t, tgt_t, fname

    @staticmethod
    def _to_tensor(image):
        array = np.array(image, dtype=np.float32, copy=True)
        return torch.from_numpy(array.transpose(2, 0, 1)).div_(255.0)

    @staticmethod
    def _random_crop(inp, tgt, patch_size):
        w, h = inp.size
        if w < patch_size or h < patch_size:
            # upscale small images so a crop is always possible
            scale = patch_size / min(w, h)
            new_size = (max(patch_size, int(w * scale)), max(patch_size, int(h * scale)))
            inp = inp.resize(new_size, Image.BICUBIC)
            tgt = tgt.resize(new_size, Image.BICUBIC)
            w, h = inp.size

        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        box = (x, y, x + patch_size, y + patch_size)
        return inp.crop(box), tgt.crop(box)

    @staticmethod
    def _random_flip(inp, tgt):
        if random.random() < 0.5:
            inp = inp.transpose(Image.FLIP_LEFT_RIGHT)
            tgt = tgt.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            inp = inp.transpose(Image.FLIP_TOP_BOTTOM)
            tgt = tgt.transpose(Image.FLIP_TOP_BOTTOM)
        return inp, tgt

    @staticmethod
    def _resize_and_pad(inp, tgt, patch_size):
        """Fit a pair inside a square while preserving its aspect ratio."""
        w, h = inp.size
        scale = min(patch_size / w, patch_size / h)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        inp = inp.resize(size, Image.Resampling.BICUBIC)
        tgt = tgt.resize(size, Image.Resampling.BICUBIC)
        pad_w, pad_h = patch_size - size[0], patch_size - size[1]
        left, right = pad_w // 2, pad_w - pad_w // 2
        top, bottom = pad_h // 2, pad_h - pad_h // 2

        def reflect_pad(image):
            array = np.asarray(image)
            # Reflect padding requires an axis larger than one pixel.
            mode = "reflect" if min(array.shape[:2]) > 1 else "edge"
            padded = np.pad(array, ((top, bottom), (left, right), (0, 0)), mode=mode)
            return Image.fromarray(padded, mode="RGB")

        return reflect_pad(inp), reflect_pad(tgt)


class UnpairedInferenceDataset(Dataset):
    """Simple dataset for running inference on a folder of raw images (no targets needed)."""

    def __init__(self, folder):
        self.folder = folder
        self.filenames = sorted([
            f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTENSIONS)
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.folder, fname)).convert("RGB")
        array = np.array(img, dtype=np.float32, copy=True)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).div_(255.0)
        return tensor, fname, img.size  # (w, h) for later resize-back
