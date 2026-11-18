"""
Loss functions for UICE-MIRNet training.

The paper does not specify its training loss explicitly, so this follows the
common recipe used by MIRNet and most underwater-enhancement papers:
Charbonnier (smooth L1) reconstruction loss + an SSIM term for structural
similarity, optionally combined with a color-consistency term to encourage
colorfulness (in line with the paper's stated goal of increasing UCIQE/UIQM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """A differentiable, robust variant of L1 loss: sqrt((x-y)^2 + eps^2)."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


def gaussian_window(window_size, sigma, channels, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    """1 - SSIM, computed per-channel with a Gaussian window."""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, pred, target):
        channels = pred.size(1)
        window = gaussian_window(self.window_size, self.sigma, channels, pred.device).to(pred.dtype)
        pad = self.window_size // 2

        mu_p = F.conv2d(pred, window, padding=pad, groups=channels)
        mu_t = F.conv2d(target, window, padding=pad, groups=channels)

        mu_p_sq, mu_t_sq, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

        sigma_p_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_p_sq
        sigma_t_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_t_sq
        sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pt

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / \
                   ((mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2))
        return 1.0 - ssim_map.mean()


class ColorConsistencyLoss(nn.Module):
    """
    Penalizes the angular difference between mean color vectors of the
    prediction and target, to discourage color-cast drift (encourages
    higher colorfulness / better UCIQE-style behaviour).
    """

    def forward(self, pred, target):
        pred_mean = pred.mean(dim=[2, 3])      # (B,3)
        target_mean = target.mean(dim=[2, 3])  # (B,3)
        pred_norm = F.normalize(pred_mean, dim=1)
        target_norm = F.normalize(target_mean, dim=1)
        cos_sim = (pred_norm * target_norm).sum(dim=1)
        return (1.0 - cos_sim).mean()


class UICEMIRNetLoss(nn.Module):
    def __init__(self, w_char=1.0, w_ssim=0.2, w_color=0.05):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.color = ColorConsistencyLoss()
        self.w_char = w_char
        self.w_ssim = w_ssim
        self.w_color = w_color

    def forward(self, pred, target):
        l_char = self.charbonnier(pred, target)
        l_ssim = self.ssim(pred, target)
        l_color = self.color(pred, target)
        total = self.w_char * l_char + self.w_ssim * l_ssim + self.w_color * l_color
        return total, {"charbonnier": l_char.item(), "ssim": l_ssim.item(), "color": l_color.item()}
