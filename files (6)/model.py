

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias)


def conv1x1(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)


def downsample(x, factor):
    if factor == 1:
        return x
    return F.avg_pool2d(x, kernel_size=factor, stride=factor)


def upsample_like(x, ref):
    """Upsample x to match ref's spatial size (bilinear)."""
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def upsample(x, factor):
    return F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False)


class SCAB(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        # F -> CONV -> ReLU -> CONV  => FM   (Eq. 5)
        self.fm_branch = nn.Sequential(
            conv3x3(channels, channels),
            nn.ReLU(inplace=True),
            conv3x3(channels, channels),
        )

        # Spatial Attention (SA): GAP + GMP over channel dim -> concat -> conv -> sigmoid (Eq. 6)
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)

        # Modified Channel Attention (M-CA): GAP -> conv-relu-conv -> ReLU (replacing sigmoid) (Eq. 7)
        mid = max(channels // reduction, 4)
        self.mca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(channels, mid),
            nn.ReLU(inplace=True),
            conv1x1(mid, channels),
            nn.ReLU(inplace=True),  # paper replaces sigmoid with ReLU here
        )

        # Final fusion conv after concatenating SA*FM and M-CA*FM
        self.fuse_conv = conv1x1(channels * 2, channels)

    def forward(self, x):
        fm = self.fm_branch(x)

        # Spatial attention map
        avg_pool = torch.mean(fm, dim=1, keepdim=True)   # channel-wise GAP -> (B,1,H,W)
        max_pool, _ = torch.max(fm, dim=1, keepdim=True)  # channel-wise GMP -> (B,1,H,W)
        sa_map = torch.sigmoid(self.sa_conv(torch.cat([avg_pool, max_pool], dim=1)))
        sa_out = fm * sa_map

        # Modified channel attention
        ca_weights = self.mca(fm)  # (B,C,1,1)
        ca_out = fm * ca_weights

        combined = self.fuse_conv(torch.cat([sa_out, ca_out], dim=1))
        return x + combined  # Eq. 4 residual

class FSFB(nn.Module):
    """
    Fuses an arbitrary number of feature streams (all already resampled to
    the SAME spatial resolution) using a Selective-Kernel-style
    fuse + select mechanism.
    """

    def __init__(self, channels, num_streams=4, reduction=8):
        super().__init__()
        self.num_streams = num_streams
        mid = max(channels // reduction, 4)

        # Channel-wise statistic + compact representation: GAP -> conv -> ReLU
        self.descriptor = nn.Sequential(
            conv1x1(channels, mid),
            nn.ReLU(inplace=True),
        )

        # One conv per stream to produce that stream's (pre-softmax) attention logits
        self.select_convs = nn.ModuleList([
            conv1x1(mid, channels) for _ in range(num_streams)
        ])

    def forward(self, streams):
        assert len(streams) == self.num_streams, \
            f"Expected {self.num_streams} streams, got {len(streams)}"

        # Fuse: element-wise sum
        fused = torch.stack(streams, dim=0).sum(dim=0)  # (B,C,H,W)

        # GAP -> channel-wise statistic
        gap = F.adaptive_avg_pool2d(fused, 1)  # (B,C,1,1)
        z = self.descriptor(gap)               # (B,mid,1,1)

        # Select: per-stream logits, softmax ACROSS streams (per channel)
        logits = [conv(z) for conv in self.select_convs]        # list of (B,C,1,1)
        logits = torch.stack(logits, dim=0)                      # (S,B,C,1,1)
        weights = torch.softmax(logits, dim=0)                   # softmax over streams

        # Re-calibrate and aggregate (Eq. 3)
        stacked_streams = torch.stack(streams, dim=0)             # (S,B,C,H,W)
        out = (stacked_streams * weights).sum(dim=0)               # (B,C,H,W)
        return out

class UICEB(nn.Module):
    """
    4-resolution multi-scale block: input (1x), L1 (1/2), L2 (1/4), L3 (1/8).
    Each is passed through SCAB, cross-fused via FSFB at every resolution,
    passed through SCAB again, then fused back to the input resolution.
    """

    def __init__(self, channels):
        super().__init__()
        self.scab_stage1 = nn.ModuleList([SCAB(channels) for _ in range(4)])
        self.fsfb_stage1 = nn.ModuleList([FSFB(channels, num_streams=4) for _ in range(4)])
        self.scab_stage2 = nn.ModuleList([SCAB(channels) for _ in range(4)])
        self.fsfb_final = FSFB(channels, num_streams=4)
        self.out_conv = conv1x1(channels, channels)

    def forward(self, x):
        # Build the 4 resolution streams: 1x, 1/2, 1/4, 1/8
        s0 = x
        s1 = downsample(x, 2)
        s2 = downsample(s1, 2)
        s3 = downsample(s2, 2)

        streams = [s0, s1, s2, s3]

        # First SCAB pass (identify low-colorfulness areas per-scale)
        streams = [scab(s) for scab, s in zip(self.scab_stage1, streams)]

        # Cross-scale fusion: for each target resolution, resample every
        # other stream to that resolution and fuse with FSFB.
        fused = []
        for target_idx in range(4):
            target = streams[target_idx]
            resampled = []
            for src_idx, s in enumerate(streams):
                if src_idx == target_idx:
                    resampled.append(s)
                else:
                    resampled.append(upsample_like(s, target) if src_idx > target_idx
                                      else downsample(s, 2 ** (target_idx - src_idx)))
            fused.append(self.fsfb_stage1[target_idx](resampled))

        # Second SCAB pass
        fused = [scab(f) for scab, f in zip(self.scab_stage2, fused)]

        # Final fusion back to the original (1x) resolution
        final_inputs = [fused[0]] + [upsample_like(f, fused[0]) for f in fused[1:]]
        merged = self.fsfb_final(final_inputs)
        out = self.out_conv(merged)

        return x + out  # residual


# --------------------------------------------------------------------------
# Recursive Residual Block (RRB) -- Eq. 2
# --------------------------------------------------------------------------
class RRB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_in = conv3x3(channels, channels)
        self.ceb1 = UICEB(channels)
        self.ceb2 = UICEB(channels)
        self.conv_out = conv3x3(channels, channels)

    def forward(self, x):
        f = self.conv_in(x)
        f = self.ceb1(f)
        f = self.ceb2(f)
        f = self.conv_out(f)
        return x + f  # residual


# --------------------------------------------------------------------------
# Full UICE-MIRNet -- Fig. 1 / Eq. 1
# --------------------------------------------------------------------------
class UICEMIRNet(nn.Module):
    def __init__(self, in_channels=3, channels=32, num_rrb=3):
        super().__init__()
        self.conv_in = conv3x3(in_channels, channels)
        self.rrbs = nn.ModuleList([RRB(channels) for _ in range(num_rrb)])
        self.conv_out = conv3x3(channels, in_channels)

    def forward(self, x):
        x0 = self.conv_in(x)
        f = x0
        for rrb in self.rrbs:
            f = rrb(f)
        r = self.conv_out(f)
        out = r + x  # Eq. 1: Iout = R + I
        return out


if __name__ == "__main__":
    # Quick shape sanity check
    model = UICEMIRNet(in_channels=3, channels=32, num_rrb=3)
    dummy = torch.randn(1, 3, 416, 416)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print("Input :", dummy.shape)
    print("Output:", out.shape)
    print(f"Params: {n_params:,}")
