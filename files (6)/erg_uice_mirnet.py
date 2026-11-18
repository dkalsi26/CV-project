"""
Edge-Reliability-Guided UICE-MIRNet (ERG-UICE-MIRNet).

This is a research variant of ``model.UICEMIRNet``.  It deliberately leaves
``model.py`` unchanged and retains the original RRB, SCAB, residual, and
four-resolution-stream design.  Only the final FSFB in each UI-CEB receives
edge guidance.

Inputs follow the existing project convention: RGB float tensors in [0, 1].
The surrounding dataset/inference code is responsible for any reflect padding
to a multiple of eight; this module uses the same zero-padded 3x3 convention
as the base model's convolutions.
"""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import FSFB, RRB, SCAB, conv1x1, conv3x3, downsample, upsample_like


FusionMode = Literal[
    "feature",
    "reliability_weights",
    "feature_and_reliability_weights",
]


class EdgeExtractionModule(nn.Module):
    """Fixed, differentiable Sobel edge extractor for an RGB input tensor.

    Returns ``[magnitude, Gx, Gy]`` with shape ``(B, 3, H, W)``.  The Sobel
    kernels are buffers (not Parameters), so they move with ``model.to()`` and
    are included in checkpoints without becoming trainable.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @staticmethod
    def luminance(rgb: torch.Tensor) -> torch.Tensor:
        """Differentiable Rec. 709 luminance for an RGB tensor."""
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(
                "EdgeExtractionModule expects an RGB tensor of shape (B, 3, H, W), "
                f"got {tuple(rgb.shape)}"
            )
        # The project supplies unnormalised [0, 1] RGB.  No clipping or image
        # re-normalisation is applied, preserving gradients into the input.
        return (
            0.2126 * rgb[:, 0:1]
            + 0.7152 * rgb[:, 1:2]
            + 0.0722 * rgb[:, 2:3]
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        luminance = self.luminance(rgb)
        gx = F.conv2d(luminance, self.sobel_x, padding=1)
        gy = F.conv2d(luminance, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + self.epsilon)
        return torch.cat((magnitude, gx, gy), dim=1)


class EdgeReliabilityModule(nn.Module):
    """Predict a soft confidence map for Sobel edges and shallow RGB cues."""

    def __init__(self, edge_channels: int = 3, hidden_channels: int = 16):
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be at least 1")
        self.rgb_projection = nn.Sequential(
            conv3x3(3, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.estimator = nn.Sequential(
            conv3x3(edge_channels + hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            conv3x3(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            conv1x1(hidden_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        if rgb.shape[0] != edge_features.shape[0] or rgb.shape[-2:] != edge_features.shape[-2:]:
            raise ValueError("RGB and edge features must have the same batch and spatial dimensions")
        shallow_rgb = self.rgb_projection(rgb)
        return self.estimator(torch.cat((edge_features, shallow_rgb), dim=1))


class EdgeGuidedFSFB(FSFB):
    """Original FSFB with optional edge injection at its first (1x) stream.

    ``fusion_mode='feature'`` is the default research model: it preserves the
    original FSFB attention-weight computation exactly.  ``'reliability_weights'``
    is the ablation that changes only its weights.  The explicit combined mode
    is available for a later experiment, rather than silently mixing the two
    interventions.
    """

    valid_fusion_modes = {
        "feature",
        "reliability_weights",
        "feature_and_reliability_weights",
    }

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        fusion_mode: FusionMode = "feature",
        gamma_init: float = 0.0,
    ):
        super().__init__(channels=channels, num_streams=4, reduction=reduction)
        if fusion_mode not in self.valid_fusion_modes:
            raise ValueError(
                f"fusion_mode must be one of {sorted(self.valid_fusion_modes)}, got {fusion_mode!r}"
            )
        self.fusion_mode = fusion_mode
        mid = max(channels // reduction, 4)
        self.edge_projection = conv1x1(3, channels)
        self.gate = conv1x1(channels + channels + 1, channels)
        # A zero initial value makes the feature-injection variants initially
        # identical to UICE-MIRNet, while gamma remains trainable.
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        # Only used by the weight-modulation ablations.  It maps a global
        # reliability descriptor into the original FSFB descriptor space.
        self.reliability_descriptor = conv1x1(1, mid)

    def _inject_edges(
        self,
        first_stream: torch.Tensor,
        edge_features: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        edge_at_f1 = F.interpolate(
            edge_features, size=first_stream.shape[-2:], mode="bilinear", align_corners=False
        )
        reliability_at_f1 = F.interpolate(
            reliability, size=first_stream.shape[-2:], mode="bilinear", align_corners=False
        )
        projected_edges = self.edge_projection(edge_at_f1)
        gate = torch.sigmoid(
            self.gate(torch.cat((first_stream, projected_edges, reliability_at_f1), dim=1))
        )
        return first_stream + self.gamma * gate * projected_edges

    def _reliability_weighted_fusion(
        self, streams: Iterable[torch.Tensor], reliability: torch.Tensor
    ) -> torch.Tensor:
        """Original FSFB selection, with a reliability descriptor added to z."""
        streams = list(streams)
        fused = torch.stack(streams, dim=0).sum(dim=0)
        z = self.descriptor(F.adaptive_avg_pool2d(fused, 1))
        reliability_at_f1 = F.interpolate(
            reliability, size=streams[0].shape[-2:], mode="bilinear", align_corners=False
        )
        reliability_z = self.reliability_descriptor(
            F.adaptive_avg_pool2d(reliability_at_f1, 1)
        )
        logits = torch.stack([conv(z + reliability_z) for conv in self.select_convs], dim=0)
        weights = torch.softmax(logits, dim=0)
        return (torch.stack(streams, dim=0) * weights).sum(dim=0)

    def forward(
        self,
        streams: Iterable[torch.Tensor],
        edge_features: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        streams = list(streams)
        if len(streams) != 4:
            raise ValueError(f"Expected four final-fusion streams, got {len(streams)}")
        if edge_features.shape[1] != 3 or reliability.shape[1] != 1:
            raise ValueError("Expected edge features (B, 3, H, W) and reliability (B, 1, H, W)")

        inject_features = self.fusion_mode in {"feature", "feature_and_reliability_weights"}
        modulate_weights = self.fusion_mode in {
            "reliability_weights",
            "feature_and_reliability_weights",
        }
        if inject_features:
            streams[0] = self._inject_edges(streams[0], edge_features, reliability)
        if modulate_weights:
            return self._reliability_weighted_fusion(streams, reliability)
        # Calls the original FSFB implementation unchanged, including its
        # native attention-weight generation.
        return super().forward(streams)


class ERGUICEB(nn.Module):
    """UI-CEB whose final (and only final) FSFB is edge guided."""

    def __init__(self, channels: int, fusion_mode: FusionMode = "feature", gamma_init: float = 0.0):
        super().__init__()
        self.scab_stage1 = nn.ModuleList([SCAB(channels) for _ in range(4)])
        self.fsfb_stage1 = nn.ModuleList([FSFB(channels, num_streams=4) for _ in range(4)])
        self.scab_stage2 = nn.ModuleList([SCAB(channels) for _ in range(4)])
        self.fsfb_final = EdgeGuidedFSFB(
            channels, fusion_mode=fusion_mode, gamma_init=gamma_init
        )
        self.out_conv = conv1x1(channels, channels)

    def forward(
        self, x: torch.Tensor, edge_features: torch.Tensor, reliability: torch.Tensor
    ) -> torch.Tensor:
        s0 = x
        s1 = downsample(x, 2)
        s2 = downsample(s1, 2)
        s3 = downsample(s2, 2)
        streams = [s0, s1, s2, s3]

        streams = [scab(s) for scab, s in zip(self.scab_stage1, streams)]
        fused = []
        for target_idx in range(4):
            target = streams[target_idx]
            resampled = []
            for src_idx, stream in enumerate(streams):
                if src_idx == target_idx:
                    resampled.append(stream)
                elif src_idx > target_idx:
                    resampled.append(upsample_like(stream, target))
                else:
                    resampled.append(downsample(stream, 2 ** (target_idx - src_idx)))
            fused.append(self.fsfb_stage1[target_idx](resampled))

        fused = [scab(feature) for scab, feature in zip(self.scab_stage2, fused)]
        final_inputs = [fused[0]] + [upsample_like(feature, fused[0]) for feature in fused[1:]]
        merged = self.fsfb_final(final_inputs, edge_features, reliability)
        return x + self.out_conv(merged)


class ERGRRB(nn.Module):
    """Original RRB topology, passing one cached guidance representation onward."""

    def __init__(self, channels: int, fusion_mode: FusionMode = "feature", gamma_init: float = 0.0):
        super().__init__()
        self.conv_in = conv3x3(channels, channels)
        self.ceb1 = ERGUICEB(channels, fusion_mode=fusion_mode, gamma_init=gamma_init)
        self.ceb2 = ERGUICEB(channels, fusion_mode=fusion_mode, gamma_init=gamma_init)
        self.conv_out = conv3x3(channels, channels)

    def forward(
        self, x: torch.Tensor, edge_features: torch.Tensor, reliability: torch.Tensor
    ) -> torch.Tensor:
        features = self.conv_in(x)
        features = self.ceb1(features, edge_features, reliability)
        features = self.ceb2(features, edge_features, reliability)
        return x + self.conv_out(features)


class ERGUICEMIRNet(nn.Module):
    """UICE-MIRNet with six final-FSFB edge-guidance sites by default.

    With ``num_rrb=3`` there are three RRBs times two UI-CEBs, so edge maps
    are reused by exactly six edge-guided final FSFBs.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 32,
        num_rrb: int = 3,
        reliability_channels: int = 16,
        fusion_mode: FusionMode = "feature",
        gamma_init: float = 0.0,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("ERG-UICE-MIRNet currently requires RGB input (in_channels=3)")
        if num_rrb < 1:
            raise ValueError("num_rrb must be at least 1")
        self.edge_extractor = EdgeExtractionModule()
        self.edge_reliability = EdgeReliabilityModule(hidden_channels=reliability_channels)
        self.conv_in = conv3x3(in_channels, channels)
        self.rrbs = nn.ModuleList(
            [ERGRRB(channels, fusion_mode=fusion_mode, gamma_init=gamma_init) for _ in range(num_rrb)]
        )
        self.conv_out = conv3x3(channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # These are deliberately computed once and shared by all six UI-CEBs.
        edge_features = self.edge_extractor(x)
        reliability = self.edge_reliability(x, edge_features)

        features = self.conv_in(x)
        for rrb in self.rrbs:
            features = rrb(features, edge_features, reliability)
        return self.conv_out(features) + x

    def load_baseline_weights(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[list[str], list[str]]:
        """Load UICE-MIRNet weights and leave ERG-specific layers freshly initialised.

        Pass either a baseline model state dict or ``checkpoint['model_state']``.
        The return value is ``(missing_keys, unexpected_keys)`` for logging.
        """
        incompatible = self.load_state_dict(state_dict, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


if __name__ == "__main__":
    model = ERGUICEMIRNet(in_channels=3, channels=32, num_rrb=3)
    dummy = torch.randn(2, 3, 128, 160, requires_grad=True)
    output = model(dummy)
    output.mean().backward()
    print("Input :", dummy.shape)
    print("Output:", output.shape)
    print("Input gradient is finite:", bool(torch.isfinite(dummy.grad).all()))
    print("Trainable parameters:", sum(parameter.numel() for parameter in model.parameters()))
