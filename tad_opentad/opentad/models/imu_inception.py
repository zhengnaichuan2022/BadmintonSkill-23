"""PyTorch iSPL-Inception style 1D CNN backbone for IMU TAD (time-preserving)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .builder import MODELS


class _InceptionModule1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int = 64,
        bottleneck: int = 32,
        max_kernel: int = 68,
        use_bottleneck: bool = True,
        stride: int = 1,
    ):
        super().__init__()
        branch_in = in_channels
        if use_bottleneck and in_channels > 1:
            self.bottleneck = nn.Sequential(
                nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False),
                nn.BatchNorm1d(bottleneck),
                nn.ReLU(inplace=True),
            )
            branch_in = bottleneck
        else:
            self.bottleneck = None

        kernel_sizes = [max(max_kernel // (2**i), 3) for i in range(3)]
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(branch_in, filters, kernel_size=k, padding=k // 2, stride=stride, bias=False),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(inplace=True),
                )
                for k in kernel_sizes
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=stride, padding=1),
            nn.Conv1d(in_channels, filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
        )
        self.out_channels = filters * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_in = self.bottleneck(x) if self.bottleneck is not None else x
        outs = [branch(branch_in) for branch in self.branches]
        outs.append(self.pool_branch(x))
        target_len = min(o.shape[-1] for o in outs)
        outs = [o[..., :target_len] for o in outs]
        return torch.cat(outs, dim=1)


@MODELS.register_module()
class IMUInceptionBackbone(nn.Module):
    """iSPL-Inception backbone for IMU sequences.

    Input:  ``[B, C_in, T]`` raw IMU window
    Output: ``[B, C_out, T // stride]``
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 256,
        network_depth: int = 5,
        filters_number: int = 64,
        max_kernel_size: int = 68,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        stride: int = 4,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.stride = stride

        self.input_norm = nn.BatchNorm1d(in_channels)
        self.modules_list = nn.ModuleList()
        self.shortcut_convs = nn.ModuleList()
        channels = in_channels
        for depth in range(network_depth):
            module = _InceptionModule1D(
                channels,
                filters=filters_number,
                max_kernel=max_kernel_size,
                use_bottleneck=use_bottleneck,
                stride=1,
            )
            self.modules_list.append(module)
            if use_residual and depth % 3 == 2:
                if channels != module.out_channels:
                    self.shortcut_convs.append(
                        nn.Conv1d(channels, module.out_channels, kernel_size=1, bias=False)
                    )
                else:
                    self.shortcut_convs.append(nn.Identity())
            else:
                self.shortcut_convs.append(None)
            channels = module.out_channels

        self.out_channels = out_channels
        self.proj = (
            nn.Identity()
            if channels == out_channels
            else nn.Conv1d(channels, out_channels, kernel_size=1, bias=False)
        )
        self.temporal_pool = nn.MaxPool1d(kernel_size=stride, stride=stride)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                x = self._forward_features(x)
        else:
            x = self._forward_features(x)

        if masks is not None and x.dim() == 3:
            feat_len = x.shape[-1]
            raw_len = masks.shape[-1]
            idx = torch.linspace(0, raw_len - 1, feat_len, device=masks.device).long()
            feat_masks = masks[:, idx]
            x = x * feat_masks.unsqueeze(1).float()
        return x

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        for module, shortcut in zip(self.modules_list, self.shortcut_convs):
            out = module(x)
            if shortcut is not None:
                out = out + shortcut(x)
            x = out
        x = self.proj(x)
        x = self.temporal_pool(x)
        return x
