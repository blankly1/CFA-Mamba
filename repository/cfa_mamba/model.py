from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class SemanticGatedMambaAdapter(nn.Module):
    """Injects CLIP semantic distributions into high-level SfMamba features."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.semantic_proj = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.visual_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.norm = nn.BatchNorm2d(feature_dim)

    def forward(self, feature: torch.Tensor, clip_probs: torch.Tensor, channel_first: bool = True) -> torch.Tensor:
        if clip_probs is None:
            return feature

        if channel_first:
            x = feature
        else:
            x = feature.permute(0, 3, 1, 2).contiguous()

        semantic = self.semantic_proj(clip_probs.float()).to(dtype=x.dtype)
        semantic = semantic[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        visual = self.visual_proj(x)
        gate = self.gate(torch.cat([visual, semantic], dim=1))
        fused = gate * visual + (1.0 - gate) * semantic
        out = x + self.norm(self.out_proj(fused))

        if channel_first:
            return out
        return out.permute(0, 2, 3, 1).contiguous()


class SemanticConditionedScanAdapter(nn.Module):
    """A lightweight CLIP-conditioned state-space adapter over high-level tokens."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False)
        self.semantic_proj = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.delta_proj = nn.Linear(hidden_dim, feature_dim)
        self.b_proj = nn.Linear(hidden_dim, feature_dim)
        self.c_proj = nn.Linear(hidden_dim, feature_dim)
        self.semantic_value = nn.Linear(hidden_dim, feature_dim)
        self.A_log = nn.Parameter(torch.zeros(feature_dim))
        self.D = nn.Parameter(torch.ones(feature_dim))
        self.out_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.norm = nn.BatchNorm2d(feature_dim)
        self.gate = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def _scan(self, tokens: torch.Tensor, decay: torch.Tensor, b_gain: torch.Tensor, c_gain: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = tokens.shape
        state = tokens.new_zeros(batch_size, channels)
        outputs = []
        d_gain = self.D.to(dtype=tokens.dtype)[None, :]
        for step in range(length):
            current = tokens[:, :, step]
            state = decay * state + b_gain * current
            outputs.append((c_gain * state + d_gain * current).unsqueeze(-1))
        return torch.cat(outputs, dim=-1)

    def forward(self, feature: torch.Tensor, clip_probs: torch.Tensor, channel_first: bool = True) -> torch.Tensor:
        if clip_probs is None:
            return feature

        if channel_first:
            x = feature
        else:
            x = feature.permute(0, 3, 1, 2).contiguous()

        visual = self.input_proj(x)
        semantic = self.semantic_proj(clip_probs.float()).to(dtype=x.dtype)
        delta = torch.nn.functional.softplus(self.delta_proj(semantic)).unsqueeze(-1)
        b_gain = torch.sigmoid(self.b_proj(semantic))
        c_gain = torch.sigmoid(self.c_proj(semantic))
        a = -torch.exp(self.A_log.to(dtype=x.dtype))[None, :, None]
        decay = torch.exp(a * delta).squeeze(-1)

        tokens = visual.flatten(2)
        scanned = self._scan(tokens, decay, b_gain, c_gain)
        if self.bidirectional:
            reverse = self._scan(tokens.flip(-1), decay, b_gain, c_gain).flip(-1)
            scanned = 0.5 * (scanned + reverse)

        scanned = scanned.view_as(visual)
        semantic_map = self.semantic_value(semantic)[:, :, None, None].expand_as(scanned)
        gate = self.gate(torch.cat([scanned, semantic_map], dim=1))
        fused = gate * scanned + (1.0 - gate) * semantic_map
        out = x + self.norm(self.out_proj(fused))

        if channel_first:
            return out
        return out.permute(0, 2, 3, 1).contiguous()


class ClipGuidedSfMamba(nn.Module):
    """SfMamba student with a CLIP-conditioned semantic gate before classification."""

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        gate_hidden_dim: int | None = None,
        gate_dropout: float = 0.0,
        inject_after_fusion: bool = True,
        adapter_type: str = "scan",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.inject_after_fusion = inject_after_fusion
        if adapter_type == "gate":
            self.semantic_gate = SemanticGatedMambaAdapter(
                num_classes=num_classes,
                feature_dim=base_model.num_features,
                hidden_dim=gate_hidden_dim,
                dropout=gate_dropout,
            )
        elif adapter_type == "scan":
            self.semantic_gate = SemanticConditionedScanAdapter(
                num_classes=num_classes,
                feature_dim=base_model.num_features,
                hidden_dim=gate_hidden_dim,
                dropout=gate_dropout,
            )
        else:
            raise ValueError(f"Unsupported adapter_type: {adapter_type}")

    @property
    def patch_embed(self):
        return self.base_model.patch_embed

    @property
    def layers(self):
        return self.base_model.layers

    @property
    def fusion_blocks(self):
        return self.base_model.fusion_blocks

    @property
    def classifier(self):
        return self.base_model.classifier

    @property
    def channel_first(self) -> bool:
        return self.base_model.channel_first

    def no_weight_decay(self):
        if hasattr(self.base_model, "no_weight_decay"):
            return self.base_model.no_weight_decay()
        return set()

    def no_weight_decay_keywords(self):
        if hasattr(self.base_model, "no_weight_decay_keywords"):
            return self.base_model.no_weight_decay_keywords()
        return set()

    def forward_features(self, x: torch.Tensor, clip_probs: torch.Tensor | None = None) -> torch.Tensor:
        x = self.base_model.patch_embed(x)
        if self.base_model.pos_embed is not None:
            pos_embed = self.base_model.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.base_model.pos_embed
            x = x + pos_embed

        for i, layer in enumerate(self.base_model.layers):
            x = layer.blocks(x)
            if i != self.base_model.num_layers - 1:
                if layer.downsample is not None:
                    x = layer.downsample(x)
                continue

            for fusion_block in self.base_model.fusion_blocks:
                x = fusion_block(x)

            if self.inject_after_fusion:
                x = self.semantic_gate(x, clip_probs, channel_first=self.channel_first)

            if layer.downsample is not None:
                x = layer.downsample(x)

        if getattr(self.base_model, "relu_bef_cls", False):
            x = self.base_model.relu(x)
        if not self.inject_after_fusion:
            x = self.semantic_gate(x, clip_probs, channel_first=self.channel_first)
        return x

    def forward(self, x: torch.Tensor, clip_probs: torch.Tensor | None = None, return_feat: bool = False):
        feature = self.forward_features(x, clip_probs=clip_probs)
        logits = self.base_model.classifier(feature)
        if return_feat:
            return logits, feature
        return logits

    def save_target_parts(self, output_dir: str) -> OrderedDict:
        state = OrderedDict()
        state["patch_embed"] = self.base_model.patch_embed.state_dict()
        state["layers"] = self.base_model.layers.state_dict()
        state["fusion_blocks"] = self.base_model.fusion_blocks.state_dict()
        state["classifier"] = self.base_model.classifier.state_dict()
        state["semantic_gate"] = self.semantic_gate.state_dict()
        return state
