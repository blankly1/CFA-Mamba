"""Experimental controls; the original scan class remains unchanged."""
import torch
from torch import nn
from cfa_mamba.model import SemanticConditionedScanAdapter


class NoAdapter(nn.Module):
    def forward(self, feature, clip_probs=None, channel_first=True):
        return feature


class ResetState(SemanticConditionedScanAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.A_log.requires_grad_(False)
        for parameter in self.delta_proj.parameters():
            parameter.requires_grad_(False)

    def _scan(self, tokens, decay, b_gain, c_gain):
        return (c_gain * b_gain + self.D[None, :]).unsqueeze(-1) * tokens


class ConditionalMLP(SemanticConditionedScanAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d = kwargs['feature_dim']
        del self.delta_proj, self.b_proj, self.c_proj, self.A_log, self.D
        self.pointwise = nn.Sequential(nn.Linear(2*d, d+1), nn.GELU(), nn.Linear(d+1, d))

    def forward(self, feature, clip_probs, channel_first=True):
        x = feature if channel_first else feature.permute(0, 3, 1, 2).contiguous()
        semantic = self.semantic_proj(clip_probs.float()).to(dtype=x.dtype)
        visual = self.input_proj(x)
        tokens = visual.flatten(2).transpose(1, 2)
        condition = semantic[:, None, :].expand(-1, tokens.shape[1], -1)
        scanned = self.pointwise(torch.cat([tokens, condition], -1)).transpose(1, 2).reshape_as(visual)
        semantic_map = self.semantic_value(semantic)[:, :, None, None].expand_as(scanned)
        gate = self.gate(torch.cat([scanned, semantic_map], dim=1))
        out = x + self.norm(self.out_proj(gate * scanned + (1-gate) * semantic_map))
        return out if channel_first else out.permute(0, 2, 3, 1).contiguous()


class SharedCondition(SemanticConditionedScanAdapter):
    def __init__(self, mean_probs, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('mean_probs', mean_probs.reshape(1, -1).float())

    def forward(self, feature, clip_probs, channel_first=True):
        x = feature if channel_first else feature.permute(0, 3, 1, 2).contiguous()
        visual = self.input_proj(x)
        semantic = self.semantic_proj(clip_probs.float()).to(x.dtype)
        shared = self.semantic_proj(self.mean_probs).to(x.dtype)
        delta = torch.nn.functional.softplus(self.delta_proj(shared))
        decay = torch.exp(-torch.exp(self.A_log.to(x.dtype))[None, :] * delta)
        b = torch.sigmoid(self.b_proj(shared))
        c = torch.sigmoid(self.c_proj(shared))
        tokens = visual.flatten(2)
        scanned = self._scan(tokens, decay, b, c)
        if self.bidirectional:
            scanned = 0.5 * (scanned + self._scan(tokens.flip(-1), decay, b, c).flip(-1))
        scanned = scanned.reshape_as(visual)
        semantic_map = self.semantic_value(semantic)[:, :, None, None].expand_as(scanned)
        gate = self.gate(torch.cat([scanned, semantic_map], dim=1))
        out = x + self.norm(self.out_proj(gate*scanned + (1-gate)*semantic_map))
        return out if channel_first else out.permute(0, 2, 3, 1).contiguous()
