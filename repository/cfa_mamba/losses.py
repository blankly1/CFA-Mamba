from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CGMambaLossOutput:
    total: torch.Tensor
    pseudo: torch.Tensor
    consistency: torch.Tensor
    entropy: torch.Tensor
    diversity: torch.Tensor
    anchor: torch.Tensor
    selected_ratio: float


class CGMambaLoss(nn.Module):
    """CFA-Mamba objectives (historical class name retained).

    Reliability r = exp(-KL(CLIP || source)) * max(1-source_conf, 0.05).
    Hard pseudo-label CE uses normalized mask*r*CLIP_conf weights.
    Soft KL(CLIP || target) uses normalized mask*r weights.
    The epoch schedule scales only these two CLIP output losses.
    Anchor = mean(source_conf * KL(source || target)).
    """

    def __init__(
        self,
        lambda_pseudo: float = 1.0,
        lambda_consistency: float = 1.0,
        lambda_entropy: float = 0.1,
        lambda_diversity: float = 1.0,
        lambda_anchor: float = 0.1,
        confidence_threshold: float = 0.55,
        use_clip_hard_labels: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_pseudo = lambda_pseudo
        self.lambda_consistency = lambda_consistency
        self.lambda_entropy = lambda_entropy
        self.lambda_diversity = lambda_diversity
        self.lambda_anchor = lambda_anchor
        self.confidence_threshold = confidence_threshold
        self.use_clip_hard_labels = use_clip_hard_labels

    def _zero(self, t: torch.Tensor) -> torch.Tensor:
        return t.sum() * 0.0

    @torch.no_grad()
    def _source_weights(
        self,
        clip_probs: torch.Tensor,
        source_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Per-sample weight = agreement(source, CLIP) × (1 - source_conf).
        High only when source is uncertain AND agrees with CLIP.
        """
        cp = clip_probs[mask] if mask is not None else clip_probs
        sl = source_logits[mask] if mask is not None else source_logits
        src_probs = F.softmax(sl.float(), dim=1)
        kl = F.kl_div(torch.log(src_probs + 1e-8), cp, reduction="none").sum(1).clamp(0)
        agreement_w = torch.exp(-kl)
        source_unc_w = (1.0 - src_probs.max(1).values).clamp(min=0.05)
        return agreement_w * source_unc_w  # [N]

    def pseudo_label_loss(
        self,
        logits: torch.Tensor,
        clip_probs: torch.Tensor,
        source_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        confidence, pseudo_label = clip_probs.max(dim=1)
        selected = confidence.ge(self.confidence_threshold)
        if not selected.any():
            return self._zero(logits), selected

        sel_logits = logits[selected]
        with torch.no_grad():
            if source_logits is not None:
                w = self._source_weights(clip_probs, source_logits, mask=selected)
                w = w * confidence[selected]
            else:
                w = confidence[selected]

        loss_per = F.cross_entropy(sel_logits, pseudo_label[selected], reduction="none")
        loss = (w * loss_per).sum() / (w.sum() + 1e-8)
        return loss, selected

    def consistency_loss(
        self,
        logits: torch.Tensor,
        clip_probs: torch.Tensor,
        source_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = clip_probs.max(1).values.ge(self.confidence_threshold)
        if not mask.any():
            return self._zero(logits)
        kl_per = F.kl_div(
            F.log_softmax(logits[mask], dim=1),
            clip_probs[mask].detach(),
            reduction="none",
        ).sum(1)
        if source_logits is not None:
            w = self._source_weights(clip_probs, source_logits, mask=mask)
            return (w * kl_per).sum() / (w.sum() + 1e-8)
        return kl_per.mean()

    def entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, 1)
        return -(p * F.log_softmax(logits, 1)).sum(1).mean()

    def diversity_loss(self, logits: torch.Tensor) -> torch.Tensor:
        mp = F.softmax(logits, 1).mean(0)
        return (mp * torch.log(mp + 1e-6)).sum()

    def anchor_loss(self, logits: torch.Tensor, source_logits: torch.Tensor) -> torch.Tensor:
        """KL(student ∥ source) weighted by source_conf: confident source → strong anchor."""
        with torch.no_grad():
            src_probs = F.softmax(source_logits.float(), dim=1)
            src_conf = src_probs.max(1).values  # [B]
        kl_per = F.kl_div(
            F.log_softmax(logits, 1), src_probs.detach(), reduction="none"
        ).sum(1)
        return (src_conf * kl_per).mean()

    def forward(
        self,
        logits: torch.Tensor,
        clip_probs: torch.Tensor,
        clip_weight: float = 1.0,
        source_logits: torch.Tensor | None = None,
    ) -> CGMambaLossOutput:
        pseudo, selected = self.pseudo_label_loss(logits, clip_probs, source_logits)
        consistency = self.consistency_loss(logits, clip_probs, source_logits)
        entropy = self.entropy_loss(logits)
        diversity = self.diversity_loss(logits)
        anchor = self.anchor_loss(logits, source_logits) if source_logits is not None \
            else self._zero(logits)

        total = (
            clip_weight * (self.lambda_pseudo * pseudo + self.lambda_consistency * consistency)
            + self.lambda_entropy * entropy
            + self.lambda_diversity * diversity
            + self.lambda_anchor * anchor
        )
        return CGMambaLossOutput(
            total=total,
            pseudo=pseudo.detach(),
            consistency=consistency.detach(),
            entropy=entropy.detach(),
            diversity=diversity.detach(),
            anchor=anchor.detach(),
            selected_ratio=float(selected.float().mean()),
        )
