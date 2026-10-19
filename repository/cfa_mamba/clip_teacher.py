from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


DEFAULT_PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a clean photo of a {}.",
    "a close-up photo of a {}.",
    "a rendering of a {}.",
]


class ClipSemanticTeacher(nn.Module):
    """Frozen external CLIP supervision; historical class name retained for compatibility."""

    def __init__(
        self,
        class_names: Sequence[str],
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        backend: str = "auto",
        prompt_templates: Iterable[str] | None = None,
        temperature: float = 0.01,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.class_names = list(class_names)
        self.model_name = model_name
        self.pretrained = pretrained
        self.backend = backend
        self.prompt_templates = list(prompt_templates or DEFAULT_PROMPT_TEMPLATES)
        self.temperature = temperature
        self.device = torch.device(device)

        self.model, self.preprocess, self.tokenize, self.backend = self._load_clip_backend()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        text_features = self._build_text_features()
        self.register_buffer("text_features", text_features, persistent=False)

    def _load_clip_backend(self):
        backend = self.backend.lower()
        if backend not in {"auto", "open_clip", "clip"}:
            raise ValueError(f"Unsupported CLIP backend: {self.backend}")

        if backend in {"auto", "open_clip"}:
            try:
                import open_clip

                if self.pretrained and os.path.isfile(self.pretrained):
                    model, _, preprocess_val = open_clip.create_model_and_transforms(
                        self.model_name,
                        pretrained=None,
                        device=self.device,
                    )
                    open_clip.load_checkpoint(model, self.pretrained)
                else:
                    model, _, preprocess_val = open_clip.create_model_and_transforms(
                        self.model_name,
                        pretrained=self.pretrained,
                        device=self.device,
                    )
                tokenizer = open_clip.get_tokenizer(self.model_name)
                return model, preprocess_val, tokenizer, "open_clip"
            except ImportError:
                if backend == "open_clip":
                    raise

        try:
            import clip

            openai_name = self.model_name.replace("ViT-B-16", "ViT-B/16").replace("ViT-L-14", "ViT-L/14")
            model, preprocess = clip.load(openai_name, device=self.device)
            return model, preprocess, clip.tokenize, "clip"
        except ImportError as exc:
            raise ImportError(
                "Install either open_clip_torch or OpenAI CLIP on the server. "
                "Recommended: pip install open_clip_torch"
            ) from exc

    @torch.no_grad()
    def _build_text_features(self) -> torch.Tensor:
        per_template_features = []
        for template in self.prompt_templates:
            prompts = [template.format(name) for name in self.class_names]
            tokens = self.tokenize(prompts).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features.float(), dim=-1)
            per_template_features.append(text_features)

        text_features = torch.stack(per_template_features, dim=0).mean(dim=0)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.model.encode_image(images.to(self.device))
        image_features = F.normalize(image_features.float(), dim=-1)
        logits = image_features @ self.text_features.t()
        return F.softmax(logits / self.temperature, dim=-1)

    @torch.no_grad()
    def build_cache(
        self,
        loader: DataLoader,
        dataset_size: int,
        cache_path: str = "",
        force_recompute: bool = False,
        amp: bool = True,
        logger=None,
    ) -> torch.Tensor:
        if cache_path and os.path.exists(cache_path) and not force_recompute:
            if logger is not None:
                logger.info(f"Loading CLIP semantic cache from {cache_path}")
            return torch.load(cache_path, map_location="cpu")

        if logger is not None:
            logger.info("Building CLIP semantic cache for target domain")

        probs = torch.zeros(dataset_size, len(self.class_names), dtype=torch.float32)
        self.eval()
        for step, batch in enumerate(loader):
            images, _, indices = batch
            images = images.to(self.device, non_blocking=True)
            indices = indices.long()
            with torch.cuda.amp.autocast(enabled=amp and self.device.type == "cuda"):
                batch_probs = self.forward(images)
            probs[indices] = batch_probs.detach().cpu()

            if logger is not None and step % 100 == 0:
                logger.info(f"CLIP cache batch {step}/{len(loader)}")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(probs, cache_path)
            if logger is not None:
                logger.info(f"Saved CLIP semantic cache to {cache_path}")
        return probs
