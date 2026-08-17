"""
SAEG: Synonym Averaged Embedding Generator
============================================

Implements Equations 1-3 from the paper (Section 3.2 "Unknown Object Labelling").

Algorithm:
    Eq 1: f_syn = mean(encode_text(template.format(synonym)) for template in templates)
    Eq 2: f_cls = mean(f_syn for synonym in class_to_synonyms[classname])
    Eq 3: F = stack(f_cls for classname in class_names)  → shape (num_classes, D)

Reference:
    Enhancing Novel Object Detection via Cooperative Foundational Models
    Bharadwaj et al., WACV 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
from typing import List, Optional, Dict
from vild_templates import format_class_prompts


class SAEGTextFeatureGenerator:
    """
    Synonym Averaged Embedding Generator (SAEG).

    Generates robust text features for class names by:
      1. Averaging CLIP embeddings across multiple prompt templates (Eq 1)
      2. Averaging across synonyms for each class (Eq 2)
      3. Stacking into a single feature matrix (Eq 3)
    """

    def __init__(
        self,
        clip_model: torch.nn.Module,
        tokenizer,
        templates: List[str],
        class_to_synonyms: Dict[str, List[str]],
        device: torch.device = None,
        show_progress: bool = True,
    ):
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.templates = templates
        self.class_to_synonyms = class_to_synonyms
        self.device = device or next(clip_model.parameters()).device
        self.show_progress = show_progress

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of texts and return L2-normalized embeddings."""
        tokens = self.tokenizer(texts, context_length=self.clip_model.context_length).to(self.device)
        # Match autocast dtype to model params to avoid Half vs BFloat16 mismatch
        model_dtype = next(self.clip_model.parameters()).dtype
        with torch.cuda.amp.autocast(dtype=model_dtype):
            embeddings = self.clip_model.encode_text(tokens)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings.float()

    def _get_embedding_dim(self) -> int:
        embed_dim = getattr(self.clip_model, 'output_dim', None)
        if embed_dim is not None:
            return embed_dim
        embed_dim = getattr(self.clip_model.text, 'output_dim', None)
        if embed_dim is not None:
            return embed_dim
        with torch.no_grad():
            dummy = self.tokenizer([""], context_length=self.clip_model.context_length).to(self.device)
            dummy_feats = self.clip_model.encode_text(dummy)
            return dummy_feats.shape[-1]

    @torch.no_grad()
    def generate_text_features(
        self,
        class_names: List[str],
        clear_cache_every: int = 50,
    ) -> torch.Tensor:
        """
        Generate SAEG text features for all class names.

        Args:
            class_names: List of LVIS class names (e.g., 1203 classes)
            clear_cache_every: Clear GPU cache every N classes

        Returns:
            text_features: Tensor of shape (num_classes, embedding_dim),
                           L2-normalized along the last dimension
        """
        num_classes = len(class_names)
        embedding_dim = self._get_embedding_dim()

        all_text_features = []
        self._per_class_template_embeds = []

        iterator = tqdm(class_names, desc="[SAEG] Generating text features") if self.show_progress else class_names

        for cls_idx, classname in enumerate(iterator):
            synonyms = self.class_to_synonyms.get(classname, [classname])

            syn_features = []
            all_template_embeds = []

            for syn in synonyms:
                prompts = format_class_prompts(syn, templates=self.templates)

                syn_embeddings = []
                batch_size = 1000
                for batch_start in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[batch_start:batch_start + batch_size]
                    batch_embeds = self.encode_texts(batch_prompts)
                    syn_embeddings.append(batch_embeds.cpu())

                if syn_embeddings:
                    syn_embeddings = torch.cat(syn_embeddings, dim=0)
                    all_template_embeds.append(syn_embeddings)
                    syn_embedding = syn_embeddings.mean(dim=0)
                    syn_embedding = F.normalize(syn_embedding, dim=0)
                    syn_features.append(syn_embedding)

            if syn_features:
                syn_features = torch.stack(syn_features, dim=0)
                cls_embedding = syn_features.mean(dim=0)
                cls_embedding = F.normalize(cls_embedding, dim=0)
                self._per_class_template_embeds.append(torch.cat(all_template_embeds, dim=0))
            else:
                cls_embedding = torch.zeros(embedding_dim)
                self._per_class_template_embeds.append(torch.empty(0, embedding_dim))

            all_text_features.append(cls_embedding)

            if (cls_idx + 1) % clear_cache_every == 0 and self.device.type == "cuda":
                torch.cuda.empty_cache()

        text_features = torch.stack(all_text_features, dim=0)
        text_features = F.normalize(text_features, dim=-1)

        return text_features.to(self.device)

    @torch.no_grad()
    def compute_covariance(self, class_means: torch.Tensor, lambda_reg: float = 0.1) -> torch.Tensor:
        """
        Compute regularized inverse covariance from per-template embeddings
        centered on SAEG class means. Uses synonyms and templates consistently.

        Args:
            class_means: (K, D) SAEG class mean features (L2-normalized)
            lambda_reg: Tikhonov regularization parameter

        Returns:
            cov_inv: (D, D) regularized inverse covariance matrix
        """
        all_centered = []
        for cls_idx, embeds in enumerate(self._per_class_template_embeds):
            if embeds.shape[0] == 0:
                continue
            cls_mean = class_means[cls_idx].cpu()
            centered = embeds - cls_mean.unsqueeze(0)
            all_centered.append(centered)

        if len(all_centered) == 0:
            D = class_means.shape[1]
            return torch.eye(D, device=class_means.device)

        all_centered = torch.cat(all_centered, dim=0)
        N = all_centered.shape[0]
        D = all_centered.shape[1]

        cov_emp = (all_centered.T @ all_centered) / (N - 1)
        cov_shrink = cov_emp + lambda_reg * torch.eye(D, dtype=cov_emp.dtype)
        cov_inv = torch.linalg.inv(cov_shrink.float())

        print(f"[SAEG] Covariance from {N} synonym-template embeddings (λ={lambda_reg}), Det: {torch.det(cov_shrink):.2e}")
        return cov_inv.to(device=class_means.device, dtype=class_means.dtype)


from config_utils import resolve_path

def load_class_to_synonyms(path: str = "lvis_original_class_to_synonyms.pkl") -> Dict[str, List[str]]:
    """Load the LVIS class-to-synonyms mapping from pickle file."""
    resolved_path = resolve_path(path)
    with open(resolved_path, "rb") as f:
        return pickle.load(f)
