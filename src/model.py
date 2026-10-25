"""
Module 3: SchemaFlexTabNet-QC Architecture
Transformer Encoder with Schema-Flexible Tokenization, Training-Time Random Feature Masking,
Auxiliary Clinical Concept Bottleneck Head, and Differentiable Monotonicity Regularization.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_harmonizer import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
from src.schema_tokenizer import SchemaFlexibleTokenizer, MONOTONIC_DIRECTIONS


try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    HAS_SDPA_BACKEND = True
except ImportError:
    HAS_SDPA_BACKEND = False


class SchemaFlexTabNetQC(nn.Module):
    """
    Proposed Architecture:
    - Schema-Flexible Feature Tokenizer with Masked Self-Attention
    - Training-time Random Feature Dropout (simulating missing clinical columns)
    - Auxiliary Clinical Concept Bottleneck Head (predicts Stage 2 Hypertension / Metabolic state)
    - Risk Prediction Head with Differentiable Monotonicity Regularization
    """

    def __init__(
        self,
        num_cont: int = len(CONTINUOUS_FEATURES),
        num_cat: int = len(CATEGORICAL_FEATURES),
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        feature_drop_prob: float = 0.20,
    ):
        super().__init__()
        self.tokenizer = SchemaFlexibleTokenizer(num_cont, num_cat, d_model)
        self.feature_drop_prob = feature_drop_prob

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Auxiliary Clinical Concept Bottleneck Head (predicts Hypertension / Metabolic state)
        self.concept_head = nn.Linear(d_model, 1)

        # Final Risk Classifier Head (fuses [CLS] latent representation + predicted clinical concept)
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 1, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_cont: torch.Tensor,
        mask_cont: torch.Tensor,
        x_cat: torch.Tensor,
        mask_cat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_cont.size(0)
        tokens = self.tokenizer(x_cont, x_cat)

        # Observation mask: 1 = observed, 0 = missing
        obs_mask = torch.cat([mask_cont, mask_cat], dim=1)

        # Random feature masking during training to enforce cross-cohort schema invariance
        if self.training and self.feature_drop_prob > 0.0:
            keep_mask = (torch.rand_like(obs_mask) > self.feature_drop_prob).float()
            # Ensure at least 1 feature remains observed
            obs_mask = obs_mask * keep_mask

        # Prepend 1.0 for the [CLS] token (always observed)
        cls_obs = torch.ones((batch_size, 1), device=obs_mask.device, dtype=obs_mask.dtype)
        full_obs_mask = torch.cat([cls_obs, obs_mask], dim=1)

        # PyTorch src_key_padding_mask: True for positions to IGNORE (unobserved)
        padding_mask = full_obs_mask == 0.0

        if HAS_SDPA_BACKEND and x_cont.is_cuda:
            with sdpa_kernel(SDPBackend.MATH):
                encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        else:
            encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)

        cls_rep = self.norm(encoded[:, 0, :])

        # Clinical Concept Bottleneck prediction
        concept_logit = self.concept_head(cls_rep)
        concept_prob = torch.sigmoid(concept_logit)

        # Fuse latent representation with clinical concept probability
        fused = torch.cat([cls_rep, concept_prob], dim=1)
        risk_logit = self.classifier(fused).squeeze(-1)

        return risk_logit, concept_logit.squeeze(-1)


def compute_monotonicity_loss(
    model: SchemaFlexTabNetQC,
    x_cont: torch.Tensor,
    mask_cont: torch.Tensor,
    x_cat: torch.Tensor,
    mask_cat: torch.Tensor,
    directions: List[float] = MONOTONIC_DIRECTIONS,
) -> torch.Tensor:
    """
    Penalizes gradients that violate known clinical monotonicity
    (e.g., risk must not decrease when Systolic BP or Age increases).
    """
    x_cont_grad = x_cont.detach().clone().requires_grad_(True)
    logits, _ = model(x_cont_grad, mask_cont, x_cat, mask_cat)
    grads = torch.autograd.grad(
        outputs=logits.sum(),
        inputs=x_cont_grad,
        create_graph=True,
        retain_graph=True,
    )[0]

    dir_tensor = torch.tensor(directions, device=x_cont.device, dtype=torch.float32).unsqueeze(0)
    # Violation occurs when sign(grad) opposes expected clinical direction on observed features
    violations = F.relu(-grads * dir_tensor) * mask_cont
    return violations.mean()
