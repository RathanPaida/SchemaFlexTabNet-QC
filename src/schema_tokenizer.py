"""
Module 2: Schema-Flexible Dataset & Tokenizer (No Inner-Join Required)
Converts heterogeneous patient records with missing or non-overlapping columns into
unified token representations with attention masks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.data_harmonizer import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES

MONOTONIC_DIRECTIONS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0]


class MultiCohortTabularDataset(Dataset):
    """
    Produces variable-availability feature tensors with binary observation masks.
    Missing columns in a cohort are marked with mask=0 rather than dropped via inner-join.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cont_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

        # Compute training fold mean & std solely on observed (non-NaN) values
        if cont_stats is None:
            self.cont_stats = {}
            for col in CONTINUOUS_FEATURES:
                if col in self.df.columns and self.df[col].notna().any():
                    mu = float(self.df[col].dropna().mean())
                    sigma = float(self.df[col].dropna().std())
                    if sigma < 1e-6:
                        sigma = 1.0
                    self.cont_stats[col] = (mu, sigma)
                else:
                    self.cont_stats[col] = (0.0, 1.0)
        else:
            self.cont_stats = cont_stats

        n = len(self.df)
        self.x_cont = np.zeros((n, len(CONTINUOUS_FEATURES)), dtype=np.float32)
        self.mask_cont = np.zeros((n, len(CONTINUOUS_FEATURES)), dtype=np.float32)

        for j, col in enumerate(CONTINUOUS_FEATURES):
            if col in self.df.columns:
                obs = self.df[col].notna().values
                self.mask_cont[:, j] = obs.astype(np.float32)
                mu, sigma = self.cont_stats[col]
                vals = self.df[col].fillna(mu).values.astype(np.float32)
                self.x_cont[:, j] = (vals - mu) / sigma

        self.x_cat = np.zeros((n, len(CATEGORICAL_FEATURES)), dtype=np.int64)
        self.mask_cat = np.zeros((n, len(CATEGORICAL_FEATURES)), dtype=np.float32)

        for j, col in enumerate(CATEGORICAL_FEATURES):
            if col in self.df.columns:
                obs = self.df[col].notna().values
                self.mask_cat[:, j] = obs.astype(np.float32)
                vals = self.df[col].fillna(0).values.astype(np.int64)
                self.x_cat[:, j] = np.clip(vals, 0, 9)

        self.y = self.df["target"].values.astype(np.float32)
        self.concept_htn = (
            self.df["concept_hypertension"].values.astype(np.float32)
            if "concept_hypertension" in self.df.columns
            else np.zeros(n, dtype=np.float32)
        )

        # Store cohort metadata if present
        self.cohorts = self.df["cohort"].values if "cohort" in self.df.columns else np.array(["Unknown"] * n)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_cont": torch.tensor(self.x_cont[idx], dtype=torch.float32),
            "mask_cont": torch.tensor(self.mask_cont[idx], dtype=torch.float32),
            "x_cat": torch.tensor(self.x_cat[idx], dtype=torch.long),
            "mask_cat": torch.tensor(self.mask_cat[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "concept_htn": torch.tensor(self.concept_htn[idx], dtype=torch.float32),
        }


class SchemaFlexibleTokenizer(nn.Module):
    """
    Tokenizes each feature j as:
        Token_{i,j} = FeatureNameEmbedding(j) + ValueProjection(x_{i,j})
    Prepends a learned [CLS] token for patient-level aggregation.
    """

    def __init__(
        self,
        num_cont: int = len(CONTINUOUS_FEATURES),
        num_cat: int = len(CATEGORICAL_FEATURES),
        d_model: int = 64,
    ):
        super().__init__()
        self.num_cont = num_cont
        self.num_cat = num_cat
        self.total_features = num_cont + num_cat

        # Learned semantic identity embedding for every canonical feature name
        self.feature_name_embeddings = nn.Parameter(
            torch.randn(1, self.total_features, d_model) * 0.02
        )

        # Continuous value projections (feature-specific linear layer)
        self.cont_weight = nn.Parameter(torch.randn(1, num_cont, d_model) * 0.02)
        self.cont_bias = nn.Parameter(torch.zeros(1, num_cont, d_model))

        # Categorical lookup tables (up to 10 classes per category)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(10, d_model) for _ in range(num_cat)]
        )

        # Learnable [CLS] summary token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(
        self, x_cont: torch.Tensor, x_cat: torch.Tensor
    ) -> torch.Tensor:
        batch_size = x_cont.size(0)

        # Continuous tokens: (B, num_cont, 1) * (1, num_cont, d_model) + bias
        cont_tokens = x_cont.unsqueeze(-1) * self.cont_weight + self.cont_bias

        # Categorical tokens: list of (B, 1, d_model) -> (B, num_cat, d_model)
        cat_token_list = [
            emb(x_cat[:, j]).unsqueeze(1) for j, emb in enumerate(self.cat_embeddings)
        ]
        cat_tokens = torch.cat(cat_token_list, dim=1)

        # Combine value tokens with feature name identity embeddings
        value_tokens = torch.cat([cont_tokens, cat_tokens], dim=1)
        tokens = value_tokens + self.feature_name_embeddings

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        all_tokens = torch.cat([cls_tokens, tokens], dim=1)
        return all_tokens
