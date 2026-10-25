"""
Module 4: Post-Hoc Temperature Calibration & Split Conformal Abstention
Provides finite-sample coverage guarantees and patient-level abstention flags
for ambiguous or out-of-distribution clinical presentations.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """
    Post-hoc Temperature Scaling to minimize Expected Calibration Error (ECE)
    and provide well-calibrated probabilities for clinical risk prediction.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def fit(self, logits: np.ndarray, y_true: np.ndarray, max_iter: int = 100) -> float:
        logits_t = torch.tensor(logits, dtype=torch.float32)
        y_t = torch.tensor(y_true, dtype=torch.float32)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.05, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            t = torch.clamp(self.temperature, min=0.05, max=10.0)
            scaled_logits = logits_t / t
            loss = F.binary_cross_entropy_with_logits(scaled_logits, y_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits_t = torch.tensor(logits, dtype=torch.float32)
            t = torch.clamp(self.temperature, min=0.05, max=10.0)
            probs = torch.sigmoid(logits_t / t).numpy()
        return probs


class SplitConformalAbstainer:
    """
    Constructs finite-sample valid prediction sets C(X_i) subset of {0, 1}
    with marginal coverage guarantee P(Y in C(X)) >= 1 - alpha.
    Whenever |C(X_i)| != 1 (either both classes plausible or empty),
    the model abstains and flags the patient for clinician review.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_hat: Optional[float] = None

    def calibrate(self, probs_cal: np.ndarray, y_cal: np.ndarray) -> float:
        """Computes the conformal non-conformity quantile threshold on calibration data."""
        n = len(y_cal)
        # Non-conformity score: 1 - probability assigned to the true class
        scores = np.where(y_cal == 1, 1.0 - probs_cal, probs_cal)
        q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        q_level = np.clip(q_level, 0.0, 1.0)
        self.q_hat = float(np.quantile(scores, q_level, method="higher"))
        return self.q_hat

    def predict_with_abstention(self, probs_test: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Produces prediction sets and binary abstention flags for test patients.
        """
        if self.q_hat is None:
            raise ValueError("SplitConformalAbstainer must be calibrated before predicting!")

        include_pos = (1.0 - probs_test) <= self.q_hat
        include_neg = probs_test <= self.q_hat

        set_size = include_pos.astype(int) + include_neg.astype(int)
        # Abstain when ambiguous ({0, 1}) or empty set ({})
        abstain_mask = set_size != 1
        confident_mask = ~abstain_mask

        # Point prediction for confident samples: 1 if include_pos else 0
        point_preds = np.where(include_pos & ~include_neg, 1, 0)

        return {
            "probs": probs_test,
            "include_pos": include_pos,
            "include_neg": include_neg,
            "set_size": set_size,
            "abstain_mask": abstain_mask,
            "confident_mask": confident_mask,
            "point_preds": point_preds,
            "coverage": float(np.mean(confident_mask)),
        }
