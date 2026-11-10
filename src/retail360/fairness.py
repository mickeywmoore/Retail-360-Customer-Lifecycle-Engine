"""Regional Mean Absolute Error (MAE) disparity auditing to prevent geographic bias."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


@dataclass
class FairnessAuditReport:
    """Structured fairness evaluation across regional cohorts."""

    overall_mae: float
    group_mae: dict[str, float]
    max_disparity: float
    threshold: float
    passed: bool


def audit_regional_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regions: pd.Series | np.ndarray,
    max_disparity_threshold: float = 5.0,
) -> FairnessAuditReport:
    """Compute subgroup MAE across geographic regions and flag threshold violations."""
    eval_df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
            "region": np.asarray(regions, dtype=str),
        }
    )
    if eval_df.empty:
        raise ValueError("Cannot run fairness audit on an empty evaluation set.")

    overall_mae = float(mean_absolute_error(eval_df["y_true"], eval_df["y_pred"]))

    group_mae: dict[str, float] = {}
    for region_name, group in eval_df.groupby("region"):
        group_mae[str(region_name)] = round(
            float(mean_absolute_error(group["y_true"], group["y_pred"])), 2
        )

    disparity = (
        max(group_mae.values()) - min(group_mae.values())
        if len(group_mae) > 1
        else 0.0
    )
    disparity_rounded = round(float(disparity), 2)

    return FairnessAuditReport(
        overall_mae=round(overall_mae, 2),
        group_mae=dict(sorted(group_mae.items())),
        max_disparity=disparity_rounded,
        threshold=max_disparity_threshold,
        passed=disparity_rounded <= max_disparity_threshold,
    )
