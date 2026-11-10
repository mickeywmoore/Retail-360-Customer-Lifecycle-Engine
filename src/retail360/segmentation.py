"""Log-transformed K-Means customer persona segmentation and WCSS diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RFM_COLUMNS = ["recency_days", "frequency", "monetary_value"]


def preprocess_rfm_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Apply log1p skewness correction and standard scaling to RFM metrics."""
    missing = [col for col in RFM_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required RFM columns: {missing}")

    df_log = np.log1p(df[RFM_COLUMNS].clip(lower=0))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_log)
    return scaled, scaler


def compute_elbow_wcss(
    rfm_scaled: np.ndarray,
    max_k: int = 10,
    seed: int = 42,
) -> dict[int, float]:
    """Evaluate Within-Cluster Sum of Squares (WCSS) across candidate k values."""
    upper_k = min(max_k, len(rfm_scaled))
    wcss: dict[int, float] = {}
    for k in range(1, upper_k + 1):
        model = KMeans(n_clusters=k, init="k-means++", random_state=seed, n_init=10)
        model.fit(rfm_scaled)
        wcss[k] = round(float(model.inertia_), 2)
    return wcss


def fit_customer_segments(
    df: pd.DataFrame,
    n_clusters: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign K-Means cluster labels and build an aggregated persona summary table."""
    rfm_scaled, _ = preprocess_rfm_features(df)
    kmeans = KMeans(
        n_clusters=n_clusters, init="k-means++", random_state=seed, n_init=10
    )
    segmented_df = df.copy()
    segmented_df["cluster"] = kmeans.fit_predict(rfm_scaled)

    summary = (
        segmented_df.groupby("cluster", as_index=False)
        .agg(
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_spend=("monetary_value", "mean"),
            customer_count=("customer_unique_id", "count"),
        )
        .round(2)
        .sort_values("avg_spend", ascending=False)
        .reset_index(drop=True)
    )
    return segmented_df, summary
