"""Neural Network Customer Lifetime Value (LTV) forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["recency_days", "frequency", "monetary_value"]


@dataclass
class LTVTrainingResult:
    """Container for trained LTV model artifacts and holdout evaluation metrics."""

    backend: str
    mae: float
    rmse: float
    test_frame: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray


def synthesize_90d_ltv_targets(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate 90-day forward spend labels incorporating recency decay penalties."""
    rng = np.random.default_rng(seed)
    enriched = df.copy()
    noise = rng.normal(loc=0.0, scale=15.0, size=len(enriched))
    raw_target = (enriched["monetary_value"] * 0.20) + (enriched["frequency"] * 10.0) + noise
    decay_factor = np.where(enriched["recency_days"] > 100, 0.50, 1.0)
    enriched["future_spend_90d"] = np.clip(raw_target * decay_factor, a_min=0.0, a_max=None)
    return enriched


def train_and_evaluate_ltv(
    df: pd.DataFrame,
    epochs: int = 50,
    test_size: float = 0.20,
    seed: int = 42,
    prefer_tensorflow: bool = False,
) -> LTVTrainingResult:
    """Train a multi-layer neural regressor (64-32-16) without train/test leakage."""
    labeled_df = (
        df if "future_spend_90d" in df.columns else synthesize_90d_ltv_targets(df, seed=seed)
    )

    train_df, test_df = train_test_split(labeled_df, test_size=test_size, random_state=seed)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    x_test = scaler.transform(test_df[FEATURE_COLUMNS])
    y_train = train_df["future_spend_90d"].to_numpy(dtype=float)
    y_test = test_df["future_spend_90d"].to_numpy(dtype=float)

    backend_used = "scikit-learn-mlp"
    y_pred: np.ndarray

    if prefer_tensorflow:
        try:
            import tensorflow as tf  # type: ignore

            tf.random.set_seed(seed)
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(x_train.shape[1],)),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(1, activation="linear"),
                ]
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="mse",
                metrics=["mae"],
            )
            model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=0,
            )
            y_pred = model.predict(x_test, verbose=0).flatten()
            backend_used = "tensorflow-keras"
        except ImportError:
            prefer_tensorflow = False

    if not prefer_tensorflow:
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=max(epochs * 4, 200),
            random_state=seed,
        )
        mlp.fit(x_train, y_train)
        y_pred = mlp.predict(x_test)

    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    return LTVTrainingResult(
        backend=backend_used,
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        test_frame=test_df.reset_index(drop=True),
        y_true=y_test,
        y_pred=y_pred,
    )
