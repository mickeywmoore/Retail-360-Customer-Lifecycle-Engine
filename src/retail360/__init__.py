"""Retail-360 Customer Lifecycle & Neural LTV Forecasting CLI Engine."""

from retail360.data_store import build_rfm_feature_store, connect_db, seed_synthetic_olist_data
from retail360.fairness import FairnessAuditReport, audit_regional_fairness
from retail360.ltv_model import LTVTrainingResult, train_and_evaluate_ltv
from retail360.segmentation import fit_customer_segments

__all__ = [
    "FairnessAuditReport",
    "LTVTrainingResult",
    "audit_regional_fairness",
    "build_rfm_feature_store",
    "connect_db",
    "fit_customer_segments",
    "seed_synthetic_olist_data",
    "train_and_evaluate_ltv",
]
__version__ = "2.0.0"
