"""Automated unit and integration tests for the Retail-360 CLI engine."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from retail360.cli import main
from retail360.data_store import (
    build_rfm_feature_store,
    connect_db,
    seed_synthetic_olist_data,
)
from retail360.fairness import audit_regional_fairness
from retail360.ltv_model import train_and_evaluate_ltv
from retail360.segmentation import fit_customer_segments


class TestRetail360Pipeline(unittest.TestCase):
    def test_sql_rfm_feature_engineering(self) -> None:
        conn = connect_db(":memory:")
        counts = seed_synthetic_olist_data(conn, n_orders=600, n_unique_customers=150, seed=7)
        self.assertEqual(counts["orders"], 600)

        rfm_df = build_rfm_feature_store(conn)
        self.assertGreater(len(rfm_df), 50)
        self.assertTrue(
            {"customer_unique_id", "region", "recency_days", "frequency", "monetary_value"}.issubset(
                rfm_df.columns
            )
        )
        self.assertTrue((rfm_df["monetary_value"] > 0).all())
        self.assertTrue((rfm_df["recency_days"] >= 0).all())

    def test_segmentation_and_ltv_training(self) -> None:
        conn = connect_db(":memory:")
        seed_synthetic_olist_data(conn, n_orders=800, n_unique_customers=200, seed=11)
        rfm_df = build_rfm_feature_store(conn)

        segmented_df, summary = fit_customer_segments(rfm_df, n_clusters=4, seed=11)
        self.assertEqual(len(summary), 4)
        self.assertEqual(int(summary["customer_count"].sum()), len(segmented_df))

        ltv_res = train_and_evaluate_ltv(segmented_df, epochs=25, seed=11)
        self.assertGreater(ltv_res.mae, 0.0)
        self.assertEqual(len(ltv_res.y_true), len(ltv_res.y_pred))

    def test_fairness_disparity_flagging(self) -> None:
        y_true = np.array([100.0, 100.0, 100.0, 100.0])
        y_pred = np.array([101.0, 101.0, 120.0, 120.0])
        regions = np.array(["North", "North", "South", "South"])

        report = audit_regional_fairness(y_true, y_pred, regions, max_disparity_threshold=5.0)
        self.assertFalse(report.passed)
        self.assertAlmostEqual(report.max_disparity, 19.0, places=2)

    def test_cli_end_to_end_json_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "report.json"
            exit_code = main(
                [
                    "run-pipeline",
                    "--orders",
                    "500",
                    "--customers",
                    "120",
                    "--epochs",
                    "20",
                    "--output-json",
                    str(out_file),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.exists())
            payload = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertIn("ltv_metrics", payload)
            self.assertIn("fairness_audit", payload)


if __name__ == "__main__":
    unittest.main()
