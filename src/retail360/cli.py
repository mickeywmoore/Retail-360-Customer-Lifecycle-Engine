"""Command-line interface for running the Retail-360 Customer Lifecycle Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retail360.data_store import (
    build_rfm_feature_store,
    connect_db,
    seed_synthetic_olist_data,
)
from retail360.fairness import audit_regional_fairness
from retail360.ltv_model import train_and_evaluate_ltv
from retail360.segmentation import fit_customer_segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retail360",
        description="End-to-end SQL RFM feature engineering, K-Means segmentation, LTV forecasting, and fairness audit CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_cmd = subparsers.add_parser(
        "run-pipeline",
        help="Execute the full SQL -> K-Means -> Neural LTV -> Fairness Audit pipeline.",
    )
    run_cmd.add_argument(
        "--orders", type=int, default=10_000, help="Synthetic orders to seed (default: 10000)."
    )
    run_cmd.add_argument(
        "--customers", type=int, default=2_000, help="Unique customers to simulate (default: 2000)."
    )
    run_cmd.add_argument(
        "--clusters", type=int, default=4, help="Number of K-Means persona clusters (default: 4)."
    )
    run_cmd.add_argument(
        "--epochs", type=int, default=50, help="Neural regressor training epochs (default: 50)."
    )
    run_cmd.add_argument(
        "--disparity-threshold",
        type=float,
        default=5.0,
        help="Maximum acceptable regional MAE disparity in USD (default: 5.0).",
    )
    run_cmd.add_argument(
        "--db-path",
        type=str,
        default=":memory:",
        help="SQLite database path or ':memory:' (default: :memory:).",
    )
    run_cmd.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to export structured pipeline metrics and audit report as JSON.",
    )
    run_cmd.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic reproducibility."
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> int:
    conn = connect_db(args.db_path)
    counts = seed_synthetic_olist_data(
        conn,
        n_orders=args.orders,
        n_unique_customers=args.customers,
        seed=args.seed,
    )
    rfm_df = build_rfm_feature_store(conn)
    segmented_df, cluster_summary = fit_customer_segments(
        rfm_df, n_clusters=args.clusters, seed=args.seed
    )
    ltv_result = train_and_evaluate_ltv(
        segmented_df, epochs=args.epochs, seed=args.seed
    )
    audit = audit_regional_fairness(
        y_true=ltv_result.y_true,
        y_pred=ltv_result.y_pred,
        regions=ltv_result.test_frame["region"],
        max_disparity_threshold=args.disparity_threshold,
    )

    print("=== Retail-360 Lifecycle Pipeline Summary ===")
    print(
        f"Ingested SQL Tables : {counts['orders']} orders | {len(rfm_df)} active customer profiles"
    )
    print("\n--- Customer Persona Segments (K-Means) ---")
    print(cluster_summary.to_string(index=False))

    print(f"\n--- 90-Day Neural LTV Forecast ({ltv_result.backend}) ---")
    print(f"Holdout MAE         : ${ltv_result.mae:.2f}")
    print(f"Holdout RMSE        : ${ltv_result.rmse:.2f}")

    status_label = "PASS" if audit.passed else "ALERT - REVIEW REQUIRED"
    print(f"\n--- Regional Fairness Audit [{status_label}] ---")
    print(f"Overall Test MAE    : ${audit.overall_mae:.2f}")
    for region, mae_val in audit.group_mae.items():
        print(f"  Region {region:<10} : ${mae_val:.2f}")
    print(
        f"Max Regional Gap    : ${audit.max_disparity:.2f} (Threshold: ${audit.threshold:.2f})"
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ingestion_counts": counts,
            "active_profiles": len(rfm_df),
            "clusters": cluster_summary.to_dict(orient="records"),
            "ltv_metrics": {
                "backend": ltv_result.backend,
                "mae": ltv_result.mae,
                "rmse": ltv_result.rmse,
            },
            "fairness_audit": {
                "overall_mae": audit.overall_mae,
                "group_mae": audit.group_mae,
                "max_disparity": audit.max_disparity,
                "threshold": audit.threshold,
                "passed": audit.passed,
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report -> {out_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-pipeline":
        return run_pipeline(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
