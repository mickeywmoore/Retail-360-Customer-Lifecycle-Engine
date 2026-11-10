"""SQLite Olist schema generator and SQL CTE RFM feature engineering store."""

from __future__ import annotations

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd


RFM_SQL_QUERY = """
WITH order_payments AS (
    SELECT
        order_id,
        SUM(payment_value) AS total_order_value
    FROM olist_order_payments_dataset
    GROUP BY order_id
),
customer_orders AS (
    SELECT
        c.customer_unique_id,
        c.region,
        o.order_id,
        o.order_purchase_timestamp,
        COALESCE(p.total_order_value, 0.0) AS total_order_value,
        r.review_score
    FROM olist_orders_dataset o
    INNER JOIN olist_customers_dataset c
        ON o.customer_id = c.customer_id
    LEFT JOIN order_payments p
        ON o.order_id = p.order_id
    LEFT JOIN olist_order_reviews_dataset r
        ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
),
rfm_calculation AS (
    SELECT
        customer_unique_id,
        MAX(region) AS region,
        COUNT(DISTINCT order_id) AS frequency,
        SUM(total_order_value) AS monetary_value,
        CAST(
            julianday(:reference_date) - julianday(MAX(order_purchase_timestamp))
            AS INTEGER
        ) AS recency_days,
        AVG(review_score) AS avg_review_score
    FROM customer_orders
    GROUP BY customer_unique_id
)
SELECT
    customer_unique_id,
    region,
    recency_days,
    frequency,
    monetary_value,
    ROUND(avg_review_score, 3) AS avg_review_score
FROM rfm_calculation
WHERE monetary_value > 0
ORDER BY monetary_value DESC;
"""


def connect_db(db_path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open a SQLite connection for either an in-memory or file-backed store."""
    return sqlite3.connect(str(db_path))


def seed_synthetic_olist_data(
    conn: sqlite3.Connection,
    n_orders: int = 10_000,
    n_unique_customers: int = 2_000,
    seed: int = 42,
) -> dict[str, int]:
    """Populate relational Olist-style tables with deterministic synthetic records."""
    if n_orders <= 0 or n_unique_customers <= 0:
        raise ValueError("n_orders and n_unique_customers must be positive integers.")

    rng = np.random.default_rng(seed)
    order_ids = [f"order_{i}" for i in range(n_orders)]
    temp_customer_ids = [f"temp_cust_{i}" for i in range(n_orders)]

    regions_pool = np.array(["North", "South", "East", "West"])
    user_regions = {
        f"user_{uid}": str(rng.choice(regions_pool))
        for uid in range(n_unique_customers)
    }
    assigned_users = [
        f"user_{int(uid)}" for uid in rng.integers(0, n_unique_customers, size=n_orders)
    ]

    orders_df = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": temp_customer_ids,
            "order_status": rng.choice(
                ["delivered", "shipped", "canceled"],
                size=n_orders,
                p=[0.94, 0.04, 0.02],
            ),
            "order_purchase_timestamp": pd.date_range(
                start="2025-01-01", periods=n_orders, freq="45min"
            ).astype(str),
        }
    )

    customers_df = pd.DataFrame(
        {
            "customer_id": temp_customer_ids,
            "customer_unique_id": assigned_users,
            "region": [user_regions[u] for u in assigned_users],
        }
    )

    payments_df = pd.DataFrame(
        {
            "order_id": order_ids,
            "payment_value": np.round(rng.exponential(scale=150.0, size=n_orders), 2),
        }
    )

    reviews_df = pd.DataFrame(
        {
            "order_id": order_ids,
            "review_score": rng.choice(
                [1, 2, 3, 4, 5], size=n_orders, p=[0.10, 0.05, 0.10, 0.25, 0.50]
            ),
        }
    )

    orders_df.to_sql("olist_orders_dataset", conn, index=False, if_exists="replace")
    customers_df.to_sql("olist_customers_dataset", conn, index=False, if_exists="replace")
    payments_df.to_sql("olist_order_payments_dataset", conn, index=False, if_exists="replace")
    reviews_df.to_sql("olist_order_reviews_dataset", conn, index=False, if_exists="replace")

    return {
        "orders": len(orders_df),
        "customers": n_unique_customers,
        "payments": len(payments_df),
        "reviews": len(reviews_df),
    }


def build_rfm_feature_store(
    conn: sqlite3.Connection,
    reference_date: str = "2026-01-05",
) -> pd.DataFrame:
    """Execute the multi-CTE SQL transformation to produce customer-level RFM features."""
    df = pd.read_sql_query(
        RFM_SQL_QUERY,
        conn,
        params={"reference_date": reference_date},
    )
    df["recency_days"] = df["recency_days"].clip(lower=0)
    return df
