from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def build_clean_customers(
    spark: SparkSession,
    row_count: int = 10_000,
    seed: int = 42,
) -> DataFrame:
    """
    Generate a deterministic, clean customer dataset.
    """

    rows = []

    countries = ["IN", "US", "GB", "DE", "AU", "SG"]

    for customer_id in range(1, row_count + 1):
        country = countries[(customer_id - 1) % len(countries)]

        rows.append(
            (
                customer_id,
                f"customer{customer_id}@example.com",
                country,
                datetime(2024, 1, 1) + timedelta(days=customer_id % 730),
            )
        )

    schema = StructType(
        [
            StructField("customer_id", LongType(), nullable=False),
            StructField("email", StringType(), nullable=False),
            StructField("country", StringType(), nullable=False),
            StructField("created_at", TimestampType(), nullable=False),
        ]
    )

    return spark.createDataFrame(rows, schema)


def build_clean_orders(
    spark: SparkSession,
    row_count: int = 25_000,
    customer_count: int = 10_000,
    seed: int = 42,
) -> DataFrame:
    """
    Generate clean orders referencing existing customers.
    """

    rows = []

    statuses = ["PLACED", "SHIPPED", "DELIVERED", "CANCELLED"]

    start_date = date(2024, 1, 1)

    for order_id in range(1, row_count + 1):
        customer_id = ((order_id - 1) % customer_count) + 1

        order_date = start_date + timedelta(days=order_id % 730)

        status = statuses[(order_id - 1) % len(statuses)]

        if status in {"SHIPPED", "DELIVERED"}:
            shipped_date = datetime.combine(
                order_date + timedelta(days=(order_id % 7) + 1),
                datetime.min.time(),
            )
        else:
            shipped_date = None

        rows.append(
            (
                order_id,
                customer_id,
                datetime.combine(order_date, datetime.min.time()),
                shipped_date,
                status,
                datetime.combine(
                    order_date,
                    datetime.min.time(),
                )
                + timedelta(hours=12),
            )
        )

    schema = StructType(
        [
            StructField("order_id", LongType(), nullable=False),
            StructField("customer_id", LongType(), nullable=False),
            StructField("order_date", TimestampType(), nullable=False),
            StructField("shipped_date", TimestampType(), nullable=True),
            StructField("status", StringType(), nullable=False),
            StructField("load_ts", TimestampType(), nullable=False),
        ]
    )

    return spark.createDataFrame(rows, schema)


def build_clean_order_items(
    spark: SparkSession,
    row_count: int = 60_000,
    order_count: int = 25_000,
    seed: int = 42,
) -> DataFrame:
    """
    Generate order items referencing existing orders.
    """

    rows = []

    for item_id in range(1, row_count + 1):
        order_id = ((item_id - 1) % order_count) + 1
        product_id = ((item_id - 1) % 500) + 1

        quantity = (item_id % 5) + 1
        unit_price = round(10.0 + ((item_id * 13) % 9900) / 100.0, 2)

        rows.append(
            (
                order_id,
                product_id,
                quantity,
                unit_price,
            )
        )

    schema = StructType(
        [
            StructField("order_id", LongType(), nullable=False),
            StructField("product_id", LongType(), nullable=False),
            StructField("quantity", IntegerType(), nullable=False),
            StructField("unit_price", DoubleType(), nullable=False),
        ]
    )

    return spark.createDataFrame(rows, schema)


def build_clean_payments(
    spark: SparkSession,
    row_count: int = 25_000,
    order_count: int = 25_000,
    seed: int = 42,
) -> DataFrame:
    """
    Generate clean payments referencing existing orders.
    """

    rows = []

    methods = ["CARD", "UPI", "NET_BANKING", "WALLET"]

    for payment_id in range(1, row_count + 1):
        order_id = ((payment_id - 1) % order_count) + 1

        amount = round(
            50.0 + ((payment_id * 17) % 9950) / 100.0,
            2,
        )

        method = methods[(payment_id - 1) % len(methods)]

        paid_at = datetime(2024, 1, 1) + timedelta(
            days=payment_id % 730,
            hours=14,
        )

        rows.append(
            (
                payment_id,
                order_id,
                amount,
                method,
                paid_at,
            )
        )

    schema = StructType(
        [
            StructField("payment_id", LongType(), nullable=False),
            StructField("order_id", LongType(), nullable=False),
            StructField("amount", DoubleType(), nullable=False),
            StructField("method", StringType(), nullable=False),
            StructField("paid_at", TimestampType(), nullable=False),
        ]
    )

    return spark.createDataFrame(rows, schema)


def generate_clean_dataset(
    spark: SparkSession,
    row_counts: Dict[str, int],
    seed: int = 42,
) -> Dict[str, DataFrame]:
    """
    Generate all clean tables.
    """

    customers = build_clean_customers(
        spark=spark,
        row_count=row_counts["customers"],
        seed=seed,
    )

    orders = build_clean_orders(
        spark=spark,
        row_count=row_counts["orders"],
        customer_count=row_counts["customers"],
        seed=seed,
    )

    order_items = build_clean_order_items(
        spark=spark,
        row_count=row_counts["order_items"],
        order_count=row_counts["orders"],
        seed=seed,
    )

    payments = build_clean_payments(
        spark=spark,
        row_count=row_counts["payments"],
        order_count=row_counts["orders"],
        seed=seed,
    )

    return {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
    }


def write_delta_tables(
    datasets: Dict[str, DataFrame],
    catalog: str,
    schema: str,
    mode: str = "overwrite",
) -> None:
    """
    Write DataFrames to Delta tables.
    """

    for table_name, dataframe in datasets.items():
        table_identifier = f"{catalog}.{schema}.{table_name}"

        (
            dataframe.write
            .format("delta")
            .mode(mode)
            .option("overwriteSchema", "true")
            .saveAsTable(table_identifier)
        )

        print(f"Written: {table_identifier}")