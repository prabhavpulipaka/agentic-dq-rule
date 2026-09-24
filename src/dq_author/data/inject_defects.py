from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def _select_ids(
    dataframe: DataFrame,
    id_column: str,
    count: int,
    offset: int = 0,
) -> List[int]:
    """
    Select deterministic IDs from a dataset.

    The selected IDs are taken from distinct, non-null values
    and ordered deterministically.
    """

    values = (
        dataframe
        .select(id_column)
        .where(F.col(id_column).isNotNull())
        .distinct()
        .orderBy(F.col(id_column))
        .collect()
    )

    ids = [row[id_column] for row in values]

    if not ids:
        raise ValueError(f"No IDs found in column: {id_column}")

    if count > len(ids):
        raise ValueError(
            f"Requested {count} IDs from {len(ids)} available IDs "
            f"in column {id_column}"
        )

    return ids[offset:offset + count]


def _add_manifest_entry(
    manifest: List[Tuple],
    defect_id: str,
    table_name: str,
    column_name: str,
    defect_type: str,
    affected_row_count: int,
    injection_run_id: str,
    injected_at: datetime,
) -> None:
    manifest.append(
        (
            defect_id,
            table_name,
            column_name,
            defect_type,
            affected_row_count,
            injection_run_id,
            injected_at,
        )
    )


def inject_defects(
    spark: SparkSession,
    clean_tables: Dict[str, DataFrame],
    injection_run_id: str = "run_001",
    seed: int = 42,
) -> Tuple[Dict[str, DataFrame], List[Tuple]]:
    """
    Create defective copies from clean DataFrames.

    Important design choice:
    All defect target IDs are selected from the original clean
    DataFrames before duplicate rows are introduced.

    This keeps the ground-truth manifest deterministic and
    prevents duplicate order IDs from changing the affected
    row counts of later defects.

    Returns:
        defective_tables
        manifest_rows
    """

    defective = {
        name: dataframe
        for name, dataframe in clean_tables.items()
    }

    manifest: List[Tuple] = []

    injected_at = datetime.now(timezone.utc).replace(tzinfo=None)

    # Keep references to the original clean datasets.
    clean_orders = clean_tables["orders"]
    clean_order_items = clean_tables["order_items"]
    clean_customers = clean_tables["customers"]

    # ---------------------------------------------
    # Select ALL target IDs from the clean baseline
    # ---------------------------------------------
    #
    # Each range is intentionally separated so that
    # defects do not overlap on the same order.
    #
    # seed changes the starting point while preserving
    # deterministic selection.

    seed_offset = seed % 10

    null_customer_ids = _select_ids(
        clean_orders,
        "order_id",
        count=100,
        offset=0 + seed_offset,
    )

    duplicate_order_ids = _select_ids(
        clean_orders,
        "order_id",
        count=100,
        offset=200 + seed_offset,
    )

    shipped_before_ids = _select_ids(
        clean_orders,
        "order_id",
        count=100,
        offset=400 + seed_offset,
    )

    invalid_status_ids = _select_ids(
        clean_orders,
        "order_id",
        count=100,
        offset=600 + seed_offset,
    )

    orphan_ids = _select_ids(
        clean_orders,
        "order_id",
        count=100,
        offset=800 + seed_offset,
    )

    future_order_ids = _select_ids(
        clean_orders,
        "order_id",
        count=100,
        offset=1000 + seed_offset,
    )

    negative_quantity_ids = _select_ids(
        clean_order_items,
        "order_id",
        count=100,
        offset=20 + seed_offset,
    )

    malformed_email_ids = _select_ids(
        clean_customers,
        "customer_id",
        count=100,
        offset=100 + seed_offset,
    )

    # ---------------------------------------------
    # 1. Null customer_id
    # ---------------------------------------------

    orders = clean_orders.withColumn(
        "customer_id",
        F.when(
            F.col("order_id").isin(null_customer_ids),
            F.lit(None).cast(LongType()),
        ).otherwise(F.col("customer_id")),
    )

    _add_manifest_entry(
        manifest,
        "defect_001",
        "orders",
        "customer_id",
        "null_customer_id",
        100,
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # 2. Negative quantity
    # ---------------------------------------------

    order_items = clean_order_items.withColumn(
        "quantity",
        F.when(
            F.col("order_id").isin(negative_quantity_ids),
            F.lit(-1),
        ).otherwise(F.col("quantity")),
    )

    _add_manifest_entry(
        manifest,
        "defect_002",
        "order_items",
        "quantity",
        "negative_quantity",
        100,
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # 3. Duplicate order_id
    # ---------------------------------------------
    #
    # Duplicate rows are created from the clean baseline.
    # This happens AFTER the target IDs for all other defects
    # have already been selected.
    #
    # Therefore duplicate rows cannot accidentally expand
    # the affected row count of later defects.

    duplicate_rows = clean_orders.filter(
        F.col("order_id").isin(duplicate_order_ids)
    )

    orders = orders.unionByName(duplicate_rows)

    _add_manifest_entry(
        manifest,
        "defect_003",
        "orders",
        "order_id",
        "duplicate_order_id",
        100,
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # 4. Shipped date before order date
    # ---------------------------------------------

    orders = orders.withColumn(
        "shipped_date",
        F.when(
            F.col("order_id").isin(shipped_before_ids),
            F.col("order_date") - F.expr("INTERVAL 1 DAY"),
        ).otherwise(F.col("shipped_date")),
    )

    _add_manifest_entry(
        manifest,
        "defect_004",
        "orders",
        "shipped_date,order_date",
        "shipped_before_order",
        100,
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # 5. Invalid status
    # ---------------------------------------------

    orders = orders.withColumn(
        "status",
        F.when(
            F.col("order_id").isin(invalid_status_ids),
            F.lit("INVALID_STATUS"),
        ).otherwise(F.col("status")),
    )

    _add_manifest_entry(
        manifest,
        "defect_005",
        "orders",
        "status",
        "invalid_status",
        100,
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # 6. Orphan customer_id
    # ---------------------------------------------

    orders = orders.withColumn(
        "customer_id",
        F.when(
            F.col("order_id").isin(orphan_ids),
            F.lit(999_999_999).cast(LongType()),
        ).otherwise(F.col("customer_id")),
    )

    _add_manifest_entry(
        manifest,
        "defect_006",
        "orders",
        "customer_id",
        "orphan_customer_id",
        100,
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # 7. Malformed email
    # ---------------------------------------------

    customers = clean_customers.withColumn(
        "email",
        F.when(
            F.col("customer_id").isin(malformed_email_ids),
            F.lit("invalid-email"),
        ).otherwise(F.col("email")),
    )

    _add_manifest_entry(
        manifest,
        "defect_007",
        "customers",
        "email",
        "malformed_email",
        100,
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # 8. Future order date
    # ---------------------------------------------

    orders = orders.withColumn(
        "order_date",
        F.when(
            F.col("order_id").isin(future_order_ids),
            F.lit("2099-01-01").cast(TimestampType()),
        ).otherwise(F.col("order_date")),
    )

    _add_manifest_entry(
        manifest,
        "defect_008",
        "orders",
        "order_date",
        "future_order_date",
        100,
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # Final defective datasets
    # ---------------------------------------------

    defective["orders"] = orders
    defective["order_items"] = order_items
    defective["customers"] = customers

    # Payments intentionally remain unchanged for this phase.

    return defective, manifest


def manifest_schema() -> StructType:
    return StructType(
        [
            StructField("defect_id", StringType(), nullable=False),
            StructField("table_name", StringType(), nullable=False),
            StructField("column_name", StringType(), nullable=False),
            StructField("defect_type", StringType(), nullable=False),
            StructField("affected_row_count", IntegerType(), nullable=False),
            StructField("injection_run_id", StringType(), nullable=False),
            StructField("injected_at", TimestampType(), nullable=False),
        ]
    )


def write_manifest(
    spark: SparkSession,
    manifest_rows: List[Tuple],
    catalog: str,
    schema: str,
    mode: str = "append",
) -> None:
    """
    Write the ground-truth manifest as a Delta table.
    """

    manifest_df = spark.createDataFrame(
        manifest_rows,
        schema=manifest_schema(),
    )

    (
        manifest_df.write
        .format("delta")
        .mode(mode)
        .option("mergeSchema", "true")
        .saveAsTable(
            f"{catalog}.{schema}.injected_defects"
        )
    )