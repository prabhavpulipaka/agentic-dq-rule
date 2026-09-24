from __future__ import annotations

from datetime import datetime
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

    This function assumes the selected ID column is numeric.
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

    selected = []

    for index in range(count):
        selected.append(ids[(offset + index) % len(ids)])

    return selected


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

    Returns:
        defective_tables
        manifest_rows
    """

    defective = {
        name: dataframe
        for name, dataframe in clean_tables.items()
    }

    manifest: List[Tuple] = []

    injected_at = datetime.utcnow()

    # ---------------------------------------------
    # 1. Null customer_id
    # ---------------------------------------------

    orders = defective["orders"]

    null_customer_ids = _select_ids(
        orders,
        "order_id",
        count=100,
        offset=seed % 10,
    )

    orders = orders.withColumn(
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
        len(null_customer_ids),
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # 2. Negative quantity
    # ---------------------------------------------

    order_items = defective["order_items"]

    negative_quantity_ids = _select_ids(
        order_items,
        "order_id",
        count=100,
        offset=20 + (seed % 10),
    )

    order_items = order_items.withColumn(
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
        len(negative_quantity_ids),
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # 3. Duplicate order_id
    # ---------------------------------------------

    duplicate_order_ids = _select_ids(
        orders,
        "order_id",
        count=100,
        offset=200,
    )

    duplicate_rows = (
        orders
        .filter(F.col("order_id").isin(duplicate_order_ids))
        .withColumn(
            "customer_id",
            F.col("customer_id"),
        )
    )

    # Duplicate rows preserve the same order_id.
    # The resulting table has an additional copy of
    # each selected order.

    orders = orders.unionByName(duplicate_rows)

    _add_manifest_entry(
        manifest,
        "defect_003",
        "orders",
        "order_id",
        "duplicate_order_id",
        len(duplicate_order_ids),
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # 4. Shipped date before order date
    # ---------------------------------------------

    shipped_before_ids = _select_ids(
        orders,
        "order_id",
        count=100,
        offset=400,
    )

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
        len(shipped_before_ids),
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # 5. Invalid status
    # ---------------------------------------------

    invalid_status_ids = _select_ids(
        orders,
        "order_id",
        count=100,
        offset=600,
    )

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
        len(invalid_status_ids),
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # 6. Orphan customer_id
    # ---------------------------------------------

    orphan_ids = _select_ids(
        orders,
        "order_id",
        count=100,
        offset=800,
    )

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
        len(orphan_ids),
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # 7. Malformed email
    # ---------------------------------------------

    customers = defective["customers"]

    malformed_email_ids = _select_ids(
        customers,
        "customer_id",
        count=100,
        offset=100,
    )

    customers = customers.withColumn(
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
        len(malformed_email_ids),
        injection_run_id,
        injected_at,
    )

    # ---------------------------------------------
    # 8. Future order date
    # ---------------------------------------------

    future_order_ids = _select_ids(
        orders,
        "order_id",
        count=100,
        offset=1000,
    )

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
        len(future_order_ids),
        injection_run_id,
        injected_at,
    )

    defective["orders"] = orders
    defective["order_items"] = order_items
    defective["customers"] = customers

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

    manifest_df.write \
        .format("delta") \
        .mode(mode) \
        .option("mergeSchema", "true") \
        .saveAsTable(
            f"{catalog}.{schema}.injected_defects"
        )