import pytest

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from dq_author.data.generate_clean import (
    generate_clean_dataset,
)

from dq_author.data.inject_defects import (
    inject_defects,
)


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("dq-author-tests")
        .getOrCreate()
    )

    yield session

    session.stop()


@pytest.fixture
def row_counts():
    return {
        "customers": 100,
        "orders": 250,
        "order_items": 500,
        "payments": 250,
    }


def test_clean_dataset_counts(spark, row_counts):
    datasets = generate_clean_dataset(
        spark=spark,
        row_counts=row_counts,
        seed=42,
    )

    assert datasets["customers"].count() == 100
    assert datasets["orders"].count() == 250
    assert datasets["order_items"].count() == 500
    assert datasets["payments"].count() == 250


def test_clean_orders_have_valid_customer_ids(
    spark,
    row_counts,
):
    datasets = generate_clean_dataset(
        spark=spark,
        row_counts=row_counts,
        seed=42,
    )

    customers = datasets["customers"]
    orders = datasets["orders"]

    orphan_count = (
        orders
        .join(
            customers,
            on="customer_id",
            how="left_anti",
        )
        .count()
    )

    assert orphan_count == 0


def test_clean_quantities_are_positive(
    spark,
    row_counts,
):
    datasets = generate_clean_dataset(
        spark=spark,
        row_counts=row_counts,
        seed=42,
    )

    invalid_count = (
        datasets["order_items"]
        .where(F.col("quantity") <= 0)
        .count()
    )

    assert invalid_count == 0


def test_injection_creates_manifest(
    spark,
    row_counts,
):
    clean_tables = generate_clean_dataset(
        spark=spark,
        row_counts=row_counts,
        seed=42,
    )

    defective_tables, manifest_rows = inject_defects(
        spark=spark,
        clean_tables=clean_tables,
        injection_run_id="test_run",
        seed=42,
    )

    assert len(manifest_rows) == 8

    defect_types = {
        row[3]
        for row in manifest_rows
    }

    expected_types = {
        "null_customer_id",
        "negative_quantity",
        "duplicate_order_id",
        "shipped_before_order",
        "invalid_status",
        "orphan_customer_id",
        "malformed_email",
        "future_order_date",
    }

    assert defect_types == expected_types


def test_injection_creates_negative_quantities(
    spark,
    row_counts,
):
    clean_tables = generate_clean_dataset(
        spark=spark,
        row_counts=row_counts,
        seed=42,
    )

    defective_tables, _ = inject_defects(
        spark=spark,
        clean_tables=clean_tables,
        injection_run_id="test_run",
        seed=42,
    )

    negative_count = (
        defective_tables["order_items"]
        .where(F.col("quantity") < 0)
        .count()
    )

    assert negative_count > 0