from pyspark.sql import SparkSession

from dq_author.data.generate_clean import (
    generate_clean_dataset,
    write_delta_tables,
)

# ---------------------------------------------
# Configuration
# ---------------------------------------------

CATALOG = "main"
SCHEMA = "dq_demo"
SEED = 42

ROW_COUNTS = {
    "customers": 10_000,
    "orders": 25_000,
    "order_items": 60_000,
    "payments": 25_000,
}

spark = SparkSession.builder.getOrCreate()

# ---------------------------------------------
# Create schema
# ---------------------------------------------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# ---------------------------------------------
# Generate clean data
# ---------------------------------------------

datasets = generate_clean_dataset(
    spark=spark,
    row_counts=ROW_COUNTS,
    seed=SEED,
)

# ---------------------------------------------
# Write Delta tables
# ---------------------------------------------

write_delta_tables(
    datasets=datasets,
    catalog=CATALOG,
    schema=SCHEMA,
    mode="overwrite",
)

# ---------------------------------------------
# Basic verification
# ---------------------------------------------

for table_name in datasets:
    table_identifier = f"{CATALOG}.{SCHEMA}.{table_name}"

    count = spark.table(table_identifier).count()

    print(f"{table_identifier}: {count} rows")