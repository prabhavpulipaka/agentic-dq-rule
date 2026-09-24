from pyspark.sql import SparkSession

from dq_author.data.inject_defects import (
    inject_defects,
    write_manifest,
)

# ---------------------------------------------
# Configuration
# ---------------------------------------------

CATALOG = "main"
SCHEMA = "dq_demo"

INJECTION_RUN_ID = "run_001"
SEED = 42

spark = SparkSession.builder.getOrCreate()

# ---------------------------------------------
# Read clean baseline
# ---------------------------------------------

table_names = [
    "customers",
    "orders",
    "order_items",
    "payments",
]

clean_tables = {
    name: spark.table(f"{CATALOG}.{SCHEMA}.{name}")
    for name in table_names
}

# ---------------------------------------------
# Inject defects
# ---------------------------------------------

defective_tables, manifest_rows = inject_defects(
    spark=spark,
    clean_tables=clean_tables,
    injection_run_id=INJECTION_RUN_ID,
    seed=SEED,
)

# ---------------------------------------------
# Write defective tables
# ---------------------------------------------

for table_name, dataframe in defective_tables.items():
    table_identifier = (
        f"{CATALOG}.{SCHEMA}.{table_name}_defective"
    )

    (
        dataframe.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_identifier)
    )

    print(f"Written: {table_identifier}")

# ---------------------------------------------
# Write ground-truth manifest
# ---------------------------------------------

write_manifest(
    spark=spark,
    manifest_rows=manifest_rows,
    catalog=CATALOG,
    schema=SCHEMA,
    mode="append",
)

display(
    spark.table(
        f"{CATALOG}.{SCHEMA}.injected_defects"
    )
)