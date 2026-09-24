import sys
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
sys.path.append("/Workspace/Users/pulipakaprabhav@gmail.com/agentic-dq-rule/src")
from datetime import datetime, timezone
from pyspark.sql import functions as F
from pyspark.sql.types import (
    NumericType,
    StringType,
    TimestampType,
)

# ---------------------------------------------
# Configuration
# ---------------------------------------------

CATALOG = "workspace"
SCHEMA = "dq_demo"

PROFILE_TABLE = f"{CATALOG}.{SCHEMA}.dq_profiles"

TABLES_TO_PROFILE = [
    "customers_defective",
    "orders_defective",
    "order_items_defective",
    "payments_defective",
]

spark = SparkSession.builder.getOrCreate()

# ---------------------------------------------
# Profiling helper
# ---------------------------------------------


def profile_table(
    table_identifier: str,
    table_name: str,
):
    dataframe = spark.table(table_identifier)

    total_rows = dataframe.count()

    profile_rows = []

    for field in dataframe.schema.fields:
        column_name = field.name
        data_type = field.dataType.simpleString()

        column_df = dataframe.select(F.col(column_name))

        null_count = (
            column_df
            .where(F.col(column_name).isNull())
            .count()
        )

        null_percentage = (
            (null_count / total_rows) * 100
            if total_rows > 0
            else 0.0
        )

        distinct_count = (
            column_df
            .where(F.col(column_name).isNotNull())
            .distinct()
            .count()
        )

        min_value = None
        max_value = None

        if isinstance(
            field.dataType,
            (NumericType, TimestampType),
        ):
            bounds = dataframe.select(
                F.min(F.col(column_name)).alias("min_value"),
                F.max(F.col(column_name)).alias("max_value"),
            ).first()

            min_value = (
                str(bounds["min_value"])
                if bounds["min_value"] is not None
                else None
            )

            max_value = (
                str(bounds["max_value"])
                if bounds["max_value"] is not None
                else None
            )

        top_values = (
            dataframe
            .groupBy(F.col(column_name))
            .count()
            .orderBy(F.desc("count"))
            .limit(10)
            .collect()
        )

        value_distribution = [
    {
        "value": (
            "<NULL>"
            if row[column_name] is None
            else str(row[column_name])
        ),
        "count": row["count"],
    }
    for row in top_values
    ]

        profile_rows.append(
            {
                "table_name": table_name,
                "column_name": column_name,
                "data_type": data_type,
                "total_rows": total_rows,
                "null_count": null_count,
                "null_percentage": float(null_percentage),
                "distinct_count": distinct_count,
                "min_value": min_value,
                "max_value": max_value,
                "value_distribution": value_distribution,
                "profiled_at": datetime.now(timezone.utc).replace(tzinfo=None),
            }
        )

    return profile_rows


# ---------------------------------------------
# Run profiling
# ---------------------------------------------

all_profile_rows = []

for table_name in TABLES_TO_PROFILE:
    table_identifier = f"{CATALOG}.{SCHEMA}.{table_name}"

    print(f"Profiling: {table_identifier}")

    all_profile_rows.extend(
        profile_table(
            table_identifier=table_identifier,
            table_name=table_name,
        )
    )

# ---------------------------------------------
# Create DataFrame and write Delta
# ---------------------------------------------

profile_df = spark.createDataFrame(all_profile_rows)

(
    profile_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(PROFILE_TABLE)
)

display(spark.table(PROFILE_TABLE))