"""
Data Pipeline Utility: 
This code file basically:
1. Handles the creation of Bronze, Silver and Gold tables respectively.
2. Processes data partition by partition - one month at a time. 

"""

import os
import glob
from datetime import datetime
from collections import Counter

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StringType, IntegerType, FloatType, DateType, MapType, ArrayType, NumericType
)
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer


# Bronze Layer 

def process_bronze_table(table_name, source_filepath, bronze_dir, snapshot_date_str, spark):
    """
    This layer is tasked to reads raw CSV files provided and then saves a filtered partition to bronze.
    Each month shall get its own parquet file.
    
    Why partition by month?
    - Practically in production, new data arrives monthly
    - Partitioning allows to process/reprocess one month at a time
    - Supports Airflow backfilling
    """
    df = spark.read.csv(source_filepath, header=True, inferSchema=False)
    
    # Lowercase all column names for consistency
    df = df.toDF(*[c.lower() for c in df.columns])

    # Filter to only the current month's snapshot
    df = df.filter(col("snapshot_date") == snapshot_date_str)

    # Add ingestion metadata
    df = df.withColumn("ingestion_timestamp", F.lit(datetime.utcnow().isoformat()))
    df = df.withColumn("source_file", F.lit(os.path.basename(source_filepath)))

    # Save as parquet partition
    partition_name = snapshot_date_str.replace("-", "_") + ".parquet"
    output_path = os.path.join(bronze_dir, table_name, partition_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.write.mode("overwrite").parquet(output_path)


# Silver Layer

def _process_attributes(df):
    """
    Clean customer attributes:
    - Extract numeric age
    - Enforce valid age range (0-120)
    - Null invalid occupations
    - Drop PII (name, ssn) as they never reach downstream tables
    """
    numeric_regex = r"([-+]?\d*\.?\d+)"
    df = df.withColumn("age", F.regexp_extract(col("age"), numeric_regex, 1))

    type_map = {
        "customer_id": StringType(),
        "age": IntegerType(),
        "occupation": StringType(),
        "snapshot_date": DateType(),
    }
    for c, t in type_map.items():
        df = df.withColumn(c, col(c).cast(t))

    # Enforce valid age
    df = df.withColumn(
        "age",
        F.when((col("age") >= 0) & (col("age") <= 120), col("age")).otherwise(None)
    )

    # Null invalid occupation
    df = df.withColumn(
        "occupation",
        F.when(col("occupation") == "_______", None).otherwise(col("occupation"))
    )

    # Drop PII: Name and SSN must never reach gold
    df = df.drop("name", "ssn")
    return df


def _process_clickstream(df):
    """
    Casting all 20 behavioural feature columns to integer.
    """
    type_map = {
        **{f"fe_{i}": IntegerType() for i in range(1, 21)},
        "customer_id": StringType(),
        "snapshot_date": DateType(),
    }
    for c, t in type_map.items():
        df = df.withColumn(c, col(c).cast(t))
    return df


def _split_loan_type(loan_type):
    if not isinstance(loan_type, str):
        return {}
    loans = loan_type.replace(" and ", ",").split(",")
    cleaned = [l.strip().replace(" ", "_").lower() for l in loans if l.strip()]
    return dict(Counter(cleaned))


def _process_financials(df, silver_dir, snapshot_date_str):
    """
    Clean financial features:
    - Extracting numeric values from dirty strings (e.g. '52312.68_')
    - Parsing Credit_History_Age into years and months
    - Splitting Payment_Behaviour into spent level and value level
    - Null invalid Credit_Mix values
    - Clipping outliers at 97th percentile
    - Splitting Type_of_Loan into its own silver table
    """
    numeric_regex = r"([-+]?\d*\.?\d+)"

    numeric_cols = {
        "annual_income": FloatType(),
        "monthly_inhand_salary": FloatType(),
        "num_bank_accounts": IntegerType(),
        "num_credit_card": IntegerType(),
        "interest_rate": IntegerType(),
        "num_of_loan": IntegerType(),
        "delay_from_due_date": IntegerType(),
        "num_of_delayed_payment": IntegerType(),
        "changed_credit_limit": FloatType(),
        "num_credit_inquiries": FloatType(),
        "outstanding_debt": FloatType(),
        "credit_utilization_ratio": FloatType(),
        "total_emi_per_month": FloatType(),
        "amount_invested_monthly": FloatType(),
        "monthly_balance": FloatType(),
    }

    for c, t in numeric_cols.items():
        df = df.withColumn(c, F.regexp_extract(col(c), numeric_regex, 1))
        df = df.withColumn(c, col(c).cast(t))

    # Parse credit history age
    df = df.withColumn("credit_history_age_year",
                       F.regexp_extract(col("credit_history_age"), r"(\d+)\s+Year", 1).cast(IntegerType()))
    df = df.withColumn("credit_history_age_month",
                       F.regexp_extract(col("credit_history_age"), r"(\d+)\s+Month", 1).cast(IntegerType()))

    # Null negative values in columns that shouldn't be negative
    for c in ["num_of_loan", "delay_from_due_date", "num_of_delayed_payment"]:
        df = df.withColumn(c, F.when(col(c) >= 0, col(c)).otherwise(None))

    # Clip outliers at 97th percentile
    for c in ["num_bank_accounts", "num_credit_card", "interest_rate",
              "num_of_loan", "num_of_delayed_payment"]:
        p97 = df.approxQuantile(c, [0.97], 0.01)[0]
        df = df.withColumn(c, F.when(col(c) > p97, p97).otherwise(col(c)))

    # Split payment behaviour into two clean columns
    pb_regex = r"(Low|High)_spent_(Small|Medium|Large)_value"
    df = df.withColumn("payment_behaviour_spent",
                       F.regexp_extract(col("payment_behaviour"), pb_regex, 1))
    df = df.withColumn("payment_behaviour_spent",
                       F.when(col("payment_behaviour_spent") != "", col("payment_behaviour_spent")).otherwise(None))
    df = df.withColumn("payment_behaviour_value",
                       F.regexp_extract(col("payment_behaviour"), pb_regex, 2))
    df = df.withColumn("payment_behaviour_value",
                       F.when(col("payment_behaviour_value") != "", col("payment_behaviour_value")).otherwise(None))

    # Null invalid credit mix
    df = df.withColumn("credit_mix",
                       F.when(col("credit_mix") == "_", None).otherwise(col("credit_mix")))

    # Build and save loan_type silver table separately
    df_loan = df.select("customer_id", "snapshot_date", "type_of_loan")
    split_udf = F.udf(_split_loan_type, MapType(StringType(), IntegerType()))
    df_loan = df_loan.withColumn("loan_type_counts", split_udf(col("type_of_loan")))
    all_keys = (
        df_loan.select("loan_type_counts")
        .rdd.flatMap(lambda r: r["loan_type_counts"].keys() if r["loan_type_counts"] else [])
        .distinct().collect()
    )
    for key in all_keys:
        df_loan = df_loan.withColumn(key, F.coalesce(col("loan_type_counts").getItem(key), F.lit(0)))
    df_loan = df_loan.drop("loan_type_counts", "type_of_loan")
    partition_name = snapshot_date_str.replace("-", "_") + ".parquet"
    loan_path = os.path.join(silver_dir, "loan_type", partition_name)
    os.makedirs(os.path.dirname(loan_path), exist_ok=True)
    df_loan.write.mode("overwrite").parquet(loan_path)

    return df.drop("payment_behaviour", "type_of_loan", "credit_history_age")


def _process_lms(df):
    """
    Process Loan Management System data:
    - Casting all columns to correct types
    - Adding mob (month on book) which is theinstallment number
    - Adding dpd (days past due) which indicates how many days overdue
    
    mob and dpd are the industry-standard way on how banks track loan health.
    Label definition: default = dpd >= 30 at mob == 6
    """
    type_map = {
        "loan_id": StringType(),
        "customer_id": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for c, t in type_map.items():
        df = df.withColumn(c, col(c).cast(t))

    # mob = month on book (how many months into the loan)
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # dpd = days past due (how many days overdue is the customer)
    df = df.withColumn("installments_missed",
                       F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date",
                       F.when(col("installments_missed") > 0,
                              F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd",
                       F.when(col("overdue_amt") > 0.0,
                              F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))
    return df


def process_silver_table(table_name, bronze_dir, silver_dir, snapshot_date_str, spark):
    """
    Read bronze partition then clean and finally save to silver.
    """
    partition_name = snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = os.path.join(bronze_dir, table_name, partition_name)
    df = spark.read.parquet(filepath)
    df = df.toDF(*[c.lower() for c in df.columns])

    # Drop bronze metadata columns
    df = df.drop("ingestion_timestamp", "source_file")

    if table_name == "attributes":
        df = _process_attributes(df)
    elif table_name == "clickstream":
        df = _process_clickstream(df)
    elif table_name == "financials":
        df = _process_financials(df, silver_dir, snapshot_date_str)
    elif table_name == "lms":
        df = _process_lms(df)

    out_path = os.path.join(silver_dir, table_name, partition_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.write.mode("overwrite").parquet(out_path)


# Gold Layer (Label store and Feature Store)

def _read_silver(table, silver_dir, spark):
    """Read all the partitions of a silver table."""
    folder = os.path.join(silver_dir, table)
    files = glob.glob(os.path.join(folder, "*.parquet"))
    return spark.read.parquet(*files)


def _one_hot_encode(df, col_name):
    """One-hot encode a categorical column using Spark ML."""
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index", handleInvalid="keep")
    model = indexer.fit(df)
    df = model.transform(df)
    encoder = OneHotEncoder(inputCol=f"{col_name}_index", outputCol=f"{col_name}_ohe", dropLast=False)
    df = encoder.fit(df).transform(df)
    to_arr = F.udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
    df = df.withColumn(f"{col_name}_arr", to_arr(f"{col_name}_ohe"))
    for i, cat in enumerate([c.lower() for c in model.labels]):
        df = df.withColumn(f"{col_name}_{cat}", df[f"{col_name}_arr"][i].cast(IntegerType()))
    df = df.drop(col_name, f"{col_name}_index", f"{col_name}_ohe", f"{col_name}_arr")
    return df


def build_label_store(silver_dir, gold_dir, snapshot_date_str, spark, mob=6, dpd=30):
    """
    Building label store for a given snapshot date.
    
    Label definition:
    - Looking at each loan at mob=6 (6 months into the loan)
    - If dpd >= 30 (30+ days overdue) then is_default = 1
    - Otherwise - is_default = 0

    """
    df_lms = _read_silver("lms", silver_dir, spark)

    # Get each loan at exactly mob=6
    df = df_lms.filter(col("mob") == mob)

    # Apply default definition
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob"))

    df = df.select("loan_id", "customer_id", "label", "label_def", "snapshot_date")

    # Filter to current snapshot date
    df = df.filter(col("snapshot_date") == snapshot_date_str)

    partition_name = snapshot_date_str.replace("-", "_") + ".parquet"
    out_path = os.path.join(gold_dir, "label_store", partition_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.write.mode("overwrite").parquet(out_path)
    return df


def build_feature_store(silver_dir, gold_dir, snapshot_date_str, spark):
    """
    Building feature store for a given snapshot date.
    Joins attributes + financials + loan_type + clickstream.
    Then apply one-hot encoding and imputation.
    Only use data available before loan start date to ensure no data leakage.
    """
    df_attr = _read_silver("attributes", silver_dir, spark)
    df_fin = _read_silver("financials", silver_dir, spark)
    df_loan_type = _read_silver("loan_type", silver_dir, spark)
    df_cs = _read_silver("clickstream", silver_dir, spark)
    df_lms = _read_silver("lms", silver_dir, spark)

    # Filter all tables to current snapshot
    df_attr = df_attr.filter(col("snapshot_date") == snapshot_date_str)
    df_fin = df_fin.filter(col("snapshot_date") == snapshot_date_str)
    df_loan_type = df_loan_type.filter(col("snapshot_date") == snapshot_date_str)

    # Join attributes, financials, loan types
    df = df_attr.join(df_fin, on=["customer_id", "snapshot_date"], how="inner")
    df = df.join(df_loan_type, on=["customer_id", "snapshot_date"], how="inner")
    df = df.drop("credit_history_age")

    # Combine credit history age into total months
    df = df.withColumn("credit_history_age_months",
                       F.col("credit_history_age_year") * 12 + F.col("credit_history_age_month"))
    df = df.drop("credit_history_age_year", "credit_history_age_month")

    # Impute nulls with mean for numeric columns
    numeric_cols = [c for c in df.columns if isinstance(df.schema[c].dataType, NumericType)]
    imputer = Imputer(inputCols=numeric_cols, outputCols=numeric_cols)
    df = imputer.fit(df).transform(df)

    # One-hot encode categorical columns
    for cat_col in ["occupation", "payment_of_min_amount", "credit_mix",
                    "payment_behaviour_spent", "payment_behaviour_value"]:
        if cat_col in df.columns:
            df = _one_hot_encode(df, cat_col)

    # Aggregate clickstream: mean of fe_1..fe_20 over all history before loan start
    df_lms_mob0 = df_lms.filter(col("mob") == 0).select("customer_id", col("snapshot_date").alias("loan_date"))
    df_cs_filtered = df_cs.join(df_lms_mob0, on="customer_id", how="inner")
    df_cs_filtered = df_cs_filtered.filter(col("snapshot_date") <= col("loan_date"))
    agg_exprs = [F.avg(f"fe_{i}").alias(f"avg_fe_{i}") for i in range(1, 21)]
    df_cs_agg = df_cs_filtered.groupBy("customer_id").agg(*agg_exprs)

    df = df.join(df_cs_agg, on="customer_id", how="left")

    # Save partition
    partition_name = snapshot_date_str.replace("-", "_") + ".parquet"
    out_path = os.path.join(gold_dir, "feature_store", partition_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.write.mode("overwrite").parquet(out_path)
    return df
