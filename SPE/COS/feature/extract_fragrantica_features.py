# Databricks notebook source
# MAGIC %run "/utils/spark_utils"

# COMMAND ----------

import numpy as np
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import *

# COMMAND ----------

im = spark.table("lc_dev.ml_house_of_fragrance_silver.lc_fragrantica_matching")

accords = [F.coalesce(F.col(f"main_accords.{c}"), F.lit(0)).cast("double") for c in im.select("main_accords.*").columns]
accords = im.select("atg_code", F.array(accords).alias("main_accords_vt"))

normalize_udf = F.udf(lambda x: [value / sum(x) if sum(x) != 0 else 0 for value in x], F.ArrayType(DoubleType()))
accords = accords.withColumn("main_accords_vt", normalize_udf(F.col("main_accords_vt")))

create_or_insertoverwrite_table(
    accords, 
    "lc_dev",
    "ml_house_of_fragrance_silver",
    "intermediate_accord",
    ds_managed=True,
    )

# COMMAND ----------

season = [F.coalesce(F.col(f"season_rating.{c}"), F.lit(0)).cast("double") for c in im.select("season_rating.*").columns]
season = im.select("atg_code", F.array(season).alias("season_rating_vt"))

season = season.withColumn("season_rating_vt", normalize_udf(F.col("season_rating_vt")))
season = season.na.fill(0)
create_or_insertoverwrite_table(
    season, 
    "lc_dev",
    "ml_house_of_fragrance_silver",
    "intermediate_season",
    ds_managed=True,
    )

# COMMAND ----------

note = im.select("atg_code", "middle_notes")
create_or_insertoverwrite_table(
    note, 
    "lc_dev",
    "ml_house_of_fragrance_silver",
    "intermediate_note",
    ds_managed=True,
    )
