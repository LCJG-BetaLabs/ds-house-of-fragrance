# Databricks notebook source
dbutils.widgets.removeAll()
dbutils.widgets.text("result_path", "")

result_path = getArgument("result_path")

# COMMAND ----------

import os
import pyspark.sql.functions as f

from utils.enviroment import BASE_DIR, LC_FRAGRANTICA_MATCHING

# COMMAND ----------

# MAGIC %run ./notes_grouping

# COMMAND ----------

model_dir = os.path.join(BASE_DIR, "model")
result = spark.read.parquet(result_path)
matching = spark.table(LC_FRAGRANTICA_MATCHING)

# COMMAND ----------

result = result.withColumn("cluster", f.concat(f.lit("cluster "), f.col("cluster")))
result = result.withColumn("dummy", f.lit(1))

# COMMAND ----------

final_df = matching.join(result, on="atg_code", how="inner")
final_df.createOrReplaceTempView("final_df")

# COMMAND ----------

def sum_pivot_table(table, group_by_col, agg_col):
    df = table.groupBy("cluster", group_by_col).agg(f.sum(agg_col))
    pivot_table = (
        df.groupBy(group_by_col).pivot("cluster").agg(f.sum(f"sum({agg_col})"))
    ).fillna(0)
    display(pivot_table)
    return pivot_table


def count_pivot_table(table, group_by_col, agg_col):
    df = table.groupBy("cluster", group_by_col).agg(f.countDistinct(agg_col).alias("count"))
    pivot_table = (
        df.groupBy(group_by_col)
        .pivot("cluster")
        .agg(f.sum(f"count"))
    ).fillna(0)
    display(pivot_table)
    return pivot_table

# COMMAND ----------

count_pivot_table(final_df, "dummy", "atg_code")

# COMMAND ----------

count_pivot_table(final_df, "color_desc", "atg_code")

# COMMAND ----------

count_pivot_table(final_df, "class_desc", "atg_code")

# COMMAND ----------

count_pivot_table(final_df, "subclass_desc", "atg_code")

# COMMAND ----------

count_pivot_table(final_df, "brand_desc", "atg_code")

# COMMAND ----------

count_pivot_table(final_df, "for_gender", "atg_code")

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

def group_notes_udf(mapping):
    def group_notes(note):
        for k, v in mapping.items():
            if note in v:
                return k

    return udf(group_notes, StringType())

# COMMAND ----------

# base/middle/top notes
df = spark.sql("SELECT atg_code, base_notes, cluster FROM final_df")
df = df.withColumn("base_notes", f.explode(f.col("base_notes")))
count_pivot_table(df, "base_notes", "atg_code")

# COMMAND ----------

udf_group_notes = group_notes_udf(base_notes_mapping)
df = df.withColumn("group", udf_group_notes(df["base_notes"]))
count_pivot_table(df, "group", "atg_code")

# COMMAND ----------

# base/middle/top notes
df = spark.sql("SELECT atg_code, middle_notes, cluster FROM final_df")
df = df.withColumn("middle_notes", f.explode(f.col("middle_notes")))
count_pivot_table(df, "middle_notes", "atg_code")

# COMMAND ----------

udf_group_notes = group_notes_udf(middle_notes_mapping)
df = df.withColumn("group", udf_group_notes(df["middle_notes"]))
count_pivot_table(df, "group", "atg_code")

# COMMAND ----------

# base/middle/top notes
df = spark.sql("SELECT atg_code, top_notes, cluster FROM final_df")
df = df.withColumn("top_notes", f.explode(f.col("top_notes")))
count_pivot_table(df, "top_notes", "atg_code")

# COMMAND ----------

udf_group_notes = group_notes_udf(top_notes_mapping)
df = df.withColumn("group", udf_group_notes(df["top_notes"]))
count_pivot_table(df, "group", "atg_code")

# COMMAND ----------

# main accords
df = spark.sql("SELECT atg_code, main_accords, cluster FROM final_df").toPandas()
def preprocess_main_accords(d):
    return [k for k,v in d.items() if v is not None]

df["main_accords"] = df["main_accords"].apply(lambda d: preprocess_main_accords(d))
df_exploded = df.explode("main_accords")
count_pivot_table(spark.createDataFrame(df_exploded), "main_accords", "atg_code")

# COMMAND ----------

df = spark.sql("SELECT atg_code, main_accords, cluster FROM final_df").toPandas()

def group_accords(accords):
    for k, v in main_accords_grouping.items():
        if accords in v:
            return k
        
def get_max_dict_key(d):
    d = {key: value if value is not None else 0 for key, value in d.items()}
    return max(d, key=d.get)
    
df["main_accords"] = df["main_accords"].apply(lambda x: get_max_dict_key(x))
df["main_accords"] = df["main_accords"].apply(lambda x: group_accords(x))
count_pivot_table(spark.createDataFrame(df), "main_accords", "atg_code")

# COMMAND ----------

# process voting data
# show the one with highest vote

# COMMAND ----------

df = spark.sql("SELECT atg_code, season_rating, cluster FROM final_df").toPandas()

def get_day_night(d):
    if d["day"] > d["night"]:
        return "day"
    else:
        return "night"

def get_season(d):
    d = {key: value for key, value in d.items() if key not in ["day", "night"]}
    return max(d, key=d.get)
    
df["day_night"] = df["season_rating"].apply(lambda d: get_day_night(d))
df["season"] = df["season_rating"].apply(lambda d: get_season(d))


# COMMAND ----------

count_pivot_table(spark.createDataFrame(df), "day_night", "atg_code")

# COMMAND ----------

count_pivot_table(spark.createDataFrame(df), "season", "atg_code")

# COMMAND ----------

# sillage & longevity

df = spark.sql("SELECT atg_code, sillage, longevity, price_value, cluster FROM final_df").toPandas()

def get_max_dict_key(d):
    d = {key: value for key, value in d.items()}
    return max(d, key=d.get)

df["sillage"] = df["sillage"].apply(lambda d: get_max_dict_key(d))
df["longevity"] = df["longevity"].apply(lambda d: get_max_dict_key(d))
df["price_value"] = df["price_value"].apply(lambda d: get_max_dict_key(d))

# COMMAND ----------

count_pivot_table(spark.createDataFrame(df), "sillage", "atg_code")

# COMMAND ----------

count_pivot_table(spark.createDataFrame(df), "longevity", "atg_code")

# COMMAND ----------

count_pivot_table(spark.createDataFrame(df), "price_value", "atg_code")

# COMMAND ----------


