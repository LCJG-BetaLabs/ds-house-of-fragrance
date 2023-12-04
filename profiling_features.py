# Databricks notebook source
import os
import pyspark.sql.functions as f

from utils.enviroment import BASE_DIR, LC_FRAGRANTICA_MATCHING

# COMMAND ----------

model_dir = os.path.join(BASE_DIR, "model")
result = spark.read.parquet(os.path.join(model_dir.replace("/dbfs", ""), "clustering_result.parquet"))
matching = spark.table(LC_FRAGRANTICA_MATCHING)

# COMMAND ----------

result = result.withColumn("cluster", f.concat(f.lit("cluster "), f.col("cluster")))

# COMMAND ----------

display(result)

# COMMAND ----------

display(matching)

# COMMAND ----------

final_df = matching.join(result, on="atg_code", how="left")

# COMMAND ----------

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

count_pivot_table(final_df, "color_desc", "atg_code")

# COMMAND ----------

count_pivot_table(final_df, "class_desc", "atg_code")

# COMMAND ----------

count_pivot_table(final_df, "subclass_desc", "atg_code")

# COMMAND ----------

count_pivot_table(final_df, "brand_desc", "atg_code")

# COMMAND ----------

# price point (H, M, L from percentile)


# COMMAND ----------

count_pivot_table(final_df, "for_gender", "atg_code")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT atg_code, top_notes, middle_notes, base_notes FROM final_df

# COMMAND ----------

# base/middle/top notes
df = spark.sql("SELECT atg_code, base_notes, cluster FROM final_df")
df = df.withColumn("base_notes", f.explode(f.col("base_notes")))
count_pivot_table(df, "base_notes", "atg_code")

# COMMAND ----------

# base/middle/top notes
df = spark.sql("SELECT atg_code, middle_notes, cluster FROM final_df")
df = df.withColumn("middle_notes", f.explode(f.col("middle_notes")))
count_pivot_table(df, "middle_notes", "atg_code")

# COMMAND ----------

# base/middle/top notes
df = spark.sql("SELECT atg_code, top_notes, cluster FROM final_df")
df = df.withColumn("top_notes", f.explode(f.col("top_notes")))
count_pivot_table(df, "top_notes", "atg_code")

# COMMAND ----------

# main accords
df = spark.sql("SELECT atg_code, main_accords, cluster FROM final_df").toPandas()
def preprocess_main_accords(d):
    return [k for k,v in d.items() if v is not None]

df["main_accords"] = df["main_accords"].apply(lambda d: preprocess_main_accords(d))
df_exploded = df.explode("main_accords")
count_pivot_table(spark.createDataFrame(df_exploded), "main_accords", "atg_code")

# COMMAND ----------

# process voting data
# show the one with highest vote

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT atg_code, season_rating, cluster FROM final_df

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

df = spark.sql("SELECT atg_code, sillage, longevity, cluster FROM final_df").toPandas()

def get_max_dict_key(d):
    d = {key: value for key, value in d.items()}
    return max(d, key=d.get)

df["sillage"] = df["sillage"].apply(lambda d: get_max_dict_key(d))
df["longevity"] = df["longevity"].apply(lambda d: get_max_dict_key(d))

# COMMAND ----------

count_pivot_table(spark.createDataFrame(df), "sillage", "atg_code")

# COMMAND ----------

count_pivot_table(spark.createDataFrame(df), "longevity", "atg_code")

# COMMAND ----------


