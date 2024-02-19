# Databricks notebook source
# MAGIC %pip install unidecode

# COMMAND ----------

import os
import glob

from functools import reduce
from pyspark.sql import DataFrame
import pyspark.sql.functions as f

import pandas as pd
from unidecode import unidecode
from utils.enviroment import BASE_DIR, LC_FRAGRANTICA_MATCHING

# not filtering by stock level in dev
# from utils.get_data import get_product_feed

# product_feed = get_product_feed()

# COMMAND ----------

path = "/dbfs/mnt/stg/house_of_fragrance/input/cbo_fragrance_category_persona.csv"
mapping = pd.read_csv(path).fillna("")
mapping["brand"] = mapping["brand"].apply(lambda b: unidecode(b).lower())
mapping = mapping[["brand", "category"]].rename(columns={"brand": "brand_desc"})
mapping = spark.createDataFrame(mapping)

# COMMAND ----------

matching = spark.table(LC_FRAGRANTICA_MATCHING)

# COMMAND ----------

main_accords = ["citrus", "earthy", "floral", "spicy", "sweet"]
dfs = []
for ma in main_accords:
    result_path = os.path.join(BASE_DIR.replace("/dbfs", ""), f"{ma}.parquet")
    result = spark.read.parquet(result_path)
    dfs.append(result)
clustering_results = reduce(DataFrame.unionAll, dfs)

# COMMAND ----------

results = clustering_results.join(matching.select("atg_code", "brand_desc"), on="atg_code", how="left")
results = results.withColumn("brand_desc", udf(lambda b: unidecode(b).lower())("brand_desc"))
results = results.join(mapping, on="brand_desc", how="left")

# COMMAND ----------

# check null
null_count = results.select(f.sum(f.col("category").isNull().cast("int"))).collect()[0][0]
print(f"# of products without category: {null_count}/{results.count()} ({(null_count/results.count())*100:.2f}%)")

# COMMAND ----------

# check null brand
brand = results.select("brand_desc", "category").dropDuplicates()
null_count = brand.select(f.sum(f.col("category").isNull().cast("int"))).collect()[0][0]
num_brand = brand.count()
p = null_count/num_brand
print(f"# of brand without category: {null_count}/{num_brand} ({p*100:.2f}%)")

# COMMAND ----------

# all brands that dun have category
display(brand)

# COMMAND ----------

# add another column for {main_accord}_{category}
# keep only those that has brand
nonnull_results = results.filter(f.col("category").isNotNull())
nonnull_results = nonnull_results.withColumn("cluster", f.concat(f.col("cluster"), f.lit("_"), f.col("category")))
nonnull_results = nonnull_results.withColumn("category", f.trim(f.col("category")))

# COMMAND ----------

# save
nonnull_results.write.parquet("/mnt/stg/house_of_fragrance/result_with_category.parquet", mode="overwrite")
display(nonnull_results)

# COMMAND ----------

display(nonnull_results.select("cluster").distinct())
