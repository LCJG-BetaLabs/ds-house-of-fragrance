# Databricks notebook source
# MAGIC %pip install unidecode

# COMMAND ----------

import os
import itertools
import pandas as pd
from functools import reduce
from pyspark.sql import DataFrame
import pyspark.sql.functions as f
from unidecode import unidecode
from utils.enviroment import LC_FRAGRANTICA_MATCHING, BASE_DIR
from utils.functions import (
    get_season, get_max_dict_key, group_accords, group_notes, get_day_night,
)

# COMMAND ----------

# MAGIC %run ./notes_grouping

# COMMAND ----------

# get matching result
matching_result = spark.table(LC_FRAGRANTICA_MATCHING)
matching_result = matching_result.select(
    "atg_code",
    "prod_desc_eng",
    "brand_desc",
    "long_desc",
    "care",
    "for_gender",
    "rating",
    "number_votes",
    "main_accords",
    "season_rating",
    "description",
    "top_notes",
    "middle_notes",
    "base_notes",
    "longevity",
    "sillage",
    "gender_vote",
    "price_value",
)

# COMMAND ----------

# rules:
# - top_main_accords
#   - divide into floral, spicy, sweet+vanilla, woody+earthy+musky, citrus
# - season
#   - divide into AW, SS
# - middle note
#   - divide into "Fresh & Light", "Warm & Earthy", "Sweet & Rich"
# total 5*2*3 = 30 segments

# if one segment has less than 5 items, will be merge with neighbor segment

# COMMAND ----------

main_accords_final_group = {
    "floral": ["floral"],
    "spicy": ["spicy"],
    "sweet": ["vanilla", "sweet"],
    "earthy": ["earthy", "musky", "powdery", "woody"],
    "citrus": ["citrus", "fresh"],
    "mineral": ["mineral"]
}

middle_notes_final_group = {
    "Fresh & Light": ["Citrus/Fruity", "Fresh/Aquatic", "Green/Herbal"],
    "Warm & Earthy": ["Earthy/Spicy", "Spicy", "Woody"],
    "Sweet & Rich": ["Floral", "Gourmand"],
}

# COMMAND ----------

matching_result_pd = matching_result.toPandas()
matching_result_pd = matching_result_pd[["atg_code", "for_gender", "season_rating", "main_accords", "middle_notes", "sillage", "longevity"]]
matching_result_pd["main_accords"] = matching_result_pd["main_accords"].apply(lambda x: get_max_dict_key(x))
matching_result_pd["main_accords"] = matching_result_pd["main_accords"].apply(
    lambda x: group_accords(x, main_accords_grouping))
matching_result_pd["main_accords"] = matching_result_pd["main_accords"].apply(
    lambda x: group_accords(x, main_accords_final_group))
matching_result_pd["season"] = matching_result_pd["season_rating"].apply(lambda d: get_season(d))

# middle notes
matching_result_pd = matching_result_pd.explode("middle_notes")
matching_result_pd["middle_notes"] = matching_result_pd["middle_notes"].apply(
    lambda x: group_notes(x, middle_notes_mapping))
matching_result_pd["middle_notes"] = matching_result_pd["middle_notes"].apply(
    lambda x: group_notes(x, middle_notes_final_group))

season_mapping = matching_result_pd.groupby("season")["atg_code"].apply(set).to_dict()
main_accords_mapping = matching_result_pd.groupby("main_accords")["atg_code"].apply(set).to_dict()
middle_notes_mapping_ = matching_result_pd.groupby("middle_notes")["atg_code"].apply(set).to_dict()

# COMMAND ----------

matching_result_pd
for_gender_mapping = matching_result_pd.groupby("for_gender")["atg_code"].apply(set).to_dict()
day_night_mapping = matching_result_pd.groupby("for_gender")["atg_code"].apply(set).to_dict()
sillage_mapping = matching_result_pd.groupby("for_gender")["atg_code"].apply(set).to_dict()
longevity_mapping = matching_result_pd.groupby("for_gender")["atg_code"].apply(set).to_dict()

# COMMAND ----------

combinations = list(itertools.product(season_mapping.keys(), middle_notes_mapping_.keys()))
print(combinations)

# COMMAND ----------

all_result = []
for ma in main_accords_mapping.keys():
    all_dfs = []
    for i, (s, mn) in enumerate(combinations):
        atgs = main_accords_mapping[ma] & season_mapping[s] & middle_notes_mapping_[mn]
        df = pd.DataFrame(atgs, columns=["atg_code"])
        df["main_accord"] = ma
        df["season"] = s
        df["middle_notes"] = mn
        df["cluster"] = f"{ma}_{s}_{mn}"
        all_dfs.append(df)

    result = pd.concat(all_dfs)[["atg_code", "cluster"]]
    result_path = os.path.join(BASE_DIR.replace("/dbfs", ""), "result_v3", f"{ma}.parquet")
    print(ma, len(result))

    if len(result) > 0:
        result_df = spark.createDataFrame(result)
        result_df.write.parquet(result_path, mode="overwrite")
        all_result.append(result_df)

final_result = reduce(DataFrame.unionAll, all_result)

# COMMAND ----------

result_path

# COMMAND ----------

display(final_result)

# COMMAND ----------

# consider "category"
path = "/dbfs/mnt/stg/house_of_fragrance/input/cbo_fragrance_category_persona.csv"
mapping = pd.read_csv(path).fillna("")
mapping["brand"] = mapping["brand"].apply(lambda b: unidecode(b).lower())
mapping = mapping[["brand", "category"]].rename(columns={"brand": "brand_desc"})
mapping = spark.createDataFrame(mapping)

results = final_result.join(matching_result.select("atg_code", "brand_desc"), on="atg_code", how="left")
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
save_path = "/mnt/stg/house_of_fragrance/result_with_category.parquet"
nonnull_results.write.parquet(save_path, mode="overwrite")
display(nonnull_results)

# COMMAND ----------

dbutils.notebook.run(
    "./profiling_features",
    0,
    {
        "result_path": save_path
    }
)

# COMMAND ----------

# rank
import numpy as np

cluster_path = "/mnt/stg/house_of_fragrance/result_with_category.parquet"
cluster = spark.read.parquet(cluster_path)
cluster_pd = cluster.toPandas()
cluster_pd["cluster"] = cluster_pd["cluster"].apply(
    lambda x: "_".join([x.split("_")[0], x.split("_")[1], x.split("_")[3]])
)
cluster = spark.createDataFrame(cluster_pd)
cluster.createOrReplaceTempView("df")

# COMMAND ----------

# get representative items
profiling = pd.read_csv("/dbfs/mnt/stg/house_of_fragrance/profiling_result.csv")
profiling = profiling.set_index("dummy")
profiling = profiling.drop(index=[np.nan])

# COMMAND ----------

features = ["for_gender", "group", "main_accords", "day_night", "season", "sillage", "longevity"]
result_dict = {}
for i, f in enumerate(features):
    if i < len(features) - 1:
        subdf = profiling[features[i]:features[i+1]]
    else:
        subdf = profiling[features[i]:]
    subdf = subdf.drop(index=[fe for fe in features if fe in subdf.index])

    for column in subdf.columns:
        
        max_index = subdf[[column]].astype(int).idxmax()
        if column not in result_dict:
            result_dict[column] = {f: max_index.iloc[0]}
        else:
            result_dict[column][f] = max_index.iloc[0]

# COMMAND ----------

result_dict

# COMMAND ----------

# for each cluster, get most representative item


# COMMAND ----------

# filter for business requirement


# COMMAND ----------

# save output parquat and excel file
