# Databricks notebook source
import os
import itertools
import pandas as pd
from utils.enviroment import LC_FRAGRANTICA_MATCHING, BASE_DIR
from utils.functions import (
    get_season, get_max_dict_key, group_accords, group_notes
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
}

middle_notes_final_group = {
    "Fresh & Light": ["Citrus/Fruity", "Fresh/Aquatic", "Green/Herbal"],
    "Warm & Earthy": ["Earthy/Spicy", "Spicy", "Woody"],
    "Sweet & Rich": ["Floral", "Gourmand"],
}

# COMMAND ----------

matching_result_pd = matching_result.toPandas()
matching_result_pd = matching_result_pd[["atg_code", "for_gender", "season_rating", "main_accords", "middle_notes"]]
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

combinations = list(itertools.product(season_mapping.keys(), middle_notes_mapping_.keys()))
print(combinations)

# COMMAND ----------

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
    result_path = os.path.join(BASE_DIR.replace("/dbfs", ""), f"{ma}.parquet")
    spark.createDataFrame(result).write.parquet(result_path, mode="overwrite")

    dbutils.notebook.run(
        "./profiling_features",
        0,
        {
            "result_path": result_path
        }
    )

# COMMAND ----------
