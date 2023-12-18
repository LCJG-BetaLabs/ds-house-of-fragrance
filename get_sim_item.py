# Databricks notebook source
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
import pyspark.sql.functions as f
from utils.functions import (
    get_season, get_max_dict_key, group_accords, group_notes, get_group_table
)
from utils.enviroment import LC_FRAGRANTICA_MATCHING

path = "/mnt/stg/house_of_fragrance/similar_product_engine/fragrance/sim_table"
sim_table = spark.read.parquet(path)
display(sim_table)

# COMMAND ----------

weights = [0.4, 0.3, 0.1, 0.2]


@udf(returnType=FloatType())
def calculate_weighted_score(col1, col2, col3, col4):
    return col1 * weights[0] + col2 * weights[1] + col3 * weights[2] + col4 * weights[3]


# Apply the UDF to calculate the weighted score for each row
sim_table = sim_table.withColumn(
    "weighted_score",
    calculate_weighted_score(f.col("bert_sim"), f.col("accord_sim"), f.col("season_sim"), f.col("note_sim"))
)

# COMMAND ----------

family = ["floral", "spicy", "sweet", "earthy", "citrus"]
middle_note = ['Fresh & Light', 'Warm & Earthy', 'Sweet & Rich']

merge_cluster = {
    "citrus_AW": [f"citrus_AW_{m}" for m in middle_note],
}


def group_cluster(x, mapping):
    for k, v in mapping.items():
        if x in v:
            return k
    return x


dfs = []
for _family in family:
    family_df = get_group_table(_family).toPandas()
    family_df["new_group"] = family_df["cluster"].apply(lambda x: group_cluster(x, merge_cluster))
    dfs.append(family_df)

result = pd.concat(dfs)

# COMMAND ----------

# select item that represent the cluster
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

# MAGIC %run ./notes_grouping

# COMMAND ----------

def get_day_night(d):
    day = d["day"]
    night = d["night"]
    if day > night:
        return "day"
    else:
        return "night"


# COMMAND ----------

matching_result_pd = matching_result.toPandas()
matching_result_pd = matching_result_pd[
    ["atg_code", "for_gender", "season_rating", "main_accords", "middle_notes", "sillage", "longevity"]]
matching_result_pd["main_accords"] = matching_result_pd["main_accords"].apply(lambda x: get_max_dict_key(x))
matching_result_pd["main_accords"] = matching_result_pd["main_accords"].apply(
    lambda x: group_accords(x, main_accords_grouping))
matching_result_pd["main_accords"] = matching_result_pd["main_accords"].apply(
    lambda x: group_accords(x, main_accords_final_group))

matching_result_pd["day_night"] = matching_result_pd["season_rating"].apply(lambda d: get_day_night(d))
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
day_night_mapping = matching_result_pd.groupby("day_night")["atg_code"].apply(set).to_dict()

# COMMAND ----------

haiii = citrus.groupby("new_group")["atg_code"].apply(set).to_dict()

# COMMAND ----------

haiii

# COMMAND ----------

representative = {
    "AW": {
        "season": "AW",
        "day_night": "day",
        "main_accords": "citrus",
        "gender": "for women and men",
    }
}

# COMMAND ----------

haiii["AW"] & season_mapping["AW"] & main_accords_mapping["citrus"] & day_night_mapping["day"]

# COMMAND ----------

import pandas as pd


# COMMAND ----------

def rank_sim(item_list, represent):
    df = pd.DataFrame(item_list, columns=["source"])
    df["sim"] = represent
    df = spark.createDataFrame(df)
    df = df.join(sim_table.select("source", "sim", "weighted_score"), how="inner", on=["source", "sim"])
    return df


# COMMAND ----------

test = rank_sim(list(haiii["AW"]), "BKU306")

# COMMAND ----------

display(test)

# COMMAND ----------

represent = "BKU306"

# COMMAND ----------

season_mapping = matching_result_pd.groupby("season")["atg_code"].apply(set).to_dict()
main_accords_mapping = matching_result_pd.groupby("main_accords")["atg_code"].apply(set).to_dict()
middle_notes_mapping_ = matching_result_pd.groupby("middle_notes")["atg_code"].apply(set).to_dict()
day_night_mapping
