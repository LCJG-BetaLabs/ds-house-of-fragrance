# Databricks notebook source
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
import pyspark.sql.functions as f
from utils.functions import (
    get_season, get_max_dict_key, group_accords, group_notes, get_group_table, get_day_night
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

# merged_cluster_name : [cluster to be merged]
merge_cluster = {
    "citrus_AW": [f"citrus_AW_{m}" for m in middle_note],
    "earthy_AW_Warm & Earthy & Fresh & Light": ["earthy_AW_Warm & Earthy", "earthy_AW_Fresh & Light"],
    "earthy_SS_Fresh & Light & Warm & Earthy": ['earthy_SS_Fresh & Light', 'earthy_SS_Warm & Earthy'],
    "floral_AW_Fresh & Light & Sweet & Rich": ['floral_AW_Fresh & Light', 'floral_AW_Sweet & Rich'],
    "sweet_AW_Warm & Earthy & Fresh & Light": ['sweet_AW_Warm & Earthy', 'sweet_SS_Fresh & Light',],
    "sweet_SS_Warm & Earthy & Fresh & Light": ['sweet_SS_Warm & Earthy', 'sweet_AW_Fresh & Light',],
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

result

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

matching_result_pd = matching_result.toPandas()
matching_result_pd = matching_result_pd[
    ["atg_code", "for_gender", "season_rating", "main_accords", "middle_notes", "sillage", "longevity"]]

# main_accords/day night/season
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

gender_mapping = matching_result_pd.groupby("for_gender")["atg_code"].apply(set).to_dict()

# COMMAND ----------

matching_result_pd

# COMMAND ----------

def combine(data, key1, key2, _as=None):
    combined = data.get(key1, 0) + data.get(key2, 0)
    data.pop(key1, None)
    data.pop(key2, None)
    data[_as] = combined
    return data

# sillage
matching_result_pd["sillage"] = matching_result_pd["sillage"].apply(lambda x: combine(x, "strong", "enormous", _as="strong"))
matching_result_pd["sillage"] = matching_result_pd["sillage"].apply(lambda x: get_max_dict_key(x))
sillage_mapping = matching_result_pd.groupby("sillage")["atg_code"].apply(set).to_dict()

# longevity
matching_result_pd["longevity"] = matching_result_pd["longevity"].apply(lambda x: combine(x, "eternal", "long_lasting", _as="long_lasting"))
matching_result_pd["longevity"] = matching_result_pd["longevity"].apply(lambda x: combine(x, "very_weak", "weak", _as="weak"))
matching_result_pd["longevity"] = matching_result_pd["longevity"].apply(lambda x: get_max_dict_key(x))
longevity_mapping = matching_result_pd.groupby("longevity")["atg_code"].apply(set).to_dict()

# COMMAND ----------

cluster_mapping = result.groupby("new_group")["atg_code"].apply(set).to_dict()

# COMMAND ----------

cluster_mapping.keys()

# COMMAND ----------

# hardcode for now, may automate later
representative_items = {
    "citrus_AW": {
        "gender": "for women and men",
        "middle_notes": "Sweet & Rich",
        "main_accords": "citrus",
        "day_night": "day",
        "season": "AW",
        "sillage": "moderate",
        "longevity": "long_lasting",
    }
}

# COMMAND ----------

import random


def get_representative_item(cluster_name):
    d = representative_items[cluster_name]
    items = cluster_mapping[cluster_name] & gender_mapping[d["gender"]] & day_night_mapping[d["day_night"]] & season_mapping[d["season"]] & sillage_mapping[d["sillage"]] & longevity_mapping[d["longevity"]]
    print(items)
    random_item = random.choice(list(items))
    return random_item

# COMMAND ----------

get_representative_item("citrus_AW")

# COMMAND ----------

def rank_sim(item_list, represent):
    df = pd.DataFrame(item_list, columns=["source"])
    df["sim"] = represent
    df = spark.createDataFrame(df)
    df = df.join(sim_table.select("source", "sim", "weighted_score"), how="inner", on=["source", "sim"])
    return df

# COMMAND ----------

test = rank_sim(list(cluster_mapping["citrus_AW"]), get_representative_item("citrus_AW"))

# COMMAND ----------

display(test)
