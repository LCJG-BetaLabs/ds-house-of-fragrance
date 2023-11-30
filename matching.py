# Databricks notebook source
# MAGIC %run "/utils/spark_utils"

# COMMAND ----------

# MAGIC %run ./vit/utils

# COMMAND ----------

import numpy as np
import pandas as pd
from utils.enviroment import (
    LC_ATTRIBUTE,
    FRAGRANTICA_ATTRIBUTE,
    LC_VIT_TABLE,
    LC_VIT_ENCODING_PATH,
    FRAGRANTICA_VIT_TABLE,
    FRAGRANTICA_VIT_ENCODING_PATH,
)

from sklearn.metrics.pairwise import cosine_similarity

# COMMAND ----------

# get lc product vit embedding
lc_attr = spark.table(LC_ATTRIBUTE)

items = np.array(lc_attr[["atg_code"]].collect()).flatten()
image_list = [atg + "_in_xl.jpg" for atg in items]
print("number of items: ", len(items))

lc_vit_encoding, lc_vit_encoding_id, _ = read_encoding(
    LC_VIT_TABLE,
    LC_VIT_ENCODING_PATH,
    image_list=image_list,  # ids are image name
    is_latest_only=True,
)

# COMMAND ----------

# convert image name back to atgs
atgs = [img_name[:6] for img_name in lc_vit_encoding_id]

# COMMAND ----------

lc_vit_df = pd.DataFrame(
    atgs,
    columns=["atg_code"]
)
lc_vit_df["encoding"] = lc_vit_encoding.tolist()

# COMMAND ----------

lc_vit_encoding = lc_vit_df["encoding"].values
lc_vit_encoding_id = lc_vit_df["atg_code"].values

# COMMAND ----------

# get fragrantica product vit embedding
f_attr = spark.table(FRAGRANTICA_ATTRIBUTE)
# use image name as id
image_list = np.array(f_attr[["image_name"]].collect()).flatten()
print("number of items: ", len(image_list))

f_vit_encoding, f_vit_encoding_id, _ = read_encoding(
    FRAGRANTICA_VIT_TABLE,
    FRAGRANTICA_VIT_ENCODING_PATH,
    image_list=image_list,  # ids are image name
    is_latest_only=True,
)

# COMMAND ----------

f_vit_df = pd.DataFrame(
    f_vit_encoding_id,
    columns=["image_name"]
)
f_vit_df["encoding"] = f_vit_encoding.tolist()

# COMMAND ----------

f_vit_encoding = f_vit_df["encoding"].values
f_vit_encoding_id = f_vit_df["image_name"].values

# COMMAND ----------

f_attr = f_attr.toPandas()
lc_attr = lc_attr.toPandas()

f_attr = f_attr.merge(f_vit_df, on="image_name", how="inner")
lc_attr = lc_attr.merge(lc_vit_df, on="atg_code", how="inner")

# COMMAND ----------

# brand name mapping for handling discrepancy of brand name
brand_name_mapping = pd.read_csv("/dbfs/mnt/stg/house_of_fragrance/fragrantica/scraping/brand_name_mapping.csv")
brand_name_mapping

# COMMAND ----------

f_attr = f_attr.merge(
    brand_name_mapping[["lc_brand", "fragrantica_brand"]],
    how="left",
    left_on="company",
    right_on="fragrantica_brand",
).rename(columns={"lc_brand": "brand_desc"})

# COMMAND ----------

# pre-process product name

f_attr["name"] = f_attr["name"].str.lower()
lc_attr["prod_desc_eng"] = lc_attr["prod_desc_eng"].str.lower()

f_attr["name_cleaned"] = f_attr["name"].str.replace("eau de parfum", "")
f_attr["name_cleaned"] = f_attr["name_cleaned"].str.replace("eau de cologne", "")
f_attr["name_cleaned"] = f_attr["name_cleaned"].str.replace("eau de toilette", "")

lc_attr["prod_desc_eng_cleaned"] = lc_attr["prod_desc_eng"].str.replace("eau de parfum", "")
lc_attr["prod_desc_eng_cleaned"] = lc_attr["prod_desc_eng_cleaned"].str.replace("eau de cologne", "")
lc_attr["prod_desc_eng_cleaned"] = lc_attr["prod_desc_eng_cleaned"].str.replace("eau de toilette", "")
lc_attr["prod_desc_eng_cleaned"] = lc_attr["prod_desc_eng_cleaned"].str.replace(r'\b\d+(?:ml|g)\b', '', regex=True)

# COMMAND ----------

# sim compute by brand

lc_attr = lc_attr.fillna("")
f_attr = f_attr.fillna("")

lc_brand = lc_attr["brand_desc"].str.upper()
f_brand = f_attr["brand_desc"].str.upper()

common_brands = np.intersect1d(lc_brand.unique(), f_brand.unique())

lc_ids = lc_attr["atg_code"].values
f_ids = f_attr["image_name"]

lc_attr["key"] = 0
f_attr["key"] = 0

matching_result_by_brand = {}  # result dict

# COMMAND ----------

set(lc_brand.unique())

# COMMAND ----------

len(lc_brand.unique())

# COMMAND ----------

f_brand.unique()

# COMMAND ----------

common_brands

# COMMAND ----------

def get_sim(lc_id, f_id, sim_array):
    sim = sim_array[_lc_ids == lc_id, _f_ids == f_id]
    return sim[0]

# COMMAND ----------

lc_attr

# COMMAND ----------

def get_token(string):
    token = set(string.split(" "))
    token.discard(" ")
    token.discard("")
    return token

# COMMAND ----------

def jaccard_similarity(set1, set2):
    """
    Computes the Jaccard similarity between two sets.
    """
    print(set1, set2)
    if len(set1) == 0 or len(set2) == 0:
        return 0
    intersection = len(set1.intersection(set2))
    print(intersection)
    union = len(set1.union(set2))
    print(union)
    similarity = intersection / union
    return similarity

# COMMAND ----------

result = []
for brand in common_brands:
    _lc_attr = lc_attr[lc_attr["brand_desc"] == brand]
    _f_attr = f_attr[f_attr["brand_desc"] == brand]

    _lc_vit_encoding = np.array(_lc_attr[["encoding"]].values.tolist()).reshape(-1, 768)
    _f_vit_encoding = np.array(_f_attr[["encoding"]].values.tolist()).reshape(-1, 768)

    _lc_ids = _lc_attr["atg_code"].values
    _f_ids = _f_attr["image_name"].values

    _sim = cosine_similarity(_lc_vit_encoding, _f_vit_encoding)

    matching = pd.merge(
        _lc_attr,
        _f_attr,
        how="outer",
        on="key",
    )

    matching["sim"] = matching.apply(
        lambda row: get_sim(row["atg_code"], row["image_name"], _sim), axis=1
    )
    matching["similar_name"] = matching.apply(
        lambda row: jaccard_similarity(
            get_token(row["prod_desc_eng_cleaned"]),
            get_token(row["name_cleaned"])
        ), axis=1
    )
    result.append(matching)

# COMMAND ----------

result = pd.concat(result)

# COMMAND ----------

result

# COMMAND ----------

test = result[result["similar_name"] > 0]

# COMMAND ----------

max_value_rows = test.loc[test.groupby("atg_code")["sim"].transform(max) == test["sim"]]

# COMMAND ----------

get_token("signature oud")

# COMMAND ----------

get_token("sandalo")

# COMMAND ----------

jaccard_similarity(get_token("signature oud"), get_token("sandalo"))

# COMMAND ----------

max_value_rows

# COMMAND ----------

# fix
max_value_rows["main_accords"] = max_value_rows["main_accords"].apply(
    lambda x: {key.replace(' ', '_'):value for key,value in x.items()} 
)
max_value_rows["longevity"] = max_value_rows["longevity"].apply(
    lambda x: {key.replace(' ', '_'):value for key,value in x.items()} 
)
max_value_rows["price_value"] = max_value_rows["price_value"].apply(
    lambda x: {key.replace(' ', '_'):value for key,value in x.items()} 
)
max_value_rows["gender_vote"] = max_value_rows["gender_vote"].apply(
    lambda x: {key.replace(' ', '_'):value for key,value in x.items()} 
)

# COMMAND ----------

max_value_rows.loc[69, "main_accords"]

# COMMAND ----------

max_value_rows[["atg_code", "prod_desc_eng", "brand_desc_x", "name", "image", "sim", "similar_name", "prod_desc_eng_cleaned", "name_cleaned"]]

# COMMAND ----------

pd.set_option('display.max_rows', 5)

# COMMAND ----------

LC_FRAGRANTICA_MATCHING = "lc_dev.ml_house_of_fragrance_silver.lc_fragrantica_matching"

create_or_insertoverwrite_table(
    spark.createDataFrame(max_value_rows),
    LC_FRAGRANTICA_MATCHING.split(".")[0],
    LC_FRAGRANTICA_MATCHING.split(".")[1],
    LC_FRAGRANTICA_MATCHING.split(".")[2],
    ds_managed=True,
)

# COMMAND ----------


