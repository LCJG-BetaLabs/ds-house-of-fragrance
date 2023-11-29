# Databricks notebook source
# MAGIC %run "/utils/spark_utils"

# COMMAND ----------

import os
import glob
import json
import pandas as pd

from utils.get_data import get_product_feed, get_lc_perfume_data
from utils.download_image import download_image
from utils.enviroment import (
    FRAGRANTICA_DIR,
    FRAGRANTICA_IMAGE_DIR,
    FRAGRANTICA_ATTRIBUTE,
    LC_ATTRIBUTE,
)


def insert_image_name_to_df(df, prod_name, image_name):
    df.loc[df["name"] == prod_name, "image_name"] = image_name


# COMMAND ----------

product_feed = get_product_feed()
lc_data = get_lc_perfume_data(product_feed)

# COMMAND ----------

# get fragrantica data
fragrantica_path = os.path.join(FRAGRANTICA_DIR, "scraping")
brand_folder = [f for f in os.listdir(fragrantica_path) if os.path.isdir(os.path.join(fragrantica_path, f))]

all_products = []
for brand in brand_folder:
    if brand == "brand_json":
        continue
    path = os.path.join(fragrantica_path, brand)
    products = []
    product_files = glob.glob(os.path.join(path, "*.json"))
    for pf in product_files:
        with open(pf, 'r') as f:
            data = json.load(f)
            data = pd.json_normalize(data, max_level=0)
            products.append(data)
    products = pd.concat(products)
    all_products.append(products)

frag_data = pd.concat(all_products)

# COMMAND ----------

frag_data.columns = frag_data.columns.str.replace(" ", "_")

# COMMAND ----------

if "image_name" not in frag_data.columns:
    frag_data["image_name"] = None

# COMMAND ----------

for prod_name, image_url in frag_data[["name", "image"]].values:
    image_name = download_image(image_url, prod_name, FRAGRANTICA_IMAGE_DIR)
    insert_image_name_to_df(frag_data, prod_name, image_name)

# COMMAND ----------


create_or_insertoverwrite_table(
    spark.createDataFrame(frag_data.fillna("")),
    FRAGRANTICA_ATTRIBUTE.split(".")[0],
    FRAGRANTICA_ATTRIBUTE.split(".")[1],
    FRAGRANTICA_ATTRIBUTE.split(".")[2],
    ds_managed=True,
)

create_or_insertoverwrite_table(
    lc_data,
    LC_ATTRIBUTE.split(".")[0],
    LC_ATTRIBUTE.split(".")[1],
    LC_ATTRIBUTE.split(".")[2],
    ds_managed=True,
)

# COMMAND ----------


