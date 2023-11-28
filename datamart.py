# Databricks notebook source
import os
import glob
import json
import pandas as pd

from utils.get_data import get_product_feed, get_lc_perfume_data
from utils.download_image import download_image
from utils.enviroment import FRAGRANTICA_IMAGE_DIR


def insert_image_name_to_df(df, prod_name, image_name):
    df.loc[df["name"] == prod_name, "image_name"] = image_name


# COMMAND ----------

product_feed = get_product_feed()
lc_data = get_lc_perfume_data(product_feed)

# COMMAND ----------

# get fragrantica data
fragrantica_path = "/dbfs/mnt/stg/house_of_fragrance/fragrantica_scraping/"
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

if "image_name" not in frag_data.columns:
    frag_data["image_name"] = None

# COMMAND ----------

for prod_name, image_url in frag_data[["name", "image"]].values:
    image_name = download_image(image_url, prod_name, FRAGRANTICA_IMAGE_DIR)
    insert_image_name_to_df(frag_data, prod_name, image_name)

# COMMAND ----------

# TODO: use uc table
spark.createDataFrame(frag_data).write.parquet(
    "/mnt/stg/house_of_fragrance/fragrantica_attribute.parquet",
    mode="overwrite",
)

lc_data.write.parquet(
    "/mnt/stg/house_of_fragrance/lc_perfume_attribute.parquet",
    mode="overwrite",
)

# COMMAND ----------


