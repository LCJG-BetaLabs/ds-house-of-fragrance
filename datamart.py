# Databricks notebook source
def get_product_feed():
    spark.sql(
        """CREATE
        OR REPLACE TEMPORARY VIEW ProductFeed AS
        SELECT
            sku AS atg_code,
            region,
            stockLevel AS stock_level,
            price
        FROM
            lc_prd.api_product_feed_silver.lc_product_feed 
        WHERE
            load_date = (SELECT MAX(load_date) FROM lc_prd.api_product_feed_silver.lc_product_feed)
        """
    )
    stock = spark.sql(
        """
        SELECT
            atg_code,
            INT(IFNULL(stock_level_cn, 0)) AS stock_level_cn,
            INT(IFNULL(stock_level_hk, 0)) AS stock_level_hk,
            INT(IFNULL(stock_level_row, 0)) AS stock_level_row
        FROM (
            SELECT atg_code, region, stock_level
            FROM ProductFeed
        )
        PIVOT (
            FIRST_VALUE(stock_level)
            FOR region
            IN ("cn" AS stock_level_cn, "hk" AS stock_level_hk, "row" AS stock_level_row)
        )"""
    )
    price = spark.sql(
        """
        SELECT
            atg_code,
            INT(IFNULL(price_cn, 0)) AS price_cn,
            INT(IFNULL(price_hk, 0)) AS price_hk,
            INT(IFNULL(price_row, 0)) AS price_row
        FROM (
            SELECT atg_code, region, price
            FROM ProductFeed
        )
        PIVOT (
            FIRST_VALUE(price)
            FOR region
            IN ("cn" AS price_cn, "hk" AS price_hk, "row" AS price_row)
        )
        """
    )
    product_feed = stock.join(price, on="atg_code", how="inner")
    return product_feed

# COMMAND ----------

product_feed = get_product_feed()

# COMMAND ----------

def get_lc_perfume_data(product_feed):
    lc_data = spark.sql(
        """
        SELECT 
            atg_code,
            style,
            color_desc,
            prod_desc_eng,
            prod_desc_tc,
            brand_desc,
            category_desc,
            class_desc,
            subclass_desc,
            display_name,
            long_desc,
            care,
            img_list
        FROM lc_prd.ml_data_preproc_silver.attribute
        WHERE class_desc = "Perfume"
        """
    )
    lc_data = lc_data.join(product_feed, on="atg_code", how="inner")
    return lc_data

# COMMAND ----------

lc_data = get_lc_perfume_data(product_feed)

# COMMAND ----------

display(lc_data)

# COMMAND ----------

# get fragrantica data
import os

fragrantica_path = "/dbfs/mnt/stg/house_of_fragrance/fragrantica_scraping/"
brand_folder = [f for f in os.listdir(fragrantica_path) if os.path.isdir(os.path.join(fragrantica_path, f))]

# COMMAND ----------

import glob
import json

import pandas as pd

# COMMAND ----------

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

# COMMAND ----------

frag_data = pd.concat(all_products)

# COMMAND ----------

frag_data

# COMMAND ----------

# download image
import requests
import http.client
import time


image_dir = "/dbfs/mnt/stg/house_of_fragrance/fragrantica_images"
os.makedirs(image_dir, exist_ok=True)


def get_image_name(product_name, image_url):
    _id = image_url.split(".")[-2]
    product_name = product_name.replace(" ", "_")
    product_name = product_name.replace("/", "_")
    return f"{product_name}__{_id}.jpg"


def _retry_get_request(url, tries: int=-1, delay: float=0, max_delay: float=None, backoff: int=1):
    """
    Retries `requests.get(url)` automatically.
    Copied from `retry` package: https://github.com/invl/retry

    :param url:
    :param tries: the maximum number of attempts. default: -1 (infinite).
    :param delay: initial delay between attempts. default: 0.
    :param max_delay: the maximum value of delay. default: None (no limit).
    :param backoff: multiplier applied to delay between attempts. default: 1 (no backoff).
    :returns: the response from url
    """
    while tries:
        try:
            return requests.get(url, stream=True)
        except (requests.RequestException, http.client.IncompleteRead):
            tries -= 1
            if not tries:
                raise

            time.sleep(delay)
            delay *= backoff

            if max_delay is not None:
                delay = min(delay, max_delay)


def download_image(image_url, product_name, image_dir):
    image_name = get_image_name(product_name, image_url)
    dest_path = os.path.join(image_dir, image_name)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return

    with _retry_get_request(image_url, tries=5, delay=1, max_delay=10, backoff=2) as r:
        if r.status_code != 200:
            return

        r.raw.decode_content = True
        content = r.raw.read()
        if len(content) == 0:
            return

        with open(dest_path, 'wb') as f:
            f.write(content)
    return image_name

    

# COMMAND ----------

if "image_name" not in frag_data.columns:
    frag_data["image_name"] = None

# COMMAND ----------

def insert_image_name_to_df(df, prod_name, image_name):
    df.loc[df["name"] == prod_name, "image_name"] = image_name

# COMMAND ----------

for prod_name, image_url in frag_data[["name", "image"]].values:
    image_name = download_image(image_url, prod_name, image_dir)
    insert_image_name_to_df(frag_data, prod_name, image_name)

# COMMAND ----------

spark.createDataFrame(frag_data).write.parquet("/mnt/stg/house_of_fragrance/fragrantica_attribute.parquet")

# COMMAND ----------


