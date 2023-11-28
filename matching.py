# Databricks notebook source
# MAGIC %run "/utils/spark_utils"

# COMMAND ----------

import numpy as np
from vit.utils import read_encoding
from utils.enviroment import (
    LC_ATTRIBUTE,
    FRAGRANTICA_ATTRIBUTE,
    LC_VIT_TABLE,
    LC_VIT_ENCODING_PATH,
    FRAGRANTICA_VIT_TABLE,
    FRAGRANTICA_VIT_ENCODING_PATH,
)

# COMMAND ----------

# get lc product vit embedding
lc_attr = spark.table(LC_ATTRIBUTE)
# use image name as id
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
