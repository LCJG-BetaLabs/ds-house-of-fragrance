# Databricks notebook source
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typings import *
    from vit.transformer_vit import VitTransformer

from utils.enviroment import FRAGRANTICA_IMAGE_DIR

# COMMAND ----------

# MAGIC %run "./vit/transformer_vit"

# COMMAND ----------

# same vit model for lc images
model_name = "vit_base_patch16_224_miil_in21k"
model_path = "/Volumes/lc_prd/ml_data_preproc_silver/model/timm/vit_base_patch16_224_in21k_miil.pth" # TODO: use HOF volume
encoding_path = "/dbfs/mnt/stg/house_of_fragrance/encoding/vit/fragrantica/encoding" # TODO: use HOF volume

# feature store db & table
db_table_name = f"lc_dev.ml_house_of_fragrance.fragrantica_encoding_vit"

# COMMAND ----------

# get the image list from lc attr table
attribute_table = spark.read.parquet("/mnt/stg/house_of_fragrance/fragrantica_attribute.parquet")
# spark.table("lc_dev.ml_house_of_fragrance.fragrantica_attribute")

image_list = attribute_table.select("image_name").toPandas().values
print(f"num images: {len(image_list)}")

# COMMAND ----------

# if atg is in existing records, don't encode again
if spark.catalog.tableExists(db_table_name):
    spark.createDataFrame(
        pd.DataFrame(image_list, columns=["image_name"])
    ).createOrReplaceTempView("AllImages")

    image_list = spark.sql(f"""
SELECT a.image_name 
FROM AllImages a 
LEFT ANTI JOIN {db_table_name} b USING (image_name)
""").toPandas().values.flatten()
print(f"data size: {len(image_list)}")

# COMMAND ----------

# encoding & push to feature store
transformer = VitTransformer(
    model_name=model_name,
    model_path=model_path,
    image_path=FRAGRANTICA_IMAGE_DIR,
    encoding_path=encoding_path,
    id_name="image_name",
    table_name=db_table_name,
)
transformer.transform(
    id_list=image_list,
    partition_size=4000,
    batch_size=512,
)

# COMMAND ----------
