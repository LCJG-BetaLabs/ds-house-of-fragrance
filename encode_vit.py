# Databricks notebook source
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typings import *
    from vit.transformer_vit import VitTransformer

from utils.enviroment import (
    FRAGRANTICA_IMAGE_DIR,
    VIT_MODEL_PATH,
    FRAGRANTICA_VIT_ENCODING_PATH,
    FRAGRANTICA_VIT_TABLE,
    FRAGRANTICA_ATTRIBUTE,
)

# COMMAND ----------

# MAGIC %pip install timm==0.4.12

# COMMAND ----------

# MAGIC %run ./vit/transformer_vit

# COMMAND ----------

# same vit model for lc images
model_name = "vit_base_patch16_224_miil_in21k"
# get the image list from attr table
attribute_table = spark.table(FRAGRANTICA_ATTRIBUTE)
image_list = attribute_table.select("image_name").toPandas().values
print(f"num images: {len(image_list)}")

# COMMAND ----------

# if prod is in existing records, don't encode again
if spark.catalog.tableExists(FRAGRANTICA_VIT_TABLE):
    spark.createDataFrame(
        pd.DataFrame(image_list, columns=["image_name"])
    ).createOrReplaceTempView("AllImages")

    image_list = spark.sql(f"""
SELECT a.image_name 
FROM AllImages a 
LEFT ANTI JOIN {FRAGRANTICA_VIT_TABLE} b USING (image_name)
""").toPandas().values.flatten()
print(f"data size: {len(image_list)}")

# COMMAND ----------

# encoding & push to feature store
transformer = VitTransformer(
    model_name=model_name,
    model_path=VIT_MODEL_PATH,
    image_path=FRAGRANTICA_IMAGE_DIR,
    encoding_path=FRAGRANTICA_VIT_ENCODING_PATH,
    id_name="image_name",
    table_name=FRAGRANTICA_VIT_TABLE,
)
transformer.transform(
    id_list=image_list,
    partition_size=4000,
    batch_size=512,
)
