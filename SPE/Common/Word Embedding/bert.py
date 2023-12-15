# Databricks notebook source
"""get bert encoding from ml_data_preproc_silver"""

# COMMAND ----------

# MAGIC %run "/utils/spark_utils"

# COMMAND ----------

import numpy as np
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, DoubleType
from pyspark.sql import Row

from SPE.Common.read_encoding import read_encoding

# COMMAND ----------

item_master = spark.table("lc_dev.ml_house_of_fragrance_silver.lc_fragrantica_matching")
items = np.array(item_master[["atg_code"]].collect()).flatten()
print("number of items: ", len(items))

# COMMAND ----------

bert_encoding, bert_encoding_id, _ = read_encoding(
    "lc_prd.ml_data_preproc_silver.encoding_bert",
    "/Volumes/lc_prd/ml_data_preproc_silver/encoding/bert",
    image_list=items,
    is_latest_only=True,
)
schema = StructType([
    StructField("atg_code", StringType(), nullable=False),
    StructField("embedding", ArrayType(DoubleType()), nullable=False)
])
rows = [
    Row(ids=string, encodings=row) for string, row in zip(
        bert_encoding_id.tolist(), bert_encoding.tolist()
    )
]
result = spark.createDataFrame(rows, schema=schema)
create_or_insertoverwrite_table(
    result,
    "lc_dev",
    "ml_house_of_fragrance_silver",
    "intermediate_bert",
    ds_managed=True,
)

