# Databricks notebook source
# MAGIC %run "./generator_factory"

# COMMAND ----------

# MAGIC %run "../../Common/Utils/configuration"

# COMMAND ----------

# MAGIC %run "../../Common/Utils/data_source"

# COMMAND ----------

# MAGIC %run "../../Common/Utils/logger"

# COMMAND ----------

# MAGIC %run "../../Common/Utils/spark_func"

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import StringType, ArrayType, DoubleType
from pyspark.sql.utils import AnalysisException
import os
import re
import gc
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generator_factory import (
        create_similarity_generator_by_type,
        SimilarityConfiguration,
        SimilarityType,
    )
    from ..Utils.configuration import Configuration
    from ..Utils.data_source import DataSource
    from ..Utils.logger import get_logger
    from ..Utils.spark_func import compact_delta_table, write_temp_parquet
    from ..Utils.typings import *


# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("base_dir", "")
dbutils.widgets.text("bu_desc", "")
dbutils.widgets.text("similarities", "")
base_dir = getArgument("base_dir")
similarities = [s.strip() for s in getArgument("similarities").split(",")]
bu_desc = getArgument("bu_desc")
config = Configuration(base_dir, bu_desc)
data_source = DataSource(config)

logger = get_logger()

# COMMAND ----------

PRODUCT_ID_COL = "atg_code"
SOURCE_COL = "source"
SIM_COL = "sim"
SCORE_COL = "class_sim"
PARTITION_COLS = ["class_desc"]

# COMMAND ----------


def get_atg_code(p: str) -> str:
    return p.split("/")[-1].split("_")[0]


get_atg_code_udf = f.udf(get_atg_code, StringType())


def color_to_embedding(df: DataFrame, input_col: str) -> DataFrame:
    df = df.withColumn(
        input_col,
        f.struct(
            f.col("color").alias("colors"),
            f.col("count").cast("array<int>").alias("size"),
        ),
    )
    return df


def dict_to_list_normalized(input_dict: dict):
    input_dict = [value if value is not None else 0 for key, value in input_dict.items()]
    input_dict = np.linalg.norm(input_dict)
    return input_dict


dict_to_list_normalized_udf = udf(dict_to_list_normalized, ArrayType(DoubleType()))


def dict_to_vector(df: DataFrame, input_col: dict) -> DataFrame:
    df = df.withColumn(
        input_col,
        dict_to_list_normalized_udf(input_col)
    )
    return df


similarity_configs = {
    SimilarityType.DOMINANT_COLOR: SimilarityConfiguration(
        input_name="dominant_color",
        output_name="color_sim",
        input_col="embedding",
        preprocess_func=color_to_embedding,
    ),
    SimilarityType.DOMINANT_COLOR_V2: SimilarityConfiguration(
        input_name="dominant_color",
        output_name="dominant_color",
        input_col="embedding",
        preprocess_func=color_to_embedding,
    ),
    SimilarityType.WORD2VEC: SimilarityConfiguration(
        input_name="word2vec", output_name="word2vec", input_col="embedding"
    ),
    SimilarityType.RESNET: SimilarityConfiguration(
        input_name="resnet", output_name="resnet", input_col="embedding"
    ),
    SimilarityType.BERT: SimilarityConfiguration(
        input_name="bert", output_name="bert", input_col="embedding"
    ),
    SimilarityType.VIT: SimilarityConfiguration(
        input_name="vit", output_name="vit", input_col="embedding"
    ),
    SimilarityType.CONTOUR: SimilarityConfiguration(
        input_name="contour", output_name="contour", input_col="embedding"
    ),
    SimilarityType.SUBCLASS: SimilarityConfiguration(
        input_name="improved_class_hierarchy",
        output_name="subclass",
        input_col="ich_subclass_desc",
    ),
    SimilarityType.ASPECT_RATIO: SimilarityConfiguration(
        input_name="aspect_ratio", output_name="aspect_ratio", input_col="ratio"
    ),
    SimilarityType.KEYWORD: SimilarityConfiguration(
        input_name="keyword",
        output_name="keyword",
        input_col="keyword_list",
    ),
    SimilarityType.SEASON: SimilarityConfiguration(
        input_name="season",
        output_name="season",
        input_col="season_rating_vt",
        # preprocess_func=dict_to_vector,
    ),
    SimilarityType.ACCORD: SimilarityConfiguration(
        input_name="accord",
        output_name="accord",
        input_col="main_accords_vt",
        # preprocess_func=dict_to_vector,
    ),
    SimilarityType.NOTE: SimilarityConfiguration(
        input_name="note",
        output_name="note",
        input_col="middle_notes",
    ),
}


def read_feature(input_name: str) -> DataFrame:
    return spark.table(f"lc_dev.ml_house_of_fragrance_silver.intermediate_{input_name}")


def clean_string(string):
    return re.sub(r"[^a-z0-9]", "", string.lower())


im_table = config.get_filtered_item_master_table_name()
im = spark.table(im_table)[[PRODUCT_ID_COL] + PARTITION_COLS]

for similarity in similarities:
    similarity = similarity.upper()
    logger.info("===================================================")
    logger.info(similarity)

    similarity_type = SimilarityType[similarity.upper()]
    similarity_config = similarity_configs[similarity_type]
    feature_df = read_feature(similarity_config.input_name)
    feature_df = feature_df.join(im, how="inner", on=PRODUCT_ID_COL)
    if similarity_config.preprocess_func:
        feature_df = similarity_config.preprocess_func(
            feature_df, similarity_config.input_col
        )

    if "path" in feature_df.columns and PRODUCT_ID_COL not in feature_df.columns:
        # backward compatibility
        feature_df = feature_df.withColumn(
            PRODUCT_ID_COL, get_atg_code_udf(f.col("path"))
        )

    generator = create_similarity_generator_by_type(
        similarity_type,
        product_id_col=PRODUCT_ID_COL,
        input_col=similarity_config.input_col,
        source_col=SOURCE_COL,
        sim_col=SIM_COL,
        score_col=SCORE_COL,
    )

    def pre_group_callback(_generator, group_names):
        class_desc = clean_string(group_names[0])
        similarity_path = config.get_similarity_class_delta_path(
            similarity_config.output_name,
            class_desc
        )
        try:
            # materialize
            path = write_temp_parquet(spark.read.format("delta").load(similarity_path))
            anti_df = spark.read.parquet(path)
            anti_df = anti_df.withColumn("key", f.concat(f.col(SOURCE_COL), f.col(SIM_COL)))
            _generator.set_anti_pairs_df(anti_df)
        except AnalysisException:
            _generator.set_anti_pairs_df(None)

    def pre_chunk_callback(df, group_names):
        df = df.withColumn("key", f.concat(f.col(SOURCE_COL), f.col(SIM_COL)))
        return df

    def post_chunk_callback(df, group_names):
        class_desc = clean_string(group_names[0])
        similarity_path = config.get_similarity_class_delta_path(
            similarity_config.output_name,
            class_desc
        )
        df.write.mode("append").format("delta").save(similarity_path)

    def post_group_callback(_generator, group_names):
        class_desc = clean_string(group_names[0])
        similarity_path = config.get_similarity_class_delta_path(
            similarity_config.output_name,
            class_desc
        )
        compact_delta_table(similarity_path)
        gc.collect()
        spark.sparkContext._jvm.System.gc()
        spark.catalog.clearCache()
        spark.sql("CLEAR CACHE")

    generator.generate_similarity(
        feature_df, 
        partition_col=PARTITION_COLS,
        pre_group_callback=pre_group_callback,
        pre_chunk_callback=pre_chunk_callback,
        post_chunk_callback=post_chunk_callback,
        post_group_callback=post_group_callback,
    )

    logger.info(f"{similarity} completed")

# COMMAND ----------


# # Read the Parquet files into separate DataFrames
# keyword = spark.read.format("delta").load("/mnt/stg/house_of_fragrance/similar_product_engine/fragrance/intermediate/similarity_by_class/keyword/perfume").withColumnRenamed("class_sim", "keyword_sim")
# bert = spark.read.format("delta").load("/mnt/stg/house_of_fragrance/similar_product_engine/fragrance/intermediate/similarity_by_class/bert/perfume").withColumnRenamed("class_sim", "bert_sim")
# accord = spark.read.format("delta").load("/mnt/stg/house_of_fragrance/similar_product_engine/fragrance/intermediate/similarity_by_class/accord/perfume").withColumnRenamed("class_sim", "accord_sim")
# season = spark.read.format("delta").load("/mnt/stg/house_of_fragrance/similar_product_engine/fragrance/intermediate/similarity_by_class/season/perfume").withColumnRenamed("class_sim", "season_sim")
# note = spark.read.format("delta").load("/mnt/stg/house_of_fragrance/similar_product_engine/fragrance/intermediate/similarity_by_class/note/perfume").withColumnRenamed("class_sim", "note_sim")

# joined_df = keyword.join(bert, ["source","sim"]).join(accord, ["source","sim"]).join(season, ["source","sim"]).join(note, ["source","sim"])
# joined_df.write.format("parquet").save("/mnt/stg/house_of_fragrance/similar_product_engine/fragrance/sim_table/sim.parquet")
