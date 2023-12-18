# Databricks notebook source
# MAGIC %run "../Utils/spark_func"

# COMMAND ----------

import glob

import pyspark.sql
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..Utils.configuration import Configuration
    from ..Utils.spark_func import union_read

    spark = pyspark.sql.SparkSession.getActiveSession()

SOURCE_COLUMN_NAME = "source"
SIM_COLUMN_NAME = "sim"
SCORE_COLUMN_NAME = "score"


def normalize_column(df: DataFrame, column: str) -> DataFrame:
    """MinMax normalize a float column in df. Maintains the columns in df."""
    if df.count() == 0:
        return df
    assembler = VectorAssembler().setInputCols([column]).setOutputCol("features")
    df = assembler.transform(df)
    scaler = MinMaxScaler(inputCol="features", outputCol="output")
    model = scaler.fit(df)
    df = model.transform(df)
    df = df.withColumn(column, vector_to_array(f.col("output")).getItem(0))
    df = df.drop("features", "output")
    return df


class Score:
    def __init__(self, name: str, df: DataFrame, normalize: bool = False):
        self.name = name
        self.df = df
        self.normalize = normalize

    def process_df(self) -> DataFrame:
        """
        :return: columns: ["source", "sim", self.name]
        """
        df = self.df.select(SOURCE_COLUMN_NAME, SIM_COLUMN_NAME, SCORE_COLUMN_NAME)
        if self.normalize:
            df = normalize_column(df, SCORE_COLUMN_NAME)
        df = df.withColumnRenamed(SCORE_COLUMN_NAME, self.name)
        return df

    @classmethod
    def from_similarity_type(
        cls,
        similarity_type: str,
        config: "Configuration",
        file_type: str = "delta",
        name: str = None,
        normalize: bool = False,
    ):
        """

        :param similarity_type:
        :param config:
        :param file_type:
        :param name: if None, will use `similarity_type` instead
        :param normalize:
        :return:
        """
        pattern = config.get_similarity_class_delta_path(similarity_type, "*", local=True)
        paths = [p.replace("/dbfs", "") for p in glob.glob(pattern)]
        df = union_read(paths, file_type)
        df = df.dropDuplicates(subset=["source", "sim"]) # hard code fix duplicate row issue in similarity df
        df = df.withColumnRenamed("class_sim", SCORE_COLUMN_NAME)
        return cls(name or similarity_type, df, normalize)
