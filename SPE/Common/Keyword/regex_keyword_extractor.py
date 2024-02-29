# Databricks notebook source
# MAGIC %run "/utils/spark_utils"

# COMMAND ----------

import re
from pyspark.sql.functions import udf
import pyspark.sql.functions as f
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql import DataFrame
from typing import Callable, List


class RegexKeywordExtractor:
    def __init__(
        self,
        item_master: DataFrame,
        column: str,
        preproc: Callable,
        patterns: List[str],
        postproc: Callable,
    ):
        self.item_master = item_master
        self.column = column
        self.preproc = preproc
        self.patterns = patterns
        self.postproc = postproc
        self.COLUMN_NAME = "keyword_list"

    def get_extract_keywords_udf(self):
        preproc = self.preproc
        patterns = self.patterns
        postproc = self.postproc

        @udf(returnType=ArrayType(StringType()))
        def extract_keywords(string: str) -> List[str]:
            """keyword udf"""
            string = preproc(string)
            keyword_list = []
            for pattern in patterns:
                matches = re.findall(pattern, string)
                keyword_list.extend(set(matches))
            return postproc(keyword_list)

        return extract_keywords

    def save_similarity_table(self, output_table: str = None):
        """save similarity dataframe to uc table"""
        extract_keywords = self.get_extract_keywords_udf()
        result = self.item_master.withColumn(
            self.COLUMN_NAME, extract_keywords(f.col(self.column))
        )
        result = result.select("atg_code", "keyword_list")
        create_or_insertoverwrite_table(
            result, 
            output_table.split(".")[0], 
            output_table.split(".")[1], 
            output_table.split(".")[2],
            ds_managed=True,
        )
