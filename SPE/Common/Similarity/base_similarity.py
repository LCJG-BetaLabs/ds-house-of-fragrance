# Databricks notebook source
# MAGIC %run "../Utils/spark_func"

# COMMAND ----------

# MAGIC %run "../Utils/logger"

# COMMAND ----------

from abc import ABC, abstractmethod
from pyspark.sql import functions as f
from pyspark.sql import DataFrame
from pyspark.sql.udf import UserDefinedFunction
from typing import TYPE_CHECKING, Callable, Optional, List
from typing_extensions import Self


if TYPE_CHECKING:
    from ..Utils.spark_func import (
        iterate_groups,
        cross_join_by_chunks,
        write_temp_parquet,
        union_read,
    )
    from ..Utils.logger import get_logger
    from ..Utils.typings import *

logger = get_logger()


class BaseSimilarityGenerator(ABC):
    def __init__(
        self,
        product_id_col: str,
        input_col: str,
        source_col: str = "source",
        sim_col: str = "sim",
        score_col: str = "class_sim",
    ):
        """
        :param product_id_col:
        :param input_col:
        :param source_col:
        :param sim_col:
        :param score_col:
        """
        self.product_id_col = product_id_col
        self.input_col = input_col
        self.source_col = source_col
        self.sim_col = sim_col
        self.score_col = score_col
        self._has_stock_df: Optional[DataFrame] = None
        self._pairs_df: Optional[DataFrame] = None
        self._anti_pairs_df: Optional[DataFrame] = None
        self._similarity_df: Optional[DataFrame] = None

    @abstractmethod
    def create_similarity_udf(self) -> UserDefinedFunction:
        raise NotImplementedError

    def generate_similarity(
        self,
        df: DataFrame,
        partition_col: List[str] = None,
        filter_func: Callable = None,
        pre_group_callback: Callable[[Self, List[str]], None] = None,
        post_group_callback: Callable[[Self, List[str]], None] = None,
        pre_chunk_callback: Callable[[DataFrame, List[str]], DataFrame] = None,
        post_chunk_callback: Callable[[DataFrame, List[str]], None] = None,
    ):
        """
        :param df:
        :param partition_col: if specified, only generate similarity within the same partition
        :param filter_func:
        :param pre_group_callback: called before processing each group on the dataframe
            (Self, group_names)
        :param post_group_callback: called after processing each group on the dataframe
            (Self, group_names)
        :param pre_chunk_callback: called after processing each chunk on the dataframe
        :param post_chunk_callback: called after processing each chunk on the dataframe
        :return:
        """
        temp_paths = []
        for group_names, group in iterate_groups(df, partition_col):
            source_input_col = self.input_col + "_1"
            sim_input_col = self.input_col + "_2"
            source_df = group.select(self.product_id_col, self.input_col).toDF(
                self.source_col, source_input_col
            )
            sim_df = group.select(self.product_id_col, self.input_col).toDF(
                self.sim_col, sim_input_col
            )

            # filter OOS
            sim_df = self._filter_out_of_stock(sim_df, self.sim_col)

            if pre_group_callback is not None:
                pre_group_callback(self, group_names)

            # cross join
            for i, result_df in enumerate(cross_join_by_chunks(
                source_df,
                sim_df,
                chunk_size=4500,
                repartition=128,
                left_repartition_col=self.source_col,
                right_repartition_col=self.sim_col,
            )):
                logger.info(f"Processing group '{group_names}' chunk #{i} (size={result_df.count()})")

                if pre_chunk_callback is not None:
                    result_df = pre_chunk_callback(result_df, group_names)

                # join pairs
                result_df = self._join_pairs_df(result_df)

                # join anti df
                result_df = self._join_anti_pairs_df(result_df)

                if result_df.count() == 0:
                    continue

                # generate similarity
                similarity_func = self.create_similarity_udf()
                result_df = result_df.withColumn(
                    self.score_col,
                    similarity_func(f.col(source_input_col), f.col(sim_input_col)),
                )
                result_df = result_df.select(self.source_col, self.sim_col, self.score_col)
                if filter_func is not None:
                    result_df = filter_func(result_df)

                if result_df.count() == 0:
                    continue

                path = write_temp_parquet(result_df)
                temp_paths.append(path)

                if post_chunk_callback is not None:
                    post_chunk_callback(result_df, group_names)

            if post_group_callback is not None:
                post_group_callback(self, group_names)

        if len(temp_paths) > 0:
            df = union_read(temp_paths)
            path = write_temp_parquet(df)
            self._similarity_df = spark.read.parquet(path)
        return self._similarity_df

    def _filter_out_of_stock(self, df: DataFrame, column: str):
        if self._has_stock_df:
            has_stock_df = (
                self._has_stock_df.select(self.product_id_col)
                .toDF(column)
                .drop_duplicates(subset=[column])
            )
            df = df.join(has_stock_df, on=column, how="inner")
        return df

    def _join_pairs_df(self, df):
        if self._pairs_df:
            df = df.join(
                self._pairs_df, how="inner", on=[self.sim_col, self.source_col]
            )
        return df

    def _join_anti_pairs_df(self, df):
        if self._anti_pairs_df:
            df = df.join(
                self._anti_pairs_df, how="left_anti", on="key"
            )
        return df

    def set_has_stock_df(self, has_stock_df: Optional[DataFrame]):
        self._has_stock_df = has_stock_df

    def set_pairs_df(self, pairs_df: Optional[DataFrame]):
        self._pairs_df = pairs_df

    def set_anti_pairs_df(self, anti_pairs_df: Optional[DataFrame]):
        """
        :param anti_pairs_df: DataFrame containing pairs [`source_col`, `sim_col`] to NOT generate similarity on (anti join)
        """
        self._anti_pairs_df = anti_pairs_df

    def write_similarity_to_delta(self, path):
        if self._similarity_df is not None and self._similarity_df.count() > 0:
            self._similarity_df.write.mode("append").format("delta").save(path)
