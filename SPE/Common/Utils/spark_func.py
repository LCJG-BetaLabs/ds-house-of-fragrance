# Databricks notebook source
# MAGIC %run "./logger"

# COMMAND ----------

import glob
import json
import time
import tempfile
from functools import reduce
from pyspark.sql import functions as f
from pyspark.sql import DataFrame, Window
from pyspark.sql.utils import IllegalArgumentException
from typing import TYPE_CHECKING, List, Tuple, Generator


if TYPE_CHECKING:
    from .logger import get_logger
    from .typings import *

logger = get_logger()


def iterate_groups(
    df: DataFrame,
    groupby_columns: List[str] = None,
) -> Generator[Tuple[List[str], DataFrame], None, None]:
    """
    Groupby df and iterate over the groups.
    Same as pandas version of
    ```for group_name, group in df.groupby():```
    """
    groupby_columns = groupby_columns or []  # if None, make it []
    rows = df.select(*groupby_columns).distinct().collect()
    for row in rows:
        group = df
        for k, v in row.asDict().items():
            group = group.filter(f.col(k) == v)
        yield list(row), group


def _add_row_number(df: DataFrame) -> DataFrame:
    df = df.withColumn("_id", f.monotonically_increasing_id()) # this will skip numbers
    w = Window().orderBy("_id")
    df = df.withColumn("id", (f.row_number().over(w))).drop('_id')
    return df


def cross_join_by_chunks(
    left: DataFrame,
    right: DataFrame = None,
    chunk_size: int = 4000,
    repartition: int = None,
    left_repartition_col: str = None,
    right_repartition_col: str = None,
):
    """
    :param chunk_size: size of one chunk before cross join
    """
    if not right:
        right = left.alias("")

    left_size = left.count()
    right_size = right.count()

    left = _add_row_number(left)
    right = _add_row_number(right)

    for i0 in range(0, left_size, chunk_size):
        i1 = min(i0 + chunk_size, left_size)
        
        for j0 in range(0, right_size, chunk_size):
            j1 = min(j0 + chunk_size, right_size)
            
            l = left.filter(f.col("id").between(i0, i1 - 1)).drop("id")  # end inclusive
            r = right.filter(f.col("id").between(j0, j1 - 1)).drop("id")  # end inclusive
            
            if repartition:
                l = l.repartition(repartition, left_repartition_col)
                r = r.repartition(repartition, right_repartition_col)

            l = f.broadcast(l)
            r = f.broadcast(r)
            
            yield l.crossJoin(r)


def _count_parquet_files(path: str):
    return len(glob.glob("/dbfs" + path + "/*.parquet"))


def compact_delta_table(table_path: str, threshold: int = 4):
    """
    :param table_path: path to the delta table
    :param threshold: only compact if the delta table has more parquet files than (or equal to) this number
    """
    num_files = _count_parquet_files(table_path)
    logger.info(f"Compacting {table_path} ({num_files} files)")
    if num_files < threshold:
        return

    _df = spark.sql(f"""OPTIMIZE delta.`{table_path}`;""")
    logger.info(f"Optimized table {table_path} ({_count_parquet_files(table_path)} files)")
    logger.info(f"Logs: {json.loads(_df.toJSON().first())}")
    time.sleep(5)

    spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
    try:
        spark.sql(f"""VACUUM delta.`{table_path}` RETAIN 0 HOURS;""")
        logger.info(f"Vacuumed table {table_path} ({_count_parquet_files(table_path)} files)")
    except IllegalArgumentException:
        logger.exception(f"Failed to vacuum table")


def write_temp_parquet(df):
    path = tempfile.mkdtemp()
    df.write.parquet(path)
    return path


def union_read(paths: List[str], file_type: str = "parquet"):
    dfs = [spark.read.format(file_type).load(p) for p in paths]
    return reduce(lambda a, b: a.union(b), dfs)
