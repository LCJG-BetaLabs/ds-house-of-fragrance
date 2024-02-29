# Databricks notebook source
# MAGIC %run "../Utils/configuration"

# COMMAND ----------

# MAGIC %run "../Utils/data_source"

# COMMAND ----------

# MAGIC %run "./score_aggregator"

# COMMAND ----------

# MAGIC %run "./filters"

# COMMAND ----------

# MAGIC %run "./default_filters"

# COMMAND ----------

# MAGIC %run "./ranker"

# COMMAND ----------

# MAGIC %run "./export"

# COMMAND ----------

import os
import pyspark.sql
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from typing import List, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    spark = pyspark.sql.SparkSession.getActiveSession()
    from ..Utils.configuration import Configuration
    from ..Utils.data_source import DataSource
    from .score_aggregator import ScoreAggregator
    from .filters import FilterRunner, FilterType
    from .ranker import Ranker
    from .export import export_json_by_class, export_html_by_class
    from .default_filters import (
        add_conflicting_subclass_filter,
        add_product_tagging_filter,
    )


# COMMAND ----------

def _read_item_master(config: "Configuration") -> DataFrame:
    """read item master with out of stock items removed"""
    return (
        spark.table(config.get_filtered_item_master_table_name())
        # .withColumnRenamed(f"stock_level_{config.region}", "SOH")
        # .withColumnRenamed(f"price_{config.region}", "price")
        # .filter(col("SOH") > 0)
    )


def recommend(
    config: "Configuration",
    data_source: "DataSource",
    score_aggregator: ScoreAggregator,
    filter_runner: FilterRunner,
    ranker: Ranker,
    filter_item_master: Callable[[DataFrame], DataFrame]=None,
    recommendations_per_item: int = 10,
    overwrite_outputs: bool = True,
    tagging_dimensions: List[str] = None
):
    # TODO: Probably need a better way to add these filters
    # add remaining default filters
    add_product_tagging_filter(tagging_dimensions)
    add_conflicting_subclass_filter()
    add_price_point_filter(config)
    add_persona_filter(config)
    
    # read item master
    item_master = _read_item_master(config)

    # filter item master
    if filter_item_master is not None:
        item_master = filter_item_master(item_master)

    if item_master.count() == 0:
        print("Empty item master, exiting function")
        return
    
    # aggregate scores
    aggregated_score: DataFrame = score_aggregator.aggregate()

    # run filters
    filter_runner.with_item_master(item_master).run_filter_rules(aggregated_score)

    # run ranking
    required_ranked_score: DataFrame = ranker.rank(
        filter_runner.get_filtered_df(FilterType.REQUIRED_FILTER),
        limit=recommendations_per_item,
    )
    deploy_ranked_score: DataFrame = ranker.rank(
        filter_runner.get_filtered_df(FilterType.DEPLOY_FILTER),
        limit=recommendations_per_item,
    )

    export_json_by_class(
        rec=required_ranked_score,
        output_dir=config.get_output_dir(deploy=False),
        region=config.region,
        score_columns=score_aggregator.column_names,
        overwrite_outputs=overwrite_outputs,
    )
    export_json_by_class(
        rec=deploy_ranked_score,
        output_dir=config.get_output_dir(deploy=True),
        region=config.region,
        score_columns=score_aggregator.column_names,
        overwrite_outputs=overwrite_outputs
    )

    export_html_by_class(
        output_root=config.get_output_root(),
        output_dir=config.get_output_dir(deploy=False),
        bu_desc=config.bu_desc,
        region=config.region,
        simtypes=score_aggregator.column_names,
        repos_root=os.path.abspath(".."),
    )
