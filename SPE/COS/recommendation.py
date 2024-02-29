# Databricks notebook source
# MAGIC %run "../Common/Recommender/filters"

# COMMAND ----------

# MAGIC %run "../Common/Recommender/ranker"

# COMMAND ----------

# MAGIC %run "../Common/Recommender/recommend"

# COMMAND ----------

# MAGIC %run "../Common/Recommender/score_aggregator"

# COMMAND ----------

import pyspark.sql
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    spark = pyspark.sql.SparkSession.getActiveSession()
    from Common.Utils.configuration import Configuration
    from Common.Utils.data_source import DataSource
    from Common.Recommender.score_aggregator import ScoreAggregator
    from Common.Recommender.filters import (
        FilterRunner,
        FilterType,
        register_filter_udf,
        register_filter,
    )
    from Common.Recommender.recommend import recommend
    from Common.Recommender.ranker import Ranker

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("base_dir", "")
dbutils.widgets.text("bu_desc", "")
dbutils.widgets.text("region", "")
dbutils.widgets.text("tagging_dimensions", "")
base_dir = getArgument("base_dir")
bu_desc = getArgument("bu_desc")
region = getArgument("region")
tagging_dimensions = getArgument("tagging_dimensions").split(",") if getArgument("tagging_dimensions") != "" else None
config = Configuration(base_dir, bu_desc=bu_desc, region=region)
data_source = DataSource(config)

# COMMAND ----------

# add filter args udf
# @register_filter_udf(name="similar_price")
def similar_price_udf(source_price: Column, sim_price: Column) -> Column:
    return (
        (0.5 <= f.abs(sim_price / source_price - 1))
        & (f.abs(sim_price / source_price - 1) <= 2)
    ).cast("int")

# COMMAND ----------

def recommend_fragrance(overwrite_outputs=False):
    # add rules
    @register_filter(name="total_score_threshold", type=FilterType.REQUIRED_FILTER)
    def total_score_threshold():
        return (f.col("score_total") >= 0.2).cast("int")

    def fragrance_filter_item_master(item_master: DataFrame) -> DataFrame:
        item_master = item_master.filter(item_master.category_desc == "Fragrance")
        return item_master

    fragrance_score_aggregator = (
        ScoreAggregator()
        .set_config(config)
        .add_score_from_similarity_type("bert", weight=0.4)
        # .add_score_from_similarity_type("keyword", weight=0.6)
        .add_score_from_similarity_type("accord", weight=0.3)
        .add_score_from_similarity_type("season", weight=0.1)
        .add_score_from_similarity_type("note", weight=0.2)

    )

    filter_runner = FilterRunner()
    ranker = Ranker()

    recommend(
        config=config,
        data_source=data_source,
        score_aggregator=fragrance_score_aggregator,
        filter_runner=filter_runner,
        ranker=ranker,
        filter_item_master=fragrance_filter_item_master,
        overwrite_outputs=overwrite_outputs,
        tagging_dimensions=tagging_dimensions,
    )

# COMMAND ----------

recommend_fragrance(overwrite_outputs=False)
