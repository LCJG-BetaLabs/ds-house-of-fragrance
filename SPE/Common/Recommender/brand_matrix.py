# Databricks notebook source
# MAGIC %run "../Utils/configuration"

# COMMAND ----------

import pandas as pd
from pyspark.sql import DataFrame, functions as f
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.window import Window
from enum import Enum
from typing import TYPE_CHECKING, Dict, Set, List

if TYPE_CHECKING:
    from ..Utils.configuration import Configuration


class BrandAdjacency(Enum):
    SAME_BRAND = 1
    SIMILAR_BRAND = 2
    ANY_BRAND = 3


def _generate_positions(positions: List[BrandAdjacency], n):
    """
    Example:

    .. code-block:: python

        _generate_positions(['B', 'B', 'A', 'A', 'A'], 10)

    Output:

    .. code-block::

        ['B_00000001', 'B_00000002', 'A_00000001', 'A_00000002', 'A_00000003',
        'B_00000003', 'B_00000004', 'A_00000004', 'A_00000005', 'A_00000006']

    """
    output = []
    counts = {pos.name: 0 for pos in set(positions)}
    for i in range(n):
        pos = positions[i % len(positions)].name
        counts[pos] += 1
        output.append(pos + "_" + str(counts[pos]).zfill(8))
    remaining = {b.name for b in BrandAdjacency} - {b.name for b in positions}
    for b in remaining:
        output.extend([b + "_" + str(i).zfill(8) for i in range(1, n + 1)])
    return output


class BrandMatrix:
    """
    Usage:
    To be passed into :py:func:`Common.Recommender.ranker.Ranker`

    .. code-block:: python

        positions = [
            BrandAdjacency.SAME_BRAND,
            BrandAdjacency.SAME_BRAND,
            BrandAdjacency.SIMILAR_BRAND,
            BrandAdjacency.SIMILAR_BRAND,
            BrandAdjacency.SIMILAR_BRAND,
        ]
        brand_matrix = BrandMatrix(config, positions)
        ranker = Ranker(reranker=brand_matrix)
    """
    def __init__(
        self,
        config: "Configuration",
        positions: List[BrandAdjacency],
        is_brand_matrix_file_exist=True,
        num_similar_brands: int = 5,
    ):
        self.config = config
        self.num_similar_brands = num_similar_brands
        self.positions = positions
        self.is_brand_matrix_file_exist = is_brand_matrix_file_exist

    def get_brand_adjacency_udf(self):
        if self.is_brand_matrix_file_exist:
            path = self.config.get_brand_matrix_path()
            # columns: [region, bu_desc, brand_desc, similar_brand_1, similar_brand_2, ...]
            df = pd.read_csv(path)
            columns = [f"similar_brand_{i}" for i in range(1, self.num_similar_brands + 1)]
            # {brand: {similar_brand_1, similar_brand_2, ...}, ...}
            mapping: Dict[str, Set[str]] = (
                df.query(f"region == '{self.config.region}'")
                .query(f"bu_desc == '{self.config.bu_desc}'")
                .set_index("brand_desc")[columns]
                .apply(set, axis=1)
                .to_dict()
            )
        else:
          if BrandAdjacency.SIMILAR_BRAND in self.positions:
            raise ValueError("Brand matrix file does not exist, BrandAjacency.SIMILAR_BRAND cannot be used")
          else:
            mapping = {}

        @f.udf(returnType=StringType())
        def adjacency_udf(source_brand: str, sim_brand: str):
            if source_brand == sim_brand:
                return BrandAdjacency.SAME_BRAND.name
            if sim_brand in mapping.get(source_brand, set()):
                return BrandAdjacency.SIMILAR_BRAND.name
            return BrandAdjacency.ANY_BRAND.name

        return adjacency_udf

    def rerank(self, rec: DataFrame) -> DataFrame:
        # check brand adjacency for each pair
        brand_adjacency_udf = self.get_brand_adjacency_udf()
        rec = rec.withColumn(
            "bm_adjacency",
            brand_adjacency_udf(f.col("source_brand_desc"), f.col("sim_brand_desc")),
        )

        # groupby adjacency and rank by score in desc order
        rec = rec.withColumn(
            "bm_grouped_rank",
            f.row_number().over(
                Window.partitionBy("source", "bm_adjacency").orderBy(f.col("score_total").desc())
            ),
        )

        # rank according to positions
        get_key = f.udf(lambda t, r: t + "_" + str(r).zfill(8), StringType())
        rec = rec.withColumn("bm_key", get_key(f.col("bm_adjacency"), f.col("bm_grouped_rank")))

        n = rec.groupby("source").count().select(f.max(f.col("count"))).collect()[0][0]
        positions = _generate_positions(self.positions, n * len(self.positions))
        _p2rank = {p: i + 1 for i, p in enumerate(positions)}
        get_index = f.udf(lambda key: _p2rank[key], IntegerType())
        rec = rec.withColumn("rank", get_index(f.col("bm_key")))

        # tidy up rank
        rec = rec.withColumn(
            "rank",
            f.row_number().over(Window.partitionBy("source").orderBy(f.col("rank"))),
        )
        rec = rec.drop("bm_adjacency", "bm_grouped_rank", "bm_key")
        return rec
