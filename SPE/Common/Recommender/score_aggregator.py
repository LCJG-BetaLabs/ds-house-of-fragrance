# Databricks notebook source
# MAGIC %run "./score"

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql import DataFrame
from functools import reduce
from typing import TYPE_CHECKING, List

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

if TYPE_CHECKING:
    from .score import Score
    from ..Utils.configuration import Configuration

SOURCE_COLUMN_NAME = "source"
SIM_COLUMN_NAME = "sim"
TOTAL_SCORE_COLUMN_NAME = "score_total"


class ScoreAggregator:
    def __init__(self, config: "Configuration" = None):
        self.scores: List[Score] = []
        self.weights: List[float] = []
        self.names: List[str] = []
        self.column_names: List[str] = []
        self.config = config

    def __repr__(self):
        r = "ScoreAggregator({"
        r += ", ".join(
            [
                f'"{name}": {weight:.2f}'
                for name, weight in zip(self.names, self.weights)
            ]
        )
        r += "})"
        return r

    def set_config(self, config: "Configuration") -> Self:
        self.config = config
        return self

    def add_score(self, score: Score, weight: float) -> Self:
        """
        :param score: a Score instance
        :param weight: between 0 and 1
        """
        if weight <= 0 or weight > 1:
            raise ValueError(f"Invalid weight: {weight}, must be between [0, 1).")
        if score.name in self.names:
            raise ValueError(f"Score '{score.name}' is already added.")
        self.scores.append(score)
        self.weights.append(weight)
        self.names.append(score.name)
        self.column_names.append("score_" + score.name)
        return self

    def add_score_from_similarity_type(
        self,
        similarity_type: str,
        weight: float,
        file_type: str = "delta",
        name: str = None,
        normalize: bool = False,
    ) -> Self:
        """

        :param similarity_type:
        :param weight: between 0 and 1
        :param file_type:
        :param name: if None, will use `similarity_type` instead
        :param normalize:
        :return:
        """
        if not self.config:
            raise ValueError(f"Config not set. Use set_config() to set config.")
        score = Score.from_similarity_type(
            similarity_type=similarity_type,
            config=self.config,
            file_type=file_type,
            name=name,
            normalize=normalize,
        )
        self.add_score(score, weight)
        return self

    def _process_scores(self) -> List[DataFrame]:
        """Process score dataframe and weight the scores"""
        score_dfs = []
        for score, weight in zip(self.scores, self.weights):
            score_df = score.process_df()
            score_df = score_df.withColumn(
                score.name + "_weighted", (f.col(score.name) * weight).cast("float")
            )
            score_dfs.append(score_df)
        return score_dfs

    def aggregate(self) -> DataFrame:
        if not self.scores:
            raise Exception("No scores added.")

        score_dfs = self._process_scores()
        result_df = reduce(
            lambda a, b: a.join(
                b, how="outer", on=[SOURCE_COLUMN_NAME, SIM_COLUMN_NAME]
            ),
            score_dfs,
        )
        result_df = result_df.fillna(0).withColumn(
            TOTAL_SCORE_COLUMN_NAME,
            sum(f.col(name + "_weighted") for name in self.names),
        )
        for name, column_name in zip(self.names, self.column_names):
            result_df = result_df.withColumnRenamed(name, column_name)
        result_df = result_df.select(
            SOURCE_COLUMN_NAME,
            SIM_COLUMN_NAME,
            TOTAL_SCORE_COLUMN_NAME,
            *self.column_names,
        )
        return result_df
