# Databricks notebook source
from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from typing import Protocol

class Reranker(Protocol):
    def rerank(self, rec: DataFrame) -> DataFrame:
        pass


# TODO: add priority ranking (subclass -> color)
class Ranker:
    def __init__(self, reranker: Reranker = None):
        self.reranker = reranker

    def rank(self, rec: DataFrame, persist: bool = True, limit: int = 10) -> DataFrame:
        rec = rec.withColumn(
            "rank",
            f.rank().over(Window.partitionBy("source").orderBy(f.desc("score_total"))),
        )
        if self.reranker:
            rec = self.reranker.rerank(rec)
        rec = rec.filter(f.col("rank") <= limit)
        if persist:
            rec = rec.repartition(512).persist(StorageLevel.MEMORY_ONLY)
        return rec
