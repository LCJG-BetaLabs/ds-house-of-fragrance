# Databricks notebook source
import os
import numpy as np
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
import pyspark.sql.functions as f
from utils.enviroment import BASE_DIR

path = "/mnt/stg/house_of_fragrance/similar_product_engine/fragrance/sim_table"
sim_table = spark.read.parquet(path)
display(sim_table)

weights = [0.4, 0.3, 0.1, 0.2]


@udf(returnType=FloatType())
def calculate_weighted_score(col1, col2, col3, col4):
    return col1 * weights[0] + col2 * weights[1] + col3 * weights[2] + col4 * weights[3]


def rank_sim(item_list, represent):
    df = pd.DataFrame(item_list, columns=["source"])
    df["sim"] = represent
    df = spark.createDataFrame(df)
    df = df.join(sim_table.select("source", "sim", "weighted_score"), how="inner", on=["source", "sim"])
    return df


sim_table = sim_table.withColumn(
    "weighted_score",
    calculate_weighted_score(f.col("bert_sim"), f.col("accord_sim"), f.col("season_sim"), f.col("note_sim"))
)

cluster_output = pd.read_csv(os.path.join(BASE_DIR, "output.csv"))

all_rank = []
for cluster in np.unique(cluster_output["cluster"]):
    subdf = cluster_output[cluster_output["cluster"] == cluster]
    rep = np.random.choice(subdf[subdf["rank"] == 1]["atg_code"].values, 1)
    rank = rank_sim(subdf[["atg_code"]].values, rep).toPandas()
    rank = rank.sort_values(by="weighted_score", ascending=False).reset_index(name="rank")
    all_rank.append(rank)

all_rank = pd.concat(all_rank).rename(columns={"source": "atg_code"})[["atg_code", "rank"]]
cluster_output = cluster_output.merge(all_rank, how="inner", on="atg_code")
display(cluster_output)
