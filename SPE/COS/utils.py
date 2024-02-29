# Databricks notebook source
from os import path
import csv
import shutil


def save_df_to_single_csv(df, path):
    tmp_dir = path + ".tmp"
    df.repartition(1).write.mode("overwrite").format("com.databricks.spark.csv").option(
        "header", True
    ).option("multiLine", True).option("quote", '"').option("escape", '"').option(
        "quoteAll", True
    ).save(
        tmp_dir
    )
    partition_path = [
        f.path for f in dbutils.fs.ls("dbfs:" + tmp_dir) if f.name.endswith(".csv")
    ][0].replace("dbfs:", "")
    print("/dbfs" + partition_path)
    print("/dbfs" + path.replace("dbfs:", ""))
    shutil.copyfile("/dbfs" + partition_path, "/dbfs" + path.replace("dbfs:", ""))
    dbutils.fs.mv(partition_path, "dbfs:" + path)
    dbutils.fs.rm(tmp_dir, recurse=True)


def get_item_pricing_csv_path():
    p = path.join(getArgument("base_dir"), "input", "csv", "item_pricing.csv")
    return p


def get_neo_item_master_parquet_path():
    p = path.join(getArgument("base_dir"), "input", "parquet", "neo_item_master")
    return p


def get_atg_item_master_parquet_path():
    p = path.join(getArgument("base_dir"), "input", "parquet", "atg_item_master")
    return p


def get_bu_mapping_csv_path():
    p = path.join(getArgument("base_dir"), "input", "csv", "bu_mapping.csv")
    return p


def get_aggregated_item_master_json_path():
    p = path.join(getArgument("base_dir"), "input", "json", "item_master.json")
    return p


def get_aggregated_item_master_csv_path():
    p = path.join(getArgument("base_dir"), "input", "csv", "item_master.csv")
    return p


def get_score_csv_path(bu_desc, class_desc):
    p = path.join(
        getArgument("base_dir"), "intermediate", bu_desc, class_desc, "score.csv"
    )
    return p


def read_item_master():
    # p = get_aggregated_item_master_csv_path()
    # return pd.read_csv(p, escapechar="\\", quoting=csv.QUOTE_ALL)
    p = get_aggregated_item_master_json_path()
    return pd.read_json(p, orient="records")


def get_aggregated_item_master_parquet_path():
    p = path.join(getArgument("base_dir"), "input", "parquet", "item_master")
    return p
