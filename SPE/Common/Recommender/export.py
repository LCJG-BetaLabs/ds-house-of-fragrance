# Databricks notebook source
# MAGIC %run "../Utils/spark_func"

# COMMAND ----------

import os
import shutil
import re
import pandas as pd
from jinja2 import Template
from pyspark.sql import DataFrame
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..Utils.spark_func import iterate_groups


def _safe_text(s: str) -> str:
    if pd.isnull(s):
        return ""
    return (
        s.replace("\n", " ").replace(",", " ").replace('""', " ").replace("'", "")[0:]
    )


def _get_lc_image_url(atg_code: str) -> str:
    if not atg_code:
        return ""
    return f"https://media.lanecrawford.com/{atg_code[0]}/{atg_code[1]}/{atg_code[2]}/{atg_code}_in_xl.jpg"


def _save_json(df: pd.DataFrame, output_dir: str, class_desc: str, region: str):
    if len(df) > 0:
        path = os.path.join(output_dir, f"lsh_output_{class_desc}_{region}.json")
        df.to_json(path, orient="records", indent=2)
        print(f"Saved class_desc '{class_desc}' ({len(df)} rows) to {path}")
    else:
        print(f"Empty output for class_desc '{class_desc}'")


JSON_COLUMNS = [
    "source",
    "sim",
    "rank",
    "score_total",
    "region",
    "source_brand_desc",
    "source_long_desc",
    "source_img",
    "source_class_desc",
    "source_subclass_desc",
    # "source_real_atg_class_desc",
    # "source_real_atg_subclass_desc",
    # "source_price",
    # "source_SOH",
    "sim_brand_desc",
    "sim_long_desc",
    "sim_img",
    "sim_class_desc",
    "sim_subclass_desc",
    # "sim_real_atg_class_desc",
    # "sim_real_atg_subclass_desc",
    # "sim_price",
    # "sim_SOH",
]


def export_json_by_class(rec: DataFrame, output_dir: str, region: str, score_columns: List[str], overwrite_outputs: bool=True):
    # clear previous recommendations
    if overwrite_outputs:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    for class_descs, group in iterate_groups(rec, ["source_class_desc"]):
        group = group.toPandas()
        group = group.drop_duplicates(subset=["source", "sim"])
        group = group.sort_values(by=["source", "rank"])
        group["source_long_desc"] = group["source_long_desc"].apply(_safe_text)
        group["sim_long_desc"] = group["sim_long_desc"].apply(_safe_text)
        group["region"] = region
        group["source_img"] = group["source"].apply(_get_lc_image_url)
        group["sim_img"] = group["sim"].apply(_get_lc_image_url)
        group["sim_score"] = group["score_total"]
        group = group[JSON_COLUMNS + score_columns]
        _save_json(group, output_dir, class_descs[0], region)


def _get_class_desc_from_output_filename(filename):
    pattern = re.compile(r"lsh_output_([A-Za-z-& ',]+)_[A-Z]+.json")
    try:
        return pattern.search(filename).group(1)
    except AttributeError:
        raise ValueError(f"Failed to get class_desc from filename '{filename}'")


def export_html_by_class(
    output_root: str,
    output_dir: str,
    bu_desc: str,
    region: str,
    simtypes: List[str],
    repos_root: str = "../..",
):
    class_descs = list(
        map(_get_class_desc_from_output_filename, os.listdir(output_dir))
    )
    print(class_descs)

    # render manifest
    with open(
        os.path.abspath(os.path.join(repos_root, "Common/HTML/manifest.json.jinja")),
        "r",
    ) as f:
        content = f.read()
    manifest = Template(content).render(
        bu_desc=bu_desc,
        class_descs=class_descs,
        regions=[region],
    )
    with open(os.path.join(output_root, "manifest.json"), "w") as f:
        f.write(manifest)

    # render html
    with open(
        os.path.abspath(os.path.join(repos_root, "Common/HTML/recommender.html.jinja")),
        "r",
    ) as f:
        content = f.read()
    html = Template(content).render(simtypes=simtypes)
    with open(os.path.join(output_root, "recommender.html"), "w") as f:
        f.write(html)
