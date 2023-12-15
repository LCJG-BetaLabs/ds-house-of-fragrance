# Databricks notebook source
"""
COS Extension
- Currently similar products for Skincare is now using tag `stable/cos-v2.0.0`
- Other category of COS products are using master and production branch
- Workflow are separated
"""

import os

base_dir_no_dbfs = "/mnt/stg/house_of_fragrance/similar_product_engine/fragrance"
base_dir = "/dbfs" + base_dir_no_dbfs
bu_desc = "fragrance"

# dimension for product tagging filter
tagging_dimensions = "cos_fragrance_type"

jobs = [
    # Item Master Workflow
    {
        "notebook_path": "../Item Master",
        "arguments": {},
    },
    # Word Feature Workflow
    {
        "notebook_path": "../../Common/Word Embedding/bert",
        "arguments": {},
    },
    {
        "notebook_path": "../feature/extract_keyword",
        "arguments": {},
    },
    Similarity
    {
        "notebook_path": "../../Common/Similarity/generate_all_class_similarity",
        "arguments": {
            "base_dir": base_dir_no_dbfs,
            "bu_desc": bu_desc,
            "similarities": "KEYWORD,BERT,SEASON,ACCORD,NOTE",
        },
    },
]

jobs += [
    {
        "notebook_path": "../recommendation",
        "arguments": {
            "bu_desc": bu_desc,
            "base_dir": base_dir_no_dbfs,
            "region": region,
            "tagging_dimensions": tagging_dimensions,
        },
    }
    for region in ["HK"]
]

for job in jobs:
    dbutils.notebook.run(job["notebook_path"], 0, job["arguments"])

# COMMAND ----------

# %run "../../Common/Database/cosmos"

# COMMAND ----------

# cosmos_container_upload(
#     os.path.join(base_dir_no_dbfs, "output", "deployment", "HK"),
#     os.path.join(base_dir_no_dbfs, "output", "deployment", "CN"),
#     max_concurrency=3,
#     max_pending_operations=5,
# )
