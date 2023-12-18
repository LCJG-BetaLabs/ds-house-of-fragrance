# Databricks notebook source
# MAGIC %run "../../Common/Keyword/regex_keyword_extractor"

# COMMAND ----------

output_table = "lc_dev.ml_house_of_fragrance_silver.intermediate_keyword"

# COMMAND ----------

import re
from typing import List

# COMMAND ----------

def preproc(string): 
    return string


def postproc(keyword_list: List[str]):
    new_list = []
    for k in keyword_list:
        new_list.extend(re.split(r", | and", k))
    new_list = [n.lower().replace("and ", "").strip() for n in new_list]
    return new_list


im = spark.table("lc_dev.ml_house_of_fragrance_silver.lc_fragrantica_matching")

patterns = [r"(?:<li>(?:Heart|Key) notes: )([A-Za-z ,]+)(?=<\/li>)"]

keyword_extractor = RegexKeywordExtractor(
    item_master=im,
    column="care",
    preproc=preproc,
    patterns=patterns,
    postproc=postproc,
)

keyword_extractor.save_similarity_table(output_table=output_table)
