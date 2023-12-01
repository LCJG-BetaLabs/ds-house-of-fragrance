# Databricks notebook source
from utils.enviroment import LC_FRAGRANTICA_MATCHING
import pandas as pd
import pyspark.sql.functions as f

# COMMAND ----------

matching_result = spark.table(LC_FRAGRANTICA_MATCHING)

# COMMAND ----------

matching_result = matching_result.select(
    "atg_code",
    "prod_desc_eng",
    "brand_desc",
    "long_desc",
    "care",
    "for_gender",
    "rating",
    "number_votes",
    "main_accords",
    "description",
    "top_notes",
    "middle_notes",
    "base_notes",
    "longevity",
    "sillage",
    "gender_vote",
    "price_value",
)

# COMMAND ----------

def features_in_list_by_vip(feature, table=matching_result):
    grouped_df = table.groupBy("atg_code").agg(f.collect_list(feature).alias(feature))
    return grouped_df

# COMMAND ----------

def one_hot_encoding(
    feature, table=matching_result, prefix="brand_", postfix="", explode=False
):

    if explode:
        table = table.select("atg_code", f.explode(feature).alias(feature))
        # Get distinct fruits
        distinct_fruits = (
            table.select(feature).distinct().rdd.flatMap(lambda x: x).collect()
        )
        # Create one-hot encoded columns
        encoded_cols = [
            f.expr(
                "IF(array_contains({}, '{}'), 1, 0) AS {}{}{}".format(
                    feature + "_ex",
                    fruit,
                    prefix,
                    fruit.replace("(", "")
                    .replace(")", "")
                    .replace("-", " ")
                    .replace(" ", "_"),
                    postfix,
                )
            )
            for fruit in distinct_fruits
        ]
        # Group by id and aggregate using the encoded columns
        encoded_df = table.groupBy("atg_code").agg(*encoded_cols)
    else:
        table = table.filter((f.col(feature).isNotNull()) & (f.col(feature) != ""))
        distinct_labels = (
            table.select(feature).distinct().rdd.flatMap(lambda x: x).collect()
        )
        encoded_df = table.select(
            "atg_code",
            *[
                f.expr(
                    "IF({} = '{}', 1, 0) AS `{}{}{}`".format(
                        feature, label, prefix, label.replace(" ", "_"), postfix
                    )
                )
                for label in distinct_labels
            ]
        )
    return encoded_df

# COMMAND ----------

brand = one_hot_encoding("brand_desc")
display(brand)

# COMMAND ----------

def one_hot_pd(df, feature):
    df = df.explode(feature)
    one_hot = pd.get_dummies(df[feature])
    df = df.join(one_hot)
    df = df.groupby("atg_code").sum().reset_index()
    columns_to_process = df.columns.drop("atg_code")
    df[columns_to_process] = df[columns_to_process].applymap(lambda x: 1 if x > 1 else 0)
    df = df.rename(columns=lambda x: f'{feature}_' + x.replace("-", "").replace("(", "").replace(")", "").replace(" ", "_") if x != 'atg_code' else x)
    return df

# COMMAND ----------

df = matching_result.select("atg_code", "top_notes").toPandas()
top_notes = one_hot_pd(df, "top_notes")
top_notes

# COMMAND ----------

df = matching_result.select("atg_code", "middle_notes").toPandas()
middle_notes = one_hot_pd(df, "middle_notes")
middle_notes

# COMMAND ----------

df = matching_result.select("atg_code", "base_notes").toPandas()
base_notes = one_hot_pd(df, "base_notes")
base_notes

# COMMAND ----------

def encode_column_of_dict(df, feature):
    df = matching_result.select("atg_code", feature).toPandas()
    df = df[["atg_code"]].join(pd.json_normalize(df[feature])).fillna(0)
    return df

# COMMAND ----------

main_accords = encode_column_of_dict(matching_result, "main_accords")
main_accords

# COMMAND ----------

longevity = encode_column_of_dict(matching_result, "longevity")
longevity

# COMMAND ----------

sillage = encode_column_of_dict(matching_result, "sillage")
sillage

# COMMAND ----------

gender_vote = encode_column_of_dict(matching_result, "gender_vote")
gender_vote

# COMMAND ----------

price_value = encode_column_of_dict(matching_result, "price_value")
price_value

# COMMAND ----------


