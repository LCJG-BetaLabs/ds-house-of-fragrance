# Databricks notebook source
import os
import pandas as pd
from utils.enviroment import BASE_DIR, LC_FRAGRANTICA_MATCHING
import pyspark.sql.functions as f

# COMMAND ----------

# MAGIC %run ./notes_grouping

# COMMAND ----------

feature_dir = os.path.join(BASE_DIR, "model_features")
os.makedirs(feature_dir, exist_ok=True)

matching_result = spark.table(LC_FRAGRANTICA_MATCHING)
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
    "season_rating",
    "description",
    "top_notes",
    "middle_notes",
    "base_notes",
    "longevity",
    "sillage",
    "gender_vote",
    "price_value",
)
matching_result.createOrReplaceTempView("matching")

# COMMAND ----------


def save_feature_df(df, filename):
    df.write.parquet(os.path.join(feature_dir.replace("/dbfs", ""), f"{filename}.parquet"), mode="overwrite")


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
                    """IF({} = "{}", 1, 0) AS `{}{}{}`""".format(
                        feature, label, prefix, label.replace(" ", "_"), postfix
                    )
                )
                for label in distinct_labels
            ]
        )
    return encoded_df


def group_notes(note, mapping):
    for k, v in mapping.items():
        if note in v:
            return k
        

def one_hot_pd(df, feature, mapping=None):
    df = df.explode(feature)
    if mapping:
        df[feature] = df[feature].apply(lambda x: group_notes(x, mapping))
    one_hot = pd.get_dummies(df[feature])
    df = df.join(one_hot)
    df = df.groupby("atg_code").sum().reset_index()
    columns_to_process = df.columns.drop("atg_code")
    df[columns_to_process] = df[columns_to_process].applymap(lambda x: 1 if x > 1 else 0)
    df = df.rename(columns=lambda x: f'{feature}_' + x.replace("-", "").replace("(", "").replace(")", "").replace(" ", "_") if x != 'atg_code' else x)
    return df


def encode_column_of_dict(df, feature):
    df = matching_result.select("atg_code", feature).toPandas()
    df = df[["atg_code"]].join(pd.json_normalize(df[feature])).fillna(0)
    return df


def add_prefix(df, prefix):
    new_colname = [prefix + col for col in df.columns if col != "atg_code"]
    df.columns = ["atg_code"] + new_colname


def get_max_dict_key(d):
    d = {key: value if value is not None else 0 for key, value in d.items()}
    return max(d, key=d.get)


def group_accords(accords):
    for k, v in main_accords_grouping.items():
        if accords in v:
            return k
        

def combine(data, key1, key2, _as=None):
    combined = data[key1] + data[key2]
    del data[key1]
    del data[key2]
    data[_as] = combined
    return data


def convert_to_percentage(df):
    cols_to_convert = df.columns[1:]
    row_sums = df[cols_to_convert].sum(axis=1)
    df_percentage = df[cols_to_convert].div(row_sums, axis=0)
    df_percentage = pd.concat([df['atg_code'], df_percentage], axis=1)
    return df_percentage
    

# COMMAND ----------

# brand
brand = one_hot_encoding("brand_desc")
save_feature_df(brand, "brand")

# COMMAND ----------

# top_notes
df = matching_result.select("atg_code", "top_notes").toPandas()
top_notes = one_hot_pd(df, "top_notes", top_notes_mapping)
save_feature_df(spark.createDataFrame(top_notes), "top_notes")

# COMMAND ----------

df = matching_result.select("atg_code", "middle_notes").toPandas()
middle_notes = one_hot_pd(df, "middle_notes", middle_notes_mapping)
save_feature_df(spark.createDataFrame(middle_notes), "middle_notes")

# COMMAND ----------

df = matching_result.select("atg_code", "base_notes").toPandas()
base_notes = one_hot_pd(df, "base_notes", base_notes_mapping)
save_feature_df(spark.createDataFrame(base_notes), "base_notes")

# COMMAND ----------

df = matching_result.select("atg_code", "main_accords").toPandas()
df["main_accords"] = df["main_accords"].apply(lambda x: get_max_dict_key(x))
distinct_main_accords = df["main_accords"].unique()
df["main_accords"] = df["main_accords"].apply(lambda x: group_accords(x))
df = pd.get_dummies(df, columns=["main_accords"])
save_feature_df(spark.createDataFrame(df), "main_accords")

# COMMAND ----------

longevity = matching_result.select("atg_code", "longevity").toPandas()
longevity["longevity"] = longevity["longevity"].apply(lambda x: combine(x, "eternal", "long_lasting", _as="long_lasting"))
longevity["longevity"] = longevity["longevity"].apply(lambda x: combine(x, "very_weak", "weak", _as="weak"))
longevity["longevity"] = longevity["longevity"].apply(lambda x: get_max_dict_key(x))
longevity = pd.get_dummies(longevity, columns=["longevity"])
save_feature_df(spark.createDataFrame(longevity), "longevity")

# COMMAND ----------

sillage = matching_result.select("atg_code", "sillage").toPandas()
sillage["sillage"] = sillage["sillage"].apply(lambda x: combine(x, "strong", "enormous", _as="strong"))
sillage["sillage"] = sillage["sillage"].apply(lambda x: get_max_dict_key(x))
sillage = pd.get_dummies(sillage, columns=["sillage"])
save_feature_df(spark.createDataFrame(sillage), "sillage")

# COMMAND ----------

gender_vote = matching_result.select("atg_code", "gender_vote").toPandas()
gender_vote["gender_vote"] = gender_vote["gender_vote"].apply(lambda x: get_max_dict_key(x))
gender_vote = pd.get_dummies(gender_vote, columns=["gender_vote"])
save_feature_df(spark.createDataFrame(gender_vote), "gender_vote")

# COMMAND ----------

price_value = encode_column_of_dict(matching_result, "price_value")
add_prefix(price_value, "price_value_")
price_value = convert_to_percentage(price_value)
save_feature_df(spark.createDataFrame(price_value), "price_value")

# COMMAND ----------

season = encode_column_of_dict(matching_result, "season_rating")
add_prefix(season, "season_")
save_feature_df(spark.createDataFrame(season), "season")

# COMMAND ----------


