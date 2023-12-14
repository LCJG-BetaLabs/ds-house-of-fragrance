# Databricks notebook source
pip install scikit-learn-extra

# COMMAND ----------

import os
import glob
import pyspark.sql.functions as f

import numpy as np
import pandas as pd
import joblib

from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from utils.enviroment import BASE_DIR, LC_FRAGRANTICA_MATCHING

# COMMAND ----------

# get model feature
feature_dir = os.path.join(BASE_DIR, "model_features")
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

files = glob.glob(feature_dir + "/*.parquet")

all_features_df = None
for path in files:
    if "main_accords.parquet" in path:
        # continue
        df = spark.read.parquet(path.replace("/dbfs", ""))
        print(df.count())
        if all_features_df is None:
            all_features_df = df
        else:
            all_features_df = all_features_df.join(df, on="atg_code", how="inner")

# COMMAND ----------

# get matching result
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

# COMMAND ----------

def get_feature(atg_codes, remove_zero_percentage=0.85):
    features_df = all_features_df.filter(f.col("atg_code").isin(atg_codes))
    features_df = features_df.fillna(0)
    # remove features with a lot of zero
    if remove_zero_percentage is not None:
        zero_features = []
        total_rows = features_df.count()
        for column_name in features_df.columns:
            zero_count = features_df.filter(f.col(column_name) == 0).count()
            zero_percentage = zero_count / total_rows

            if zero_percentage >= remove_zero_percentage:
                zero_features.append(column_name)
        print("zero_features: ", zero_features, len(zero_features))
        features_to_keep = [col for col in features_df.columns if col not in zero_features]
        print("features_to_keep", features_to_keep, len(features_to_keep))
        features_df_filtered = features_df.select(features_to_keep)

    features_df_filtered = features_df
    feature_cols = [c for c in features_df_filtered.columns if c != "atg_code"]
    all_prod = features_df_filtered.select("atg_code").toPandas().values.reshape(1, -1)
    features_df_filtered = features_df_filtered.replace("NA", "0")

    # get feature array
    pandas_df = features_df_filtered.select(feature_cols).toPandas()
    features_array = pandas_df.values

    # standardization
    scaler = StandardScaler()
    standardized_df = scaler.fit_transform(features_array)
    standardized_df = np.nan_to_num(standardized_df)

    features_embed = standardized_df
    return standardized_df

# COMMAND ----------

matching_result_pd = matching_result.toPandas()
atg_codes = matching_result_pd["atg_code"].values
atg_codes.shape

# COMMAND ----------

features_embed = get_feature(list(atg_codes), percentage=0.9)
print(features_embed.shape)

# COMMAND ----------

# KMedoids
from sklearn_extra.cluster import KMedoids

sum_of_distances = [] # sum of distances of samples to their closest cluster center.
for i in range(1, 21):
    kmedoids = KMedoids(n_clusters=i, init='k-medoids++')
    kmedoids.fit(features_embed)
    sum_of_distances.append(kmedoids.inertia_)

# COMMAND ----------

# Elbow-method
plt.figure(figsize=(10, 8))
plt.plot(range(1, 21), sum_of_distances, marker="o", linestyle="--")
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of distances of samples to their closest cluster center")
plt.show()

# COMMAND ----------

kmedoids = KMedoids(n_clusters=5, init='k-medoids++')
kmedoids.fit(features_embed)

# COMMAND ----------

print(np.unique(kmedoids.labels_, return_counts=True))

# COMMAND ----------

result_df = pd.DataFrame(
    np.concatenate((np.array(atg_codes).reshape(-1, 1), kmedoids.labels_.reshape(-1, 1)), axis=1),
    columns=["atg_code", "cluster"]
)

# COMMAND ----------

spark.createDataFrame(result_df).write.parquet(os.path.join(model_dir.replace("/dbfs", ""), "kmedoids_clustering_result.parquet"), mode="overwrite")

# save model
joblib.dump(kmedoids, os.path.join(model_dir, "kmedoids_model.pkl"))

# COMMAND ----------


