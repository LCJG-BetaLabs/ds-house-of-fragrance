# Databricks notebook source
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

from utils.enviroment import BASE_DIR

# COMMAND ----------

feature_dir = os.path.join(BASE_DIR, "model_features")
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

files = glob.glob(feature_dir + "/*.parquet")

features_df = None
for path in files:
    if "brand.parquet" in path or "price_value.parquet" in path:
        continue
    df = spark.read.parquet(path.replace("/dbfs", ""))
    if features_df is None:
        features_df = df
    else:
        features_df = features_df.join(df, on="atg_code", how="left")

# COMMAND ----------

# remove features with > 80% zero
features_df = features_df.fillna(0)
zero_percentage_threshold = 0.8
zero_features = []

total_rows = features_df.count()
for column_name in features_df.columns:
    zero_count = features_df.filter(f.col(column_name) == 0).count()
    zero_percentage = zero_count / total_rows

    if zero_percentage >= zero_percentage_threshold:
        zero_features.append(column_name)

print(zero_features, len(zero_features))

# COMMAND ----------

features_to_keep = [col for col in features_df.columns if col not in zero_features]

# COMMAND ----------

print(features_to_keep, len(features_to_keep))

# COMMAND ----------

features_df_filtered = features_df.select(features_to_keep)

# COMMAND ----------

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

# COMMAND ----------

# FactorAnalysis
n_components = 25

fa = FactorAnalysis(n_components=n_components)
fa.fit(standardized_df)

# COMMAND ----------

factor_loadings = fa.components_
selected_features = np.abs(factor_loadings).sum(axis=0).argsort()[:n_components]

factor_table = pd.DataFrame(factor_loadings.T, index=np.array(feature_cols))
features_embed = standardized_df[:, selected_features]

# COMMAND ----------

np.array(feature_cols)[selected_features]

# COMMAND ----------

factor_table

# COMMAND ----------

# kmeans
wcss = []  # Within-Cluster Sum of Square -> variability of the observations within each cluster
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, n_init=10)
    kmeans.fit(features_embed)
    wcss.append(kmeans.inertia_)

# COMMAND ----------

# Elbow-method
plt.figure(figsize=(10, 8))
plt.plot(range(1, 21), wcss, marker="o", linestyle="--")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# COMMAND ----------

# k-mean final
kmeans = KMeans(n_clusters=10)
kmeans.fit(features_embed)

# COMMAND ----------

# visualize cluster
# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(features_embed)

# Create scatter plot with colored clusters
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans.labels_)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('K-means Clustering Result (t-SNE Visualization)')
plt.show()

# COMMAND ----------

print(np.unique(kmeans.labels_, return_counts=True))

# COMMAND ----------

# save result
result_df = pd.DataFrame(
    np.concatenate((all_prod.reshape(-1, 1), kmeans.labels_.reshape(-1, 1)), axis=1),
    columns=["atg_code", "cluster"]
)
spark.createDataFrame(result_df).write.parquet(os.path.join(model_dir.replace("/dbfs", ""), "clustering_result.parquet"))

# save model
joblib.dump(kmeans, os.path.join(model_dir, "kmeans_model.pkl"))

# COMMAND ----------


