# Databricks notebook source
# MAGIC %run "/utils/spark_utils"

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE 
# MAGIC OR REPLACE TEMPORARY VIEW ItemMaster AS
# MAGIC SELECT *
# MAGIC FROM lc_dev.ml_house_of_fragrance_silver.lc_fragrantica_matching
