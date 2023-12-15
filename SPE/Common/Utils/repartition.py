# Databricks notebook source
  def repartition(df, num_partitions, key):
    """
    Repartition of Koalas DataFrame, the current databricks only installed koalas 1.2 and repartition API is not implemented
    """
    repartitioned_sdf = df.to_spark().repartition(num_partitions, key)
    return ks.DataFrame(repartitioned_sdf)
