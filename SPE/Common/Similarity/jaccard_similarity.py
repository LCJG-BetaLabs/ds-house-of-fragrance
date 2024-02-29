# Databricks notebook source
# MAGIC %run "./base_similarity"

# COMMAND ----------

from numba import jit
from scipy.spatial.distance import cosine
from pyspark.sql.functions import pandas_udf, PandasUDFType
from sklearn.metrics.pairwise import linear_kernel
from pyspark.sql.types import FloatType
import numpy as np


class JaccardSimilarityGenerator(BaseSimilarityGenerator):
    def create_similarity_udf(self):
        def cos_func(X: pd.Series, Y: pd.Series) -> pd.Series:
            def _jaccard(v):
                a, b = v
                if a is None:
                    a = []
                if b is None:
                    b = []
                s1 = set(a)
                s2 = set(b)
                if not s1 and not s2:
                    return 0.0
                intersect = s1.intersection(s2)
                return float(len(intersect) / len(s1.union(s2)))

            values = np.column_stack((X.values, Y.values))

            a = np.apply_along_axis(_jaccard, -1, values)
            return pd.Series(a)

        return pandas_udf(cos_func, returnType=FloatType())
