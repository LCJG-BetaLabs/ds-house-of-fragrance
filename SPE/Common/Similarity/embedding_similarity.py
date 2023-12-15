# Databricks notebook source
# MAGIC %run "./base_similarity"

# COMMAND ----------

from numba import jit
from scipy.spatial.distance import cosine
from pyspark.sql.functions import pandas_udf, PandasUDFType
from sklearn.metrics.pairwise import linear_kernel
from pyspark.sql.types import FloatType
import numpy as np


class EmbeddingSimilarityGenerator(BaseSimilarityGenerator):
    def create_similarity_udf(self):
        @jit(nopython=True, fastmath=True)
        def _cosine_similarity_numba(u, v):
            uv = np.dot(u, v)
            uu = np.dot(u, u)
            vv = np.dot(v, v)
            cos_theta = 1.0
            print(uu,vv)
            if uu != 0 and vv != 0:
                cos_theta = uv / np.sqrt(uu * vv)
            return cos_theta

        def cos_func(X: pd.Series, Y: pd.Series) -> pd.Series:
            def _cos(v):
                a, b = v
                return _cosine_similarity_numba(a, b)

            values = np.column_stack((X.values, Y.values))
            a = np.apply_along_axis(_cos, -1, values)
            return pd.Series(a)

        return pandas_udf(cos_func, returnType=FloatType())
