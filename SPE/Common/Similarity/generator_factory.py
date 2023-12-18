# Databricks notebook source
# MAGIC %run "./jaccard_similarity"

# COMMAND ----------

# MAGIC %run "./embedding_similarity"

# COMMAND ----------

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class SimilarityType(Enum):
    DOMINANT_COLOR = 1
    DOMINANT_COLOR_V2 = 2
    WORD2VEC = 3
    RESNET = 4
    BERT = 5
    VIT = 6
    CLASS = 7
    SUBCLASS = 8
    CONTOUR = 9
    ASPECT_RATIO = 10
    KEYWORD = 11
    ACCORD = 12
    SEASON = 13
    NOTE = 14


@dataclass
class SimilarityConfiguration:
    input_name: str
    input_col: str
    output_name: str
    preprocess_func: Callable = None


SIMILARITY_CLASS_TYPES = {
    SimilarityType.BERT: EmbeddingSimilarityGenerator,
    SimilarityType.KEYWORD: JaccardSimilarityGenerator,
    SimilarityType.ACCORD: EmbeddingSimilarityGenerator,
    SimilarityType.SEASON: EmbeddingSimilarityGenerator,
    SimilarityType.NOTE: JaccardSimilarityGenerator,
}

# COMMAND ----------


def create_similarity_generator_by_type(
    sim_type: SimilarityType, *args, **kwargs
) -> BaseSimilarityGenerator:
    class_type = SIMILARITY_CLASS_TYPES.get(sim_type, None)
    if class_type is None:
        raise KeyError(f"Similarity {sim_type} not supported")
    generator = class_type(**kwargs)
    return generator
