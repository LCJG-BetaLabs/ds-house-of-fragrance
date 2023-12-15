# Databricks notebook source
# MAGIC %run "./filters"

# COMMAND ----------

import pandas as pd
import pyspark.sql
from pyspark.sql import functions as f
from pyspark.sql.udf import UserDefinedFunction
from pyspark.sql.types import IntegerType
from typing import TYPE_CHECKING, Dict, Tuple, List

if TYPE_CHECKING:
    spark = pyspark.sql.SparkSession.getActiveSession()
    from ..Utils.configuration import Configuration
    from .filters import FilterType, register_filter


# @register_filter(name="not_same_item", type=FilterType.REQUIRED_FILTER)
def not_same_item():
    return (~f.col("source").eqNullSafe(f.col("sim"))).cast("int")


# @register_filter(name="same_gender", type=FilterType.REQUIRED_FILTER)
def same_gender():
    return f.col("source_gender").eqNullSafe(f.col("sim_gender")).cast("int")


# @register_filter(name="same_class_desc", type=FilterType.REQUIRED_FILTER)
def same_class_desc():
    return f.col("source_class_desc").eqNullSafe(f.col("sim_class_desc")).cast("int")


# @register_filter(name="similar_price", type=FilterType.DEPLOY_FILTER)
def similar_price(udf: UserDefinedFunction):
    return udf(f.col("source_price"), f.col("sim_price"))


# @register_filter(name="in_stock", type=FilterType.DEPLOY_FILTER)
def in_stock():
    return ((f.col("source_SOH") > 0) & (f.col("sim_SOH") > 0)).cast("int")


def _is_same(s1: str, s2: str, default: str = "") -> bool:
    return s1 == default or s2 == default or s1 == s2


def add_persona_filter(config: "Configuration"):
    def read_persona() -> Dict[str, Dict[str, str]]:
        path = config.get_persona_path()
        if os.path.exists(path):
            persona = pd.read_csv(path)
            persona = persona.fillna("")
            persona = persona.apply(lambda c: c.str.strip())
            persona = persona[persona["bu_desc"] == config.bu_desc]
            persona = persona.drop_duplicates("brand_desc", keep="first")
            persona = persona.set_index("brand_desc")[["persona"]]
            persona = persona.to_dict(orient="index")
            return persona
        else:
            print(f"persona file does not exist in path {path}, filter is skipped")
            return {}

    persona = read_persona()

    # add filter if len(persona) > 0, since some bu/region do not have persona info
    if len(persona) > 0:
        @f.udf(returnType=IntegerType())
        def udf(brand1: str, brand2: str) -> int:
            if brand1 not in persona or brand2 not in persona:
                return True
            p1 = persona[brand1]["persona"]
            p2 = persona[brand2]["persona"]
            return int(_is_same(p1, p2))

        @register_filter(name="same_persona", type=FilterType.REQUIRED_FILTER)
        def same_persona():
            return udf(f.col("source_brand_desc"), f.col("sim_brand_desc"))
        

def add_price_point_filter(config: "Configuration"):
    def read_price_point() -> Dict[str, Dict[str, str]]:
        path = config.get_price_point_path()
        if os.path.exists(path):
            price_point = pd.read_csv(path)
            price_point = price_point.fillna("")
            price_point = price_point.apply(lambda c: c.str.strip())
            price_point = price_point[price_point["bu_desc"] == config.bu_desc]
            price_point = price_point.drop_duplicates("brand_desc", keep="first")
            price_point = price_point.set_index("brand_desc")[["price_point"]]
            price_point = price_point.to_dict(orient="index")
            return price_point
        else:
            print(f"price point file does not exist in path {path}, filter is skipped")
            return {}

    price_point = read_price_point()

    # add filter if len(price_point) > 0, since some bu/region do not have price point info
    if len(price_point) > 0:
        @f.udf(returnType=IntegerType())
        def udf(brand1: str, brand2: str) -> int:
            if brand1 not in price_point or brand2 not in price_point:
                return True
            # good" and "better" can be match, and "best" and "better" can be match
            p1 = price_point[brand1]["price_point"]
            p2 = price_point[brand2]["price_point"]
            is_match = True if {p1, p2} <= {"good", "better"} or {p1, p2} <= {"best", "better"} else False
            return int(is_match)

        @register_filter(name="same_price_point", type=FilterType.REQUIRED_FILTER)
        def same_price_point():
            return udf(f.col("source_brand_desc"), f.col("sim_brand_desc"))
    

def add_product_tagging_filter(tagging_dimensions: List[str] = None):
    if tagging_dimensions is None:
        return
    
    tagging = spark.sql(
        """
        SELECT *
        FROM lc_prd.ml_product_tagging_silver.product_tagging
    """
    ).toPandas()

    tagging = tagging[tagging["dimension"].isin(tagging_dimensions)]

    # Validate product tagging
    is_duplicated = tagging.duplicated(subset=["atg_code", "dimension"])
    if any(is_duplicated):
        duplication = (
            tagging.loc[is_duplicated, ["dimension"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(f"Duplicated tags! {duplication}")

    # tagging_dict: a nested dictionary
    #     {
    #         atg1: {
    #             dimension1: tag1,
    #             dimension2: tag2,
    #         },
    #         ...
    #     }
    tagging_dict = (
        tagging.groupby("atg_code")[["dimension", "tag"]]
        .apply(lambda x: x.set_index("dimension")["tag"].to_dict())
        .to_dict()
    )

    @f.udf(returnType=IntegerType())
    def udf(source: str, sim: str) -> int:
        # get all dimensions for both source & sim
        src_tags = tagging_dict.get(source, {})
        sim_tags = tagging_dict.get(sim, {})
        # get common dimensions
        # if one item has tag and one has no tag, can still recommend
        all_dimensions = set(src_tags.keys()) & set(sim_tags.keys())
        for dim in all_dimensions:
            if src_tags[dim] != sim_tags[dim]:
                return 0
        return 1

    @register_filter(name="same_product_tagging", type=FilterType.REQUIRED_FILTER)
    def same_product_tagging():
        return udf(f.col("source"), f.col("sim"))


def add_conflicting_subclass_filter():
    not_match = [
        # puffer and other coats (WW)
        ["Single Breasted", "Puffer"],
        ["Fur", "Puffer"],
        ["Trench", "Puffer"],
        ["Others", "Puffer"],
        ["Gilets", "Puffer"],
        ["Zip Up", "Puffer"],
        ["Leather", "Puffer"],
        ["Double Breasted", "Puffer"],
        ["Capes", "Puffer"],
        ["Long Sleeves", "Puffer"],
        ["Short Sleeves", "Puffer"],
        ["Hoodies", "Puffer"],
        ["Shift", "Puffer"],
        ["LIGHTWEIGHT JERSEY 1", "Puffer"],
        # WW underwear
        ["Bras", "Panties"],
        ["Bras", "Shapewear"],
        ["Bras", "Lingerie and Shapewear"],
        ["Bras", "Bodysuits"],
        ["Bras", "Leggings"],
        ["Tops", "Panties"],
        ["Tops", "Shapewear"],
        ["Tops", "Lingerie and Shapewear"],
        ["Tops", "Bodysuits"],
        ["Tops", "Leggings"],
        ["Camisole", "Panties"],
        ["Camisole", "Shapewear"],
        ["Camisole", "Lingerie and Shapewear"],
        ["Camisole", "Bodysuits"],
        ["Camisole", "Leggings"],
        # WW Shapewear
        ["Panties", "Shapewear"],
        ["Panties", "Bodysuits"],
        ["Bodysuits", "Shapewear"],
        # WW Sleepwear
        ["Bodysuits", "Tops"],
        ["Tops", "Sleepwear"],
        ["Bras", "Sleepwear"],
        ["Bodysuits", "Sleepwear"],
        # WW jumpsuit and rompers problem
        ["Jumpsuits & Rompers", "Shorts"],
        ["Jumpsuits & Rompers", "Wide"],
        ["Jumpsuits & Rompers", "Casual"],
        ["Jumpsuits & Rompers", "Cropped"],
        ["Jumpsuits & Rompers", "Pants"],
        ["Jumpsuits & Rompers", "Straight"],
        ["Jumpsuits & Rompers", "Leggings"],
        ["Jumpsuits & Rompers", "Skinny"],
        # MW underwear
        ["Underwear", "Briefs"],
        ["Underwear", "Boxers"],
        ["Underwear", "Undershirts"],
        ["Undershirts", "Briefs"],
        ["Undershirts", "Boxers"],
        # LSA
        ["Slip-On", "Ballerina"],
        ["Slip-On", "Sandals"],
        ["Slip-On", "Loafers & Moccasins"],
        ["Slip-On", "Low-Top"],
        ["Ballerina", "Sandals"],
        ["Ballerina", "Loafers & Moccasins"],
        ["Ballerina", "Low-Top"],
        ["Sandals", "Loafers & Moccasins"],
        ["Sandals", "Low-Top"],
        ["Loafers & Moccasins", "Low-Top"],
        # MSA
        ["Sneakers", "Oxfords"],
        ["Sneakers", "Derbies"],
        ["Sneakers", "Lace Ups"],
        # MW activewear
        ["Tops", "Shorts"],
        ["Tops", "Pants"],
    ]
    # sort
    not_match = [sorted(row) for row in not_match]
    conflict_subclass = pd.DataFrame(not_match, columns=["subclass1", "subclass2"])
    # all values default to be False -> not matching
    matching_dict: Dict[Tuple[str, str], int] = (
        conflict_subclass.groupby(["subclass1", "subclass2"])
        .apply(lambda x: 0)
        .to_dict()
    )

    @f.udf(returnType=IntegerType())
    def udf(subclass1: str, subclass2: str) -> int:
        subclass1, subclass2 = sorted((subclass1, subclass2))
        is_match = matching_dict.get((subclass1, subclass2), 1)
        return is_match

    @register_filter(name="not_conflicting_subclass", type=FilterType.REQUIRED_FILTER)
    def not_conflicting_subclass():
        return udf(f.col("source_subclass_desc"), f.col("sim_subclass_desc"))
