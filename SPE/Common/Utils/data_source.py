# Databricks notebook source
import pandas as pd
import json
from typing import List


class DataSource:
    def __init__(self, configuration):
        self.configuration = configuration

    def read_item_master(self):
        p = self.configuration.get_aggregated_item_master_json_path()
        return pd.read_json(p, orient="records")

    def read_available_inventory(self, region=None):
        inv_df = pd.read_csv(self.configuration.get_lc_feed_api_csv_path())
        if region:
            region = region.lower()
            inv_df = inv_df[inv_df["region"] == region]
        inv_df = inv_df[["sku"]].drop_duplicates()
        return inv_df.rename(columns={"sku": "atg_code"})

    def read_inventory_pricing_discount(self, region: str = None):
        region = region or self.configuration.region
        if not region:
            raise ValueError(f"Invalid region: {region}")
        inv_df = spark.sql(
            """
            SELECT
                sku,
                region,
                stockLevel,
                price,
                discount
            FROM
                lc_prd.api_product_feed_silver.lc_product_feed 
            WHERE
                load_date = (SELECT MAX(load_date) FROM lc_prd.api_product_feed_silver.lc_product_feed)
            """
        ).toPandas()
        region = region.lower()
        inv_df = pd.read_csv(self.configuration.get_lc_feed_api_csv_path())
        inv_df = inv_df[inv_df["region"] == region]
        inv_df = inv_df[["sku", "stockLevel", "price", "discount"]]
        inv_df = inv_df.drop_duplicates(subset=["sku"])
        inv_df = inv_df.set_index("sku")
        inv_df = inv_df.to_dict("index")
        price_dict = {k: v["price"] for k, v in inv_df.items()}
        for k, v in inv_df.items():
            price_dict[k] = v["price"]
        inv_dict = {k: v["stockLevel"] for k, v in inv_df.items()}
        discount_dict = {k: v["discount"] != "0.0%" for k, v in inv_df.items()}
        return inv_dict, price_dict, discount_dict

    def read_inventory_pricing_discount_df(self, region: str = None):
        inv, price, discount = self.read_inventory_pricing_discount(region)
        inv = spark.createDataFrame(
            pd.DataFrame(inv.items(), columns=["atg_code", "SOH"])
        )
        price = spark.createDataFrame(
            pd.DataFrame(price.items(), columns=["atg_code", "price"])
        )
        discount = spark.createDataFrame(
            pd.DataFrame(discount.items(), columns=["atg_code", "discount"])
        )
        return inv.join(price, how="left", on="atg_code").join(
            discount, how="left", on="atg_code"
        )

    def read_keyword_overlap_dict(self, bu_desc, class_desc):
        p = self.configuration.get_keyword_overlap_path(bu_desc, class_desc)
        overlap_df = pd.read_csv(p)
        if len(overlap_df) == 0:
            return {}
        overlap_df = (
            overlap_df.groupby(["source", "sim"])["keyword"].apply(list).reset_index()
        )
        overlap_df["column"] = overlap_df.apply(
            lambda r: (r["source"], r["sim"]), axis=1
        )
        overlap_dict = overlap_df.set_index("column").to_dict(orient="index")
        overlap_dict = {k: v["keyword"] for k, v in overlap_dict.items()}
        return overlap_dict

    def read_highlight_brands(self):
        p = self.configuration.get_highlight_brands_csv_path()
        brands = pd.read_csv(p).rename(columns={"Brand Code": "brand_code"})
        brands = brands[~brands["brand_code"].isnull()]
        brands = brands[~brands["brand_code"].isnull()]
        brands["brand_code"] = brands["brand_code"].astype(int)
        return brands

    def read_subclass_filter_df(self):
        p = self.configuration.get_subclass_filter_csv_path()
        return pd.read_csv(p)

    def read_similarity_config(self, class_desc: str) -> List[str]:
        """
        Returns config of which class should use which similarity scores.
        Requires two files:

        1. `similarity_keys.json`, mapping a key to a similarity score name,
            e.g. "TEXT": "bert,word2vec"
        2. `similarity_config.json`, mapping a class to a key,
            e.g. "Bath Mats": "TEXT,VISUAL"

        :param (str) class_desc: item class desc
        :return: list of similarity score names e.g. ["bert", "word2vec", "vit"]
        """
        with open(self.configuration.get_similarity_config_path(), "r") as file:
            similarity_config = json.load(file)

        with open(self.configuration.get_similarity_keys_path(), "r") as file:
            similarity_keys = json.load(file)

        l = similarity_config[class_desc]
        l = [s.strip() for s in l.split(",")]
        l = [similarity_keys[s] for s in l]
        l = ",".join(l)
        similarities = l.split(",")

        return similarities

