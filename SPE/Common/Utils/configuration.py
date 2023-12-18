# Databricks notebook source
import os


class Configuration:
    def __init__(
        self,
        base_dir: str,
        bu_desc: str = None,
        region: str = None,
        product_feed_date: str = "latest",
    ):
        """
        :param base_dir: Project base directory.
        :param product_feed_date: For time travel to get historical LC product feed. Either "latest" or date in format "YYYYMMDD".
        """
        self.base_dir = base_dir
        self.common_dir = os.path.join(os.path.dirname(os.path.normpath(self.base_dir)), "common")
        self.bu_desc = bu_desc
        self.region = region
        self.product_feed_date = product_feed_date

    def get_base_dir(self, include_dbfs=True):
        if include_dbfs and not self.base_dir.startswith("/dbfs"):
            return "/dbfs" + self.base_dir
        return self.base_dir

    def get_score_path(self, bu_desc, class_desc):
        return os.path.join(
            self.get_base_dir(), "intermediate", bu_desc, class_desc, "score.csv"
        )

    def get_item_pricing_csv_path(self):
        return os.path.join(self.get_base_dir(), "input", "csv", "item_pricing.csv")

    def get_aggregated_item_master_json_path(self, local=True):
        p = os.path.join(self.get_base_dir(local), "input", "json", "item_master.json")
        return p

    def get_filtered_item_master_table_name(self):
        return "lc_dev.ml_house_of_fragrance_silver.lc_fragrantica_matching"

    def get_image_url_path(self, size, pos, local=False):
        p = os.path.join(
            self.get_base_dir(local),
            "intermediate",
            "delta",
            "_".join(["image_url", pos, size]),
        )
        return p

    def get_image_output_blob_path(self, size, pos, local=False):
        p = os.path.join(
            self.get_base_dir(local),
            "intermediate",
            "delta",
            "_".join(["image_output_blob", pos, size]),
        )
        return p

    def get_image_dir(self, size, pos, local=True):
        p = os.path.join(
            self.get_base_dir(local), "input", "_".join(["images", pos, size])
        )
        return p

    def get_image_download_result_path(self, size, pos, local=False):
        p = os.path.join(
            self.get_base_dir(local),
            "intermediate",
            "delta",
            "_".join(["image_download_result", pos, size]),
        )
        return p

    def get_image_delta_path(self, size, pos, local=False):
        p = os.path.join(
            self.get_base_dir(local),
            "intermediate",
            "delta",
            "_".join(["images", pos, size]),
        )
        return p

    def get_atg_category_parquet_path(self, local=False):
        p = os.path.join(self.get_base_dir(local), "input", "parquet", "atg_category")
        return p

    def get_atg_product_text_parquet_path(self, local=False):
        p = os.path.join(
            self.get_base_dir(local), "input", "parquet", "atg_product_text"
        )
        return p

    def get_neo_class_desc_keyword_parquet_path(self, local=False):
        p = os.path.join(
            self.get_base_dir(local), "intermediate", "parquet", "neo_class_keyword"
        )
        return p

    def get_neo_subclass_desc_keyword_parquet_path(self, local=False):
        p = os.path.join(
            self.get_base_dir(local), "intermediate", "parquet", "neo_subclass_keyword"
        )
        return p

    def get_atg_category_keyword_parquet_path(self, local=False):
        p = os.path.join(
            self.get_base_dir(local), "intermediate", "parquet", "atg_category_keyword"
        )
        return p

    def get_atg_display_name_keyword_parquet_path(self, local=False):
        p = os.path.join(
            self.get_base_dir(local),
            "intermediate",
            "parquet",
            "atg_display_name_keyword",
        )
        return p

    def get_long_desc_keyword_parquet_path(self, local=False):
        p = os.path.join(
            self.get_base_dir(local), "intermediate", "parquet", "atg_long_desc_keyword"
        )
        return p

    def get_details_keyword_parquet_path(self, local=False):
        p = os.path.join(
            self.get_base_dir(local), "intermediate", "parquet", "atg_details_keyword"
        )
        return p

    def get_combined_keyword_parquet_path(self, local=False):
        p = os.path.join(
            self.get_base_dir(local), "intermediate", "parquet", "atg_combined_keyword"
        )
        return p

    def get_improved_hierarchy_parquet_path(self, local=False):
        p = os.path.join(
            self.get_base_dir(local),
            "intermediate",
            "parquet",
            "improved_class_hierarchy",
        )
        return p

    def get_improved_hierarchy_json_path(self, local=True):
        p = os.path.join(
            self.get_base_dir(local), "input", "json", "improved_class_hierarchy.json"
        )
        return p

    def get_improved_hierarchy_delta_path(self, local=False):
        p = os.path.join(
            self.get_base_dir(local),
            "intermediate",
            "delta",
            "improved_class_hierarchy",
        )
        return p

    def get_similar_class_pairs_parquet_path(self, local=False):
        p = os.path.join(
            self.get_base_dir(local), "intermediate", "parquet", "similar_class_pairs"
        )
        return p

    def get_lc_feed_api_csv_path(self):
        return (
            f"/dbfs/mnt/shared/lc/product_feed/lc_feed_api_{self.product_feed_date}.csv"
        )

    def get_keyword_overlap_path(self, bu_desc, class_desc):
        return os.path.join(
            self.get_base_dir(),
            "intermediate",
            bu_desc,
            class_desc,
            "keyword_overlap.csv",
        )

    def get_highlight_brands_csv_path(self):
        return os.path.join(self.get_base_dir(), "input", "csv", "highlight_brands.csv")

    def get_similarity_config_path(self):
        return os.path.join(
            self.get_base_dir(), "input", "json", "similarity_config.json"
        )

    def get_similarity_keys_path(self):
        return os.path.join(
            self.get_base_dir(), "input", "json", "similarity_keys.json"
        )

    def get_class_keyword_mapping_csv_path(self, local=True):
        return os.path.join(
            self.get_base_dir(local), "input", "csv", "class_keyword_mapping.csv"
        )

    def get_keyword_mapping_csv_path(self, local=True):
        return os.path.join(
            self.get_base_dir(local), "input", "csv", "keyword_mapping.csv"
        )

    def get_negative_keyword_mapping_csv_path(self, local=True):
        return os.path.join(
            self.get_base_dir(local), "input", "csv", "negative_keyword_mapping.csv"
        )

    def get_class_mapping_csv_path(self, local=True):
        return os.path.join(
            self.get_base_dir(local), "input", "csv", "class_mapping.csv"
        )

    def get_fashion_segment_class_level_mapping_csv_path(self, local=False):
        return os.path.join(
            self.get_base_dir(local),
            "input",
            "csv",
            "fashion_segment_class_level_mask_type.csv",
        )

    def get_fashion_segment_mapping_delta_path(self, local=False):
        return os.path.join(
            self.get_base_dir(local), "intermediate", "delta", "fashion_segment_mapping"
        )

    def get_similarity_all_delta_path(self, sim_type, local=False):
        return os.path.join(
            self.get_base_dir(local), "intermediate", "similarity", sim_type
        )

    def get_similarity_class_delta_path(self, sim_type, class_desc, local=False):
        return os.path.join(
            self.get_base_dir(local), "intermediate", "similarity_by_class", sim_type, class_desc
        )

    def get_brand_group_path(self, local=True):
        return os.path.join(self.get_base_dir(local), "input", "csv", "brand_group.csv")

    def get_output_root(self):
        return os.path.join(self.get_base_dir(True), "output")

    def get_output_dir(
        self, bu_desc: str = None, region: str = None, deploy: bool = False
    ):
        if deploy:
            folder = "deployment"
        else:
            folder = bu_desc or self.bu_desc
        region = region or self.region
        return os.path.join(self.get_base_dir(True), "output", folder, region)

    def get_brand_matrix_path(self, local=True):
        return os.path.join(self.get_base_dir(local), "input", "csv", "brand_matrix.csv")
    
    def get_persona_path(self):
        return "/dbfs" + os.path.join(self.common_dir, f"persona_{self.region.lower()}.csv")
    
    def get_price_point_path(self):
        return "/dbfs" + os.path.join(self.common_dir, f"price_point_{self.region.lower()}.csv")
    
