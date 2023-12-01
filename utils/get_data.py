from databricks.sdk.runtime import spark


def get_product_feed():
    spark.sql(
        """CREATE
        OR REPLACE TEMPORARY VIEW ProductFeed AS
        SELECT
            sku AS atg_code,
            region,
            stockLevel AS stock_level,
            price
        FROM
            lc_prd.api_product_feed_silver.lc_product_feed 
        WHERE
            load_date = (SELECT MAX(load_date) FROM lc_prd.api_product_feed_silver.lc_product_feed)
        """
    )
    stock = spark.sql(
        """
        SELECT
            atg_code,
            INT(IFNULL(stock_level_cn, 0)) AS stock_level_cn,
            INT(IFNULL(stock_level_hk, 0)) AS stock_level_hk,
            INT(IFNULL(stock_level_row, 0)) AS stock_level_row
        FROM (
            SELECT atg_code, region, stock_level
            FROM ProductFeed
        )
        PIVOT (
            FIRST_VALUE(stock_level)
            FOR region
            IN ("cn" AS stock_level_cn, "hk" AS stock_level_hk, "row" AS stock_level_row)
        )"""
    )
    price = spark.sql(
        """
        SELECT
            atg_code,
            INT(IFNULL(price_cn, 0)) AS price_cn,
            INT(IFNULL(price_hk, 0)) AS price_hk,
            INT(IFNULL(price_row, 0)) AS price_row
        FROM (
            SELECT atg_code, region, price
            FROM ProductFeed
        )
        PIVOT (
            FIRST_VALUE(price)
            FOR region
            IN ("cn" AS price_cn, "hk" AS price_hk, "row" AS price_row)
        )
        """
    )
    product_feed = stock.join(price, on="atg_code", how="inner")
    return product_feed


def get_lc_perfume_data(product_feed):
    lc_data = spark.sql(
        """
        SELECT 
            atg_code,
            style,
            color_desc,
            prod_desc_eng,
            prod_desc_tc,
            brand_desc,
            category_desc,
            class_desc,
            subclass_desc,
            display_name,
            long_desc,
            care,
            img_list
        FROM lc_prd.ml_data_preproc_silver.attribute
        WHERE class_desc = "Perfume"
        """
    )
    lc_data = lc_data.join(product_feed, on="atg_code", how="inner")
    return lc_data
