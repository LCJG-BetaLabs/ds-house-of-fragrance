import os

FRAGRANTICA_DIR = "/dbfs/mnt/stg/house_of_fragrance/fragrantica"
FRAGRANTICA_IMAGE_DIR = os.path.join(FRAGRANTICA_DIR, "images") # "/dbfs/mnt/stg/house_of_fragrance/fragrantica_images"
os.makedirs(FRAGRANTICA_IMAGE_DIR, exist_ok=True)

catalog = "lc_dev"
FRAGRANTICA_ATTRIBUTE = f"{catalog}.ml_house_of_fragrance_sliver.fragrantica_attribute"

LC_ATTRIBUTE = f"{catalog}.ml_house_of_fragrance_sliver.lc_attribute"
LC_VIT_ENCODING_PATH = "/Volumes/lc_prd/ml_data_preproc_silver/encoding/vit"
LC_VIT_TABLE = "lc_prd.ml_data_preproc_silver.encoding_vit"

FRAGRANTICA_VIT_TABLE = f"{catalog}.ml_house_of_fragrance.fragrantica_encoding_vit"
VIT_MODEL_PATH = "/Volumes/lc_prd/ml_data_preproc_silver/model/timm/vit_base_patch16_224_in21k_miil.pth" # TODO: use HOF volume
FRAGRANTICA_VIT_ENCODING_PATH = "/dbfs/mnt/stg/house_of_fragrance/encoding/vit/fragrantica/encoding" # TODO: use HOF volume
