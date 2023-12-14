# Databricks notebook source
import numpy as np
import pandas as pd
from utils.enviroment import (
    LC_FRAGRANTICA_MATCHING,
)

# COMMAND ----------

df = spark.table(LC_FRAGRANTICA_MATCHING)
arr = df.select("atg_code", "image_name", "vit_sim", "similar_name", "brand_desc").toPandas().values

# COMMAND ----------

print(f"all brands: {np.unique(arr[:, 4])}")

# COMMAND ----------

def show_fullset_by_brand(brand):
    tmp = arr[arr[:, 4] == brand, :]
    for row in tmp:
        show_fullset(
            row[0],
            row[1],
            f"""{row[2]}, {row[3]}""",
        )

# COMMAND ----------

# visualization

import os
import requests
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np

# matplotlib.use('Agg')


def read_image(atg):
    url = f"https://media.lanecrawford.com/{atg[0]}/{atg[1]}/{atg[2]}/{atg}_in_m.jpg"
    r = requests.get(url)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content))
    return img


def read_fragrantica_image(image_name):
    img = Image.open(os.path.join("/dbfs/mnt/stg/house_of_fragrance/fragrantica/images/", image_name))
    return img


def show_fullset(atg, ff, score):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    img = read_image(atg)
    axes[0].imshow(img)
    axes[0].set_title(atg)
    axes[0].grid(False)
    axes[0].axis("off")

    img = read_fragrantica_image(ff)
    axes[1].imshow(img)
    axes[1].set_title(ff)
    axes[1].grid(False)
    axes[1].axis("off")

    # plt.suptitle(f"score: {score:.4f}")
    plt.suptitle(score)
    plt.tight_layout()
    display(plt.show())

# COMMAND ----------

idx = np.random.choice(len(arr), size=30)

# COMMAND ----------

for row in arr[idx]:
    show_fullset(
        row[0],
        row[1],
        f"""{row[2]}, {row[3]}""",
    )

# COMMAND ----------

show_fullset_by_brand('TOM FORD')

# COMMAND ----------


