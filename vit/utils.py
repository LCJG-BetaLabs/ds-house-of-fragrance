# Databricks notebook source
import os
import math
import numpy as np
import pandas as pd

from PIL import Image
from torchvision import transforms
import torch

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from typings import *


# COMMAND ----------

def save_table(
        db_table_name: str,
        df: pd.DataFrame,
        checker: Optional[Callable[[pd.DataFrame], bool]] = None,
        overwrite: bool = False,
        schema=None,
        add_insert_time_column: bool = False,
):
    print(f"Saving to {db_table_name}")

    if checker is not None and not checker(df):
        raise ValueError(f"check failed {db_table_name}")

    if add_insert_time_column:
        df["InsertTime"] = pd.to_datetime('now').replace(microsecond=0)

    sdf = spark.createDataFrame(df, schema=schema)

    table_exist = spark.catalog.tableExists(db_table_name)
    if table_exist:
        print(f'Found table {db_table_name}, inserting to table (overwrite={overwrite}).')

        try:
            sdf.write.insertInto(db_table_name, overwrite=overwrite)
            print(f'Updated table {db_table_name}')
        except Exception as e:
            print(f'Failed to update the table {db_table_name} .')
            raise e
    else:
        print(f'Table {db_table_name} not found, creating table.')
        owner = "betalabsds"
        try:
            sdf.write.format('delta').mode('overwrite').saveAsTable(db_table_name)
            spark.sql(f"ALTER TABLE {db_table_name} OWNER TO {owner}")
            print(f'Created table {db_table_name}')

        except Exception as e:
            print(f'Failed to create table {db_table_name}.')
            raise e


def _partitioned_filenames(folder: str, prefix: str, part: int) -> Tuple[str, str]:
    enc_filename = os.path.join(folder, f"{prefix}_{str(part).zfill(8)}.npy")
    id_filename = os.path.join(folder, f"{prefix}_id_{str(part).zfill(8)}.npy")
    return enc_filename, id_filename


def save_encoding(
        encoding_path: str,
        encoding: np.ndarray,
        id_list: np.ndarray,
        partition_prefix: str = "part",
        partition_size: int = 2000,
) -> np.ndarray:
    """
    Save numpy encoding arrays into partitioned files. It will try to partition into multiple chunks evenly.

    :param encoding_path:
    :param encoding:
    :param id_list:
    :param partition_prefix: underlying encoding filename prefix, default: "part"
    :param partition_size: maximum number of entries in each partition file
    :return:
    """
    i = 0
    # partition encoding array into chunks
    enc_splits = np.array_split(encoding, math.ceil(len(encoding) / partition_size))
    id_splits = np.array_split(id_list, math.ceil(len(encoding) / partition_size))
    parts = []
    for (enc_split, id_split) in zip(enc_splits, id_splits):
        # find next available part file
        while True:
            enc_filename, id_filename = _partitioned_filenames(encoding_path, partition_prefix, i)
            if not os.path.exists(enc_filename):
                break
            i += 1

        # save partition
        np.save(enc_filename, enc_split)
        np.save(id_filename, id_split)
        parts.append(np.full(len(enc_split), fill_value=i, dtype=int))
        i += 1
        print(f"saved encoding {enc_split.shape} to {enc_filename}")

    return np.concatenate(parts)


def read_encoding(
        table_name: str,
        encoding_path: str,
        partition_prefix: str = "part",
        image_list: Optional[List[str]] = None,
        is_latest_only: bool = True,
        time_range: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    :param table_name: feature store database & table containing encoding metadata
    :param encoding_path: path to the encodings files
    :param partition_prefix: underlying encoding filename prefix, default: "part"
    :param image_list: list of desired image names
    :param is_latest_only: if True, only get the most recent encoding for each image
    :param time_range:
    :return:
        - data_encoding: encodings, shape (N * embed_dim)
        - data_id: image names, shape (N)
        - data_part: part ids, shape (N)
    """
    # read the image_name and parts in feature store table
    catalog = spark.table(table_name).toPandas().values
    image_names = catalog[:, 0]
    image_parts = catalog[:, 1].astype(int)

    # inner join user input list
    if image_list is not None:
        keep_idx = np.in1d(image_names, image_list)
        image_names = image_names[keep_idx]
        image_parts = image_parts[keep_idx]

    # read the encoding files by part
    data_id = []  # image names
    data_encoding = []  # actual encoding
    data_part = []  # part ids
    unique_parts = np.unique(image_parts)
    for part in unique_parts:
        enc_filename, id_filename = _partitioned_filenames(encoding_path, partition_prefix, part)
        part_encoding = np.load(enc_filename, allow_pickle=True)
        part_id = np.load(id_filename, allow_pickle=True)
        # only keep those in our needed list
        keep_idx = np.in1d(part_id, image_names)
        data_id.append(part_id[keep_idx])
        data_encoding.append(part_encoding[keep_idx])
        data_part.append(np.full(np.sum(keep_idx), fill_value=part, dtype=int))

    if len(data_id) > 0:
        data_id = np.concatenate(data_id)
        data_encoding = np.vstack(data_encoding)
        data_part = np.concatenate(data_part)
    else:
        data_id = np.array([])
        data_encoding = np.array([])
        data_part = np.array([])

    # deduplicate the encodings if some image_name exists in multiple parts
    if is_latest_only:
        data_id_temp, data_idx_temp = np.unique(data_id[::-1], return_index=True)
        data_idx_temp = len(data_id) - data_idx_temp - 1

        data_id = data_id_temp
        data_encoding = data_encoding[data_idx_temp]
        data_part = data_part[data_idx_temp]
    return data_encoding, data_id, data_part


class PadToSquare:
    """Custom torchvision transform to pad image to square with white background"""

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.
        Returns:
            PIL Image or Tensor: Padded image.
        """
        if isinstance(img, Image.Image):
            w, h = img.size
            fill = 255
        elif isinstance(img, torch.Tensor):
            _, h, w = img.size()
            fill = 1
        else:
            raise TypeError(f"Invalid img type: expected PIL Image or torch Tensor, but found {type(img)}")

        # Calculate padding size
        s = max(h, w)
        top_pad = (s - h) // 2
        bot_pad = (s - h) - top_pad
        left_pad = (s - w) // 2
        right_pad = (s - w) - left_pad

        padding = [left_pad, top_pad, right_pad, bot_pad]
        return transforms.functional.pad(img, padding, fill, "constant")

    def __repr__(self):
        return self.__class__.__name__ + "()"
