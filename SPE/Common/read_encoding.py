"""Copy and modified from ds-feature-store/data_preproc/core_feature_store.py"""
import os
import numpy as np
from typing import List, Optional, Tuple
from databricks.sdk.runtime import spark


def _partitioned_filenames(folder: str, prefix: str, part: int) -> Tuple[str, str]:
    enc_filename = os.path.join(folder, f"{prefix}_{str(part).zfill(8)}.npy")
    id_filename = os.path.join(folder, f"{prefix}_id_{str(part).zfill(8)}.npy")
    return enc_filename, id_filename


def read_encoding(
    table_name: str,
    encoding_path: str,
    partition_prefix: str="part",
    image_list: Optional[List[str]]=None,
    is_latest_only: bool=True,
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
        - vernum: data version number
    """
    # read the image_name and parts
    catalog = np.array(spark.sql(f"SELECT * FROM {table_name}").collect())
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