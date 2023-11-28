# Databricks notebook source
import datetime
import numpy as np
import os
import pandas as pd
import time
from abc import ABC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vit.utils import save_encoding, save_table


# COMMAND ----------

# MAGIC %run "./utils"

# COMMAND ----------

class BaseTransformer(ABC):
    """
    Transformer base class.

    Usage:
        - Define `init_model(self, model)` method that returns a model instance, which will be stored in `self.model`.
        - Define `batch_predict(self, data_list)` method that takes in a batch input, and output the batch encoding.
        - Call `transform(self, ...)` to perform transformation, by batch and with regular flushing to FS.
    """

    def __init__(self,
                 encoding_path: str,
                 encoding_dim: int,
                 id_name: str,
                 table_name: str,
                 encoding_dtype: type = float,
                 **kwargs):
        self.encoding_path = encoding_path
        self.encoding_dim = encoding_dim
        self.id_name = id_name
        self.table_name = table_name
        self.encoding_dtype = encoding_dtype

        # init the model
        self.model = self.init_model()

    def init_model(self) -> Any:
        raise NotImplementedError()

    def batch_predict(self, data_list: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def transform(self,
                  id_list: np.ndarray,
                  data_list: np.ndarray = None,
                  batch_size: int = 256,
                  flush_every: int = 30,
                  partition_size: int = 2000):
        """
        Perform transformation on the input data and insert into feature store table

        :param id_list: Array of data id. This is the values for the column `self.id_name` in FS table.
        :param data_list: Array of data. This is the argument passed into `self.batch_predict(data_list)` (by batch).
                If None, will use `id_list` to replace it.
        :param batch_size: Encoding batch size for the model.
        :param flush_every: Flush into FS every X batches.
        :param partition_size: Size of each underlying partition encoding file.
        :return:
        :rtype:
        """
        if len(id_list) == 0:
            print("empty data, exiting")
            return

        if data_list is None:
            data_list = id_list.view()

        if len(id_list) != len(data_list):
            raise ValueError(f"id_list ({len(id_list)}) must have the same size as data_list ({len(data_list)})")

        os.makedirs(self.encoding_path, exist_ok=True)

        # make a buffer array to store N batches, where N = `flush_every`
        buffer_size = batch_size * flush_every
        encoded_buffer = np.zeros((buffer_size, self.encoding_dim), dtype=self.encoding_dtype)
        id_buffer = np.full(buffer_size, fill_value="", dtype=object)

        # predict by batch
        t0 = time.perf_counter()
        breakpoints = list(range(0, len(id_list), batch_size))
        for b, dstart in enumerate(breakpoints):
            dend = min(dstart + batch_size, len(id_list))
            batch_id = id_list[dstart:dend]
            batch_input = data_list[dstart:dend]

            # indices in the buffer array
            bstart = dstart % buffer_size
            bend = (dend - 1) % buffer_size + 1
            encoded_buffer[bstart:bend] = self.batch_predict(batch_input)
            id_buffer[bstart:bend] = batch_id

            # flush the buffer
            if (b + 1) % flush_every == 0 or (b + 1) == len(breakpoints):
                encoded_buffer = encoded_buffer[:bend]
                id_buffer = id_buffer[:bend]
                time_elapsed = str(datetime.timedelta(seconds=round(time.perf_counter() - t0)))
                print(f"Progress: [{dend}/{len(id_list)}]\t| time elapsed: {time_elapsed}")

                # save to db
                parts = save_encoding(self.encoding_path, encoded_buffer, id_buffer, partition_size=partition_size)
                df = pd.DataFrame({self.id_name: id_buffer, "part": parts})
                save_table(self.table_name, df, add_insert_time_column=True)

                # make a new buffer array
                encoded_buffer = np.zeros((buffer_size, self.encoding_dim), dtype=self.encoding_dtype)
                id_buffer = np.full(buffer_size, fill_value="", dtype=object)
