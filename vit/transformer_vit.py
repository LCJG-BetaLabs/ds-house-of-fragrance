# Databricks notebook source
import os
import numpy as np
import torch
from typing import TYPE_CHECKING

torch.set_grad_enabled(False)

if TYPE_CHECKING:
    from vit.base_transformer import BaseTransformer
    from vit.vit_model import create_timm_vit_model, open_and_preprocess


# COMMAND ----------

# MAGIC %run "../base_transformer"

# COMMAND ----------

# MAGIC %run "./vit_model"

# COMMAND ----------

class VitTransformer(BaseTransformer):
    def __init__(self,
                 model_name: str,
                 model_path: str,
                 image_path: str,
                 encoding_path: str,
                 id_name: str,
                 table_name: str,
                 **kwargs):
        self.model_name = model_name
        self.model_path = model_path
        self.image_path = image_path

        super(VitTransformer, self).__init__(
            encoding_path=encoding_path,
            encoding_dim=768,
            id_name=id_name,
            table_name=table_name
        )

    def init_model(self):
        model, transform, is_distilled = create_timm_vit_model(self.model_name, self.model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return {"model": model, "transform": transform, "is_distilled": is_distilled, "device": device}

    def batch_predict(self, data_list: np.ndarray) -> np.ndarray:
        """Predict on a batch of images and return the encoding arrays (data_list is list of base filenames)"""
        with torch.inference_mode():
            model = self.model["model"]
            is_distilled = self.model["is_distilled"]
            batch = self.make_batch(data_list)
            predictions = model.forward_features(batch)
            if is_distilled:
                predictions = torch.hstack(predictions)
            predictions = predictions.detach().cpu().numpy()
            predictions = np.array([p.flatten() for p in predictions])
        return predictions

    def make_batch(self, base_filenames: np.ndarray) -> np.ndarray:
        """Read images and tidy up as a batched image array"""
        transform = self.model["transform"]
        device = self.model["device"]
        images = [open_and_preprocess(os.path.join(self.image_path, img), preprocess_fn=transform) for img in
                  base_filenames]
        return torch.stack(images).to(device)
