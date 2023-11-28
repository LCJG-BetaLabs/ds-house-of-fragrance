# Databricks notebook source

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torchvision import transforms

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from vit.utils import PadToSquare


def open_and_preprocess(img_filename: str, preprocess_fn=None) -> torch.Tensor:
    try:
        img = Image.open(img_filename).convert('RGB')

        if preprocess_fn is not None:
            img = preprocess_fn(img)
        else:
            img = transforms.ToTensor()(img)

        return img
    except Exception:
        print(f"error processing image {img_filename}")
        raise


def create_timm_vit_model(model_name: str, checkpoint_path: str):
    """
    Create timm PyTorch ViT model and transformation function.
    Please see https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    for the appropriate weight files.

    Args:
        model_name (str): timm model name
        checkpoint_path (str): path to model weight
    Returns:
        model (torch.nn.Module): torch model
        transform (torchvision.transforms.Compose): preprocessing transformation
        is_distilled (bool): whether the model is a distilled deit. If True, the model should output a tuple
    """
    model = timm.create_model(model_name, pretrained=False, checkpoint_path=checkpoint_path)
    model.eval()
    config = resolve_data_config({}, model=model)
    # as our image is already cropped image, we set crop pct back to 1
    config["crop_pct"] = 1.0
    transform = create_transform(**config)
    # add a layer to pad image to square at the beginning
    # as there is a CenterCrop transform later
    transform.transforms.insert(0, PadToSquare())

    # deit distilled models have tuple output so we need a flag
    # see https://github.com/huggingface/pytorch-image-models/blob/v0.4.12/timm/models/vision_transformer.py#L255
    is_distilled = (model.num_tokens > 1)

    return model, transform, is_distilled
