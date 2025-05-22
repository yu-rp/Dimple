import copy, math, torch, numpy, os, random
from io import BytesIO
from typing import BinaryIO
from PIL import Image
from PIL.Image import Image as ImageObject
from copy import deepcopy
from typing import List, Dict, Any
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
import warnings
import torch.nn.functional as F

def _regularize_images(images: list["ImageInput"], **kwargs) -> dict[str, list["ImageObject"]]:
    r"""Regularize images to avoid error. Including reading and pre-processing."""
    results = []
    for image in images:
        if isinstance(image, (str, BinaryIO)):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if not isinstance(image, ImageObject):
            raise ValueError(f"Expect input is a list of images, but got {type(image)}.")

        results.append(_preprocess_image(image, **kwargs))

    return {"images": results}

def _preprocess_image(image: "ImageObject", **kwargs) -> "ImageObject":

    if (image.width * image.height) > kwargs["image_max_pixels"]:
        resize_factor = math.sqrt(kwargs["image_max_pixels"] / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < kwargs["image_min_pixels"]:
        resize_factor = math.sqrt(kwargs["image_min_pixels"] / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    if min(image.width, image.height) < 28:
        width, height = max(image.width, 28), max(image.height, 28)
        image = image.resize((width, height))

    if image.width / image.height > 200:
        width, height = image.height * 180, image.height
        image = image.resize((width, height))

    if image.height / image.width > 200:
        width, height = image.width, image.width * 180
        image = image.resize((width, height))

    return image