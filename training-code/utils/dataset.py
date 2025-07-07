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
from transformers.utils import logging

logger = logging.get_logger("dimple." + __name__)

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

def exp_dynamic_padding(n: int) -> int:
    if n < 16:
        return 2 ** math.ceil(math.log2(n + 1))
    else:
        return ((n // 16) + 1) * 16

def mul_dynamic_padding(n: int) -> int:
    if n < 8:
        return ((n // 2) + 1) * 2
    elif n < 16:
        return ((n // 4) + 1) * 4
    elif n < 32:
        return ((n // 8) + 1) * 8
    else:
        return ((n // 16) + 1) * 16

def phrase_dynamic_padding(n: int) -> int:
    if n < 64:
        return 64
    elif n < 256:
        return 256
    elif n < 1024:
        return 1024

def random_exp_dynamic_padding(n: int) -> int:
    if n < 16:
        upper_bound = 2 ** math.ceil(math.log2(n + 1))
    else:
        upper_bound = ((n // 16) + 1) * 16
    lower_bound = n + 1
    return random.randint(lower_bound, upper_bound)

class CustomizedDataProcessor:

    def __init__(
        self, 
        tokenizer, 
        image_processor, 
        default_system="You are a helpful assistant.",
        image_token="<|image_pad|>", 
        image_placeholder="<image>", 
        stop_words= "<|im_end|>",
        ans_seq_length = -1,
        answer_padding_strategy = "fixed",
        max_seq_length = -1,
        only_text = False,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.default_system = default_system
        self.image_token = image_token
        self.image_placeholder = image_placeholder
        self.stop_words = stop_words
        self.ans_seq_length = ans_seq_length
        self.answer_padding_strategy = answer_padding_strategy
        self.max_seq_length = max_seq_length
        self.text_only = only_text

    def gen_2d_attention_mask(self, cut_offs):
        attn_mask = []
        length = cut_offs[-1]
        for cut_off in cut_offs:
            previous_length = len(attn_mask)
            attn_mask.extend(
                [
                    [1] * cut_off + [0] * (length - cut_off) for _ in range(cut_off - previous_length)
                ]
            )
        return attn_mask
        
    def process(
        self,
        item: Dict[str, Any]
    ):
        item = deepcopy(item)

        merge_length: int = getattr(self.image_processor, "merge_size") ** 2

        if self.text_only:
            logger.warning_once("text_only is set to True, so images will be ignored.")
            item["images"] = []
            for m_idx, message in enumerate(item["messages"]):
                content = message["content"]
                content = content.replace(self.image_placeholder, "")
                item["messages"][m_idx]["content"] = content

        if len(item["images"]) != 0:
            images = _regularize_images(
                item["images"],
                image_max_pixels=getattr(self.image_processor, "max_pixels", 768 * 768),
                image_min_pixels=getattr(self.image_processor, "min_pixels", 32 * 32),
            )["images"]
            image_grid_thw = self.image_processor(images, return_tensors="pt")["image_grid_thw"]
        else:
            images = []
            image_grid_thw = None

        image_idx = 0
        input_ids = []
        labels = []
        mask_locations = []

        attn_cut_points = None
        previous_length = None

        for m_idx, message in enumerate(item["messages"]):
            content = message["content"]
            while self.image_placeholder in content:
                if image_idx >= len(image_grid_thw):
                    raise ValueError(f"`len(images)` is less than the number of {self.image_placeholder} tokens.")

                image_seqlen = image_grid_thw[image_idx].prod() // merge_length
                content = content.replace(
                    self.image_placeholder, f"<|vision_start|>{self.image_token * image_seqlen}<|vision_end|>", 1
                )
                image_idx += 1
            
            role = message["role"]
            
            if m_idx == 0:
                formatted_content = f"<|im_start|>system\n{self.default_system}<|im_end|>"
                ids = self.tokenizer.encode(formatted_content, add_special_tokens=False)

                labels.append([-100] * len(ids))
                mask_locations.append([0] * len(ids))
                input_ids.append(ids)

            if role == "user":
                formatted_content = f"\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
                ids = self.tokenizer.encode(formatted_content, add_special_tokens=False)

                labels.append([-100] * len(ids))
                mask_locations.append([0] * len(ids))
                input_ids.append(ids)

            elif role == "assistant":
                formatted_content = f"{content}"
                if self.answer_padding_strategy == "fixed":
                    if self.ans_seq_length > 0:
                        ids = self.tokenizer.encode(
                            formatted_content, 
                            add_special_tokens=False,
                            padding = "max_length",
                            truncation = True,
                            max_length = self.ans_seq_length,
                            padding_side = "right",
                            )
                    else:
                        logger.warning_once("ans_seq_length is not set, so no padding and truncation will be applied.")
                        ids = self.tokenizer.encode(formatted_content, add_special_tokens=False)
                elif self.answer_padding_strategy == "dynamic":
                    ids = self.tokenizer.encode(
                        formatted_content, 
                        add_special_tokens=False,
                        padding = False,
                        truncation = False,
                    )
                    dynamic_length = random_exp_dynamic_padding(len(ids))
                    ids = ids + [self.tokenizer.pad_token_id for _ in range(dynamic_length - len(ids))]
                else:
                    raise ValueError(f"Unsupported answer padding strategy: {self.answer_padding_strategy}")

                labels.append(deepcopy(ids))
                mask_locations.append([1] * len(ids))
                input_ids.append(ids)


            else:
                raise NotImplementedError("Unexpected role: {}".format(role))

        if len(images) != image_idx:
            raise ValueError(f"The number of images does not match the number of {self.image_placeholder} tokens.")

        input_ids = sum(input_ids, [])
        labels = sum(labels, [])
        mask_locations = sum(mask_locations, [])
        attn_mask = None
        order_token = None

        assert len(input_ids) == len(labels) == len(mask_locations)

        if self.max_seq_length > 0:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            mask_locations = mask_locations[:self.max_seq_length]

        inputs = {
                "images": item["images"],
                "input_ids": input_ids,
                "labels": labels,
                "mask_locations": mask_locations,
            }

        logger.debug(f"processed inputs: {inputs.keys()}")
        return inputs


class CustomizedDataCollator(DataCollatorForSeq2Seq):

    def __init__(self, 
            image_processor, 
            position_ids_function, 
            need_mask_location = True, 
            *args, 
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.image_processor = image_processor
        self.position_ids_function = position_ids_function
        self.need_mask_location = need_mask_location
        
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        batch_images = []
        for feature in features:
            batch_images.extend(feature["images"])

        if len(batch_images) > 0:
            batch_images = _regularize_images(
                batch_images,
                image_max_pixels=getattr(self.image_processor, "max_pixels", 768 * 768),
                image_min_pixels=getattr(self.image_processor, "min_pixels", 32 * 32),
            )["images"]
            inputs = self.image_processor(batch_images, return_tensors="pt")
        else:
            inputs = {}

        features_ip_lb = [{"input_ids": feature["input_ids"], "labels": feature["labels"]} for feature in features]
        features_ip_lb = super().__call__(features_ip_lb)
        features_ip_ms = [{"input_ids": feature["input_ids"], "labels": feature["mask_locations"]} for feature in features]
        features_ip_ms = super().__call__(features_ip_ms)
        assert torch.allclose(features_ip_lb["input_ids"], features_ip_ms["input_ids"])
        assert torch.allclose(features_ip_lb["attention_mask"], features_ip_ms["attention_mask"])
        inputs.update(features_ip_lb)
        mask_locations = features_ip_ms["labels"].clone()
        mask_locations = torch.where(mask_locations == 1, 1, 0)
        if self.need_mask_location:
            inputs["mask_locations"] = mask_locations.bool()

        inputs["position_ids"], inputs["rope_deltas"] = self.position_ids_function(
            inputs["input_ids"],
            image_grid_thw = inputs.get("image_grid_thw", None),
            attention_mask = inputs["attention_mask"],
        )

        for k in inputs.keys():
            inputs[k] = inputs[k].clone()
        logger.debug(f"collected inputs: {inputs.keys()}")
        return inputs

def load_llava_alignment_dataset(split: str, **kwargs):
    if kwargs["shard"]:
        logger.warning_once("shard is set to True, so the dataset will be loaded in streaming mode.")
        assert kwargs["shuffle"], "shard should be used with shuffle"
    else:
        pass

    if split == "train":
        if kwargs["shard"]:
            return load_dataset(
                    "json", 
                    data_files="/your/path/to/llava_alignment_train_shards/*.json", 
                    split="train",
                    streaming = True
                ).shuffle(seed=kwargs["seed"], buffer_size=65536)
        else:
            if split == "train":
                return load_dataset(
                    "json",
                    data_files="/your/path/to/llava_alignment_train.json",
                    streaming=True
                )["train"].shuffle(seed=kwargs["seed"], buffer_size=65536)
    elif split == "eval":
        return None
    else:
        raise ValueError(f"Unsupported split: {split}")

def load_llava_next_dataset(seed, shuffle = True, shard = False):
    if shard:
        logger.warning_once("shard is set to True, so the dataset will be loaded in streaming mode.")
        assert shuffle, "shard should be used with shuffle"
    else:
        pass

    if shard:

        return load_dataset(
                "json", 
                data_files="/your/path/to/llava_next_train_shards/*.json", 
                split="train",
                streaming = True
            ).shuffle(seed=seed, buffer_size=65536)
    else:
        if shuffle:
            return load_dataset(
                "json",
                data_files="/your/path/to/llava_next_train.json",
                streaming=True
            )["train"].shuffle(seed=seed, buffer_size=65536)
        else:
            return load_dataset(
                "json",
                data_files="/your/path/to/llava_next_train.json",
                streaming=True
            )["train"]

def from_llava_format_2_qwen_format(item):
    new_item = []
    images = iter(item["images"])
    for message in item["messages"]:
        content = message["content"]
        if "<image>" in content:
            content = content.strip("\n\t ")
            if content.endswith("<image>"):
                content = content[:-7]
                new_item.append({
                    "role": message["role"],
                    "content": [
                        {"type": "text", "text": content.replace("<image>", "")},
                        {"type": "image", "image": next(images)},
                    ],
                })
            elif content.startswith("<image>"):
                content = content[:-7]
                new_item.append({
                    "role": message["role"],
                    "content": [
                        {"type": "image", "image": next(images)},
                        {"type": "text", "text": content.replace("<image>", "")},
                    ],
                })
            else:
                raise ValueError("not well formed")
        else:
            new_item.append(message)
    return {"messages": new_item, "images": item["images"]}