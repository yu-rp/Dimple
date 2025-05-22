import sys

import base64
import re, math
from io import BytesIO
from typing import List, Optional, Tuple, Union
import time

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer
)

from lmms_eval.models.dimple_model.processing_dimple import DimpleProcessor
from lmms_eval.models.dimple_model.modeling_dimple import DimpleModel
from lmms_eval.models.dimple_model.tokenization_dimple import DimpleTokenizer
from lmms_eval.models.dimple_model.image_processing_dimple import DimpleImageProcessor
from lmms_eval.models.dimple_model.data_utils import _regularize_images

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

import time
import socket

class Timer:
    def __init__(self, accelerator=None, verbose=False, batch_size=1):
        self.times = [] 
        self.tokens = [] 
        self.start_time = None
        self.verbose = verbose
        self.accelerator = accelerator
        self.batch_size = batch_size

    def start(self):
        if self.start_time is not None:
            raise RuntimeError("Timer is already running. Call stop() before start().")
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")
        elapsed = time.time() - self.start_time
        self.times.append(elapsed)
        self.start_time = None
        if self.verbose:
            self.print_time(elapsed)
        return elapsed

    def record_token_length(self, tokens):
        # assert len(self.times) * self.batch_size == len(self.tokens) + self.batch_size, "You must call stop() before recording token length."
        # assert len(tokens) == self.batch_size, f"Expected {self.batch_size} tokens, but got {len(tokens)}."
        self.tokens.extend([len(token) for token in tokens])

    def reset(self):
        self.start_time = None
        self.times = []
        self.tokens = []

    def get_total_time(self):
        return sum(self.times)

    def get_avg_time(self):
        return sum(self.times) / len(self.times) if self.times else 0.0

    def get_max_time(self):
        return max(self.times) if self.times else 0.0

    def get_min_time(self):
        return min(self.times) if self.times else 0.0

    def get_count(self):
        return len(self.times)

    def get_total_tokens(self):
        return sum(self.tokens)

    def get_token_per_second(self):
        if self.get_total_time() == 0:
            return 0.0
        return self.get_total_tokens() / self.get_total_time()

    def get_num_samples(self):
        return len(self.tokens)

    def print_summary(self):
        rank = self.get_rank()
        print(f"[Rank: {rank}] Timer Summary - Runs: {self.get_count()}, Total: {self.get_total_time():.4f}s, Avg: {self.get_avg_time():.4f}s, Max: {self.get_max_time():.4f}s, Min: {self.get_min_time():.4f}s")
        print(f"[Rank: {rank}] Timer Summary - Num Samples: {self.get_num_samples()}, Total Tokens: {self.get_total_tokens()}, Tokens/Second: {self.get_token_per_second():.2f}")

    def print_time(self, elapsed):
        rank = self.get_rank()
        print(f"[Rank: {rank}] Elapsed Time: {elapsed:.4f} seconds")

    def get_rank(self):
        if self.accelerator is not None:
            return self.accelerator.process_index
        return 0

@register_model("dimple")
class Dimple(lmms):

    def __init__(
        self,
        pretrained: str = "rp-yu/Dimple-7B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=False,
        causal_attention: Optional[bool] = False,
        system_prompt: Optional[str] = None,
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        max_pixel_scale: Optional[int] = -1,
        max_length: Optional[int] = 1024,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device
        self.timer = Timer(accelerator=accelerator, verbose=False, batch_size=int(batch_size))

        self._model = DimpleModel.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map).eval()

        if causal_attention:
            self._model.config.full_attn_mask = False
        else:
            self._model.config.full_attn_mask = True

        tokenizer = DimpleTokenizer.from_pretrained(pretrained, padding_side="left")
        img_processor = DimpleImageProcessor.from_pretrained(pretrained)

        if max_pixel_scale > 0:
            img_processor.max_pixels = (14*max_pixel_scale) ** 2
            img_processor.size["max_pixels"] = (14*max_pixel_scale) ** 2
            eval_logger.info(f"Using max pixel value: {img_processor.max_pixels}={math.sqrt(img_processor.max_pixels)}")
        else:
            eval_logger.info("Using original max pixel value")

        processor = DimpleProcessor(
            image_processor=img_processor, 
            tokenizer=tokenizer, 
            chat_template=tokenizer.chat_template,
        )

        self.max_pixels = processor.image_processor.max_pixels
        self.min_pixels = processor.image_processor.min_pixels

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        self.processor = processor
        self._tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = max_length
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def extract_images_from_batched_messages(self, batched_messages):
        images = []
        for messages in batched_messages:
            for input_part in messages:
                for content in input_part["content"]:
                    if content["type"] == "image":
                        image_data = content["image"]
                        if image_data.startswith("data:image/jpeg;base64,"):
                            base64_str = image_data.split(",")[1]
                            image_bytes = base64.b64decode(base64_str)
                            image = Image.open(BytesIO(image_bytes))
                            images.append(image)
                        else:
                            raise ValueError(f"Unsupported image format: {image_data}")
        if len(images) == 0:
            images = None
        else:
            images = _regularize_images(
                    images,
                    image_max_pixels=getattr(self.processor.image_processor, "max_pixels", 768 * 768),
                    image_min_pixels=getattr(self.processor.image_processor, "min_pixels", 32 * 32),
                )["images"]
        return images

    def generate_until(self, requests: List[Instance]) -> List[str]:
        '''
        Generate text until a specified token is reached.
        '''
        self.timer.reset()
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                if self.system_prompt is None:
                    message = []
                else:
                    message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                for visual in visual_list[i]:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        vr = decord.VideoReader(visual)
                        first_frame = vr[0].asnumpy()
                        height, width = first_frame.shape[:2]
                        # max_pixels = height * width
                        processed_visuals.append({"type": "video", "video": visual})
                        # processed_visuals.append({"type": "video", "video": ""})
                    elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                        # processed_visuals.append({"type": "image", "image": f""})

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:  # currently support find <image x> in the context
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)

            texts = self.processor.apply_chat_template(
                batched_messages, tokenize=False, add_generation_prompt=True, add_vision_id = False
            )
            images = self.extract_images_from_batched_messages(batched_messages)

            inputs = self.processor(
                text=texts,
                images=images,
                videos=None,
                padding="longest",
                return_tensors="pt",
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "steps": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": 0.01,
                "alg": "origin",
                "alg_temp": 0.0,
                "use_cache": True,
                "alg_p_threshold": 0.0,
                "use_original_confidence": False,
                "decoding_pipeline": "dim"
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            if current_gen_kwargs["steps"] == "auto":
                current_gen_kwargs["steps"] = current_gen_kwargs["max_new_tokens"]

            input_ids = inputs.pop("input_ids")
            self.timer.start()
            output = self.model.diffusion_generate(
                input_ids,
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                output_history=True,
                return_dict_in_generate=True,
                steps=current_gen_kwargs["steps"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                alg=current_gen_kwargs["alg"],
                alg_temp=current_gen_kwargs["alg_temp"],
                use_cache=current_gen_kwargs["use_cache"],
                alg_p_threshold=current_gen_kwargs["alg_p_threshold"],
                use_original_confidence=current_gen_kwargs["use_original_confidence"],
                decoding_pipeline=current_gen_kwargs["decoding_pipeline"],
                **inputs,
            )
            self.timer.stop()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output.sequences)]
            self.timer.record_token_length(generated_ids_trimmed)
            answers = [
                self.processor.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                for ids in generated_ids_trimmed
            ]
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            # reorder this group of results back to original unsorted form

        res = re_ords.get_original(res)

        pbar.close()
        self.timer.print_summary()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
