# coding=utf-8
# Copyright 2024 The Dimple team and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig,
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)
from transformers.cache_utils import (
    Cache,
    DynamicCache,
)
from transformers.generation.utils import GenerationMixin

logger = logging.get_logger("dimple."+__name__)


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False, use_original_confidence = True):

    if use_original_confidence:
        logger.debug(f"Using original confidence: {use_original_confidence}")
        original_logits = logits.clone()
        original_probs = torch.softmax(original_logits, dim=-1)

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            if use_original_confidence:
                logger.debug(f"Sampling: Using original confidence: {use_original_confidence}")
                confidence = torch.gather(original_probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            else:
                logger.debug(f"Sampling: Using original confidence (should be False): {use_original_confidence}")
                confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            if use_original_confidence:
                logger.debug(f"Sampling Fail: Using original confidence: {use_original_confidence}")
                _, x0 = probs.max(dim=-1)
                confidence = torch.gather(original_probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            else:
                logger.debug(f"Sampling Fail: Using original confidence (should be False): {use_original_confidence}")
                confidence, x0 = probs.max(dim=-1)
    else:
        if use_original_confidence:
            logger.debug(f"No Sampling: Using original confidence: {use_original_confidence}")
            _, x0 = probs.max(dim=-1)
            confidence = torch.gather(original_probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        else:
            logger.debug(f"No Sampling: Using original confidence (should be False): {use_original_confidence}")
            confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        if use_original_confidence:
            logger.debug(f"margin_confidence: Using original confidence: {use_original_confidence}")
            sorted_probs, _ = torch.sort(original_probs, dim=-1, descending=True)
            # Extract top1 and top2 probabilities
            top1_probs = sorted_probs[:, 0] 
            top2_probs = sorted_probs[:, 1] 
            # Calculate confidence as top1 - top2
            confidence = top1_probs - top2_probs 
        else:
            logger.debug(f"margin_confidence: Using original confidence (should be False): {use_original_confidence}")
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            # Extract top1 and top2 probabilities
            top1_probs = sorted_probs[:, 0] 
            top2_probs = sorted_probs[:, 1] 
            # Calculate confidence as top1 - top2
            confidence = top1_probs - top2_probs 
            
    
    if neg_entropy:
        if use_original_confidence:
            logger.debug(f"neg_entropy: Using original confidence: {use_original_confidence}")
            epsilon = 1e-10
            log_probs = torch.log(original_probs + epsilon)
            confidence = torch.sum(original_probs * log_probs, dim=-1)
        else:
            logger.debug(f"neg_entropy: Using original confidence (should be False): {use_original_confidence}")
            epsilon = 1e-10
            log_probs = torch.log(probs + epsilon)
            confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


@dataclass
class DimpleModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DimpleGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        # cache parameter
        self.use_cache: bool = kwargs.pop("use_cache", False)
        # general generation parameter
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)
        self.alg_p_threshold: Optional[float] = kwargs.pop("alg_p_threshold", None)
        # highly recommended to be True!
        self.use_original_confidence: Optional[bool] = kwargs.pop("use_original_confidence", True) 
        # dim or dream. the original dream decoding pipeline might be problematic for some cases.
        self.decoding_pipeline: Optional[str] = kwargs.pop("decoding_pipeline", "dim") 
        self.block_attention: Optional[bool] = kwargs.pop("block_attention", False)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DimpleGenerationMixin:
    # in Dream

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        pixel_values = model_kwargs.get("pixel_values", None)
        image_grid_thw = model_kwargs.get("image_grid_thw", None)
        if expand_size == 1:
            logger.debug(
                f"Expanding inputs for generation with `expand_size` = {expand_size}."
            )
            return GenerationMixin._expand_inputs_for_generation(
                expand_size=expand_size,
                is_encoder_decoder=is_encoder_decoder,
                input_ids=input_ids,
                **model_kwargs
            )
        elif pixel_values is None and image_grid_thw is None:
            logger.debug(
                f"Expanding inputs for generation with `expand_size` = {expand_size} without image inputs."
            )
            return GenerationMixin._expand_inputs_for_generation(
                expand_size=expand_size,
                is_encoder_decoder=is_encoder_decoder,
                input_ids=input_ids,
                **model_kwargs
            )
        else:
            raise ValueError(
                "Does not support expansion for image inputs. "
            )

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            logger.warning_once(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation."
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning_once(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DimpleGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DimpleGenerationConfig], **kwargs: Dict
    ) -> DimpleGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DimpleGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            model_kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config, model_kwargs

    def _prepare_special_tokens(
        self,
        generation_config: DimpleGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning_once(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    def gen_4d_attention_mask_with_block_attention(self, model_kwargs) -> torch.Tensor:
        attention_mask = model_kwargs.get("attention_mask", None)
        bs = attention_mask.shape[0]
        prompt_length = model_kwargs["prompt_length"]
        totol_sequence_length = model_kwargs["total_sequence_length"]
        attention_mask_4d = torch.zeros((bs, 1, totol_sequence_length, totol_sequence_length), device=attention_mask.device, dtype = attention_mask.dtype)
        attention_mask_4d[:, :, :prompt_length, :prompt_length] = 1.0
        attention_mask_4d[:, :, prompt_length:] = 1.0
        return attention_mask_4d

    def _mask_pad_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        generation_config: DimpleGenerationConfig,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """
        pad tokens in the input ids and attentions for generation. This is used to insert mask tokens into the input_ids
        """
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        attention_mask = model_kwargs.get("attention_mask", None)

        # pad input_ids to max_length
        input_ids = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            model_kwargs["attention_mask"] = attention_mask
        else:
            raise ValueError(
                "attention_mask should be provided. "
            )

        return input_ids, model_kwargs

    def compare_past_key_values(self, old, new):
        if len(old) != len(new):
            return False
        for (k_old, v_old), (k_new, v_new) in zip(old, new):
            if not (torch.equal(k_old, k_new) and torch.equal(v_old, v_new)):
                return False
        return True

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        # update past_key_values keeping its naming used in model code
        if model_kwargs["use_cache"]:
            assert outputs.past_key_values is not None, "Cache should not be None if use_cache is True"
            assert outputs.past_key_values.get_seq_length() == model_kwargs["total_sequence_length"], \
                f"Cache length {outputs.past_key_values.get_seq_length()} should be equal to the total sequence length {model_kwargs['total_sequence_length']}"
            # The crop operation requires "left padding for batch processing"
            outputs.past_key_values.crop(max_length = model_kwargs["prompt_length"])
        else:
            assert outputs.past_key_values is None, "Cache should be None if use_cache is False"
        model_kwargs["past_key_values"] = outputs.past_key_values

        # update cache position
        if model_kwargs["use_cache"]:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-(model_kwargs["total_sequence_length"] - model_kwargs["prompt_length"]):]
        else:
            assert model_kwargs["cache_position"] is None, "Cache position should be None if use_cache is False"

        if model_kwargs.get("rope_deltas", None) is not None:
            assert torch.equal(
                model_kwargs["rope_deltas"], outputs.rope_deltas), \
                f"Rope deltas {model_kwargs['rope_deltas']} should be equal to the new rope deltas {outputs.rope_deltas}"
        model_kwargs["rope_deltas"] = outputs.rope_deltas
        return model_kwargs

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DimpleGenerationConfig] = None,
        **kwargs,
    ) -> Union[DimpleModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = model_kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = model_kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = model_kwargs.get("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            logger.warning_once(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`."
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            logger.warning_once(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs."
            )

        # 5. initialize kv cache
        model_kwargs["use_cache"] = generation_config.use_cache
        logger.debug(f"Using cache: {model_kwargs['use_cache']}. ")
        if model_kwargs["use_cache"]:
            model_kwargs["past_key_values"] = DynamicCache()
            model_kwargs["prompt_length"] = input_ids.shape[1] - 1
            logger.debug(
                f"The cache is initialized with {model_kwargs['past_key_values']}."
            )
            logger.debug(
                f"Set prompt length to {model_kwargs['prompt_length']}."
            )
        else:
            model_kwargs["past_key_values"] = None
            model_kwargs["prompt_length"] = input_ids.shape[1]  - 1
            logger.debug(
                f"The cache is initialized with {model_kwargs['past_key_values']}."
            )
            logger.debug(
                f"Set prompt length to {model_kwargs['prompt_length']}."
            )

        # 6. Expand inputs for generation
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 7. pad mask for generation
        input_ids, model_kwargs = self._mask_pad_inputs_for_generation(
            input_ids=input_ids,
            generation_config=generation_config,
            **model_kwargs,
        )
        model_kwargs["total_sequence_length"] = input_ids.shape[1]
        logger.debug(
            f"Set total sequence length to {model_kwargs['total_sequence_length']}."
        )

        # 8. set block attention
        block_attention = generation_config.block_attention
        if block_attention:
            model_kwargs["attention_mask_4d"] = self.gen_4d_attention_mask_with_block_attention(model_kwargs)                

        # 9. initialize cache position
        if model_kwargs["use_cache"]:
            model_kwargs["cache_position"] = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
            logger.debug(
                f"The cache position is initialized with {type(model_kwargs['cache_position'])} {model_kwargs['cache_position'].shape}."
            )
        else:
            model_kwargs["cache_position"] = None
            logger.debug(
                f"The cache position is initialized with {model_kwargs['cache_position']}."
            )
        # print(f"pipeline: {generation_config.decoding_pipeline},use cache: {model_kwargs['use_cache']}, block attention: {block_attention}, use original confidence: {generation_config.use_original_confidence}, alg: {generation_config.alg}, alg_temp: {generation_config.alg_temp}, alg_p_threshold: {generation_config.alg_p_threshold}")
        # 10. Generate
        result = self._sample(
            input_ids,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
            **model_kwargs,
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        generation_config: DimpleGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        **model_kwargs,
    ) -> Union[DimpleModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        alg_p_threshold = generation_config.alg_p_threshold
        decoding_pipeline = generation_config.decoding_pipeline
        use_original_confidence = generation_config.use_original_confidence
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        attention_mask = model_kwargs.get("attention_mask", None)
        attention_mask_4d = model_kwargs.get("attention_mask_4d", None)

        histories = [] if (return_dict_in_generate and output_history) else None

        timesteps = torch.linspace(1, eps, steps + 1, device=input_ids.device)

        input_ids = generation_tokens_hook_func(None, input_ids, None)

        num_total_mask = (input_ids == mask_token_id).sum()

        # this allows user-defined token control of the intermediate steps
        for i in range(steps):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            x = model_inputs.pop("input_ids").clone()
            mask_index = (x == mask_token_id)
            outputs = self(x, **model_inputs)

            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            
            logits = outputs.logits
            assert torch.all(x[:,0] != mask_token_id), "The first token should not be a mask token"
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]

            if decoding_pipeline == 'dream':
                # raise NotImplementedError("Dream decoding pipeline is copied from the original code.")
                t = timesteps[i]
                s = timesteps[i + 1]
            
                if alg == 'origin':
                    p_transfer = 1 - s / t if i < steps - 1 else 1
                    x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                    transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                    _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k, use_original_confidence = False)
                    x[mask_index] = x0.clone()
                elif alg == 'autoregressive':
                    x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                    transfer_index_t_s = torch.zeros(*x.shape, device=self.device, dtype=torch.bool)
                    transfer_index_t_s[torch.arange(x.shape[0]), mask_index.max(dim = 1)[1]] = True
                    mask_transfer_index_t_s = transfer_index_t_s[mask_index]
                    _, x0[mask_transfer_index_t_s]= sample_tokens(mask_logits[mask_transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k, use_original_confidence = False)
                    x[mask_index] = x0.clone()
                else:
                    if alg == 'maskgit_plus':
                        confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, use_original_confidence = False)
                    elif alg == 'topk_margin':
                        confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True, use_original_confidence = False)
                    elif alg == 'entropy':
                        confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True, use_original_confidence = False)
                    else:
                        raise RuntimeError(f"Unknown alg: {alg}")
                    num_mask_token = mask_index.sum()
                    number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                        else:
                            confidence = confidence / alg_temp
                            confidence = F.softmax(confidence, dim=-1)
                            transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                        x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_token_id
                        x0_[transfer_index] = x0[transfer_index].clone()
                        x[mask_index] = x0_

            elif decoding_pipeline == 'dim':

                if alg == 'origin':
                    confidence, x0= sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, use_original_confidence = use_original_confidence)
                elif alg == 'origin-ratio':
                    confidence, x0= sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, use_original_confidence = use_original_confidence)
                elif alg == 'autoregressive':
                    confidence, x0= sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, use_original_confidence = use_original_confidence)
                elif alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, use_original_confidence = use_original_confidence)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True, use_original_confidence = use_original_confidence)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True, use_original_confidence = use_original_confidence)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")

                p_threshold_transfer_index_sum = 0
                if alg_p_threshold is not None and alg_p_threshold > 0:
                    if i == steps - 1:
                        # all tokens should be transfered
                        transfer_index = torch.ones_like(confidence, device=self.device, dtype=torch.bool)
                    else:
                        transfer_index = confidence > alg_p_threshold
                    p_threshold_transfer_index_sum = transfer_index.sum()
                    if p_threshold_transfer_index_sum == 0:
                        pass
                    else:
                        logger.debug(f"p threshold is activated, make transfer")
                        x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_token_id
                        x0_[transfer_index] = x0[transfer_index].clone()
                        x[mask_index] = x0_.clone()
                else:
                    pass

                if p_threshold_transfer_index_sum == 0:

                    num_cur_mask = mask_index.sum()
                    ratio_cur_mask = num_cur_mask / num_total_mask
                    if ratio_cur_mask <= timesteps[-1]:
                        raise ValueError(f"ratio_cur_mask {ratio_cur_mask} should be larger than timesteps[-1] {timesteps[-1]}")
                    
                    valid_s_indices = (timesteps < ratio_cur_mask).nonzero(as_tuple=True)[0]
                    s_idx_start = max(valid_s_indices.min().item(), i + 1)
                    t = ratio_cur_mask
                    for s_idx in range(s_idx_start, steps + 1):
                        s = timesteps[s_idx]
                        number_transfer_tokens = int(num_cur_mask * (1 - s / t)) if s > timesteps[-1] else num_cur_mask
                        if number_transfer_tokens >= 1:
                            break
                        else:
                            continue

                    if alg == 'origin':
                        x0_ = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                        transfer_index_t_s = torch.randperm(x0_.shape[0])[:number_transfer_tokens]
                        x0_[transfer_index_t_s]= x0[transfer_index_t_s].clone()
                        x[mask_index] = x0_.clone()
                    elif alg == 'origin-ratio':
                        p_transfer = 1 - s / t if s > timesteps[-1] else 1
                        x0_ = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                        transfer_index_t_s = torch.rand(*x0_.shape, device=self.device) < p_transfer
                        x0_[transfer_index_t_s]= x0[transfer_index_t_s].clone()
                        x[mask_index] = x0_.clone()
                    elif alg == 'autoregressive':
                        x0_ = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                        transfer_index_t_s = torch.zeros(*x.shape, device=self.device, dtype=torch.bool)
                        transfer_index_t_s[torch.arange(x.shape[0]), mask_index.max(dim = 1)[1]] = True
                        mask_transfer_index_t_s = transfer_index_t_s[mask_index]
                        x0_[mask_transfer_index_t_s]= x0[mask_transfer_index_t_s].clone()
                        x[mask_index] = x0_.clone()
                    elif alg in ['maskgit_plus', 'topk_margin', 'entropy']:
                        assert number_transfer_tokens > 0, f"number_transfer_tokens {number_transfer_tokens} should be larger than 0"
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                        else:
                            confidence = confidence / alg_temp
                            confidence = F.softmax(confidence, dim=-1)
                            transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                        x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_token_id
                        x0_[transfer_index] = x0[transfer_index].clone()
                        x[mask_index] = x0_.clone()
            else:
                raise ValueError(f"Unknown decoding pipeline: {decoding_pipeline}")

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            answer_token_length = model_kwargs['total_sequence_length'] - model_kwargs['prompt_length']
            if x.shape[1] == model_kwargs['total_sequence_length']:
                assert torch.all(x[:,:model_kwargs['prompt_length']+1] == input_ids[:,:model_kwargs['prompt_length']+1]), "prompt tokens should not be changed"
            elif x.shape[1] == answer_token_length:
                assert torch.all(
                    x[:,0] == input_ids[:,-answer_token_length]), "The first token in x should be the same as the input_ids"
            input_ids[:, -answer_token_length:] = x[:, -answer_token_length:].clone()

            if histories is not None:
                histories.append(input_ids.clone())

            if decoding_pipeline == 'dim' and torch.all(input_ids != mask_token_id):
                break
        
        if return_dict_in_generate:
            return DimpleModelOutput(
                sequences=input_ids,
                history=histories,
            )
        else:
            return input_ids