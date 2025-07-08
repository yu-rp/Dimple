import sys
import json
import torch
from transformers import AutoTokenizer
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
dimple_chat_template = qwen_tokenizer.chat_template

from models.tokenization_dimple import DimpleTokenizer
DimpleTokenizer.register_for_auto_class("AutoTokenizer")
dimple_tokenizer = DimpleTokenizer.from_pretrained(
    "Dream-org/Dream-v0-Instruct-7B", 
    padding_side="left",
    chat_template=dimple_chat_template,
    trust_remote_code=True
    )
dimple_tokenizer.add_special_tokens(special_tokens_dict = {
    "bos_token":"<|im_start|>",
    'eos_token': '<|im_end|>',
    'additional_special_tokens': ["<|vision_start|>", "<|vision_end|>", "<|vision_pad|>","<|image_pad|>","<|video_pad|>"]
    })

from models.image_processing_dimple import DimpleImageProcessor
DimpleImageProcessor.register_for_auto_class("AutoImageProcessor")
dimple_image_processor = DimpleImageProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

from models.processing_dimple import DimpleProcessor
DimpleProcessor.register_for_auto_class("AutoProcessor")

dimple_processor = DimpleProcessor(
    image_processor=dimple_image_processor, 
    tokenizer=dimple_tokenizer, 
    chat_template=dimple_chat_template,
)
dimple_processor.save_pretrained("/somewhere/dimple")

qwen_vision_config_dict = json.load(
    open("/vision/config/of/qwen", "r")
)
from models.configuration_dimple import DimpleConfig, DimpleVisionConfig
DimpleVisionConfig.register_for_auto_class("AutoConfig")
from transformers import AutoModel

# First load the model to access its visual config directly
qwen_model = AutoModel.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
qwen_vision_config = qwen_model.visual.config
dimple_vision_config = DimpleVisionConfig(**qwen_vision_config.to_dict())
DimpleConfig.register_for_auto_class("AutoConfig")
dimple_config = DimpleConfig.from_pretrained("Dream-org/Dream-v0-Instruct-7B", vision_config = dimple_vision_config, use_cache = False)

from models.modeling_dimple import DimpleModel
DimpleModel.register_for_auto_class("AutoModel")
model = DimpleModel(dimple_config)

vision_state_dict = qwen_model.visual.state_dict()
load_res = model.visual.load_state_dict(vision_state_dict)

from transformers import AutoModel

dream = AutoModel.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
lm_head_state_dict = dream.lm_head.state_dict()
load_res = model.lm_head.load_state_dict(lm_head_state_dict)

model_state_dict = dream.model.state_dict()
load_res = model.model.load_state_dict(model_state_dict)

model.save_pretrained("/somewhere/dimple")