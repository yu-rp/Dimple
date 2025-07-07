
import torch
from transformers import AutoProcessor, AutoModel, TextIteratorStreamer

class FullSequenceStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.mask_token = tokenizer.mask_token_id
        self.placeholder_token = tokenizer.convert_tokens_to_ids("_")
        self.placeholder_token = tokenizer.encode("␣")[0]

    def put(self, value, stream_end=False):
        # change mask tokens to space token
        value = value.clone()
        value[value == self.mask_token] = self.placeholder_token
        # Assume full token_ids are passed in every time
        decoded = self.tokenizer.batch_decode(value, **self.decode_kwargs)
        self.text_queue.put(decoded)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def end(self):
        self.text_queue.put(self.stop_signal, timeout=self.timeout)

def get_model(device):
    
    model_name = "rp-yu/Dimple-7B"
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.eval()
    model = model.to(device)

    return model, processor

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def get_qwen(device):
    
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval()
    model = model.to(device)

    return model, processor