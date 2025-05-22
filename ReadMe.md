<img src="https://cdn-uploads.huggingface.co/production/uploads/635364b3c41f548fe39db945/T6ffjtAkFkI76QjXmN6iR.png" alt="Dimple" style="width:100%;"/>

<p align="center">
🤗 <a href="https://huggingface.co/rp-yu/Dimple-7B">Model</a>&nbsp&nbsp | &nbsp&nbsp 💬 <a href="https://huggingface.co/spaces/rp-yu/Dimple-7B">Demo: Chat with Dimple</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/">Paper</a>&nbsp&nbsp | &nbsp&nbsp ✨ <a href="https://github.com/yu-rp/Dimple">Code</a>&nbsp&nbsp
</p>

# 💧 Dimple


**Dimple** is the first Discrete Diffusion Multimodal Large Language Model (DMLLM) that leverages a hybrid training paradigm combining autoregressive and diffusion-based instruction tuning. The model architecture is similar to Qwen and LLaVA, while introducing an **autoregressive-then-diffusion** training strategy:

* **Stage 1**: Autoregressive fine-tuning for alignment and initial instruction tuning.
* **Stage 2**: Diffusion-based fine-tuning for enhanced instruction-following capabilities.

Trained on the same dataset as LLaVA-NEXT, **Dimple-7B surpasses LLaVA-NEXT-7B by 3.9%**, demonstrating that diffusion-based multimodal large language models can match its autoregressive counterparts under similar training budget.

---

## 🔍 Highlights

* **Hybrid Training**: Combines autoregressive and diffusion training.
* **Diffusion Decoding**: Supports confident decoding, random decoding, maskgit-style decoding, and entropy-based decoding.
* **Controllable Generation**: Enables fine-grained control over format, structure, and length via structure priors.
* **Autoregressive-like Prefilling**: Enhances inference speed using prefilling techniques.

---

## 📊 Evaluation Results

| Benchmark             | Dimple-7B (ours) | LLaVA-1.5-7B | LLaVA-NEXT-7B | Eagle-7B | Eagle2-9B | Qwen-VL-7B | Qwen2.5-VL-7B |
| --------------------- | ---------------- | ------------ | ------------- | -------- | --------- | ---------- | ------------- |
| **Training Samples**  | 1.3M             | 1.2M         | 1.3M          | 2.4M     | 27.8M     | 1.5B       | -             |
| **Training Tokens**   | 0.8B             | -            | -             | -        | -         | -          | 2.6T          |
| **Base LLM**          | Dream (Qwen2.5)  | Vicuna       | Vicuna-1.5    | Vicuna   | Qwen2.5   | Qwen       | Qwen2.5       |
| **GQA**               | 59.2             | 62.0         | 64.8          | 64.9 | -         | 59.3       | -             |
| **MMBench (en test)** | 74.6         | 64.3         | 68.7          | 68.4     | -         | -          | 83.5      |
| **MME (Perception)**  | 1514             | 1510         | 1519          | 1528 | -         | -          | -             |
| **MME (Cognition)**   | 432          | -            | 332           | -        | -         | -          | -             |
| **MME (Total)**       | 1946         | -            | 1851          | -        | -         | -          | 2347      |
| **POPE**              | 86.2             | 85.8         | 86.7          | 88.8 | -         | -          | -             |
| **MMMU (val)**        | 45.2         | -            | 35.8          | 36.3     | 56.1      | -          | 58.6      |
| **SQA (img)**         | 77.1         | 66.8         | 72.8          | 70.0     | -         | -          | -             |
| **AI2D**              | 74.4         | -            | 65.4          | -        | 83.9  | 62.3       | 83.9      |
| **ChartQA**           | 63.4             | -            | 54.9          | 67.7 | 86.4  | 65.7       | 87.3      |
| **TextVQA**           | 61.6             | -            | 64.8      | -        | 83.0  | -          | -             |
| **OCRBench**          | 565          | -            | 490           | 529      | -         | -          | -             |
| **MathVista (mini)**  | 42.3         | -            | 33.0          | -        | 63.8  | 37.0       | 68.2      |
| **MMVet**             | 41.2             | 31.1         | 47.3      | -        | 62.2  | -          | 67.1      |

---

## 🛠️ Environment

Make sure your environment includes the following versions:

```bash
transformers==4.46.2
torch==2.5.1
accelerate==1.6.0
```

---

## 🚀 Inference Example

```python
import torch
from transformers import AutoProcessor, AutoModel
import json, requests
from PIL import Image

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

image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
messages = [
    [{"role": "user", "content": [
        {"type": "image", "image": image_url},
        {"type": "text", "text": "Describe this image."}
    ]}],
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, add_vision_id=False
)
images = [
    Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
]

inputs = processor(
    text=text,
    images=images,
    videos=None,
    padding="longest",
    return_tensors="pt",
)

input_ids = inputs.pop("input_ids")
output = model.diffusion_generate(
    input_ids,
    max_new_tokens=64,
    output_history=True,
    return_dict_in_generate=True,
    steps=64,
    temperature=0.2,
    top_p=0.95,
    alg="origin",
    use_cache=True,
    alg_p_threshold=0.95,
    use_original_confidence=True,
    decoding_pipeline="dim",
    **inputs
)

generations = [
    processor.tokenizer.decode(g[len(p):].cpu().tolist())
    for p, g in zip(input_ids, output.sequences)
]

for j in range(len(messages)):
    print("output:", j, generations[j].split(processor.tokenizer.eos_token)[0])

# output: 0 In the image, a woman wearing a shirt with a plaid and a dog are sitting together on a beach. The sun appears to be setting in the background, creating a warm and serene atmosphere.
```

---

## 📢 Community
Feel free to join the Dimple Community for in-depth discussions and idea exchange!
![cef5c275065254f3f5d2d9d7117d9f6](https://github.com/user-attachments/assets/f9ac2cb6-fffa-4df3-a780-a69efd086fcb)


## 📚 Citation

> Citation information will be provided soon.
> Please stay tuned if you are interested in citing **Dimple** in your work.
