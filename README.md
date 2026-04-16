# [ICMR 2026]Ivy-Fake: A Unified Explainable Framework and Benchmark for Image and Video AIGC Detection

<div align="center">

<h4>
<a href="https://arxiv.org/abs/2506.00979">📄 arXiv Paper</a> &nbsp; 
<a href="https://pi3ai.github.io/IvyFake">🌐 Project Page</a> &nbsp; 
<a href="https://huggingface.co/AI-Safeguard/Ivy-Fake">🤗 Hugging Face Models</a>
<a href="https://huggingface.co/datasets/AI-Safeguard/Ivy-Fake">🤗 Hugging Face Datasets</a>
</h4>

</div>


## News

- 🔥 `2026.2` We release our models 🚀[**Ivy-xDetector**](https://huggingface.co/AI-Safeguard/Ivy-Fake) for **AI-generated image and video detection**🔥🔥🔥!
- 🔥 `2025.12` The [**Ivy-Fake**](https://huggingface.co/datasets/AI-Safeguard/Ivy-Fake) is released.
- 🔥 `2025.5` We release the [**Arxiv**](https://arxiv.org/abs/2506.00979)

## 🚀 Getting Started

### 1. Inference Example

The following snippet demonstrates how to perform inference using our model with the `transformers` library.

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Initialize Model and Processor
model_id = "AI-Safeguard/Ivy-Fake"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Define the Detection Prompt
messages = [
    {
        "role": "system",
        "content": "You are an AI-generated content detector. Classify the media as real or fake. Provide reasoning inside <think>...</think> tags. End with exactly one word—real or fake—wrapped in <conclusion>...</conclusion>."
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://path-to-your-image.jpg", # Replace with your media path
            },
            {"type": "text", "text": "Is this image real or fake?"},
        ],
    }
]

# Preparation for Inference
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

# Generation
generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text[0])
```

---

## 📝 Citation

If you find Ivy-Fake or IVY-XDETECTOR useful in your research, please cite:

```bibtex
@article{jiang2025ivy,
  title={Ivy-fake: A unified explainable framework and benchmark for image and video aigc detection},
  author={Jiang, Changjiang and Dong, Wenhui and Zhang, Zhonghao and Si, Chenyang and Yu, Fengchang and Peng, Wei and Yuan, Xinbin and Bi, Yifei and Zhao, Ming and Zhou, Zian and others},
  journal={arXiv preprint arXiv:2506.00979},
  year={2025}
}
```