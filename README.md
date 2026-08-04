# 📖 GPT-2 TinyStories Fine-Tuning

A lightweight pipeline to fine-tune **GPT-2** on the **TinyStories** dataset using Hugging Face `transformers`, `datasets`, and PyTorch.

---

## ⚡ Quickstart

### 1. Requirements
```bash
pip install torch transformers datasets huggingface_hub
```

### 2. Pipeline Overview
1. **Load Base Model:** Instantiates `gpt2` with aligned `pad_token_id`.
2. **Download Data:** Fetches `roneneldan/TinyStories` via `snapshot_download`.
3. **Preprocess:** Filters empty/short stories, tokenizes sequences (max length: 256), and collates for causal LM.
4. **Train:** Fine-tunes model with `Trainer` for 3 epochs using mixed-precision (`fp16`).
5. **Inference:** Generates stories from trained checkpoints using nucleus sampling.

---

## 📊 Training Summary

* **Dataset Split:** 429 Train Samples | 189 Validation Samples
* **Hyperparameters:** `learning_rate=5e-5`, `batch_size=8`, `epochs=3`

| Epoch | Training Loss | Validation Loss |
| :---: | :---: | :---: |
| 1 | 2.5551 | 2.3304 |
| 2 | 2.1712 | 2.1154 |
| **3** | **2.1975** | **2.0347** |

---

## 🚀 Running Inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = "./gpt2-tinystories112/checkpoint-162"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "Once, there was a happy family of cats"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.7, 
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
## Output
```
Once, there was a happy family of cats. The mother looked at the cat and said, "Hello, cat. Are you hungry?" The cat said, "No, I am hungry!" So, the mom said, "Let's go visit the shop and pick something up. We can buy some food." The cat liked the idea and decided to let the mom and dad go and clean the shop. The cat liked having a good time and wanted to eat more. So, they went to the shop and picked something to eat. They

```


---

## 📁 Essential Checkpoint Files
To save storage, you only need these **5 files** (~450 MB) for inference:
* `model.safetensors`
* `config.json`
* `generation_config.json`
* `tokenizer.json`
* `tokenizer_config.json`

*(You can delete `optimizer.pt` and `scheduler.pt` to save ~900 MB).*
