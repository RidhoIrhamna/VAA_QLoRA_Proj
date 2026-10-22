"""Minimal example for people who only want to use the public adapter.

Requires: NVIDIA GPU, Hugging Face login for the gated Mistral base model.
    pip install torch transformers peft bitsandbytes accelerate
    hf auth login
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER = "ridoley/mistral7b-vaa-general-qlora"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "You are a Lean/BPM expert. Classify ALL activities as VA, BVA, or NVA. "
            "Return ONLY valid JSON with process_name and activities "
            "(activity_name, label, waste_type, justification)."
        ),
    },
    {
        "role": "user",
        "content": (
            "Process Name: Goods Delivery Process\n"
            "1. Receive shipping order document\n"
            "2. Receive goods from customer\n"
            "3. Deliver goods to customer address\n"
            "Classify ALL activities. JSON only."
        ),
    },
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))
