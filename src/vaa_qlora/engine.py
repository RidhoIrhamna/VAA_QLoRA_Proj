from __future__ import annotations

import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .parse import parse_process_output
from .prompt import build_user_prompt, load_system_prompt

DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_ADAPTER = "ridoley/mistral7b-vaa-general-qlora"


class VAAEngine:
    """Load Mistral-7B Instruct + the public VAA QLoRA adapter and classify a process."""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_id: str = DEFAULT_ADAPTER,
        max_new_tokens: int = 2048,
        cache_dir: str | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA GPU is required. This demo loads Mistral-7B in 4-bit QLoRA."
            )

        self.max_new_tokens = max_new_tokens
        cache_dir = cache_dir or str(Path(__file__).resolve().parents[2] / ".hf_cache")
        os.makedirs(cache_dir, exist_ok=True)

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        auth = {"token": token} if token else {}

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_id, cache_dir=cache_dir, **auth)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bf16 = False
        try:
            bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            pass

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb,
            device_map="auto",
            cache_dir=cache_dir,
            **auth,
        )
        self.model = PeftModel.from_pretrained(base, adapter_id, cache_dir=cache_dir, **auth)
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        self.model.eval()
        self.system_prompt = load_system_prompt()

    def classify(self, process_name: str, activities: list[dict]) -> dict:
        user_text = build_user_prompt(process_name, activities)
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
            )

        gen = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        result = parse_process_output(gen, process_name=process_name)
        result["prompt_tokens"] = int(prompt_len)
        return result
