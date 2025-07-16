from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, BitsAndBytesConfig)
from trl import DPOTrainer
import glob, re, os

BASE = "sft-out"                         # 基盤は常に sft-out
NEXT = f"dpo-out-{len(glob.glob('dpo-out-*'))+1}"

tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True,
                         bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype="bfloat16")
model = AutoModelForCausalLM.from_pretrained(BASE,
                                             quantization_config=bnb,
                                             device_map="auto")

pairs = load_dataset("json", data_files="dpo_pairs.jsonl", split="train")

trainer = DPOTrainer(
    model=model, tokenizer=tok, train_dataset=pairs, beta=0.1,
    args=TrainingArguments(
        output_dir=NEXT,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=1e-5)
)
trainer.train()
trainer.save_model(NEXT); tok.save_pretrained(NEXT)
print(f"✓ DPO 完了 → {NEXT}/")
