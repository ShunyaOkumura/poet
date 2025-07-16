from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer

BASE = "tokyotech-llm/Swallow-7b-plus-hf"

# トークナイザー
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.pad_token = tok.eos_token

# データセット
ds = load_dataset("json", data_files="sft_poems.jsonl", split="train")
ds = ds.map(lambda x: {"text": x["text"]})

# モデル＆4bit量子化設定
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,
    device_map="auto",
)

# LoRA 設定
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],  # Llama系なら q_proj/v_proj など
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# TrainingArguments
training_args = TrainingArguments(
    output_dir="sft-out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=8,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="epoch",
)

# SFTTrainer の初期化
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    peft_config=peft_config,
    args=training_args,
    tokenizer=tok,
)

# 学習開始
trainer.train()
trainer.save_model("sft-out")
tok.save_pretrained("sft-out")
print("✓ SFT 完了 → sft-out/")
