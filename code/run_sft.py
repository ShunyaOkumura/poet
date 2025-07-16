from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,          # import from transformers
)
from trl import SFTTrainer        # tokenizer no longer passed here

BASE = "tokyotech-llm/Swallow-7b-plus-hf"

# 1. Tokenizer
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.pad_token = tok.eos_token

# 2. Dataset
ds = load_dataset("json", data_files="sft_poems.jsonl", split="train")
ds = ds.map(lambda x: {"text": x["text"]})

# 3. Model & 4-bit config
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,
    device_map="auto"
)

# 4. TrainingArguments
training_args = TrainingArguments(
    output_dir="sft-out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=8,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="epoch"
)

# 5. SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1
)

# 6. Run
trainer.train()
trainer.save_model("sft-out")
tok.save_pretrained("sft-out")
print("✓ SFT 完了 → sft-out/")
