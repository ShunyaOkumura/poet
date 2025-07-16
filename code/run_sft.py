from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, BitsAndBytesConfig

BASE = "tokyotech-llm/Swallow-7b-plus-hf"

# 1. トークナイザー
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.pad_token = tok.eos_token

# 2. データ読み込み
ds = load_dataset("json", data_files="sft_poems.jsonl", split="train")
ds = ds.map(lambda x: {"text": x["text"]})

# 3. モデル & 4bit 設定
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

# 4. トレーニング引数
training_args = TrainingArguments(
    output_dir="sft-out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=8,      # 例: 6 epoch に
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="epoch"
)

# 5. SFTTrainer の初期化
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    # tokenizer は内部で model に紐づくので不要
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1
)

# 6. 学習実行
trainer.train()
trainer.save_model("sft-out")
tok.save_pretrained("sft-out")
print("✓ SFT 完了 → sft-out/")
