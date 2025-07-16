from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

BASE = "tokyotech-llm/Swallow-7b-plus-hf"
tok  = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token

ds = load_dataset("json", data_files="sft_poems.jsonl", split="train")
ds = ds.map(lambda x: {"text": x["text"]})

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    quantization_config={"load_in_4bit": True}  # 新しい記法
)

training_args = TrainingArguments(
    output_dir="sft-out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=7,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="epoch"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    tokenizer=tok,  # 最新の `trl` では TrainingArgumentsと同階層に移動
    args=training_args,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1
)

trainer.train()
trainer.save_model("sft-out")
tok.save_pretrained("sft-out")
print("✓ SFT 完了 → sft-out/")
