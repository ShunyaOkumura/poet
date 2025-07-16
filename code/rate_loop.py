import json, datetime, glob, re
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)

# 最新 dpo-out-* or sft-out をロード
latest = sorted(glob.glob("dpo-out-*"), key=lambda s:int(re.findall(r'\d+',s)[-1]))[-1] \
         if glob.glob("dpo-out-*") else "sft-out"

tok = AutoTokenizer.from_pretrained(latest); tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    latest,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                           bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype="bfloat16"),
    device_map="auto")

def gen_one(theme):
    prompt = f"15字以内4行で詩を書け。テーマ: {theme}\n### 詩\n"
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, do_sample=True, temperature=0.9,
                         top_p=0.95, max_new_tokens=80)
    return tok.decode(out[0, ids['input_ids'].shape[1]:],
                      skip_special_tokens=True).strip()

logf = open("logs/rated.jsonl","a",encoding="utf-8")
while True:
    theme = input("\n★ テーマ (Ctrl‑C 終了) > ").strip()
    poem  = gen_one(theme)
    print("\n── 詩 ──\n", poem)
    try:
        score = int(input("点数 (-3..3) > ") or "0")
        score = max(-3, min(3, score))
    except ValueError:
        score = 0
    logf.write(json.dumps({"theme":theme,"poem":poem,"score":score,
                           "time":datetime.datetime.now().isoformat()},
                          ensure_ascii=False)+"\n"); logf.flush()
