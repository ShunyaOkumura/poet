from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import glob, re

latest = sorted(glob.glob("dpo-out-*"), key=lambda s:int(re.findall(r'\d+',s)[-1]))[-1]
tok = AutoTokenizer.from_pretrained(latest); tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    latest,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                           bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype="bfloat16"),
    device_map="auto")

def is_ok(poem, n=15, lines=4):
    L=[l.strip() for l in poem.splitlines() if l.strip()]
    return len(L)<=lines and all(len(l)<=n for l in L)

def generate(theme,trials=5):
    prompt=f"15字以内4行で詩を書け。テーマ: {theme}\n### 詩\n"
    for _ in range(trials):
        ids=tok(prompt,return_tensors="pt").to(model.device)
        out=model.generate(**ids,max_new_tokens=80,do_sample=True,
                           temperature=0.9,top_p=0.95)
        poem=tok.decode(out[0,ids['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        if is_ok(poem): return poem
    return poem

while True:
    print("\n─ 詩 ─\n"+generate(input("テーマ > ").strip()))
