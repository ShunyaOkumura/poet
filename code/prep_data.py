# 1) 520 poems  → sft_poems.jsonl
import json, uuid, random, os

# ---------- SFT 用 ----------
poems = [p.strip() for p in open("C:\\Users\\manis\\OneDrive\\Haiku\\Gen\\source\\poems.txt", encoding="utf-8")
         .read().split("---作品区切り---") if p.strip()]
with open("sft_poems.jsonl", "w", encoding="utf-8") as f:
    for p in poems:
        f.write(json.dumps({"id": str(uuid.uuid4()), "text": p},
                           ensure_ascii=False) + "\n")
print("✓ sft_poems.jsonl を生成")

# ---------- DPO 用 ----------
if os.path.exists("logs/rated.jsonl"):
    chosen, rejected = [], []
    for ln in open("logs/rated.jsonl", encoding="utf-8"):
        d = json.loads(ln)
        (chosen if d["score"] >= 0 else rejected).append(d)
    random.shuffle(chosen); random.shuffle(rejected)
    pairs = [{"prompt": "詩を書いてください。",
              "chosen": c["poem"],
              "rejected": r["poem"]}
             for c, r in zip(chosen[:len(rejected)], rejected)]
    with open("dpo_pairs.jsonl", "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False)
    print(f"✓ dpo_pairs.jsonl を生成 ({len(pairs)} ペア)")
else:
    print("logs/rated.jsonl がまだ無いので DPO ペアは作成しません")
