import json, collections, random
by = collections.defaultdict(list)
for ln in open("logs/rated.jsonl", encoding="utf-8"):
    d=json.loads(ln); by[d["theme"]].append(d)

pairs=[]
for theme,items in by.items():
    items.sort(key=lambda x:x["score"], reverse=True)
    hi, lo = 0, len(items)-1
    while hi < lo and items[hi]["score"]>items[lo]["score"]:
        pairs.append({"prompt": theme,
                      "chosen": items[hi]["poem"],
                      "rejected": items[lo]["poem"]})
        hi += 1; lo -= 1
random.shuffle(pairs)
json.dump(pairs, open("dpo_pairs.jsonl","w",encoding="utf-8"), ensure_ascii=False)
print("✓ 作ったペア:",len(pairs))
