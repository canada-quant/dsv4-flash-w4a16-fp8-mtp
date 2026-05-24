#!/usr/bin/env python3
"""GSM8K-50 chat-mode sanity bench. Greedy, no thinking, sequential (concurrency=1)
to avoid the SM 12.x cuda-graphs+thinking race that crashes under load.
Saves results JSON for inclusion as evidence on the dequant'd-artifact card."""
import json, urllib.request, re, time, sys
from datasets import load_dataset

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "DSV4-W4A16-FP8-MTP"
N = 50

ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42).select(range(N))
print(f"Loaded {len(ds)} GSM8K problems", file=sys.stderr)

results = []
t0 = time.monotonic()
for i, row in enumerate(ds):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": row["question"] + "\n\nGive the final number as the last thing in your answer."}],
        "max_tokens": 1024,
        "temperature": 0,
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        out = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
    except Exception as e:
        out = ""
        usage = {}
        print(f"[{i+1}/{N}] ERROR {e}", file=sys.stderr)
    expected = row["answer"].split("####")[-1].strip().replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", out)
    pred = nums[-1].replace(",", "") if nums else None
    ok = (pred == expected) if pred is not None else False
    results.append({"idx": i, "expected": expected, "predicted": pred, "correct": ok,
                    "completion_tokens": usage.get("completion_tokens", 0)})
    if (i + 1) % 5 == 0 or i < 3:
        nc = sum(1 for r in results if r['correct'])
        print(f"[{i+1}/{N}] running {nc}/{i+1} = {nc/(i+1)*100:.1f}% (last exp={expected} pred={pred} {'OK' if ok else 'NO'})", file=sys.stderr)
wall = time.monotonic() - t0
correct = sum(1 for r in results if r['correct'])
summary = {
    "model": MODEL, "n": N, "correct": correct, "accuracy": correct / N,
    "wall_clock_s": wall,
    "avg_completion_tokens": sum(r['completion_tokens'] for r in results) / N,
    "methodology": "chat-mode, no thinking, concurrency=1, greedy, last-number extraction",
    "results": results,
}
print(json.dumps(summary, indent=2))
print(f"\nGSM8K-50: {correct}/{N} = {correct/N*100:.1f}% ({wall:.0f}s)", file=sys.stderr)
