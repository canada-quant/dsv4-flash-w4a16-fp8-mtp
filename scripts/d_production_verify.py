#!/usr/bin/env python3
"""Verify Card D dequant'd artifact under PRODUCTION config (cuda graphs ON, MTP enabled, sequential).
Captures spec_decode acceptance% from vLLM Prometheus metrics before/after a short workload.
"""
import json, urllib.request, re, sys
from datasets import load_dataset

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "DSV4-W4A16-FP8-MTP"
PROM = "http://localhost:8000/metrics"

def chat(prompt, max_tokens=512):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return resp["choices"][0]["message"]["content"], resp.get("usage", {})

def fetch_prom():
    try:
        return urllib.request.urlopen(PROM, timeout=5).read().decode()
    except Exception:
        return ""

def parse_metric(txt, name):
    total = 0.0
    for line in txt.splitlines():
        if line.startswith(name + '{') or line.startswith(name + ' '):
            try: total += float(line.rsplit(' ', 1)[1])
            except: pass
    return total

# Baseline metrics
p0 = fetch_prom()
d0 = parse_metric(p0, 'vllm:spec_decode_num_drafts_total')
dt0 = parse_metric(p0, 'vllm:spec_decode_num_draft_tokens_total')
at0 = parse_metric(p0, 'vllm:spec_decode_num_accepted_tokens_total')
print(f"baseline drafts={d0:.0f} draft_tokens={dt0:.0f} accepted={at0:.0f}", flush=True)

# 3 chat prompts smoke
print("\n=== 3 chat prompts (cuda graphs ON, MTP enabled) ===", flush=True)
for q in ["What is 17*23?", "What is the capital of France?", "Write a Python function to reverse a string."]:
    print(f"Q: {q}", flush=True)
    try:
        a, u = chat(q, 200)
        print(f"A: {a[:200]}", flush=True)
        print(f"  completion_tokens={u.get('completion_tokens')}", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

# GSM8K-20 sequential
print("\n=== GSM8K-20 chat-mode sequential (cuda graphs ON, MTP enabled) ===", flush=True)
ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42).select(range(20))
correct = 0
total_tok = 0
for i, row in enumerate(ds):
    try:
        out, u = chat(row["question"] + "\n\nGive the final number as the last thing in your answer.", 1024)
        expected = row["answer"].split("####")[-1].strip().replace(",", "")
        nums = re.findall(r"-?\d[\d,]*", out)
        pred = nums[-1].replace(",", "") if nums else None
        ok = (pred == expected)
        correct += int(ok)
        total_tok += u.get('completion_tokens', 0)
        print(f"[{i+1}/20] exp={expected} pred={pred} tok={u.get('completion_tokens')} {'OK' if ok else 'NO'}", flush=True)
    except Exception as e:
        print(f"[{i+1}/20] ERROR {e}", flush=True)

# Final metrics
p1 = fetch_prom()
d1 = parse_metric(p1, 'vllm:spec_decode_num_drafts_total')
dt1 = parse_metric(p1, 'vllm:spec_decode_num_draft_tokens_total')
at1 = parse_metric(p1, 'vllm:spec_decode_num_accepted_tokens_total')
delta_d = d1 - d0
delta_dt = dt1 - dt0
delta_at = at1 - at0
acc = (delta_at / delta_dt * 100) if delta_dt > 0 else None
print(f"\nfinal drafts={d1:.0f} draft_tokens={dt1:.0f} accepted={at1:.0f}", flush=True)
print(f"delta drafts={delta_d:.0f} draft_tokens={delta_dt:.0f} accepted={delta_at:.0f}", flush=True)
print(f"\n=== Production-config verification ===", flush=True)
print(f"GSM8K-20: {correct}/20 = {correct*5}%", flush=True)
print(f"Total completion tokens: {total_tok}", flush=True)
print(f"MTP draft acceptance:  {acc:.2f}%" if acc is not None else "MTP draft acceptance: n/a", flush=True)
print(f"Drafts emitted: {int(delta_d)}, draft tokens: {int(delta_dt)}, accepted: {int(delta_at)}", flush=True)
print(f"  -> MTP {'IS' if delta_d > 0 else 'IS NOT'} firing under cuda graphs ON", flush=True)
