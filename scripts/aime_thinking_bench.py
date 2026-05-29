#!/usr/bin/env python3
"""AIME-2024 thinking-mode bench against a running vLLM serve.

Produces JSON in the `benchmarks/rtxpro6000/cardb_aime30_c*_thinking.json` format:
- meta: timestamp, dataset, model, concurrency, max_tokens, thinking, n
- totals: correct, errors, wallclock_s, finish_reasons
- mtp_delta: d_drafts, d_draft_tokens, d_accepted_tokens, acceptance_rate, tokens_per_step

Usage:
  python3 aime_thinking_bench.py \
      --base-url http://127.0.0.1:8000 \
      --model DSV4-NVFP4-FP8-MTP \
      --concurrency 4 \
      --max-tokens 16384 \
      --out benchmarks/cardb_aime30_c4_thinking.json
"""

import argparse
import asyncio
import datetime as _dt
import json
import re
import time
from urllib.request import urlopen

import aiohttp
from datasets import load_dataset


ANSWER_RE = re.compile(r"\\boxed\{(\d+)\}|answer\s*(?:is|:)\s*(\d+)|####\s*(\d+)", re.IGNORECASE)


def extract_answer(text: str) -> int | None:
    if not text:
        return None
    for m in reversed(list(ANSWER_RE.finditer(text))):
        for g in m.groups():
            if g is not None:
                try:
                    return int(g)
                except ValueError:
                    pass
    nums = re.findall(r"\b(\d+)\b", text)
    return int(nums[-1]) if nums else None


def fetch_metrics(base_url: str) -> dict:
    try:
        text = urlopen(base_url + "/metrics", timeout=10).read().decode()
    except Exception:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        if not any(
            tag in line
            for tag in (
                "spec_decode",
                "draft_acceptance",
                "num_accepted",
                "num_draft",
                "num_emitted",
                "tokens_per_step",
            )
        ):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            pass
    return out


def metric_delta(before: dict, after: dict) -> dict:
    out = {}
    for k, v in after.items():
        out[k] = v - before.get(k, 0.0)
    return out


def find_first(delta: dict, needle: str) -> float | None:
    for k, v in delta.items():
        if needle in k:
            return v
    return None


async def one_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    thinking: "bool | str",
    semaphore: asyncio.Semaphore,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    # `thinking` is either True/False (legacy) or a string mode:
    # "none"/"chat" → chat mode, "low"/"medium"/"high"/"max" → reasoning_effort.
    # DS-V4 parser aliases low/medium → high, so the effective modes are
    # chat | high | max.
    if isinstance(thinking, str):
        mode = thinking.lower()
        if mode in ("none", "chat", "false", "off"):
            pass  # no chat_template_kwargs → chat mode
        else:
            payload["chat_template_kwargs"] = {
                "thinking": True,
                "reasoning_effort": mode,
            }
    elif thinking:
        payload["chat_template_kwargs"] = {"thinking": True}

    async with semaphore:
        t0 = time.time()
        try:
            async with session.post(
                base_url + "/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=3600),
            ) as resp:
                data = await resp.json()
            dt = time.time() - t0
            choice = data["choices"][0]
            msg = choice.get("message", {}) or {}
            content_field = msg.get("content") or ""
            # vllm's OpenAI serving layer renames reasoning_content → reasoning per
            # the OpenAI spec. On thinking-mode length-truncated responses content
            # is null because no </think> was emitted; the entire model output sits
            # in `reasoning`. Pull from reasoning when content is empty so we can
            # still extract a "Answer: NNN" the model may have written partway.
            reasoning_field = (
                msg.get("reasoning")
                or msg.get("reasoning_content")  # legacy field name
                or ""
            )
            content = content_field if content_field else reasoning_field
            return {
                "ok": True,
                "wall_s": dt,
                "finish_reason": choice.get("finish_reason"),
                "content": content,
                "reasoning_present": bool(reasoning_field),
                "content_present": bool(content_field),
                "reasoning_chars": len(reasoning_field),
                "content_chars": len(content_field),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "wall_s": time.time() - t0}


async def run(args):
    print(f"[aime] loading dataset Maxwell-Jia/AIME_2024", flush=True)
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    items = list(ds)
    if args.n and args.n < len(items):
        items = items[: args.n]
    mode_desc = args.thinking if isinstance(args.thinking, str) else ("high" if args.thinking else "chat")
    print(f"[aime] {len(items)} problems, concurrency={args.concurrency}, max_tokens={args.max_tokens}, mode={mode_desc}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    metrics_before = fetch_metrics(args.base_url)

    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [
            one_request(
                session,
                args.base_url,
                args.model,
                f"{p['Problem']}\n\nGive your final answer as a single integer 0-999, on its own line, prefixed by 'Answer: '.",
                args.max_tokens,
                args.thinking,
                sem,
            )
            for p in items
        ]
        results = await asyncio.gather(*tasks)
    wall = time.time() - t0
    metrics_after = fetch_metrics(args.base_url)

    correct = 0
    errors = 0
    finish_reasons: dict[str, int] = {}
    details = []
    for p, r in zip(items, results):
        if not r["ok"]:
            errors += 1
            details.append({"id": p.get("ID", ""), "ok": False, "error": r.get("error")})
            continue
        fr = r.get("finish_reason") or "unknown"
        finish_reasons[fr] = finish_reasons.get(fr, 0) + 1
        pred = extract_answer(r["content"])
        gt = int(p["Answer"]) if "Answer" in p else None
        ok = (gt is not None and pred is not None and pred == gt)
        if ok:
            correct += 1
        details.append({
            "id": p.get("ID", ""),
            "ok": ok,
            "finish_reason": fr,
            "wall_s": r["wall_s"],
            "completion_tokens": r.get("completion_tokens", 0),
            "reasoning_present": r.get("reasoning_present"),
            "content_present": r.get("content_present"),
            "reasoning_chars": r.get("reasoning_chars"),
            "content_chars": r.get("content_chars"),
            "predicted": pred,
            "ground_truth": gt,
        })

    delta = metric_delta(metrics_before, metrics_after)
    d_drafts = find_first(delta, "num_draft")
    d_accepted = find_first(delta, "num_accepted")
    d_emitted = find_first(delta, "num_emitted")
    acceptance: float | None = None
    if d_drafts and d_drafts > 0 and d_accepted is not None:
        acceptance = d_accepted / d_drafts
    tokens_per_step: float | None = None
    if d_emitted and d_emitted > 0 and d_accepted is not None:
        tokens_per_step = (d_accepted + d_emitted) / d_emitted

    out = {
        "meta": {
            "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
            "dataset": "Maxwell-Jia/AIME_2024",
            "model": args.model,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "thinking": args.thinking,
            "n": len(items),
        },
        "totals": {
            "correct": correct,
            "errors": errors,
            "wallclock_s": wall,
            "finish_reasons": finish_reasons,
        },
        "mtp_delta": {
            "d_drafts": d_drafts,
            "d_draft_tokens": d_drafts,
            "d_accepted_tokens": d_accepted,
            "acceptance_rate": acceptance,
            "tokens_per_step": tokens_per_step,
        },
        "details": details,
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[aime] wrote {args.out}", flush=True)
    accept_s = f"{acceptance:.4f}" if acceptance is not None else "n/a"
    print(
        f"[aime] correct={correct}/{len(items)} errors={errors} "
        f"wall={wall:.0f}s mtp_accept={accept_s}",
        flush=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--thinking", action="store_true", default=True)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.add_argument(
        "--reasoning-effort",
        choices=["none", "chat", "low", "medium", "high", "max"],
        default=None,
        help="When set, overrides --thinking/--no-thinking. 'none'/'chat' = no thinking; "
             "DS-V4 parser aliases low/medium → high, so the effective distinct modes "
             "are chat | high | max.",
    )
    ap.add_argument("--n", type=int, default=0, help="0 = all 30")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.reasoning_effort is not None:
        args.thinking = args.reasoning_effort
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
