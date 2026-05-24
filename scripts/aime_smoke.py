#!/usr/bin/env python3
"""1-shot AIME smoke. Pulls problem #1 from Maxwell-Jia/AIME_2024, chat-templates with thinking=high,
extracts integer from \\boxed{...}, prints pass/fail. Used as a gate before full bench."""
import argparse, asyncio, json, re, sys
from openai import AsyncOpenAI
from datasets import load_dataset

ANSWER_PATTERNS = [
    re.compile(r'\\boxed\{(-?\d+)\}'),
    re.compile(r'final answer[^\d-]*(-?\d{1,4})', re.IGNORECASE),
    re.compile(r'answer is[^\d-]*(-?\d{1,4})', re.IGNORECASE),
    re.compile(r'\b(-?\d{1,4})\b\s*$'),
]

def extract(text):
    if not text: return None
    for pat in ANSWER_PATTERNS:
        m = pat.findall(text)
        if m:
            try: return int(m[-1])
            except: continue
    return None

async def main(args):
    ds = load_dataset(args.dataset, split='train')
    row = ds[args.idx]
    print(f"problem id={row.get('ID', row.get('id'))} expected={row['Answer']}", file=sys.stderr)
    client = AsyncOpenAI(base_url=args.base_url, api_key='dummy', timeout=600)
    resp = await client.chat.completions.create(
        model=args.model,
        messages=[{'role':'user','content': row['Problem'] + '\n\nReason step-by-step then place your final integer answer (0-999) inside \\boxed{...}.'}],
        max_tokens=args.max_tokens,
        temperature=0.0,
        extra_body={'chat_template_kwargs':{'thinking':True,'reasoning_effort':'high'}} if args.thinking else None,
    )
    msg = resp.choices[0].message
    content = (msg.content or '') + (getattr(msg, 'reasoning', None) or '')
    pred = extract(content)
    expected = int(str(row['Answer']).strip())
    finish = resp.choices[0].finish_reason
    tok = resp.usage.completion_tokens
    print(f"finish={finish} completion_tokens={tok}", file=sys.stderr)
    print(f"prediction={pred} expected={expected} correct={pred==expected}", file=sys.stderr)
    print(f"--- last 400 chars of generation ---", file=sys.stderr)
    print(content[-400:], file=sys.stderr)
    # Exit code: 0 if got a non-None prediction (gate passes), 1 if None (gate fails)
    sys.exit(0 if pred is not None else 1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://localhost:8000/v1')
    ap.add_argument('--model', default='DSV4-W4A16-FP8-MTP')
    ap.add_argument('--dataset', default='Maxwell-Jia/AIME_2024')
    ap.add_argument('--idx', type=int, default=0)
    ap.add_argument('--thinking', action='store_true', default=True)
    ap.add_argument('--max-tokens', type=int, default=32768)
    asyncio.run(main(ap.parse_args()))
