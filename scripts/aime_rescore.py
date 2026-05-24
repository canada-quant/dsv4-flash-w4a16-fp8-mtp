#!/usr/bin/env python3
"""Re-score an AIME bench output with correct \\boxed{N} extraction.
The original aime_bench.py has a buggy regex (r'\\\\\\\\boxed\\\\{(\\\\d+)\\\\}'
matches literal backslashes, not regex digit class). We salvage the existing
bench JSON by re-running extraction on the saved generations."""
import json, re, sys, glob

# Correct patterns
PATTERNS = [
    re.compile(r'\\boxed\{(-?\d+)\}'),          # \boxed{N}
    re.compile(r'\\boxed\{\\?frac\{(-?\d+)\}\{(-?\d+)\}\}'),  # \boxed{\frac{a}{b}} - returns (a,b)
    re.compile(r'final answer[^\d-]*(-?\d{1,4})', re.IGNORECASE),
    re.compile(r'answer is[^\d-]*(-?\d{1,4})', re.IGNORECASE),
    re.compile(r'\b(-?\d{1,3})\b\s*$'),
]

def extract(text):
    if not text: return None
    for pat in PATTERNS:
        m = pat.findall(text)
        if m:
            last = m[-1]
            try:
                if isinstance(last, tuple):
                    # \frac{a}{b} - prefer answers expressed as m+n where m and n are coprime
                    # AIME often asks for m+n; sum them
                    return int(last[0]) + int(last[1])
                return int(last)
            except: continue
    return None

def main(path):
    raw = open(path).read()
    idx = raw.find('{')
    d = json.loads(raw[idx:])
    n = d['n_total']
    new_correct = 0
    flipped = []
    for r in d['results']:
        new_pred = extract(r['content'])
        try:
            expected = int(str(r['expected']).strip())
        except:
            expected = None
        ok = (new_pred is not None and expected is not None and new_pred == expected)
        if ok != r['correct']:
            flipped.append((r['id'], r['predicted'], new_pred, r['expected'], 'NOW_OK' if ok else 'WAS_OK_NOW_NO'))
        if ok:
            new_correct += 1
        r['rescored_predicted'] = new_pred
        r['rescored_correct'] = ok

    trunc = sum(1 for r in d['results'] if r['finish']=='length')
    errs = sum(1 for r in d['results'] if r.get('finish')=='error')
    nt = [r for r in d['results'] if r['finish'] not in ('length','error')]
    nt_correct = sum(1 for r in nt if r['rescored_correct'])

    print(f"=== Re-scored {path} ===")
    print(f"  original  pass@1 = {d['n_correct']}/{n} = {d['n_correct']/n*100:.2f}%")
    print(f"  rescored  pass@1 = {new_correct}/{n} = {new_correct/n*100:.2f}%")
    print(f"  rescored  non-truncated pass@1 = {nt_correct}/{len(nt)} = {(nt_correct/len(nt)*100 if nt else 0):.2f}%")
    print(f"  truncated/errored = {trunc}/{errs}/{n}")
    print(f"  MTP acceptance = {d['spec_decode'].get('acceptance_pct')}%")
    print(f"  wall_clock = {d['wall_clock_s']:.0f}s, avg_tokens = {d['avg_completion_tokens']:.0f}")
    print(f"  flipped entries (original->rescored): {len(flipped)}")
    if flipped[:5]:
        print("  sample flips:")
        for f in flipped[:5]:
            print(f"    {f}")
    # write rescored
    out_path = path.replace('.json', '.rescored.json')
    with open(out_path, 'w') as f:
        json.dump(d, f, indent=2)
    print(f"\n  rescored JSON saved: {out_path}")

if __name__ == '__main__':
    paths = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob('/tmp/d-rebench/aime24_a937_*.json'))
    for p in paths:
        main(p)
