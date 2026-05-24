# Context-length sweep (RTX 6000 Pro TP=2 cudagraph, 2026-05-24)

Tried `vllm bench serve` with input lengths 256, 2048, 8192, 32000
(output 256, bs=1, 8 prompts). Serve config:
`--max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 8192
--gpu-memory-utilization 0.95 --speculative-config k=1 --disable-custom-all-reduce`.

## Results

| Input length | Output tok/s | TPOT median (ms) | TTFT median (ms) | Completed / total | MTP acceptance | Status |
|---|---|---|---|---|---|---|
| 256 | **96.31** | 8.72 | 183.0 | 8/8 | 71.83% | ✅ clean run |
| 2048 | 0.07 | 6.40 | 6726 | 1/8 | 100% (n=1) | ⚠️ 7 of 8 returned HTTP 500 |
| 8192 | 0.00 | 0.00 | 0.00 | 0/8 | n/a | ❌ serve died first; client got `Cannot connect to host localhost:8000` |
| 32000 | 0.00 | 0.00 | 0.00 | 0/8 | n/a | ❌ same — serve was dead by this point |

## What happened

The serve process accepted requests at in=256 cleanly (matches the
baseline `tp2_2026-05-24T010311Z` headline numbers within noise). At
in=2048, **one** request succeeded and **seven** returned HTTP 500 from
the API server. Between in=2048 and in=8192 the serve process shut
down completely (`INFO: Shutting down` in the log) — by the time the
in=8192 client tried, the server socket was closed.

## Hypothesis (not fully validated)

Most likely a **chunked-prefill + spec-decode interaction** at long
contexts on the SM 12.0 sparse-MLA path. With `max-num-batched-tokens
8192`, a 32k prompt gets split into 4 prefill chunks. The MTP
spec-decode drafter requires the full prefill to complete before it
can run; the mid-chunk state of chunked prefill + spec-decode together
may have an edge case the SM 12.0 sparse-MLA path doesn't handle.

Other plausible causes (not investigated):
- KV-cache eviction race when long-prompt allocations exceed the
  `max_num_seqs=4` budget × 32768 = 131072 token KV slots
- The `--no-enable-prefix-caching` flag combined with long prompts
  re-allocating KV blocks on every retry
- Some Triton kernel autotune bailing on shapes >8K

## What works

`vllm bench serve` at in=256 with TP=2 cudagraph + MTP-spec k=1
**works cleanly**. Numbers match the baseline run on the same TP=2
config:

| Metric | 2026-05-24 headline (in=256) | This sweep (in=256) |
|---|---|---|
| output_throughput tok/s | 98.83 | 96.31 |
| TPOT median (ms) | 8.55 | 8.72 |
| MTP acceptance | 71.39% | 71.83% |

So **the dynamo-safe cudagraph stack is stable for the standard
serving regime**. The long-context failure is a separate, narrower
issue at >2K input.

## What to investigate next

1. **Disable spec-decode and re-test long contexts.** If the long-input
   500 errors go away without MTP, that confirms the chunked-prefill +
   MTP interaction hypothesis.
2. **Try max-num-batched-tokens = 16384 or 32768** (matching
   max-model-len). Eliminates chunked prefill for prompts under the
   max-model-len; tests whether chunked prefill is the trigger.
3. **Check the server-side traceback** for the 500s — they're in
   `/tmp/serve_longctx.log` lines marked `POST /v1/completions HTTP/1.1 500`.
4. **Try in=4096 between 2048 and 8192** — narrow the failure boundary.

These are follow-up items, not blockers for the published headline
numbers.

## Raw JSONs

`bench_mtp_in{256,2048,8192,32000}_bs1.json` in this directory.
