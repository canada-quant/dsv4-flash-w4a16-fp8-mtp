# Bench matrix — tp2_64k (2026-05-29T09:09:18Z)
Endpoint: http://127.0.0.1:8000  Model: DSV4-W4A16-FP8-MTP  max_model_len=65536  AIME max_tokens=65036

## AIME-2024 thinking-mode sweep at c=4 (max_tokens=65036, n=30)
| mode | correct/30 | errors | stop | length | MTP accept | wall s |
|---|---|---|---|---|---|---|
| chat | 18/30 | 0 | 29 | 1 | 95.78% | 3175 |
| high | 27/30 | 2 | 27 | 1 | 91.97% | 9161 |
| max | 24/30 | 5 | 24 | 1 | 92.52% | 10583 |

## AIME-2024 c=1 single-shot (mode=high, max_tokens=65036)
| c | correct/30 | errors | stop | length | MTP accept | wall s |
|---|---|---|---|---|---|---|
| 1 | 27/30 | 0 | 28 | 2 | 91.68% | 2894 |

## GSM8K-50 c=8 (strict-match, 8-shot)
    see /workspace/bench-out/tp2_64k_gsm8k50_c8.log

## Throughput sweep — random 256/256 (MTP-on)
| bs | output tok/s | TPOT median ms | TPOT p99 ms | wall s |
|---|---|---|---|---|
| 1 | 95.2 | 8.05 | 8.64 | 10.8 |
| 4 | 40.6 | 83.18 | 146.41 | 100.8 |
| 8 | 45.7 | 79.02 | 123.21 | 179.4 |
| 16 | 34.9 | 81.77 | 237.23 | 469.7 |

## Throughput sweep — random 1024/1024 (MTP-on)
| bs | output tok/s | TPOT median ms | TPOT p99 ms | wall s |
|---|---|---|---|---|
| 1 | 30.7 | 7.90 | 63.60 | 133.3 |
| 4 | 45.1 | 86.43 | 98.41 | 363.2 |

