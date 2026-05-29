# Bench matrix — tp4_64k (2026-05-29T17:09:42Z)
Endpoint: http://127.0.0.1:8000  Model: DSV4-W4A16-FP8-MTP  max_model_len=65536  AIME max_tokens=65036

## AIME-2024 thinking-mode sweep at c=4 (max_tokens=65036, n=30)
| mode | correct/30 | errors | stop | length | MTP accept | wall s |
|---|---|---|---|---|---|---|
| chat | 19/30 | 0 | 30 | 0 | 93.06% | 298 |
| high | 29/30 | 0 | 29 | 1 | 91.01% | 784 |
| max | 27/30 | 0 | 25 | 5 | 91.68% | 1548 |

## AIME-2024 c=1 single-shot (mode=high, max_tokens=65036)
| c | correct/30 | errors | stop | length | MTP accept | wall s |
|---|---|---|---|---|---|---|
| 1 | 28/30 | 0 | 28 | 2 | 90.76% | 2481 |

## GSM8K-50 c=8 (strict-match, 8-shot)
    |gsm8k|      3|flexible-extract|     8|exact_match|↑  | 0.86|±  |0.0496|
    |     |       |strict-match    |     8|exact_match|↑  | 0.80|±  |0.0571|
    

## Throughput sweep — random 256/256 (MTP-on)
| bs | output tok/s | TPOT median ms | TPOT p99 ms | wall s |
|---|---|---|---|---|
| 1 | 108.1 | 7.32 | 7.88 | 9.5 |
| 4 | 104.3 | 11.31 | 63.03 | 39.3 |
| 8 | 433.2 | 16.44 | 20.46 | 18.9 |
| 16 | 164.3 | 90.18 | 111.40 | 99.7 |

## Throughput sweep — random 1024/1024 (MTP-on)
| bs | output tok/s | TPOT median ms | TPOT p99 ms | wall s |
|---|---|---|---|---|
| 1 | 138.1 | 6.93 | 7.34 | 29.7 |
| 4 | 363.7 | 10.18 | 11.75 | 45.0 |

