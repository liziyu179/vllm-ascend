# Layerwise vs Nonlayerwise Log Analysis

Data source:
- `vllm-ascend/vllm-layerwise/test_proxy.log`
- `vllm-ascend/vllm-layerwise/test_p.log`
- `vllm-ascend/vllm-layerwise/test_d.log`
- `vllm-ascend/vllm-nonlayerwise/test_proxy.log`
- `vllm-ascend/vllm-nonlayerwise/test_p.log`
- `vllm-ascend/vllm-nonlayerwise/test_d.log`

Matching rule:
- Use proxy request UUID as the request key.
- Match `chatcmpl-<uuid>-<suffix>` in `test_p.log` and `test_d.log`.
- `worker exec` duration is computed as `max(after worker exec) - min(before worker exec)` for the same request.

Sample count:
- `layerwise`: 100 requests
- `nonlayerwise`: 100 requests

## Mean Time (ms)

| Metric | layerwise | nonlayerwise | delta(non-layer - layer) |
| --- | ---: | ---: | ---: |
| Proxy request -> first token | 324.13 | 292.08 | -32.05 |
| Proxy -> P schedule get | 43.01 | 36.26 | -6.75 |
| Proxy -> D schedule get | 40.06 | 34.41 | -5.65 |
| P queue (`start sched - get req`) | 19.80 | 0.12 | -19.68 |
| P exec | 210.09 | 183.99 | -26.10 |
| D queue (`start sched - get req`) | 252.75 | 226.12 | -26.63 |
| D exec | 15.59 | 16.40 | +0.81 |
| P end -> D start | 19.35 | 39.71 | +20.36 |
| D end -> proxy first token | 15.30 | 14.67 | -0.63 |

## Trimmed Mean (drop top/bottom 5%) (ms)

This view is more robust because both runs contain several long-tail requests.

| Metric | layerwise | nonlayerwise | delta(non-layer - layer) |
| --- | ---: | ---: | ---: |
| Proxy request -> first token | 295.77 | 276.11 | -19.66 |
| Proxy -> P schedule get | 43.99 | 35.88 | -8.11 |
| Proxy -> D schedule get | 27.99 | 21.59 | -6.40 |
| P queue | 4.59 | 0.08 | -4.51 |
| P exec | 195.32 | 181.57 | -13.76 |
| D queue | 236.21 | 222.87 | -13.34 |
| D exec | 15.58 | 16.40 | +0.82 |
| P end -> D start | 19.27 | 26.74 | +7.48 |
| D end -> proxy first token | 15.29 | 14.66 | -0.63 |

## Main Findings

1. In this batch of logs, `nonlayerwise` is faster for first-token latency.
   - Mean: `292.08 ms` vs `324.13 ms`
   - Trimmed mean: `276.11 ms` vs `295.77 ms`

2. The largest stable advantage of `nonlayerwise` is on the P side, not the D execution itself.
   - `P exec` is about `13.76 ms` faster in trimmed mean.
   - `P queue` is about `4.51 ms` lower in trimmed mean.

3. `D exec` is almost the same in both modes.
   - `layerwise`: `15.58 ms`
   - `nonlayerwise`: `16.40 ms`
   - Difference is less than `1 ms`.

4. `D queue` is also lower in `nonlayerwise`, by about `13.34 ms` in trimmed mean.
   - This suggests the time from D receiving the request to actually starting scheduling is not improved by the current `layerwise` path in these logs.

5. `layerwise` only shows an advantage in the handoff gap after P finishes.
   - `P end -> D start`:
   - `layerwise`: `19.27 ms`
   - `nonlayerwise`: `26.74 ms`
   - So `layerwise` saves about `7.48 ms` here.
   - But that gain is smaller than the losses on `P exec` and `D queue`, so it does not translate into a lower end-to-end latency.

## Where The Performance Gap Comes From

Using trimmed mean as the main reference:

- `nonlayerwise` wins about `19.66 ms` on first-token latency overall.
- The biggest contributors are:
  - `P exec`: `-13.76 ms`
  - `D queue`: `-13.34 ms`
  - `P queue`: `-4.51 ms`
- `layerwise` gets back only:
  - `P end -> D start`: `+7.48 ms` advantage over `nonlayerwise`

So the current bottleneck difference is:
- `layerwise` does reduce the final handoff gap from P to D,
- but it appears to make P slower and does not reduce D's wait-to-schedule enough,
- therefore the overall first-token latency is still worse.

## Notable Long-Tail Requests

These requests strongly skew the plain mean and are worth checking separately.

### layerwise

- `a4ef26e2-4189-45cd-ba43-9c58f26e5520`
  - proxy first token: `1612 ms`
  - P exec: `887 ms`
  - D queue: `931 ms`
  - proxy -> D get: `649 ms`
- `7ef9159e-eb80-4bac-80e5-f5425942bc0a`
  - proxy first token: `1206 ms`
  - P exec: `976 ms`
  - D queue: `1117 ms`
- `dd3d4ae2-da53-4d18-8b5d-04bd34bf5bc1`
  - proxy -> D get: `637 ms`

### nonlayerwise

- `f7006583-068c-424b-b372-341283341027`
  - proxy first token: `997 ms`
  - proxy -> D get: `661 ms`
  - P end -> D start: `699 ms`
- `f7e44d82-ff85-4277-8891-1a13bc061dd1`
  - proxy first token: `712 ms`
  - proxy -> D get: `650 ms`
  - P end -> D start: `456 ms`
- `13930b81-cb6f-4966-af41-9503a13a6265`
  - proxy first token: `653 ms`
  - P exec: `340 ms`
  - D queue: `570 ms`

## Suggested Next Checks

1. Focus on why `layerwise` increases `P exec`.
   - This is the largest stable regression.
2. Check why `D queue` is still high in `layerwise`.
   - If layerwise push is working as intended, D should ideally start earlier or wait less.
3. Inspect the long-tail requests above first.
   - They may indicate warmup, scheduling stalls, or delayed log/reporting on D.
