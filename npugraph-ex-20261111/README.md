# Qwen3-8B TP1 benchmark, 2026-11-11

Source: 0b45c3eb1091665b5851bceb10f6e9f3af250ebc, based on upstream 43d0ecf9.
Ascend A3, BF16, TP1, fixed random input/output 1024/512, 160 requests, concurrency 32, seed 42.
A, ex, and static each use a fresh server process. Within each process, the identical benchmark command runs three times without restarting the server.
Pass 1 is the discarded full warmup. Passes 2 and 3 are measured. The PR main table and images select pass 3; all three passes are disclosed below.
Profiler is off. CPU governor performance, swappiness=0, numa_balancing=0, sched_migration_cost_ns=50000.

| Pass | A output tokens/s | A mean TPOT ms | ex output tokens/s | ex mean TPOT ms | static output tokens/s | static mean TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (warmup) | 1028.401 | 27.659 | 1026.909 | 27.703 | 1038.489 | 27.361 |
| 2 | 1032.549 | 27.679 | 1030.491 | 27.732 | 1039.200 | 27.475 |
| 3 (selected) | 1024.034 | 27.937 | 1030.336 | 27.743 | 1040.487 | 27.441 |

One independent service process per route. These results do not establish a stable speedup across restarts or a benefit from changing the migration setting.
Static operator packages were reused from a validated cold compilation of the same source/environment; each process reinstalls them and captures fresh graphs.
All 8 capture sizes passed. The 99-request per-route token/logprob probes matched the cold reference exactly.
Fresh-image GSM8K results are pending; no historical accuracy scores are represented as current results.
