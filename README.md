# EVIE-4.5B + JEV-27B-VL: one-pass visual reranking for ViDoRe V3

This repository holds the pipeline we submitted to the [ViDoRe V3 pipeline leaderboard](https://github.com/illuin-tech/vidore-benchmark).

1. [EVIE-4.5B](https://huggingface.co/tencent/EVIE-4.5B) retrieves the top 100 pages. It scores page images with late-interaction MaxSim. The corpus is encoded in `index()`.
2. [JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) reranks the top 16 pages. It sees the 16 page images (longest side 896 px, in EVIE order, each labelled with its document name). It answers one multiple-choice question: "Which page contains the information that best helps answer the query?" It returns a probability for every page in one forward pass. The 16 pages are sorted by this probability. Ranks 17 to 100 keep the EVIE order.

Each query needs one prefill. There is no text generation, no query rewriting and no OCR text. A pointwise reranker scores each page alone. Here the 16 candidates are judged together as the options of one question.

## Results

We ran the official CLI (`vidore-benchmark pipeline evaluate --language english`) at vidore-benchmark commit `a70f23a`. Hardware: one NVIDIA H200 (141 GB), 16 CPU cores, 192 GB RAM. 8 queries run concurrently.

| Dataset | Queries | nDCG@10 | Recall@10 | Recall@100 | Indexing (s) | Search (s/query) |
|---|---:|---:|---:|---:|---:|---:|
| computer_science | 215 | 0.8401 | 0.8323 | 0.9942 | 146 | 1.14 |
| energy | 308 | 0.7342 | 0.7792 | 0.9690 | 249 | 1.06 |
| finance_en | 309 | 0.7449 | 0.7728 | 0.9608 | 311 | 1.15 |
| finance_fr | 320 | 0.5696 | 0.6276 | 0.8931 | 249 | 1.09 |
| hr | 318 | 0.7168 | 0.7259 | 0.9556 | 119 | 1.06 |
| industrial | 283 | 0.6357 | 0.6413 | 0.8742 | 558 | 1.09 |
| pharmaceuticals | 364 | 0.7242 | 0.7317 | 0.9444 | 220 | 0.97 |
| physics | 302 | 0.4942 | 0.5395 | 0.9216 | 206 | 0.86 |
| **Average** | 2419 | **0.6825** | 0.7063 | 0.9391 | 2056 (total) | 1.05 |

EVIE-4.5B alone scores 0.6676 average nDCG@10 with the same index.

Search time is the CLI's `search_throughput_ms_per_query`. It is the wall time of `search()` divided by the number of queries. With 8 queries in flight, one query takes about 8 s from request to answer. Most of this time is the JEV prefill. The EVIE search takes about 0.07 s. Indexing covers page rendering and EVIE encoding.

## Code

| File | Role |
|---|---|
| `src/jevrank_pipeline.py` | The evaluated `BasePipeline` (`JevRerankPipeline`). This is the `module_path`. |
| `src/vdr_server.py` | EVIE retrieval server with live `/index` and `/search`. Its `/region` and `/judge` endpoints are not used here. They need `src/pagestore.py` and `src/jev_client.py`. |
| `bin/jev-serve` | Starts JEV-27B-VL in the `vllm/vllm-openai:v0.30.0` image. It uses the model repo's `serve_decide.py` (System 1 at `POST /v1/decide`) with the flags from the model card. |
| `slurm/jevrank_bench.sh` | The job that produced the results. It starts JEV, then EVIE. It runs a 2-query smoke test, the CLI on all 8 datasets, and the CI check. |
| `src/finalize_submission.py`, `src/validate_submission.py` | The CLI writes `num_queries` only per language, but the CI also needs it in `overall`. These scripts copy it there and run the repository's CI check locally. |

The scripts take local paths from environment variables. They are listed at the top of `slurm/jevrank_bench.sh` and `bin/jev-serve`. They replace the paths of our machine, and our site's module loads were removed. The logic is the same as in the run. `src/jevrank_pipeline.py` is the exact file that ran.

## Versions

- vidore-benchmark `a70f23a`.
- EVIE-4.5B, revision `b732dca`. It runs through the colpali fork in [Tencent/EVIE](https://github.com/Tencent/EVIE): 1024 visual tokens per page and the official MaxSim scoring.
- JEV-27B-VL, revision `fee8c1d`, served by vLLM 0.30.0 (`vllm/vllm-openai:v0.30.0`). Settings: `--max-num-seqs 8` (required by the model card), `--max-model-len 32768`, `--limit-mm-per-prompt {"image": 16}`, `--gpu-memory-utilization 0.55`. Thinking is off.

## Reproduce

Set the variables listed at the top of `slurm/jevrank_bench.sh`. Then run it from the repository root on a machine with one GPU of at least 141 GB:

```bash
mkdir -p logs
sbatch slurm/jevrank_bench.sh
```

The leaderboard files are written to `runs/<exp>/submission/results/`. The file `runs/<exp>/decisions/<dataset>.jsonl` keeps, for every query, the EVIE top 100, the 16 JEV probabilities, the prompt tokens and the timings.

## License

Apache-2.0 (see `LICENSE`). The models keep their own licences: EVIE-4.5B and JEV-27B-VL are Apache-2.0.
