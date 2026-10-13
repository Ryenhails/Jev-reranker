"""EVIE + JEV as a vidore-benchmark pipeline: EVIE-4.5B retrieves, then JEV-27B-VL System 1 re-ranks EVIE's top-16
page images with ONE `choice` decision per query (one prefill, no decoding, no query rewriting).

    vidore-benchmark pipeline evaluate --dataset-name vidore/vidore_v3_hr --language english \
        --module-path src/jevrank_pipeline.py --class-name JevRerankPipeline \
        --pipeline-args '{"num_concurrent": 8}'

index(): every page image the framework passes in is encoded live by the EVIE server (lossless PNG, as in
  visagentret_pipeline.py) and kept in memory as a JPEG with longest side `page_max_side` (896) for JEV.
search(): EVIE top-`retriever_top_k` (100) for the query; the first `candidate_k` (16) pages, in EVIE order, go into
  one JEV decision: state = the query, then each page as "Page <A..P> (document: <doc_id>):" + its image; question =
  which page best helps answer the query; options = the page labels. JEV's probabilities re-order those pages; EVIE's
  ranks k+1..100 follow. If JEV fails for one query after retries, that query keeps EVIE's order (listed in infos).
  Every decision (EVIE top-100, probabilities, tokens, times) is logged to $JEVRANK_RUN_DIR/<domain>.jsonl.

Endpoints come from the environment (deployment, not method):
  JEVRANK_JEV_URL        bin/jev-serve root (POST /v1/decide)
  JEVRANK_JEV_KEY_ENV    name of the env var holding its key (default VLLM_API_KEY)
  JEVRANK_RETRIEVER_URL  src/vdr_server.py --backend evie
  JEVRANK_RUN_DIR        decision logs
  JEVRANK_WORK_DIR       node-local scratch for the exported corpus
"""
import base64
import concurrent.futures as cf
import io
import json
import os
import shutil
import threading
import time
import urllib.request
import zipfile

from PIL import Image
from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline

LABELS = "ABCDEFGHIJKLMNOP"
QUESTION = "Which page contains the information that best helps answer the query? Query: {q}"


def _post(url, payload, timeout, key=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, method="POST", data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class JevRerankPipeline(BasePipeline):
    def __init__(self, retriever_model="tencent/EVIE-4.5B", reranker_model="autotrust/JEV-27B-VL", candidate_k=16,
                 retriever_top_k=100, page_max_side=896, page_jpeg_quality=85, num_concurrent=8, max_retries=2,
                 query_limit=None):
        if retriever_model != "tencent/EVIE-4.5B":
            raise ValueError("the retrieval server implements tencent/EVIE-4.5B only")
        if not 2 <= int(candidate_k) <= len(LABELS):
            raise ValueError(f"candidate_k must be 2..{len(LABELS)} (the decision head's native option labels)")
        self.retriever_model, self.reranker_model = retriever_model, reranker_model
        self.k, self.top_k = int(candidate_k), int(retriever_top_k)
        self.page_max_side, self.jpeg_quality = int(page_max_side), int(page_jpeg_quality)
        self.num_concurrent, self.max_retries, self.query_limit = int(num_concurrent), int(max_retries), query_limit
        env = os.environ
        self.jev_url = env["JEVRANK_JEV_URL"].rstrip("/")
        self.jev_key = env.get(env.get("JEVRANK_JEV_KEY_ENV", "VLLM_API_KEY"), "")
        self.retriever_url = env["JEVRANK_RETRIEVER_URL"].rstrip("/")
        self.run_dir = env["JEVRANK_RUN_DIR"]
        self.work_dir = env.get("JEVRANK_WORK_DIR") or f"/tmp/{env.get('USER', 'u')}_jevrank_{env.get('SLURM_JOB_ID', 'x')}"

    # ------------------------------------------------------------------ index
    def index(self, corpus_ids, corpus_images, corpus_texts, dataset_name=None):
        if not dataset_name:
            raise ValueError("dataset_name is required (passed by vidore-benchmark's evaluate_retrieval)")
        t0 = time.time()
        self.domain = dataset_name.split("/")[-1].removeprefix("vidore_v3_")
        self.index_name = f"jev/{self.domain}"
        base = f"{self.work_dir}/{self.domain}"
        shutil.rmtree(base, ignore_errors=True)
        os.makedirs(base)
        from datasets import load_dataset
        meta = load_dataset(dataset_name, data_dir="corpus", split="test").select_columns(["corpus_id", "doc_id"])
        doc_of = dict(zip(map(str, meta["corpus_id"]), meta["doc_id"]))
        self.doc = {str(c): doc_of[str(c)] for c in corpus_ids}

        def prep(i):
            im = corpus_images[i].convert("RGB")
            png = io.BytesIO()
            im.save(png, "PNG", compress_level=1)
            im.thumbnail((self.page_max_side, self.page_max_side), Image.LANCZOS)
            jpg = io.BytesIO()
            im.save(jpg, "JPEG", quality=self.jpeg_quality, optimize=True)
            return str(corpus_ids[i]), png.getvalue(), "data:image/jpeg;base64," + base64.b64encode(jpg.getvalue()).decode()

        self.images = {}
        evie_zip = f"{base}/pages_for_retriever.zip"
        with zipfile.ZipFile(evie_zip, "w", zipfile.ZIP_STORED) as ez, cf.ThreadPoolExecutor(min(16, os.cpu_count() or 4)) as ex:
            for cid, png, jpg in ex.map(prep, range(len(corpus_ids))):
                ez.writestr(f"{cid}.png", png)
                self.images[cid] = jpg
        t_export = time.time() - t0
        info = _post(f"{self.retriever_url}/index", {"name": self.index_name, "pages_zip": evie_zip, "texts": None}, timeout=4 * 3600)
        os.remove(evie_zip)
        self.index_info = {"pages": len(corpus_ids), "export_ms": round(1000 * t_export),
                           "retriever_encode_ms": info["encode_ms"], "retriever_mean_tokens_per_page": info["mean_tokens"],
                           "jev_page_jpeg_mean_kb": round(sum(len(v) for v in self.images.values()) * 3 / 4 / 1024 / max(1, len(self.images)), 1)}
        print(f"[{self.index_name}] indexed: {self.index_info}", flush=True)

    # ------------------------------------------------------------------ search
    def _jev_ok(self):
        req = urllib.request.Request(f"{self.jev_url}/v1/models", headers={"Authorization": f"Bearer {self.jev_key}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status == 200
        except Exception:
            return False

    def search(self, query_ids, queries):
        pairs = list(zip(query_ids, queries))
        if self.query_limit:
            pairs = pairs[: int(self.query_limit)]
        os.makedirs(self.run_dir, exist_ok=True)
        log_path = f"{self.run_dir}/{self.domain}.jsonl"
        lock, done, t0 = threading.Lock(), [0], time.time()
        abort = threading.Event()
        log = open(log_path, "w")

        def one(qid, query):
            t = time.time()
            hits = _post(f"{self.retriever_url}/search", {"domain": self.index_name, "query": query, "k": self.top_k}, timeout=600)["results"]
            evie = [str(h["corpus_id"]) for h in hits]
            t_ret = time.time() - t
            cands = evie[: self.k]
            state = [f"Query: {query}\nCandidate pages:"]
            for lab, c in zip(LABELS, cands):
                state += [f"\nPage {lab} (document: {self.doc.get(c, '?')}):", {"image": self.images[c]}]
            body = {"kind": "choice", "state": state, "question": QUESTION.format(q=query),
                    "options": [f"Page {lab}" for lab in LABELS[: len(cands)]], "strategy": "single", "thinking": "off"}
            rec = {"qid": str(qid), "evie": evie[: self.k], "probs": None, "prompt_tokens": 0, "error": None,
                   "evie_top": evie}  # EVIE's full order for the original query (analysis only)
            t = time.time()
            for attempt in range(1 + self.max_retries):
                if abort.is_set() or len(cands) < 2:
                    break
                try:
                    r = _post(f"{self.jev_url}/v1/decide", body, timeout=900, key=self.jev_key)
                    rec["probs"], rec["prompt_tokens"] = r["probabilities"], r["usage"]["prompt_tokens"]
                    break
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
                    if not self._jev_ok():
                        print(f"[{self.index_name}] JEV server not answering; aborting", flush=True)
                        abort.set()
                        break
                    time.sleep(10)
            rec["retriever_ms"], rec["jev_ms"] = round(1000 * t_ret), round(1000 * (time.time() - t))
            if rec["probs"]:
                p = dict(zip(cands, rec["probs"]))
                head = sorted(cands, key=lambda c: (-p[c], cands.index(c)))
            else:
                head = cands  # JEV failed for this query: EVIE's order
            ranking = head + evie[self.k:]
            rec["ranking_top10"] = ranking[:10]
            with lock:
                log.write(json.dumps(rec) + "\n")
                log.flush()
                done[0] += 1
                if done[0] % 50 == 0 or done[0] == len(pairs):
                    el = time.time() - t0
                    print(f"[{self.index_name}] {done[0]}/{len(pairs)} done, {el / 60:.1f} min elapsed, "
                          f"ETA {(len(pairs) - done[0]) * el / done[0] / 60:.1f} min", flush=True)
            return rec, ranking

        try:
            with cf.ThreadPoolExecutor(self.num_concurrent) as ex:
                out = list(ex.map(lambda a: one(*a), pairs))
        finally:
            log.close()
        if abort.is_set():
            raise RuntimeError("JEV server failed during the run")
        run = {rec["qid"]: {cid: 1.0 / (i + 1) for i, cid in enumerate(ranking)} for rec, ranking in out}
        recs = [rec for rec, _ in out]
        ok = [r for r in recs if r["probs"]]
        infos = {"pipeline_infos": {
            "retriever_model": self.retriever_model, "reranker_model": self.reranker_model, "candidate_k": self.k,
            "retriever_top_k": self.top_k, "page_max_side": self.page_max_side, "num_concurrent": self.num_concurrent,
            "index": self.index_info, "num_queries_run": len(pairs),
            "per_query_retrieval_time_milliseconds": {r["qid"]: r["retriever_ms"] + r["jev_ms"] for r in recs},
            "reranker_failed_query_ids": [r["qid"] for r in recs if not r["probs"]],
            "reranker_total_prompt_tokens": sum(r["prompt_tokens"] for r in ok),
            "reranker_calls": len(ok), "reranker_completion_tokens": 0,
            "avg_retriever_ms": sum(r["retriever_ms"] for r in recs) / max(1, len(recs)),
            "avg_reranker_ms": sum(r["jev_ms"] for r in ok) / max(1, len(ok)),
        }}
        return run, infos
