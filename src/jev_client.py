"""Client for JEV-27B-VL System 1 page judgments (POST /v1/decide on bin/jev-serve). Stdlib only.

judge_pages(url, key, domain, query, ids) -> [{"corpus_id", "p_relevant", "prompt_tokens"}], one decision per page:
kind "noul", state = document name + the page image (the 1600px JPEG the visual agent sees), question = the query.
Used by jev_rerank.py (stage A) and by vdr_server.py's /judge endpoint (the agent's judge tool, stage B).
"""
import base64
import concurrent.futures as cf
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pagestore import DATA, page_image_bytes  # noqa: E402

QUESTION = "Does this document page contain information that helps answer the following query? Query: {q}"
_DOCS = {}


def doc_of(domain, cid):
    if domain not in _DOCS:
        out = {}
        with open(f"{DATA}/{domain}/corpus_img/documents.tsv") as fh:
            next(fh)
            for line in fh:
                doc, _, first, last = line.rstrip("\n").split("\t")
                for c in range(int(first), int(last) + 1):
                    out[c] = doc
        _DOCS[domain] = out
    return _DOCS[domain].get(int(cid), "?")


def page_state(domain, cid):
    img = base64.b64encode(page_image_bytes(domain, cid)).decode()
    return [f"Document: {doc_of(domain, cid)}\nPage image:", {"image": "data:image/jpeg;base64," + img}]


def decide(url, key, state, question, timeout=600):
    req = urllib.request.Request(f"{url}/v1/decide", method="POST",
                                 data=json.dumps({"kind": "noul", "state": state, "question": question}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def judge_pages(url, key, domain, query, ids, workers=8):
    def one(cid):
        r = decide(url, key, page_state(domain, cid), QUESTION.format(q=query))
        return {"corpus_id": int(cid), "p_relevant": r["probabilities"][1], "prompt_tokens": r["usage"]["prompt_tokens"]}
    with cf.ThreadPoolExecutor(min(workers, max(1, len(ids)))) as ex:
        return list(ex.map(one, ids))
