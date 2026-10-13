"""Packed page stores: one file per domain instead of one file per page.

    data/<domain>/corpus/pages.jsonl        {"corpus_id", "doc_id", "page", "markdown"} per line (OCR text)
    data/<domain>/corpus_img/images.zip     <corpus_id>.jpg members, stored uncompressed (JPEG already is)

Readers fall back to the old per-page layout (corpus/pages/<cid>.md, corpus_img/images/<cid>.jpg).
"""
import json
import os
import zipfile

DATA = os.environ.get("VDR_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
_TEXT, _ZIP = {}, {}


def page_texts(domain):
    """{corpus_id (int): markdown} for a domain (cached)."""
    if domain not in _TEXT:
        p = f"{DATA}/{domain}/corpus/pages.jsonl"
        if os.path.exists(p):
            with open(p) as fh:
                _TEXT[domain] = {(r := json.loads(l))["corpus_id"]: r["markdown"] or "" for l in fh}
        else:  # legacy per-page files
            out = {}
            d = f"{DATA}/{domain}/corpus/pages"
            for f in os.listdir(d):
                with open(f"{d}/{f}") as fh:
                    out[int(f[:-3])] = fh.read().partition("\n---\n")[2]
            _TEXT[domain] = out
    return _TEXT[domain]


def page_image_bytes(domain, cid):
    """JPEG bytes of one page from the image-only corpus."""
    z = f"{DATA}/{domain}/corpus_img/images.zip"
    if os.path.exists(z):
        if domain not in _ZIP:
            _ZIP[domain] = zipfile.ZipFile(z)
        return _ZIP[domain].read(f"{int(cid)}.jpg")
    with open(f"{DATA}/{domain}/corpus_img/images/{int(cid)}.jpg", "rb") as fh:
        return fh.read()
