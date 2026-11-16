"""Download a balanced, seeded sample of the ZJU-Leaper fabric defect dataset (Hugging Face mirror
mohamedemam/zju-leaper-fabric-defect-yolo, CC BY-NC-SA 4.0, non-commercial).

  fit   : 3000 defective + 3000 normal from the official train split
  val   :  600 defective +  600 normal from the official train split (disjoint from fit)
  test  : 1000 defective + 1000 normal from the official test split ("val" in the mirror) - never used for any choice
Writes data/external/zju/{images,labels}/... and sample.json.
"""
from __future__ import annotations

import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "external" / "zju"
BASE = "https://huggingface.co/datasets/mohamedemam/zju-leaper-fabric-defect-yolo/resolve/main/"
rows = json.loads((ROOT / "labels_index.json").read_text())
rng = random.Random(2026)


def pick(split: str, defective: bool, n: int, exclude=frozenset()):
    pool = [r for r in rows if r["split"] == split and (r["size"] > 0) == defective and r["label"] not in exclude]
    return rng.sample(pool, n)


fit = pick("train", True, 3000) + pick("train", False, 3000)
used = {r["label"] for r in fit}
val = pick("train", True, 600, used) + pick("train", False, 600, used)
test = pick("val", True, 1000) + pick("val", False, 1000)
sample = [{**r, "role": role} for role, rs in (("fit", fit), ("val", val), ("test", test)) for r in rs]


def get(rel: str) -> None:
    out = ROOT / rel
    if out.exists() and out.stat().st_size > 0:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(BASE + rel, timeout=120) as r:
                out.write_bytes(r.read())
            return
        except Exception:
            if attempt == 4:
                raise


def job(r: dict) -> None:
    img = r["label"].replace("labels/", "images/", 1)[:-4] + ".jpg"
    r["image"] = img
    get(img)
    if r["size"] > 0:
        get(r["label"])


with ThreadPoolExecutor(10) as ex:
    for i, _ in enumerate(ex.map(job, sample), 1):
        if i % 1000 == 0:
            print(f"{i}/{len(sample)}", flush=True)
(ROOT / "sample.json").write_text(json.dumps(sample))
print("done", len(sample))
