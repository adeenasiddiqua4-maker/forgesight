"""Download the MVTec AD texture categories used by ForgeSight (leather, carpet, wood) from the
Hugging Face mirror Voxel51/mvtec-ad, resized to 512x512 PNG, with defect masks.

Licence: MVTec AD is CC BY-NC-SA 4.0 (non-commercial). Output: data/external/mvtec/<category>/{images,masks}/ + index.json
"""
from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "data" / "external" / "mvtec"
BASE = "https://huggingface.co/datasets/Voxel51/mvtec-ad/resolve/main/"
CATS = sys.argv[1:] or ["leather", "carpet", "wood"]


def get(path: str) -> np.ndarray:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(BASE + path, timeout=120) as r:
                img = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img
        except Exception as exc:  # retry transient network errors
            err = exc
    raise RuntimeError(f"{path}: {err}")


def job(s: dict) -> dict:
    cat, label, split = s["category"]["label"], s["defect"]["label"], s["split"]
    stem = s["filepath"].replace("/", "_").removesuffix(".png")
    out = ROOT / cat / "images" / f"{stem}.png"
    rec = {"category": cat, "label": label, "split": split, "image": str(out.relative_to(ROOT)), "mask": None}
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        img = get(s["filepath"])
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(out), cv2.resize(img[..., :3], (512, 512), interpolation=cv2.INTER_AREA))
    if s.get("defect_mask"):
        mout = ROOT / cat / "masks" / f"{stem}.png"
        if not mout.exists():
            mout.parent.mkdir(parents=True, exist_ok=True)
            m = get(s["defect_mask"]["mask_path"])
            m = m if m.ndim == 2 else m[..., 0]
            cv2.imwrite(str(mout), (cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) * 255)
        rec["mask"] = str(mout.relative_to(ROOT))
    return rec


samples = [s for s in json.load(open(ROOT / "samples.json"))["samples"] if s["category"]["label"] in CATS]
print(f"{len(samples)} files to fetch for {CATS}", flush=True)
recs = []
with ThreadPoolExecutor(6) as ex:
    for i, r in enumerate(ex.map(job, samples), 1):
        recs.append(r)
        if i % 100 == 0:
            print(f"  {i}/{len(samples)}", flush=True)
idx = ROOT / "index.json"
old = json.loads(idx.read_text()) if idx.exists() else []
keep = [r for r in old if r["category"] not in CATS]
idx.write_text(json.dumps(keep + recs, indent=0))
print("done", len(recs))
