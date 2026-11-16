"""Train and validate PatchCore inspection profiles for MVTec AD texture categories (leather, carpet, wood).

Protocol (no test leakage):
  * fit:        MVTec 'train' split, defect-free images only
  * validation: half of MVTec 'test' (hash split by file) -> calibration + review-band thresholds
  * test:       the other half, never used for any choice -> every reported number
Writes data/models/<name>/1.0.0/{model.pt, card.json}, profiles/<key>.json, evaluations/textures.json.
Usage: python scripts/train_textures.py [leather carpet wood]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml import metrics as M  # noqa: E402
from app.ml.calibration import Platt, select_review_band  # noqa: E402
from app.ml.detectors.patchcore import PatchCore  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "external" / "mvtec"
OUT = ROOT / "data" / "models"
VERSION = "1.0.0"
INPUT = (320, 320)
TOL = 0.02
torch.set_num_threads(4)
MATERIAL = {"leather": ("leather", "Leather surface (close-up)", "Finished leather, flat close-up view"),
            "carpet": ("textiles", "Carpet / pile textile (close-up)", "Tufted carpet or pile fabric, flat close-up view"),
            "wood": ("wood", "Wood surface (close-up)", "Planed or finished wood, flat close-up view")}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def half(path: str) -> str:
    return "val" if int(hashlib.sha256(f"tex-v1:{path}".encode()).hexdigest()[:8], 16) % 2 == 0 else "test"


def run(cat: str) -> dict:
    idx = [r for r in json.loads((SRC / "index.json").read_text()) if r["category"] == cat]
    train = [r for r in idx if r["split"] == "train"]
    rest = [r for r in idx if r["split"] == "test"]
    for r in rest:
        r["half"] = half(r["image"])
    val = [r for r in rest if r["half"] == "val"]
    test = [r for r in rest if r["half"] == "test"]
    log(f"{cat}: fit {len(train)} good | val {len(val)} ({sum(r['label'] != 'good' for r in val)} defective) | "
        f"test {len(test)} ({sum(r['label'] != 'good' for r in test)} defective)")
    load = lambda r: cv2.imread(str(SRC / r["image"]))  # noqa: E731
    name = f"{cat}_patchcore"
    pc = PatchCore(input_size=INPUT, name=name, version=VERSION)
    t0 = time.time()
    pc.fit_embeddings(pc.embed_many([load(r) for r in train]), coreset=4000, presample=40000)
    log(f"  fitted in {time.time() - t0:.0f}s")

    def score(rs):
        emb = pc.embed_many([load(r) for r in rs])
        maps = pc.distance_maps(emb)
        return np.array([pc.image_score(m) for m in maps]), maps

    sv, mv = score(val)
    st, mt = score(test)
    yv = np.array([r["label"] != "good" for r in val], int)
    yt = np.array([r["label"] != "good" for r in test], int)
    platt = Platt.fit(sv, yv)
    pv, pt = platt(sv), platt(st)
    t_low, t_high = select_review_band(pv, yv, max_escape=TOL, max_false_reject=TOL)

    # pixel threshold: maximise pixel F1 on validation defect images (masks at 512 -> evaluate at 256 for speed)
    def pix(rs, maps, thr):
        g, p = [], []
        for r, m in zip(rs, maps):
            if not r["mask"]:
                continue
            gt = cv2.resize(cv2.imread(str(SRC / r["mask"]), 0), (256, 256), interpolation=cv2.INTER_NEAREST) > 127
            g.append(gt.ravel())
            p.append((pc.full_map(m, 256, 256) > thr).ravel())
        return M.pixel_report(np.concatenate(g), np.concatenate(p))

    good_v = np.concatenate([m.ravel() for m, r in zip(mv, val) if r["label"] == "good"])
    cands = np.quantile(good_v, [0.99, 0.995, 0.999, 0.9995, 0.9999])
    best = max(((pix(val, mv, t)["pixel_f1"] or 0), float(t)) for t in cands)
    pc.pixel_threshold = best[1]
    pc.score_scale = 2 * pc.pixel_threshold
    pc.validated = True

    per_defect = {}
    for d in sorted({r["label"] for r in test} - {"good"}):
        sel = np.array([r["label"] == d for r in test])
        n = int(sel.sum())
        hit = int((pt[sel] >= 0.5).sum())
        notpass = int((pt[sel] > t_low).sum())
        per_defect[d] = {"support": n, "detected_at_0.5": hit, "recall_at_0.5": round(hit / max(1, n), 4), "recall_ci95": M.wilson(hit, n),
                         "not_auto_passed": notpass, "caught_or_reviewed_rate": round(notpass / max(1, n), 4)}
    test_m = {"auroc": M.auroc(yt, st), **M.binary_report(yt, pt >= 0.5), "calibration": M.ece(yt, pt)}
    ops = []
    for tol in (0.0, 0.02, 0.05):
        lo, hi = select_review_band(pv, yv, max_escape=tol, max_false_reject=tol)
        ops.append({"validation_tolerance": tol, "shipped": tol == TOL, "val": M.triage_report(yv, pv, lo, hi), "test": M.triage_report(yt, pt, lo, hi)})
    metrics = {"val": {"auroc": M.auroc(yv, sv), **M.binary_report(yv, pv >= 0.5)}, "test": test_m,
               "test_triage": M.triage_report(yt, pt, t_low, t_high), "test_per_defect": per_defect, "operating_points": ops,
               "test_pixel": pix(test, mt, pc.pixel_threshold)}
    log(f"  TEST: AUROC {test_m['auroc']}  acc {test_m['accuracy']} {test_m['accuracy_ci95']}  recall {test_m['recall']}  FPR {test_m['false_positive_rate']}")
    log(f"  TEST triage: {json.dumps(metrics['test_triage'])}")

    base = OUT / name / VERSION
    pc.save(base / "model.pt")
    mat, pname, pcat = MATERIAL[cat]
    lim = [f"Trained and measured only on MVTec AD '{cat}': flat, evenly lit close-ups of one {cat} type under one camera.",
           f"Test half has {int(yt.sum())} defective and {int((1 - yt).sum())} good images; see the 95% intervals.",
           "Does not transfer to whole hides / rolls on a conveyor, other colours, finishes or lighting without new data.",
           "Data licence: MVTec AD is CC BY-NC-SA 4.0 (non-commercial use only)."]
    card = {"name": name, "version": VERSION, "kind": "anomaly", "material": mat, "architecture": "PatchCore: frozen ImageNet MobileNetV2 (blocks 6+13), k-center coreset 4000, 320x320",
            "dataset": f"MVTec AD {cat} (Hugging Face mirror Voxel51/mvtec-ad)", "trained_on": "MVTec train split, defect-free images only",
            "splits": {"fit_good": len(train), "val": len(val), "test": len(test)}, "calibration": platt.to_dict(), "pixel_threshold": pc.pixel_threshold,
            "metrics": metrics, "limitations": lim, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (base / "card.json").write_text(json.dumps(card, indent=2))
    (OUT / "profiles" / f"{cat}_surface.json").write_text(json.dumps({
        "key": f"{cat}_surface", "material": mat, "name": pname, "product_category": pcat, "requires_reference": False, "mm_per_pixel": None,
        "detectors": [{"type": "patchcore", "model": name, "version": VERSION, "calibration": platt.to_dict()}],
        "decision": {"mode": "fusion", "features": [name], "coef": [platt.a], "intercept": platt.b, "t_low": t_low, "t_high": t_high,
                     "image_level_validated": True, "validated": True, "evidence": f"{name}-test"},
        "drift_reference": {name: np.quantile(sv[yv == 0], np.linspace(0, 1, 11)).round(4).tolist()}}, indent=2))
    return {"id": f"{name}-test", "model": name, "version": VERSION, "material": mat, "dataset": f"MVTec AD {cat}", "split": "test (held-out half)",
            "task": "anomaly detection (image-level) + localization, review band", "n": len(test), "metrics": metrics, "notes": lim,
            "created_at": card["created_at"], "hardware": f"CPU x{torch.get_num_threads()} threads (no GPU)"}


if __name__ == "__main__":
    cats = sys.argv[1:] or ["leather", "carpet", "wood"]
    evals_path = OUT / "evaluations" / "textures.json"
    evals = [e for e in (json.loads(evals_path.read_text()) if evals_path.exists() else []) if e["model"].split("_")[0] not in cats]
    for c in cats:
        evals.append(run(c))
    evals_path.write_text(json.dumps(evals, indent=2))
    log("done")
