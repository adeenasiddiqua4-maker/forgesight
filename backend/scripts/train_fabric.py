"""Train and validate the woven-fabric defect model on the ZJU-Leaper sample (scripts/fetch_zju_fabric.py).

Supervised: MobileNetV2 encoder (ImageNet transfer) + FPN decoder, one class 'defect', defect regions from the
dataset's boxes (derived from its masks). Image-level decision = calibrated top-k defect probability.
  fit  : 6000 images (official train split)    val : 1200 images (official train split, disjoint) -> thresholds
  test : 2000 images (official TEST split, never used for any choice) -> every reported number
"""
from __future__ import annotations

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
from app.ml.detectors.segnet import SegDetector  # noqa: E402
from app.ml.training import train_segnet  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "external" / "zju"
OUT = ROOT / "data" / "models"
VERSION, NAME, SIZE, TOL = "1.0.0", "fabric_segnet", 256, 0.02
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
torch.set_num_threads(4)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(r):
    img = cv2.resize(cv2.imread(str(SRC / r["image"])), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    m = np.zeros((SIZE, SIZE), np.uint8)
    if r["size"] > 0:
        for line in (SRC / r["label"]).read_text().split("\n"):
            p = line.split()
            if len(p) == 5:
                cx, cy, w, h = (float(v) * SIZE for v in p[1:])
                cv2.rectangle(m, (int(cx - w / 2), int(cy - h / 2)), (int(np.ceil(cx + w / 2)), int(np.ceil(cy + h / 2))), 1, -1)
    return img, m


sample = json.loads((SRC / "sample.json").read_text())
role = {k: [r for r in sample if r["role"] == k] for k in ("fit", "val", "test")}
log({k: (len(v), sum(r["size"] > 0 for r in v)) for k, v in role.items()})
fit = [load(r) for r in role["fit"]]
det = SegDetector(["defect"], input_size=(SIZE, SIZE), name=NAME, version=VERSION)
tlog = train_segnet(det, [a for a, _ in fit], [b for _, b in fit], epochs=EPOCHS, batch=24, lr=1e-3, balance_empty=False, log=log)
del fit


def scores(rs):
    imgs = [load(r)[0] for r in rs]
    out = []
    for i in range(0, len(imgs), 32):
        for p in det.prob_maps(imgs[i:i + 32]):
            out.append(det.image_score_from_prob(1 - p[0]))
    return np.array(out)


sv, st = scores(role["val"]), scores(role["test"])
yv = np.array([r["size"] > 0 for r in role["val"]], int)
yt = np.array([r["size"] > 0 for r in role["test"]], int)
platt = Platt.fit(sv, yv)
pv, pt = platt(sv), platt(st)
t_low, t_high = select_review_band(pv, yv, max_escape=TOL, max_false_reject=TOL)
# single-threshold operating point chosen on validation (max accuracy), reported on test
cands = np.linspace(0.05, 0.95, 91)
thr = float(cands[int(np.argmax([((pv >= c) == yv).mean() for c in cands]))])
test_m = {"auroc": M.auroc(yt, st), **M.binary_report(yt, pt >= thr), "decision_threshold": round(thr, 3), "calibration": M.ece(yt, pt)}
ops = []
for tol in (0.0, 0.02, 0.05, 0.1):
    lo, hi = select_review_band(pv, yv, max_escape=tol, max_false_reject=tol)
    ops.append({"validation_tolerance": tol, "shipped": tol == TOL, "val": M.triage_report(yv, pv, lo, hi), "test": M.triage_report(yt, pt, lo, hi)})
metrics = {"val": {"auroc": M.auroc(yv, sv), **M.binary_report(yv, pv >= thr)}, "test": test_m,
           "test_triage": M.triage_report(yt, pt, t_low, t_high), "operating_points": ops}
log(f"TEST: AUROC {test_m['auroc']} accuracy {test_m['accuracy']} {test_m['accuracy_ci95']} recall {test_m['recall']} FPR {test_m['false_positive_rate']}")
for o in ops:
    t = o["test"]
    log(f"  tolerance {o['validation_tolerance']}: auto-pass {t['auto_pass']} auto-NG {t['auto_defective']} review {t['sent_to_review']} "
        f"escaped {t['escaped_defects']} false-rejects {t['false_rejects']} auto-accuracy {t['auto_decision_accuracy']}")

det.pixel_threshold = 0.5
base = OUT / NAME / VERSION
det.save(base / "model.pt")
lim = ["Trained and measured on ZJU-Leaper woven fabric images (19 patterns, 512x512 line-scan crops). Other fabrics, knits, prints, "
       "colours, cameras or lighting need their own validation.",
       "Defect regions come from boxes around the dataset's masks, so outlines are coarse. Single class 'defect': no defect types.",
       f"Test sample: {int(yt.sum())} defective + {int((1 - yt).sum())} normal images from the official test split; the real defect rate is lower (about 25%).",
       "Data licence: ZJU-Leaper is CC BY-NC-SA 4.0 (non-commercial use only)."]
card = {"name": NAME, "version": VERSION, "kind": "segmentation", "material": "textiles", "classes": ["defect"],
        "architecture": f"MobileNetV2 encoder (ImageNet transfer) + FPN decoder, {SIZE}x{SIZE}, CE + Dice", "dataset": "ZJU-Leaper (HF mirror mohamedemam/zju-leaper-fabric-defect-yolo)",
        "splits": {k: len(v) for k, v in role.items()}, "training": tlog, "calibration": platt.to_dict(), "metrics": metrics, "limitations": lim,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
(base / "card.json").write_text(json.dumps(card, indent=2))
(OUT / "profiles" / "woven_fabric.json").write_text(json.dumps({
    "key": "woven_fabric", "material": "textiles", "name": "Woven fabric", "product_category": "Woven fabric web (line-scan crops)",
    "requires_reference": False, "mm_per_pixel": None, "detectors": [{"type": "segnet", "model": NAME, "version": VERSION, "calibration": platt.to_dict()}],
    "decision": {"mode": "fusion", "features": [NAME], "coef": [platt.a], "intercept": platt.b, "t_low": t_low, "t_high": t_high,
                 "image_level_validated": True, "validated": True, "evidence": f"{NAME}-test"}}, indent=2))
ev_path = OUT / "evaluations" / "fabric.json"
ev_path.write_text(json.dumps([{"id": f"{NAME}-test", "model": NAME, "version": VERSION, "material": "textiles", "dataset": "ZJU-Leaper woven fabric",
                                "split": "official test split (sample)", "task": "defect detection (image-level) + localization", "n": len(role["test"]),
                                "metrics": metrics, "notes": lim, "created_at": card["created_at"], "hardware": "CPU x4 threads (no GPU)"}], indent=2))
log("done")
