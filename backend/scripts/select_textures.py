"""Model selection for texture PatchCore models using the VALIDATION half only (the test half is not loaded)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml import metrics as M  # noqa: E402
from app.ml.detectors.patchcore import PatchCore  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "data" / "external" / "mvtec"
torch.set_num_threads(4)
half = lambda p: "val" if int(hashlib.sha256(f"tex-v1:{p}".encode()).hexdigest()[:8], 16) % 2 == 0 else "test"  # noqa: E731

for cat in sys.argv[1:] or ["carpet", "wood", "leather"]:
    idx = [r for r in json.loads((SRC / "index.json").read_text()) if r["category"] == cat]
    train = [cv2.imread(str(SRC / r["image"])) for r in idx if r["split"] == "train"]
    val = [r for r in idx if r["split"] == "test" and half(r["image"]) == "val"]
    vimg = [cv2.imread(str(SRC / r["image"])) for r in val]
    y = np.array([r["label"] != "good" for r in val], int)
    for size in (256, 320, 448):
        for taps in ((6, 13), (3, 6, 13), (13, 17)):
            pc = PatchCore(input_size=(size, size), taps=taps)
            pc.fit_embeddings(pc.embed_many(train), coreset=4000, presample=40000)
            s = [pc.image_score(m) for m in pc.distance_maps(pc.embed_many(vimg))]
            print(f"{cat} size={size} taps={taps}: val AUROC {M.auroc(y, s)}", flush=True)
