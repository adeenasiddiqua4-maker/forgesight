"""Export the models used by the browser-only web app (webapp/) to ONNX, plus the data they need.

  features.onnx   image -> patch embeddings (frozen MobileNetV2 blocks 6+13, as PatchCore uses)
  knn.onnx        (embeddings, memory bank) -> nearest-neighbour distance per patch
  crack_seg.onnx  pavement crack segmentation   tile_seg.onnx  magnetic tile segmentation
  tile_memory.bin PatchCore memory bank of the tile model (float32, row-major)
  config.json     thresholds, fusion weights, class names, defect knowledge base
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml import knowledge as K  # noqa: E402
from app.ml.backbones import IMAGENET_MEAN, IMAGENET_STD, mobilenetv2_imagenet  # noqa: E402
from app.ml.detectors.patchcore import PatchCore  # noqa: E402
from app.ml.detectors.segnet import SegDetector  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models"
OUT = ROOT / "webapp" / "public" / "models"
OUT.mkdir(parents=True, exist_ok=True)


class Norm(nn.Module):
    """Input: RGB float 0..1, NCHW. Normalisation lives in the graph so the browser only sends raw pixels."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = Norm()
        self.net = mobilenetv2_imagenet(True)

    def forward(self, x):
        f6, f13 = self.net.forward_taps(self.norm(x), (6, 13))
        a = F.avg_pool2d(f6, 3, 1, 1)
        b = F.interpolate(F.avg_pool2d(f13, 3, 1, 1), size=a.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([a, b], 1)[0].permute(1, 2, 0).reshape(-1, 128)  # [P, 128], row-major over the H/8 x W/8 grid


class Knn(nn.Module):
    def forward(self, emb, mem):
        d2 = (emb * emb).sum(1, keepdim=True) + (mem * mem).sum(1)[None] - 2.0 * emb @ mem.t()
        return torch.sqrt(torch.clamp(d2.min(1).values, min=0.0))


class Seg(nn.Module):
    def __init__(self, det: SegDetector):
        super().__init__()
        self.norm = Norm()
        self.net = det.net

    def forward(self, x):
        return torch.softmax(self.net(self.norm(x)), 1)[0]  # [K+1, H, W]


def export(module, args, name, inputs, outputs, dynamic=None):
    module.eval()
    torch.onnx.export(module, args, str(OUT / name), input_names=inputs, output_names=outputs, dynamic_axes=dynamic, opset_version=17, dynamo=False)
    print(f"{name}: {(OUT / name).stat().st_size / 1e6:.1f} MB")


with torch.no_grad():
    export(Features(), (torch.rand(1, 3, 320, 448),), "features.onnx", ["image"], ["embeddings"], {"image": {2: "h", 3: "w"}, "embeddings": {0: "p"}})
    export(Knn(), (torch.rand(100, 128), torch.rand(300, 128)), "knn.onnx", ["emb", "mem"], ["dist"], {"emb": {0: "p"}, "mem": {0: "m"}, "dist": {0: "p"}})
    crack = SegDetector.load(MODELS / "concrete_crack_segnet" / "1.0.0" / "model.pt")
    export(Seg(crack), (torch.rand(1, 3, *crack.input_size),), "crack_seg.onnx", ["image"], ["probs"])
    tile = SegDetector.load(MODELS / "mt_segnet" / "1.0.0" / "model.pt")
    export(Seg(tile), (torch.rand(1, 3, *tile.input_size),), "tile_seg.onnx", ["image"], ["probs"])

pc = PatchCore.load(MODELS / "mt_patchcore" / "1.0.0" / "model.pt")
pc.memory.numpy().astype(np.float32).tofile(OUT / "tile_memory.bin")
prof = json.loads((MODELS / "profiles" / "magnetic_tile.json").read_text())
crack_prof = json.loads((MODELS / "profiles" / "pavement_crack.json").read_text())
textures = {}
TEX_NAMES = {"leather": "MVTec AD leather", "carpet": "MVTec AD carpet", "wood": "MVTec AD wood"}
for kind in ("leather", "carpet", "wood"):
    base = MODELS / f"{kind}_patchcore" / "1.0.0"
    if not (base / "model.pt").exists():
        continue
    tpc = PatchCore.load(base / "model.pt")
    if tuple(tpc.taps) != (6, 13):
        raise SystemExit(f"{kind}: web export supports taps (6, 13) only, got {tpc.taps}")
    tpc.memory.numpy().astype(np.float32).tofile(OUT / f"{kind}_memory.bin")
    tprof = json.loads((MODELS / "profiles" / f"{kind}_surface.json").read_text())
    textures[kind] = {"input": int(tpc.input_size[0]), "memoryRows": int(tpc.memory.shape[0]), "platt": tprof["detectors"][0]["calibration"],
                      "t_low": tprof["decision"]["t_low"], "t_high": tprof["decision"]["t_high"], "pixelThreshold": tpc.pixel_threshold,
                      "scoreScale": tpc.score_scale, "name": TEX_NAMES[kind]}

fabric = None
fab_base = MODELS / "fabric_segnet" / "1.0.0"
if (fab_base / "model.pt").exists():
    fab = SegDetector.load(fab_base / "model.pt")
    with torch.no_grad():
        export(Seg(fab), (torch.rand(1, 3, *fab.input_size),), "fabric_seg.onnx", ["image"], ["probs"])
    fprof = json.loads((MODELS / "profiles" / "woven_fabric.json").read_text())
    fabric = {"input": list(fab.input_size), "platt": fprof["detectors"][0]["calibration"], "t_low": fprof["decision"]["t_low"],
              "t_high": fprof["decision"]["t_high"], "pixelThreshold": fab.pixel_threshold, "name": "ZJU-Leaper woven fabric"}

config = {
    "tile": {"segInput": list(tile.input_size), "pcInput": list(pc.input_size), "classes": tile.classes, "segPixelThreshold": tile.pixel_threshold,
             "pcPixelThreshold": pc.pixel_threshold, "memoryRows": int(pc.memory.shape[0]),
             "fusion": {k: prof["decision"][k] for k in ("coef", "intercept", "t_low", "t_high")},
             "platt": {d["model"]: d["calibration"] for d in prof["detectors"]}, "disagreementDelta": 0.6},
    "crack": {"segInput": list(crack.input_size), "classes": crack.classes, "segPixelThreshold": crack.pixel_threshold,
              "regionConfident": crack_prof["decision"]["region_confident"]},
    "textures": textures,
    "fabric": fabric,
    "materials": [{"key": m["key"], "name": m["name"], "family": m["family"], "imaging": m["imaging_notes"]} for m in K.MATERIALS],
    "taxonomy": {k: {"label": v[0], "category": v[1], "crit": v[2]} for k, v in K.TAXONOMY.items()},
    "causes": {f"{t}|{m['key']}": K.cause_hypotheses(t, m["key"]) for t in K.TAXONOMY for m in K.MATERIALS if K.cause_hypotheses(t, m["key"])},
}
(OUT / "config.json").write_text(json.dumps(config))
print("config.json + tile_memory.bin written")

# numerical parity of the exported graphs against PyTorch
import onnxruntime as ort  # noqa: E402

x = torch.rand(1, 3, 288, 416)
with torch.no_grad():
    ref = Features().eval()(x).numpy()  # eval: BatchNorm must use stored statistics
got = ort.InferenceSession(str(OUT / "features.onnx")).run(None, {"image": x.numpy()})[0]
print("features max abs diff:", float(np.abs(ref - got).max()), ref.shape)
e, m = torch.rand(500, 128), torch.rand(900, 128)
got = ort.InferenceSession(str(OUT / "knn.onnx")).run(None, {"emb": e.numpy(), "mem": m.numpy()})[0]
print("knn max abs diff:", float(np.abs(torch.cdist(e, m).min(1).values.numpy() - got).max()))
x = torch.rand(1, 3, *crack.input_size)
with torch.no_grad():
    ref = Seg(crack)(x).numpy()
print("crack_seg max abs diff:", float(np.abs(ref - ort.InferenceSession(str(OUT / "crack_seg.onnx")).run(None, {"image": x.numpy()})[0]).max()))
