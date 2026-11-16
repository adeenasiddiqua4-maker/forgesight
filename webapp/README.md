# ForgeSight Inspector (browser-only web app)

A static web app: all detection runs in the visitor's browser (ONNX Runtime Web / WebAssembly). No server, no database,
no paid hosting, and footage never leaves the computer.

Live copy: https://claude.ai/artifact/HsE6spe1fpFqRsTGbXChnQ (hosted by Claude; private to its owner until shared).

## What it does
* Video, photos or camera, analysed frame by frame with defects outlined, labelled and graded, plus an OK / CHECK / NG status bar.
* **Learn normal from this footage**: any material (leather included). The first seconds of defect-free footage teach it what normal
  looks like; later frames are compared against that. A screening aid with no measured accuracy on your material.
* **Trained models**: pavement / concrete cracks, and ferrite magnetic tiles (calibrated pass / check / reject).

## Verified against the server version
* Ferrite tile model, 277 held-out test images: identical outcome on 277/277 images (same AUROC 0.9983, 0 escapes, 1 false reject,
  121 sent to review). About 0.18 s per image in Chromium on a 4-core CPU.
* Crack model on the sample video: 24/24 frames NG, same as the server. Learned mode on the tile sample: same 5 frames flagged.

## Build and host anywhere for free
```bash
npm install
npm run build        # -> dist/
```
`dist/` is plain static files: drag it onto Netlify Drop, or publish it with GitHub Pages / Cloudflare Pages / Vercel (free tiers).
`dist/artifact.html` is the variant published on claude.ai (models loaded as base64 text, since that host only serves web file types).

Models are exported from the trained PyTorch models with `python backend/scripts/export_web.py` (writes `public/models/`).

## Not in the browser version
Accounts, shared database, dataset labelling and training, PDF reports and the analytics dashboard need the full platform
(`backend/` + `frontend/`). The golden-reference PCB profile is not included (it needs a template image per board).
