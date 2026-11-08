# Bragg-spot benchmark — smoke-test results

Notebook: `dinov3/notebooks/dummy_data_explore.ipynb`. Data: simulated SiO₂, probe α = 1 / 2 / 4 mrad.

**Smoke test only:** CPU, 4 thicknesses (20, 200, 420, 460 Å → indices 0, 9, 20, 22), decoder trained 300 steps on 6 patterns.
Full run (25 thicknesses, 1500 steps on GPU) still to do.

## Pipeline
1. **GT** from `dataPots`: maxima ≥ 5 % of max, ≥ 5 px apart, direct beam removed → ~105 lattice spots/pattern, same for every angle.
   "Visible" = disk in `dataMeas` ≥ 3× its surrounding ring.
2. **DINOv3 ViT-L/16** (frozen): crop 128 px → resize 1024 → 64×64 tokens, 2 detector px/token.
3. **Detection maps:** DINO averaged query (cosine), DINO spatial template (token NCC), classical probe correlation p = 0 / 0.5 / 1.
   Peaks ≥ 4 px apart, sub-px refinement, per-method centre calibration on the probe image.
4. **Scoring:** TP = within 2 px of an unmatched GT spot; FP otherwise; direct beam (and invisible spots in the "vis" protocol) ignored.
   Threshold chosen per method/angle so FP = 5 per image.
5. **Trained decoder:** DINO tokens + raw image + probe → heatmap + offsets; train t < 400 Å, test t ≥ 400 Å (all angles together).

## TP / FP / FN at 5 FP per image — all lattice reflections, 4 thicknesses (map refinement)

| method | 1 mrad TP / FP / FN | recall | 2 mrad TP / FP / FN | recall | 4 mrad TP / FP / FN | recall |
|---|---|---|---|---|---|---|
| GT spots | 425 | | 436 | | 429 | |
| DINO avg-query | 61 / 20 / 364 | 0.14 | 4 / 20 / 432 | 0.01 | 0 / 20 / 429 | 0.00 |
| DINO spatial | 39 / 15 / 386 | 0.09 | 19 / 13 / 417 | 0.04 | 7 / 11 / 422 | 0.02 |
| classical p=0 | 9 / 20 / 416 | 0.02 | 25 / 20 / 411 | 0.06 | 33 / 20 / 396 | 0.08 |
| classical p=0.5 | 64 / 20 / 361 | 0.15 | 38 / 20 / 398 | 0.09 | 25 / 20 / 404 | 0.06 |
| **classical p=1** | **207 / 20 / 218** | **0.49** | **70 / 20 / 366** | **0.16** | 6 / 10 / 423 | 0.01 |

## Held-out thicknesses (≥ 400 Å, 2 patterns per angle), all reflections, 5 FP/img

| method | 1 mrad TP / FP / FN | 2 mrad TP / FP / FN | 4 mrad TP / FP / FN |
|---|---|---|---|
| GT spots | 224 | 223 | 222 |
| classical p=1 | 95 / 10 / 129 | 31 / 10 / 192 | 2 / 9 / 220 |
| classical p=0 | 5 / 10 / 219 | 14 / 10 / 209 | **23 / 10 / 199** |
| DINO spatial | 25 / 10 / 199 | 9 / 9 / 214 | 5 / 8 / 217 |
| **trained decoder** | **126 / 10 / 98** | **44 / 10 / 179** | 3 / 10 / 219 |

## Visible spots only (4 thicknesses, 5 FP/img)
GT visible: 283 / 85 / 1. Classical p=1: 184/20/99 (1 mrad), 54/20/31 (2 mrad). DINO spatial: 39/15/244, 13/13/72.

## Takeaways
- Classical cross-correlation (p = 1) is the best untrained method at 1 and 2 mrad: 3–5× the TPs of either DINO map at the same FP rate.
- Frozen DINO features add no detection value yet; spatial matching beats the averaged query only at 2 and 4 mrad.
- At 4 mrad the disks overlap, and every method finds < 10 % of spots (phase correlation p = 0 is least bad).
- The small decoder already beats classical p=1 on held-out thicknesses (126 vs 95 TP at 1 mrad, 44 vs 31 at 2 mrad), but it is trained on only 6 patterns.
- Single crystal / one orientation → this is a debugging benchmark, not a generalisation test.

Full tables (incl. no-threshold counts): `tp_fp_fn_counts.txt`; run log: `benchmark_smoke_test.log`.

Figures: `froc.png`, `recall_vs_thickness.png`, `example_maps_t200.png`, `decoder_heatmaps.png`.
