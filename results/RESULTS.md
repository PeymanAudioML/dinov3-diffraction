# Bragg-spot detection with a trained decoder — smoke-test results

Notebook: `dinov3/notebooks/dummy_data_explore.ipynb`. Data: simulated SiO₂, probe α = 1 / 2 / 4 mrad.

**Smoke test only:** CPU, 4 thicknesses (20, 200, 420, 460 Å), decoder trained 300 steps.
Full run (25 thicknesses, 1500 steps on GPU) still to do.

## Pipeline
1. **Ground truth** from `dataPots`: maxima ≥ 5 % of max, ≥ 5 px apart, direct beam removed → ~112 lattice spots per
   image, identical for every angle. "Visible" = disk in `dataMeas` ≥ 3× its surrounding ring.
2. **DINOv3 ViT-L/16** (frozen): crop 128 px → resize 1024 → 64×64 tokens (2 detector px per token).
3. **Decoder:** DINO tokens + raw pattern + probe image → spot heatmap + sub-pixel offsets.
4. **Split:** train on thickness < 400 Å, test on ≥ 400 Å; all three angles of a thickness on the same side.
   Smoke test: 6 train / 6 test images (full run: 57 / 18).
5. **Scoring:** TP = within 2 px of an unmatched GT spot, FP otherwise (direct beam ignored), FN = GT spot not found.
   Threshold per angle chosen so FP = 5 per image.

## TP / FP / FN per held-out test image (all lattice reflections, 5 FP/img threshold)

| angle | thickness | GT | TP | FP | FN |
|---|---|---|---|---|---|
| 1 mrad | 420 Å | 112 | 62 | 4 | 50 |
| 1 mrad | 460 Å | 112 | 64 | 6 | 48 |
| 2 mrad | 420 Å | 111 | 23 | 6 | 88 |
| 2 mrad | 460 Å | 112 | 21 | 4 | 91 |
| 4 mrad | 420 Å | 110 | 2 | 3 | 108 |
| 4 mrad | 460 Å | 112 | 1 | 7 | 111 |

## Recall and position error

| | 1 mrad | 2 mrad | 4 mrad |
|---|---|---|---|
| recall @ 5 FP/img, all reflections | 0.56 | 0.20 | 0.01 |
| recall @ 5 FP/img, visible spots | 0.61 | 0.39 | – (no visible spots) |
| median position error (px) | 0.38 | 0.78 | 1.04 |

## Takeaways
- 1 mrad: more than half of all spots are found, with sub-pixel accuracy. Most FN are faint outer reflections.
- 2 mrad: the central disks are found, but outer and overlapping ones are missed.
- 4 mrad: the disks merge into one blob and the decoder finds almost nothing.
- **Caveat:** all images share identical spot positions (one crystal, one orientation), so the decoder may memorise
  the lattice (it marks spots in dark areas where nothing is visible). A real test needs other crystals / orientations
  or shifted patterns.

Figures: `decoder_tp_fp_fn.png` (green + TP, red × FP, yellow ○ FN on every test image),
`decoder_froc_recall.png` (FROC + recall vs thickness), `decoder_heatmaps.png`. Log: `decoder_smoke_test.log`.
