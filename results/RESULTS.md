# Bragg-spot detection with a trained decoder: results

Notebook: `dinov3/notebooks/dummy_data_explore.ipynb`. Every number and figure here comes from the outputs saved in that
notebook (one top-to-bottom run, execution counts 1–15). Decoder training on GPU is not bit-for-bit reproducible, so a
rerun shifts the numbers by a few percent.

## Setup
1. **Data:** simulated SiO₂ (P3₂21), probe α = 1 / 2 / 4 mrad, 25 thicknesses (20–500 Å) per angle = 75 patterns.
2. **Ground truth** from `dataPots`: local maxima ≥ 5 % of the strongest reflection, direct beam removed → ~110 lattice
   spots per image in the scored region, the same positions for every angle. "Visible" = disk in `dataMeas` ≥ 3× its
   surrounding ring and ≥ 1e-3 of the brightest disk.
3. **DINOv3 ViT-L/16** (SAT-493M, frozen): central 128 px crop → 1024 px → 64×64 tokens (2 detector px per token).
4. **Decoder (model A):** DINO tokens + log-scaled pattern + probe image → spot heatmap + sub-pixel offsets.
   205,507 trainable parameters, 1500 steps, batch 4, AdamW + OneCycle, T4 GPU. Final training loss 0.046.
5. **Split:** train on thickness < 400 Å (57 images), test on 400–500 Å (18 images = 6 thicknesses × 3 angles).
6. **Scoring:** TP = within 2 px of an unmatched GT spot, FP otherwise (direct beam ignored), FN = GT spot not found.
   Per-angle threshold set so there are 5 FP per image on the test set.

## 1. Full run on SiO₂ (model A, held-out thicknesses 400–500 Å)

| | 1 mrad | 2 mrad | 4 mrad |
|---|---|---|---|
| recall @ 1 FP/img, all reflections | 0.951 | 0.714 | 0.131 |
| recall @ 5 FP/img, all reflections | **0.967** | **0.915** | **0.351** |
| median position error @ 5 FP/img (px) | 0.039 | 0.050 | 0.093 |
| recall @ 1 FP/img, visible spots | 0.941 | 0.627 | – |
| recall @ 5 FP/img, visible spots | 0.963 | 0.849 | – |
| GT spots: all / visible | 670 / 407 | 669 / 126 | 666 / 1 |

At 4 mrad only 1 spot in all 6 test images is "visible" (the disks overlap), so the visible-spot recall there is
meaningless and left out.

Totals at the 5 FP/img threshold:

| angle | GT | TP | FP | FN |
|---|---|---|---|---|
| 1 mrad | 670 | 648 | 30 | 22 |
| 2 mrad | 669 | 612 | 30 | 57 |
| 4 mrad | 666 | 234 | 30 | 432 |

Per test image:

| angle | thickness | GT | TP | FP | FN |
|---|---|---|---|---|---|
| 1 mrad | 400 Å | 110 | 108 | 4 | 2 |
| 1 mrad | 420 Å | 112 | 110 | 3 | 2 |
| 1 mrad | 440 Å | 112 | 108 | 3 | 4 |
| 1 mrad | 460 Å | 112 | 110 | 3 | 2 |
| 1 mrad | 480 Å | 112 | 107 | 5 | 5 |
| 1 mrad | 500 Å | 112 | 105 | 12 | 7 |
| 2 mrad | 400 Å | 111 | 108 | 2 | 3 |
| 2 mrad | 420 Å | 111 | 109 | 4 | 2 |
| 2 mrad | 440 Å | 111 | 102 | 5 | 9 |
| 2 mrad | 460 Å | 112 | 98 | 6 | 14 |
| 2 mrad | 480 Å | 112 | 100 | 7 | 12 |
| 2 mrad | 500 Å | 112 | 95 | 6 | 17 |
| 4 mrad | 400 Å | 110 | 57 | 0 | 53 |
| 4 mrad | 420 Å | 110 | 40 | 5 | 70 |
| 4 mrad | 440 Å | 110 | 35 | 4 | 75 |
| 4 mrad | 460 Å | 112 | 41 | 5 | 71 |
| 4 mrad | 480 Å | 112 | 33 | 9 | 79 |
| 4 mrad | 500 Å | 112 | 28 | 7 | 84 |

- 1 and 2 mrad: nearly all spots are found, with position errors well under 0.1 px. Recall drops slowly with thickness.
- 4 mrad: about a third of the spots are found; recall falls from 0.52 at 400 Å to 0.25 at 500 Å.

Figures: `decoder_froc_recall.png` (FROC + recall vs thickness), `decoder_heatmaps.png` (decoder heatmaps at 400 Å),
`decoder_tp_fp_fn.png` (TP/FP/FN on every test image).

## 2. Ablation: what do the DINOv3 features add?
Model B is model A without the DINO branch (pattern + probe only, 102,915 parameters). Everything else is identical:
training images, targets, steps, schedule, seed, test images and scoring. B's final training loss is 0.704 (A: 0.046).

| | A 1 mrad | B 1 mrad | A 2 mrad | B 2 mrad | A 4 mrad | B 4 mrad |
|---|---|---|---|---|---|---|
| recall @ 1 FP/img, all reflections | 0.951 | 0.458 | 0.714 | 0.372 | 0.131 | 0.123 |
| recall @ 5 FP/img, all reflections | **0.967** | 0.499 | **0.915** | 0.441 | **0.351** | 0.198 |
| median position error @ 5 FP/img (px) | 0.039 | 0.127 | 0.050 | 0.144 | 0.093 | 0.257 |
| recall @ 1 FP/img, visible spots | 0.941 | 0.715 | 0.627 | **0.810** | – | – |
| recall @ 5 FP/img, visible spots | 0.963 | 0.762 | 0.849 | 0.857 | – | – |

Totals at each model's 5 FP/img threshold (30 FP per angle for both):

| angle | GT | TP A | TP B | FN A | FN B |
|---|---|---|---|---|---|
| 1 mrad | 670 | 648 | 334 | 22 | 336 |
| 2 mrad | 669 | 612 | 295 | 57 | 374 |
| 4 mrad | 666 | 234 | 132 | 432 | 534 |

- With DINO features, recall on all reflections roughly doubles at every angle, and the position error is about 3× smaller.
- On **visible** spots the gap is much smaller: at 2 mrad, B matches A at 5 FP/img and beats it at 1 FP/img.
  Most of A's lead is on reflections that are faint or invisible in the pattern.
- Caveat: every SiO₂ image has the same spot positions (one crystal, one orientation), so A can learn where the lattice
  is and mark it even where nothing is visible. The cross-crystal check below supports this.

Figures: `ablation_froc.png` (A solid, B dashed), `ablation_tp_fp_fn.png` (TP/FP/FN, A and B, every test image).

## 3. Cross-crystal check: SiO₂-trained decoder on AlCo₃
Model A, unchanged, applied to AlCo₃ (Pm-3m, cubic, mp-1018101) from
`dummy_data/selected_folders/0c6670dd-6029-42cf-ab7b-cdbc8713c22d`: 3 angles × all 25 thicknesses (20–500 Å),
at the per-angle 5 FP/img thresholds from the SiO₂ test set (not re-tuned).

| angle | GT | TP | FP | FN | recall | precision | FP / image |
|---|---|---|---|---|---|---|---|
| 1 mrad | 1866 | 1105 | 2094 | 761 | 0.592 | 0.345 | 83.8 |
| 2 mrad | 1860 | 528 | 2167 | 1332 | 0.284 | 0.196 | 86.7 |
| 4 mrad | 1876 | 95 | 1305 | 1781 | 0.051 | 0.068 | 52.2 |

Threshold chosen on AlCo₃ itself (optimistic):

| | 1 mrad | 2 mrad | 4 mrad |
|---|---|---|---|
| recall @ 5 FP/img, all reflections | 0.342 | 0.077 | 0.006 |
| recall @ 5 FP/img, visible spots | 0.444 | 0.126 | 0.000 |
| median position error @ 5 FP/img (px) | 0.093 | 0.122 | 1.159 |

- At 1 mrad the bright, well-separated spots are still found accurately (~0.1 px).
- The SiO₂ thresholds are far too low for this crystal: ~80 FP per image instead of 5.
- At 5 FP/img recall drops from 0.97 / 0.92 / 0.35 (SiO₂) to 0.34 / 0.08 / 0.01: the decoder does not generalise to a
  new lattice. Training needs many crystals (e.g. the 24 in `selected_folders`) with the split done by crystal.
- The cell also lists every predicted spot centre in detector px and q (1/Å): 7,301 spots over the 75 patterns.

Figures: `cross_crystal_AlCo3_recall.png` (recall vs thickness), `cross_crystal_AlCo3_tp_fp_fn.png`
(3 angles × 6 thicknesses).

## Earlier smoke test
`decoder_smoke_test.log` is an earlier CPU run (4 thicknesses, 300 steps, 6 test images) that only checked that the
pipeline runs. Its numbers are superseded by section 1.
