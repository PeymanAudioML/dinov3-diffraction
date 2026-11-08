# DINOv3 + DPT (CHMv2) on simulated diffraction patterns: results

Companion notes for `dummy_data_explore.ipynb`. The notebook now holds only the DINOv3 + DPT experiments; the earlier
trained-decoder experiments are summarised in section 9 and live in git history (commit `c72ba57`).

## 1. Setup

**Model:** `dinov3_vitl16_chmv2` (`dinov3.hub.depthers`), used as is, nothing trained.

| part | details | weights |
|---|---|---|
| backbone | DINOv3 ViT-L/16, SAT-493M (satellite) pretraining, 24 blocks, 1024-d, ~300 M params, frozen | `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` |
| head | DPT decoder on blocks 5, 11, 17, 23, trained to predict canopy height in metres (0.001–96) | `dinov3_vitl16_chmv2_dpt_head-3703d643.pth` |

**Data:** multislice (pyprismatic) simulations, probe semiangle α = 1, 2, 4 mrad × 25 thicknesses (20–500 Å), 256×256 px.
- SiO2: `dummy_data/` (spot spacing ≈ 8.5 px).
- AlCo3 (Pm-3m, cubic): `dummy_data/selected_folders/0c6670dd-…` (spot spacing ≈ 14 px).

**Input:** central 128 px crop → scaled to [0, 1] → 3 channels → bilinear upsample to 1024×1024 (64×64 tokens, 2 detector px
per token) → CHMv2 mean/std. Two scalings are compared:
- **log1p** (used so far): log1p then 1–99.9 % stretch. Because `dataMeas` peaks at ~1 and log1p(x) ≈ x for x ≪ 1, this is
  almost linear: only the brightest disks are visible.
- **true log:** log10(x + ε·max x) mapped to [0, 1]; ε sets the floor (ε = 3e-4 in the scaling study, 1e-4 in stage 1).

**Ground truth and scoring:** spot centres = local maxima of `dataPots` (direct beam excluded). A reflection is *visible*
if its disk in `dataMeas` is ≥ 3× its surrounding ring (r+1 … r+3 px). Detections are matched greedily within 2 px;
invisible lattice points and the direct beam are ignored, not counted as FP. Recall is reported at 1 and 5 false positives
per image, over all 25 thicknesses. Thresholds are picked on the evaluated set, so the numbers are optimistic.

## 2. Raw CHMv2 output on AlCo3

The model returns a 1024×1024 "canopy height" map per pattern. On diffraction input (log1p):
- Most of the map sits at the output floor (0.0098 m); the mean is 0.01–0.3 m.
- The maximum jumps between 0.02 m and 53 m from one thickness to the next, with no trend in thickness or angle.
- Correlation of the output with log intensity is weak: 0.11–0.30 at 1 mrad, −0.04–0.13 at 2 mrad, 0.00–0.09 at 4 mrad.

The output responds to some disks but is not a stable function of the diffraction intensity.

## 3. Input scaling: log1p vs true log (AlCo3, ε = 3e-4)

| | log1p 1 / 2 / 4 mrad | true log 1 / 2 / 4 mrad |
|---|---|---|
| reflections standing out in the input (disk ≥ 0.05 above its ring) | 0.13 / 0.12 / 0.07 | **0.82 / 0.76 / 0.23** |
| CHMv2 mean on disks | 0.76 / 0.013 / 0.053 | 5.88 / 0.30 / 0.88 |
| CHMv2 mean off disks | 0.017 / 0.011 / 0.040 | 13.3 / 1.28 / 2.43 |
| on / off ratio | 44 / 1.2 / 1.3 | **0.44 / 0.23 / 0.36** |

- True log makes 6× more reflections visible to the backbone at 1–2 mrad.
- The CHMv2 output **flips polarity**: with log1p the disks are bright (ratio > 1), with true log they are dark holes in a
  high background (ratio < 1). The head scores textured regions (rims, background speckle) as "canopy" and flat disk
  interiors as "ground".

Observations from the figures:
1. Polarity flips with preprocessing, so no fixed peak rule works.
2. Under true log each disk is a flat plateau: no single extremum per disk.
3. Rims and crescents light up (log1p at 1 mrad, 4 mrad outlines), so peaks sit about one radius off-centre.
4. A low ε floor amplifies background speckle, which the head scores as canopy.
5. Strong dependence on disk size: 2 mrad disks barely register under log1p.
6. Thin samples (20 Å) give uniform disks without internal texture and behave differently.

**Diagnosis:** the frozen features do outline the disks (every disk is visible in the true-log output), but the canopy
head converts this into "texture vs smooth", which has no peak at the disk centre and whose sign depends on preprocessing.

## 4. Stage 1: geometric disk-centre readout

To test whether the right readout can recover centres from the CHMv2 output, each map is filtered with a disk-ring
kernel: mean inside radius r (the probe radius) minus mean in a ring r+1 … r+3 px. This peaks at disk centres for
plateaus of either sign. Peaks: local maxima (min distance 4 px), robust z-score, parabolic sub-pixel refinement.

| score map | what it tests |
|---|---|
| CHMv2 raw output (log1p) | the baseline that failed |
| CHMv2 true log + disk-ring (sign −1) | can the features locate disks once the readout is right? |
| input true log + disk-ring | the same readout on the input image, no network |
| raw input × probe disk | classical matched filter on linear intensity |

**Recall of visible spots at 5 FP per image (1 FP in brackets):**

| score map | SiO2 1 mrad | 2 mrad | 4 mrad | AlCo3 1 mrad | 2 mrad | 4 mrad |
|---|---|---|---|---|---|---|
| CHMv2 raw output | 0.31 (0.17) | 0.00 (0.00) | 0.00 (0.00) | 0.30 (0.01) | 0.00 (0.00) | 0.00 (0.00) |
| CHMv2 true log + disk-ring | 0.16 (0.03) | 0.01 (0.00) | 0.07 (0.00) | 0.38 (0.18) | 0.03 (0.00) | 0.19 (0.10) |
| **input true log + disk-ring** | **0.96 (0.89)** | **0.94 (0.78)** | **0.50 (0.36)** | **0.94 (0.83)** | **0.86 (0.72)** | **0.76 (0.51)** |
| raw input × probe disk | 0.65 (0.60) | 0.71 (0.52) | 0.14 (0.14) | 0.93 (0.83) | 0.80 (0.57) | 0.55 (0.33) |

Findings:
- **The geometric readout does not rescue the CHMv2 output.** It is no better than the raw output on SiO2 and only
  slightly better on AlCo3; both stay near zero at 2 and 4 mrad. The canopy head loses the disk-centre information, so
  its map cannot serve as a spot score.
- **The no-network baseline is strong.** The disk-ring filter on the true-log input finds 86–96 % of visible spots at
  1–2 mrad on both crystals and beats the classical linear matched filter everywhere, most at 4 mrad. The true-log
  scaling is what makes faint disks usable.
- **SiO2 ≈ AlCo3** for the geometric readouts: no crystal-specific tuning is involved.
- **SiO2 at 4 mrad** is lowest for every method: the disks overlap (diameter ≈ 2.4× spot spacing), a physical limit.

**What disk-ring is:** a correlation of the image with a zero-mean template of the probe disk: +1/N inside radius r,
−1/M in the ring r+1 … r+3 px, 0 elsewhere (computed by FFT convolution; the template is symmetric). The score is
"mean inside the disk − mean just around it", so it peaks at disk centres and a flat background scores zero. It is a
matched filter, not normalised cross-correlation (no division by local brightness or variance), and uses no network
and no labels.

## 5. Adaptive preprocessing and Friedel / lattice post-processing

A fixed true-log floor works at 20 Å but fails at 200–500 Å, where the diffuse background rises above the floor. The
`adaptive01` preprocessing therefore (1) subtracts a smooth background (30th-percentile filter, ~3 disk diameters), (2) sets
the log floor at 3× the image's own residual level and (3) scales the top to the Bragg disks with the direct beam masked.
Post-processing adds a **Friedel check** (score = min of a detection and its mirror −g about the direct beam), a
**lattice fit** (two vectors fitted to the 40 strongest detections; detections > 1.5 px off-lattice dropped) and a
per-image **polarity check** for the CHMv2 map.

## 6. All 24 crystals in `selected_folders`

Run with `scripts/run_all_crystals.py` (inference only, ~50 min on a T4; 1,800 patterns); per-crystal results in
`results/all_crystals_recall.csv`, plotted in the notebook's last section.

**Median over crystals, visible-spot recall at 5 FP per image (1 / 2 / 4 mrad):**

| score map | no post-processing | + Friedel + lattice |
|---|---|---|
| true log + disk-ring | **0.86 / 0.86 / 0.65** | 0.83 / 0.79 / 0.49 |
| adaptive + disk-ring | 0.85 / 0.85 / 0.61 | 0.85 / 0.79 / 0.44 |
| CHMv2(adaptive) + disk-ring | 0.08 / 0.01 / 0.00 | 0.14 / 0.03 / 0.04 |

- **CHMv2 fails on every crystal:** it never beats the filter (0 of 72 crystal × angle cases) and still shows disks
  as dark in 89 % of images.
- **Adaptive ≈ true log** across crystals; its gain on SiO2 / AlCo3 does not generalise.
- **Friedel + lattice does not help on average** and hurts at 4 mrad: the lattice fitted from detections is unreliable
  for low-symmetry crystals (P1, Cm, Cc). Fitting the lattice from the image autocorrelation is the likely fix.
- **Crystal matters most:** most crystals reach 0.8–1.0 at 1 mrad; ErCo5Si3 and HoTe3 stay low for every method.

## 7. DINOSim with DINOv3 features

DINOSim ([González-Marfil et al., bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.03.09.642092v1);
code [AAitorG/napari-DINOSim](https://github.com/AAitorG/napari-DINOSim)) compares each patch embedding with those at a
few clicked reference points and refines with k-nearest neighbours (k = 5). It was run with the frozen DINOv3 ViT-L/16
SAT-493M backbone instead of DINOv2 (`scripts/run_dinosim.py`); reference = 5 random visible spots on the 200 Å image of
each file, the other 24 thicknesses scored. Results in `results/dinosim_recall.csv`.

| visible recall @ 5 FP (1 / 2 / 4 mrad) | SiO2 | AlCo3 |
|---|---|---|
| DINOSim (true log) | 0.80 / 0.14 / 0.00 | 0.84 / 0.08 / 0.00 |
| **DINOSim (adaptive)** | **0.97** / 0.04 / 0.00 | **0.93** / 0.04 / 0.02 |
| disk-ring (true log) | 0.96 / 0.94 / 0.54 | 0.94 / 0.86 / 0.72 |
| disk-ring (adaptive) | 0.97 / 0.93 / 0.38 | 0.94 / 0.89 / 0.63 |

- At 1 mrad with the adaptive input DINOSim **matches** the filter: the frozen DINOv3 features do locate small disks,
  unlike the CHMv2 head.
- At 2–4 mrad it collapses: larger disks span several patches, so the similarity map marks whole disks rather than
  centres, and peak picking finds several points per disk.
- DINOSim uses 5 labelled spots per file; the filter uses none.

## 8. Conclusions and next steps

- The pretrained DINOv3 + DPT canopy-height model, used as is, **cannot locate Bragg disks** on any of the 24 crystals,
  with either input scaling or with a geometric readout of its output. The bottleneck is the head, not the backbone
  features: DINOSim on the same frozen backbone matches the filter at 1 mrad.
- A simple filter on true-log input with no network (median 0.86 recall at 5 FP, 1–2 mrad, over 24 crystals) is the
  bar any DINOv3-based detector has to beat.
- **DINOSim + disk-ring readout:** read the DINOSim similarity map with the disk-ring filter to recover centres at
  2–4 mrad; then all 24 crystals, and a probe-disk reference to make it label-free.
- **Lattice from the image autocorrelation** instead of from detections.
- **Stage 2:** train a small heatmap + offset head on the frozen DINOv3 features (blocks 5, 11, 17, 23), with fixed
  true-log preprocessing (ε = 1e-4), several crystals from `selected_folders` split into train / validation / test by
  crystal, shift / flip / 90° augmentation, threshold chosen on the validation crystals. Compare against "input true
  log + disk-ring" and the classical filter on the held-out crystals.
- **Other DINOv3 models:** only the ViT-L/16 SAT-493M backbone and the CHMv2 head are downloaded. Worth trying for stage 2:
  ViT-L/16 **LVD-1689M** (web-image pretraining, same cost) and ViT-S/B for the effect of size. All other pretrained
  heads (depth, detection, segmentation) exist only for ViT-7B, which does not fit on the 15 GB T4.
- Caveats: thresholds come from the evaluated set; the sign of the CHMv2 true-log map (−1) was set from the figures,
  not tuned.

## 9. Earlier experiments (removed from the notebook)

Trained decoders on frozen plain DINOv3 ViT-L/16 features (SiO2, train on thicknesses < 400 Å, test 400–500 Å):
- **Model A** (DINO features + image + probe): recall @ 5 FP on all reflections 0.97 / 0.92 / 0.35 at 1 / 2 / 4 mrad,
  median position error ≤ 0.09 px.
- **Model B** (same decoder without DINO): 0.50 / 0.44 / 0.20. On *visible* spots B is close to A; much of A's lead is on
  faint reflections whose positions it learns from the single fixed SiO2 lattice.
- **Cross-crystal:** A applied unchanged to AlCo3 gives 0.59 / 0.28 / 0.05 with ~80 FP per image: it does not generalise
  to a new lattice. This motivated the crystal-split training in stage 2.
- `dataPots` is most likely |FT of the projected potential| through thickness t: good for spot locations, not for
  measured intensities.
