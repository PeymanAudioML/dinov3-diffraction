# DINOv3 on dummy diffraction data — notes

Companion notes for `dummy_data_explore.ipynb` (multislice / pyprismatic SiO2 simulations, probe α = 1, 2, 4 mrad,
thickness 20–500 Å in 25 steps).

## 1. Model: plain DINOv3 backbone
- DINOv3 ViT-L/16, SAT-493M pretraining, frozen. No DPT head / decoder.
- `encode_sim(s, t_idx)` returns, for one simulation:

| key | shape | meaning |
|---|---|---|
| `diff_cls` | (T, 1024) | global (CLS-token) embedding of each diffraction pattern; T = number of thicknesses (25, or 1 with `t_idx`) |
| `diff_patch` | (T, 1024, g, g) | patch-token embeddings of each diffraction pattern |
| `probe_cls` | (1024,) | global embedding of the probe aperture image |
| `probe_patch` | (1024, g, g) | patch-token embeddings of the probe aperture image |

- Preprocessing: diffraction = log1p → 1–99.9 % stretch; probe (≈ binary) = min–max. Then 3 channels, resize, ImageNet norm.
- **CLS score** = cos(`diff_cls`, `probe_cls`): one number per image pair, how alike the whole images are.

## 2. Bragg spots from probe-aperture similarity
1. **Query** = coverage-weighted mean of the probe patch embeddings that contain the aperture.
2. **Similarity map** = cos(query, every diffraction patch) → (g, g) map. Bragg discs in CBED are shifted copies of the
   probe aperture, so high values mark disc locations.
3. **Peaks** = local maxima above `min + PEAK_REL·(max − min)`, at least `MIN_DIST` patches apart.
4. **Sub-pixel refinement** = intensity-weighted centroid of the raw pattern in a ±`REFINE_R` px window.

Output per file: `reps[i]["peaks_px"]` (detector px), `reps[i]["peaks_q"]` (qx, qy in 1/Å).

### Resolution: change input size, not patch size
Patch size (16) is fixed by the pretrained weights. Each patch covers fewer detector pixels if the input is larger.
The spots lie within ≈ ±1 1/Å, so the image is centre-cropped first:

| setting | grid | detector px / patch | result |
|---|---|---|---|
| full 256 → 512 | 32×32 | 8 | spot region lights up as one blob |
| crop 128 → 1024 (current) | 64×64 | 2 | individual discs separate |

### Results at t = 200 Å (`PEAK_REL = 0.35`, `MIN_DIST = 2`, tuned by eye)
| probe | spots found | quality |
|---|---|---|
| 1 mrad | 25 | good: small, well-separated discs |
| 2 mrad | 28 | mostly good; a few peaks between discs |
| 4 mrad | 30 | unreliable: discs overlap (physical limit) |

Not yet scored against ground truth.

## 3. What is `dataPots`?
Not documented in the files; inferred from the data:

| check | result |
|---|---|
| same across 1/2/4 mrad files | yes (within 0.5 % of max) → specimen property, not probe-dependent |
| maximum | at q = 0 |
| symmetry | P(q) = P(−q) (Friedel) |
| sparsity | 4 % of pixels > 1 % of max → peaks at reciprocal-lattice points |
| P(0) vs thickness | P(0)/t ≈ 8.6 for t ≥ 100 Å → linear in t → a magnitude, not |F|² |

Most likely `dataPots` = |FT of the projected potential through thickness t|. At Bragg positions
`dataPots(g) ∝ t·|V_g| ∝ t·|F_e(g)|` (electron structure factor), with these differences from a strict structure factor:
magnitude only (no phase), continuous map with finite peak width, zero-order Laue zone only (HOLZ likely in `dataQz`),
scaled by thickness.

**As ground truth:** good for Bragg spot **locations** and for |F| up to a scale; not for measured spot intensities
(`dataMeas` includes dynamical scattering and the probe); no phases. Confirm with the simulation script (`script_name`
in the metadata).

## 4. Will a bigger DINOv3 help?
Probably only a little: the main limits are domain gap, patch resolution and using frozen features without training.
- Worth trying: ViT-L/16 **LVD-1689M** (same cost), ViT-H+/16 (fits a T4, ~3× slower).
- ViT-7B does not fit a 15 GB T4.
- Likely bigger gains: intermediate-layer features (`get_intermediate_layers`, e.g. blocks 12–18), a small trained head
  (linear probe / DPT) on spot labels from `dataPots`, finer resolution, and comparing against classical
  template matching (cross-correlation with the probe disc).

## 5. Next steps
- Score detected spots against `dataPots` peaks (precision, recall, position error), using the same columns as the
  earlier `predicted_vs_gt_per_pattern.csv`.
- Compare layers / models only after that evaluation exists.

## 6. Cross-crystal check: decoder applied to a new crystal
Last notebook cell: the decoder (model A, trained on SiO2 with thicknesses < 400 Å) is applied, unchanged, to AlCo3
(Pm-3m, cubic, mp-1018101) from `dummy_data/selected_folders/0c6670dd-…`. All 3 probe angles × 25 thicknesses are used.
Cut-off scores are the per-angle 5 FP/image thresholds from the SiO2 held-out set.

| α (mrad) | recall (all reflections) | precision | FP / image |
|---|---|---|---|
| 1 | 0.62 | 0.36 | 82 |
| 2 | 0.26 | 0.18 | 86 |
| 4 | 0.04 | 0.05 | 56 |

- 1 mrad: the bright, separated spots are found, with ~0.1 px position error on matches.
- The SiO2 cut-offs are too low here, so many detections land on background noise.
- 2 and 4 mrad: the overlapping disks differ from SiO2 and recall collapses.
- As expected for a decoder trained on one crystal and one orientation. Next: train across many crystals from
  `selected_folders` and split by crystal.
