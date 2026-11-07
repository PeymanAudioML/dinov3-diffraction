# dinov3-diffraction

Exploring frozen [DINOv3](https://github.com/facebookresearch/dinov3) features on simulated electron diffraction
(4D-STEM / CBED) patterns: encode diffraction patterns and their probe apertures, compare them with cosine
similarity, and locate Bragg discs from the similarity map.

- [`dinov3/notebooks/dummy_data_explore.ipynb`](dinov3/notebooks/dummy_data_explore.ipynb): data loading, DINOv3 encoding,
  probe-aperture similarity maps and Bragg-spot detection.
- [`NOTES.md`](NOTES.md): method, results, what `dataPots` is, and next steps.

## Method in short
1. DINOv3 ViT-L/16 (SAT-493M) backbone, frozen, no decoder head.
2. Centre-crop the 256×256 pattern to 128×128 and upsample to 1024 → 64×64 patch grid (2 detector px per patch).
3. Query = embedding of the probe-image patches that contain the aperture.
4. Cosine similarity of the query with every diffraction patch → local maxima → sub-pixel centroid refinement.

## Repository layout
- `dinov3/notebooks/dummy_data_explore.ipynb`, `NOTES.md`: this project.
- `dinov3/`: a copy of the [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) code
  (commit `6876159`), used through `torch.hub.load(..., source="local")`. It is © Meta Platforms, Inc. and
  distributed under the [DINOv3 License](dinov3/LICENSE.md); see that folder's own README.

## Requirements
- The bundled `dinov3/` folder: set `DINOV3_LOCATION` in the notebook to its path.
- DINOv3 weights `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`, obtained from Meta under the
  [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). Weights are not included here.
- Python packages: `torch`, `numpy`, `scipy`, `h5py`, `pandas`, `matplotlib`.
- Simulation data (`*_training.h5`, `simulation_dataframe.pkl`) is not included.

Paths in the notebook (`DINOV3_LOCATION`, `BACKBONE_WEIGHTS`, `DATA_DIR`) point to the original machine; edit them
to match yours.
