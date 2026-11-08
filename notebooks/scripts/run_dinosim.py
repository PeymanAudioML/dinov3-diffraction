# Usage: python run_dinosim.py OUT.pkl two|all
# DINOSim (github.com/AAitorG/napari-DINOSim, cloned to /mnt/data/DINOV3/napari-DINOSim) with the frozen DINOv3
# ViT-L/16 SAT-493M backbone. Reference = 5 random visible spots on the 200 Å image of each file; the other 24
# thicknesses are scored against the disc-ring filter on the same images. "two" = SiO2 + AlCo3, "all" = 24 crystals.
# DINOSim (DINOv2-style patch-similarity + kNN, github.com/AAitorG/napari-DINOSim) on diffraction patterns,
# with the frozen DINOv3 ViT-L/16 SAT-493M backbone. Reference = a few "clicked" visible spots on one thickness.
import json, sys, types, importlib, pickle, time
from pathlib import Path
OUT, WHICH = Path(sys.argv[1]), sys.argv[2]            # WHICH: "two" (SiO2 + AlCo3) or "all"
N_CLICK, T_REF, K_NN, IMG_SIZE = 5, 9, 5, 1024

# Load DINOSim's pipeline module without its napari GUI / file-reading deps (cv2, tifffile are only used to read files)
import importlib.machinery
def _stub(name, **attrs):
    m = types.ModuleType(name); m.__spec__ = importlib.machinery.ModuleSpec(name, None); m.__dict__.update(attrs)
    sys.modules[name] = m
_stub("tqdm", tqdm=lambda x, **k: x); _stub("cv2", imread=None); _stub("tifffile", imread=None)
pkg = types.ModuleType("dinosim_utils"); pkg.__path__ = ["/mnt/data/DINOV3/napari-DINOSim/src/napari_dinosim/utils"]
sys.modules["dinosim_utils"] = pkg
DINOSim_pipeline = importlib.import_module("dinosim_utils.dinoSim_pipeline").DINOSim_pipeline
get_img_processing_f = importlib.import_module("dinosim_utils.utils").get_img_processing_f

nb = json.load(open("/mnt/data/DINOV3/dinov3/notebooks/dummy_data_explore.ipynb")); c = nb["cells"]
src = lambda i: "".join(c[i]["source"])
s10 = src(10); s10 = s10[:s10.index("tl_in, tl_out, tl_rows")]
s12 = src(12); s12 = s12[:s12.index("loc_rows = []")]
s15 = src(15); s15 = s15[:s15.index("AD_CRYSTALS = ")]
g = dict(OUT=OUT, WHICH=WHICH, N_CLICK=N_CLICK, T_REF=T_REF, K_NN=K_NN, IMG_SIZE=IMG_SIZE,
         DINOSim_pipeline=DINOSim_pipeline, get_img_processing_f=get_img_processing_f)
pre = "import matplotlib; matplotlib.use('Agg')\nimport torch\ndisplay = print\nDPT_SIZE = 1024\n"
exec(compile(pre + src(1) + "\n" + src(2) + "\n" + src(4) + "\n" + s10 + "\n" + s12 + "\n" + s15, "setup", "exec"), g)
exec(compile(r'''
import time, pickle
backbone = torch.hub.load(DINOV3_LOCATION, "dinov3_vitl16", source="local", weights=BACKBONE_WEIGHTS).to(DEVICE).eval()
ROOT = Path("/mnt/data/DINOV3/dinov3/dummy_data/selected_folders")
def load_crystal(folder):
    meta = pd.read_pickle(folder / "simulation_dataframe.pkl"); meta.index = meta.index.astype(str)
    ss = [load_sim(p) for p in sorted(folder.glob("*_training.h5"))]
    for s in ss: s["meta"] = meta.loc[s["uuid"]]
    return sorted(ss, key=lambda s: s["meta"]["probeSemiangle"])
crystals = [("SiO2", sims)]
folders = sorted(f for f in ROOT.iterdir() if f.is_dir())
if WHICH == "two":
    folders = [f for f in folders if f.name.startswith("0c6670dd")]
rows = pickle.load(open(OUT, "rb")) if OUT.exists() else []
done_ = {(r["crystal"]) for r in rows}
rng = np.random.default_rng(0)
t0 = time.time()

def evaluate(name, crystal, s, t, m, gt):
    det, sc = pick_peaks(m); det = det + c0
    keep = in_region(det); det, sc = det[keep], sc[keep]
    return dict(crystal=crystal, method=name, mrad=s["meta"]["probeSemiangle"], t_idx=t, scores=sc.astype(np.float32),
                **{k: v.astype(np.int8) if k.startswith("status") else v for k, v in match(det, gt).items() if k.startswith("status")},
                n_vis=int(gt["visible"].sum()), n_all=len(gt["centers"]))

def run_crystal(crystal, ss):
    for s in ss:
        _, r = probe_geometry(s["dataProbe"]); beam = beam_xy(s); T = len(s["thicknesses"])
        for inp_name, prep in (("true log", lambda img: true_log01(img, EPS_LOC)), ("adaptive", lambda img: adaptive01(img, r, beam))):
            x = np.stack([prep(s["dataMeas"][..., t]) for t in range(T)]).astype(np.float32)[..., None]    # (T,128,128,1)
            pipe = DINOSim_pipeline(backbone, backbone.patch_size, torch.device(DEVICE),
                                    get_img_processing_f(resize_size=IMG_SIZE), backbone.embed_dim, dino_image_size=IMG_SIZE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                pipe.pre_compute_embeddings(x, overlap=(0, 0), padding=(0, 0), crop_shape=(cw, cw, 1), verbose=False)
            gref = ground_truth(s, T_REF); vis = gt_c = gref["centers"][gref["visible"]] - c0               # crop px
            pick = rng.choice(len(vis), size=min(N_CLICK, len(vis)), replace=False)
            pipe.set_reference_vector([(T_REF, float(p[1]), float(p[0])) for p in vis[pick]])           # (n, x=col, y=row)
            dist = pipe.get_ds_distances_sameRef(verbose=False, k=K_NN)                                  # (T, 64, 64), low = like the clicks
            dist = pipe.distance_post_processing(dist, None, "bilinear")[..., 0]                         # (T, 128, 128)
            del pipe; torch.cuda.empty_cache()
            for t in range(T):
                if t == T_REF:
                    continue                                                                             # reference image not scored
                gt = ground_truth(s, t)
                rows.append(evaluate(f"DINOSim ({inp_name})", crystal, s, t, -dist[t], gt))
                rows.append(evaluate(f"disc-ring ({inp_name})", crystal, s, t, disc_centre_map(x[t, ..., 0], r, +1), gt))

if "SiO2" not in done_:
    run_crystal("SiO2", sims); pickle.dump(rows, open(OUT, "wb")); print("SiO2", round(time.time() - t0), "s", flush=True)
for ci, folder in enumerate(folders):
    ss = load_crystal(folder); m0 = ss[0]["meta"]; crystal = f"{m0['pretty_formula']} ({m0['space_group_symbol']})"
    if crystal in done_:
        continue
    run_crystal(crystal, ss); pickle.dump(rows, open(OUT, "wb"))
    print(f"{ci + 1}/{len(folders)} {crystal}  {time.time() - t0:.0f} s", flush=True)
print("done")
''', "run", "exec"), g)
