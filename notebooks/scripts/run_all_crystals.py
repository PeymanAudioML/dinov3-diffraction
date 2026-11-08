# Usage: python run_all_crystals.py OUT.pkl
# Runs the adaptive-preprocessing / Friedel / lattice cell of dummy_data_explore.ipynb on all 24 crystals in
# dummy_data/selected_folders (inference only, ~2 min per crystal on a T4) and saves one row per image × method.
# Reuses the notebook's own code: cells 1, 2, 4, 6, 10, 12, 15 (indices must match the notebook). Resumes if OUT exists.
# Then: python summarize.py OUT.pkl ../results/all_crystals_recall.csv
# Adaptive / CHMv2(adaptive) readouts on every crystal in selected_folders (inference only).
import json, sys, time, pickle
from pathlib import Path
OUT = Path(sys.argv[1])
nb = json.load(open("/mnt/data/DINOV3/dinov3/notebooks/dummy_data_explore.ipynb")); c = nb["cells"]
src = lambda i: "".join(c[i]["source"])
s10 = src(10); s10 = s10[:s10.index("tl_in, tl_out, tl_rows")]
s12 = src(12); s12 = s12[:s12.index("loc_rows = []")]
s15 = src(15); s15 = s15[:s15.index("AD_CRYSTALS = ")]
pre = "import matplotlib; matplotlib.use('Agg')\nimport torch\ndisplay = print\n"
g = {"OUT": OUT}
exec(compile(pre + src(1) + "\n" + src(2) + "\n" + src(4) + "\n" + src(6) + "\n" + s10 + "\n" + s12 + "\n" + s15, "setup", "exec"), g)
exec(compile('\n# Memory-light, mathematically identical replacement for the CHMv2 soft-argmax (avoids a 1 GB temporary).\nimport dinov3.eval.depth.models as _dm\ndef _mixlog_lowmem(input, bins, max_clamp_value=1e-4, eps_shift=1e-8, eps=1e-12):\n    y = torch.relu(input)\n    shift = (-y.amin(dim=1, keepdim=True)).clamp_min(0.0).clamp_max(max_clamp_value) + eps_shift\n    y.add_(shift)\n    denom = torch.nan_to_num(y.sum(dim=1, keepdim=True), nan=1.0, posinf=1.0, neginf=1.0).clamp_min(eps)\n    num = torch.einsum("nbhw,b->nhw", y, bins.clamp_min(eps).to(y.dtype)).unsqueeze(1)\n    return (num / denom).clamp_min(eps) * 8.0\n_dm.create_outputs_with_chmv2_mixlog_norm = _mixlog_lowmem\n' + r'''
import time, pickle
ROOT = Path("/mnt/data/DINOV3/dinov3/dummy_data/selected_folders")
def load_crystal(folder):
    meta = pd.read_pickle(folder / "simulation_dataframe.pkl"); meta.index = meta.index.astype(str)
    ss = [load_sim(p) for p in sorted(folder.glob("*_training.h5"))]
    for s in ss: s["meta"] = meta.loc[s["uuid"]]
    return sorted(ss, key=lambda s: s["meta"]["probeSemiangle"])

rows = pickle.load(open(OUT, "rb")) if OUT.exists() else []
done_ = {r["uuid"] for r in rows}; t0 = time.time()
print(f"resuming: {len(done_)} crystals already saved", flush=True)
folders = sorted(f for f in ROOT.iterdir() if f.is_dir())
for ci, folder in enumerate(folders):
    if folder.name in done_:
        continue
    ss = load_crystal(folder); m0 = ss[0]["meta"]
    crystal = f"{m0['pretty_formula']} ({m0['space_group_symbol']})"
    for s in ss:
        _, r = probe_geometry(s["dataProbe"]); beam = beam_xy(s); mr = s["meta"]["probeSemiangle"]
        for t in range(len(s["thicknesses"])):
            img, gt = s["dataMeas"][..., t], ground_truth(s, t)
            x_ad = adaptive01(img, r, beam).astype(np.float32)
            x_tl = true_log01(img, EPS_LOC).astype(np.float32)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                out_ad = F.interpolate(chmv2_model(to_chmv2(x_ad).to(DEVICE)).float(),
                                       size=(cw, cw), mode="area")[0, 0].cpu().numpy()
            assert np.isfinite(out_ad).all(), (crystal, mr, t)
            ref = disc_centre_map(x_ad, r, +1)
            chm, sign = oriented(disc_centre_map(np.log10(np.clip(out_ad, 0, None) + 1e-2), r, +1), ref)
            maps = {"Input true log ε=1e-4 + disc-ring (stage 1)": disc_centre_map(x_tl, r, +1),
                    "Input adaptive + disc-ring": ref, "CHMv2(adaptive) + disc-ring": chm}
            for name, m in maps.items():
                z = robust_z(m); det, sc = pick_peaks(m)
                for post in POSTS:
                    d, c_ = det, sc
                    if post != "none": c_ = friedel_rescore(d, c_, z, beam)
                    if post == "Friedel + lattice": d, c_ = lattice_filter(d, c_, beam, fit_lattice(d, c_, beam))
                    o = np.argsort(-c_); d, c_ = d[o] + c0, c_[o]
                    keep = in_region(d); d, c_ = d[keep], c_[keep]
                    mt = match(d, gt)
                    rows.append(dict(uuid=folder.name, crystal=crystal, method=name, post=post, mrad=mr, t_idx=t,
                                     scores=c_.astype(np.float32), status=mt["status"].astype(np.int8),
                                     status_all=mt["status_all"].astype(np.int8),
                                     sign=sign if name.startswith("CHMv2") else np.nan,
                                     n_vis=int(gt["visible"].sum()), n_all=len(gt["centers"])))
    print(f"{ci + 1}/{len(folders)} {crystal}  {time.time() - t0:.0f} s", flush=True)
    pickle.dump(rows, open(OUT, "wb"))
print("done")
''', "run", "exec"), g)
