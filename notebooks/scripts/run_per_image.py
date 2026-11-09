# Usage: python run_per_image.py OUT.csv [--crystals NAME ...] [CHMV2_ROWS.pkl DINOSIM_ROWS.pkl]
#   --crystals: only crystals whose name starts with one of these (e.g. --crystals ZrSn3); default all.
# Scores every image separately (one crystal, one probe angle, one thickness). For each image the threshold is the
# lowest score that keeps at most k false positives in that image (k = 1, 5); recall = found / visible in that image.
# Images with no visible reflection get NaN. CPU only. Methods:
#   "true log + disc-ring"           the method (reflect padding, as in the notebook)          SiO2 + 24 crystals
#   "true log + disc-ring, zero pad" the same with zero padding (edge-fix comparison)          SiO2 + 24 crystals
#   "filled disk (linear)"           raw intensity correlated with a filled disk                SiO2 + AlCo3
#   "true log + Sobel probe match"   Sobel magnitude of true log x Sobel magnitude of dataProbe SiO2 + 24 crystals
# Two cutoffs, both chosen per image using the ground truth: "vis" (≤ k false detections among visible reflections)
# and "all" (≤ 5 false detections when every lattice reflection is a target). At the "all" cutoff the script also
# reports the false-detection count and the median / 90th-percentile centre error of the correct detections.
# Optional: per-image rows saved by run_all_crystals.py (CHMv2) and run_dinosim.py (DINOSim) are scored the same way.
import json, sys, pickle
args = sys.argv[1:]; OUT = args.pop(0); CRYS = None
if "--crystals" in args:
    i = args.index("--crystals"); j = next((k for k in range(i + 1, len(args)) if args[k].endswith(".pkl")), len(args))
    CRYS = args[i + 1:j]; del args[i:j]
EXTRA = args
nb = json.load(open("/mnt/data/DINOV3/dinov3/notebooks/dummy_data_explore.ipynb")); c = nb["cells"]
src = lambda i: "".join(c[i]["source"])
s10 = src(10); s10 = s10[:s10.index("tl_in, tl_out, tl_rows")]
s12 = src(12); s12 = s12[:s12.index("loc_rows = []")]
g = {"OUT": OUT, "EXTRA": EXTRA, "CRYS": CRYS}
exec(compile("import matplotlib; matplotlib.use('Agg')\nimport torch\ndisplay=print\nDPT_SIZE=1024\n" + src(1) + "\n" + src(2) + "\n" + src(4) + "\n" + s10 + "\n" + s12, "setup", "exec"), g)
exec(compile(r'''
import time, pickle
ROOT = Path("/mnt/data/DINOV3/dinov3/dummy_data/selected_folders")
def load_crystal(folder):
    meta = pd.read_pickle(folder / "simulation_dataframe.pkl"); meta.index = meta.index.astype(str)
    ss = [load_sim(p) for p in sorted(folder.glob("*_training.h5"))]
    for s in ss: s["meta"] = meta.loc[s["uuid"]]
    return sorted(ss, key=lambda s: s["meta"]["probeSemiangle"])

def score_image(row):
    """One image → recall at ≤1 / ≤5 FP in this image, all-reflection recall, errors of the kept TPs."""
    o = dict(n_vis=row["n_vis"], n_all=row["n_all"])
    thr5, rec5 = operating_point([row], 5, "vis"); thr1, rec1 = operating_point([row], 1, "vis")
    o["vis5"] = rec5 if row["n_vis"] else np.nan; o["vis1"] = rec1 if row["n_vis"] else np.nan
    o["all5"] = operating_point([row], 5, "all")[1] if row["n_all"] else np.nan
    o["fp5"] = int(((row["status"] == 0) & (row["scores"] >= thr5)).sum())
    if "err" in row:
        e = row["err"][(row["status"] == 1) & (row["scores"] >= thr5)]
        o["err_med"] = float(np.median(e)) if len(e) else np.nan; o["err_p90"] = float(np.percentile(e, 90)) if len(e) else np.nan
    if "err_all" in row:                                        # all-reflection cutoff: ≤ 5 false detections per image
        thrA = operating_point([row], 5, "all")[0]; keep = row["scores"] >= thrA
        tp, fp = keep & (row["status_all"] == 1), keep & (row["status_all"] == 0)
        e = row["err_all"][tp]
        o.update(tp_all5=int(tp.sum()), fp_all5=int(fp.sum()),
                 err_all_med=float(np.median(e)) if len(e) else np.nan, err_all_p90=float(np.percentile(e, 90)) if len(e) else np.nan)
    return o

def run(m, gt):
    det, sc = pick_peaks(m); det = det + c0
    keep = in_region(det); det, sc = det[keep], sc[keep]
    return dict(scores=sc, **match(det, gt), n_vis=int(gt["visible"].sum()), n_all=len(gt["centers"]))

crystals = [("SiO2", sims)]
for f in sorted(ROOT.iterdir()):
    if f.is_dir():
        ss = load_crystal(f); m0 = ss[0]["meta"]; crystals.append((f"{m0['pretty_formula']} ({m0['space_group_symbol']})", ss))
if CRYS:
    crystals = [(cr, ss) for cr, ss in crystals if any(cr.startswith(k) for k in CRYS)]
out, t0 = [], time.time()
for ci, (cr, ss) in enumerate(crystals):
    for s in ss:
        _, r = probe_geometry(s["dataProbe"]); mr = s["meta"]["probeSemiangle"]; K = disc_ring_kernel(r)
        cen = ground_truth(s, 0, lo=0, hi=255)["centers"]; dm = np.hypot(*(cen[:, None] - cen[None]).transpose(2, 0, 1))
        np.fill_diagonal(dm, np.inf); sp = float(np.median(dm.min(1)))
        for t, th in enumerate(s["thicknesses"]):
            gt = ground_truth(s, t); x = true_log01(s["dataMeas"][..., t], EPS_LOC)
            maps = {"true log + disc-ring": disc_centre_map(x, r, +1),
                    "true log + disc-ring, zero pad": fftconvolve(x, K, mode="same"),
                    "true log + Sobel probe match": sobel_probe_map(x, s["dataProbe"])}
            if cr in ("SiO2", "AlCo3 (Pm-3m)"):
                maps["filled disk (linear)"] = fftconvolve(crop(s["dataMeas"][..., t]).astype(np.float64), disc_kernel(r), mode="same")
            for meth, m in maps.items():
                out.append(dict(crystal=cr, mrad=mr, t_idx=t, thickness=float(th), method=meth, r=r, spacing=sp,
                                **score_image(run(m, gt))))
    print(f"{ci + 1}/{len(crystals)} {cr} {time.time() - t0:.0f}s", flush=True)

TH = {(o["crystal"], o["mrad"], o["t_idx"]): o["thickness"] for o in out}
TH.update({("AlCo3", k[1], k[2]): v for k, v in TH.items() if k[0] == "AlCo3 (Pm-3m)"})
for p in EXTRA:
    for row in pickle.load(open(p, "rb")):
        if row.get("post", "none") != "none" or not (row["method"].startswith("CHMv2") or row["method"].startswith("DINOSim")):
            continue
        cr = row["crystal"]; key = (cr, row["mrad"], row["t_idx"])
        out.append(dict(crystal=cr, mrad=row["mrad"], t_idx=row["t_idx"], thickness=TH.get(key, np.nan),
                        method=row["method"], **score_image(row)))
pd.DataFrame(out).round(4).to_csv(OUT, index=False); print("saved", len(out), "rows")
''', "run", "exec"), g)
