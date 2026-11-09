# Usage: python run_overlap_analysis.py OUT_real.csv OUT_synthetic.csv      (CPU, ~4 min)
# Why disk overlap makes Bragg disks hard to locate with true log + disc-ring.
#  real:      every image of SiO2 + the 24 crystals: overlap ratio, how much of each disk's ring lies on neighbouring
#             disks, filter response at the true centres, fraction of disks darker than their surroundings.
#  synthetic: a square lattice (spacing 14 px) of flat disks whose radius is swept, so only the overlap changes;
#             disks add either incoherently (intensities) or coherently (amplitudes with random phases).
import json, sys
OUT_R, OUT_S = sys.argv[1], sys.argv[2]
nb = json.load(open("/mnt/data/DINOV3/dinov3/notebooks/dummy_data_explore.ipynb")); c = nb["cells"]
src = lambda i: "".join(c[i]["source"])
s10 = src(10); s10 = s10[:s10.index("tl_in, tl_out, tl_rows")]
s12 = src(12); s12 = s12[:s12.index("loc_rows = []")]
g = {"OUT_R": OUT_R, "OUT_S": OUT_S}
exec(compile("import matplotlib; matplotlib.use('Agg')\nimport torch\ndisplay=print\nDPT_SIZE=1024\n" + src(1) + "\n" + src(2) + "\n" + src(4) + "\n" + s10 + "\n" + s12, "setup", "exec"), g)
exec(compile(r'''
import time
from scipy.ndimage import map_coordinates, gaussian_filter
ROOT = Path("/mnt/data/DINOV3/dinov3/dummy_data/selected_folders")
GAP, WIDTH = 1.0, 2.0                                    # disc-ring kernel gap and ring width (px)

def load_crystal(folder):
    meta = pd.read_pickle(folder / "simulation_dataframe.pkl"); meta.index = meta.index.astype(str)
    ss = [load_sim(p) for p in sorted(folder.glob("*_training.h5"))]
    for s in ss: s["meta"] = meta.loc[s["uuid"]]
    return sorted(ss, key=lambda s: s["meta"]["probeSemiangle"])

def robust_z(S):
    med = np.median(S); return (S - med) / (1.4826 * np.median(np.abs(S - med)) + 1e-12)

def ring_on_neighbours(cen_eval, cen_all, r):
    """Fraction of each disk's ring (r+GAP, r+GAP+WIDTH] that lies inside another disk (radius r)."""
    R = int(np.ceil(r + GAP + WIDTH)) + 1; yy, xx = np.mgrid[-R:R + 1, -R:R + 1]; d = np.hypot(yy, xx)
    ring = np.stack([yy[(d > r + GAP) & (d <= r + GAP + WIDTH)], xx[(d > r + GAP) & (d <= r + GAP + WIDTH)]], 1).astype(float)
    out = []
    for p in cen_eval:
        pts = p + ring; others = cen_all[np.hypot(*(cen_all - p).T) > 0.5]
        dd = np.hypot(pts[:, None, 0] - others[None, :, 0], pts[:, None, 1] - others[None, :, 1]).min(1)
        out.append(np.mean(dd <= r))
    return np.asarray(out)

def rim_step(M, cen, r, d=1.5, n=48):
    """Median over the rim of log10(M just inside / M just outside); < 0 means the disk is darker than outside."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False); u = np.stack([np.sin(th), np.cos(th)], 1); out = []
    for p in cen:
        a = map_coordinates(M, (p + (r - d) * u).T, order=1, mode="nearest")
        b = map_coordinates(M, (p + (r + d) * u).T, order=1, mode="nearest")
        out.append(np.median(np.log10(np.maximum(a, 1e-30) / np.maximum(b, 1e-30))))
    return np.asarray(out)

def dark_stats(step, clear=np.log10(1.5)):
    """Disks with a clear rim (|step| >= x1.5): their fraction, and the fraction of those that are darker inside."""
    ok = np.abs(step) >= clear
    return dict(frac_rim=float(ok.mean()) if len(step) else np.nan,
                frac_dark=float((step[ok] < 0).mean()) if ok.any() else np.nan)

# ── real data ───────────────────────────────────────────────────────────────────────────────────────────────
crystals = [("SiO2", sims)]
for f in sorted(ROOT.iterdir()):
    if f.is_dir():
        ss = load_crystal(f); m0 = ss[0]["meta"]; crystals.append((f"{m0['pretty_formula']} ({m0['space_group_symbol']})", ss))
rows, t0 = [], time.time()
for ci, (cr, ss) in enumerate(crystals):
    for s in ss:
        _, r = probe_geometry(s["dataProbe"]); mr = s["meta"]["probeSemiangle"]
        full = ground_truth(s, 0, lo=0, hi=255)["centers"]                       # whole-detector lattice (for neighbours)
        for t, th_ in enumerate(s["thicknesses"]):
            gt = ground_truth(s, t); M = s["dataMeas"][..., t].astype(np.float64); cen = gt["centers"]
            z = robust_z(disc_centre_map(true_log01(M, EPS_LOC), r, +1))
            ij = np.clip(np.round(cen - c0).astype(int), 0, z.shape[0] - 1)
            rows.append(dict(crystal=cr, mrad=mr, t_idx=t, thickness=float(th_), r=r, n_all=len(cen), n_vis=int(gt["visible"].sum()),
                             ring_cover=float(np.median(ring_on_neighbours(cen, full, r))),
                             z_centre=float(np.median(z[tuple(ij.T)])),
                             **dark_stats(rim_step(M, cen, r))))
    print(f"{ci + 1}/{len(crystals)} {cr} {time.time() - t0:.0f}s", flush=True)
pd.DataFrame(rows).round(4).to_csv(OUT_R, index=False)

# ── synthetic overlap sweep ─────────────────────────────────────────────────────────────────────────────────
SP, N = 14.0, 256; yy, xx = np.mgrid[:N, :N].astype(float); ctr = np.array([N / 2, N / 2])
rng = np.random.default_rng(0); out = []
for r in np.arange(2.0, 14.5, 1.0):
    for rep in range(8):
        a = rng.uniform(0, np.pi / 2); R2 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        hk = np.array([(h, k) for h in range(-14, 15) for k in range(-14, 15)], float)
        pos = ctr + hk @ R2.T * SP; pos = pos[np.all((pos > -r) & (pos < N + r), 1)]
        direct = np.all(np.abs(pos - ctr) < 1e-6, 1)
        inten = np.where(direct, 1.0, 10 ** rng.uniform(-3, -1, len(pos)))      # weak to strong reflections
        phase = rng.uniform(0, 2 * np.pi, len(pos))
        psi = np.zeros((N, N), complex); I_inc = np.zeros((N, N))
        for p, I0, ph in zip(pos, inten, phase):
            disk = np.clip(r + 0.5 - np.hypot(yy - p[0], xx - p[1]), 0, 1)        # flat disk, 1-px soft edge
            I_inc += I0 * disk; psi += np.sqrt(I0) * np.exp(1j * ph) * disk
        bg = 1e-4 * np.exp(-((yy - ctr[0]) ** 2 + (xx - ctr[1]) ** 2) / (2 * 60.0 ** 2))   # diffuse background
        gt = dict(centers=pos[~direct], visible=np.ones((~direct).sum(), bool), direct=ctr)
        keep = in_region(gt["centers"]); gt["centers"], gt["visible"] = gt["centers"][keep], gt["visible"][keep]
        for model, I in (("incoherent", I_inc + bg), ("coherent", np.abs(psi) ** 2 + bg)):
            x = true_log01(I, EPS_LOC); S = disc_centre_map(x, r, +1); z = robust_z(S)
            det, sc = pick_peaks(S); det = det + c0; k = in_region(det); det, sc = det[k], sc[k]
            row = dict(scores=sc, **match(det, gt), n_vis=len(gt["centers"]), n_all=len(gt["centers"]))
            thr, rec = operating_point([row], 5, "vis")
            e = row["err"][(row["status"] == 1) & (row["scores"] >= thr)]
            ij = np.clip(np.round(gt["centers"] - c0).astype(int), 0, z.shape[0] - 1)
            out.append(dict(model=model, r=r, spacing=SP, eta=2 * r / SP, rep=rep, recall5=rec,
                            err_med=float(np.median(e)) if len(e) else np.nan, z_centre=float(np.median(z[tuple(ij.T)])),
                            **dark_stats(rim_step(I, gt["centers"], r))))
    print("synthetic r =", r, flush=True)
pd.DataFrame(out).round(4).to_csv(OUT_S, index=False); print("done")
''', "run", "exec"), g)
