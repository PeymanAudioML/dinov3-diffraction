# Usage: python compare_sobel_zrsn3.py PER_IMAGE.csv OUT_DIR
# Sobel-magnitude probe matching vs disk-ring on ZrSn3 (1, 2, 4 mrad x 25 thicknesses), paired per image.
# PER_IMAGE.csv: output of  run_per_image.py OUT.csv --crystals ZrSn3
# Primary metric: recall of all lattice reflections at <= 5 false detections per image; the cutoff is selected per image
# using the ground truth (as in the rest of the benchmark). Writes:
#   sobel_vs_discring_ZrSn3_paired.csv    one row per image, both methods side by side
#   sobel_vs_discring_ZrSn3_summary.csv   one row per probe angle
#   sobel_vs_discring_ZrSn3_4mrad.png     overlays at 4 mrad, 20 / 200 / 500 A
import json, sys
PI, OUTD = sys.argv[1], sys.argv[2]
nb = json.load(open("/mnt/data/DINOV3/dinov3/notebooks/dummy_data_explore.ipynb")); c = nb["cells"]
src = lambda i: "".join(c[i]["source"])
s10 = src(10); s10 = s10[:s10.index("tl_in, tl_out, tl_rows")]
s12 = src(12); s12 = s12[:s12.index("loc_rows = []")]
g = {"PI": PI, "OUTD": OUTD}
exec(compile("import matplotlib; matplotlib.use('Agg')\nimport torch\ndisplay=print\nDPT_SIZE=1024\n" + src(1) + "\n" + src(2) + "\n" + src(4) + "\n" + s10 + "\n" + s12, "setup", "exec"), g)
exec(compile(r'''
from scipy.stats import wilcoxon
from matplotlib.lines import Line2D
OUTD = Path(OUTD); OUTD.mkdir(parents=True, exist_ok=True)
A, B = "true log + disc-ring", "true log + Sobel probe match"
COLS = ["all5", "tp_all5", "fp_all5", "err_all_med", "err_all_p90"]

# ── paired per-image table ──────────────────────────────────────────────────────────────────────────────────
d = pd.read_csv(PI); d = d[d.method.isin([A, B])]
key = ["crystal", "mrad", "t_idx", "thickness", "n_all"]
p = d[d.method == A][key + COLS].merge(d[d.method == B][key + COLS], on=key, suffixes=("_discring", "_sobel"))
p["d_all5"] = p.all5_sobel - p.all5_discring; p["d_tp_all5"] = p.tp_all5_sobel - p.tp_all5_discring
p = p.sort_values(["mrad", "t_idx"]); p.round(4).to_csv(OUTD / "sobel_vs_discring_ZrSn3_paired.csv", index=False)

# ── summary by probe angle ──────────────────────────────────────────────────────────────────────────────────
rows = []
for mr, q in p.groupby("mrad"):
    dd = q.d_all5.values; nz = dd[dd != 0]
    o = dict(mrad=mr, images=len(q), lattice_reflections_per_image=float(q.n_all.median()),
             sobel_better=int((dd > 0).sum()), sobel_worse=int((dd < 0).sum()), tie=int((dd == 0).sum()),
             wilcoxon_p=float(wilcoxon(nz).pvalue) if len(nz) >= 5 else np.nan)
    for m in ("discring", "sobel"):
        o[f"{m}_median_all5"] = float(q[f"all5_{m}"].median()); o[f"{m}_mean_all5"] = float(q[f"all5_{m}"].mean())
        o[f"{m}_true_centres_found"] = int(q[f"tp_all5_{m}"].sum())
        o[f"{m}_median_fp"] = float(q[f"fp_all5_{m}"].median())
        o[f"{m}_median_err_px"] = float(q[f"err_all_med_{m}"].median())
        o[f"{m}_median_err_p90_px"] = float(q[f"err_all_p90_{m}"].median())
    o["lattice_reflections_total"] = int(q.n_all.sum())
    rows.append(o)
S = pd.DataFrame(rows); S.round(4).to_csv(OUTD / "sobel_vs_discring_ZrSn3_summary.csv", index=False)
print(S.round(3).T.to_string())

# ── overlays at 4 mrad ──────────────────────────────────────────────────────────────────────────────────────
ROOT = Path("/mnt/data/DINOV3/dinov3/dummy_data/selected_folders")
f = next(f for f in ROOT.iterdir() if f.is_dir() and pd.read_pickle(f / "simulation_dataframe.pkl")["pretty_formula"].iloc[0] == "ZrSn3")
meta = pd.read_pickle(f / "simulation_dataframe.pkl"); meta.index = meta.index.astype(str)
ss = [load_sim(q_) for q_ in sorted(f.glob("*_training.h5"))]
s = next(s_ for s_ in ss if meta.loc[s_["uuid"]]["probeSemiangle"] == 4); _, r = probe_geometry(s["dataProbe"])
INK, MUTED = "#1f1f1e", "#6b6a64"; C_TP, C_FP, C_FN, C_GT = "#1baf7a", "#eb6834", "#eda100", "#2a78d6"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "savefig.dpi": 200})
fig, axes = plt.subplots(2, 3, figsize=(6.6, 5.0), layout="constrained")
for j, t in enumerate([0, 9, 24]):
    gt = ground_truth(s, t); x = true_log01(s["dataMeas"][..., t], EPS_LOC); cen = gt["centers"]
    for i, (name, m) in enumerate((("disk-ring", disc_centre_map(x, r, +1)), ("Sobel probe match", sobel_probe_map(x, s["dataProbe"])))):
        det, sc = pick_peaks(m); det = det + c0; k = in_region(det); det, sc = det[k], sc[k]
        row = dict(scores=sc, **match(det, gt), n_vis=int(gt["visible"].sum()), n_all=len(cen))
        thr, rec = operating_point([row], 5, "all"); keep = sc >= thr; st = row["status_all"][keep]; dk = det[keep]
        hit = np.zeros(len(cen), bool)
        for q_ in dk[st == 1]: hit[np.argmin(np.hypot(*(cen - q_).T))] = True
        ax = axes[i, j]; ax.imshow(x, cmap="gray", vmin=0, vmax=1, extent=[c0 - .5, c0 + cw - .5, c0 + cw - .5, c0 - .5])
        ax.scatter(cen[hit, 1], cen[hit, 0], s=30, facecolors="none", edgecolors=C_GT, lw=1.0)
        ax.scatter(cen[~hit, 1], cen[~hit, 0], s=26, facecolors="none", edgecolors=C_FN, lw=0.9)
        ax.scatter(dk[st == 1, 1], dk[st == 1, 0], s=18, c=C_TP, marker="+", lw=1.2)
        ax.scatter(dk[st == 0, 1], dk[st == 0, 0], s=16, c=C_FP, marker="x", lw=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}, {s['thicknesses'][t]:.0f} Å\nR_all = {rec:.2f}  ({int((st == 1).sum())} of {len(cen)}, FP {int((st == 0).sum())})", color=INK)
h = [Line2D([], [], marker="+", ls="", color=C_TP, ms=7, mew=1.3, label="detection on a lattice point"),
     Line2D([], [], marker="x", ls="", color=C_FP, ms=6, mew=1.3, label="false detection"),
     Line2D([], [], marker="o", ls="", mfc="none", mec=C_GT, ms=7, label="lattice point, found"),
     Line2D([], [], marker="o", ls="", mfc="none", mec=C_FN, ms=7, label="lattice point, missed")]
fig.legend(handles=h, loc="outside lower center", ncol=2, frameon=False)
fig.suptitle("ZrSn3, 4 mrad: all lattice reflections, cutoff per image at ≤ 5 false detections (selected with ground truth)",
             fontsize=8, color=INK)
fig.savefig(OUTD / "sobel_vs_discring_ZrSn3_4mrad.png", bbox_inches="tight"); print("saved overlays")
''', "run", "exec"), g)
