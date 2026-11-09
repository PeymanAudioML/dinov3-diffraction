# Figures for the tutorial section "Disk-ring versus Sobel probe matching" (CPU only).
# Usage: python make_figs_sobel.py figs
# Needs results/per_image_sobel_all.csv from:  scripts/run_per_image.py results/per_image_sobel_all.csv
# Metric: all lattice reflections are targets; cutoff per image at <= 5 false detections, selected with ground truth.
import json, sys
nb = json.load(open("/mnt/data/DINOV3/dinov3/notebooks/dummy_data_explore.ipynb")); c = nb["cells"]
src = lambda i: "".join(c[i]["source"])
s10 = src(10); s10 = s10[:s10.index("tl_in, tl_out, tl_rows")]
s12 = src(12); s12 = s12[:s12.index("loc_rows = []")]
g = {"FIG": sys.argv[1]}
pre = "import matplotlib; matplotlib.use('Agg')\nimport torch\ndisplay = print\nDPT_SIZE = 1024\n"
exec(compile(pre + src(1) + "\n" + src(2) + "\n" + src(4) + "\n" + s10 + "\n" + s12, "setup", "exec"), g)
exec(compile(r'''
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
FIG = Path(FIG); RES = Path("/mnt/data/DINOV3/dinov3/notebooks/results")
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
ANG = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a"}
C_TP, C_FP, C_FN, C_GT = "#1baf7a", "#eb6834", "#eda100", "#2a78d6"
W = 6.5
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 200, "font.family": "DejaVu Sans"})
def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig); print("saved", name, flush=True)

A, B = "true log + disc-ring", "true log + Sobel probe match"
NAME = {A: "disk–ring", B: "Sobel probe match"}
d = pd.read_csv(RES / "per_image_sobel_all.csv"); d = d[d.method.isin([A, B])].copy()
d["fn_all5"] = d.n_all - d.tp_all5; d["eta"] = 2 * d.r / d.spacing

# ── summary table (printed, copied into the LaTeX) ─────────────────────────────────────────────────────────
p = d[d.method == A].merge(d[d.method == B], on=["crystal", "mrad", "t_idx"], suffixes=("_d", "_s")); p["diff"] = p.all5_s - p.all5_d
for mr, q in p.groupby("mrad"):
    print(mr, "lattice", int(q.n_all_d.sum()), "| better/worse/tie", int((q["diff"] > 0).sum()), int((q["diff"] < 0).sum()), int((q["diff"] == 0).sum()))
print(d.groupby(["mrad", "method"]).agg(R=("all5", "median"), TP=("tp_all5", "sum"), FN=("fn_all5", "sum"), FP=("fp_all5", "sum"),
      err=("err_all_med", "median"), p90=("err_all_p90", "median")).round(3).to_string())
p["bin"] = pd.cut(p.eta_d, [0, 0.5, 0.8, 1.0, 1.5, 2.5, 6])
print(p.groupby("bin", observed=True).agg(n=("diff", "size"), Rd=("all5_d", "median"), Rs=("all5_s", "median"),
      better=("diff", lambda v: (v > 0).mean()), worse=("diff", lambda v: (v < 0).mean())).round(3).to_string())
cr = p.groupby(["crystal", "mrad"]).agg(Rd=("all5_d", "median"), Rs=("all5_s", "median"), eta=("eta_d", "first")).reset_index()
cr.round(3).to_csv(RES / "sobel_vs_discring_all_by_crystal.csv", index=False)

# ── Fig S1: summary ────────────────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(W, 2.6), layout="constrained", width_ratios=[1, 1.15, 1.25])
ax = axes[0]
for mr, col in ANG.items():
    u = cr[cr.mrad == mr]; ax.scatter(u.Rd, u.Rs, s=14, color=col, alpha=0.85, lw=0, label=f"{mr} mrad")
ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("disk–ring, median $R_{all}$"); ax.set_ylabel("Sobel, median $R_{all}$")
ax.set_title("(a) per crystal and angle", color=INK); ax.legend(frameon=False, loc="upper left", handletextpad=0.2); tidy(ax)
ax = axes[1]; bins = [0, 0.5, 0.8, 1.0, 1.5, 2.5, 6]; mids = np.arange(len(bins) - 1)
for k, (m, col, mk) in enumerate(((A, INK, "o"), (B, "#eda100", "s"))):
    u = d[d.method == m]; med = u.groupby(pd.cut(u.eta, bins), observed=False).all5.median().values
    ax.plot(mids, med, color=col, marker=mk, ms=3.5, lw=1.5, label=NAME[m])
ax.set_xticks(mids, ["≤0.5", "0.5–0.8", "0.8–1", "1–1.5", "1.5–2.5", ">2.5"], rotation=35)
ax.set_xlabel("overlap ratio η"); ax.set_ylabel("median $R_{all}$"); ax.set_ylim(0, 0.8)
ax.set_title("(b) versus overlap", color=INK); ax.legend(frameon=False); tidy(ax)
ax = axes[2]; x = 0; ticks, labs = [], []
for mr in (1, 2, 4):
    for m in (A, B):
        u = d[(d.method == m) & (d.mrad == mr)]; tp, fn, fp = u.tp_all5.sum(), u.fn_all5.sum(), u.fp_all5.sum()
        ax.bar(x, tp, 0.8, color=C_TP); ax.bar(x, fn, 0.8, bottom=tp, color=C_FN); ax.bar(x, fp, 0.8, bottom=tp + fn, color=C_FP)
        ticks.append(x); labs.append(f"{'DR' if m == A else 'Sob'}\n{mr} mrad" if m == A else "Sob"); x += 1
    x += 0.6
ax.set_xticks(ticks, [f"DR\n{mr}" if i % 2 == 0 else f"Sob\n{mr}" for i, mr in enumerate([1, 1, 2, 2, 4, 4])])
ax.set_xlabel("method and probe angle (mrad)"); ax.set_ylabel("count, all 625 images per angle")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
ax.legend(handles=[Patch(color=C_TP, label="TP"), Patch(color=C_FN, label="FN"), Patch(color=C_FP, label="FP")],
          frameon=False, ncol=3, loc="upper center")
ax.set_ylim(0, 152000)
ax.set_title("(c) TP, FN, FP totals", color=INK); tidy(ax)
save(fig, "sobel_summary.png")

# ── Fig S2…: overlays for every material, disk-ring (top) vs Sobel (bottom), 1/2/4 mrad at 200 A ─────────────
ROOT = Path("/mnt/data/DINOV3/dinov3/dummy_data/selected_folders")
def load_crystal(folder):
    meta = pd.read_pickle(folder / "simulation_dataframe.pkl"); meta.index = meta.index.astype(str)
    ss = [load_sim(q_) for q_ in sorted(folder.glob("*_training.h5"))]
    for s in ss: s["meta"] = meta.loc[s["uuid"]]
    return sorted(ss, key=lambda s: s["meta"]["probeSemiangle"])
mats = [("SiO2", "SiO2", sims)]
for f in sorted(ROOT.iterdir()):
    if f.is_dir():
        ss = load_crystal(f); m0 = ss[0]["meta"]; mats.append((m0["pretty_formula"], f"{m0['pretty_formula']} ({m0['space_group_symbol']})", ss))
mats = [mats[0]] + sorted(mats[1:], key=lambda z: z[0].lower())
T_IDX = 9                                                                      # 200 A
def panel(ax, s, mname, title):
    _, r = probe_geometry(s["dataProbe"]); x = true_log01(s["dataMeas"][..., T_IDX], EPS_LOC); gt = ground_truth(s, T_IDX); cen = gt["centers"]
    m = disc_centre_map(x, r, +1) if mname == A else sobel_probe_map(x, s["dataProbe"])
    det, sc = pick_peaks(m); det = det + c0; k = in_region(det); det, sc = det[k], sc[k]
    row = dict(scores=sc, **match(det, gt), n_vis=int(gt["visible"].sum()), n_all=len(cen))
    thr, rec = operating_point([row], 5, "all"); keep = sc >= thr; st = row["status_all"][keep]; dk = det[keep]
    hit = np.zeros(len(cen), bool)
    for q_ in dk[st == 1]: hit[np.argmin(np.hypot(*(cen - q_).T))] = True
    ax.imshow(x, cmap="gray", vmin=0, vmax=1, extent=[c0 - .5, c0 + cw - .5, c0 + cw - .5, c0 - .5])
    ax.scatter(cen[~hit, 1], cen[~hit, 0], s=16, facecolors="none", edgecolors=C_FN, lw=0.8)
    ax.scatter(dk[st == 1, 1], dk[st == 1, 0], s=14, c=C_TP, marker="+", lw=1.1)
    ax.scatter(dk[st == 0, 1], dk[st == 0, 0], s=13, c=C_FP, marker="x", lw=1.1)
    ax.set_xticks([]); ax.set_yticks([]); [sp.set_color(MUTED) for sp in ax.spines.values()]
    ax.set_title(title, color=INK, fontsize=7.5)
    ax.text(0.02, 0.02, f"TP {int((st == 1).sum())}  FN {int((~hit).sum())}  FP {int((st == 0).sum())}", transform=ax.transAxes,
            color="white", fontsize=6.3, va="bottom", bbox=dict(fc="black", alpha=0.6, lw=0, pad=1.2))
    return rec
LEG = [Line2D([], [], marker="+", ls="", color=C_TP, ms=6, mew=1.2, label="TP: detection on a lattice point"),
       Line2D([], [], marker="o", ls="", mfc="none", mec=C_FN, ms=6, label="FN: lattice point missed"),
       Line2D([], [], marker="x", ls="", color=C_FP, ms=5, mew=1.2, label="FP: false detection")]
for short, full, ss in mats:
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.55), layout="constrained")
    for j, s in enumerate(ss):
        mr = s["meta"]["probeSemiangle"]
        for i, mname in enumerate((A, B)):
            panel(axes[i, j], s, mname, f"{NAME[mname]}, {mr} mrad")
    fig.legend(handles=LEG, loc="outside lower center", ncol=3, frameon=False)
    save(fig, "sobel_overlay_" + short.replace("(", "").replace(")", "") + ".png")
print("materials:", [m[0] for m in mats])
''', "figs", "exec"), g)
