# Figures for the tutorial "Locating Bragg disks ... and why disk overlap makes it hard" (CPU only).
# Usage: python make_figs.py figs
# Needs results/per_image_recall.csv (scripts/run_per_image.py) and results/overlap_{real,synthetic}.csv
# (scripts/run_overlap_analysis.py).
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
from matplotlib.patches import Circle, Wedge
from matplotlib.ticker import NullFormatter, NullLocator
FIG = Path(FIG); RES = Path("/mnt/data/DINOV3/dinov3/notebooks/results")
# one style for every figure: text width 6.5 in, 8-9 pt text, fixed colours per probe angle
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
ANG = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a"}
C_TP, C_FP, C_FN, C_GT = "#1baf7a", "#eb6834", "#eda100", "#2a78d6"
W = 6.5
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 200, "font.family": "DejaVu Sans"})
def tidy(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid: ax.grid(axis=grid, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
def noaxis(ax):
    ax.set_xticks([]); ax.set_yticks([]); [sp.set_color(MUTED) for sp in ax.spines.values()]
def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig); print("saved", name, flush=True)

ROOT = Path("/mnt/data/DINOV3/dinov3/dummy_data/selected_folders")
def load_crystal(folder):
    meta = pd.read_pickle(folder / "simulation_dataframe.pkl"); meta.index = meta.index.astype(str)
    ss = [load_sim(p) for p in sorted(folder.glob("*_training.h5"))]
    for s in ss: s["meta"] = meta.loc[s["uuid"]]
    return sorted(ss, key=lambda s: s["meta"]["probeSemiangle"])
by_name = {pd.read_pickle(f / "simulation_dataframe.pkl")["pretty_formula"].iloc[0]: f for f in ROOT.iterdir() if f.is_dir()}
CRYS = {"SiO2": sims, "ZrSn3": load_crystal(by_name["ZrSn3"])}

PI = pd.read_csv(RES / "per_image_recall.csv"); TL = PI[PI.method == "true log + disc-ring"].copy()
TL["eta"] = 2 * TL.r / TL.spacing
OR = pd.read_csv(RES / "overlap_real.csv").merge(TL[["crystal", "mrad", "t_idx", "spacing", "eta", "vis5", "all5", "err_med"]],
                                                  on=["crystal", "mrad", "t_idx"])
SY = pd.read_csv(RES / "overlap_synthetic.csv")
ETA_C = 1 - 3.0 / 14.0                                         # ring condition for the synthetic lattice (s = 14 px)

def image_result(s, t):
    _, r = probe_geometry(s["dataProbe"]); x = true_log01(s["dataMeas"][..., t], EPS_LOC)
    S = disc_centre_map(x, r, +1); det, sc = pick_peaks(S); det = det + c0; k = in_region(det); det, sc = det[k], sc[k]
    gt = ground_truth(s, t); row = dict(x=x, S=S, det=det, scores=sc, gt=gt, r=r, **match(det, gt),
                                       n_vis=int(gt["visible"].sum()), n_all=len(gt["centers"]))
    row["thr"], row["recall"] = operating_point([row], 5, "vis"); return row

def overlay(ax, row, title):
    x, det, sc, gt, thr = row["x"], row["det"], row["scores"], row["gt"], row["thr"]
    ax.imshow(x, cmap="gray", extent=[c0 - .5, c0 + cw - .5, c0 + cw - .5, c0 - .5], vmin=0, vmax=1)
    keep = sc >= thr; st = row["status"][keep]; d = det[keep]; cen, vis = gt["centers"], gt["visible"]
    hit = np.zeros(len(cen), bool)
    for p in d[st == 1]: hit[np.argmin(np.hypot(*(cen - p).T))] = True
    ax.scatter(cen[~vis, 1], cen[~vis, 0], s=4, c="#9a998f", marker=".", lw=0)
    ax.scatter(cen[vis & hit, 1], cen[vis & hit, 0], s=30, facecolors="none", edgecolors=C_GT, lw=1.0)
    ax.scatter(cen[vis & ~hit, 1], cen[vis & ~hit, 0], s=38, facecolors="none", edgecolors=C_FN, lw=1.3)
    ax.scatter(d[st == 1, 1], d[st == 1, 0], s=16, c=C_TP, marker="+", lw=1.1)
    ax.scatter(d[st == 0, 1], d[st == 0, 0], s=14, c=C_FP, marker="x", lw=1.1)
    ax.set_title(title, color=INK); noaxis(ax)
    ax.text(0.02, 0.02, f"TP {int((st == 1).sum())}  FP {int((st == 0).sum())}  FN {int((vis & ~hit).sum())}",
            transform=ax.transAxes, color="white", fontsize=6.5, va="bottom", bbox=dict(fc="black", alpha=0.55, lw=0, pad=1.2))
LEGEND = [Line2D([], [], marker="+", ls="", color=C_TP, ms=7, mew=1.3, label="detection, correct"),
          Line2D([], [], marker="x", ls="", color=C_FP, ms=6, mew=1.3, label="detection, false"),
          Line2D([], [], marker="o", ls="", mfc="none", mec=C_GT, ms=7, label="visible reflection, found"),
          Line2D([], [], marker="o", ls="", mfc="none", mec=C_FN, ms=7, mew=1.3, label="visible reflection, missed"),
          Line2D([], [], marker=".", ls="", color="#9a998f", ms=5, label="lattice point, not visible")]

# Fig 1 — geometry: the same lattice at three probe angles, with the filter's disk and ring drawn on one reflection
ss = CRYS["ZrSn3"]; fig, axes = plt.subplots(1, 3, figsize=(W, 2.45), layout="constrained")
for ax, s in zip(axes, ss):
    _, r = probe_geometry(s["dataProbe"]); mr = s["meta"]["probeSemiangle"]; x = true_log01(s["dataMeas"][..., 0], EPS_LOC)
    cen = ground_truth(s, 0)["centers"] - c0; ctr = np.array([cw / 2, cw / 2]); h = 34
    ax.imshow(x, cmap="gray", vmin=0, vmax=1); ax.set_xlim(ctr[1] - h, ctr[1] + h); ax.set_ylim(ctr[0] + h, ctr[0] - h)
    d = np.hypot(*(cen - ctr).T); p = cen[np.argmin(np.abs(d - 2 * 13.8))]          # a reflection two spacings out
    ax.add_patch(Wedge((p[1], p[0]), r + 3, 0, 360, width=2, fc="#eb6834", alpha=0.55, ec="none"))
    ax.add_patch(Circle((p[1], p[0]), r, fc="none", ec="#2a78d6", lw=1.6))
    nb_ = cen[(np.hypot(*(cen - p).T) > 1) & (np.hypot(*(cen - p).T) < 2.2 * 13.8)]
    for q in nb_: ax.add_patch(Circle((q[1], q[0]), r, fc="none", ec="white", lw=0.6, ls="--", alpha=0.8))
    eta = TL[(TL.crystal == "ZrSn3 (I4/mmm)") & (TL.mrad == mr)].eta.iloc[0]
    ax.set_title(f"α = {mr} mrad,  r = {r:.1f} px,  η = {eta:.2f}", color=INK); noaxis(ax)
h = [Line2D([], [], color="#2a78d6", lw=1.6, label="filter disk D (radius r)"),
     Line2D([], [], color="#eb6834", lw=5, alpha=0.6, label="filter ring R"),
     Line2D([], [], color=MUTED, lw=0.8, ls="--", label="neighbouring disks")]
fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False)
save(fig, "geometry.png")

# Fig 2 — pipeline on one SiO2 pattern
s = sims[0]; t = 9; row = image_result(s, t); ext = [c0 - .5, c0 + cw - .5, c0 + cw - .5, c0 - .5]
fig, axes = plt.subplots(1, 4, figsize=(W, 1.9), layout="constrained")
for ax, (im, ttl, kw) in zip(axes, [(crop(s["dataMeas"][..., t]), "(a) raw intensity M", dict(cmap="gray")),
                                    (row["x"], "(b) true log T", dict(cmap="gray", vmin=0, vmax=1)),
                                    (row["S"], "(c) filter response S", dict(cmap="magma"))]):
    ax.imshow(im, extent=ext, **kw); ax.set_title(ttl, color=INK); noaxis(ax)
overlay(axes[3], row, "(d) detections")
save(fig, "pipeline.png")

# Fig 3 — the kernel
r = row["r"]; K = disc_ring_kernel(r); R = (K.shape[0] - 1) // 2; v = np.abs(K).max()
fig, axes = plt.subplots(1, 2, figsize=(W * 0.8, 2.2), layout="constrained", width_ratios=[1, 1.4])
im = axes[0].imshow(K, cmap="RdBu_r", vmin=-v, vmax=v, extent=[-R - .5, R + .5, R + .5, -R - .5])
axes[0].set_title(f"(a) K for r = {r:.1f} px", color=INK); axes[0].set_xlabel("px"); axes[0].set_ylabel("px")
fig.colorbar(im, ax=axes[0], shrink=0.85)
axes[1].step(np.arange(-R, R + 1), K[R], where="mid", color=C_GT, lw=1.6); axes[1].axhline(0, color=MUTED, lw=0.7)
axes[1].set_xlabel("offset through the centre (px)"); axes[1].set_ylabel("weight"); axes[1].set_title("(b) profile", color=INK)
tidy(axes[1]); save(fig, "kernel.png")

# Fig 4 — synthetic overlap sweep: examples, recall and centre response versus η
def synth(r, coherent, seed=3, SP=14.0, N=256):
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[:N, :N].astype(float); ctr = np.array([N / 2, N / 2])
    a = 0.3; R2 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    hk = np.array([(h, k) for h in range(-14, 15) for k in range(-14, 15)], float)
    pos = ctr + hk @ R2.T * SP; pos = pos[np.all((pos > -r) & (pos < N + r), 1)]; direct = np.all(np.abs(pos - ctr) < 1e-6, 1)
    inten = np.where(direct, 1.0, 10 ** rng.uniform(-3, -1, len(pos))); ph = rng.uniform(0, 2 * np.pi, len(pos))
    acc = np.zeros((N, N), complex if coherent else float)
    for p, I0, f in zip(pos, inten, ph):
        disk = np.clip(r + 0.5 - np.hypot(yy - p[0], xx - p[1]), 0, 1)
        acc += (np.sqrt(I0) * np.exp(1j * f) if coherent else I0) * disk
    I = np.abs(acc) ** 2 if coherent else acc
    return true_log01(I + 1e-4 * np.exp(-((yy - ctr[0]) ** 2 + (xx - ctr[1]) ** 2) / (2 * 60.0 ** 2)), EPS_LOC)
fig = plt.figure(figsize=(W, 4.3), layout="constrained"); gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05])
for j, (rr, coh) in enumerate([(4, False), (7, False), (10, False), (4, True), (7, True), (10, True)]):
    ax = fig.add_subplot(gs[0, j]); x = synth(rr, coh)
    ax.imshow(x[32:96, 32:96], cmap="gray", vmin=0, vmax=1); noaxis(ax)
    ax.set_title(f"{'coherent' if coh else 'incoherent'}\nη = {2 * rr / 14:.2f}", color=INK, fontsize=7.5)
q = SY.groupby(["model", "eta"]).agg(rec=("recall5", "median"), z=("z_centre", "median"),
                                     rlo=("recall5", lambda v: v.quantile(.25)), rhi=("recall5", lambda v: v.quantile(.75))).reset_index()
MOD = {"incoherent": ("#1f1f1e", "o", "incoherent (intensities add)"), "coherent": ("#eda100", "s", "coherent (amplitudes add)")}
ax1, ax2 = fig.add_subplot(gs[1, :3]), fig.add_subplot(gs[1, 3:])
for mdl, (col, mk, lab) in MOD.items():
    u = q[q.model == mdl]
    ax1.fill_between(u.eta, u.rlo, u.rhi, color=col, alpha=0.12, lw=0)
    ax1.plot(u.eta, u.rec, color=col, marker=mk, ms=3.5, lw=1.5, label=lab)
    ax2.plot(u.eta, u.z, color=col, marker=mk, ms=3.5, lw=1.5, label=lab)
for ax in (ax1, ax2):
    ax.axvline(ETA_C, color=MUTED, lw=0.8, ls="--"); ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("overlap ratio η = 2r / s"); tidy(ax)
ax1.text(ETA_C - 0.03, 0.05, "ring reaches\nneighbours", ha="right", fontsize=6.5, color=MUTED)
ax1.text(1.03, 0.05, "disks touch", ha="left", fontsize=6.5, color=MUTED)
ax1.set_ylim(0, 1.04); ax1.set_ylabel("recall (≤ 5 false positives)"); ax1.set_title("(b) detection", color=INK)
ax2.set_yscale("symlog", linthresh=1); ax2.set_yticks([0, 1, 10], ["0", "1", "10"]); ax2.yaxis.set_minor_locator(NullLocator())
ax2.set_ylabel("median z at true centres"); ax2.set_title("(c) filter response at the centres", color=INK)
ax1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 0.18))
fig.text(0.01, 0.985, "(a)", fontsize=8.5, color=INK, va="top")
save(fig, "synthetic.png")

# Fig 5 — real data: every image of SiO2 + 24 crystals versus its overlap ratio
bins = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.5, 5.5])
def binned(ax, xv, yv, col=INK):
    mid, med = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        k = (xv >= a) & (xv < b) & np.isfinite(yv)
        if k.sum() >= 10: mid.append(np.sqrt(a * b)); med.append(np.median(yv[k]))
    ax.plot(mid, med, color=col, lw=1.8, zorder=5)
fig, axes = plt.subplots(2, 3, figsize=(W, 4.4), layout="constrained")
OR["visfrac"] = OR.n_vis / OR.n_all
panels = [("ring_cover", "(a) ring on neighbouring disks", "fraction of ring area", (0, 1.02), False),
          ("z_centre", "(b) response at true centres", "median z", None, True),
          ("frac_rim", "(c) disks with a clear rim", "fraction of lattice points", (0, 1.02), False),
          ("visfrac", "(d) visible reflections", "fraction of lattice points", (0, 1.02), False),
          ("all5", "(e) lattice reflections found", "recall (≤ 5 false positives)", (0, 1.02), False),
          ("err_med", "(f) localisation error", "median error of correct\ndetections (px)", (0, 1.6), False)]
for ax, (col, ttl, yl, ylim, sl) in zip(axes.flat, panels):
    for mr, cc in ANG.items():
        u = OR[OR.mrad == mr]; ax.scatter(u.eta, u[col], s=3, color=cc, alpha=0.35, lw=0, label=f"{mr} mrad")
    binned(ax, OR.eta.values, OR[col].values)
    ax.set_xscale("log"); ax.set_xticks([0.3, 0.5, 1, 2, 4], ["0.3", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.axvline(1, color=MUTED, lw=0.8, ls=":")
    if sl: ax.set_yscale("symlog", linthresh=1); ax.set_yticks([-1, 0, 1, 10], ["−1", "0", "1", "10"]); ax.yaxis.set_minor_locator(NullLocator())
    if ylim: ax.set_ylim(*ylim)
    ax.set_title(ttl, color=INK); ax.set_ylabel(yl); tidy(ax)
for ax in axes[1]: ax.set_xlabel("overlap ratio η = 2r / s")
h = [Line2D([], [], marker="o", ls="", color=cc, ms=4, label=f"{mr} mrad") for mr, cc in ANG.items()] + \
    [Line2D([], [], color=INK, lw=1.8, label="median in bins of η")]
fig.legend(handles=h, loc="outside lower center", ncol=4, frameon=False)
save(fig, "overlap_real.png")

# Fig 6/7 — overlays: SiO2 (reference) and ZrSn3 (worst case); 3 angles × 3 thicknesses, threshold per image
for cr in ("SiO2", "ZrSn3"):
    fig, axes = plt.subplots(3, 3, figsize=(W, 7.0), layout="constrained")
    for i, s in enumerate(CRYS[cr]):
        mr = s["meta"]["probeSemiangle"]
        for j, t in enumerate([0, 9, 24]):
            rw = image_result(s, t); rc = f"R_vis = {rw['recall']:.2f}" if rw["n_vis"] >= 3 else "not scored"
            overlay(axes[i, j], rw, f"{mr} mrad, {s['thicknesses'][t]:.0f} Å: {rc}")
    fig.legend(handles=LEGEND, loc="outside lower center", ncol=3, frameon=False)
    save(fig, f"overlay_{cr}.png")

# Fig 8 — every image of the 24 crystals, rows sorted by spot spacing (densest lattice at the bottom)
c24 = TL[TL.crystal != "SiO2"]; sp = c24.groupby("crystal").spacing.median().sort_values(ascending=False)
order = sp.index; th = np.sort(c24.thickness.unique())
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(W, 5.0), layout="constrained")
fig.get_layout_engine().set(wspace=0.08)
cm = plt.get_cmap("Blues").copy(); cm.set_bad("#d9d8d3")
for ax, mr in zip(axes, [1, 2, 4]):
    u = c24[c24.mrad == mr]
    v = u.pivot_table(index="crystal", columns="thickness", values="vis5", dropna=False).reindex(index=order, columns=th)
    n = u.pivot_table(index="crystal", columns="thickness", values="n_vis").reindex(index=order, columns=th)
    im = ax.imshow(np.ma.masked_where(~(n.values >= 3), v.values), cmap=cm, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    k = [i for i, t_ in enumerate(th) if t_ in (20, 200, 400)]
    ax.set_xticks(k, [f"{th[i]:.0f}" for i in k]); ax.set_xlabel("thickness (Å)")
    ax.set_title(f"{mr} mrad\n{int((n.values >= 3).sum())} of {n.size} images scored", color=INK, fontsize=7.5)
    ax.tick_params(length=0); [s_.set_visible(False) for s_ in ax.spines.values()]
    et = u.groupby("crystal").eta.first().reindex(order)
    for i, e in enumerate(et.values):
        ax.text(len(th) - 0.3, i, f"{e:.1f}", va="center", ha="left", fontsize=5.5, color=INK if e < 1 else "#c0392b")
labels = [f"{c_.split()[0]}  (s = {sp[c_]:.1f})" for c_ in order]
axes[0].set_yticks(range(len(order)), labels, fontsize=6)
fig.colorbar(im, ax=axes, shrink=0.5, label="visible-reflection recall of the image")
save(fig, "crystals24.png")

# Fig 9 — thickness dependence: median over the crystals, per thickness and angle
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(W * 0.72, 3.4), height_ratios=[3, 1.2], layout="constrained")
for mr, col in ANG.items():
    u = c24[(c24.mrad == mr) & (c24.n_vis >= 3)].groupby("thickness").vis5.quantile([.25, .5, .75]).unstack()
    axes[0].fill_between(u.index, u[.25], u[.75], color=col, alpha=0.13, lw=0)
    axes[0].plot(u.index, u[.5], color=col, lw=1.6, marker="o", ms=2.5, label=f"{mr} mrad")
    nn = c24[c24.mrad == mr].assign(ok=lambda z: z.n_vis >= 3).groupby("thickness").ok.sum()
    axes[1].plot(nn.index, nn.values, color=col, lw=1.6, marker="o", ms=2.5)
axes[0].set_ylim(0, 1.03); axes[0].set_ylabel("visible-reflection recall"); axes[0].legend(frameon=False, ncol=3, loc="lower left")
axes[1].set_ylim(0, 25); axes[1].set_ylabel("crystals\nscored"); axes[1].set_xlabel("thickness (Å)")
for ax in axes: tidy(ax)
save(fig, "thickness24.png")
''', "figs", "exec"), g)
