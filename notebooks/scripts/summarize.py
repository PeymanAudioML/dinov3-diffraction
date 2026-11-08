# Usage: python summarize.py ROWS.pkl OUT.csv  — per-crystal recall table + console summary of run_all_crystals.py output.
import pickle, sys, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)
rows = pickle.load(open(sys.argv[1], "rb"))
df = pd.DataFrame(rows)

def froc(rs, proto):
    sk, nk = ("status", "n_vis") if proto == "vis" else ("status_all", "n_all")
    sc = np.concatenate([r["scores"] for r in rs]); st = np.concatenate([r[sk] for r in rs])
    k = st >= 0; sc, st = sc[k], st[k]; o = np.argsort(-sc); sc, st = sc[o], st[o]
    return np.cumsum(st == 0) / len(rs), np.cumsum(st == 1) / max(sum(r[nk] for r in rs), 1), sc
def op(rs, fp, proto="vis"):
    f, r, sc = froc(rs, proto); ok = np.where(f <= fp)[0]
    return (sc[ok[-1]], r[ok[-1]]) if len(ok) else (np.inf, 0.0)
def at(rs, thr, proto="vis"):
    sk, nk = ("status", "n_vis") if proto == "vis" else ("status_all", "n_all")
    tp = sum(((r[sk] == 1) & (r["scores"] >= thr)).sum() for r in rs); fp = sum(((r[sk] == 0) & (r["scores"] >= thr)).sum() for r in rs)
    return tp / max(sum(r[nk] for r in rs), 1), fp / len(rs)

short = {"Input true log ε=1e-4 + disc-ring (stage 1)": "true log", "Input adaptive + disc-ring": "adaptive",
         "CHMv2(adaptive) + disc-ring": "CHMv2(adaptive)"}
df["m"] = df.method.map(short)
print(f"{df.crystal.nunique()} crystals, {len(df) // 9} images")

# A) per crystal, thresholds per crystal & angle (optimistic): visible recall at 1 / 5 FP
per = []
for (cr, m, p, mr), g in df.groupby(["crystal", "m", "post", "mrad"], sort=False):
    g = g.to_dict("records")
    per.append(dict(crystal=cr, m=m, post=p, mrad=mr, n_vis=np.mean([r["n_vis"] for r in g]),
                    vis1=op(g, 1)[1], vis5=op(g, 5)[1], all5=op(g, 5, "all")[1]))
per = pd.DataFrame(per)
per.to_csv(sys.argv[2], index=False)
print("\nA) Median over crystals of per-crystal recall (visible @1FP, visible @5FP, all @5FP):")
print(per.groupby(["m", "post", "mrad"], sort=False)[["vis1", "vis5", "all5"]].median().unstack("mrad").round(3).to_string())
print("\nA2) Mean over crystals:")
print(per.groupby(["m", "post", "mrad"], sort=False)[["vis1", "vis5"]].mean().unstack("mrad").round(3).to_string())

# B) single global threshold per method/post/angle, chosen on all crystals pooled at 5 FP/img
print("\nB) One threshold for all crystals (pooled 5 FP/img): pooled recall; % crystals where FP/img > 10")
glob = []
for (m, p, mr), g in df.groupby(["m", "post", "mrad"], sort=False):
    thr, rec = op(g.to_dict("records"), 5)
    fpc = [at(h.to_dict("records"), thr)[1] for _, h in g.groupby("crystal")]
    glob.append(dict(m=m, post=p, mrad=mr, vis_recall=rec, crystals_fp_gt10=np.mean(np.array(fpc) > 10)))
print(pd.DataFrame(glob).set_index(["m", "post", "mrad"]).unstack("mrad").round(2).to_string())

# C) CHMv2 vs adaptive head-to-head per crystal (Friedel + lattice, 5 FP)
fl = per[per.post == "Friedel + lattice"].pivot_table(index=["crystal", "mrad"], columns="m", values="vis5")
print("\nC) Crystals × angles where CHMv2(adaptive) beats the adaptive input (Friedel + lattice, visible @5FP):",
      int((fl["CHMv2(adaptive)"] > fl["adaptive"] + 0.02).sum()), "of", len(fl))
print("   ... and where it is worse by > 0.05:", int((fl["CHMv2(adaptive)"] < fl["adaptive"] - 0.05).sum()))

# D) polarity of CHMv2(adaptive)
pol = df[(df.m == "CHMv2(adaptive)") & (df.post == "none")]
print("\nD) CHMv2(adaptive): fraction of images with disc = bright (no flip):", round((pol.sign > 0).mean(), 3))
print(pol.assign(b=pol.sign > 0).groupby("mrad").b.mean().round(3).to_string())

# E) per-crystal table (adaptive vs CHMv2, Friedel + lattice, visible @5FP)
print("\nE) Per crystal, Friedel + lattice, visible recall @5FP:")
print(fl.unstack("mrad").round(2).to_string())
