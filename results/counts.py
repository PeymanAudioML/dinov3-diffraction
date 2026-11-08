import pickle, numpy as np, pandas as pd, sys
SP = sys.argv[1]; d = pickle.load(open(f"{SP}/rows.pkl", "rb"))
def froc_thr(rs, sk, nk, fp):
    sc = np.concatenate([r["scores"] for r in rs]); st = np.concatenate([r[sk] for r in rs])
    k = st >= 0; sc, st = sc[k], st[k]; o = np.argsort(-sc); sc, st = sc[o], st[o]
    ok = np.where(np.cumsum(st == 0) / len(rs) <= fp)[0]
    return sc[ok[-1]] if len(ok) else np.inf
def count(rs, proto, fp=None):
    sk, nk = ("status", "n_vis") if proto == "vis" else ("status_all", "n_all")
    thr = -np.inf if fp is None else froc_thr(rs, sk, nk, fp)
    st = np.concatenate([r[sk][r["scores"] >= thr] for r in rs])
    gt = sum(r[nk] for r in rs); tp = int((st == 1).sum()); fpn = int((st == 0).sum())
    return dict(imgs=len(rs), GT=gt, det=len(st), TP=tp, FP=fpn, FN=gt - tp, ign=int((st == -1).sum()),
                recall=round(tp / max(gt, 1), 3), precision=round(tp / max(tp + fpn, 1), 3))
def table(rs, proto, fp, title):
    df = pd.DataFrame(rs); out = []
    for (m, a), g in df.groupby(["method", "mrad"], sort=False):
        out.append(dict(method=m, mrad=a, **count(g.to_dict("records"), proto, fp)))
    print(f"\n### {title}\n" + pd.DataFrame(out).sort_values(["mrad", "method"]).to_string(index=False))
rows = [r for r in d["rows"] if r["refine"] == "map"]
test = {(r["file"], r["t_idx"]) for r in d["dec_rows"]}
held = [r for r in rows if (r["file"], r["t_idx"]) in test] + d["dec_rows"]
for proto in ("all", "vis"):
    table(rows, proto, 5, f"{proto} | all 4 thicknesses | threshold 5 FP/img")
    table(held, proto, 5, f"{proto} | held-out t>=400 | threshold 5 FP/img")
table(rows, "all", None, "all | all 4 thicknesses | NO threshold (every peak)")
