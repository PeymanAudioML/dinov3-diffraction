import json, pickle, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = sys.argv[1]; nb = json.load(open("dummy_data_explore.ipynb")); src = lambda i: "".join(nb["cells"][i]["source"])
g = {"T_IDX": 9}
for i in [1, 3, 5, 6]: exec(src(i), g)
g.update(IMG=256, SIZE=1024, CROP=128, C0=64, backbone=type("B", (), {"patch_size": 16})(),
         crop=lambda img, c=128: img[64:64 + c, 64:64 + c])
exec(src(16), g); g.update(pickle.load(open(f"{SP}/rows.pkl", "rb")))
exec(src(26).replace("plt.show()", f"plt.savefig('{SP}/tp_fp_fn.png', dpi=70, bbox_inches='tight')"), g); print("ok")
