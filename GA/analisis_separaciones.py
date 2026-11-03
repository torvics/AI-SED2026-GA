import numpy as np, sys
sys.path.insert(0, ".")
from nsga2_placement import *

D = Data("board_data.json"); prob = Placement(D)
X = np.load("out/pareto_X.npy"); F = np.load("out/pareto_F.npy"); vs = np.load("out/chosen_vec.npy")
bv = baseline_vector(D)
o = {}; prob._evaluate(bv[None, :], o); Fb = o["F"][0]

def geom(v):
    x, y, rot, w, h = decode(D, v[None, :])
    return x, y, rot, w, h

def g(v, a, b):
    x, y, rot, w, h = geom(v)
    return float(gap(x, y, w, h, D.idx[a], D.idx[b])[0])

def loop_area(v):
    x, y, *_ = geom(v); L = [D.idx[r] for r in SWITCH_LOOP]
    return float((x[0, L].max() - x[0, L].min()) * (y[0, L].max() - y[0, L].min()))

cands = {
    "ACTUAL (ARES)": bv,
    "ELEGIDA (compromiso)": vs,
    "Solo HPWL (extremo Pareto)": X[np.argmin(F[:, 0])],
    "Solo EMI (extremo Pareto)": X[np.argmin(F[:, 2])],
    "Solo térmico (extremo Pareto)": X[np.argmin(F[:, 1])],
}
pairs_emi = [(a, v) for a in EMI_AGGRESSORS for v in EMI_VICTIMS if a != v]
print("== Índices ==")
print(f"{'':32s}{'HPWL':>8s}{'Térm':>8s}{'EMI':>8s}")
for n, v in cands.items():
    o = {}; prob._evaluate(v[None, :], o); f = o["F"][0]
    print(f"{n:32s}{f[0]:8.0f}{f[1]:8.2f}{f[2]:8.2f}")

print("\n== EMI: separación borde a borde agresor -> víctima (mm) ==")
hdr = "".join(f"{n.split(' ')[0]:>11s}" for n in cands)
print(f"{'par':10s}{hdr}")
for a, v in pairs_emi:
    print(f"{a+'->'+v:10s}" + "".join(f"{g(c,a,v):11.1f}" for c in cands.values()))
print(f"{'Área lazo U3-D4-L1 (mm²)':26s}" + "".join(f"{loop_area(c):11.1f}" for c in cands.values()))

print("\n== TÉRMICO: separación fuente -> víctima (mm) ==")
print(f"{'par':10s}{hdr}")
for v in THERMAL_VICTIMS:
    for s_ in ["U3", "U2", "D4", "L1"]:
        print(f"{s_+'->'+v:10s}" + "".join(f"{g(c,s_,v):11.1f}" for c in cands.values()))
print("calor-calor:")
for a, b in [("U3", "L1"), ("U3", "D4"), ("L1", "D4"), ("U2", "U1")]:
    print(f"{a+'-'+b:10s}" + "".join(f"{g(c,a,b):11.1f}" for c in cands.values()))
