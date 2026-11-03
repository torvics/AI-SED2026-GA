"""
NSGA-II para placement de PCB  (HPWL, Térmico, EMI)  +  restricciones duras
Lee board_data.json (generado por parse_proteus.py).

Uso:   python nsga2_placement.py              (corrida normal)
       python nsga2_placement.py --gens 300 --pop 200 --seed 7

Salidas (carpeta out/):
  pareto.csv, placement_elegido.csv (ref,x,y,rot para aplicar en ARES),
  pareto.png, convergencia.png, layout_antes_despues.png, calor_antes_despues.png
"""
import argparse, json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

# =====================  CONFIGURACIÓN (lo que TÚ decides)  =====================
IGNORE_NETS = {"GND"}                 # van por plano de cobre
CRITICAL_CLASSES = {"POWER_03A"}      # redes de potencia/conmutación -> peso 3
CRITICAL_NETS = {"#00000", "#00008"}  # cristal -> peso 3
W_CRIT, W_NORMAL = 3.0, 1.0

# Desacople: capacitor -> IC que desacopla (distancia corta = mejor)
DECOUPLING = {"C10": ("U1", "14"), "C6": ("U3", "6"), "C11": ("U2", "10")}  # cap -> (IC, pin de alimentación)
W_DECOUP = 2.0

# Térmico: víctima(s) y peso. Fuentes de calor salen de power_w en board_data.json
THERMAL_VICTIMS = {"X1": 1.0, "U1": 0.5}
D0_THERMAL = 3.0                      # mm, suavizado 1/(d+d0)

# EMI: agresores y víctimas (peso de cada uno)
EMI_AGGRESSORS = {"U3": 1.0, "L1": 1.0, "D4": 0.7, "U2": 0.5}
EMI_VICTIMS = {"X1": 1.0, "C1": 0.5, "C2": 0.5, "U1": 0.5}
D0_EMI = 3.0
SWITCH_LOOP = ["U3", "D4", "L1"]      # componentes del lazo caliente del buck
W_LOOP = 0.05                         # peso del área del lazo (mm²)

# Mecánico
HOLES = [(5, 5), (5, 45), (70, 5), (70, 45)]  # agujeros de montaje (estimado)
HOLE_KEEPOUT = 4.0                    # mm de radio alrededor del centro
GRID_TRY = [0.5, 0.25, 0.1]           # rejilla final de exportación (mm)
# ================================================================================


class Data:
    def __init__(self, path):
        d = json.load(open(path))
        # Aviso claro si board_data.json viene de una versión vieja de parse_proteus.py
        _p = next(iter(d["nets"].values()))["pins"][0]
        if len(_p) != 4:
            raise SystemExit("board_data.json tiene pines con 2 valores [ref, pin]; se esperan 4 [ref, pin, dx, dy].\n"
                             "Estás usando una versión vieja de parse_proteus.py: reemplázala por la nueva y vuelve a ejecutarla.")
        self.board = d["board"]
        self.W, self.H = self.board["w"], self.board["h"]
        self.refs = list(d["components"].keys())
        self.N = len(self.refs)
        self.idx = {r: i for i, r in enumerate(self.refs)}
        C = d["components"]
        self.x0 = np.array([C[r]["x"] for r in self.refs])
        self.y0 = np.array([C[r]["y"] for r in self.refs])
        self.rot0 = np.array([C[r]["rot"] for r in self.refs])
        self.w = np.array([C[r]["w"] for r in self.refs])
        self.h = np.array([C[r]["h"] for r in self.refs])
        self.fixed = np.array([C[r]["fixed"] for r in self.refs])
        self.power = np.array([C[r]["power_w"] for r in self.refs])
        self.mov = np.where(~self.fixed)[0]
        self.M = len(self.mov)
        # redes (cada pin con su offset a rot 0) y tabla global de offsets
        self.pin_off = {}
        self.nets = []
        for name, v in d["nets"].items():
            for r, p, dx, dy in v["pins"]:
                self.pin_off[(r, p)] = (dx, dy)
            if name in IGNORE_NETS:
                continue
            pins = [(self.idx[r], dx, dy) for r, p, dx, dy in v["pins"] if r in self.idx]
            if len(pins) < 2:
                continue
            wgt = W_CRIT if (v["class"] in CRITICAL_CLASSES or name in CRITICAL_NETS) else W_NORMAL
            ids = np.array([q[0] for q in pins]); dxs = np.array([q[1] for q in pins]); dys = np.array([q[2] for q in pins])
            self.nets.append((name, ids, dxs, dys, wgt))
        self.heat = np.where(self.power > 0)[0]
        # pares para traslape (se omiten fijo-fijo)
        I, J = np.triu_indices(self.N, 1)
        keep = ~(self.fixed[I] & self.fixed[J])
        self.pI, self.pJ = I[keep], J[keep]


def decode(D, X):
    """X: (P, 3M) -> pos x,y (P,N), rot (P,N), w,h efectivos (P,N)."""
    P = X.shape[0]
    x = np.tile(D.x0, (P, 1)); y = np.tile(D.y0, (P, 1)); rot = np.tile(D.rot0, (P, 1)).astype(float)
    m = D.mov
    x[:, m] = X[:, 0:D.M]
    y[:, m] = X[:, D.M:2 * D.M]
    rot[:, m] = (np.floor(X[:, 2 * D.M:3 * D.M]).clip(0, 3)) * 90
    swap = (np.round(rot / 90).astype(int) % 2) == 1
    w = np.where(swap, D.h, D.w); h = np.where(swap, D.w, D.h)
    return x, y, rot, w, h


def gap(x, y, w, h, i, j):
    gx = np.maximum(0, np.abs(x[:, i] - x[:, j]) - (w[:, i] + w[:, j]) / 2)
    gy = np.maximum(0, np.abs(y[:, i] - y[:, j]) - (h[:, i] + h[:, j]) / 2)
    return np.hypot(gx, gy)


def pin_xy(D, x, y, rot, ids, dxs, dys):
    th = np.deg2rad(rot[:, ids])
    return (x[:, ids] + dxs * np.cos(th) - dys * np.sin(th),
            y[:, ids] + dxs * np.sin(th) + dys * np.cos(th))


def objectives(D, x, y, rot, w, h):
    P = x.shape[0]
    # --- 1) HPWL ponderado con posición REAL de pines + desacople al pin de alimentación
    f1 = np.zeros(P)
    for _, ids, dxs, dys, wgt in D.nets:
        px, py = pin_xy(D, x, y, rot, ids, dxs, dys)
        f1 += wgt * ((px.max(1) - px.min(1)) + (py.max(1) - py.min(1)))
    for cap, (ic, pin) in DECOUPLING.items():
        ii = np.array([D.idx[ic]]); dx, dy = D.pin_off[(ic, pin)]
        px, py = pin_xy(D, x, y, rot, ii, np.array([dx]), np.array([dy]))
        f1 += W_DECOUP * np.hypot(x[:, D.idx[cap]] - px[:, 0], y[:, D.idx[cap]] - py[:, 0])
    # --- 2) Térmico: fuente->víctima + calor-calor
    f2 = np.zeros(P)
    for v, wv in THERMAL_VICTIMS.items():
        iv = D.idx[v]
        for s in D.heat:
            if s != iv:
                f2 += wv * D.power[s] / (gap(x, y, w, h, s, iv) + D0_THERMAL)
    for a in range(len(D.heat)):
        for b in range(a + 1, len(D.heat)):
            i, j = D.heat[a], D.heat[b]
            f2 += D.power[i] * D.power[j] / (gap(x, y, w, h, i, j) + D0_THERMAL)
    f2 *= 100  # escala legible
    # --- 3) EMI: agresor->víctima + área del lazo del buck
    f3 = np.zeros(P)
    for a, wa in EMI_AGGRESSORS.items():
        for v, wv in EMI_VICTIMS.items():
            f3 += wa * wv / (gap(x, y, w, h, D.idx[a], D.idx[v]) + D0_EMI)
    L = [D.idx[r] for r in SWITCH_LOOP]
    f3 += W_LOOP * (x[:, L].max(1) - x[:, L].min(1)) * (y[:, L].max(1) - y[:, L].min(1))
    return np.column_stack([f1, f2, f3])


def constraints(D, x, y, w, h):
    """Cada columna debe ser <= 0. Devuelve violaciones en mm."""
    # traslape (penetración mínima cuando se solapan en x e y)
    I, J = D.pI, D.pJ
    dx = np.abs(x[:, I] - x[:, J]) - (w[:, I] + w[:, J]) / 2
    dy = np.abs(y[:, I] - y[:, J]) - (h[:, I] + h[:, J]) / 2
    ov = np.where((dx < 0) & (dy < 0), np.minimum(-dx, -dy), 0).sum(1)
    # fuera de placa
    out = (np.maximum(0, w / 2 - x) + np.maximum(0, x + w / 2 - D.W) +
           np.maximum(0, h / 2 - y) + np.maximum(0, y + h / 2 - D.H)).sum(1)
    # agujeros de montaje (distancia del centro del hueco al rectángulo)
    hole = np.zeros(x.shape[0])
    for hx, hy in HOLES:
        ddx = np.maximum(0, np.abs(x - hx) - w / 2)
        ddy = np.maximum(0, np.abs(y - hy) - h / 2)
        hole += np.maximum(0, HOLE_KEEPOUT - np.hypot(ddx, ddy)).sum(1)
    return np.column_stack([ov, out, hole])


class Placement(Problem):
    def __init__(self, D):
        self.D = D
        M = D.M
        xl = np.r_[np.full(M, 0.0), np.full(M, 0.0), np.zeros(M)]
        xu = np.r_[np.full(M, D.W), np.full(M, D.H), np.full(M, 3.999)]
        super().__init__(n_var=3 * M, n_obj=3, n_ieq_constr=3, xl=xl, xu=xu)

    def _evaluate(self, X, out, *a, **k):
        x, y, rot, w, h = decode(self.D, X)
        out["F"] = objectives(self.D, x, y, rot, w, h)
        out["G"] = constraints(self.D, x, y, w, h)


def baseline_vector(D):
    rot_idx = (np.round(D.rot0[D.mov] / 90).astype(int) % 4).astype(float) + 0.5
    return np.r_[D.x0[D.mov], D.y0[D.mov], rot_idx]


def initial_population(D, prob, n, rng):
    base = baseline_vector(D)
    pop = [base]
    for k in range(n - 1):
        if k < int(0.4 * n):            # variaciones del placement actual
            s = rng.uniform(1.0, 6.0)
            v = base.copy()
            v[:D.M] += rng.normal(0, s, D.M); v[D.M:2 * D.M] += rng.normal(0, s, D.M)
            flip = rng.random(D.M) < 0.1
            v[2 * D.M:][flip] = rng.integers(0, 4, flip.sum()) + 0.5
        else:                           # aleatorios
            v = rng.uniform(prob.xl, prob.xu)
        pop.append(np.clip(v, prob.xl, prob.xu))
    return np.array(pop)


class Track(Callback):
    def __init__(self):
        super().__init__()
        self.hist = []

    def notify(self, algo):
        F, CV = algo.pop.get("F"), algo.pop.get("CV")[:, 0]
        ok = CV <= 1e-9
        self.hist.append([F[ok, 0].min() if ok.any() else np.nan,
                          F[ok, 1].min() if ok.any() else np.nan,
                          F[ok, 2].min() if ok.any() else np.nan, ok.sum()])


def snap_solution(D, prob, x_vec):
    """Redondea a rejilla; usa la más gruesa que siga siendo factible."""
    for g in GRID_TRY:
        v = x_vec.copy()
        v[:2 * D.M] = np.round(v[:2 * D.M] / g) * g
        out = {}
        prob._evaluate(v[None, :], out)
        if (out["G"] <= 1e-9).all():
            return v, g, out["F"][0]
    out = {}; prob._evaluate(x_vec[None, :], out)
    return x_vec, 0.0, out["F"][0]


# ---------------------------  térmico: mapa de calor  ---------------------------
def heat_map(D, x, y, w, h, kt=0.01, h_conv=2 * 10.0, cell=1.0):
    """Régimen permanente 2D: -kt∇²T + h·T = q. Valores RELATIVOS (cualitativos)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    nx, ny = int(D.W / cell), int(D.H / cell)
    q = np.zeros((ny, nx))
    for i in D.heat:
        xs = slice(max(0, int((x[i] - w[i] / 2) / cell)), min(nx, int(np.ceil((x[i] + w[i] / 2) / cell))))
        ys = slice(max(0, int((y[i] - h[i] / 2) / cell)), min(ny, int(np.ceil((y[i] + h[i] / 2) / cell))))
        n = max(1, (xs.stop - xs.start) * (ys.stop - ys.start))
        q[ys, xs] += D.power[i] / n
    g = kt                                   # conductancia entre celdas vecinas (W/K)
    gc = h_conv * (cell * 1e-3) ** 2         # pérdida por convección por celda (W/K)
    idx = lambda r, c: r * nx + c
    A = sp.lil_matrix((nx * ny, nx * ny))
    for r in range(ny):
        for c in range(nx):
            k = idx(r, c); A[k, k] += gc
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < ny and 0 <= cc < nx:
                    A[k, k] += g; A[k, idx(rr, cc)] -= g
    T = spsolve(A.tocsr(), q.ravel()).reshape(ny, nx)
    return T


def draw_layout(ax, D, x, y, w, h, title):
    ax.add_patch(Rectangle((0, 0), D.W, D.H, fill=False, ec="k", lw=1.5))
    for hx, hy in HOLES:
        ax.add_patch(Circle((hx, hy), 1.6, fc="0.7", ec="k"))
    for i, r in enumerate(D.refs):
        col = "#d9534f" if D.power[i] > 0 else ("#5b9bd5" if D.fixed[i] else "#9fd89f")
        ax.add_patch(Rectangle((x[i] - w[i] / 2, y[i] - h[i] / 2), w[i], h[i], fc=col, ec="k", lw=0.5, alpha=0.85))
        ax.text(x[i], y[i], r, ha="center", va="center", fontsize=5)
    ax.set_xlim(-2, D.W + 2); ax.set_ylim(-2, D.H + 2); ax.set_aspect("equal"); ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="board_data.json")
    ap.add_argument("--pop", type=int, default=150)
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    Path(a.out).mkdir(exist_ok=True)

    D = Data(a.data)
    prob = Placement(D)
    rng = np.random.default_rng(a.seed)

    # ---- baseline (sin redondear, tal como está en ARES)
    bv = baseline_vector(D)
    o = {}; prob._evaluate(bv[None, :], o)
    F_base, G_base = o["F"][0], o["G"][0]
    print(f"Baseline  HPWL={F_base[0]:.1f}  Térmico={F_base[1]:.3f}  EMI={F_base[2]:.3f}")
    print(f"Baseline violaciones (traslape, borde, agujeros): {np.round(G_base, 2)}  (0 = factible)")

    algo = NSGA2(pop_size=a.pop, sampling=initial_population(D, prob, a.pop, rng),
                 crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20), eliminate_duplicates=True)
    tr = Track()
    res = minimize(prob, algo, ("n_gen", a.gens), seed=a.seed, callback=tr, verbose=False)

    F, X = res.F, res.X
    if F is None or len(F) == 0:
        print("Sin soluciones factibles: relaja restricciones o aumenta gens/pop."); return
    print(f"Frente de Pareto factible: {len(F)} soluciones")

    # ---- elegir compromiso: mejor suma de razones vs baseline que sobreviva al redondeo a rejilla
    ratio = F / F_base
    order = np.argsort(ratio.sum(1))
    k = None
    for cand in order:
        vs, grid, Fs = snap_solution(D, prob, X[cand])
        if grid > 0:
            k = int(cand); break
    if k is None:
        k = int(order[0]); vs, grid, Fs = snap_solution(D, prob, X[k])
    print(f"Elegida #{k} (rejilla {grid} mm): HPWL={Fs[0]:.1f} ({Fs[0]/F_base[0]-1:+.1%})  "
          f"Térmico={Fs[1]:.3f} ({Fs[1]/F_base[1]-1:+.1%})  EMI={Fs[2]:.3f} ({Fs[2]/F_base[2]-1:+.1%})")

    np.save(f"{a.out}/pareto_X.npy", X); np.save(f"{a.out}/pareto_F.npy", F); np.save(f"{a.out}/chosen_vec.npy", vs)
    # ---- exportar
    x, y, rot, w, h = decode(D, vs[None, :])
    with open(f"{a.out}/placement_elegido.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["Ref", "X_mm", "Y_mm", "Rot_deg", "Movido"])
        for i, r in enumerate(D.refs):
            wr.writerow([r, round(x[0, i], 3), round(y[0, i], 3), int(rot[0, i]) % 360, "no" if D.fixed[i] else "si"])
    with open(f"{a.out}/pareto.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["HPWL", "Termico", "EMI", "HPWL_vs_base", "Term_vs_base", "EMI_vs_base"])
        for f, r_ in zip(F, ratio):
            wr.writerow(list(np.round(f, 4)) + list(np.round(r_, 4)))

    # ---- figuras
    H_ = np.array(tr.hist)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for j, n in enumerate(["HPWL", "Térmico", "EMI"]):
        ax[0].plot(H_[:, j] / F_base[j], label=n)
    ax[0].axhline(1, c="k", ls="--", lw=0.8); ax[0].set_xlabel("Generación"); ax[0].set_ylabel("Mejor factible / baseline")
    ax[0].legend(); ax[0].set_title("Convergencia (<1 = mejor que el diseño actual)")
    ax[1].plot(H_[:, 3]); ax[1].set_xlabel("Generación"); ax[1].set_ylabel("Individuos factibles"); ax[1].set_title("Factibilidad de la población")
    plt.tight_layout(); plt.savefig(f"{a.out}/convergencia.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    pairs = [(0, 1), (0, 2), (1, 2)]; nm = ["HPWL (mm)", "Térmico", "EMI"]
    for axx, (i, j) in zip(ax, pairs):
        sc = axx.scatter(F[:, i], F[:, j], c=F[:, 3 - i - j], cmap="viridis", s=14)
        axx.scatter(*F_base[[i, j]], c="r", marker="*", s=180, label="Actual", zorder=5)
        axx.scatter(*Fs[[i, j]], c="k", marker="D", s=50, label="Elegida", zorder=5)
        axx.set_xlabel(nm[i]); axx.set_ylabel(nm[j]); plt.colorbar(sc, ax=axx, label=nm[3 - i - j])
    ax[0].legend(); plt.suptitle("Frente de Pareto (soluciones factibles)")
    plt.tight_layout(); plt.savefig(f"{a.out}/pareto.png", dpi=150); plt.close()

    xb, yb, rb, wb, hb = decode(D, bv[None, :])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    draw_layout(ax[0], D, xb[0], yb[0], wb[0], hb[0], f"ANTES  HPWL={F_base[0]:.0f}")
    draw_layout(ax[1], D, x[0], y[0], w[0], h[0], f"DESPUÉS  HPWL={Fs[0]:.0f}")
    plt.tight_layout(); plt.savefig(f"{a.out}/layout_antes_despues.png", dpi=150); plt.close()

    Tb = heat_map(D, xb[0], yb[0], wb[0], hb[0]); Ta = heat_map(D, x[0], y[0], w[0], h[0])
    vmax = max(Tb.max(), Ta.max())
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for axx, T, t in zip(ax, [Tb, Ta], ["ANTES", "DESPUÉS"]):
        im = axx.imshow(T, origin="lower", extent=[0, D.W, 0, D.H], cmap="inferno", vmin=0, vmax=vmax)
        for v, c in [("X1", "c"), ("U1", "w")]:
            i = D.idx[v]; xx, yy = (xb, yb) if t == "ANTES" else (x, y)
            axx.plot(xx[0, i], yy[0, i], "o", c=c, ms=5)
            axx.text(xx[0, i] + 1, yy[0, i] + 1, v, color=c, fontsize=8)
        axx.set_title(f"{t}: ΔT relativo (víctima X1 = {T[int(y[0, D.idx['X1']] if t != 'ANTES' else yb[0, D.idx['X1']]), int(x[0, D.idx['X1']] if t != 'ANTES' else xb[0, D.idx['X1']])]:.3f})", fontsize=9)
        plt.colorbar(im, ax=axx, label="ΔT relativo (cualitativo)")
    plt.tight_layout(); plt.savefig(f"{a.out}/calor_antes_despues.png", dpi=150); plt.close()
    print(f"Listo. Archivos en {a.out}/")


if __name__ == "__main__":
    main()
