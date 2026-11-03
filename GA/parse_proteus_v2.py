"""
Parser de Proteus -> board_data.json
Entradas: netlist SDF, Pick&Place CSV (mm), BOM CSV (opcional)
Salida:   board_data.json  (componentes, redes, contorno) + reporte de validación
Uso:      python parse_proteus.py
"""
import csv, json, re, itertools, math
from pathlib import Path

SDF = "MotorPaso_PIC16_Placed_Despues_IA.SDF"
PNP = "MotorPaso_PIC16_Placed_Despues_IA_Base_Design.csv"
OUT = "board_data.json"

# Tamaño (ancho, alto) en mm a rotación 0, courtyard ESTIMADO (calibrar en ARES).
FOOTPRINT_MM = {
    "CAP20": (7.5, 4.0), "CAPC3225X100": (5.1, 3.4), "CAPC2012X100": (3.5, 2.0),
    "CAPC1608X87": (2.9, 1.6), "CAPMP7343X430": (8.5, 4.9), "CAPMP7343X310": (8.5, 4.9),
    "CAP_BUCK_EEHZK1E471P": (10.5, 11.9), "RES40": (11.5, 2.5), "RESC2012X50": (3.5, 2.0),
    "DIL18": (23.0, 10.0), "SO8": (5.0, 6.2), "XTAL18": (10.5, 5.5),
    "DIOM5336X240": (7.2, 4.4), "SMB": (7.0, 4.2), "LED": (5.0, 5.0),
    "FUSM7243X305": (8.2, 4.7), "IND58X58": (6.0, 6.0), "TEMP_BLK": (7.5, 6.5),
    "TP_CIRCLE_SMD": (2.0, 2.0), "CON6_1X6_HDR_2663": (13.5, 3.0),
    "CON6_2X3_HDR_2664": (7.2, 4.3), "SHDR2W80P0X381_1X2_901X920X740P": (9.0, 9.2),
}
# Componentes que NO se mueven (conectores, agujeros). Ajusta a tu criterio.
FIXED = {"J1", "J2", "M1", "S1", "S2", "S3"}
# Disipación estimada (W) -> luego la afinas con datasheets
POWER_W = {"U3": 0.35, "L1": 0.10, "D4": 0.15, "U2": 0.40, "U1": 0.05, "R5": 0.05}

def parse_sdf(path):
    txt = Path(path).read_text(encoding="latin-1").replace("\r", "")
    # --- partlist
    part_block = txt.split("*PARTLIST")[1].split("*NETLIST")[0].splitlines()[1:]
    parts = {}
    for line in part_block:
        if not line.strip():
            continue
        ref = line.split(",")[0]
        m = re.search(r"GROUP=(\d+)", line)
        pk = re.search(r"PACKAGE=([^,]+)", line)
        parts[ref] = {"group": int(m.group(1)) if m else 0,
                      "package": pk.group(1) if pk else ""}
    # --- netlist
    net_block = txt.split("*NETLIST")[1].splitlines()[1:]
    nets, cur = {}, None
    for line in net_block:
        line = line.strip()
        if not line:
            cur = None
            continue
        f = line.split(",")
        if cur is None:                      # encabezado de red
            cur = f[0]
            nets[cur] = {"pins": [], "class": next((x.split("=")[1] for x in f if x.startswith("CLASS=")), "")}
        elif f[0] in parts:                  # pin real (ignora LBL/GT/PR/PT)
            nets[cur]["pins"].append([f[0], f[2]])
    return parts, nets

def parse_pnp(path):
    comps = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            rot = int(float(r["Rotation"])) % 360
            comps[r["Part ID"]] = {"x": float(r["X"]), "y": float(r["Y"]), "rot": rot,
                                   "layer": r["Layer"], "value": r["Value"], "package": r["Package"]}
    return comps


# ---------------- Geometría de pines (offset desde el centro, rot 0, mm) ----------------
# Medida de capturas de ARES y verificada contra el CSV. Rotación CCW positiva (como Proteus).
def _dil18():
    d = {}
    for n in range(1, 10):  d[str(n)] = (-10.16 + 2.54 * (n - 1), -3.81)
    for n in range(10, 19): d[str(n)] = (10.16 - 2.54 * (n - 10), 3.81)
    return d
def _two(a, b=None):
    b = a if b is None else b
    return {"1": (-a, 0.0), "2": (b, 0.0)}
PINS = {
    "DIL18": _dil18(),
    "SO8": {"1": (-2.635, 1.905), "2": (-2.635, 0.635), "3": (-2.635, -0.635), "4": (-2.635, -1.905),
            "5": (2.635, -1.905), "6": (2.635, -0.635), "7": (2.635, 0.635), "8": (2.635, 1.905)},
    "CON6_1X6_HDR_2663": {str(n): ((n - 3.5) * 2.0, 0.0) for n in range(1, 7)},
    "CON6_2X3_HDR_2664": {"1": (-2, -1), "2": (-2, 1), "3": (0, -1), "4": (0, 1), "5": (2, -1), "6": (2, 1)},
    "CAP20": _two(2.54), "XTAL18": _two(2.54), "RES40": _two(5.08), "TEMP_BLK": _two(2.55),
    "CAPC3225X100": _two(1.5), "CAPC2012X100": _two(0.91), "RESC2012X50": _two(1.0),
    "CAPC1608X87": _two(0.83), "CAPMP7343X430": _two(3.1), "CAPMP7343X310": _two(3.1),
    "CAP_BUCK_EEHZK1E471P": _two(4.33), "FUSM7243X305": _two(3.06),
    "SHDR2W80P0X381_1X2_901X920X740P": _two(1.905),
    "IND58X58": {"1": (0.0, 1.99), "2": (0.0, -1.99)},
    "TP_CIRCLE_SMD": {"1": (0.0, 0.0)},
    "LED": {"K": (1.27, 0.0), "A": (-1.27, 0.0)},
    "DIOM5336X240": {"C": (-2.28, 0.0), "A": (2.28, 0.0)},
    "SMB": {"A": (-2.22, 0.0), "K": (2.22, 0.0)},
}

def rotate(dx, dy, rot_deg):
    t = math.radians(rot_deg)
    return dx * math.cos(t) - dy * math.sin(t), dx * math.sin(t) + dy * math.cos(t)

def pin_xy(comps, ref, pin):
    c = comps[ref]
    dx, dy = PINS[c["package"]][pin]
    rx, ry = rotate(dx, dy, c["rot"])
    return c["x"] + rx, c["y"] + ry

def hpwl_pts(pts):
    xs, ys = zip(*pts)
    return (max(xs) - min(xs)) + (max(ys) - min(ys))

# Mediciones reales de ARES (cobre "Etch", mm). SWITCHING_NODE = 4.965 + 2.995 (dos tramos)
MEASURED = {"AZUL_1": 9.98, "ROSA_2": 13.938, "PGD": 57.523, "SWITCHING_NODE": 4.965 + 2.995}

def calibration(comps, nets):
    print(f"\n{'Red':16s}{'HPWL centros':>14s}{'HPWL pines':>12s}{'ARES (Etch)':>13s}{'pines/ARES':>12s}{'centros/ARES':>14s}")
    for n, meas in MEASURED.items():
        h_c = hpwl_pts([(comps[r]["x"], comps[r]["y"]) for r, _ in nets[n]["pins"]])
        h_p = hpwl_pts([pin_xy(comps, r, p) for r, p in nets[n]["pins"]])
        print(f"{n:16s}{h_c:14.2f}{h_p:12.2f}{meas:13.2f}{h_p/meas:12.2f}{h_c/meas:14.2f}")

def size(c):
    w, h = FOOTPRINT_MM[c["package"]]
    return (h, w) if c["rot"] in (90, 270) else (w, h)

def hpwl(net, pos):
    pts = [pos[r] for r, _ in net["pins"] if r in pos]
    if len(pts) < 2:
        return 0.0
    xs, ys = zip(*pts)
    return (max(xs) - min(xs)) + (max(ys) - min(ys))

def overlaps(comps):
    out = []
    for a, b in itertools.combinations(comps, 2):
        wa, ha = size(comps[a]); wb, hb = size(comps[b])
        dx = abs(comps[a]["x"] - comps[b]["x"]) - (wa + wb) / 2
        dy = abs(comps[a]["y"] - comps[b]["y"]) - (ha + hb) / 2
        if dx < 0 and dy < 0:
            out.append((a, b, round(-dx, 2), round(-dy, 2)))
    return out

if __name__ == "__main__":
    parts, nets = parse_sdf(SDF)
    comps = parse_pnp(PNP)
    for ref, c in comps.items():
        c["group"] = parts[ref]["group"]
        c["w"], c["h"] = FOOTPRINT_MM[c["package"]]
        c["fixed"] = ref in FIXED
        c["power_w"] = POWER_W.get(ref, 0.0)

    pos = {r: (c["x"], c["y"]) for r, c in comps.items()}
    nets_out = {n: v for n, v in nets.items() if len(v["pins"]) >= 2}
    base = {n: hpwl(v, pos) for n, v in nets_out.items() if n != "GND"}  # GND va por plano

    # Contorno: bbox de centros +- mitad de tamaño (cota mínima, NO es el contorno real)
    x0 = min(c["x"] - size(c)[0] / 2 for c in comps.values())
    x1 = max(c["x"] + size(c)[0] / 2 for c in comps.values())
    y0 = min(c["y"] - size(c)[1] / 2 for c in comps.values())
    y1 = max(c["y"] + size(c)[1] / 2 for c in comps.values())

    nets_json = {}
    for n, v in nets_out.items():
        nets_json[n] = {"class": v["class"],
                        "pins": [[r, p, *PINS[comps[r]["package"]][p]] for r, p in v["pins"]]}
    json.dump({"board": {"x0": 0, "y0": 0, "w": 75.0, "h": 50.0, "note": "contorno confirmado con ARES (origen abajo-izq)"},
               "components": comps, "nets": nets_json}, open(OUT, "w"), indent=1)
    calibration(comps, nets_out)

    print(f"Componentes: {len(comps)} | Redes (>=2 pines): {len(nets_out)}")
    print(f"BBox de componentes (mm): x {x0:.1f}..{x1:.1f}  y {y0:.1f}..{y1:.1f}")
    print(f"HPWL total baseline (sin GND): {sum(base.values()):.1f} mm")
    print("Top 8 redes por HPWL:")
    for n, v in sorted(base.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {n:16s} {v:6.1f} mm  ({len(nets_out[n]['pins'])} pines)")
    ov = overlaps(comps)
    print(f"Traslapes con courtyards estimados: {len(ov)}")
    for o in ov:
        print("  ", o)
    missing = set(comps) - set(parts)
    print("Refs en CSV sin partlist:", missing or "ninguno")
