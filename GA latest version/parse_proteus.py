"""
Parser de Proteus -> board_data.json   (placa PROYECTOAVANCE: monitoreo de vibración)
Entradas: netlist SDF + Pick&Place CSV (mm)
Salida:   board_data.json + reporte de validación
Uso:      python parse_proteus.py

Convención: x,y en board_data.json = CENTRO DEL COURTYARD (no el origen del footprint).
Para casi todos coinciden; el ESP32-S3-WROOM-1 tiene su origen desplazado respecto al
centro del cuerpo (la antena sobresale), por eso se guarda ox,oy y el optimizador
convierte de vuelta al origen al exportar para ARES.
"""
import csv, json, re, itertools, math, sys
from pathlib import Path

# Uso: python parse_proteus.py [archivo.SDF] [archivo_PickAndPlace.csv]
SDF = sys.argv[1] if len(sys.argv) > 1 else "PROYECTOAVANCEFUSHON.SDF"
PNP = sys.argv[2] if len(sys.argv) > 2 else "PROYECTOAVANCEFUSHON_Base_Design.csv"
OUT = "board_data.json"

# Borde = línea amarilla punteada de ARES, origen en la esquina inferior izquierda.
# 108.7 x 76.0 sale de medir la captura (confirmar leyendo la esquina superior derecha en ARES).
BOARD_W, BOARD_H = 108.7, 76.0
# Agujeros de montaje: los de antes (5,5)...(95,65) + el corrimiento de origen (+4.5062, +4.5)
HOLES = [[9.51, 9.5], [9.51, 69.5], [99.51, 9.5], [99.51, 69.5]]
HOLE_KEEPOUT = 4.0                                  # mm, radio libre alrededor del centro
EDGE_MARGIN = 5.0                                   # mm, componentes MÓVILES a >= 5 mm del borde
MIN_GAP = 1.0                                       # mm, separación mínima entre courtyards

# Fijos = conectores, interruptor, botones, USB-C (se queda donde está) y el ESP32
FIXED = {"J2", "J3", "J4", "J5", "U4", "U5", "U6", "U7"}

# Disipación estimada (W), calculada con datasheets (ver notas): sin WiFi, sin R2.
POWER_W = {"U5": 0.17, "U3": 0.08, "D4": 0.03, "L1": 0.005, "U15": 0.03}

NET_IGNORE_NAMES = {"{NC}"}                         # pseudo-red de pines sin conexión


# ----------------------------------------------------------------------------------
# Generadores de geometría de pines (offset desde el ORIGEN del footprint, rot 0, mm)
# ----------------------------------------------------------------------------------
def two(a, b=None):
    b = a if b is None else b
    return {"1": (-a, 0.0), "2": (b, 0.0)}

def soic8(xs=2.7):
    return {"1": (-xs, 1.905), "2": (-xs, 0.635), "3": (-xs, -0.635), "4": (-xs, -1.905),
            "5": (xs, -1.905), "6": (xs, -0.635), "7": (xs, 0.635), "8": (xs, 1.905)}

def sot23_6():
    return {"1": (-0.95, -1.3), "2": (0.0, -1.3), "3": (0.95, -1.3),
            "4": (0.95, 1.3), "5": (0.0, 1.3), "6": (-0.95, 1.3)}

def vson10():
    d = {}
    for i, y in enumerate([1.0, 0.5, 0.0, -0.5, -1.0]):
        d[str(i + 1)] = (-1.4, y)
        d[str(10 - i)] = (1.4, y)
    return d

def lga16():
    # Datasheet IIS3DWB10IS: 4 pines por lado, paso 0.8 mm, pads 0.4x0.6. Pin 1 arriba-izq, sentido
    # antihorario (orden ESTIMADO; la posición de los pads sí es del datasheet)
    d = {}
    for i in range(4):
        t = 1.2 - 0.8 * i
        d[str(1 + i)] = (-1.85, t)            # izquierda, de arriba a abajo
        d[str(5 + i)] = (-t, -1.85)           # abajo, de izq a der
        d[str(9 + i)] = (1.85, -t)            # derecha, de abajo a arriba
        d[str(13 + i)] = (t, 1.85)            # arriba, de der a izq
    return d

def gct_usb4105():
    """GCT USB4105-GF-A (dibujo del fabricante): pads a +-3.2 (GND), +-2.4 (VBUS), +-1.75 (CC),
    +-0.25 (D+/D-); pestañas del blindaje a +-4.3 x +-2.05. Posición en Y de las filas: ESTIMADA."""
    d = {"A1_B1": (-3.2, 0.7), "B1_A1": (3.2, 0.7), "A4_B9": (-2.4, 0.7), "B4_A9": (2.4, 0.7),
         "A5": (-1.75, 0.7), "B5": (1.75, -0.7), "A6": (-0.25, 0.7), "A7": (0.25, 0.7),
         "B7": (-0.25, -0.7), "B6": (0.25, -0.7), "A8": (1.25, 0.7), "B8": (-1.25, -0.7),
         "SH1": (-4.3, 2.05), "SH2": (4.3, 2.05), "SH3": (-4.3, -2.05), "SH4": (4.3, -2.05)}
    return d

def esp32s3_wroom1():
    """Pines relativos al ORIGEN del footprint (rot 0). Medidos de la captura de ARES (rot 90):
       pines 1-14 fila inferior, 15-26 columna derecha, 27-40 fila superior, antena a la izquierda.
       ESTIMADO (+-0.5 mm): confirmar pad 1 y pad 14 en ARES."""
    d = {}
    for n in range(1, 15):   d[str(n)] = (-8.75, 9.06 - 1.27 * (n - 1))
    for n in range(15, 27):  d[str(n)] = (-6.985 + 1.27 * (n - 15), -8.61)
    for n in range(27, 41):  d[str(n)] = (8.75, -7.45 + 1.27 * (n - 27))
    for k in range(9):       d[f"41_{k + 1}"] = (-1.43 + ((k % 3) - 1) * 1.3, 1.35 + ((k // 3) - 1) * 1.3)
    return d

# paquete -> (w, h, pines, (ox, oy) offset del centro del courtyard respecto al origen)
FP = {
    # conectores / mecánicos (tamaños medidos de las capturas de ARES)
    "TE_15460742":      (10.2, 12.5, {"1": (-2.54, 0), "2": (2.54, 0)}, (0, 0)),
    "TE_15460743":      (15.2, 12.6, {"1": (-5.08, 0), "2": (0, 0), "3": (5.08, 0)}, (0, 0)),
    "HARWIN_M209990245": (5.2, 3.2, {"1": (-1.27, 0), "2": (1.27, 0)}, (0, 0)),
    "SW_EG1218":        (11.7, 4.1, {"1": (-2.54, 0), "2": (0, 0), "3": (2.54, 0)}, (0, 0)),
    "SW_1478731":       (6.4, 3.6, two(2.4), (0, 0)),
    "GCT_USB4105GFA":   (8.94, 7.35, gct_usb4105(), (0, 0)),
    "XCVR_ESP32S3WROOM1N16R2": (18.0, 25.5, esp32s3_wroom1(), (0.0, 3.75)),
    # pasivos
    "RES40":            (12.2, 2.5, two(5.08), (0, 0)),
    "RESC2012X50":      (3.5, 2.0, two(1.0), (0, 0)),
    "CAPC3225X100":     (5.1, 3.4, two(1.5), (0, 0)),
    "CAPC3216X140":     (4.4, 2.2, two(1.5), (0, 0)),
    "CAPC2012X100":     (3.5, 2.0, two(0.91), (0, 0)),
    "0603_CAP":         (2.6, 1.3, two(0.8), (0, 0)),
    "CAP10":            (5.0, 2.8, two(1.27), (0, 0)),
    "CAPMP7343X430":    (8.5, 4.9, two(3.1), (0, 0)),
    "CAPMP7343X310":    (8.5, 4.9, two(3.1), (0, 0)),
    "SRR1260-150M":     (12.7, 12.5, {"1": (-4.85, 0), "2": (4.85, 0)}, (0, 0)),
    # diodos / LEDs
    "SMB":              (7.0, 4.2, {"A": (-2.22, 0), "K": (2.22, 0)}, (0, 0)),
    "DIOM5436X250N":    (6.6, 4.0, two(2.28), (0, 0)),
    "DIOM4325X250N":    (5.8, 3.0, two(2.0), (0, 0)),
    "DIODE30":          (10.2, 2.6, {"A": (-3.81, 0), "K": (3.81, 0)}, (0, 0)),
    "DIOC200X125X110":  (3.4, 1.8, {"+": (-1.05, 0), "-": (1.05, 0)}, (0, 0)),
    # circuitos integrados
    "SO8":              (5.0, 6.2, {"1": (-2.635, 1.905), "2": (-2.635, 0.635), "3": (-2.635, -0.635),
                                    "4": (-2.635, -1.905), "5": (2.635, -1.905), "6": (2.635, -0.635),
                                    "7": (2.635, 0.635), "8": (2.635, 1.905)}, (0, 0)),
    "SOIC127P600X1758N": (7.0, 5.0, soic8(2.7), (0, 0)),
    "SOT95P280X1456N":  (3.0, 3.7, sot23_6(), (0, 0)),
    "THVD2450VDRCR":    (4.5, 4.5, vson10(), (0, 0)),
    "IIS3DWB10ISTR":    (4.5, 4.5, lga16(), (0, 0)),
}


def parse_sdf(path):
    txt = Path(path).read_text(encoding="latin-1").replace("\r", "")
    part_block = txt.split("*PARTLIST")[1].split("*NETLIST")[0].splitlines()[1:]
    parts = {}
    for line in part_block:
        if not line.strip():
            continue
        ref = line.split(",")[0]
        m = re.search(r"GROUP=(\d+)", line)
        pk = re.search(r"PACKAGE=([^,]+)", line)
        parts[ref] = {"group": int(m.group(1)) if m else 0, "package": pk.group(1) if pk else ""}
    net_block = txt.split("*NETLIST")[1].splitlines()[1:]
    nets, cur = {}, None
    for line in net_block:
        line = line.strip()
        if not line:
            cur = None
            continue
        f = line.split(",")
        if cur is None:                                   # encabezado de red
            cur = f[0]
            nets[cur] = {"pins": [], "class": next((x.split("=")[1] for x in f if x.startswith("CLASS=")), "")}
        elif f[0] in parts:                               # pin real (ignora LBL/GT/PR/PT)
            nets[cur]["pins"].append([f[0], f[2]])
    return parts, nets


def parse_pnp(path):
    comps = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if not r["Part ID"].strip():
                continue                                  # fila fantasma sin designador (duplicado de otro componente)
            rot = int(round(float(r["Rotation"]))) % 360
            comps[r["Part ID"]] = {"x_org": float(r["X"]), "y_org": float(r["Y"]), "rot": rot,
                                   "layer": r["Layer"], "value": r["Value"], "package": r["Package"]}
    return comps


def rotate(dx, dy, rot_deg):
    t = math.radians(rot_deg)
    return dx * math.cos(t) - dy * math.sin(t), dx * math.sin(t) + dy * math.cos(t)

def size(c):
    return (c["h"], c["w"]) if c["rot"] in (90, 270) else (c["w"], c["h"])

def hpwl_pts(pts):
    xs, ys = zip(*pts)
    return (max(xs) - min(xs)) + (max(ys) - min(ys))

def pin_xy(comps, ref, pin):
    c = comps[ref]
    dx, dy = c["pins"][pin]
    rx, ry = rotate(dx, dy, c["rot"])
    return c["x"] + rx, c["y"] + ry

def overlaps(comps, gap_min=None):
    gap_min = MIN_GAP if gap_min is None else gap_min
    out = []
    for a, b in itertools.combinations(comps, 2):
        if comps[a]["fixed"] and comps[b]["fixed"]:
            continue
        wa, ha = size(comps[a]); wb, hb = size(comps[b])
        dx = abs(comps[a]["x"] - comps[b]["x"]) - (wa + wb) / 2 - gap_min
        dy = abs(comps[a]["y"] - comps[b]["y"]) - (ha + hb) / 2 - gap_min
        if dx < 0 and dy < 0:
            out.append((a, b, round(-dx, 2), round(-dy, 2)))
    return out


if __name__ == "__main__":
    parts, nets = parse_sdf(SDF)
    pnp = parse_pnp(PNP)

    comps, missing_pk = {}, set()
    for ref, c in pnp.items():
        pk = c["package"]
        if pk not in FP:
            missing_pk.add(pk); continue
        w, h, pins, (ox, oy) = FP[pk]
        cx, cy = rotate(ox, oy, c["rot"])
        comps[ref] = {"x": round(c["x_org"] + cx, 4), "y": round(c["y_org"] + cy, 4),
                      "x_org": c["x_org"], "y_org": c["y_org"], "ox": ox, "oy": oy,
                      "rot": c["rot"], "layer": c["layer"], "value": c["value"], "package": pk,
                      "group": parts.get(ref, {}).get("group", 0), "w": w, "h": h,
                      "fixed": ref in FIXED, "power_w": POWER_W.get(ref, 0.0),
                      "pins": {p: (dx - ox, dy - oy) for p, (dx, dy) in pins.items()}}   # relativos al centro
    if missing_pk:
        raise SystemExit(f"Faltan footprints en FP: {sorted(missing_pk)}")

    # redes: sin {NC}, solo pines de componentes del Pick&Place, >= 2 pines
    nets_out, no_pin = {}, set()
    for n, v in nets.items():
        if n in NET_IGNORE_NAMES:
            continue
        pins = []
        for r, p in v["pins"]:
            if r not in comps:
                continue                                  # TP1..TP4 no están en el P&P
            if p not in comps[r]["pins"]:
                no_pin.add((r, p, comps[r]["package"])); continue
            dx, dy = comps[r]["pins"][p]
            pins.append([r, p, dx, dy])
        if len(pins) >= 2:
            nets_out[n] = {"class": v["class"], "pins": pins}

    json.dump({"board": {"x0": 0, "y0": 0, "w": BOARD_W, "h": BOARD_H, "holes": HOLES,
                         "hole_keepout": HOLE_KEEPOUT, "edge_margin": EDGE_MARGIN, "min_gap": MIN_GAP,
                         "note": "borde = linea amarilla punteada de ARES, origen abajo-izq"},
               "components": {r: {**{k: v for k, v in c.items() if k != "pins"}, "pins": {pn: list(xy) for pn, xy in c["pins"].items()}} for r, c in comps.items()},
               "nets": nets_out}, open(OUT, "w"), indent=1)

    # ------------------------------ reporte ------------------------------
    pos = {r: (c["x"], c["y"]) for r, c in comps.items()}
    base = {}
    for n, v in nets_out.items():
        if n == "GND":
            continue
        base[n] = hpwl_pts([pin_xy(comps, r, p) for r, p, *_ in v["pins"]])
    print(f"Componentes: {len(comps)} (fijos: {sum(c['fixed'] for c in comps.values())}) | Redes (>=2 pines): {len(nets_out)}")
    print(f"HPWL total baseline con pines (sin GND): {sum(base.values()):.1f} mm")
    print("Top 8 redes por HPWL:")
    for n, v in sorted(base.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {n:16s} {v:6.1f} mm  ({len(nets_out[n]['pins'])} pines)")
    if no_pin:
        print("AVISO pines usados en la netlist sin geometría definida (se omiten):", sorted(no_pin))
    ov = overlaps(comps)
    print(f"Pares a menos de {MIN_GAP} mm entre courtyards (estimados): {len(ov)}")
    for o in ov:
        print("  ", o)
    near = []
    for r, c in comps.items():
        if c["fixed"]:
            continue
        w, h = size(c)
        d = min(c["x"] - w / 2, BOARD_W - (c["x"] + w / 2), c["y"] - h / 2, BOARD_H - (c["y"] + h / 2))
        if d < EDGE_MARGIN:
            near.append((r, round(d, 2)))
    print(f"Componentes móviles a menos de {EDGE_MARGIN} mm del borde: {near or 'ninguno'}")
    print("Refs sin Pick&Place (se ignoran):", sorted(set(parts) - set(pnp)))
