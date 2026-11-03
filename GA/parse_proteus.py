"""
Parser de Proteus -> board_data.json
Entradas: netlist SDF, Pick&Place CSV (mm), BOM CSV (opcional)
Salida:   board_data.json  (componentes, redes, contorno) + reporte de validación
Uso:      python parse_proteus.py
"""
# csv lee el archivo de montaje; json genera los datos consumidos por el
# optimizador; re extrae campos del SDF; itertools genera pares de componentes.
import csv, json, re, itertools, math
from pathlib import Path

# Archivos de entrada y salida relativos al directorio desde donde se ejecuta
# el script. Si cambian los nombres de exportación, actualizarlos aquí.
SDF = "MotorPaso_PIC16_Placed_Despues_IA.SDF"
PNP = "MotorPaso_PIC16_Placed_Despues_IA_Base_Design.csv"
OUT = "board_data.json"

# Tamaños aproximados de las huellas en mm a rotación 0. Se usan como
# courtyards para estimar colisiones; deben calibrarse con las huellas reales
# en ARES antes de tratar el resultado como una validación mecánica definitiva.
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
# Referencias cuyo placement se conserva fijo durante la optimización. La lista
# es una decisión del diseño, no una propiedad inferida automáticamente del SDF.
FIXED = {"J1", "J2", "M1", "S1", "S2", "S3"}
# Potencias disipadas estimadas en vatios; referencias no listadas reciben 0 W.
# Afinar estos valores con hojas de datos o mediciones mejora el modelo térmico.
POWER_W = {"U3": 0.35, "L1": 0.10, "D4": 0.15, "U2": 0.40, "U1": 0.05, "R5": 0.05}


def parse_sdf(path):
    """Extrae componentes y redes del formato SDF exportado por Proteus."""
    # latin-1 permite leer exportaciones con caracteres extendidos. Se quitan
    # los retornos de carro para procesar las líneas de forma uniforme.
    txt = Path(path).read_text(encoding="latin-1").replace("\r", "")

    # --- PARTLIST: el bloque termina al comenzar NETLIST. Se omite la primera
    # línea, que corresponde al encabezado del bloque.
    part_block = txt.split("*PARTLIST")[1].split("*NETLIST")[0].splitlines()[1:]
    parts = {}
    for line in part_block:
        if not line.strip():
            continue
        # La primera columna identifica la referencia. GROUP y PACKAGE pueden
        # no aparecer, por lo que se asignan valores vacíos/de respaldo.
        ref = line.split(",")[0]
        m = re.search(r"GROUP=(\d+)", line)
        pk = re.search(r"PACKAGE=([^,]+)", line)
        parts[ref] = {"group": int(m.group(1)) if m else 0,
                      "package": pk.group(1) if pk else ""}

    # --- NETLIST: cada encabezado abre una red; las líneas siguientes contienen
    # sus pines hasta una línea vacía, que marca el fin de esa red.
    net_block = txt.split("*NETLIST")[1].splitlines()[1:]
    nets, cur = {}, None
    for line in net_block:
        line = line.strip()
        if not line:
            # Reiniciar el estado permite interpretar la siguiente línea no
            # vacía como el nombre/encabezado de otra red.
            cur = None
            continue
        f = line.split(",")
        if cur is None:                      # encabezado de red
            cur = f[0]
            # CLASS se conserva para permitir que el optimizador pondere redes
            # críticas; si no está declarada, se guarda como cadena vacía.
            nets[cur] = {"pins": [], "class": next((x.split("=")[1] for x in f if x.startswith("CLASS=")), "")}
        elif f[0] in parts:                  # conservar pines de referencias reales
            # Los registros auxiliares de Proteus (LBL/GT/PR/PT, etc.) se
            # descartan porque su primera columna no coincide con un componente.
            nets[cur]["pins"].append([f[0], f[2]])
    return parts, nets


def parse_pnp(path):
    """Lee posiciones y atributos de componentes del CSV Pick&Place."""
    comps = {}
    # utf-8-sig acepta CSV con o sin BOM. DictReader usa los encabezados de
    # columna (Part ID, X, Y, Rotation, Layer, Value y Package).
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            # La orientación se normaliza al rango 0..359 y X/Y se convierten
            # a números para que puedan usarse en cálculos geométricos.
            rot = int(float(r["Rotation"])) % 360
            # La referencia Part ID sirve como clave común para unir esta tabla
            # con los componentes y pines que aparecen en el SDF.
            comps[r["Part ID"]] = {"x": float(r["X"]), "y": float(r["Y"]), "rot": rot,
                                   "layer": r["Layer"], "value": r["Value"], "package": r["Package"]}
    return comps


def size(c):
    """Devuelve ancho y alto de la huella ya ajustados a su orientación."""
    # FOOTPRINT_MM define dimensiones para 0 grados; al girar 90/270 grados se
    # intercambian los ejes. Otras rotaciones conservan la caja original.
    w, h = FOOTPRINT_MM[c["package"]]
    return (h, w) if c["rot"] in (90, 270) else (w, h)


def hpwl(net, pos):
    """Calcula HPWL: ancho más alto de la caja que contiene los pines de una red."""
    # Solo se usan pines cuyas referencias existen en el mapa de posiciones.
    pts = [pos[r] for r, _ in net["pins"] if r in pos]
    # Una red con menos de dos puntos no tiene longitud de conexión medible.
    if len(pts) < 2:
        return 0.0
    # HPWL (half-perimeter wirelength) aproxima la longitud mínima de conexión
    # ortogonal sumando el rango horizontal y el rango vertical.
    xs, ys = zip(*pts)
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def overlaps(comps):
    """Encuentra pares de huellas cuyos rectángulos estimados se intersectan."""
    out = []
    # combinations evita comparar cada par dos veces y excluye cada componente
    # consigo mismo.
    for a, b in itertools.combinations(comps, 2):
        # dx/dy son separaciones entre bordes: un valor negativo indica
        # penetración en ese eje.
        wa, ha = size(comps[a]); wb, hb = size(comps[b])
        dx = abs(comps[a]["x"] - comps[b]["x"]) - (wa + wb) / 2
        dy = abs(comps[a]["y"] - comps[b]["y"]) - (ha + hb) / 2
        # Solo existe traslape rectangular cuando hay penetración en ambos ejes;
        # se reporta la penetración horizontal y vertical redondeada a 0.01 mm.
        if dx < 0 and dy < 0:
            out.append((a, b, round(-dx, 2), round(-dy, 2)))
    return out


if __name__ == "__main__":
    # Este bloque solo se ejecuta al lanzar el archivo directamente, no cuando
    # sus funciones se importan desde otro módulo.

    # Lee el inventario de componentes/redes del netlist y las coordenadas del
    # Pick&Place. Las referencias sirven para asociar los dos orígenes de datos.
    parts, nets = parse_sdf(SDF)
    comps = parse_pnp(PNP)
    for ref, c in comps.items():
        # Completa cada componente con metadatos del SDF y valores configurados
        # localmente para el optimizador. parts[ref] presupone que toda referencia
        # del CSV también existe en la PARTLIST.
        c["group"] = parts[ref]["group"]
        c["w"], c["h"] = FOOTPRINT_MM[c["package"]]
        c["fixed"] = ref in FIXED
        c["power_w"] = POWER_W.get(ref, 0.0)

    # Posiciones por referencia para medir HPWL. Las redes de dos o más pines
    # son las únicas que pasan al JSON; GND se excluye solo del baseline HPWL.
    pos = {r: (c["x"], c["y"]) for r, c in comps.items()}
    nets_out = {n: v for n, v in nets.items() if len(v["pins"]) >= 2}
    base = {n: hpwl(v, pos) for n, v in nets_out.items() if n != "GND"}  # GND va por plano

    # Caja envolvente de las huellas estimadas. Es una cota mínima del espacio
    # ocupado por los componentes, no el contorno real de la placa.
    x0 = min(c["x"] - size(c)[0] / 2 for c in comps.values())
    x1 = max(c["x"] + size(c)[0] / 2 for c in comps.values())
    y0 = min(c["y"] - size(c)[1] / 2 for c in comps.values())
    y1 = max(c["y"] + size(c)[1] / 2 for c in comps.values())

    # Escribe el formato de entrada del optimizador. Las dimensiones de placa
    # aquí son valores configurados (75 x 50 mm), no se infieren de x0..y1.
    json.dump({"board": {"x0": 0, "y0": 0, "w": 75.0, "h": 50.0, "note": "ESTIMADO, confirmar en ARES"},
               "components": comps, "nets": nets_out}, open(OUT, "w"), indent=1)

    # Reporte de resumen: cantidad de datos escritos y envolvente de huellas.
    print(f"Componentes: {len(comps)} | Redes (>=2 pines): {len(nets_out)}")
    print(f"BBox de componentes (mm): x {x0:.1f}..{x1:.1f}  y {y0:.1f}..{y1:.1f}")
    print(f"HPWL total baseline (sin GND): {sum(base.values()):.1f} mm")

    # Ordena las redes de mayor a menor HPWL para destacar las ocho conexiones
    # con mayor extensión estimada en el placement actual.
    print("Top 8 redes por HPWL:")
    for n, v in sorted(base.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {n:16s} {v:6.1f} mm  ({len(nets_out[n]['pins'])} pines)")

    # Reporta cruces entre courtyards estimados y referencias del CSV que no
    # están presentes en la PARTLIST. La comparación de referencias sirve como
    # verificación de consistencia entre ambas fuentes.
    ov = overlaps(comps)
    print(f"Traslapes con courtyards estimados: {len(ov)}")
    for o in ov:
        print("  ", o)
    missing = set(comps) - set(parts)
    print("Refs en CSV sin partlist:", missing or "ninguno")
