# Resumen: prototipo de optimización de placement (AI-SED 2026, BandaoTeam)

Fecha del resumen: 2 nov 2026. Resume lo trabajado en un chat donde no se pudo guardar memoria.

## Contexto
- Mientras el equipo de electrónica termina el esquemático y el ruteo ARES de la placa real (monitoreo de vibración en un equipo de wafer dicing), se prototipó el algoritmo de optimización sobre un proyecto de la capacitación: **MotorPaso_PIC16** (motor a pasos con PIC16F628A, ULN2803, buck MC34063; 40 componentes, 2 capas, ruteado en ARES).
- Objetivo: validar el método, no obtener resultados del concurso. Esos números NO se presentan como resultados de la placa del wafer dicing.

## Decisiones tomadas
- Algoritmo: **NSGA-II** (pymoo) con 3 objetivos: HPWL, térmico, EMI. Restricciones duras: sin traslape, dentro de la placa, separación de agujeros de montaje. El SA queda como línea base opcional.
- Placa: 75 × 50 mm, origen en la esquina inferior izquierda (confirmado con capturas de ARES). Agujeros de montaje estimados en (5,5), (5,45), (70,5), (70,45).
- Componentes fijos (decisión de diseño, no sale de Proteus): J1, J2, M1, S1, S2, S3.
- Roles artificiales para probar el método: víctima = X1 (cristal 4 MHz) y U1 (PIC16F628A); agresores = U3 (MC34063A), L1 (inductor 330 µH), D4 (diodo Schottky 10BQ040), U2 (ULN2803). Potencias estimadas: U3 0.35 W, U2 0.40 W, D4 0.15 W, L1 0.10 W.
- En la placa real: víctima = acelerómetro IIS3DWB; agresores = ESP32-S3 (WiFi), regulador, RS-485. El "antes" legítimo es el primer placement manual de electrónica.

## Archivos
1. `parse_proteus.py`: lee SDF (netlist) + CSV de Pick and Place, calcula posición real de pines, imprime tabla de calibración y escribe `board_data.json` (pines con 4 valores: ref, pin, dx, dy).
2. `nsga2_placement.py`: optimizador. Salidas en `out/`: placement_elegido.csv, pareto.csv, pareto.png, convergencia.png, layout_antes_despues.png, calor_antes_despues.png, archivos .npy.
3. `analisis_separaciones.py`: separaciones en mm entre agresores y víctimas, antes/después y contra extremos del Pareto.

Uso: `pip install pymoo scipy matplotlib`, luego `python parse_proteus.py` y `python nsga2_placement.py --gens 800 --pop 300 --seed 2`.
Error típico: si `parse_proteus.py` exporta pines con 2 valores en vez de 4, es una versión vieja del parser; reemplazarla por la última.

## Calibración del HPWL contra ARES (cobre "Etch", mm)
| Red | HPWL centros | HPWL pines | ARES | pines/ARES |
|---|---|---|---|---|
| AZUL_1 | 27.3 | 10.4 | 9.98 | 1.05 |
| ROSA_2 | 27.3 | 15.5 | 13.94 | 1.11 |
| SWITCHING_NODE | 15.4 | 7.9 | 7.96 | 0.99 |
| PGD | 29.1 | 34.4 | 57.52 | 0.60 |
- Medir con centros de componentes fallaba por un factor 2 a 2.7 (U1 y U2 son DIL18 de 23 mm). Con pines reales coincide.
- Cada clic en ARES mide un tramo; en redes de 3+ pines hay que sumar tramos (SWITCHING_NODE = 4.965 + 2.995).
- PGD: ARES rodeó C1, X1 y C2 (desvío por obstáculos); el HPWL no lo ve. Es 1 red de 4: no es estadística.

## Resultados (800 generaciones, población 300, semilla 2)
- Elegida (mejor suma de razones vs actual): HPWL −0.9 %, térmico −19.8 %, EMI −49.1 %. HPWL actual = 799.9.
- El −10.5 % de HPWL de una primera corrida era un error (centros en vez de pines); se retiró.
- Extremo "solo HPWL": 733 (−8.4 %), pero X1 queda a ~24 mm de U3 y el calor no mejora.
- Entre soluciones que conservan la protección: mejor HPWL −3.2 %, con X1 a 27 a 33 mm de los agresores.
- Separaciones (actual → elegida, mm): U3→X1 23.3→35.5; L1→X1 21.4→34.5; D4→X1 14.0→30.2; U2→X1 16.3→30.3; U2→U1 6.2→17.6; D4→U1 9.9→19.7. Área del lazo U3-D4-L1: 59→29 mm².
- Esas separaciones salieron casi sin costo de cableado. El bloque X1/U1/C1/C2 se corrió ~8 a 10 mm a la izquierda.

## Limitaciones conocidas
- Courtyards, potencias, roles y agujeros son estimaciones. Solo el HPWL está validado con ARES (4 redes).
- El índice de EMI está dominado por el área del lazo (2.95 de 3.35 en el diseño actual); su peso W_LOOP = 0.05 es arbitrario. Citar separaciones en mm y área del lazo, no "−49 % EMI".
- U3, L1 y D4 quedan a 0 mm entre sí: falta una separación mínima entre componentes.
- La rotación solo cambia el tamaño del courtyard y la posición de los pines; no modela orientación de pads.
- El mapa de calor es cualitativo y no participa en la optimización.
- Ruteabilidad (congestión / A*) todavía no está implementada.

## Pendientes
1. Restricción de separación mínima entre componentes (0.5 a 1 mm).
2. Reportar métricas en mm; revisar el peso del lazo.
3. Análisis de sensibilidad liberando los componentes fijos.
4. Geometría de pines del ESP32-S3-WROOM-1, IIS3DWB y demás; zona prohibida de la antena (guía de Espressif).
5. Módulo de congestión de ruteo y prueba de 2 contra 4 capas.
6. Con la placa real: aplicar placement_elegido.csv a mano en ARES, re-rutear y medir cobre por red.
