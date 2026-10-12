# Proyecto 2 — Diseño generativo de personajes: Glimmerfen

**Curso:** Deep Learning 2026, Universidad del Valle
**Pareja:** Cindy Gualim y Gadiel Ocaña
**Universo:** *Glimmerfen*, espíritus guardianes elementales de bosque (ver `PROJECT.md` para la documentación completa del proyecto)

Este repositorio entrena una **DCGAN** (y dos variantes experimentales: WGAN-GP y
normalización espectral) para generar personajes nuevos del universo Glimmerfen,
siguiendo todos los requisitos del enunciado (`Proyecto_2_Personajes_GAN.pdf`).

## Estructura del repositorio

```
├── PROJECT.md                     # Documentación completa del proyecto (informe)
├── README.md                      # Este archivo
├── requirements.txt
├── pytest.ini
├── notebooks/
│   └── Proyecto2_Glimmerfen.ipynb # Notebook principal: dataset, entrenamiento,
│                                  # experimentos, evidencias y galería final
├── src/ganverse/                  # Paquete con la lógica reutilizable y testeada
│   ├── dataset.py                 # Generador procedural del universo + Dataset
│   ├── models.py                  # Generador y Discriminador (DCGAN)
│   ├── losses.py                  # BCE no saturante y WGAN-GP (+ gradient penalty)
│   ├── train.py                   # Bucle de entrenamiento y presets de experimentos
│   ├── evidence.py                # Las 3 evidencias obligatorias
│   ├── gallery.py                 # Selección, nombrado y guardado de la galería
│   └── utils.py
├── scripts/
│   ├── run_experiments.py         # Corre los 3 entrenamientos (base, pérdida, estabilización)
│   ├── build_evidence.py          # Genera las figuras de evidencia a partir de lo ya entrenado
│   └── regenerate_gallery.py      # Regenera la galería final SOLO desde pesos + semilla
├── tests/                         # Pruebas unitarias (pytest)
├── data/glimmerfen/                # Dataset generado (.npy) + metadata.json + muestras PNG
├── experiments/<nombre>/          # checkpoint.pt, loss_history.json, snapshots/
├── figures/                       # Curvas de pérdida, rejillas de ruido fijo, comparaciones
└── galeria/                       # Los 10 personajes finales + metadata.csv + z_vectors.pt
```

## Instalación

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Probado con **Python 3.12.4**. Librerías principales y versión usada durante el
desarrollo: `torch==2.14.0`, `numpy==2.5.2`, `pillow==12.3.0`, `matplotlib==3.11.1`,
`pytest==9.1.1`. No se usó `torchvision` (todo el pipeline de datos/imágenes se
implementó con PIL + NumPy puro).

El entrenamiento usa automáticamente **GPU Apple Metal (MPS)** si está disponible
(`ganverse.utils.get_device`), si no cae a CUDA, y si no a CPU.

## Cómo ejecutar

### 1. Pruebas unitarias

```bash
PYTHONPATH=src pytest -v
```

### 2. Generar el dataset (procedural, propio del equipo)

```bash
PYTHONPATH=src python -c "from ganverse.dataset import save_dataset; save_dataset(n=4000, out_dir='data/glimmerfen', seed=42)"
```

### 3. Entrenar los tres experimentos (base, estabilización, pérdida)

```bash
PYTHONPATH=src python scripts/run_experiments.py
```

Esto entrena, en orden, los presets `baseline`, `stab_spectral_norm` y
`loss_wgangp` definidos en `ganverse/train.py`, guardando checkpoints,
snapshots de ruido fijo y el historial de pérdidas en `experiments/<nombre>/`.
Tiempo aproximado en Apple Silicon (GPU MPS) con memoria disponible: ~7 min
(baseline), ~7 min (normalización espectral), ~35-45 min (WGAN-GP completo,
45 épocas, por los 5 pasos de crítico y la doble pasada hacia atrás que exige
la penalización de gradiente). En CPU, considerablemente más lento — se
recomienda Colab/Kaggle con GPU si no se dispone de Apple Silicon.

> **Nota sobre la corrida de WGAN-GP entregada:** durante el desarrollo, el
> equipo se quedó sin memoria disponible para completar las 45 épocas
> completas de WGAN-GP (ver `PROJECT.md`, secciones 4 y 6.3, para el detalle).
> La corrida que se entrega en `experiments/loss_wgangp/` usa un presupuesto
> reducido (`feature_maps=32`, `n_critic=3`, 15 épocas) que sí corre de forma
> confiable y rápida (~2-3 min) incluso con poca memoria disponible. Para
> reproducir la configuración completa (más fiel al paper de WGAN-GP, pero
> mucho más pesada), usar:
> ```bash
> PYTHONPATH=src python -c "
> from ganverse.train import PRESETS, train
> train(PRESETS['loss_wgangp'])
> "
> ```

### 4. Generar las evidencias

```bash
PYTHONPATH=src python scripts/build_evidence.py
```

### 5. Regenerar la galería final SOLO desde pesos + semilla

```bash
PYTHONPATH=src python scripts/regenerate_gallery.py --run stab_spectral_norm --seed 2026
```

Este script **no reentrena nada**: carga `experiments/<run>/checkpoint.pt` y
reproduce exactamente la galería de `galeria/` a partir de la semilla
declarada (ver `galeria/selection_info.json` y `galeria/metadata.csv`).

### 6. Notebook principal

`notebooks/Proyecto2_Glimmerfen.ipynb` contiene el recorrido narrativo completo
(universo, datos, arquitectura, los dos experimentos con su hipótesis previa y
análisis posterior, las tres evidencias obligatorias, la galería final y la
reflexión de cierre) y se puede ejecutar de principio a fin. Por defecto usa
los checkpoints ya entrenados en `experiments/` (rápido); cambiando la
variable `RETRAIN = True` en la segunda celda, reentrena todo desde cero.

## Semillas utilizadas

| Propósito                              | Semilla |
|-----------------------------------------|---------|
| Generación del dataset                  | 42      |
| Entrenamiento (las 3 corridas)           | 2026    |
| Ruido fijo de evaluación (compartido)    | 7       |
| Selección de la galería final            | 2026    |

## Uso de asistentes de IA

Se utilizó **Claude (Anthropic, Claude Code)** como asistente durante todo el
proyecto: para diseñar la arquitectura del código (paquete `ganverse`),
escribir el generador procedural del dataset, los modelos, las funciones de
pérdida (BCE no saturante y WGAN-GP con penalización de gradiente), el ciclo
de entrenamiento, las pruebas unitarias (`pytest`), los scripts de evidencia y
galería, el notebook y la documentación (`PROJECT.md`, este `README.md` y la
presentación). El equipo revisó el código generado, ejecutó el entrenamiento
real en su propia máquina, interpretó las curvas de pérdida y las evidencias,
y es responsable del análisis y las decisiones reportadas en `PROJECT.md`.

**Ninguna imagen de la galería final proviene de un generador externo**: las
10 imágenes de `galeria/` son salida directa del modelo DCGAN/WGAN-GP/con
normalización espectral entrenado por el equipo (ver sección 2 y 8 del
enunciado), a partir del dataset procedural propio.

## Licencia de los datos

El dataset (`data/glimmerfen/`) es generado por código propio
(`ganverse/dataset.py`, dibujado con PIL), no proviene de internet ni contiene
rostros reales ni personajes de terceros. Es propiedad del equipo (CC0 /
material original para este curso).
