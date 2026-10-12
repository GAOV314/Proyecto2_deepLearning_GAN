"""Construye presentacion.pdf (<=12 diapositivas) a partir de las figuras ya
generadas en figures/ y galeria/, siguiendo la estructura de la sección 5.1
del enunciado. Se arma con matplotlib (sin dependencias de Office/LaTeX).
"""
import csv
import sys
from pathlib import Path
from textwrap import wrap

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from ganverse.utils import load_json  # noqa: E402

FIGURES_DIR = REPO_ROOT / "figures"
GALERIA_DIR = REPO_ROOT / "galeria"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
OUT_PDF = REPO_ROOT / "presentacion.pdf"

PAGE_SIZE = (13.33, 7.5)  # 16:9
NAVY = "#1f2d3d"
ACCENT = "#2e86ab"


def new_slide():
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    return fig


def title_slide(pdf):
    fig = new_slide()
    fig.text(0.5, 0.62, "Glimmerfen", ha="center", fontsize=54, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.52, "Diseño generativo de personajes con GAN", ha="center", fontsize=22, color=ACCENT)
    fig.text(0.5, 0.40, "Proyecto 2 — Deep Learning 2026 — Universidad del Valle", ha="center", fontsize=14)
    fig.text(0.5, 0.34, "Cindy Gualim  ·  Gadiel Ocaña", ha="center", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.08, "Docente: Kevin Recinos", ha="center", fontsize=11, color="gray")
    pdf.savefig(fig); plt.close(fig)


def bullet_slide(pdf, title, bullets, subtitle=None):
    fig = new_slide()
    title_fontsize = 28 if len(title) <= 55 else 21
    fig.text(0.06, 0.90, title, fontsize=title_fontsize, fontweight="bold", color=NAVY)
    if subtitle:
        fig.text(0.06, 0.83, subtitle, fontsize=14, color=ACCENT, style="italic")
    y = 0.72 if subtitle else 0.78
    for b in bullets:
        wrapped = wrap(b, width=100)
        fig.text(0.08, y, "•  " + wrapped[0], fontsize=15, color="black")
        y -= 0.065
        for cont in wrapped[1:]:
            fig.text(0.115, y, cont, fontsize=15, color="black")
            y -= 0.065
        y -= 0.02
    pdf.savefig(fig); plt.close(fig)


def image_slide(pdf, title, image_paths, caption=None, ncols=None):
    fig = new_slide()
    fig.text(0.06, 0.93, title, fontsize=26, fontweight="bold", color=NAVY)
    n = len(image_paths)
    ncols = ncols or n
    nrows = -(-n // ncols)
    for i, path in enumerate(image_paths):
        ax = fig.add_axes([0.04 + (i % ncols) * (0.92 / ncols),
                            0.12 + (nrows - 1 - i // ncols) * (0.72 / nrows),
                            0.92 / ncols - 0.02, 0.72 / nrows - 0.02])
        ax.imshow(plt.imread(path))
        ax.axis("off")
    if caption:
        fig.text(0.5, 0.05, caption, ha="center", fontsize=11, color="gray", wrap=True)
    pdf.savefig(fig); plt.close(fig)


def table_slide(pdf, title, headers, rows, col_widths=None):
    fig = new_slide()
    fig.text(0.06, 0.93, title, fontsize=26, fontweight="bold", color=NAVY)
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.78])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left",
                      colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
    pdf.savefig(fig); plt.close(fig)


def main():
    loss_summaries = load_json(FIGURES_DIR / "loss_summaries.json") if (FIGURES_DIR / "loss_summaries.json").exists() else {}
    sel_info = load_json(GALERIA_DIR / "selection_info.json") if (GALERIA_DIR / "selection_info.json").exists() else {}

    with PdfPages(OUT_PDF) as pdf:
        # 1. Portada
        title_slide(pdf)

        # 2. Universo y datos
        bullet_slide(pdf, "Universo y datos: Glimmerfen", [
            "Espíritus guardianes elementales de un bosque brumoso: cuerpo tipo blob, ojos grandes, "
            "apéndice opcional (cuernos, alas, orejas, cola, antena).",
            "Seis elementos (fuego, agua, tierra, aire, luz, sombra), cada uno con su paleta de color.",
            "Dataset: 4000 imágenes de 64x64, generadas por código propio (PIL) — sin descargas de "
            "internet, sin rostros reales ni personajes de terceros. Licencia: CC0 / del equipo.",
            "Preparación: normalización a [-1,1]; aumento de datos por flip horizontal.",
        ])

        # 3. Arquitectura
        bullet_slide(pdf, "Arquitectura de la GAN", [
            "DCGAN (Radford et al., 2015): Generador con 5 ConvTranspose2d + BatchNorm + ReLU, salida Tanh.",
            "Discriminador con 4 Conv2d + LeakyReLU, normalización intercambiable "
            "(BatchNorm / Normalización Espectral / ninguna).",
            "Se eligió DCGAN por ser la arquitectura recomendada para empezar simple y por su "
            "comportamiento bien documentado en 64x64.",
            "Mismo Generador en las 3 corridas; la variable que cambia es siempre el Discriminador "
            "y/o la función de pérdida.",
        ])

        # 4. Experimento de pérdida
        bullet_slide(pdf, "Experimento de pérdida: BCE vs. WGAN-GP",
                     subtitle="Hipótesis: WGAN-GP mejora la estabilidad al evitar la saturación de la divergencia JS",
                     bullets=[
            "Configuración controlada: mismos datos, semilla y ruido fijo de evaluación.",
            "Cambio de literatura (no arbitrario): crítico sin BatchNorm + Adam(1e-4,(0,0.9)) + n_critic=5/3, "
            "tal como exige Gulrajani et al. (2017).",
            "LIMITACIÓN DECLARADA: por falta de memoria del equipo, esta corrida se redujo a 15 épocas "
            "(no 45) y feature_maps=32 (no 64) para poder completarse — no es comparación justa de calidad final.",
            "Resultado: la pérdida del crítico decrece monótona y suave (de -90 a -14), sin los picos "
            "bruscos de BCE — confirma la dinámica más predecible esperada, aunque sin converger del todo.",
        ])

        # 5. Experimento de estabilización
        bullet_slide(pdf, "Experimento de estabilización: BatchNorm vs. Normalización Espectral",
                     subtitle="Hipótesis: la normalización espectral acota la Lipschitz de D y evita que domine a G",
                     bullets=[
            "Configuración controlada: mismos datos, semilla, 45 épocas, ruido fijo y misma pérdida (BCE).",
            "Única variable que cambia: normalización del discriminador.",
            "Resultado: pérdida D media 0.076 (base, cerca de 0 = alarma) vs. 0.730 (espectral); "
            "varianza de G 3.34 vs. 0.53 — hipótesis confirmada, se eligió este modelo para la galería.",
        ])

        # 6. Evidencia 1: ruido fijo
        fixed_noise_imgs = [FIGURES_DIR / f"fixed_noise_evolution_{r}.png"
                             for r in ("baseline", "stab_spectral_norm", "loss_wgangp")
                             if (FIGURES_DIR / f"fixed_noise_evolution_{r}.png").exists()]
        if fixed_noise_imgs:
            image_slide(pdf, "Evidencia 1: rejilla de ruido fijo", fixed_noise_imgs, ncols=1,
                        caption="Mismo lote de 16 z en cada panel, a lo largo del entrenamiento (baseline, "
                                "normalización espectral, WGAN-GP).")

        # 7. Evidencia 2: curvas de pérdida
        loss_curve_imgs = [FIGURES_DIR / f"loss_curve_{r}.png"
                            for r in ("baseline", "stab_spectral_norm", "loss_wgangp")
                            if (FIGURES_DIR / f"loss_curve_{r}.png").exists()]
        if loss_curve_imgs:
            image_slide(pdf, "Evidencia 2: curvas de pérdida interpretadas", loss_curve_imgs, ncols=3)

        # 8. Evidencia 3: vecinos más cercanos
        if (FIGURES_DIR / "nearest_neighbors.png").exists():
            image_slide(pdf, "Evidencia 3: vecinos más cercanos", [FIGURES_DIR / "nearest_neighbors.png"],
                        caption="Fila superior: personaje generado. Fila inferior: vecino más cercano en "
                                "el set de entrenamiento (MSE de píxeles).")

        # 9. Problemas y decisiones
        bullet_slide(pdf, "Problemas encontrados y decisiones tomadas", [
            "D domina a G en la base (pérdida de D ≈0 sostenida) -> colapso de modos parcial visible en la "
            "rejilla de ruido fijo entre las épocas 27 y 44. Motivó el experimento de estabilización.",
            "Restricción real de cómputo: WGAN-GP se interrumpió varias veces por falta de memoria del "
            "equipo; se resolvió reduciendo su presupuesto (feature_maps=32, n_critic=3, 15 épocas).",
            "Bug encontrado y corregido: train() no limpiaba snapshots/ entre reintentos, mezclando "
            "imágenes de configuraciones distintas bajo el mismo experimento — corregido en el código.",
            "Primera versión de la selección de galería daba 8/10 personajes casi idénticos (prohibido "
            "por el enunciado); se corrigió con un algoritmo farthest-point que maximiza diversidad real.",
        ])

        # 10. Galería final
        if (FIGURES_DIR / "gallery_overview.png").exists():
            rate = sel_info.get("selection_rate", "?")
            image_slide(pdf, "Galería final: 10 espíritus de Glimmerfen", [FIGURES_DIR / "gallery_overview.png"],
                        caption=f"Tasa de selección declarada: {rate}.")

        # 11. Conclusiones y limitaciones + reflexión
        bullet_slide(pdf, "Conclusiones, limitaciones y reflexión final", [
            "Normalización espectral cumplió su hipótesis con evidencia cuantitativa clara; WGAN-GP mostró "
            "la dinámica esperada pero no se pudo comparar en igualdad de condiciones (limitación de cómputo).",
            "El universo es deliberadamente simple (generado por código) para poder documentar origen/licencia "
            "con certeza y entrenar algo real en el tiempo disponible — un dataset de arte a mano daría más "
            "detalle pero sacrificaría esa trazabilidad.",
            "Reflexión: Terro es el personaje más difícil de defender como 'nuevo' (vecino más cercano a solo "
            "MSE=616, 2.5-7x más cerca que el resto). Esto sugiere que el modelo aprendió las categorías del "
            "universo (paleta, apéndice, N° de ojos) y las recombina, en vez de memorizar — pero las combinaciones "
            "más comunes en el dataset quedan, por definición, más cerca de algún ejemplar real.",
        ])

        # 12. Matriz de evidencias
        table_slide(pdf, "Matriz de evidencias",
                    headers=["Evidencia", "Diapositiva", "Conclusión", "Limitación"],
                    rows=[
                        ["Rejilla de ruido fijo", "6", "Base colapsa parcial; espectral\nmantiene 16 identidades distintas", "WGAN-GP (15 épocas) no\ncomparable en diversidad"],
                        ["Curvas de pérdida", "7", "D≈0 en base (alarma); espectral estable\n(~0.73, 6x menos var. en G)", "Umbrales de \"sano\"\nson cualitativos"],
                        ["Vecinos más cercanos", "8", "Ningún personaje casi idéntico\n(todos MSE>600, umbral=60)", "Solo compara contra\nnuestro propio dataset"],
                        ["Galería final (10 personajes)", "10", "10 personajes diversos (5/6 elementos,\n1-2 ojos, apéndices variados)", "Puntaje viene del propio\ndiscriminador, no de humanos"],
                    ],
                    col_widths=[0.22, 0.12, 0.33, 0.33])

    print(f"Presentación generada en {OUT_PDF}")


if __name__ == "__main__":
    main()
