"""Genera las evidencias obligatorias (sección 3) para cada experimento ya
entrenado: curvas de pérdida, rejilla de ruido fijo evolutiva, y una
comparación lado a lado entre la configuración base y cada variante."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ganverse.evidence import fixed_noise_evolution, plot_loss_curves  # noqa: E402
from ganverse.utils import load_json, save_json  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
FIGURES_DIR = REPO_ROOT / "figures"

RUNS = {
    "baseline": "Base: DCGAN, BCE no saturante, BatchNorm en D",
    "stab_spectral_norm": "Estabilización: BCE no saturante, Normalización Espectral en D",
    "loss_wgangp": "Pérdida: WGAN-GP (crítico sin BatchNorm, n_critic=5)",
}


def main():
    summaries = {}
    for run, title in RUNS.items():
        run_dir = EXPERIMENTS_DIR / run
        if not (run_dir / "loss_history.json").exists():
            print(f"[skip] {run}: no hay loss_history.json todavía")
            continue
        summary = plot_loss_curves(run_dir / "loss_history.json",
                                    FIGURES_DIR / f"loss_curve_{run}.png", title)
        summaries[run] = summary
        fixed_noise_evolution(run_dir, FIGURES_DIR / f"fixed_noise_evolution_{run}.png", n_frames=6)
        print(f"[ok] {run}: {summary}")

    save_json(summaries, FIGURES_DIR / "loss_summaries.json")

    # Comparación directa baseline vs cada variante (mismo eje, dos paneles)
    for variant in ("loss_wgangp", "stab_spectral_norm"):
        if "baseline" not in summaries or variant not in summaries:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
        for ax, run in zip(axes, ("baseline", variant)):
            hist = load_json(EXPERIMENTS_DIR / run / "loss_history.json")
            its = [h["iter"] for h in hist]
            d = [h["d_loss"] for h in hist]
            g = [h["g_loss"] for h in hist]
            ax.plot(its, d, label="D", color="#d1495b", linewidth=1.0)
            ax.plot(its, g, label="G", color="#2e86ab", linewidth=1.0)
            ax.set_title(RUNS[run])
            ax.set_xlabel("Iteración")
            ax.grid(alpha=0.25)
            ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"comparison_baseline_vs_{variant}.png", dpi=150)
        plt.close(fig)
        print(f"[ok] comparación baseline vs {variant}")


if __name__ == "__main__":
    main()
