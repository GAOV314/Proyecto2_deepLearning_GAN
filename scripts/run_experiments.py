"""Corre los tres entrenamientos del proyecto en orden: base, pérdida (WGAN-GP)
y estabilización (normalización espectral). Guarda logs en experiments/logs/.
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ganverse.train import PRESETS, train  # noqa: E402

DATA_NPY = REPO_ROOT / "data" / "glimmerfen" / "glimmerfen_64x64.npy"
LOG_DIR = REPO_ROOT / "experiments" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

ORDER = ["baseline", "stab_spectral_norm", "loss_wgangp"]

HYPOTHESES = {
    "baseline": "Configuración base: DCGAN con BCE no saturante y BatchNorm en D.",
    "stab_spectral_norm": (
        "Creemos que la normalizacion espectral en D mejorara la estabilidad "
        "(curvas menos oscilantes, menos riesgo de que D domine) porque limita "
        "la constante de Lipschitz del discriminador capa por capa. "
        "Lo consideraremos util si las perdidas de D y G se mantienen mas "
        "acotadas y la rejilla de ruido fijo no colapsa a pocas variantes."
    ),
    "loss_wgangp": (
        "Creemos que WGAN-GP mejorara la estabilidad del entrenamiento porque "
        "aproxima la distancia de Wasserstein en vez de una divergencia JS que "
        "satura facilmente, y la penalizacion de gradiente evita los gradientes "
        "explosivos del recorte de pesos de WGAN. "
        "Lo consideraremos util si la perdida del critico decrece de forma mas "
        "monotona (sin el ruido brusco de BCE) y la diversidad de la rejilla "
        "de ruido fijo se mantiene a lo largo del entrenamiento."
    ),
}

if __name__ == "__main__":
    for name in ORDER:
        cfg = PRESETS[name]
        print(f"\n=== Entrenando preset '{name}' ===")
        print(HYPOTHESES[name])
        t0 = time.time()
        log_path = LOG_DIR / f"{name}.log"
        with open(log_path, "w") as logf:
            import contextlib
            import io

            class Tee(io.TextIOBase):
                def write(self, s):
                    sys.__stdout__.write(s)
                    logf.write(s)
                    logf.flush()
                    return len(s)

            with contextlib.redirect_stdout(Tee()):
                run_dir = train(cfg, data_npy=DATA_NPY, verbose=True)
        elapsed = time.time() - t0
        print(f"=== '{name}' listo en {elapsed/60:.1f} min -> {run_dir} ===")
    print("\nTodos los experimentos terminaron.")
