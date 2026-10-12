"""Bucle de entrenamiento de la GAN, parametrizado para correr los tres
experimentos del proyecto (base, pérdida, estabilización) cambiando una sola
variable a la vez respecto a `PRESETS["baseline"]`.

Uso:
    python -m ganverse.train --preset baseline
    python -m ganverse.train --preset loss_wgangp
    python -m ganverse.train --preset stab_spectral_norm
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import GlimmerfenDataset
from .models import Discriminator, Generator
from .losses import (
    bce_discriminator_loss, bce_generator_loss,
    gradient_penalty, wgan_gp_discriminator_loss, wgan_gp_generator_loss,
)
from .utils import get_device, save_image_grid, save_json, set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_NPY = REPO_ROOT / "data" / "glimmerfen" / "glimmerfen_64x64.npy"
FIXED_NOISE_PATH = REPO_ROOT / "experiments" / "fixed_eval_noise.pt"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


@dataclass
class TrainConfig:
    name: str
    loss_type: str = "bce"            # "bce" | "wgan_gp"
    normalization: str = "batch"      # "batch" | "spectral" | "none"
    epochs: int = 45
    batch_size: int = 128
    latent_dim: int = 100
    feature_maps: int = 64
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: tuple = (0.5, 0.999)
    n_critic: int = 1
    lambda_gp: float = 10.0
    seed: int = 2026
    fixed_noise_n: int = 16
    snapshot_every: int = 3


# Config base (DCGAN clásico, BCE no saturante, BatchNorm, Adam 2e-4/0.5,0.999).
# Los otros dos presets cambian UNA sola variable respecto a esta base, con la
# excepción documentada de los ajustes que exige la propia literatura de WGAN-GP
# (ver PROJECT.md, sección "Experimentos").
PRESETS: dict[str, TrainConfig] = {
    "baseline": TrainConfig(name="baseline", loss_type="bce", normalization="batch"),
    "loss_wgangp": TrainConfig(
        name="loss_wgangp", loss_type="wgan_gp", normalization="none",
        lr_g=1e-4, lr_d=1e-4, betas=(0.0, 0.9), n_critic=5,
    ),
    "stab_spectral_norm": TrainConfig(name="stab_spectral_norm", loss_type="bce", normalization="spectral"),
}


def get_or_create_fixed_noise(latent_dim: int, n: int, seed: int = 7,
                               path: Path = FIXED_NOISE_PATH) -> torch.Tensor:
    """Un único lote de ruido fijo compartido por TODOS los experimentos, para que
    la "rejilla de ruido fijo" (evidencia #1) sea comparable entre corridas."""
    if path.exists():
        noise = torch.load(path)
        if noise.shape == (n, latent_dim):
            return noise
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(n, latent_dim, generator=gen)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(noise, path)
    return noise


def train(cfg: TrainConfig, data_npy: Path = DATA_NPY, device: torch.device | None = None,
          verbose: bool = True, experiments_dir: Path = EXPERIMENTS_DIR,
          fixed_noise_path: Path | None = None) -> Path:
    set_seed(cfg.seed)
    device = device or get_device()
    run_dir = experiments_dir / cfg.name
    snapshots_dir = run_dir / "snapshots"
    if snapshots_dir.exists():  # evita mezclar snapshots de una corrida interrumpida previa
        for old_file in snapshots_dir.glob("epoch_*.png"):
            old_file.unlink()
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    fixed_noise_path = fixed_noise_path or (experiments_dir / "fixed_eval_noise.pt")

    dataset = GlimmerfenDataset(data_npy, augment=True)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    G = Generator(cfg.latent_dim, cfg.feature_maps).to(device)
    D = Discriminator(cfg.feature_maps, normalization=cfg.normalization).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=cfg.betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=cfg.betas)

    fixed_noise = get_or_create_fixed_noise(cfg.latent_dim, cfg.fixed_noise_n,
                                             path=fixed_noise_path).to(device)

    history: list[dict] = []
    t0 = time.time()
    it = 0
    for epoch in range(cfg.epochs):
        for real in loader:
            real = real.to(device)
            bs = real.size(0)

            # --- paso(s) del discriminador / crítico ---
            d_loss_val = 0.0
            n_steps = cfg.n_critic if cfg.loss_type == "wgan_gp" else 1
            for _ in range(n_steps):
                z = torch.randn(bs, cfg.latent_dim, device=device)
                fake = G(z).detach()
                d_real = D(real)
                d_fake = D(fake)
                if cfg.loss_type == "bce":
                    d_loss = bce_discriminator_loss(d_real, d_fake)
                else:
                    gp = gradient_penalty(D, real, fake, device)
                    d_loss = wgan_gp_discriminator_loss(d_real, d_fake, gp, cfg.lambda_gp)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
                d_loss_val = d_loss.item()

            # --- paso del generador ---
            z = torch.randn(bs, cfg.latent_dim, device=device)
            fake = G(z)
            d_fake_for_g = D(fake)
            g_loss = (bce_generator_loss(d_fake_for_g) if cfg.loss_type == "bce"
                      else wgan_gp_generator_loss(d_fake_for_g))
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            history.append({"epoch": epoch, "iter": it, "d_loss": d_loss_val, "g_loss": g_loss.item()})
            it += 1

        if epoch % cfg.snapshot_every == 0 or epoch == cfg.epochs - 1:
            G.eval()
            with torch.no_grad():
                snap = G(fixed_noise).cpu()
            save_image_grid(snap, run_dir / "snapshots" / f"epoch_{epoch:04d}.png", nrow=4)
            G.train()
            if verbose:
                elapsed = time.time() - t0
                print(f"[{cfg.name}] epoch {epoch:4d}/{cfg.epochs} "
                      f"d_loss={d_loss_val:.3f} g_loss={g_loss.item():.3f} ({elapsed:.1f}s)")

    torch.save({"G": G.state_dict(), "D": D.state_dict(), "config": asdict(cfg)},
               run_dir / "checkpoint.pt")
    save_json(history, run_dir / "loss_history.json")
    save_json(asdict(cfg), run_dir / "config.json")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Entrena una variante de la GAN de Glimmerfen.")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="baseline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data", type=str, default=str(DATA_NPY))
    args = parser.parse_args()

    cfg = PRESETS[args.preset]
    if args.epochs:
        cfg.epochs = args.epochs
    run_dir = train(cfg, data_npy=Path(args.data))
    print(f"Listo. Checkpoint y evidencias en: {run_dir}")


if __name__ == "__main__":
    main()
