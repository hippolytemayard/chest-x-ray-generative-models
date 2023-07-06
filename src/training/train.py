import logging

import torch

from src.config.settings import CONFIG_PATH, DATA_PATH
from src.models.denoising_diffusion import GaussianDiffusion
from src.models.unet import Unet
from src.training.trainer import Trainer
from src.utils.files import load_omegaconf_from_yaml

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_omegaconf_from_yaml(path=CONFIG_PATH)

    logging.info("Unet model initializaton")
    model = Unet(dim=config.ddpm.size, channels=config.ddpm.unet.channels, dim_mults=config.ddpm.unet.dim_mults)

    logging.info("Gaussian diffusion model initializaton")
    diffusion = GaussianDiffusion(
        model,
        image_size=config.ddpm.size,
        timesteps=config.ddpm.gaussian_diffusion.timesteps,
        sampling_timesteps=config.ddpm.gaussian_diffusion.sampling_timesteps,
    )

    logging.info("Trainer initializaton")
    trainer = Trainer(
        diffusion_model=diffusion,
        folder=DATA_PATH,
        train_batch_size=config.ddpm.trainer.train_batch_size,
        train_lr=config.ddpm.trainer.train_lr,
        train_num_steps=config.ddpm.trainer.train_num_steps,
        gradient_accumulate_every=config.ddpm.trainer.gradient_accumulate_every,
        device=device,
    )

    trainer.train()
