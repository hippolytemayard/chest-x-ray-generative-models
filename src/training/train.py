import logging

import torch

from src.models.denoising_diffusion import GaussianDiffusion
from src.models.unet import Unet
from src.settings import DATA_PATH
from src.training.trainer import Trainer

if __name__ == "main":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = 128

    logging.info("Unet model initializaton")
    model = Unet(dim=size, channels=1, dim_mults=(1, 2, 4, 8))

    logging.info("Gaussian diffusion model initializaton")
    diffusion = GaussianDiffusion(
        model,
        image_size=size,
        timesteps=1000,  # number of steps
        sampling_timesteps=250,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    logging.info("Trainer initializaton")
    trainer = Trainer(
        diffusion_model=diffusion,
        folder=DATA_PATH,
        train_batch_size=32,
        train_lr=8e-5,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        device="cuda",  # device
    )

    trainer.train()
