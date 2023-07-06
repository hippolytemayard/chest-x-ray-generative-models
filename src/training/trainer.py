import logging
import math
from multiprocessing import cpu_count
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import utils
from tqdm import tqdm

from src.datasets.dataset import ChestXrayDataset
from src.utils.helpers import cycle, has_int_squareroot, num_to_groups


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        device,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_horizontal_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder="./results",
        convert_image_to=None,
        max_grad_norm=1.0,
    ):
        super().__init__()

        # model
        self.device = device

        self.model = diffusion_model.to(self.device)
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        logging.info("Initiating Dataset and DataLoader")

        self.ds = ChestXrayDataset(folder, self.image_size, augment_horizontal_flip=augment_horizontal_flip)
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())

        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

    # @property
    # def device(self):
    #    return self.device

    def save(self, milestone):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": None,
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone):
        device = self.device

        data = torch.load(str(self.results_folder / f"model-{milestone}.pt"), map_location=device)

        model = self.model
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])

        if "version" in data:
            print(f"loading from version {data['version']}")

    def train(self):
        device = self.device

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    loss = self.model(data)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()

                    loss.backward()

                pbar.set_description(f"loss: {total_loss:.4f}")

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    with torch.inference_mode():
                        milestone = self.step // self.save_and_sample_every
                        batches = num_to_groups(self.num_samples, self.batch_size)
                        all_images_list = list(map(lambda n: self.model.sample(batch_size=n), batches))

                    all_images = torch.cat(all_images_list, dim=0)

                    utils.save_image(
                        all_images,
                        str(self.results_folder / f"sample-{milestone}.png"),
                        nrow=int(math.sqrt(self.num_samples)),
                    )

                    self.save(milestone)

                pbar.update(1)

        print("training complete")
