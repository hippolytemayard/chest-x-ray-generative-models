from pathlib import Path

from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        folder: str,
        image_size: tuple,
        exts: list = ["jpg", "jpeg", "png", "tiff"],
        augment_horizontal_flip: bool = False,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f"{folder}").glob(f"**/*.{ext}")]

        self.transform = T.Compose(
            [
                T.Resize(image_size),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert("L")
        return self.transform(img)
