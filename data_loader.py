import os
import cv2
import torch
import random
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image


class SaliencyDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=224):
        super().__init__()
        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.img_size = img_size
        self.split = split

        self.image_paths = sorted(glob(os.path.join(self.img_dir, "*.*")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]

        # find matching mask (same base name)
        mask_candidates = glob(os.path.join(self.mask_dir, base + ".*"))
        if not mask_candidates:
            raise FileNotFoundError(f"No mask found for {base}")
        mask_path = mask_candidates[0]

        # read with OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)   # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # convert to PIL
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        # ----- Resize first -----
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # ----- Augment only for train -----
        if self.split == "train":
            # random horizontal flip
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # brightness / contrast jitter
            factor_b = 0.8 + 0.4 * random.random()   # [0.8, 1.2]
            factor_c = 0.8 + 0.4 * random.random()
            img = TF.adjust_brightness(img, factor_b)
            img = TF.adjust_contrast(img, factor_c)

        # ----- To tensor (0-1) -----
        img = TF.to_tensor(img)           # [3, H, W], float in [0,1]
        mask = TF.to_tensor(mask)         # [1, H, W], float in [0,1]
        mask = (mask > 0.5).float()       # binarize

        return img, mask


def create_dataloaders(root_dir, img_size=224, batch_size=8):
    train_ds = SaliencyDataset(root_dir, "train", img_size)
    val_ds   = SaliencyDataset(root_dir, "val",   img_size)
    test_ds  = SaliencyDataset(root_dir, "test",  img_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
