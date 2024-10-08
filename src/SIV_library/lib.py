from .matching import window_array, get_field_shape, block_match, get_x_y, correlation_to_displacement, WindowShift, process_velocity_fields
from .optical_flow import optical_flow
from collections.abc import Generator

import torch
from torch.utils.data import Dataset, DataLoader

import os
import cv2
from tqdm import tqdm


class SIVDataset(Dataset):
    def __init__(self, folder: str, transforms: list | None = None, device: str = "cpu") -> None:
        # assume the files have the correct file type
        filenames = [os.path.join(folder, name) for name in os.listdir(folder)]
        filenames.sort(key=lambda x: int(os.path.split(x)[-1].split('.')[0]))

        self.img_pairs = list(zip(filenames[:-1], filenames[1:]))
        self.idx = [int(os.path.split(x)[-1].split('.')[0]) for x in filenames]

        self.transforms = transforms
        self.device = device

        self.img_shape = cv2.imread(filenames[0], cv2.IMREAD_GRAYSCALE).shape

    def __len__(self) -> int:
        return len(self.img_pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.img_pairs[index]
        img_a, img_b = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE), cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)

        img_a = torch.tensor(img_a, dtype=torch.uint8, device=self.device)
        img_b = torch.tensor(img_b, dtype=torch.uint8, device=self.device)
        img_a, img_b = img_a[None, None, :, :], img_b[None, None, :, :]  # batch and channel dimension for transforms

        if self.transforms is not None:
            for transform in self.transforms:
                img_a = transform(img_a) if 'a' in transform.apply_to else img_a
                img_b = transform(img_b) if 'b' in transform.apply_to else img_b
        return img_a.squeeze(), img_b.squeeze()


class SIV:
    def __init__(self,
                 folder: str,
                 device: torch.device = "cpu",
                 window_size: int = 64,
                 overlap: int = 32,
                 search_area: tuple[int, int, int, int] = (0, 0, 0, 0),
                 mode: int = 1,
                 num_passes: int = 3,
                 scale_factor: float = 1/2
                 ) -> None:

        self.dataset = SIVDataset(folder=folder, device=device)
        self.device = device
        self.window_size, self.overlap, self.search_area = window_size, overlap, search_area
        self.mode, self.num_passes, self.scale_factor = mode, num_passes, scale_factor

    def __len__(self) -> int:
        return len(self.dataset)

    def __call__(self) -> Generator:
        loader = DataLoader(self.dataset)
        for a, b in tqdm(loader, total=len(loader), desc="SAD" if self.mode == 1 else "Correlation"):
            yield self.run(a, b)

    def run(self, a, b):
        scales = [self.scale_factor ** p for p in range(self.num_passes)]
        for i, scale in enumerate(scales):
            window_size, overlap = int(self.window_size * scale), int(self.overlap * scale)
            search_area = tuple(int(pad * scale) for pad in self.search_area)

            n_rows, n_cols = get_field_shape(self.dataset.img_shape, window_size, overlap)
            xp, yp = get_x_y(self.dataset.img_shape, window_size, overlap)
            xp, yp = xp.reshape(n_rows, n_cols).to(self.device), yp.reshape(n_rows, n_cols).to(self.device)

            if i == 0:
                window = window_array(a, window_size, overlap)
                area = window_array(b, window_size, overlap, area=search_area)
            else:
                shift = WindowShift(self.dataset.img_shape, window_size, overlap, search_area, self.device)
                window, area, up, vp = shift.run(a, b, xp, yp, up, vp)

            match = block_match(window, area, self.mode)
            du, dv = correlation_to_displacement(match, search_area, n_rows, n_cols, self.mode)

            up, vp = (du, dv) if i == 0 else (up + du, vp + dv)
            
            up, vp = process_velocity_fields(up, vp, std_threshold=2.0, window_size=3) # window_size should be odd. 3 seems to be nice. 5 leads to too much smoothing.
            
        return xp, yp, up, -vp


class OpticalFlow:
    def __init__(self,
                 folder: str = None,
                 device: torch.device = "cpu",
                 alpha: float = 1000.,
                 num_iter: int = 100,
                 eps: float = 1e-5,
                 grid_size_avg: int = 64,
                 ) -> None:

        self.folder = folder
        self.dataset = SIVDataset(folder=folder, device=device)
        self.device = device
        self.alpha, self.num_iter, self.eps = alpha, num_iter, eps
        self.grid_size_avg = grid_size_avg
        
    def __len__(self) -> int:
        return len(self.dataset)

    def __call__(self) -> Generator:
        rows, cols = self.dataset.img_shape
        y, x = torch.meshgrid(torch.arange(0, rows, 1), torch.arange(0, cols, 1))
        x, y = x.to(self.device), y.to(self.device)

        loader = DataLoader(self.dataset)
        for a, b in tqdm(loader, total=len(loader), desc='Optical flow'):
            xa, ya, du, dv = self.run(a, b)
            # yield x, y, du, -dv # Commented by Sagar.
            yield xa.to(self.device), ya.to(self.device), du, -dv

    def run(self, a, b):
        return optical_flow(a, b, self.alpha, self.num_iter, self.eps, self.grid_size_avg)
