"""
PIV method based on
    "Extracting turbulence parameters of smoke via video analysis" (2021), https://doi.org/10.1063/5.0059326
Horn-Schunck optical flow (https://en.wikipedia.org/wiki/Horn-Schunck_method)

Torch adaptation of pure python implementation
    https://github.com/scivision/Optical-Flow-LucasKanade-HornSchunck/blob/master/HornSchunck.py
"""

import torch
from torch.nn.functional import conv2d, pad, avg_pool2d
from tqdm import tqdm
import numpy as np


def pad_to_multiple(tensor, multiple):
    h, w = tensor.shape[-2:]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    return pad(tensor, padding, mode='reflect')

def optical_flow(img1, img2, alpha, num_iter, eps, grid_size):
    device = img1.device

    a, b = img1[None, :, :, :].float(), img2[None, :, :, :].float()
    u, v = torch.zeros_like(a), torch.zeros_like(a)

    Ix, Iy, It = compute_derivatives(a, b)

    avg_kernel = torch.tensor([[[[1/12, 1/6, 1/12],
                                 [1/6, 0, 1/6],
                                 [1/12, 1/6, 1/12]]]], dtype=torch.float32, device=device)

    for i in range(num_iter):
        u_avg = conv2d(u, avg_kernel, padding=1)
        v_avg = conv2d(v, avg_kernel, padding=1)

        der = (Ix * u_avg + Iy * v_avg + It) / (alpha ** 2 + Ix ** 2 + Iy ** 2)

        u_new = u_avg - Ix * der
        v_new = v_avg - Iy * der

        # MSE early stopping https://www.ipol.im/pub/art/2013/20/article.pdf
        delta = torch.sum((u_new - u) ** 2) + torch.sum((v_new - v) ** 2)
        delta /= a.shape[-2] * a.shape[-1]
        
        # print(f'iter = {i}, delta = {delta}')

        if eps is not None and delta < eps:
            break

        u, v = u_new, v_new
        
    # Sagar start.
    u_padded = pad_to_multiple(u, grid_size)
    v_padded = pad_to_multiple(v, grid_size)
    
    u_avg = avg_pool2d(u_padded, kernel_size=grid_size)
    v_avg = avg_pool2d(v_padded, kernel_size=grid_size)
    
    _, _ ,h, w = u_avg.shape
    ya, xa = torch.meshgrid(torch.arange(grid_size // 2, h * grid_size, grid_size), torch.arange(grid_size // 2, w * grid_size, grid_size))
    
    # print(f"Shape of u is {u.shape}")
    # print(f"Shape of u_avg is {u_avg.shape}")
    # print(f"Shape of xa is {xa.shape}")
    # print(f"Shape of ya is {ya.shape}")
    # Sagar end.
        
    return xa, ya, u_avg.squeeze(), v_avg.squeeze()


def compute_derivatives(img1: torch.Tensor, img2: torch.Tensor):
    device = img1.device

    kernel_x = torch.tensor([[[[-1 / 4, 1 / 4],
                               [-1 / 4, 1 / 4]]]], dtype=torch.float32, device=device)
    kernel_y = torch.tensor([[[[-1 / 4, -1 / 4],
                               [1 / 4, 1 / 4]]]], dtype=torch.float32, device=device)
    kernel_t = torch.ones((1, 1, 2, 2), dtype=torch.float32, device=device) / 4

    padding = (0, 1, 0, 1)  # add a column right and a row at the bottom
    img1, img2 = pad(img1, padding), pad(img2, padding)

    fx = conv2d(img1, kernel_x) + conv2d(img2, kernel_x)
    fy = conv2d(img1, kernel_y) + conv2d(img2, kernel_y)
    ft = conv2d(img1, -kernel_t) + conv2d(img2, kernel_t)

    return fx, fy, ft