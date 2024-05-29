from torch.nn.functional import conv2d, pad
import torch
import numpy as np

from tqdm import tqdm
import cv2


def block_match(windows: torch.Tensor, areas: torch.Tensor, mode: int) -> torch.Tensor:
    windows, areas = windows.float(), areas.float()
    (count, window_rows, window_cols), (area_rows, area_cols) = windows.shape, areas.shape[-2:]

    res = torch.zeros((count, area_rows - window_rows + 1, area_cols - window_cols + 1))

    if mode == 0:  # correlation mode
        for idx, (window, area) in tqdm(enumerate(zip(windows, areas)), total=count, desc='Correlation'):
            corr = conv2d(area.unsqueeze(0).unsqueeze(0), window.unsqueeze(0).unsqueeze(0), stride=1)
            res[idx] = corr

    elif mode == 1:  # SAD mode
        for j in tqdm(range(area_rows - window_rows + 1), desc='SAD'):
            for i in range(area_cols - window_cols + 1):
                ref = areas[:, j:j + window_rows, i:i + window_cols]
                res[:, j, i] = torch.sum(torch.abs(windows - ref), dim=(1, 2))

    else:
        raise ValueError("Only mode 0 (correlation) or 1 (SAD) are supported")

    # normalized output
    return res / (window_rows * window_cols)


def window_array(array: torch.Tensor, window_size, overlap) -> torch.Tensor:
    shape = array.shape
    strides = (shape[-1] * (window_size - overlap), (window_size - overlap), shape[-1], 1)

    shape = (int((shape[-2] - window_size) / (window_size - overlap)) + 1,
             int((shape[-1] - window_size) / (window_size - overlap)) + 1,
             window_size, window_size)

    return (torch.as_strided(array, size=shape, stride=strides)
            .reshape(-1, window_size, window_size))


def search_array(array: torch.Tensor, window_size, overlap,
                 area: tuple[int, int, int, int] | None = None,
                 offsets: torch.Tensor | None = None) -> torch.Tensor:
    iters = get_field_shape(array.shape, window_size, overlap)

    dx, dy = torch.round(offsets[0]).int(), torch.round(offsets[1]).int()
    dx_max, dy_max = torch.max(torch.abs(dx)).item(), torch.max(torch.abs(dy)).item()

    left, right, top, bottom = area
    padding = (left + dx_max, right + dx_max, top + dy_max, bottom + dy_max)
    array = pad(array, padding)

    areas = torch.zeros((iters[0]*iters[1], window_size+top+bottom, window_size+left+right), dtype=torch.uint8)

    for j in range(iters[0]):
        for i in range(iters[1]):
            offset_x, offset_y = int(offsets[0, j, i]), int(offsets[1, j, i])
            area = array[j*overlap+offset_y+dy_max:j*overlap+window_size+top+bottom+offset_y+dy_max,
                         i*overlap+offset_x+dx_max:i*overlap+window_size+left+right+offset_x+dx_max]
            areas[i+j*iters[1], :, :] = area

            if i + j*iters[1] == 1541:
                print(offset_x, offset_y)

                ref = array.clone().numpy()
                start1 = (i*overlap+offset_x+dx_max,
                          j*overlap+offset_y+dy_max)
                end1 = (i*overlap+window_size+left+right+offset_x+dx_max,
                        j*overlap+window_size+top+bottom+offset_y+dy_max)
                ref = cv2.rectangle(ref, start1, end1, (255, 255, 255), 3)

                start2 = (i * overlap + dx_max + left,
                          j * overlap + dy_max + top)
                end2 = (i * overlap + window_size + dx_max + right,
                        j * overlap + window_size + dy_max + bottom)
                ref = cv2.rectangle(ref, start2, end2, (255, 255, 255), 3)

                ref = cv2.resize(ref, (500, 500))

                cv2.imshow(f'{array.shape}', ref)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    return areas


def correlation_to_displacement(corr: torch.Tensor, n_rows, n_cols, mode: int = 0):
    c, rows, cols = corr.shape

    eps = 1e-7
    corr += eps

    if mode == 0:
        m = corr.view(c, -1).argmax(-1, keepdim=True)  # correlation: argmax
    elif mode == 1:
        m = corr.view(c, -1).argmin(-1, keepdim=True)  # SAD: argmin
    else:
        raise ValueError("Mode must be either 0 or 1")

    row, col = torch.floor_divide(m, cols), torch.remainder(m, cols)
    neighbors = torch.zeros(c, 3, 3)

    no_displacements = torch.zeros(c, dtype=torch.bool)
    edge_cases = torch.zeros(c, dtype=torch.bool)

    for idx, field in enumerate(corr):
        row_idx, col_idx = row[idx].item(), col[idx].item()
        peak_val = torch.max(field) if mode == 0 else torch.min(field)

        # if multiple peak vals exist (e.g. flat field) the displacement is undetermined (set to 0)
        if rows * cols - torch.count_nonzero(field - peak_val) > 1:
            no_displacements[idx] = True
            continue

        # if peak is at the edge, mask as edge case (no interpolation will be considered)
        if row_idx in [0, rows-1] or col_idx in [0, cols-1]:
            edge_cases[idx] = True
            continue

        neighbors[idx] = field.clone()[row_idx-1:row_idx+2, col_idx-1:col_idx+2]

    # Gaussian interpolation for correlation
    if mode == 0:
        ct, cb, cl, cr, cm = (neighbors[:, 0, 1], neighbors[:, 2, 1], neighbors[:, 1, 0],
                              neighbors[:, 1, 2], neighbors[:, 1, 1])

        s_x = (torch.log(cl) - torch.log(cr)) / (2 * (torch.log(cl) + torch.log(cr)) - 4 * torch.log(cm))
        s_y = (torch.log(cb) - torch.log(ct)) / (2 * (torch.log(cb) + torch.log(ct)) - 4 * torch.log(cm))

        s_x[edge_cases], s_y[edge_cases] = 0., 0.

    # Polynomial interpolation for SAD
    if mode == 1:
        xx = torch.tensor([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]], dtype=torch.float32)
        yy = torch.tensor([[1, 1, 1],
                           [0, 0, 0],
                           [-1, -1, -1]], dtype=torch.float32)
        x, y = xx.flatten(), yy.flatten()

        # design matrix (https://www.youtube.com/watch?v=9Zve4NFBbSM)
        A = torch.stack((torch.ones_like(x), x, y, x*y, x**2, y**2)).T
        A = A.unsqueeze(0).repeat(c, 1, 1)

        B = neighbors.flatten(start_dim=1)
        res = torch.linalg.lstsq(A, B)
        coefs = res.solution

        a0, a1, a2, a3, a4, a5 = coefs[:, 0], coefs[:, 1], coefs[:, 2], coefs[:, 3], coefs[:, 4], coefs[:, 5]
        a = torch.stack([torch.stack([2*a4, a3], dim=1), torch.stack([a3, 2*a5], dim=1)], dim=2)

        ranks = torch.linalg.matrix_rank(a)
        singular = torch.where(ranks != 2, True, False)

        min_locs = torch.zeros([c, 2])

        b = torch.stack([-a1, -a2], dim=1)
        min_locs[~singular] = torch.linalg.solve(a[~singular], b[~singular])

        s_x, s_y = min_locs[:, 0], min_locs[:, 1]

        # cases where interpolation fails: override with s = 0.
        s_x = torch.where(torch.abs(s_x) >= 1., 0., s_x)
        s_y = torch.where(torch.abs(s_y) >= 1., 0., s_y)
        s_x[edge_cases], s_y[edge_cases] = 0., 0.

    m2d = torch.cat((m // rows, m % cols), -1)

    u = m2d[:, 1][:, None] + s_x[:, None]
    v = m2d[:, 0][:, None] + s_y[:, None]

    default_peak_position = corr.shape[-2:]
    v = v - int(default_peak_position[0] / 2)
    u = u - int(default_peak_position[1] / 2)

    u[no_displacements], v[no_displacements] = torch.nan, torch.nan

    torch.nan_to_num_(v)
    torch.nan_to_num_(u)

    u = u.reshape(n_rows, n_cols)
    v = v.reshape(n_rows, n_cols)

    return u, v


def get_field_shape(image_size, search_area_size, overlap):
    field_shape = (np.array(image_size) - search_area_size) // (search_area_size - overlap) + 1
    return field_shape


def get_x_y(image_size, search_area_size, overlap):
    shape = get_field_shape(image_size, search_area_size, overlap)
    x_single = np.arange(shape[1], dtype=int) * (search_area_size - overlap) + search_area_size // 2
    y_single = np.arange(shape[0], dtype=int) * (search_area_size - overlap) + search_area_size // 2
    x = np.tile(x_single, shape[0])
    y = np.tile(y_single.reshape((shape[0], 1)), shape[1]).flatten()
    return x, y
