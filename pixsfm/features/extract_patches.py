from packaging import version
import numpy as np
import torch


def numpy_get_patch(arr, corner, ps):
    if len(arr.shape) == 4:
        return arr[0, corner[1]:corner[1]+ps, corner[0]:corner[0]+ps, :]
    elif len(arr.shape) == 3:
        return arr[corner[1]:corner[1]+ps, corner[0]:corner[0]+ps, :]


@torch.no_grad()
def extract_patches_torch(
        tensor: torch.Tensor,
        required_corners_np: np.ndarray,
        ps: int) -> torch.Tensor:
    c, h, w = tensor.shape
    corner = torch.from_numpy(required_corners_np).long().to(tensor.device)
    # clamping to image range
    corner = torch.min(corner, corner.new_tensor([w, h]) - ps - 1).clamp(min=0)
    offset = torch.arange(0, ps)

    kw = {}
    if version.parse(torch.__version__) >= version.parse('1.10'):
        kw["indexing"] = "ij"
    x, y = torch.meshgrid(offset, offset, **kw)
    patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2)
    patches = patches.to(corner) + corner[None, None]

    pts = patches.reshape(-1, 2)
    sampled = tensor.permute(1, 2, 0)[tuple(pts.T)[::-1]]
    sampled = sampled.reshape(ps, ps, -1, c)
    assert sampled.shape[:3] == patches.shape[:3]
    return sampled.permute(2, 3, 1, 0)


def extract_patches_numpy(
        tensor: torch.Tensor,
        required_corners_np: np.ndarray,
        ps: int) -> np.ndarray:
    patches_torch = extract_patches_torch(tensor, required_corners_np, ps)
    # @TODO: GPU->CPU remains main performance bottleneck!
    return np.ascontiguousarray(
        patches_torch.permute(0, 3, 2, 1).cpu().numpy()
    )
