import torch
import numpy as np

# Augmentations taken/inspired from https://github.com/yanx27/Pointnet_Pointnet2_pytorch


def random_point_dropout(pcd, max_dropout_ratio=0.875):
    """Simulates missing points by cloning first point, keeping shape"""

    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pcd.shape[0])) <= dropout_ratio)[0]
    cloned_pcd = pcd.clone()
    if len(drop_idx) > 0:
        cloned_pcd[drop_idx, :] = pcd[0, :]  # set to the first point
    return cloned_pcd


def random_scale_point_cloud(pcd, scale_low=0.8, scale_high=1.25):
    """Randomly scale point cloud"""

    scale = np.random.uniform(scale_low, scale_high)
    return pcd * scale


def jitter_point_cloud(pcd, sigma=0.01, clip=0.05):
    """Randomly jitter each points"""

    assert clip > 0
    noise = torch.randn_like(pcd) * sigma
    noise = torch.clamp(noise, -clip, clip)
    return pcd + noise


def random_sample(pcd: torch.Tensor) -> torch.Tensor:
    """Randomly select half of all points in the point cloud, drop the rest"""

    choice = np.random.choice(pcd.shape[0], int(len(pcd) / 2), replace=False)
    return pcd[choice, :]
