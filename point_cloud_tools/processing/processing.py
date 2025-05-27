import torch
import numpy as np
import trimesh


def unit_sphere_normalize(pcd: torch.Tensor) -> torch.Tensor:
    """Normalize point cloud so that it is centered within a unit sphere"""

    if pcd.ndim != 2 or pcd.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {pcd.shape}")

    centroid = pcd.mean(dim=0)
    pcd = pcd - centroid

    furthest_distance = torch.norm(pcd, dim=1).max()
    if furthest_distance == 0:
        return pcd  # Already at origin

    pcd = pcd / furthest_distance
    return pcd


def tanh_normalize(point_cloud: torch.Tensor) -> torch.Tensor:
    """Normalize each axis of the point cloud independently to be within the range [-1, 1]."""
    min_vals = torch.min(point_cloud, dim=0)[0]
    max_vals = torch.max(point_cloud, dim=0)[0]

    center = (min_vals + max_vals) / 2
    scale = (max_vals - min_vals) / 2

    # Normalize each axis separately
    normalized_point_cloud = (point_cloud - center) / scale
    return normalized_point_cloud


def sigmoid_normalize(tensor_pcd) -> torch.Tensor:
    """Normalize point cloud to [0,1] range per axis to prevent scaling issues."""
    min_vals = torch.min(tensor_pcd, dim=0)[0]
    max_vals = torch.max(tensor_pcd, dim=0)[0]
    scales = max_vals - min_vals
    scales[scales == 0] = 1  # Prevent division by zero
    return (tensor_pcd - min_vals) / scales


def label_smoothing(one_hot_labels: list, epsilon: float = 0.1):
    """Apply label smoothing to a one hot label vector"""
    one_hot_labels = np.array(one_hot_labels)
    return (1 - epsilon) * one_hot_labels + epsilon / len(one_hot_labels)


def sample_point_cloud_from_off(path: str, n_points: int = 2048):
    """Convert from off mesh file to N x 3 point cloud tensor"""
    mesh = trimesh.load(path, process=False)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return torch.tensor(points, dtype=torch.float32)
