import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


def plot_point_cloud(point_cloud: torch.Tensor, labels: torch.Tensor | None = None, title: str = "Point Cloud", custom_colors: Dict[int, List[int]] | None = None, size: tuple[int, int] = (15, 5)):
    """Plot a point cloud with optional color-coded labels in 3D and 2D (XZ view).

    Args:
        point_cloud (torch.Tensor): Nx3 tensor with (X, Y, Z) coordinates.
        labels (torch.Tensor, optional): Nx1 tensor with labels for color-coding. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "Point Cloud".
        custom_colors (dict, optional): Dictionary specifying label-color mappings. Defaults to None. Example: {7: [1, 0, 0]}
        size: (tuple, optional): A width x height tuple for the plot size dimensions. Defaults to (15, 5)
    """
    pcd_np = point_cloud.detach().cpu().numpy()

    # Default color is black for all points
    colors = np.full((pcd_np.shape[0], 3), (0, 0, 0), dtype=float)

    if labels is None and custom_colors is not None:
        raise ValueError("Cannot use custom colors with label parameter")

    if custom_colors is None and labels is not None:
        raise ValueError("Label parameter passed without specifying custom colors")

    if labels is not None and custom_colors is not None:
        labels_np = labels.cpu().numpy()

        for label, color in custom_colors.items():
            # Ensure color is a valid 3D RGB vector (numpy array or tuple)
            color = np.array(color)
            if isinstance(color, (np.ndarray, tuple)) and len(color) == 3:
                # Find points with the current label
                indices = np.where(labels_np == label)[0]
                # Map them to the specified custom color
                colors[indices] = color
            else:
                raise ValueError(f"Invalid color for label {label}: {color}. Custom color must be dictionary of shape [label: [r, g, b]].")

    fig = plt.figure(figsize=size)

    # 3D View
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.set_title(title)
    ax3d.scatter(pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2], s=1, c=colors)  # type: ignore

    # 2D Side View (XZ Plane)
    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.set_title(title + " (Side View)")
    ax2d.scatter(pcd_np[:, 0], pcd_np[:, 2], s=1, c=colors)
    ax2d.set_xlabel("X")
    ax2d.set_ylabel("Z")

    plt.tight_layout()
    plt.show()
