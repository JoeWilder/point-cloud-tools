from torch.utils.data import Dataset
from typing import List, Dict, Callable
import torch
import os
from tqdm import tqdm
import pickle
import shutil

from src.utilities.file_utils import download, unzip
from src.utilities.processing import sample_point_cloud_from_off


class ModelNet40(Dataset):
    def __init__(self, modelnet_dir: str, split: str = "train", num_points: int = 2048, augmenter: Callable | None = None):

        self.modelnet_dir = modelnet_dir
        assert split in {"train", "test"}
        self.split = split
        self.modelnet_cache_path = f"{modelnet_dir}/modelnet40_{self.split}.pkl"

        self.num_sampled_points = num_points
        self.pcd_data: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.class_map: Dict[int, str] = {}

        self.augmenter = augmenter

        self.initialize_dataset()

    def _get_class_mapping(self) -> Dict[int, str]:
        """Assign an ID to each class"""
        mapping = {}
        if os.path.exists(f"{self.modelnet_dir}/ModelNet40"):
            for i, dir in enumerate(sorted(os.listdir(f"{self.modelnet_dir}/ModelNet40"))):
                mapping[i] = dir
        return mapping

    def initialize_dataset(self):
        """Load dataset, initializing data if necessary"""
        if not os.path.exists(self.modelnet_cache_path):
            if os.path.exists(self.modelnet_dir) and len(os.listdir(self.modelnet_dir)) > 0:
                raise OSError(f"ModelNet directory {self.modelnet_dir} already exists and is not empty")
            os.makedirs(self.modelnet_dir, exist_ok=True)

            self.download_modelnet40(self.modelnet_dir)
            self.parse_data()
        self.load()

    def download_modelnet40(self, modelnet_dir: str):
        print("Downloading ModelNet40 dataset...")
        download("http://modelnet.cs.princeton.edu/ModelNet40.zip", modelnet_dir)
        print("Unzipping ModelNet...")
        unzip(f"{modelnet_dir}/ModelNet40.zip", modelnet_dir)
        print("Successfully downloaded ModelNet40 dataset")

    def parse_data(self):
        """Parse training and test data from dataset files, and save to disk"""
        class_map: Dict[int, str] = self._get_class_mapping()
        print("Converting ModelNet40 to point clouds and saving cache...")
        for split in ["train", "test"]:
            total_files = sum(len(os.listdir(f"{self.modelnet_dir}/ModelNet40/{class_name}/{split}")) for class_name in class_map.values())
            with tqdm(total=total_files, desc="Loading point clouds") as pbar:
                for class_id, class_name in class_map.items():
                    class_directory = f"{self.modelnet_dir}/ModelNet40/{class_name}/{split}"
                    for filename in os.listdir(class_directory):
                        path = f"{class_directory}/{filename}"
                        self.pcd_data.append(sample_point_cloud_from_off(path, n_points=self.num_sampled_points))
                        self.labels.append(class_id)
                        pbar.update(1)
            self.save(split, class_map)
            self.pcd_data.clear()
            self.labels.clear()
        shutil.rmtree(f"{self.modelnet_dir}/ModelNet40")

    def save(self, split, class_map):
        """Save data to pickle cache"""
        with open(f"{self.modelnet_dir}/modelnet40_{split}.pkl", "wb") as f:
            pickle.dump((self.pcd_data, self.labels, class_map), f)

    def load(self):
        """Load data from pickle cache"""
        with open(self.modelnet_cache_path, "rb") as f:
            self.pcd_data, self.labels, self.class_map = pickle.load(f)

    def __len__(self) -> int:
        return len(self.pcd_data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        pcd = self.pcd_data[index]
        if self.augmenter is not None:
            pcd = self.augmenter(pcd)

        return {
            "pcd": pcd,
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }
