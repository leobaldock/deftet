import os
import numpy as np
import torch
from torch.utils.data import Dataset
from voxelFileLoader import load_voxel
from meshFileLoader import load_mesh


class VoxelizedMeshDataset(Dataset):
    """Sample dataset of generated tetrahedral meshes and their voxelized representations."""

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Root directory containing voxel and mesh data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self._data_dir = data_dir
        self._transform = transform
        # self._data = []

        # for idx in range(len(os.listdir(os.path.join(self._data_dir, "mesh")))):
            # self._data.append(self.__getitem__(idx))
        
        # self._data = np.array(self._data)


    def __len__(self):
        return len(os.listdir(os.path.join(self._data_dir, "mesh")))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.toList()

        vox_pth = os.path.join(self._data_dir, "vox", f"vox{idx}.csv")
        vox = load_voxel(vox_pth, dtype=np.float32)
        # Add channel dimension.
        vox = vox[None, :]
        vox = torch.from_numpy(vox)
        

        mesh_pth = os.path.join(self._data_dir, "mesh", f"mesh{idx}.vtk")
        # This will load the mesh into a PyTorch Geometric Data object.
        mesh = load_mesh(mesh_pth)

        if self._transform:
            self._transform(vox, mesh)

        return (vox, mesh)