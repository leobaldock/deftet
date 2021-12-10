import numpy as np

def load_voxel(path, dtype=bool):
    return np.array(np.loadtxt(path), dtype=dtype).reshape((32, 32, 32))