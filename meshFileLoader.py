"""
The functionality in this file is currently under review for a PR to the PyTorch Geometric library.
Author: Isuru Wijesinghe
PR: https://github.com/pyg-team/pytorch_geometric/pull/2808
"""

import numpy as np
import meshio
import torch
import torch_geometric as pyg
from torch_geometric.utils import to_undirected

def load_mesh(path):
    # Load the mesh with meshio
    mesh = meshio.read(path)
    # Convert to pytorch geometric graph representation
    mesh_data = from_meshio(mesh)
    # Convert tetra data to edge data
    transform = TetraToEdge(remove_tetras=False)
    return transform(mesh_data)


def to_meshio(data):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`tetrahedral .msh/.vtk format`.
    Args:
        data (torch_geometric.data.Data): The data object.
    """
    points = data.pos.detach().cpu().numpy()
    tetra = data.tetra.detach().t().cpu().numpy()

    cells = [("tetra", tetra)]

    return meshio.Mesh(points, cells)


def from_meshio(mesh):
    r"""Converts a :tetrahedral .msh/.vtk file to a
    :class:`torch_geometric.data.Data` instance.
    Args:
        mesh (meshio.read): A :obj:`meshio` mesh.
    """

    pos = torch.from_numpy(mesh.points.astype(np.float32)).to(torch.float)
    tetra = torch.from_numpy(mesh.cells_dict['tetra'].astype(np.int64)).to(
        torch.long)

    return pyg.data.Data(pos=pos, tetra=tetra)

class TetraToEdge(object):
    r"""Converts mesh tetras :obj:`[num_tetras, 4]` to edge indices
    :obj:`[2, num_edges]`.
    Args:
        remove_tetras (bool, optional): If set to :obj:`False`,the tetra tensor
            will not be removed.
    """

    def __init__(self, remove_tetras=True):
        self.remove_tetras = remove_tetras

    def __call__(self, data):
        if data.tetra is not None:
            tetra = data.tetra.t().contiguous()
            edge_index = torch.cat([tetra[:2], tetra[1:3, :], tetra[-2:],
                                    tetra[::2], tetra[::3], tetra[1::2]],
                                   dim=1)
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

            data.edge_index = edge_index
            if self.remove_tetras:
                data.tetra = None

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)