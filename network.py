import math
from typing import Tuple
import torch
from torch import Tensor
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data.batch import DynamicInheritance
from torch_geometric.data.data import Data
from torch_geometric.nn import ResGatedGraphConv

def interpolate_bilinear(x0, x1, y0, y1, p):
    return 


####################
# Responsive Encoder
####################
class ResponsiveEncoder(nn.Module):
    def __init__(self, in_size, out_size, in_channels, out_channels):
        super(ResponsiveEncoder, self).__init__()

        self._target_features = out_channels
        self._downsamples = int(math.log(in_size // out_size, 2))

        self._initial = nn.Sequential(
            nn.Conv3d(in_channels, self.feats(0), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.feats(0)),
            nn.ReLU(True)
        )

        self._blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(self.feats(i-1),
                          self.feats(i),
                          kernel_size=4,
                          stride=2, padding=1),
                nn.BatchNorm3d(self.feats(i)),
                nn.ReLU(True)
            ) for i in range(1, self._downsamples + 1)
        ])

        self._final = nn.Sequential(
            nn.Conv3d(self._target_features, self._target_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self._target_features),
        )

    def forward(self, x):
        y = self._initial(x)
        for l in self._blocks:
            y = l(y)
        y = self._final(y)

        return y

    def feats(self, i):
        return int(self._target_features*(2**(i-self._downsamples)))

####################

####################
# Positional Decoder
# 2x GCN layers
# 2x MLP layers
####################
# TODO use PyG sequential to condense this a bit.
class PosDecoder(nn.Module):
    def __init__(self):
        super(PosDecoder, self).__init__()

        self.gcn1 = ResGatedGraphConv(
            in_channels=-1,
            out_channels=256
        )

        self.gcn2 = ResGatedGraphConv(
            in_channels=256,
            out_channels=128
        )

        self.mlp1 = nn.Linear(128, 64)
        self.mlp2 = nn.Linear(64, 3)

    def forward(self, mesh_data):
        x = mesh_data.x
        edge_index = mesh_data.edge_index

        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = self.mlp1(x)
        x = F.relu(x)
        x = self.mlp2(x)
        x = F.relu(x)

        # Store back to Batch object. 
        mesh_data.x = x
        return mesh_data

####################

####################
# Occupancy Decoder
# 4x MLP layers
####################
# TODO use PyG sequential to condense this a bit.
class OccDecoder(nn.Module):
    def __init__(self):
        super(OccDecoder, self).__init__()

        self.mlp1 = nn.LazyLinear(256)
        self.mlp2 = nn.Linear(256, 128)
        self.mlp3 = nn.Linear(128, 64)
        self.mlp4 = nn.Linear(64, 3)

    def forward(self, mesh_data):
        pass

####################


class DefTet(nn.Module):
    def __init__(self, in_size, latent_size, in_channels, template_mesh):
        super(DefTet, self).__init__()
        self.latent_size = latent_size
        self.template_mesh = template_mesh
        print(template_mesh.tetra)
        
        self.pos_encoder = ResponsiveEncoder(
            in_size=in_size,
            out_size=latent_size,
            in_channels=in_channels,
            out_channels=512,
        )

        self.occ_encoder = ResponsiveEncoder(
            in_size=in_size,
            out_size=latent_size,
            in_channels=in_channels,
            out_channels=512
        )

        self.pos_decoder = PosDecoder()

        self.occ_decoder = OccDecoder()

    def forward(self, x):
        # Positional Encoder
        pos_features = self.pos_encoder(x)

        # Occupancy Encoder
        occ_features = self.occ_encoder(x)

        # TODO there's got to be some way to do this more efficiently.
        # Make a template mesh for each graph in the batch.
        meshes = []
        for gi in range(x.shape[0]):
            pos = self.template_mesh.pos

            # Scale all the node positions to match the size of the feature map.
            scaled_pos = torch.mul(pos, self.latent_size)
            
            # Create a new tensor to hold the new node features.
            node_features = torch.zeros([pos.shape[0], 515], dtype=torch.float32)
            for ni, node in enumerate(scaled_pos):
                # Find the nodes position in the feature volume. TODO Ling et. al. use bilinear interpoliation here
                # I should try to use trilinear.
                fi = pos_features[gi][:, int(node[0]-1.0), int(node[1]-1.0), int(node[2]-1.0)]
                node_features[ni] = torch.cat((pos[ni], fi))

            # Make a list of the meshes.
            meshes.append(Data(node_features, self.template_mesh.edge_index))
            
        # Create a batch of template meshes the same size as the input batch.
        mesh_batch = Batch.from_data_list(meshes)

        # Positional Decoder (2-layer GCN, 2-layer MLP)
        dec_pos = self.pos_decoder(mesh_batch)

        print(dec_pos)

        # Occupancy Decoder (4-layer MLP)

        return None