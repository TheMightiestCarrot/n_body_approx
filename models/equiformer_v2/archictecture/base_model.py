"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from .utils import (
    compute_neighbors,
)


class BaseModel(nn.Module):
    def __init__(self, num_atoms=None, bond_feat_dim=None, num_targets=None):
        super(BaseModel, self).__init__()
        self.num_atoms = num_atoms
        self.bond_feat_dim = bond_feat_dim
        self.num_targets = num_targets

    def forward(self, data):
        raise NotImplementedError

    # OCP also implements a PBC version. I removed it for simplicity for now.
    # https://github.com/Open-Catalyst-Project/ocp/blob/9108a87ce383b2982c24eff4178632f01fecb63e/ocpmodels/models/base.py#L33
    def generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        otf_graph = otf_graph or self.otf_graph

        positions = data.pos if "pos" in data else data[0]

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if otf_graph:
            edge_index = radius_graph(
                positions,
                r=cutoff,
                batch=data.batch if "batch" in data else None,
                max_num_neighbors=max_neighbors,
            )

        j, i = edge_index
        distance_vec = positions[j] - positions[i]

        edge_dist = distance_vec.norm(dim=-1)
        cell_offsets = torch.zeros(
            edge_index.shape[1], 3, device=positions.device
        )
        cell_offset_distances = torch.zeros_like(
            cell_offsets, device=positions.device
        )
        neighbors = compute_neighbors(data, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
