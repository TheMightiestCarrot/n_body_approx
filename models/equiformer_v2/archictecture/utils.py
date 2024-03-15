# These function are copied from https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/common/utils.py
# We don't need all of them and I want to keep it simple for now.


import torch
from torch_scatter import segment_coo, segment_csr


def compute_neighbors(data, edge_index, batch_size):
    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = edge_index[1].new_ones(1).expand_as(edge_index[1])
    natoms = (
        data.natoms
        if "natoms" in data
        else torch.tensor([[5]] * 32, device=data[0].device)
    )  # TODO: do properly
    num_neighbors = segment_coo(
        ones, edge_index[1], dim_size=natoms.sum() if "natoms" in data else 160
    )

    # Get number of neighbors per image
    image_indptr = torch.zeros(
        natoms.shape[0] + 1, device=data[0].device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    neighbors = segment_csr(num_neighbors, image_indptr)
    return neighbors
