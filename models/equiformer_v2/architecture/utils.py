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


# Different encodings for the atom distance embeddings
class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super(GaussianSmearing, self).__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = (
            -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        )
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
