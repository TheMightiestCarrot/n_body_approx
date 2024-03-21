import glob
import os

import torch

from datasets.nbody.dataset_gravity import GravityDataset
from models.equiformer_v2.utils import load_model


def rotate_positions(positions, rotation_matrix):
    """Rotate positions by a given rotation matrix."""
    return torch.einsum(
        "ij,nj->ni",
        rotation_matrix.to(dtype=torch.float32),
        positions.to(dtype=torch.float32),
    )


def test_equivariance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = max(filter(os.path.isfile, glob.glob("models/equiformer_v2/runs/*/*")), key=os.path.getmtime)
    model = load_model(model_path, device)
    model.eval()

    dataset = GravityDataset(
        partition="test",
        max_samples=1,
        target="pos",
        path="models/equiformer_v2/runs/2024-03-20_17-08-30/nbody_small_dataset",
    )
    loc, vel, force, mass = dataset.data
    output_dims = loc.shape[-1]

    loc = torch.from_numpy(loc[0][:1]).to(device).view(-1, output_dims)
    vel = torch.from_numpy(vel[0][:1]).to(device).view(-1, output_dims)
    force = torch.from_numpy(force[0][:1]).to(device).view(-1, output_dims)
    mass = torch.from_numpy(mass[0]).to(device)
    data = [loc, vel, force, mass]

    batch = torch.zeros(loc.shape[0], dtype=torch.long, device=device)
    pred_before_rotation = model(data, batch)

    # Generate a random rotation matrix
    rotation_matrix = torch.randn(3, 3).to(device)
    rotation_matrix, _ = torch.linalg.qr(rotation_matrix)
    # Rotate the input
    loc_rotated = rotate_positions(loc.squeeze(), rotation_matrix).view(-1, output_dims)
    data_rotated = [loc_rotated, vel, force, mass]

    # Perform inference on the rotated input
    pred_after_rotation = model(data_rotated, batch)

    # Rotate the prediction back
    pred_after_rotation_back = rotate_positions(pred_after_rotation, rotation_matrix.T)

    # Check if the prediction before rotation is close to the rotated-back prediction after rotation
    assert torch.allclose(
        pred_before_rotation.to(dtype=torch.float32),
        pred_after_rotation_back.to(dtype=torch.float32),
        atol=1e-3,
    ), "The model is not equivariant under rotation"

    print("Equivariance test passed!")


if __name__ == "__main__":
    test_equivariance()
