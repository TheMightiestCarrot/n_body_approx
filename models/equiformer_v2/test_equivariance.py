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
    model_path = max(
        filter(os.path.isfile, glob.glob("models/equiformer_v2/runs/*/*")),
        key=os.path.getmtime,
    )
    model = load_model(model_path, device)
    model.eval()

    dataset = GravityDataset(
        partition="test",
        max_samples=1,
        target="pos",
        path=os.path.join(os.path.dirname(model_path), "nbody_small_dataset"),
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

    ### Test happy path

    # Generate a random rotation matrix
    rotation_matrix = torch.randn(3, 3).to(device)
    rotation_matrix, _ = torch.linalg.qr(rotation_matrix)
    # Rotate the input
    loc_rotated = rotate_positions(loc.squeeze(), rotation_matrix).view(
        -1, output_dims
    )  # TODO: Also rotate vel and force when added to the model
    data_rotated = [loc_rotated, vel, force, mass]

    # Perform inference on the rotated input
    pred_after_rotation = model(data_rotated, batch)

    # Rotate the prediction back
    pred_after_rotation_back = rotate_positions(pred_after_rotation, rotation_matrix.T)

    # Print the prediction before and after rotation
    print(f"Prediction before rotation: {pred_before_rotation}")
    print(f"Prediction after rotation: {pred_after_rotation_back}")

    # Check if the prediction before rotation is close to the rotated-back prediction after rotation
    assert torch.allclose(
        pred_before_rotation.to(dtype=torch.float32),
        pred_after_rotation_back.to(dtype=torch.float32),
        atol=1e-3,
    ), "The model is not equivariant under rotation"

    print("Equivariance test passed!")

    ### Test failure mode

    # Generate a random rotation matrix for failure mode test without rotating back
    rotation_matrix_failure = torch.randn(3, 3).to(device)
    rotation_matrix_failure, _ = torch.linalg.qr(rotation_matrix_failure)
    # Rotate the input for failure mode test without rotating back
    loc_rotated_failure = rotate_positions(loc.squeeze(), rotation_matrix_failure).view(
        -1, output_dims
    )  # TODO: Also rotate vel and force when added to the model for failure mode without rotating back
    data_rotated_failure = [loc_rotated_failure, vel, force, mass]

    # Perform inference on the rotated input for failure mode without rotating back
    pred_after_rotation_failure = model(data_rotated_failure, batch)

    print(f"Prediction before rotation: {pred_before_rotation}")
    print(
        f"Prediction after rotation without rotating back: {pred_after_rotation_failure}"
    )

    # Check if the prediction after rotation without rotating back is significantly different
    difference = torch.abs(pred_before_rotation - pred_after_rotation_failure)
    max_difference = torch.max(difference)
    print(f"Maximum difference without rotating back: {max_difference.item()}")

    # Assert that the difference is significant to confirm failure mode
    assert (
        max_difference.item() > 1e-1
    ), "The model might not be sensitive to rotation as expected in failure mode"
    print(
        "Failure mode test without rotating back passed! Model is sensitive to rotation as expected."
    )


if __name__ == "__main__":
    with torch.no_grad():
        test_equivariance()
