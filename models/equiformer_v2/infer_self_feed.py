import glob
import os

import torch

from datasets.nbody.dataset import synthetic_sim
from datasets.nbody.dataset_gravity import GravityDataset
from models.equiformer_v2.utils import load_model
from utils import segnn_utils


def main():
    parser = segnn_utils.create_argparser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=max(
            filter(os.path.isfile, glob.glob("models/equiformer_v2/runs/*/*")),
            key=os.path.getmtime,
        ),
        help="Path to the model",
    )
    parser.add_argument(
        "--num-steps", type=int, default=20, help="Number of steps to predict"
    )
    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path, device)

    dataset_train = GravityDataset(
        partition="test",
        dataset_name=args.nbody_name,
        max_samples=1,
        neighbours=args.neighbours,
        target=args.target,
        path=os.path.join(os.path.dirname(args.model_path), "nbody_small_dataset"),
    )

    simulation_index = 0

    loc, vel, force, mass = dataset_train.data

    output_dims = loc.shape[-1]
    n_nodes = loc.shape[-2]

    num_steps = args.num_steps

    loc_initial = torch.from_numpy(loc[simulation_index][:1]).view(-1, output_dims)
    vel_initial = torch.from_numpy(vel[simulation_index][:1]).view(-1, output_dims)
    force_initial = torch.from_numpy(force[simulation_index][:1]).view(-1, output_dims)
    mass_initial = torch.from_numpy(mass[simulation_index]).repeat(1, 1)

    predicted_loc = [loc_initial]
    predicted_vel = [vel_initial]
    predicted_force = [force_initial]
    predicted_mass = [mass_initial]

    for step in range(num_steps - 1):
        print(f"Predicting step {step}")

        data = [
            predicted_loc[-1],
            predicted_vel[-1],
            predicted_force[-1],
            predicted_mass[-1],
        ]
        data = [d.to(device=device) for d in data]

        batch_size = 1
        batch = (
            torch.arange(0, batch_size)
            .repeat_interleave(n_nodes)
            .long()
            .to(device=device)
        )

        pred = model(data, batch)
        predicted_loc.append(pred.cpu())
        predicted_vel.append(
            torch.zeros_like(pred.cpu())
        )  # TODO: Assuming vel is not predicted
        predicted_force.append(
            torch.zeros_like(pred.cpu())
        )  # Assuming force is not predicted
        predicted_mass.append(
            predicted_mass[-1].cpu()
        )  # Assuming mass remains constant

    print("Finished prediction")

    predicted_loc = torch.cat(predicted_loc, dim=0)
    predicted_vel = torch.cat(predicted_vel, dim=0)
    predicted_force = torch.cat(predicted_force, dim=0)
    predicted_mass = torch.cat(predicted_mass, dim=0)

    sim = synthetic_sim.GravitySim(n_balls=5, loc_std=1)

    loc_actual = (
        torch.from_numpy(loc[simulation_index][:num_steps])
        .view(-1, n_nodes, output_dims)
        .numpy()
    )
    loc_pred = predicted_loc.view(-1, n_nodes, output_dims).numpy()

    sim.interactive_trajectory_plot_all_particles_3d(
        loc_actual, loc_pred, particle_index=None, dims=output_dims, offline_plot=False
    )


if __name__ == "__main__":
    with torch.no_grad():
        main()
