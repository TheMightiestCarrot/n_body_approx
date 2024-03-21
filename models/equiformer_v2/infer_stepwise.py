import torch

from datasets.nbody.dataset_gravity import GravityDataset
from models.equiformer_v2.utils import load_model
from utils import segnn_utils


def main():
    parser = segnn_utils.create_argparser()
    parser.add_argument(
        "--batched", action="store_true", help="inferring in batched mode"
    )
    parser.add_argument("--model-path", type=str, help="path to the model")
    parser.add_argument("--num-steps", type=int, default=20, help="number of steps for inference")
    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inferring on {device}")

    model = load_model(args.model_path, device)

    max_samples = 1

    dataset_train = GravityDataset(
        partition="test",
        dataset_name=args.nbody_name,
        max_samples=max_samples,
        neighbours=args.neighbours,
        target=args.target,
        path=f"{args.model_path}/nbody_small_dataset",
    )

    simulation_index = 0
    
    num_steps = args.num_steps

    loc, vel, force, mass = dataset_train.data

    output_dims = loc.shape[-1]
    n_nodes = loc.shape[-2]

    loc = torch.from_numpy(loc[simulation_index][:num_steps]).view(-1, output_dims)
    vel = torch.from_numpy(vel[simulation_index][:num_steps]).view(-1, output_dims)
    force = torch.from_numpy(force[simulation_index][:num_steps]).view(-1, output_dims)
    mass = torch.from_numpy(mass[simulation_index]).repeat(num_steps, 1)
    data = [loc, vel, force, mass]
    data = [d.to(device=device) for d in data]

    if args.batched:
        print("inferring in batched mode")

        batch_size = num_steps
        batch = (
            torch.arange(0, batch_size)
            .repeat_interleave(n_nodes)
            .long()
            .to(device=device)
        )
        pred = model(data, batch)
    else:
        print("inferring in stepwise mode")

        predicted_loc = []
        for step in range(num_steps):
            step_data = [
                d[step * n_nodes : (step + 1) * n_nodes].to(device=device) for d in data
            ]
            batch_size = 1
            batch = (
                torch.arange(0, batch_size)
                .repeat_interleave(n_nodes)
                .long()
                .to(device=device)
            )
            step_pred = model(step_data, batch)
            predicted_loc.append(step_pred)
        pred = torch.cat(predicted_loc, dim=0)

    from datasets.nbody.dataset import synthetic_sim

    loc = loc.view(num_steps, n_nodes, output_dims).detach().numpy()
    pred = pred.view(num_steps, n_nodes, output_dims).detach().numpy()

    sim = synthetic_sim.GravitySim(n_balls=5, loc_std=1)
    sim.interactive_trajectory_plot_all_particles_3d(
        loc, pred, particle_index=None, dims=output_dims, offline_plot=False
    )


if __name__ == "__main__":
    main()
