import argparse


def create_argparser():
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='weight decay')
    parser.add_argument('--print', type=int, default=100,
                        help='print interval')
    parser.add_argument('--log', type=bool, default=False,
                        help='logging flag')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Num workers in dataloader')
    parser.add_argument('--save_dir', type=str, default="saved models",
                        help='Directory in which to save models')

    # Data parameters
    parser.add_argument('--dataset', type=str, default="qm9",
                        help='Data set')
    parser.add_argument('--root', type=str, default="datasets",
                        help='Data set location')
    parser.add_argument('--download', type=bool, default=False,
                        help='Download flag')

    # QM9 parameters
    parser.add_argument('--target', type=str, default="alpha",
                        help='Target value, also used for gravity dataset [pos, force]')
    parser.add_argument('--radius', type=float, default=2,
                        help='Radius (Angstrom) between which atoms to add links.')
    parser.add_argument('--feature_type', type=str, default="one_hot",
                        help='Type of input feature: one-hot, or Cormorants charge thingy')

    # Nbody parameters:
    parser.add_argument('--nbody_name', type=str, default="nbody_small",
                        help='Name of nbody data [nbody, nbody_small]')
    parser.add_argument('--max_samples', type=int, default=3000,
                        help='Maximum number of samples in nbody dataset')
    parser.add_argument('--time_exp', type=bool, default=False,
                        help='Flag for timing experiment')
    parser.add_argument('--test_interval', type=int, default=5,
                        help='Test every test_interval epochs')
    parser.add_argument('--n_nodes', type=int, default=5,
                        help='How many nodes are in the graph.')

    # Gravity parameters:
    parser.add_argument('--neighbours', type=int, default=6,
                        help='Number of connected nearest neighbours')

    # Model parameters
    parser.add_argument('--model', type=str, default="segnn",
                        help='Model name')
    parser.add_argument('--hidden_features', type=int, default=128,
                        help='max degree of hidden rep')
    parser.add_argument('--lmax_h', type=int, default=2,
                        help='max degree of hidden rep')
    parser.add_argument('--lmax_attr', type=int, default=3,
                        help='max degree of geometric attribute embedding')
    parser.add_argument('--subspace_type', type=str, default="weightbalanced",
                        help='How to divide spherical harmonic subspaces')
    parser.add_argument('--layers', type=int, default=7,
                        help='Number of message passing layers')
    parser.add_argument('--norm', type=str, default="instance",
                        help='Normalisation type [instance, batch]')
    parser.add_argument('--pool', type=str, default="avg",
                        help='Pooling type type [avg, sum]')
    parser.add_argument('--conv_type', type=str, default="linear",
                        help='Linear or non-linear aggregation of local information in SEConv')

    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=0, type=int,
                        help='number of gpus to use (assumes all are on one node)')

    return parser


def self_feed_stepwise_prediction(model, data, simulation_instance, args, device,
                                  simulation_indices=(i for i in range(10)),
                                  steps=6):
    """
        Executes stepwise predictions on n-body simulations using a PyTorch model for multiple initial conditions.
        It simulates dynamics by predicting changes in position and velocity, updating the state at each step.

        Parameters:
        - model: PyTorch model for prediction.
        - data: Tuple (loc, vel, force, mass) of numpy arrays as initial conditions.
        - simulation_instance: Simulation instance with `compute_force_batched` for force calculation.
        - args: Argument parser object with `lmax_attr` and `neighbours` for graph construction and transformation.
        - device: PyTorch device for computation.
        - simulation_indices: Iterable of indices for simulations to run, default 0-9.
        - steps: Number of prediction steps per simulation.

        Returns:
        - Tuple of numpy arrays: (simulated stepwise positions, model predictions), with shapes
          [(num_simulations, steps, num_nodes, output_dims), (num_simulations, steps, num_nodes, prediction_dims)].
        """
    import torch
    import copy
    from torch_geometric.data import Data
    from torch_geometric.nn import knn_graph
    from datasets.nbody.train_gravity import O3Transform
    import numpy as np

    import importlib
    import datasets.nbody.dataset.synthetic_sim as synthetic_sim

    importlib.reload(synthetic_sim)

    transform = O3Transform(args.lmax_attr)

    simulations = []
    all_predictions = []

    for simulation_index in simulation_indices:
        print("Simulating", simulation_index)
        loc, vel, force, mass = copy.deepcopy(data)

        n_nodes = loc.shape[-2]

        loc = torch.from_numpy(loc)
        vel = torch.from_numpy(vel)
        force = torch.from_numpy(force)
        mass = torch.from_numpy(mass)

        loc, vel, force, mass = [d[simulation_index, 0, ...].to(device) for d in [loc, vel, force, mass]]
        mass = mass.repeat(n_nodes, 1)

        output_dims = loc.shape[-1]

        stepwise_prediction = []
        predictions = []
        for step in range(steps):
            graph = Data(pos=loc, vel=vel, force=force, mass=mass)
            graph.edge_index = knn_graph(loc, args.neighbours)

            graph = transform(graph)  # Add O3 attributes
            graph = graph.to(device)

            # Model prediction
            prediction = model(graph).cpu().detach().numpy()

            # Update states based on prediction
            delta_loc, delta_vel = prediction[:, :output_dims], prediction[:, output_dims:]

            # Update position and velocity
            loc = loc + torch.from_numpy(delta_loc).to(device)
            vel = vel + torch.from_numpy(delta_vel).to(device)

            # todo batchsize
            force = simulation_instance.compute_force_batched(loc.cpu().detach().numpy(), mass.cpu().detach().numpy(),
                                                              simulation_instance.interaction_strength,
                                                              simulation_instance.softening, 9999999)

            force = torch.from_numpy(force)
            # force = force_copy[simulation_index, step]

            stepwise_prediction.append(loc.clone())
            predictions.append(prediction.copy())

        stepwise_prediction = np.stack(stepwise_prediction)
        simulations.append(stepwise_prediction)
        all_predictions.append(predictions)

    return np.stack(simulations), np.stack(all_predictions)


def self_feed_batch_prediction(model, data, simulation_instance, args, device,
                               n_sims=10, steps=6):
    import torch
    import copy
    from torch_geometric.data import Data
    from torch_geometric.nn import knn_graph
    from datasets.nbody.train_gravity import O3Transform
    import numpy as np

    import importlib
    import datasets.nbody.dataset.synthetic_sim as synthetic_sim

    importlib.reload(synthetic_sim)

    transform = O3Transform(args.lmax_attr)

    loc, vel, force, mass = copy.deepcopy(data)

    output_dims = loc.shape[-1]
    n_nodes = loc.shape[-2]

    loc = torch.from_numpy(loc)
    vel = torch.from_numpy(vel)
    force = torch.from_numpy(force)
    mass = torch.from_numpy(mass)

    # get just initial states
    loc, vel, force, mass = [d[:n_sims, 0, ...].to(device) for d in [loc, vel, force, mass]]
    loc, vel, force = [d.reshape(-1, 3) for d in [loc, vel, force]]
    mass = mass.repeat(n_nodes, 1)

    predictions = []
    predicted_locs = []

    for step in range(steps):
        batch = torch.arange(0, n_sims)
        graph = Data(pos=loc, vel=vel, force=force, mass=mass)

        graph.batch = batch.repeat_interleave(n_nodes).long()
        graph.edge_index = knn_graph(loc, args.neighbours, graph.batch)
        graph = transform(graph)  # Add O3 attributes
        graph = graph.to(device)

        # Model prediction
        prediction = model(graph).cpu().detach().numpy()

        # Update states based on prediction
        delta_loc, delta_vel = prediction[:, :output_dims], prediction[:, output_dims:]

        # Update position and velocity
        loc = loc + torch.from_numpy(delta_loc).to(device)
        vel = vel + torch.from_numpy(delta_vel).to(device)

        force = simulation_instance.compute_force_batched(loc.cpu().detach().numpy(), mass.cpu().detach().numpy(),
                                                          simulation_instance.interaction_strength,
                                                          simulation_instance.softening, n_sims)

        force = torch.from_numpy(force)
        # force = force_copy[simulation_index, step]

        predictions.append(prediction.copy())
        predicted_locs.append(loc.clone())

    all_steps = np.stack(predictions)
    all_steps_reshaped = all_steps.reshape(steps, n_sims, n_nodes, output_dims * 2)
    all_steps_reshaped_transposed = all_steps_reshaped.transpose(1, 0, 2, 3)

    predicted_locs = np.stack(predicted_locs)
    predicted_locs_reshaped = predicted_locs.reshape(steps, n_sims, n_nodes, output_dims)
    predicted_locs_reshaped_transposed = predicted_locs_reshaped.transpose(1, 0, 2, 3)

    return predicted_locs_reshaped_transposed, all_steps_reshaped_transposed


def batch_prediction(model, data, args, device, simulation_indices=(i for i in range(10))):
    import copy
    from torch_geometric.data import Data
    from torch_geometric.nn import knn_graph
    from datasets.nbody.train_gravity import O3Transform
    import torch
    import numpy as np

    transform = O3Transform(args.lmax_attr)

    all_predictions = []
    for simulation_index in simulation_indices:
        loc, vel, force, mass = copy.deepcopy(data)

        data_dims = loc.shape[-1]
        batch_size = loc.shape[-3]
        n_nodes = loc.shape[-2]

        loc = torch.from_numpy(loc[simulation_index]).view(-1, data_dims)
        vel = torch.from_numpy(vel[simulation_index]).view(-1, data_dims)
        force = torch.from_numpy(force[simulation_index]).view(-1, data_dims)
        mass = torch.from_numpy(mass[simulation_index]).repeat(batch_size, 1)

        loc, vel, force, mass = [d.to(device) for d in [loc, vel, force, mass]]

        graph = Data(pos=loc, vel=vel, force=force, mass=mass)
        batch = torch.arange(0, batch_size)
        graph.batch = batch.repeat_interleave(n_nodes).long()
        graph.edge_index = knn_graph(loc, args.neighbours, graph.batch)

        graph = transform(graph)  # Add O3 attributes
        graph = graph.to(device)
        batch_prediction = model(graph).cpu().detach().numpy()
        output_dims = batch_prediction.shape[-1]
        all_predictions.append(batch_prediction.reshape(batch_size, n_nodes, output_dims))

    return np.stack(all_predictions)


def get_targets(data, simulation_index, t_delta):
    import numpy as np
    import copy
    loc, vel, force, mass = copy.deepcopy(data)

    steps = loc.shape[1]

    targets = []
    for i in range(0, steps - t_delta):
        targets.append(loc[simulation_index, i + t_delta])

    return np.array(targets)


def compare_predictions(batch_preds_np, stepwise_preds_np):
    from sklearn.metrics import mean_squared_error
    # Calculate MSE
    mse_batch = mean_squared_error(batch_preds_np.reshape(-1, 3), stepwise_preds_np.reshape(-1, 3))
    mse_stepwise = mean_squared_error(stepwise_preds_np.reshape(-1, 3), batch_preds_np.reshape(-1, 3))

    print({
        "MSE_Batch": mse_batch,
        "MSE_Stepwise": mse_stepwise,
        "MSE_Difference": abs(mse_batch - mse_stepwise)
    })
