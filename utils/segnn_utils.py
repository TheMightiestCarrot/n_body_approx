import argparse
import copy
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from datasets.nbody.train_gravity_V2 import O3Transform
import multiprocessing


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


def simulate_one_index(args_tuple):
    model, data, simulation_instance, args, device, simulation_index, steps, use_force, transform = args_tuple
    print("Simulating", simulation_index, "Using force:", use_force)
    loc, vel, force, mass = copy.deepcopy(data)

    n_nodes = loc.shape[-2]
    loc, vel, force, mass = [torch.from_numpy(d) for d in [loc, vel, force, mass]]
    loc, vel, force, mass = [d[simulation_index, 0, ...].to(device) for d in [loc, vel, force, mass]]
    mass = mass.repeat(n_nodes, 1)

    output_dims = loc.shape[-1]
    states = []

    for step in range(steps):
        graph = Data(pos=loc, vel=vel, force=force, mass=mass)
        graph.edge_index = knn_graph(loc, args.neighbours)
        graph = transform(graph)  # Add O3 attributes
        graph = graph.to(device)

        # Model prediction
        prediction = model(graph).cpu().detach().numpy()
        delta_loc, delta_vel = prediction[:, :output_dims], prediction[:, output_dims:]
        loc = loc + torch.from_numpy(delta_loc).to(device)
        vel = vel + torch.from_numpy(delta_vel).to(device)

        # Update force
        force = simulation_instance.compute_force(loc.cpu().detach().numpy(), mass.cpu().detach().numpy(),
                                                  simulation_instance.interaction_strength,
                                                  simulation_instance.softening)
        force = torch.from_numpy(force).to(device)

        states.append((loc.clone(), vel.clone(), force.clone()))

    return np.stack(states)


def self_feed_stepwise_prediction_parallel(model, data, simulation_instance, args, device, simulation_indices=range(10),
                                           steps=6):
    use_force = args.use_force
    transform = O3Transform(args.lmax_attr, use_force)

    args_list = [(model, data, simulation_instance, args, device, simulation_index, steps, use_force, transform)
                 for simulation_index in simulation_indices]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        all_states = pool.map(simulate_one_index, args_list)

    all_states = np.array(all_states)

    return all_states[:, :, 0, ...], all_states[:, :, 1, ...], all_states[:, :, 2, ...]


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

    use_force = args.use_force
    transform = O3Transform(args.lmax_attr, use_force)

    all_states = []

    for simulation_index in simulation_indices:
        print("Simulating", simulation_index, "Using force:", use_force)
        loc, vel, force, mass = copy.deepcopy(data)

        n_nodes = loc.shape[-2]

        loc = torch.from_numpy(loc)
        vel = torch.from_numpy(vel)
        force = torch.from_numpy(force)
        mass = torch.from_numpy(mass)

        loc, vel, force, mass = [d[simulation_index, 0, ...].to(device) for d in [loc, vel, force, mass]]
        mass = mass.repeat(n_nodes, 1)

        output_dims = loc.shape[-1]

        states = []
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

            force = simulation_instance.compute_force(loc.cpu().detach().numpy(), mass.cpu().detach().numpy(),
                                                      simulation_instance.interaction_strength,
                                                      simulation_instance.softening)
            force = torch.from_numpy(force)

            states.append((loc.clone(), vel.clone(), force.clone()))

        all_states.append(np.stack(states))
    all_states = np.stack(all_states)

    return all_states[:, :, 0, ...], all_states[:, :, 1, ...], all_states[:, :, 2, ...]


def self_feed_batch_prediction(model, data, simulation_instance, args, device,
                               n_sims=10, steps=6):
    transform = O3Transform(args.lmax_attr, args.use_force)

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

    states = []

    for step in range(steps):
        graph = Data(pos=loc, vel=vel, force=force, mass=mass)

        batch = torch.arange(0, n_sims)
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

        force = simulation_instance.compute_force(loc.cpu().detach().numpy(), mass.cpu().detach().numpy(),
                                                  simulation_instance.interaction_strength,
                                                  simulation_instance.softening, n_sims)

        force = torch.from_numpy(force)
        # force = force_copy[simulation_index, step]

        states.append((loc.clone(), vel.clone(), force.clone()))

    states = np.stack(states)  # [steps, loc+vel+force, n_nodes x sims, dims]
    states = states.transpose(1, 0, 2, 3)  # [loc+vel+force, steps, n_nodes x sims, dims]
    all_steps_reshaped = states.reshape(3, steps, n_sims, n_nodes,
                                        output_dims)  # [loc+vel+force, steps, sims, n_nodes, dims]
    all_steps_reshaped = all_steps_reshaped.transpose(0, 2, 1, 3, 4)  # [loc+vel+force, sims, steps, n_nodes, dims]

    return all_steps_reshaped[0, ...], all_steps_reshaped[1, ...], all_steps_reshaped[2, ...]


def batch_prediction(model, data, args, device, simulation_indices=(i for i in range(10))):
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
    loc, vel, force, mass = copy.deepcopy(data)

    steps = loc.shape[1]

    targets = []
    for i in range(0, steps - t_delta):
        targets.append(loc[simulation_index, i + t_delta])

    return np.array(targets)


def compare_batch_vs_step(model, dataset, device, args, num_simulations=5):
    torch.manual_seed(42)
    np.random.seed(42)

    use_force = args.use_force
    transform = O3Transform(args.lmax_attr, use_force)

    all_simulations_predictions = batch_prediction(model, dataset.data, args, device,
                                                   simulation_indices=(i for i in range(num_simulations)))

    all_stepwise_simulations_predictions = []  # List to store predictions for all simulations

    for simulation_index in range(num_simulations):
        loc, vel, force, mass = copy.deepcopy(dataset.data)

        output_dims = loc.shape[-1]
        n_nodes = loc.shape[-2]
        simulation_steps = len(loc[simulation_index])  # Adjust based on your simulation steps determination logic

        stepwise_prediction = []  # List to store predictions for each step in the current simulation
        for step in range(simulation_steps):
            loc_step = torch.from_numpy(loc[simulation_index][step]).view(-1, output_dims).to(device)
            vel_step = torch.from_numpy(vel[simulation_index][step]).view(-1, output_dims).to(device)
            force_step = torch.from_numpy(force[simulation_index][step]).view(-1, output_dims).to(device)
            mass_step = torch.tensor(mass[simulation_index], dtype=torch.float).repeat(n_nodes, 1).to(device)

            graph = Data(pos=loc_step, vel=vel_step, force=force_step, mass=mass_step)
            graph.edge_index = knn_graph(loc_step, args.neighbours)

            graph = transform(graph)  # Add O3 attributes
            graph = graph.to(device)
            stepwise_prediction.append(model(graph).detach().cpu().numpy())

        # Convert stepwise predictions to a single array for the current simulation
        stepwise_prediction = np.stack(stepwise_prediction)
        all_stepwise_simulations_predictions.append(stepwise_prediction)

    # Convert all simulations' predictions into a NumPy array with an extra dimension for simulations
    all_stepwise_simulations_predictions = np.array(all_stepwise_simulations_predictions)

    print("Difference:", np.abs(all_stepwise_simulations_predictions - all_simulations_predictions).mean())
    return all_simulations_predictions, all_stepwise_simulations_predictions
