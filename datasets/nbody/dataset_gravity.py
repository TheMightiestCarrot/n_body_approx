import os
import pathlib
import json
import numpy as np
import torch
from .dataset.synthetic_sim import GravitySim
import matplotlib.pyplot as plt
from multiprocessing import Pool
import random
from functools import partial


class GravityDataset():
    """
    NBodyDataset

    """

    GROUND_TRUTH_FILE_PREFIXES = ['loc', 'vel', 'forces', 'masses']
    DEFAULT_DATA_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'dataset', 'gravity')

    def __init__(self, partition='train', max_samples=1e8, dataset_name="nbody_small", bodies=5, neighbours=6,
                 target="pos", random_trajectory_sampling=False, steps_to_predict=2, path=DEFAULT_DATA_PATH):
        self.partition = partition
        if self.partition == 'val':
            self.suffix = 'valid'
        else:
            self.suffix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.suffix += f"_gravity{str(bodies)}_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.suffix += f"_gravity{str(bodies)}_initvel1small"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.metadata = {}
        self.simulation: GravitySim = None
        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.path = path

        os.makedirs(path, exist_ok=True)
        self.data, self.edges = self.load(path)
        self.neighbours = int(neighbours)
        self.target = target
        self.random_trajectory_sampling = random_trajectory_sampling
        self.steps_to_predict = steps_to_predict

    def load(self, path=None):

        if path is None:
            path = self.path

        loc, vel, force, mass = [np.load(os.path.join(path, f"{prefix}_{self.suffix}.npy")) for prefix in
                                 self.GROUND_TRUTH_FILE_PREFIXES]

        self.num_nodes = loc.shape[-1]

        metadata_path = os.path.join(path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as json_file:
                self.metadata = json.load(json_file)
                self.simulation = self.init_simulation_instance()

        loc, vel, force, mass = self.preprocess(loc, vel, force, mass)
        return (loc, vel, force, mass), None

    def preprocess(self, loc, vel, force, mass):
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        force = force[0:self.max_samples, :, :, :]
        mass = mass[0:self.max_samples]

        return loc, vel, force, mass

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, force, mass = self.data
        loc, vel, force, mass = loc[i], vel[i], force[i], mass[i]

        if self.random_trajectory_sampling:
            min_frame = 0
            separation = self.steps_to_predict
            max_frame = len(loc) - separation

            frame_0 = random.randint(min_frame, max_frame - separation)
            frame_T = frame_0 + separation

        else:
            if self.dataset_name == "nbody":
                frame_0, frame_T = 6, 8
            elif self.dataset_name == "nbody_small":
                frame_0, frame_T = 30, 40
            elif self.dataset_name == "nbody_small_out_dist":
                frame_0, frame_T = 20, 30
            else:
                raise Exception("Wrong dataset partition %s" % self.dataset_name)

        if self.target == "pos":
            y = loc[frame_T]
        elif self.target == "force":
            y = force[frame_T]
        elif self.target == "pos_dt+vel_dt":
            pos_dt = loc[frame_T] - loc[frame_0]  # Change in position
            vel_dt = vel[frame_T] - vel[frame_0]  # Change in velocity
            # y = torch.cat((pos_dt, vel_dt), dim=0)
            y = np.concatenate((pos_dt, vel_dt), axis=1)

        return loc[frame_0], vel[frame_0], force[frame_0], mass, y

    def __len__(self):
        return self.data[0].shape[0]

    def init_simulation_instance(self):
        args = self.metadata['args']
        return GravitySim(noise_var=args['noise_var'], n_balls=args['n_balls'], vel_norm=args['initial_vel_norm'],
                          interaction_strength=args['G'], dt=args['dt'])

    def get_one_sim_data(self, simulation_index):
        loc, vel, force, mass = self.data
        loc = loc[simulation_index]
        vel = vel[simulation_index]
        force = force[simulation_index]
        mass = mass[simulation_index]

        return loc, vel, force, mass

    @staticmethod
    def plot_histograms(loc, vel, force=None, bins=30):
        num_dims = loc.shape[3]  # Update to reflect new indexing due to the additional 'bodies' dimension
        dim_labels = ['x', 'y', 'z'][:num_dims]  # Labels for dimensions
        colors = ['red', 'green', 'blue'][:num_dims]  # Color for each dimension

        plt.figure(figsize=(10, 5))

        # Positions
        plt.subplot(1, 3, 1)
        for i, (color, label) in enumerate(zip(colors, dim_labels)):
            # Flatten across simulations, steps, and bodies but keep dimensions separate
            plt.hist(loc[:, :, :, i].flatten(), bins=bins, alpha=0.5, color=color, label=f'{label} position')
        plt.title('Positions')
        plt.legend()

        # Velocities
        plt.subplot(1, 3, 2)
        for i, (color, label) in enumerate(zip(colors, dim_labels)):
            # Flatten across simulations, steps, and bodies but keep dimensions separate
            plt.hist(vel[:, :, :, i].flatten(), bins=bins, alpha=0.5, color=color, label=f'{label} velocity')
        plt.title('Velocities')
        plt.legend()

        if force is not None:
            # Forces
            plt.subplot(1, 3, 3)
            for i, (color, label) in enumerate(zip(colors, dim_labels)):
                # Flatten across simulations, steps, and bodies but keep dimensions separate
                plt.hist(force[:, :, :, i].flatten(), bins=bins, alpha=0.5, color=color, label=f'{label} force')
            plt.title('Forces')
            plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_differences(loc, vel, step=2, bins=30):
        num_dims = loc.shape[3]
        dim_labels = ['x', 'y', 'z'][:num_dims]
        colors = ['red', 'green', 'blue'][:num_dims]

        plt.figure(figsize=(20, 5))

        # Position Differences
        plt.subplot(1, 2, 1)
        for i, (color, label) in enumerate(zip(colors, dim_labels)):
            # Calculate differences along the steps dimension (axis=1)
            diffs = np.diff(loc[:, :, :, i], axis=1, n=step).flatten()
            plt.hist(diffs, bins=bins, alpha=0.5, color=color, label=f'{label} position difference')
        plt.title('Position Differences')
        plt.legend()

        # Velocity Differences
        plt.subplot(1, 2, 2)
        for i, (color, label) in enumerate(zip(colors, dim_labels)):
            diffs = np.diff(vel[:, :, :, i], axis=1, n=step).flatten()
            plt.hist(diffs, bins=bins, alpha=0.5, color=color, label=f'{label} velocity difference')
        plt.title('Velocity Differences')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def simulate_one(self, sim, loc, vel, mass):
        energies = [self.simulation._energy(loc[sim, i, :, :], vel[sim, i, :, :], mass[i],
                                            self.simulation.interaction_strength) for i in
                    range(loc.shape[1])]
        return energies

    def get_energies_async(self, loc, vel, mass):
        num_simulations = loc.shape[0]

        partial_simulate_one = partial(self.simulate_one, loc=loc, vel=vel, mass=mass)

        with Pool() as pool:
            energies = pool.map(partial_simulate_one, range(0, num_simulations))

        return np.array(energies)

    def plot_energy_statistics(self, loc, vel, force=None, mass=None):
        energies_array = self.get_energies_async(loc, vel, mass)

        plt.figure(figsize=(14, 8))
        colors = {'Kinetic Energy': 'red', 'Potential Energy': 'blue', 'Total Energy': 'green'}
        energy_labels = ['Kinetic Energy', 'Potential Energy', 'Total Energy']

        for i, energy_label in enumerate(energy_labels):
            energy_mean = energies_array[:, :, i].mean(axis=0)
            energy_std = energies_array[:, :, i].std(axis=0)

            times = np.arange(energy_mean.shape[0])

            # Plot mean
            plt.plot(times, energy_mean, color=colors[energy_label], label=energy_label)
            # Plot standard deviation range
            plt.fill_between(times, energy_mean - energy_std, energy_mean + energy_std, color=colors[energy_label],
                             alpha=0.2)

        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Average Energy vs Time for Multiple Simulations with Std. Dev.')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_energy_distributions_across_all_sims(self, loc, vel, force=None, mass=None, bins=50):
        energies_array = self.get_energies_async(loc, vel, mass)

        # Flatten the energy arrays to include all time points from all simulations
        kinetic_energies = energies_array[:, :, 0].flatten()
        potential_energies = energies_array[:, :, 1].flatten()
        total_energies = energies_array[:, :, 2].flatten()

        energy_types = ['Kinetic Energy', 'Potential Energy', 'Total Energy']
        energies = [kinetic_energies, potential_energies, total_energies]
        colors = ['red', 'blue', 'green']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Energy Distributions Across All Time Points and Simulations')

        for i, ax in enumerate(axes):
            ax.hist(energies[i], bins=bins, color=colors[i], alpha=0.7, density=True)
            ax.set_title(energy_types[i])
            ax.set_xlabel('Energy')
            ax.set_ylabel('Density')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_energies_of_all_sims(self, loc, vel, force=None, mass=None):
        num_simulations = loc.shape[0]
        energies_array = self.get_energies_async(loc, vel, mass)

        plt.figure(figsize=(14, 8))

        colors = {'Kinetic Energy': 'red', 'Potential Energy': 'blue', 'Total Energy': 'green'}

        for energy_type, color in colors.items():
            plt.plot([], [], color=color, label=energy_type)

        # Plotting all three energy types for each simulation
        for sim in range(num_simulations):
            times = np.arange(energies_array[sim].shape[0])

            # Kinetic Energy
            plt.plot(times, energies_array[sim, :, 0], alpha=0.3, color=colors['Kinetic Energy'], linestyle='--')

            # Potential Energy
            plt.plot(times, energies_array[sim, :, 1], alpha=0.3, color=colors['Potential Energy'], linestyle=':')

            # Total Energy
            plt.plot(times, energies_array[sim, :, 2], alpha=0.3, color=colors['Total Energy'])

        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy vs Time for Multiple Simulations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to make sure everything fits without overlapping
        plt.show()

    @staticmethod
    def plot_trajectories_static(loc, opacity=0.4, max_sims=100):
        num_sims = loc.shape[0]
        num_dims = loc.shape[3]
        n_balls = loc.shape[2]

        # If the number of simulations exceeds max_sims, sample max_sims randomly
        if num_sims > max_sims:
            selected_sims = np.random.choice(num_sims, max_sims, replace=False)
        else:
            selected_sims = np.arange(num_sims)

        if num_dims == 2:
            plt.figure(figsize=(10, 8))
            for sim in selected_sims:
                for n in range(n_balls):
                    plt.plot(loc[sim, :, n, 0], loc[sim, :, n, 1], alpha=opacity, linewidth=0.5)

        elif num_dims == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            for sim in selected_sims:
                for n in range(n_balls):
                    ax.plot(loc[sim, :, n, 0], loc[sim, :, n, 1], loc[sim, :, n, 2], alpha=opacity, linewidth=0.5)
        else:
            raise ValueError("Dimensions not supported for plotting")

        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_trajectories_static_3D_to_2D(loc, opacity=0.4, max_sims=100):

        num_sims = loc.shape[0]
        n_balls = loc.shape[2]

        # If max_sims is specified and the number of simulations exceeds it, sample max_sims randomly
        if max_sims is not None and num_sims > max_sims:
            selected_sims = np.random.choice(num_sims, max_sims, replace=False)
        else:
            selected_sims = np.arange(num_sims)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        for sim in selected_sims:
            for n in range(n_balls):
                # XY plane
                axs[0].plot(loc[sim, :, n, 0], loc[sim, :, n, 1], alpha=opacity)
                axs[0].set_xlabel('X Position')
                axs[0].set_ylabel('Y Position')
                axs[0].set_title('XY Plane')

                # XZ plane
                axs[1].plot(loc[sim, :, n, 0], loc[sim, :, n, 2], alpha=opacity)
                axs[1].set_xlabel('X Position')
                axs[1].set_ylabel('Z Position')
                axs[1].set_title('XZ Plane')

                # YZ plane
                axs[2].plot(loc[sim, :, n, 1], loc[sim, :, n, 2], alpha=opacity)
                axs[2].set_xlabel('Y Position')
                axs[2].set_ylabel('Z Position')
                axs[2].set_title('YZ Plane')

        for ax in axs:
            ax.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    dataset = GravityDataset(dataset_name="nbody")

    for item in dataset:
        loc0, vel0, force0, mass, locT = item
        print(loc0.shape, vel0.shape, force0.shape, mass.shape, locT.shape)
        break
