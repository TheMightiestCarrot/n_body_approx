import sys
from pathlib import Path

project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

from datasets.nbody.dataset_gravity import GravityDataset
from datasets.nbody.dataset_nbody import NBodyDataset
from .synthetic_sim import ChargedParticlesSim, SpringSim, GravitySim
import numpy as np
import argparse
import multiprocessing
import os

"""
nbody_small:   python3 -u generate_dataset.py --simulation=charged --num-train 10000 --seed 43 --suffix small
gravity_small: python3 -u generate_dataset.py --simulation=gravity --num-train 10000 --seed 43 --suffix small
"""

import sys

sys.argv = [
    'generate_dataset.py', '--simulation=gravity', '--num-train=10000', '--seed=43',
    '--suffix=small', '--num-valid=2000', '--num-test=2000'
]

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='charged',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=10000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=2000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=2000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length_test', type=int, default=5000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n_balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--initial_vel', type=int, default=1,
                    help='consider initial velocity')
parser.add_argument('--suffix', type=str, default="",
                    help='add a suffix to the name')
# GRAVITY PARAMS
parser.add_argument('--G', type=int, default=1,
                    help='gravitational constant')
parser.add_argument('--dt', type=int, default=0.001,
                    help='simulation step')
parser.add_argument('--softening', type=int, default=0.1,
                    help='softening parameter of gravity simulation')

args = parser.parse_args()

initial_vel_norm = 0.5
if not args.initial_vel:
    initial_vel_norm = 1e-16

args.initial_vel_norm = initial_vel_norm
args.noise_var = 0

if args.simulation == 'springs':
    sim = SpringSim(noise_var=args.noise_var, n_balls=args.n_balls)
    suffix = '_springs'
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=args.noise_var, n_balls=args.n_balls, vel_norm=args.initial_vel_norm)
    suffix = '_charged'
    save_path = NBodyDataset.path
elif args.simulation == 'gravity':
    sim = GravitySim(noise_var=args.noise_var, n_balls=args.n_balls, vel_norm=args.initial_vel_norm,
                     interaction_strength=args.G, dt=args.dt)
    suffix = '_gravity'
    save_path = GravityDataset.path
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_balls) + "_initvel%d" % args.initial_vel + args.suffix
np.random.seed(args.seed)


def sample_trajectory_wrapper(args):
    index, length, sample_freq = args
    np.random.seed(index)  # Ensure different seeds for different processes
    return sim.sample_trajectory(T=length, sample_freq=sample_freq)


def generate_dataset(num_sims, length, sample_freq):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Create an iterable with arguments for each simulation run
        args_iter = [(i, length, sample_freq) for i in range(num_sims)]

        # Map the sample_trajectory_wrapper function across the input iterable
        results = pool.map(sample_trajectory_wrapper, args_iter)

    # Unpack results
    loc_all, vel_all, edges_all, charges_all = zip(*results)

    # Convert lists of arrays into stacked arrays
    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)
    charges_all = np.stack(charges_all)

    return loc_all, vel_all, edges_all, charges_all


if __name__ == "__main__":
    print("Generating {} training simulations".format(args.num_train))
    loc_train, vel_train, edges_train, charges_train = generate_dataset(args.num_train,
                                                                        args.length,
                                                                        args.sample_freq)

    print("Generating {} validation simulations".format(args.num_valid))
    loc_valid, vel_valid, edges_valid, charges_valid = generate_dataset(args.num_valid,
                                                                        args.length,
                                                                        args.sample_freq)

    print("Generating {} test simulations".format(args.num_test))
    loc_test, vel_test, edges_test, charges_test = generate_dataset(args.num_test,
                                                                    args.length_test,
                                                                    args.sample_freq)

    if args.simulation == 'gravity':
        edges = "forces"
        nodes = "masses"
    else:
        edges = "edges"
        nodes = "charges"

    np.save(os.path.join(save_path, 'loc_train' + suffix + '.npy'), loc_train)
    np.save(os.path.join(save_path, 'vel_train' + suffix + '.npy'), vel_train)
    np.save(os.path.join(save_path, f'{edges}_train' + suffix + '.npy'), edges_train)
    np.save(os.path.join(save_path, f'{nodes}_train' + suffix + '.npy'), charges_train)

    np.save(os.path.join(save_path, 'loc_valid' + suffix + '.npy'), loc_valid)
    np.save(os.path.join(save_path, 'vel_valid' + suffix + '.npy'), vel_valid)
    np.save(os.path.join(save_path, f'{edges}_valid' + suffix + '.npy'), edges_valid)
    np.save(os.path.join(save_path, f'{nodes}_valid' + suffix + '.npy'), charges_valid)

    np.save(os.path.join(save_path, 'loc_test' + suffix + '.npy'), loc_test)
    np.save(os.path.join(save_path, 'vel_test' + suffix + '.npy'), vel_test)
    np.save(os.path.join(save_path, f'{edges}_test' + suffix + '.npy'), edges_test)
    np.save(os.path.join(save_path, f'{nodes}_test' + suffix + '.npy'), charges_test)

    import json

    with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
        json.dump({"args": vars(args)}, f, indent=4)
