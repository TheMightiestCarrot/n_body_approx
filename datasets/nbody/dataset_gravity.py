import os
import pathlib

import numpy as np
import torch


class GravityDataset():
    """
    NBodyDataset

    """
    path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'dataset', 'gravity')
    os.makedirs(path, exist_ok=True)

    def __init__(self, partition='train', max_samples=1e8, dataset_name="nbody_small", bodies=5, neighbours=6,
                 target="pos"):
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

        assert target in ["pos", "force"]

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()
        self.neighbours = int(neighbours)
        self.target = target

    def load(self):

        loc = np.load(os.path.join(self.path, 'loc_' + self.suffix + '.npy'))
        vel = np.load(os.path.join(self.path, 'vel_' + self.suffix + '.npy'))
        force = np.load(os.path.join(self.path, 'edges_' + self.suffix + '.npy'))
        mass = np.load(os.path.join(self.path, 'charges_' + self.suffix + '.npy'))

        self.num_nodes = loc.shape[-1]

        loc, vel, force, mass = self.preprocess(loc, vel, force, mass)
        return (loc, vel, force, mass), None

    def preprocess(self, loc, vel, force, mass):
        # cast to torch and swap n_nodes <--> n_features dimensions
        # loc = torch.Tensor(loc).transpose(2, 3)
        # vel = torch.Tensor(vel).transpose(2, 3)
        # force = torch.Tensor(force).transpose(2, 3)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        force = force[0:self.max_samples, :, :, :]
        mass = mass[0:self.max_samples]

        return loc, vel, force, torch.Tensor(mass)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, force, mass = self.data
        loc, vel, force, mass = loc[i], vel[i], force[i], mass[i]

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

        return loc[frame_0], vel[frame_0], force[frame_0], mass, y

    def __len__(self):
        return self.data[0].shape[0]

    # def get_edges(self, batch_size, n_nodes):
    #     edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
    #     if batch_size == 1:
    #         return edges
    #     elif batch_size > 1:
    #         rows, cols = [], []
    #         for i in range(batch_size):
    #             rows.append(edges[0] + n_nodes * i)
    #             cols.append(edges[1] + n_nodes * i)
    #         edges = [torch.cat(rows), torch.cat(cols)]
    #     return edges


if __name__ == "__main__":
    dataset = GravityDataset(dataset_name="nbody")

    for item in dataset:
        loc0, vel0, force0, mass, locT = item
        print(loc0.shape, vel0.shape, force0.shape, mass.shape, locT.shape)
        break
