import argparse

import numpy as np
import torch
from torch_cluster import radius_graph

from datasets.nbody import GravityDataset

from .architecture.equiformer_v2_nbody import EquiformerV2_nbody


def load_model(model_path, device):
    model = EquiformerV2_nbody().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def infer(model, loader, device):
    predicted_trajectories = []
    actual_trajectories = []  # Store actual trajectories for comparison
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            batch_data = [data.to(device) for data in batch_data]
            batch_data = [d.view(-1, d.size(2)) for d in batch_data]
            # Assuming the model expects data in a specific format, similar to the training phase
            loc, vel, edge_attr, charges, target = batch_data

            batch_data = (loc, vel, edge_attr, charges)
            edge_index = radius_graph(loc, r=10.0)
            edge_distance_vec = loc[edge_index[1]] - loc[edge_index[0]]
            edge_distance = torch.norm(edge_distance_vec, dim=1)
            pred = model(batch_data, target, edge_index, edge_distance, edge_distance_vec, batch_idx)
            predicted_trajectories.append(pred.cpu().numpy())
            actual_trajectories.append(target.cpu().numpy())  # Append actual trajectory
    return predicted_trajectories, actual_trajectories

def main():
    parser = argparse.ArgumentParser(description="Inference script for N-body trajectories")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--dataset-name", type=str, default="nbody_small")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_test = GravityDataset(partition='test', dataset_name=args.dataset_name)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = load_model(args.model_path, device)
    predicted_trajectories, actual_trajectories = infer(model, loader_test, device)
    print(f"Generated trajectories from model {args.model_path}")
    # Save the predicted and actual trajectories to files
    np.save("predicted_trajectories.npy", predicted_trajectories)
    np.save("actual_trajectories.npy", actual_trajectories)

if __name__ == "__main__":
    main()
