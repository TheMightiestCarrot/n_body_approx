import argparse
import datetime
import os
import time
from typing import Iterable, Optional

import torch
import torchmetrics
from timm.utils import ModelEmaV2, dispatch_clip_grad

import wandb
from datasets.nbody.dataset_gravity import GravityDataset
from models.equiformer_v2.architecture.equiformer_v2_nbody import \
    EquiformerV2_nbody

ModelEma = ModelEmaV2



def get_args_parser():
    parser = argparse.ArgumentParser("Training EquiformerV2 on N-body", add_help=False)
    parser.add_argument("--data-path", type=str, default="datasets/nbody")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--dataset-name", type=str, default="nbody_small")
    parser.add_argument("--log", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--test-interval", type=int, default=1)
    parser.add_argument("--num-atoms", type=int, default=5)
    
    return parser

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    backprop: bool=True,
                    args = None, 
                    model_ema: Optional[ModelEma] = None,  
                    loss_scaler=None,
                    clip_grad=None,
                    print_freq: int = 100, 
                    logger=None):
    
    if backprop:
        model.train()
        criterion.train()
    else:
        model.eval()
        criterion.eval()

    res = {'epoch': epoch, 'counter': 0, 'mae': 0, 'loss': 0}
    
    loss_metric = torchmetrics.MeanMetric().to(device)
    mae_metric = torchmetrics.MeanMetric().to(device)
    
    start_time = time.perf_counter()
    
    for batch_idx, batch_data in enumerate(data_loader):
        data = [d.to(device) for d in batch_data]
        data = [d.view(-1, d.size(2)) for d in data]
        loc, vel, edge_attr, charges, loc_end = data
        
        # Create a Data object
        data = (loc, vel, edge_attr, charges)
        data = tuple(d.to(device) for d in data)
        target = loc_end.to(device)
        
        pred = model(data, batch_idx, args.batch_size, args.num_atoms)
        pred = pred.squeeze()
        
        loss = criterion(pred, target)

        if backprop:
            optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            if backprop:
                loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), 
                    value=clip_grad, mode='norm')
            if backprop:
                optimizer.step()
        
        loss_metric.update(loss)
        err = pred.detach() - target
        mae_metric.update(torch.mean(torch.abs(err)))
    
        if model_ema is not None:
            model_ema.update(model)
        
        # torch.cuda.synchronize() TODO: do we need this?
        
        # logging
        if batch_idx % print_freq == 0 or batch_idx == len(data_loader) - 1:
            w = time.perf_counter() - start_time
            e = (batch_idx + 1) / len(data_loader)
            info_str = f'Epoch: [{epoch}/{args.epochs}] \t Step: [{batch_idx}/{len(data_loader)}] \t Loss: {loss_metric.compute().item():.5f}, MAE: {mae_metric.compute().item():.5f}, time/step={(1e3 * w / e / len(data_loader)):.0f}ms'
            info_str += f', lr={optimizer.param_groups[0]["lr"]:.2e}'
            print(info_str)

        res['counter'] += batch_data[0].size(0)
        
    return mae_metric.compute().item(), loss_metric.compute().item()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_train = GravityDataset(partition='train', dataset_name=args.dataset_name,
                                 max_samples=args.max_samples)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    dataset_val = GravityDataset(partition='val', dataset_name=args.dataset_name)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    dataset_test = GravityDataset(partition='test', dataset_name=args.dataset_name)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Initialize model
    model = (
        EquiformerV2_nbody()
    )  # Update irreps_in and irreps_out based on your dataset
    model.to(device)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.MSELoss()

    wandb.init(project="EquiformerV2_nbody", config=args)


    best_val_loss = 1e8
    best_epoch = 0
    for epoch in range(0, args.epochs):
        print(f"Starting training epoch {epoch}")
        _, train_loss = train_one_epoch(model, criterion, loader_train, optimizer, device, epoch, loader_train, args=args)
        if args.log:
            wandb.log({"Train MSE": train_loss})
        if epoch % args.test_interval == 0 or epoch == args.epochs - 1:
            print(f"Starting validation epoch {epoch}")
            _, val_loss = train_one_epoch(model, criterion, loader_val, optimizer, device, epoch, backprop = False, args=args)
            wandb.log({"Val MSE": val_loss})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
            print("*** Best Val Loss: %.5f \t Best epoch %d" %
                  (best_val_loss, best_epoch))
            
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # create the directory if it doesn't exist
    save_dir_path = 'models/equiformer_v2/runs'
    os.makedirs(save_dir_path, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir_path}/{args.dataset_name}_best_model_{current_time}.pth")

    return best_val_loss, best_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "EquiformerV2 training script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
