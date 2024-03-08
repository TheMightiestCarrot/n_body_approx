import e3nn
import matplotlib.pyplot as plt
import torch
import torch_cluster
import torch_geometric
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from e3nn.nn.models.gate_points_2101 import Convolution, smooth_cutoff, tp_path_exists
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
import traceback
import os
from tqdm import tqdm


def calculate_edge_attr_old(irreps_edge_attr, edge_vec, max_radius, number_of_basis):
    edge_sh = o3.spherical_harmonics(irreps_edge_attr, edge_vec, True, normalization='component')
    edge_length = edge_vec.norm(dim=1)
    edge_length_embedded = soft_one_hot_linspace(
        x=edge_length,
        start=0.0,
        end=max_radius,
        number=number_of_basis,
        basis='gaussian',
        cutoff=False
    ).mul(number_of_basis ** 0.5)
    edge_attr = smooth_cutoff(edge_length / max_radius)[:, None] * edge_sh

    return edge_attr, edge_length_embedded


def calculate_edge_attr(l_max, edge_vec, max_radius, number_of_basis, static_edge_attr=None):
    """
        Enriches edge features:
         - radial basis function of nodal distances
         - adds spherical harmonics to edge features
    :param l_max:
    :param edge_vec:
    :param max_radius:
    :param number_of_basis:
    :param static_edge_attr: additional edge attributes besides the spherical harmonics (bond type)
    :return:
    """
    edge_sh = o3.spherical_harmonics(range(l_max + 1), edge_vec, True, normalization='component')
    edge_length = edge_vec.norm(dim=1)
    edge_length_embedded = soft_one_hot_linspace(
        x=edge_length,
        start=0.0,
        end=max_radius,
        number=number_of_basis,
        basis='smooth_finite',
        cutoff=True
    ).mul(number_of_basis ** 0.5)

    if static_edge_attr is not None:
        edge_attr = torch.cat([edge_sh, static_edge_attr], dim=1)
    else:
        edge_attr = edge_sh

    return edge_attr, edge_length_embedded


def create_fully_connected_data_with_edge_features(node_features, positions, targets, max_radius, num_basis,
                                                   l_max, node_attributes=None, static_edge_attr=None):
    num_nodes = node_features.size(0)
    device = node_features.device

    # Generate fully connected edge_index for num_nodes
    row = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    col = torch.arange(num_nodes, device=device).repeat(num_nodes)
    edge_index = torch.stack([row, col], dim=0)

    # Avoid self-loops
    edge_index = edge_index[:, row != col]

    # Calculate edge features
    edge_vec = positions[edge_index[0]] - positions[edge_index[1]]

    y = targets.detach().to(device).to(torch.float32)

    edge_attr, edge_length_embedded = calculate_edge_attr(l_max, edge_vec, max_radius, num_basis, static_edge_attr)

    data_dict = {
        'x': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'y': y,
        'edge_vec': edge_vec,
        'pos': positions,
        'edge_length_embedded': edge_length_embedded
    }

    if node_attributes is not None:
        data_dict['z'] = node_attributes

    if static_edge_attr is not None:
        data_dict['static_edge_attr'] = static_edge_attr

    return Data(**data_dict)


def create_n_body_graph_dataset(inputs_np, targets_np, batch_size=1, dims=3, max_radius=None, num_basis=None, l_max=1,
                                dtype=torch.float32):
    inputs_tensor = torch.tensor(inputs_np, dtype=dtype)
    targets_tensor = torch.tensor(targets_np, dtype=dtype)

    data_list = []

    for index, simulation_step_graph in enumerate(inputs_tensor):
        node_features = simulation_step_graph[..., dims:]  # velocity
        positions = simulation_step_graph[..., :dims]
        # targets = targets_tensor[index, ...]
        targets = targets_tensor[index, :(dims * 2)] - simulation_step_graph[..., :(dims * 2)]

        data = create_fully_connected_data_with_edge_features(node_features, positions, targets,
                                                              max_radius=max_radius, num_basis=num_basis, l_max=l_max)
        data_list.append(data)

    data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    return data_loader


# Equivariance test
def equivariance_test(model, data_loader):
    retval = 99999
    training = model.training
    device = next(model.parameters()).device
    # if training:
    #     model.eval()

    try:
        model.to(torch.device("cpu"))
        # Nahodna rotacia
        rot = o3.rand_matrix()
        # Wignerove matice tejto rotacie
        D_in = model.irreps_in.D_from_matrix(rot)

        d_edge = o3.Irreps.spherical_harmonics(model.lmax).D_from_matrix(rot)


        D_out = model.irreps_out.D_from_matrix(rot)

        for batch in data_loader:
            # rotate before
            f_in = batch.detach().cpu().clone()
            btch, node_features, node_attr, edge_src, edge_dst, edge_vec, edge_attr, edge_scalars = model.preprocess(f_in)

            # ROTACIA NODALNYCH FICUR
            f_in.x = node_features @ D_in.T
            # ROTACIA POZICIE (z ktorej sa vyrataju edgeove ficury)
            #f_in.z = node_attr @ rot.T

            f_in.edge_attr = edge_attr @ d_edge.T

            #f_in.edge_length_embedded = edge_scalars @ rot.T

            #[mp.irreps_node_input, mp.irreps_node_attr, None, None, mp.irreps_edge_attr, None],

            f_before = model(f_in)
            # rotate after
            f_after = model(batch) @ D_out.T

            # TOTO POROVNA ROZDIELY
            print(torch.allclose(f_before, f_after, rtol=1e-4, atol=1e-4))
            break

        print(f_before.abs().mean())
        print((f_after - f_before).abs().mean())
        retval = (f_after - f_before).abs().mean()
    except Exception as e:
        print(e)
        pass

    if training:
        model.train()
        model.to(device)

    return retval


def visualize_layers(model):
    fontsize = 16
    textsize = 14
    layer_dst = dict(zip(['sc', 'lin1', 'tp', 'lin2'], ['gate', 'tp', 'lin2', 'gate']))
    try:
        layers = model.mp.layers
    except:
        layers = model.layers

    num_layers = len(layers)
    num_ops = max([len([k for k in list(layers[i].first._modules.keys()) if k not in ['fc', 'alpha']])
                   for i in range(num_layers - 1)])

    fig, ax = plt.subplots(num_layers, num_ops, figsize=(14, 3.5 * num_layers))
    for i in range(num_layers - 1):
        ops = layers[i].first._modules.copy()
        ops.pop('fc', None);
        ops.pop('alpha', None)
        for j, (k, v) in enumerate(ops.items()):
            ax[i, j].set_title(k, fontsize=textsize)
            v.cpu().visualize(ax=ax[i, j])
            ax[i, j].text(0.7, -0.15, '--> to ' + layer_dst[k], fontsize=textsize - 2, transform=ax[i, j].transAxes)

    layer_dst = dict(zip(['sc', 'lin1', 'tp', 'lin2'], ['output', 'tp', 'lin2', 'output']))
    ops = layers[-1]._modules.copy()
    ops.pop('fc', None);
    ops.pop('alpha', None)
    for j, (k, v) in enumerate(ops.items()):
        ax[-1, j].set_title(k, fontsize=textsize)
        v.cpu().visualize(ax=ax[-1, j])
        ax[-1, j].text(0.7, -0.15, '--> to ' + layer_dst[k], fontsize=textsize - 2, transform=ax[-1, j].transAxes)

    fig.subplots_adjust(wspace=0.3, hspace=0.5)


def train_model(model, optimizer, criterion, data_loader, num_epochs, old_epoch=0, loggers=[], dims=3, scheduler=None,
                log_every=5, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    from IPython.display import display, clear_output
    try:
        print("training")
        last_avg_loss = 0
        model.train()
        model.to(device)

        for epoch in tqdm(range(old_epoch + 1, old_epoch + num_epochs), desc="Epochs"):
            total_metrics = {
                "loss": 0, "loss_pos": 0, "loss_vel": 0,
                "perc_error_pos": 0, "perc_error_vel": 0, "perc_error_pos_vs_vel_l1": 0, "perc_error_pos_vs_vel_l2": 0
            }
            num_batches = 0
            for batch in data_loader:
                batch.to(device)
                targets = batch.y  # Adjust based on your data loading method

                # Forward pass
                outputs = model(batch)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    predicted_pos = outputs[..., :dims]
                    target_pos = targets[..., :dims]

                    predicted_vel = outputs[..., dims:]
                    target_vel = targets[..., dims:]

                    loss_pos = criterion(predicted_pos, target_pos)
                    loss_vel = criterion(predicted_vel, target_vel)

                    # Calculate percentage errors
                    perc_error_pos = (torch.norm(predicted_pos - target_pos, dim=1) /
                                      torch.norm(target_pos, dim=1)).mean() * 100

                    perc_error_vel = (torch.norm(predicted_vel - target_vel, dim=1) /
                                      torch.norm(target_vel, dim=1)).mean() * 100

                    perc_error_pos_vs_vel_l1 = (torch.abs(predicted_pos - target_pos).mean() /
                                                torch.norm(target_vel, dim=1)).mean() * 100

                    perc_error_pos_vs_vel_l2 = (torch.norm(predicted_pos - target_pos, dim=1) /
                                                torch.norm(target_vel, dim=1)).mean() * 100

                    total_metrics["loss"] += loss.item()
                    total_metrics["loss_pos"] += loss_pos.item()
                    total_metrics["loss_vel"] += loss_vel.item()
                    total_metrics["perc_error_pos"] += perc_error_pos.item()
                    total_metrics["perc_error_vel"] += perc_error_vel.item()
                    total_metrics["perc_error_pos_vs_vel_l1"] += perc_error_pos_vs_vel_l1.item()
                    total_metrics["perc_error_pos_vs_vel_l2"] += perc_error_pos_vs_vel_l2.item()

                num_batches += 1
            if scheduler is not None:
                scheduler.step()

            avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
            clear_output(wait=True)
            display(
                f"Epoch [{epoch + 1}/{num_epochs}], lr: {optimizer.param_groups[0]['lr']:.4f}, avg_both: {avg_metrics['loss']:.5f}, avg_pos: {avg_metrics['loss_pos']: .5f}, avg_vel: {avg_metrics['loss_vel']: .5f}, perc_pos: {avg_metrics['perc_error_pos']: .5f}%, perc_vel: {avg_metrics['perc_error_vel']: .5f}%")
            if epoch % log_every == 0:

                for logger in loggers:
                    # writer.add_scalar('Loss/last_both', total_metrics["loss"], epoch)
                    logger.log_scalar('Loss/last_pos', loss_pos.item(), epoch)
                    logger.log_scalar('Loss/last_vel', loss_vel.item(), epoch)

                    # logger.log_scalar('Loss/avg_both', avg_metrics["loss"], epoch)
                    logger.log_scalar('Loss/avg_pos', avg_metrics["loss_pos"], epoch)
                    logger.log_scalar('Loss/avg_vel', avg_metrics["loss_vel"], epoch)

                    logger.log_scalar('Loss/perc_pos', avg_metrics["perc_error_pos"], epoch)
                    logger.log_scalar('Loss/perc_vel', avg_metrics["perc_error_vel"], epoch)
                    logger.log_scalar('Loss/perc_pos_vs_vel_l1', avg_metrics["perc_error_pos_vs_vel_l1"], epoch)
                    logger.log_scalar('Loss/perc_pos_vs_vel_l2', avg_metrics["perc_error_pos_vs_vel_l2"], epoch)

                    for name, weight in model.named_parameters():
                        logger.log_histogram(f'{name}/weights', weight, epoch)
                        if weight.grad is not None:
                            logger.log_histogram(f'{name}/grads', weight.grad, epoch)

                    last_avg_loss = avg_metrics["loss"]




    except KeyboardInterrupt:
        pass

    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()

    print("Saving model at the end of training")
    for logger in loggers:
        if logger.get_logdir():
            torch.save(model, os.path.join(logger.get_logdir(), "model.pth"))

    return epoch, last_avg_loss
