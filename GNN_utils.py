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


def calculate_edge_attr(irreps_edge_attr, edge_vec, max_radius, number_of_basis):
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


def create_fully_connected_data_with_edge_features(node_features, positions, targets, max_radius, num_basis, l_max):
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

    irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax=l_max)
    edge_attr, edge_length_embedded = calculate_edge_attr(irreps_edge_attr, edge_vec, max_radius, num_basis)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_vec=edge_vec,
                pos=positions, edge_length_embedded=edge_length_embedded)


def create_graph_dataset(inputs_np, targets_np, batch_size=1, dims=3, max_radius=None, num_basis=None, l_max=1,
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
    training = model.training
    if training:
        model.eval()

    try:

        # Nahodna rotacia
        rot = o3.rand_matrix()
        # Wignerove matice tejto rotacie
        D_in = model.irreps_in.D_from_matrix(rot)
        D_out = model.irreps_out.D_from_matrix(rot)

        for batch in data_loader:
            # rotate before
            f_in = batch.clone()

            # ROTACIA NODALNYCH FICUR
            f_in.x = f_in.x @ D_in.T
            # ROTACIA POZICIE (z ktorej sa vyrataju edgeove ficury)
            f_in.pos = f_in.pos @ rot.T

            f_before = model(f_in)
            # rotate after
            f_after = model(batch) @ D_out.T

            # TOTO POROVNA ROZDIELY
            print(torch.allclose(f_before, f_after, rtol=1e-4, atol=1e-4))
            break

        print(f_before.abs().mean())
        print((f_after - f_before).abs().mean())
    except:
        pass

    if training:
        model.train()


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


class CustomCompose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x


class EGCN(torch.nn.Module):
    r"""equivariant neural network
    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps` or None
        representation of the input features
        can be set to ``None`` if nodes don't have input features
    irreps_hidden : `e3nn.o3.Irreps`
        representation of the hidden features
    irreps_out : `e3nn.o3.Irreps`
        representation of the output features
    irreps_node_attr : `e3nn.o3.Irreps` or None
        representation of the nodes attributes
        can be set to ``None`` if nodes don't have attributes
    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes
        the edge attributes are :math:`h(r) Y(\vec r / r)`
        where :math:`h` is a smooth function that goes to zero at ``max_radius``
        and :math:`Y` are the spherical harmonics polynomials
    layers : int
        number of gates (non linearities)
    max_radius : float
        maximum radius for the convolution
    number_of_basis : int
        number of basis on which the edge length are projected
    radial_layers : int
        number of hidden layers in the radial fully connected network
    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network
    num_neighbors : float
        typical number of nodes at a distance ``max_radius``
    num_nodes : float
        typical number of nodes in a graph
    """

    def __init__(
            self,
            irreps_in,
            irreps_out,
            irreps_node_attr=None,
            layers=1,
            mul=1,
            lmax=1,
            max_radius=None,
            number_of_basis=10,
            radial_layers=1,
            radial_neurons=100,
            num_neighbors=1.,
            num_nodes=1.,
            reduce_output=True,
            fully_connected=True,
    ) -> None:
        super().__init__()
        self.mul = mul
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output
        self.fully_connected = fully_connected

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        self.input_has_node_in = (irreps_in is not None)
        self.input_has_node_attr = (irreps_node_attr is not None)

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if
                                        ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if
                                      ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
            irreps = gate.irreps_out
            self.layers.append(CustomCompose(conv, gate))

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
        )

    def preprocess(self, data) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        if 'edge_index' in data and self.training:
            edge_src = data['edge_index'][0]  # edge source
            edge_dst = data['edge_index'][1]  # edge destination
            edge_vec = data['edge_vec']
        else:
            if self.fully_connected:
                # Generate fully connected edge_index for num_nodes
                num_nodes = data.x.size(0)
                row = torch.arange(num_nodes).repeat_interleave(num_nodes)
                col = torch.arange(num_nodes).repeat(num_nodes)
                edge_index = torch.stack([row, col], dim=0)

                # Avoid self-loops
                edge_index = edge_index[:, row != col]
            else:
                edge_index = torch_cluster.radius_graph(data['pos'], self.max_radius, batch)

            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]

        if ('edge_attr' and 'edge_length_embedded' in data) and self.training:
            edge_attr = data.edge_attr
            edge_length_embedded = data.edge_length_embedded
        else:
            edge_attr, edge_length_embedded = calculate_edge_attr(self.irreps_edge_attr, edge_vec, self.max_radius,
                                                                  self.number_of_basis)

        return batch, edge_src, edge_dst, edge_vec, edge_attr, edge_length_embedded

    def forward(self, data) -> torch.Tensor:
        """evaluate the network
        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        """
        batch, edge_src, edge_dst, edge_vec, edge_attr, edge_length_embedded = self.preprocess(data)

        if self.input_has_node_in and 'x' in data:
            assert self.irreps_in is not None
            x = data['x']
        else:
            assert self.irreps_in is None
            x = data['pos'].new_ones((data['pos'].shape[0], 1))

        if self.input_has_node_attr and 'z' in data:
            z = data['z']
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = data['pos'].new_ones((data['pos'].shape[0], 1))

        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        if self.reduce_output:
            return scatter(x, batch, dim=0).div(self.num_nodes ** 0.5)
        else:
            return x


def train_model(model, optimizer, criterion, data_loader, num_epochs, old_epoch=0, loggers=[], dims=3, scheduler=None,
                log_every=5):
    try:
        print("training")
        last_avg_loss = 0
        model.train()

        for epoch in tqdm(range(old_epoch + 1, old_epoch + num_epochs), desc="Epochs"):
            total_metrics = {
                "loss": 0, "loss_pos": 0, "loss_vel": 0,
                "perc_error_pos": 0, "perc_error_vel": 0, "perc_error_pos_vs_vel_l1": 0, "perc_error_pos_vs_vel_l2": 0
            }
            num_batches = 0
            for batch in data_loader:
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
            print(
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
