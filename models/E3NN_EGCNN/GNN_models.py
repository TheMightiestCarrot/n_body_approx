import e3nn
import torch
import torch_cluster
import torch_geometric
from e3nn import o3
from e3nn.nn import Gate
# from e3nn.nn.models.gate_points_2101 import Convolution, tp_path_exists
from e3nn.nn.models.v2106.points_convolution import Convolution as Convolutionv2106
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists, Compose
from torch_scatter import scatter

from . import GNN_utils
import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_networks import SimpleNetwork
from e3nn.nn.models.v2106.gate_points_message_passing import MessagePassing


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)


class SimpleNetworkCustom(torch.nn.Module):
    def __init__(
            self,
            irreps_in,
            irreps_out,
            max_radius,
            num_neighbors,
            num_nodes,
            mul=32,
            layers=3,
            lmax=2,
            number_of_basis=10,
            mlp_layers=[100],
            fully_connected=False,
            pool_nodes=True,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes
        self.fully_connected = fully_connected

        irreps_node_hidden = o3.Irreps([(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_in] + layers * [irreps_node_hidden] + [irreps_out],
            irreps_node_attr="0e",
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            fc_neurons=[self.number_of_basis] + mlp_layers,
            num_neighbors=num_neighbors,
        )

        self.irreps_in = self.mp.irreps_node_input
        self.irreps_out = self.mp.irreps_node_output

    def preprocess(self, data):
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
            static_edge_attr = data.get('static_edge_attr', None)
            edge_attr, edge_length_embedded = GNN_utils.calculate_edge_attr(self.lmax, edge_vec,
                                                                            self.max_radius,
                                                                            self.number_of_basis,
                                                                            static_edge_attr)

        if 'z' in data:
            node_attr = data['z']
        else:
            # assert self.irreps_node_attr == o3.Irreps("0e")
            node_attr = data['pos'].new_ones((data['pos'].shape[0], 1))

        if 'x' in data:
            assert self.irreps_in is not None
            x = data['x']
        else:
            assert self.irreps_in is None
            x = data['pos'].new_ones((data['pos'].shape[0], 1))

        return batch, x, node_attr, edge_src, edge_dst, edge_vec, edge_attr, edge_length_embedded

    def forward(self, data):
        batch, x, node_attr, edge_src, edge_dst, edge_vec, edge_attr, edge_length_embedded = self.preprocess(data)

        x = self.mp(x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedded)

        if self.pool_nodes:
            return scatter(x, batch, dim=0).div(self.num_nodes ** 0.5)
        else:
            return x


@compile_mode("script")
class Convolutionv2106Custom(torch.nn.Module):
    def __init__(
            self, irreps_node_input, irreps_node_attr, irreps_edge_attr, irreps_node_output, fc_neurons, num_neighbors
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else None
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors

        if irreps_node_attr is not None:
            self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr,
                                                  self.irreps_node_output)

            self.lin1 = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr,
                                                    self.irreps_node_input)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        assert irreps_mid.dim > 0, (
            f"irreps_node_input={self.irreps_node_input} time irreps_edge_attr={self.irreps_edge_attr} produces nothing "
            f"in irreps_node_output={self.irreps_node_output}"
        )
        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(fc_neurons + [tp.weight_numel], torch.nn.functional.silu)
        self.tp = tp

        if self.irreps_node_attr is not None:
            self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_node_output)

            # inspired by https://arxiv.org/pdf/2002.10444.pdf
            self.alpha = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")
            with torch.no_grad():
                self.alpha.weight.zero_()
            assert (
                    self.alpha.output_mask[0] == 1.0
            ), f"irreps_mid={irreps_mid} and irreps_node_attr={self.irreps_node_attr} are not able to generate scalars"

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        weight = self.fc(edge_scalars)

        if self.irreps_node_attr is not None:
            node_self_connection = self.sc(node_input, node_attr)
            node_features = self.lin1(node_input, node_attr)
        else:
            node_features = node_input

        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(edge_features, edge_dst, dim_size=node_input.shape[0]).div(self.num_neighbors ** 0.5)

        if self.irreps_node_attr is not None:
            node_conv_out = self.lin2(node_features, node_attr)
            alpha = self.alpha(node_features, node_attr)

            m = self.sc.output_mask
            alpha = (1 - m) + alpha * m
            return node_self_connection + alpha * node_conv_out

        else:
            return node_features


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
        # self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else None
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
            # conv = Convolution(
            #     irreps,
            #     self.irreps_node_attr,
            #     self.irreps_edge_attr,
            #     gate.irreps_in,
            #     number_of_basis,
            #     radial_layers,
            #     radial_neurons,
            #     num_neighbors
            # )

            conv = Convolutionv2106Custom(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                [number_of_basis] + radial_layers * [radial_neurons],
                num_neighbors
            )
            irreps = gate.irreps_out
            # self.layers.append(CustomCompose(conv, gate))
            self.layers.append(Compose(conv, gate))

        self.layers.append(
            # Convolution(
            #     irreps,
            #     self.irreps_node_attr,
            #     self.irreps_edge_attr,
            #     self.irreps_out,
            #     number_of_basis,
            #     radial_layers,
            #     radial_neurons,
            #     num_neighbors
            # )
            Convolutionv2106Custom(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                [number_of_basis] + radial_layers * [radial_neurons],
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
            l_max, edge_vec, max_radius, number_of_basis, edge_attr = None
            static_edge_attr = data.get('static_edge_attr', None)
            edge_attr, edge_length_embedded = GNN_utils.calculate_edge_attr(self.lmax, edge_vec,
                                                                            self.max_radius,
                                                                            self.number_of_basis,
                                                                            static_edge_attr)

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
            # assert self.irreps_node_attr == o3.Irreps("0e")
            z = data['pos'].new_ones((data['pos'].shape[0], 1))

        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        if self.reduce_output:
            return scatter(x, batch, dim=0).div(self.num_nodes ** 0.5)
        else:
            return x
