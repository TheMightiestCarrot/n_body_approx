import e3nn
import torch
import torch_cluster
import torch_geometric
from e3nn import o3
from e3nn.nn import Gate
#from e3nn.nn.models.gate_points_2101 import Convolution, tp_path_exists
from e3nn.nn.models.v2106.points_convolution import Convolution as Convolutionv2106
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists, Compose
from torch_scatter import scatter

import GNN_utils


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

            conv = Convolutionv2106(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                [number_of_basis] + radial_layers * [radial_neurons],
                num_neighbors
            )
            irreps = gate.irreps_out
            #self.layers.append(CustomCompose(conv, gate))
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
            Convolutionv2106(
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
            edge_attr, edge_length_embedded = GNN_utils.calculate_edge_attr(self.irreps_edge_attr, edge_vec,
                                                                            self.max_radius,
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
