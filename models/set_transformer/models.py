import torch.nn as nn


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn=nn.Tanh, output_act=None, precision='double'):
        super(MLPLayer, self).__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        if output_act is not None:
            layers.append(output_act())
        self.layers = nn.Sequential(*layers)
        if precision == 'double':
            self.double()

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, particle_dim, model_dim, num_heads, num_layers, particle_index=None,
                 activation='relu', hparams=None):
        super(Encoder, self).__init__()
        self.hparams = hparams
        self.particle_index = particle_index
        self.input_layer = nn.Linear(particle_dim, model_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True,
                                                    activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(model_dim, particle_dim)
        if hparams.get('precision', '') == 'double':
            self.double()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.transformer_encoder(x)
        if self.particle_index is not None:
            x = x[:, self.particle_index, :]
        x = self.output_layer(x)
        return x


class EncoderWithMLP(Encoder):
    def __init__(self, particle_dim, model_dim, num_heads, num_layers, particle_index=None,
                 activation='relu', hparams=None, hidden_dims=None, mlp_act=nn.ReLU, mlp_type='output',
                 mlp_output_act=None):
        super(EncoderWithMLP, self).__init__(particle_dim, model_dim, num_heads, num_layers, particle_index, activation,
                                             hparams=hparams)
        if mlp_type == 'input':
            self.input_layer = MLPLayer(model_dim, hidden_dims, particle_dim, activation_fn=mlp_act,
                                        output_act=mlp_output_act)
        elif mlp_type == 'output':
            self.output_layer = MLPLayer(model_dim, hidden_dims, particle_dim, activation_fn=mlp_act,
                                         output_act=mlp_output_act)
        else:
            raise ValueError(f"Invalid mlp_type: {mlp_type}. Expected 'input' or 'output'.")
