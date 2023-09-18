import torch.nn as nn
import torch
import numpy as np
from einops.layers.torch import Rearrange
from torchsummary import summary
from siren_pytorch import Siren
class MLP(nn.Module):
    def __init__(self,
                 aif,
                 n_layers,
                 n_units,
                 n_inputs=1,
                 neurons_out=1,
                 bn=False,
                 act='tanh'):
        super(MLP, self).__init__()
        self.aif = aif
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_inputs = n_inputs
        self.neurons_out = neurons_out
        if not bn:
            self.bn = False
        else:
            raise NotImplementedError('Batchnorm not yet working, maybe layernorm?')
            # self.bn = True
        if act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise NotImplementedError("There is no other activation implemented.")

        self.net = self.__make_net()

    def __make_net(self):
        layers = [nn.Linear(self.n_inputs, self.n_units)]
        for i in range(self.n_layers):
            layers.append(self.act)
            layers.append(nn.Linear(self.n_units, self.n_units))
        layers.append(self.act)
        layers.append(nn.Linear(self.n_units,
                                self.neurons_out))
        return nn.Sequential(*layers)

    def forward(self, t, xy):
        if not self.aif:
            txy = torch.concat([t, xy], dim=-1)

            out = self.net(txy)
            return out
        else:
            t = self.net(t)
            return t[..., 0]

class MLP_ODE(nn.Module):
    def __init__(self,
                 n_layers,
                 n_units,
                 n_inputs=1,
                 neurons_out=1,
                 bn=False,
                 act='tanh'):
        super(MLP_ODE, self).__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_inputs = n_inputs
        self.neurons_out = neurons_out
        if not bn:
            self.bn = False
        else:
            raise NotImplementedError('Batchnorm not yet working, maybe layernorm?')
            # self.bn = True
        if act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'relu':
            self.act = nn.ReLU()

        else:
            raise NotImplementedError("There is no other activation implemented.")

        self.net = self.__make_net()

    def __make_net(self):
        layers = [nn.Linear(self.n_inputs, self.n_units)]
        for i in range(self.n_layers):
            layers.append(self.act)
            layers.append(nn.Linear(self.n_units, self.n_units))
        layers.append(self.act)
        layers.append(nn.Linear(self.n_units,
                                self.neurons_out))
        return nn.Sequential(*layers)

    def forward(self, xy):
        out = self.net(xy)
        return out

class MLP_siren(nn.Module):
    def __init__(self,
                 num_layers,
                 dim_hidden,
                 dim_in=1,
                 dim_out=1,
                 w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        super(MLP_siren, self).__init__()

        self.n_layers = num_layers
        self.n_units = dim_hidden
        self.n_inputs = dim_in
        self.neurons_out = dim_out

        self.w0 = w0
        self.w0_initial = w0_initial
        self.use_bias = use_bias
        self.final_activation = final_activation

        self.net = self.__make_net()

    def __make_net(self):
        layers = []
        for i in range(self.n_layers):
            is_first = i == 0
            layer_w0 = self.w0_initial if is_first else self.w0
            layer_dim_in = self.n_inputs if is_first else self.n_units
            layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=self.n_units,
                w0=layer_w0,
                use_bias=self.use_bias,
                is_first=is_first
            ))

        final_activation = nn.Identity() if not exists(self.final_activation) else self.final_activation
        layers.append(Siren(dim_in = self.n_units, dim_out = self.neurons_out, w0 = self.w0, use_bias = self.use_bias, activation = final_activation))
        return nn.Sequential(*layers)

    def forward(self, t, xy):
        txy = torch.concat([t, xy], dim=-1)

        out = self.net(txy)
        return out

class MLP_ODE_siren(nn.Module):
    def __init__(self,
                 num_layers,
                 dim_hidden,
                 dim_in=1,
                 dim_out=1,
                 w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        super(MLP_ODE_siren, self).__init__()

        self.n_layers = num_layers
        self.n_units = dim_hidden
        self.n_inputs = dim_in
        self.neurons_out = dim_out

        self.w0 = w0
        self.w0_initial = w0_initial
        self.use_bias = use_bias
        self.final_activation = final_activation

        self.net = self.__make_net()

    def __make_net(self):
        layers = []
        for i in range(self.n_layers):
            is_first = i == 0
            layer_w0 = self.w0_initial if is_first else self.w0
            layer_dim_in = self.n_inputs if is_first else self.n_units
            layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=self.n_units,
                w0=layer_w0,
                use_bias=self.use_bias,
                is_first=is_first
            ))

        final_activation = nn.Identity() if not exists(self.final_activation) else self.final_activation
        layers.append(Siren(dim_in = self.n_units, dim_out = self.neurons_out, w0 = self.w0, use_bias = self.use_bias, activation = final_activation))
        return nn.Sequential(*layers)

    def forward(self, xy):
        out = self.net(xy)
        return out

def exists(val):
    return val is not None

class MLP_sirenlike(nn.Module):
    def __init__(self,
                 num_layers,
                 dim_hidden,
                 dim_in=1,
                 dim_out=1,
                 w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        super(MLP_sirenlike, self).__init__()

        self.n_layers = num_layers
        self.n_units = dim_hidden
        self.n_inputs = dim_in
        self.neurons_out = dim_out
        self.use_bias = use_bias
        self.final_activation = final_activation

        self.net = self.__make_net()

    def __make_net(self):
        layers = []
        for i in range(self.n_layers):
            is_first = i == 0
            layer_dim_in = self.n_inputs if is_first else self.n_units
            layers.append(nn.Linear(
                in_features=layer_dim_in,
                out_features=self.n_units,
                bias=self.use_bias
            ))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(
                in_features=self.n_units,
                out_features=self.neurons_out,
                bias=self.use_bias
            ))
        return nn.Sequential(*layers)

    def forward(self, t, xy):
        txy = torch.concat([t, xy], dim=-1)

        out = self.net(txy)
        return out