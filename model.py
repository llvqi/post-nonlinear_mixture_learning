import torch
from torch import nn
from torch.nn import functional as F

# Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self, input_d, structure, output_d):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()

        struc = [input_d] + structure + [output_d]

        for i in range(len(struc)-1):
            self.net.append(nn.Linear(struc[i], struc[i+1]))

    def forward(self, x):
        for i in range(len(self.net)-1):
            x = F.relu(self.net[i](x))

        # For the last layer
        y = self.net[-1](x)

        return y


# The PNL unmixing model
class PNL(nn.Module):
    def __init__(self, input_dim, f_size, g_size, k, n_sample, rho, device):
        super().__init__()

        # encoding
        self.e_net = nn.ModuleList()
        # decoding
        self.d_net = nn.ModuleList()
        # input dimension
        self.in_dim = input_dim
        # latent dimension
        self.latent_dim = k
        # multiplier
        self.mult = torch.randn(n_sample, dtype=torch.double, device=device)
        self.rho = rho

        ## Encoding network
        tmp = []
        # Input layer
        tmp.append(nn.Conv1d(self.in_dim, f_size[0]*self.in_dim, 1, groups=self.in_dim))
        tmp.append(nn.ReLU())

        # Hidden layer
        for j in range(1,len(f_size)):
            tmp.append(nn.Conv1d(f_size[j-1]*self.in_dim, f_size[j]*self.in_dim,
                1, groups=self.in_dim))
            tmp.append(nn.ReLU())

        # Output layer
        tmp.append(nn.Conv1d(f_size[-1]*self.in_dim, self.in_dim, 1, groups=self.in_dim))

        self.e_net = nn.Sequential(*tmp)

        # Unmixing layer
        #self.unmixing = nn.Linear(self.in_dim, self.latent_d, bias=False)

        ## Decoding network
        tmp = []
        # Input layer
        tmp.append(nn.Conv1d(self.in_dim, g_size[0]*self.in_dim, 1, groups=self.in_dim))
        tmp.append(nn.ReLU())

        # Hidden layer
        for j in range(1,len(g_size)):
            tmp.append(nn.Conv1d(g_size[j-1]*self.in_dim, g_size[j]*self.in_dim,
                1, groups=self.in_dim))
            tmp.append(nn.ReLU())

        # Output layer
        tmp.append(nn.Conv1d(g_size[-1]*self.in_dim, self.in_dim, 1, groups=self.in_dim))
        self.d_net = nn.Sequential(*tmp)

    # Encoding function
    def encode(self, x):
        y = self.e_net(x.unsqueeze(-1))

        return y.squeeze()

    # Decoding function
    def decode(self, x):
        y = self.d_net(x.unsqueeze(-1))

        return y.squeeze()
