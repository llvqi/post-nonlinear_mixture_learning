from __future__ import print_function
import argparse
import sys
import torch
from torch import optim
import scipy.io as sio
import numpy as np

import model as mdl
import training
import util

# Argument parser
def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--s_dim", default=3, help="Dimensionality of s", type=int)

    parser.add_argument("--batch_size", default=1000, help="Batch size", type=int)

    parser.add_argument("--num_epochs", default=20, help="Number of epochs", type=int)

    parser.add_argument("--inner_iters", default=100, help="Number of inner iterations", type=int)

    parser.add_argument("--learning_rate", default=1e-3, help="Learning rate", type=float)

    parser.add_argument("--rho", default=1e2, help="Value of rho", type=float)

    parser.add_argument("--model_file_name", default='best_model_simplex.pth',
            help="File name for best model saving", type=str)

    # Structure for encoder and decoder network
    parser.add_argument("--f_num_layers", default=3, help="Number of layers for f", type=int)

    parser.add_argument("--f_hidden_size", default=128, help="Number of hidden neurons for f",
            type=int)

    parser.add_argument("--q_num_layers", default=3, help="Number of layers for q", type=int)

    parser.add_argument("--q_hidden_size", default=128, help="Number of hidden neurons for q",
            type=int)


    return parser


# Main function, performing the training and evaluation
def main(args):
    parser = get_parser()
    args = parser.parse_args(args)

    torch.manual_seed(1)
    np.random.seed(12)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read data file
    data_file = './post-nonlinear_simplex_synthetic_data.mat'
    data = sio.loadmat(data_file)

    # Read input
    x = data['x']
    s_groundtruth = data['s']
    qs = data['s_q']
    mixture = data['linear_mixture']

    # Dimension of input
    n_sample = x.shape[0]
    n_feature = x.shape[1]

    # Build the PNL unmixing model
    model = mdl.PNL(n_feature, [args.f_hidden_size]*args.f_num_layers,
            [args.q_hidden_size]*args.q_num_layers, args.s_dim, n_sample,
            args.rho, device)
    model = model.to(device).double()

    # Construct optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Generate dataloader
    dataset = util.MyDataset(x)

    train_loader = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size, shuffle=True)

    eval_loader = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size, shuffle=False)

    # Call the training function
    training.train(model, args, qs, train_loader, eval_loader, device, optimizer)

    # Evaluate the learned model and plot the figures
    util.evaluate(model, args, eval_loader, device, mixture, x)


if __name__ == "__main__":
    main(sys.argv[1:])

