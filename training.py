import torch
import numpy as np
import scipy.linalg as spalg
from torch import nn

# Loss function
def loss_function(model, fx, qfx, x, idxes):
    loss = nn.MSELoss(reduction='sum')

    tmp = torch.sum(fx, 1)-1.0

    mult = model.mult[idxes]

    reconstruct_err = loss(qfx, x) / x.shape[0]
    feasible_err = torch.dot(mult, tmp) / x.shape[0]
    augmented_err = torch.norm(tmp)**2 / x.shape[0]

    return reconstruct_err + feasible_err + model.rho/2.*augmented_err, \
            reconstruct_err, feasible_err, augmented_err


# Function that optimizes the network parameters
def update_network(epoch, model, args, train_loader, device, optimizer):
    model.train()
    train_loss = 0
    recons_err = 0
    feasible_err = 0
    augmented_err = 0
    total_num = 0

    for batch_idx, (data, idxes) in enumerate(train_loader):
        data = data.to(device)

        # Forward
        fx = model.encode(data)
        qfx = model.decode(fx)

        # Computer loss
        loss, r_e, f_e, a_e = loss_function(model, fx, qfx, data, idxes)
        train_loss += loss.item()*data.shape[0]
        recons_err += r_e.item()*data.shape[0]
        feasible_err += f_e.item()*data.shape[0]
        augmented_err += a_e.item()*data.shape[0]
        total_num += data.shape[0]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= total_num
    recons_err /= total_num
    feasible_err /= total_num
    augmented_err /= total_num

    print('====> Epoch: {} loss: {:.6f}, recons = {:.6f}, feasible = {:.6f}, augmented = {:.6f}'.format(epoch, train_loss, recons_err, feasible_err, augmented_err))


# Training function
def train(model, args, qs, train_loader, eval_loader, device, optimizer):
    best_constraint_val = float('inf')
    subspace_dist_arr = []

    for epoch in range(1, args.num_epochs + 1):
        # fix the multiplier
        for _ in range(args.inner_iters):
            update_network(epoch, model, args, train_loader, device, optimizer)

        # fix the neural networks
        model.eval()
        F = []
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(eval_loader):
                data = data.to(device)

                # forward
                fx = model.encode(data)
                F.append(fx)

            F = torch.cat(F, 0)

            diff = torch.sum(F, 1)-1.0
            model.mult += model.rho * diff

            squared_diff = torch.norm(diff)**2
            # Save the model if the constraint_val decreases
            if squared_diff < best_constraint_val:
                best_constraint_val = squared_diff
                print('Saving Model')
                torch.save(model.state_dict(), args.model_file_name)

            # Compute the subspace distance
            qf, _ = torch.qr(F)
            subspace_dist = np.sin(spalg.subspace_angles(qs, qf.cpu().numpy()))[0]
            subspace_dist_arr.append(subspace_dist)

    print(subspace_dist_arr)
