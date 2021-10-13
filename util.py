from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch

# Dataset class
class MyDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
        self.data_len = data_tensor.shape[0]

    def __getitem__(self, index):
        return self.data[index], index

    def __len__(self):
        return self.data_len


# For better visaulazation
def visual_normalization(x):
    bound = 10
    x = x-np.amin(x)
    x = x/np.amax(x)*bound

    return x


# Evaluate the learned model
def evaluate(model, args, eval_loader, device, mix, x):
    model.load_state_dict(torch.load(args.model_file_name))
    model = model.to(device)
    model = model.double()

    # Forward
    with torch.no_grad():
        F = []
        for batch_idx, (data, _) in enumerate(eval_loader):
            data = data.to(device)
            fx = model.encode(data)
            F.append(fx)

        F = torch.cat(F, 0)
        F = F.cpu().numpy()

    # Scatter plot the results
    for i in range(mix.shape[1]):
        plt.subplot(1, mix.shape[1], i+1)
        # Plot the composition f\circ g
        plt.scatter(mix[:,i], visual_normalization(F[:,i]),
                label='$\hat{f}_'+str(i+1)+'\circ g_'+str(i+1)+'$')
        # Plot the generative function g
        plt.scatter(mix[:,i], visual_normalization(x[:,i]), label='$g_'+str(i+1)+'$')

        plt.xlabel('input',fontsize=20,fontweight='bold')
        if i==0:
            plt.ylabel('output',fontsize=20,fontweight='bold')

        plt.legend(fontsize=20)

    plt.show()
