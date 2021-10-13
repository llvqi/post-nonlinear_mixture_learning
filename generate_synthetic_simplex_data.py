import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def sigmoid(x):
    return 1./(1. + np.exp(-x))

matplotlib.rc('xtick', labelsize=30)
matplotlib.rc('ytick', labelsize=30)

# Random seed
np.random.seed(8)

# The mixing matrix
A = np.random.randn(3,3)

# Number of samples
N = 5000

# Generate simplex data of 3 dim
latent = np.random.uniform(0.0, 1.0, (N, 3) )
s = latent/np.sum(latent, axis = 1, keepdims = True)

# Linear mixture, scaled for better visualization
mixture = s.dot(A)*np.array([5,4,1])

# QR decomposition of the normalized_latent, q will be used to compute
# the subspace distance (evaluation of success, see the experiment section)
q, _ = np.linalg.qr(s)

# Nonlinear mixture
nonlinear_mixture = np.zeros_like(mixture)

# Nonlinear function for the 1st dimension
nonlinear_mixture[:,0] = 5*sigmoid(mixture[:,0])+0.3*mixture[:,0]
plt.scatter(mixture[:,0], nonlinear_mixture[:,0])
plt.show()

# Nonlinear function for the 2nd dimension
nonlinear_mixture[:,1] = -3*np.tanh(mixture[:,1])-0.2*mixture[:,1]
plt.scatter(mixture[:,1], nonlinear_mixture[:,1])
plt.show()

# Nonlinear function for the 3rd dimension
nonlinear_mixture[:,2] = 0.4*np.exp((mixture[:,2]))
plt.scatter(mixture[:,2], nonlinear_mixture[:,2])
plt.show()

# Save the data as .mat file
sio.savemat('post-nonlinear_simplex_synthetic_data.mat',
        {'x': nonlinear_mixture, 's': s, 's_q': q, 'linear_mixture': mixture})
