import numpy as np
import network
import tensorflow as tf
import matplotlib.pyplot as plt


net1 = network.load("base_network_2batches.npy")
net2 = network.load("incremental_network_2batches.npy")
weights = net1.weights
biases  = net1.biases

weights2 = net2.weights
biases2 = net2.biases
# plot first set of weights

fig = plt.figure()
plt.subplot(2,1,1)

im = plt.imshow(weights[0], interpolation='nearest', vmin=-2, vmax=2)
plt.title("Batch learning")
plt.subplot(2,1,2)
im = plt.imshow(weights2[0], interpolation='nearest', vmin=-2, vmax=2)
plt.title("Incremental learning")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()

# plot distribution of weights
fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(weights[0].flatten(), bins=200)
plt.xlabel("Weight")
plt.title("1st set")
plt.subplot(1,2,2)
plt.hist(weights2[0].flatten(), bins=200)
plt.title("2nd set")
plt.show()

# plot first set of weights
fig = plt.figure()
plt.subplot(2,1,1)
im = plt.imshow(weights[1], interpolation='nearest', vmin=-2, vmax=2)
plt.title("Batch learning")
plt.subplot(2,1,2)
im = plt.imshow(weights2[1], interpolation='nearest', vmin=-2, vmax=2)
plt.title("Incremental learning")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()

# plot distribution of weights
fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(weights[1].flatten(), bins=200)
plt.xlabel("Weight")
plt.title("1st set")
plt.subplot(1,2,2)
plt.hist(weights2[1].flatten(), bins=200)
plt.title("2nd set")
plt.show()


fig = plt.figure()
plt.scatter(np.arange(256), biases[0], label='Batch')
plt.scatter(np.arange(256), biases2[0], label='incremental')
plt.legend()
plt.title("Biases")
plt.show()
# plot biases
fig = plt.figure()
plt.scatter(np.arange(10), biases[1], label='Batch')
plt.scatter(np.arange(10), biases2[1], label='incremental')
plt.legend()
plt.title("Biases")
plt.show()