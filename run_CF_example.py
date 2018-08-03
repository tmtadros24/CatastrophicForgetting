import numpy as np
import network
import tensorflow as tf
import matplotlib.pyplot as plt

# load mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
training_data = zip(mnist.train.images, mnist.train.labels)
test_data = zip(mnist.test.images, mnist.test.labels)

train_labels = np.argmax(mnist.train.labels, axis=1)
test_labels = np.argmax(mnist.test.labels, axis=1)

# Create base and incremental data
excluded_class = 6
base_train_indices = [i for i in range(len(train_labels)) if train_labels[i] in [0,1,2,3,4]]
base_test_indices = [i for i in range(len(test_labels)) if test_labels[i] in [0,1,2,3,4]]
incremental_train_indices = [i for i in range(len(train_labels)) if train_labels[i] in [5, 6, 7, 8, 9]]
incremental_test_indices = [i for i in range(len(test_labels)) if test_labels[i] in [5, 6, 7, 8, 9]]

base_training_data = zip(mnist.train.images[base_train_indices,:], 
							mnist.train.labels[base_train_indices,:])

base_test_data = zip(mnist.test.images[base_test_indices,:], 
							mnist.test.labels[base_test_indices,:])

incremental_training_data = zip(mnist.train.images[incremental_train_indices,:], 
							mnist.train.labels[incremental_train_indices,:])

incremental_test_data = zip(mnist.test.images[incremental_test_indices,:], 
							mnist.test.labels[incremental_test_indices,:])

# Initialize and train network on base data
net = network.Network([784, 256, 10], cost=network.CrossEntropyCost)
eval_cost, eval_acc1, train_cost, train_acc = net.SGD(base_training_data, 15, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

print net.accuracy(base_test_data)
print net.accuracy(incremental_test_data)

# save and plot weights and biases
net.save("base_network_2batches.npy")

# Train network on incremental task
eval_cost, eval_acc2, train_cost, train_acc = net.SGD(incremental_training_data, 15, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
print net.accuracy(base_test_data)
print net.accuracy(incremental_test_data)
net.save("incremental_network_2batches.npy")



