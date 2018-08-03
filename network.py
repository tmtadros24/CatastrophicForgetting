'''
Neural network implementation based off of textbook "neural networks and deep learning"
'''
# Standard libraries
import json
import random
import sys

# Third-party libraries
import numpy as np


# Quadratic cost function
class QuadraticCost(object):
	
	@staticmethod
	def fn(a, y):
		''' return the cost associated with an output 'a' and desired output 'y'.
		'''
		return 0.5*np.linalg.norm(a-y)**2

	@staticmethod
	def delta(z, a, y):
		'''return the error delta from the output layer.'''
		return (a-y)*sigmoid_prime(z)


# Cross entropy cost function
class CrossEntropyCost(object):

	@staticmethod
	def fn(a, y):
		'''Return the cost associated with an output 'a' and desired output 'y'.
		'''
		return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

	@staticmethod
	def delta(z,a,y):
		'''return the error delta from output layer'''
		return (a-y)

### Main neural network class
class Network(object):
	def __init__(self, sizes, cost=CrossEntropyCost):
		'''The list 'sizes' contains the number of neurons in respective layers
		of the network. For example, [2,3,1] returns a 3 layer network, with the first
		layer containing 2 neurons, etc. The biases and weights for the network
		are initialized randomly.
		'''
		self.num_layers = len(sizes)
		self.sizes 		= sizes
		self.default_weight_initializer()
		self.cost 		= cost

	def default_weight_initializer(self):
		'''
		Initialize each weight using a gaussian distribbution with mean 0 and std 1 over
		square root of number of weights connecting to same neuron. Initialize biases
		using a gaussian with mean 0 and std 1.
		'''
		self.biases = [np.random.randn(y) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]

	def large_weight_initializer(self):
		'''
		Don't normalize by sqrt
		'''
		self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y,x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]


	def feedforward(self, a):
		'''Return outputof the network if 'a' is input.'''
		for b,w in zip(self.biases, self.weights):
			x = (np.dot(w,a) + b).shape
			a = sigmoid(np.dot(w,a) + b)
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta,
			lmbda=0.0,
			evaluation_data=None,
			monitor_evaluation_cost=False,
			monitor_evaluation_accuracy=False,
			monitor_training_cost=False,
			monitor_training_accuracy=False):
		'''Train the neural network using mini bbatch SGD. 
		The trainning data is a list of tuples (x,y) representing
		the training inputs and desired outputs.'''
		if evaluation_data: n_data = len(evaluation_data)
		n = len(training_data)
		evaluation_cost, evaluation_accuracy = [], []
		training_cost, training_accuracy = [], []
		
		for j in xrange(epochs):
			random.shuffle(training_data)
			
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in xrange(0, n, mini_batch_size)]

			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
			
			print "Epoch %s training complete" % j

			if monitor_training_cost:
				cost = self.total_cost(training_data, lmbda)
				training_cost.append(cost)
				print "Cost on training data: {}".format(cost)

			if monitor_training_accuracy:
				accuracy = self.accuracy(training_data, convert=True)
				training_accuracy.append(accuracy)
				print "Accuracy on training data: {} / {}".format(accuracy, n)

			if monitor_evaluation_cost:
				cost = self.total_cost(evaluation_data, lmbda, convert=True)
				evaluation_cost.append(cost)
				print "Cost on evaluationn data: {}".format(cost)

			if monitor_evaluation_accuracy:
				accuracy = self.accuracy(evaluation_data)
				evaluation_accuracy.append(accuracy)
				print "Accuracy on evaluation data: {} / {}".format(accuracy, n_data)

			print
		return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

	def update_mini_batch(self, mini_batch, eta, lmbda, n):
		'''Update the network's weights and biases by applying gradient
		descent using backprop on a single mini batch. Eta is learninng rate, 
		lmbda is regularization parameter and n is total size of training dataset.
		'''
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x,y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

		self.weights 	= [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
		self.biases 	= [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):
		'''Return a tuble (nabla_b, nabla_w) representing the gradient for the cost function.'''
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		# Feedforward
		activation 	= x
		activations = [x] # list to store activations layer by layer
		zs 			= []  # list to store all z vectors layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		
		# Backward pass
		delta = (self.cost).delta(zs[-1], activations[-1], y)
		nabla_b[-1] = delta
		nabla_w[-1] = np.outer(delta, activations[-2])
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.outer(delta, activations[-l-1])

		return (nabla_b, nabla_w)

	def accuracy(self, data):
		'''Return number of inputs in ''data'' for which
		NN outputs the correct result. The convert flag
		should be true if data is training data.'''
		results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x,y) in data]
		return sum(int(x==y) for (x,y) in results)

	def total_cost(self, data, lmbda):
		'''Return total cost for the data set'''
		cost = 0.0
		for x,y in data:
			a = self.feedforward(x)
			cost += self.cost.fn(a,y)/len(data)

		# Regularization cost
		cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
		
		return cost

	def save(self, filename):
		'''Save NN to file'''
		data = {"sizes": self.sizes,
				"weights": [w.tolist() for w in self.weights],
				"biases": [b.tolist() for b in self.biases],
				"cost": str(self.cost.__name__)}
		f = open(filename, "w")
		json.dump(data, f)
		f.close()


### loading a network
def load(filename):
	f = open(filename, "r")
	data = json.load(f)
	f.close()
	cost = getattr(sys.modules[__name__], data["cost"])
	net = Network(data["sizes"], cost=cost)
	net.weights = [np.array(w) for w in data["weights"]]
	net.biases = [np.array(b) for b in data["biases"]]
	return net

def sigmoid(z):
	'''Compute the sigmoid function'''
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	'''derivative of sigmoid function'''
	return sigmoid(z)*(1-sigmoid(z))



