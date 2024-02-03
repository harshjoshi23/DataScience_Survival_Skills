import pickle
import gzip
import numpy as np

def unit_vector(j):
    """This function converts a digit from (0,...,9) to a 10-dimensional
    unit vector with a 1.0 in the jth position and zeroes elsewhere, which
    corresponds to the desired output of the neural network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_MNIST(path):
	# load MNIST handwritten digits data using standard library
	with gzip.open(path, 'rb') as f:
		u = pickle._Unpickler(f)
		u.encoding = 'latin1'
		training_data, validation_data, testing_data = u.load()
	
	# testing data is tuple of length 2
	# containing two numpy nd arrays of size 
	# 10000 x 784 and 10000 x 1

	## convert loaded data in a format more suitable for artificial neural network training
	# training data
	training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
	training_results = [unit_vector(y) for y in training_data[1]]
	training_data = list(zip(training_inputs, training_results))
	
	# validation data
	validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
	validation_data = list(zip(validation_inputs, validation_data[1]))

	# testing data
	testing_inputs = [np.reshape(x, (784, 1)) for x in testing_data[0]]
	testing_data = list(zip(testing_inputs, testing_data[1]))
	
	return training_data, validation_data, testing_data
