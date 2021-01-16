import matplotlib.pyplot as plt

import numpy as np

from Perceptron import Perceptron



def write_weights():

	data = np.loadtxt("input1.csv", delimiter=',')

	features1 = data[:, [0, 1]]

	label = data[:, [2]]

	length = len(data)

	temp = np.ones(length)

	temp.shape = (length, 1)

	features = np.hstack((temp, features1))

	label = label.flatten()

	percept = Perceptron()

	op = open("output1.csv", "w")



	while(1):

		percept.compute_weights(features, label)

		convergence = percept.wrong_prediction(features, label)

		op.write("%d %d %d\n" %(percept.weights[1],percept.weights[2],percept.weights[0]))

		

		if convergence == 0:

			break

			

	return 0	





if __name__ == "__main__":

	write_weights()

	

	



