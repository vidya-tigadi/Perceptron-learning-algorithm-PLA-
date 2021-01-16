import numpy as np

######## Perceptron algorithm ###########
class Perceptron:
	def __init__(self):
		self.label_set = None
		self.weights = np.zeros(3) #initialise weights to zero
####### compute weights #######################
	def compute_weights(self,features, labels):
		self.label_set = list(set(labels))
		self.label_set.sort()           # here [-1,1]
		for data,label in zip(features, labels):
			label_calculated =self.function_calculate(data)
			if label_calculated < label:
				self.weights += data  # increase value of weights for positive label
			elif label_calculated > label:
				self.weights -= data  #decrease value of weights for negative label
				
	def function_calculate(self, data):
		function = np.dot(data, self.weights)  # function of feature to get predicted label 
		if function > 0:
			return self.label_set[1]
		else:
			return self.label_set[0]					
			
	def wrong_prediction(self,features, labels):
		wrong = 0                              # counter for wrong predictions 
		for data, label in zip(features, labels):
			prediction = self.function_calculate(data)
			if prediction != label:
				wrong += 1
		print(wrong)		
		return wrong		
				
