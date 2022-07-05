import numpy as np
from layer import Layer
from optimizers import Adam

class Dense(Layer, Adam):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.eta = 0.01

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        #self.weights -= learning_rate * weights_gradient
        #self.bias -= learning_rate * output_gradient
        Adam.update(self, t=0.1 ,w=self.weights, b=self.bias, dw=weights_gradient, db=output_gradient)
        return input_gradient

