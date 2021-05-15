import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression


class LinearRegression():
    """A simple linear regression model base on this source : https://machinelearnia.com/regression-lineaire-python/ """
    def __init__(self):
        pass

    def model(self, X, θ):
        "Computes a prediction"
        return X.dot(θ)
    
    def cost_function(self, X, y, θ):
        """Compute the cost"""
        m = len(y)
        return 1/(2*m) * np.sum((self.model(X, θ) - y)**2)

    def grad(self, X, y, θ):
        """ Compute the gradiant of the cost function to minimize its value"""
        m = len(y)
        return 1/m * X.T.dot(self.model(X, θ) - y)

    def train(self, X, y, α, epochs):
        """ Training algorithme using gradient descent
        PARAMETERS :
        - X (ndarray) : Features array;
        - Y (ndarray) : Targets array;
        - α (float) : Learning rate;
        - epochs (int) : Number of training epochs
        OUTPUT :
            - θ (ndarray) : The trained model parameters array;
            - history (list) : An history of cost value evolution during the training;"""

        self.X = X
        self.Y = y.reshape(y.shape[0], 1)
        self.θ = np.random.randn(2, 1)
        self.α = α
        self.epochs = epochs

        history = []
        for i in range(epochs):
            self.θ -= α * self.grad(self.X, y, self.θ)
            history.append(self.cost_function(self.X, y, self.θ))
        history = [(i, history[i]) for i in range(len(history))]
        return history
    
    def predict(self, x):
        X = x
        for i in range(self.d-1, -1, -1):
            X = np.hstack((X, x**i)) # Add a biais column to x
        return self.model(X, self.θ)
    
    def get_param(self):
        return self.θ

class PolyRegression(): 
    """A simple Polynomial regression model base on this source : https://machinelearnia.com/regression-lineaire-python/ """
    def __init__(self):
        pass

    def model(self, X, θ):
        "Computes a prediction"
        return X.dot(θ)
    
    def cost_function(self, X, y, θ):
        """Compute the cost"""
        m = len(y)
        return 1/(2*m) * np.sum((self.model(X, θ) - y)**2)

    def grad(self, X, y, θ):
        """ Compute the gradiant of the cost function to minimize its value"""
        m = len(y)
        return 1/m * X.T.dot(self.model(X, θ) - y)

    def train(self, X, y, d, α, epochs):
        """ Training algorithme using gradient descent
        PARAMETERS :
        - X (ndarray) : Features array;
        - Y (ndarray) : Targets array;
        - d (int) : Polynom degree;
        - α (float) : Learning rate;
        - epochs (int) : Number of training epochs
        OUTPUT :
            - θ (ndarray) : The trained model parameters array;
            - history (list) : An history of cost value evolution during the training;"""

        self.X = X
        for i in range(d-1, -1, -1):
            self.X = np.hstack((self.X, X**i)) # Add a biais column to x
        print(self.X.shape)
        self.Y = y.reshape(y.shape[0], 1)
        self.d = d
        self.θ = np.random.randn(d+1, 1)
        self.α = α
        self.epochs = epochs

        history = []
        for i in range(epochs):
            self.θ -= α * self.grad(self.X, y, self.θ)
            history.append(self.cost_function(self.X, y, self.θ))
        history = [(i, history[i]) for i in range(len(history))]
        return history
    
    def predict(self, x):
        X = x
        for i in range(self.d-1, -1, -1):
            X = np.hstack((X, x**i)) # Add a biais column to x
        return self.model(X, self.θ)
    
    def get_param(self):
        return self.θ
