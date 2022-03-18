import random
"""Ordinary Least Squares Regression
   Space and time complexity:
   Visualize in plots"""
class Linear_Regression:
    def __init__(self):
        self.x = None
        self.y = None

    def fit(self, X, y, iterations=100, learning_rate=0.01):
        n, m = len(X[0], len(X))
        beta_0, beta_other = self.initialize_params(n)
        for _ in range(iterations):
            gradient_beta_0, gradient_beta_other = self.compute_gradient(
                X, y, beta_0, beta_other, n, m)
        beta_0, beta_other = self.update_params(beta_0, beta_other, gradient_beta_0,
        gradient_beta_other, learning_rate)
        return beta_0, beta_other
    
    def initialize_params(self, dimensions):
        beta_0 = 0
        beta_other = [random.random()
        for _ in range(dimensions)]
        return beta_0, beta_other

    def compute_gradient(X, y, beta_0, beta_other, dimension, n, m):
        gradient_beta_0 = 0
        gradient_beta_other = [0] * dimension

        for i in range(m):
            y_i_hat = sum(X[i][j] * beta_other[j]
            for j in range(dimension)) + beta_0 #the equation
            derror_dy = 2 * (y[i] - y_i_hat)
            for j in range(dimension):
                gradient_beta_other[j] += derror_dy * X[i][j] / n
            gradient_beta_0 += derror_dy / n
        
        return gradient_beta_0, gradient_beta_other

    def update_params(self, beta_0, beta_other, gradient_beta_0, gradient_beta_other, learning_rate):
        beta_0 += gradient_beta_0 * learning_rate
        for i in range(len(beta_0)):
            beta_other[i] += (gradient_beta_other[i] * learning_rate)
        return beta_0, beta_other


