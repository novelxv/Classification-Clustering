import numpy as np

class LogisticRegression2:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=None, lambda_=0.01, loss_function="cross_entropy"):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.loss_function = loss_function
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cross_entropy_loss(self, h, y, theta, m):
        return (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    
    def exponential_loss(self, h, y, theta, m):
        return (1/m) * np.sum(np.exp(-y * h))

    def compute_cost(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(np.dot(X, theta))
        if self.loss_function == "cross_entropy":
            cost = self.cross_entropy_loss(h, y, theta, m)
        elif self.loss_function == "exponential":
            cost = self.exponential_loss(h, y, theta, m)

        if self.regularization == "l1":
            cost += (self.lambda_ / (2 * m)) * np.sum(np.abs(theta[1:]))
        elif self.regularization == "l2":
            cost += (self.lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)

        return cost
    
    def gradient_descent(self, X, y, theta):
        m = len(y)
        for i in range(self.num_iterations):
            h = self.sigmoid(np.dot(X, theta))
            gradient = np.dot(X.T, (h - y)) / m

            if self.regularization == "l1":
                gradient[1:] += (self.lambda_ / m) * np.sign(theta[1:])
            elif self.regularization == "l2":
                gradient[1:] += (self.lambda_ / m) * theta[1:]

            theta -= self.learning_rate * gradient

            if i % 100 == 0:
                cost = self.compute_cost(X, y, theta)
                # print("Iteration {}: Cost = {}".format(i, cost))

        return theta
    
    def fit(self, X, y):
        m, n = X.shape
        X = np.hstack((np.ones((m, 1)), X))
        self.theta = np.zeros(n + 1)
        self.theta = self.gradient_descent(X, y, self.theta)

    def predict_prob(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold
    
    def get_params(self, deep=True):
        return {"learning_rate": self.learning_rate, "num_iterations": self.num_iterations, "regularization": self.regularization, "lambda_": self.lambda_}
    
    def newton_method(self, X, y):
        m, n = X.shape
        X = np.hstack((np.ones((m, 1)), X))
        self.theta = np.zeros(n + 1)
        
        for i in range(self.num_iterations):
            h = self.sigmoid(np.dot(X, self.theta))
            gradient = np.dot(X.T, (h - y)) / m

            H = np.dot(X.T, np.dot(np.diag(h * (1 - h)), X)) / m

            if self.regularization == "l1":
                gradient[1:] += (self.lambda_ / m) * np.sign(self.theta[1:])
            elif self.regularization == "l2":
                H[1:, 1:] += (self.lambda_ / m) * np.identity(n)
                gradient[1:] += (self.lambda_ / m) * self.theta[1:]

            self.theta -= np.linalg.inv(H).dot(gradient)

            if i % 100 == 0:
                cost = self.compute_cost(X, y, self.theta)
                # print("Iteration {}: Cost = {}".format(i, cost))

    def fit_newton(self, X, y):
        self.newton_method(X, y)