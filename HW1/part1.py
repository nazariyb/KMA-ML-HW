import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class MyLinearRegression:
    def __init__(self, weights_init='random', add_bias = True, learning_rate=1e-5, 
        num_iterations=1_000, verbose=False, max_error=1e-5):
        ''' Linear regression model using gradient descent 

        # Arguments
            weights_init: str
                weights initialization option ['random', 'zeros']
            add_bias: bool
                whether to add bias term 
            learning_rate: float
                learning rate value for gradient descent
            num_iterations: int 
                maximum number of iterations in gradient descent
            verbose: bool
                enabling verbose output
            max_error: float
                error tolerance term, after reaching which we stop gradient descent iterations
        '''

        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.weights_init = weights_init
        self.add_bias = add_bias
        self.verbose = verbose
        self.max_error = max_error

        self.error_history = []
    
    def initialize_weights(self, n_features):
        ''' weights initialization function '''
        if self.weights_init == 'random':
            ################

            # YOUR CODE HERE
            weights = np.random.randn(n_features)

            ################
        elif self.weights_init == 'zeros':
            ################

            # YOUR CODE HERE
            weights = np.zeros(n_features)

            ################
        else:
            raise NotImplementedError
        return weights

    def cost(self, target, pred):
        ''' calculate cost function 
        
            # Arguments:
                target: np.array
                    array of target floating point numbers 
                pred: np.array
                    array of predicted floating points numbers
        '''
        ################

        # YOUR CODE HERE
        loss = np.mean((target - pred) ** 2)

        ################
        return loss

    def fit(self, x, y):
        if self.add_bias:
            ################

            # YOUR CODE HERE
            x = np.hstack([np.ones((x.shape[0], 1)), x])

            ################

        self.weights = self.initialize_weights(x.shape[1])

        y = y.flatten()

        for i in range(self.num_iterations):
            ################

            # YOUR CODE HERE
            # step 1: calculate current_loss value
            y_pred = self.predict(x).flatten()
            current_loss = self.cost(y, y_pred)

            # step 2: calculate gradient value
            gradient = (2 / x.shape[0]) * x.T @ (y_pred - y)

            # step 3: update weights using learning rate and gradient value
            self.weights -= self.learning_rate * gradient

            # step 4: calculate new_loss value
            y_pred_new = self.predict(x)
            new_loss = self.cost(y, y_pred_new)

            # step 5: if new_loss and current_loss difference is greater than max_error -> break;
            #         if iteration is greater than max_iterations -> break
            if self.verbose:
                print(f"Iteration {i+1}, Loss: {new_loss}")
            
            self.error_history.append(new_loss)

            if abs(new_loss - current_loss) < self.max_error:
                if self.verbose:
                    print(f"Converged after {i+1} iterations.")
                break
            ################
    
    def predict(self, x):
        ''' prediction function '''
        ################

        # YOUR CODE HERE
        # print(f"x: {x.shape}, self.weights: {self.weights.shape}")
        if self.add_bias and x.shape[1] < self.weights.shape[0]:
            ################

            # YOUR CODE HERE
            x = np.hstack([np.ones((x.shape[0], 1)), x])

        y_hat = x @ self.weights

        ################
        return y_hat



def normal_equation(X, y):
    ''' TODO: implement normal equation '''

    weights = np.linalg.pinv(X.T @ X) @ X.T @ y

    return weights



if __name__ == "__main__":
    # generating data samples
    x = np.linspace(-5.0, 5.0, 100)[:, np.newaxis]
    y = 29 * x + 40 * np.random.rand(100,1)

    # normalization of input data
    x /= np.max(x)

    img_dir = "img/"
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    plt.title('Data samples')
    plt.scatter(x, y)
    plt.savefig(img_dir + 'data_samples.png')


    # Sklearn linear regression model
    sklearn_model = LinearRegression()
    sklearn_model.fit(x, y)
    y_hat_sklearn = sklearn_model.predict(x)

    plt.title('Data samples with sklearn model')
    plt.scatter(x, y)
    plt.plot(x, y_hat_sklearn, color='g')
    plt.savefig(img_dir + 'sklearn_model.png')
    print('Sklearn MSE: ', mean_squared_error(y, y_hat_sklearn))

    # Your linear regression model
    my_model = MyLinearRegression(
        verbose=False,
        learning_rate=0.0002,
        num_iterations=20_000)
    my_model.fit(x, y)
    y_hat = my_model.predict(x)

    plt.title('Data samples with my model')
    plt.scatter(x, y)
    plt.plot(x, y_hat, color='r')
    plt.savefig(img_dir + 'my_model.png')
    print('My MSE: ', mean_squared_error(y, y_hat))

    # Normal equation
    weights = normal_equation(x, y)
    y_hat_normal = x @ weights

    plt.title('Data samples with normal equation')
    plt.scatter(x, y)
    plt.plot(x, y_hat_normal, color='b')
    plt.savefig(img_dir + 'normal_equation.png')
    print('Normal equation MSE: ', mean_squared_error(y, y_hat_normal))

    plt.close()

    plt.title('Error')
    plt.plot(np.arange(my_model.num_iterations), my_model.error_history, color='b')
    plt.savefig(img_dir + 'error_history')