from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
import numpy as np


from sklearn.datasets import make_gaussian_quantiles
import numpy as np
from matplotlib import pyplot as plt
import math
# Randomly generate data from two different distributions
data_class1, _ = make_gaussian_quantiles(mean = [1,1], cov = 1.5, n_samples = 30, n_features=2, random_state=18)
data_class2, _ = make_gaussian_quantiles(mean = [-1,-1], cov= 0.75,n_samples = 8, n_features=2, random_state=4)

# Concatenate the data, and add labels
X_train = np.append(data_class1, data_class2, axis=0)
y_train = np.append(np.zeros(len(data_class1), dtype=np.int32),
                   np.ones(len(data_class2), dtype=np.int32))

# Consider new test point
X_test = np.array([[-0.6,-0.4]])

def plot_data(X_train, y_train, X_test, title = ''):
    # Plot the two data classes
    fig,ax = plt.subplots(figsize=(10,10))
    ax.plot(X_train[y_train == 0][:,0], X_train[y_train == 0][:,1], 'o', markersize=8)
    ax.plot(X_train[y_train == 1][:,0], X_train[y_train == 1][:,1], 's', markersize=8)

    # Plot test point (circles to help with distances)
    ax.plot(X_test[:,0], X_test[:,1], '*', markersize=10)

    plt.axis('square')
    ax.legend(['Class 0','Class 1','test point'])
    ax.grid()
    ax.set_title(title, fontsize=15);
    return fig, ax

def decision_boundary(model, fig, ax, levels = [0.1,0.5,0.9], labels = True):
    # Code for producing the plot
    X1 = np.linspace(*ax.get_xlim(),100)
    X2 = np.linspace(*ax.get_ylim(), 100)
    Z = np.zeros(X1.shape+X2.shape)

    for i,x1 in enumerate(X1):
        for j,x2 in enumerate(X2):
            _, y_prob = model.predict(np.array([x1,x2]).reshape(1,-1))
            Z[j,i] = y_prob[0,0]

    contours = ax.contour(X1,X2,Z, levels=levels)
    if labels:
        ax.clabel(contours, inline=1, fontsize=10)

    return fig, ax
class LogisticRegression():
    """
    Custom implementation of (binary) Logistic Regression.
    """
    def __init__(self, learning_rate, n_iterations, print_cost=False):
        """
        Parameter(s):
            learning_rate - Learning rate
            n_iterations  - Number of iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.print_cost = print_cost
        self.parameters = {}

    def initialize_parameters(self, n_features):
        """
        Initialize model parameters with zeros:
            w.shape = (n_features,)
            b.shape = (1,)
        --------------------
        Input(s):
            n_features - Number of features
        --------------------
        Output(s):
        """

        w = np.zeros(shape = (n_features,1))
        b = 0

        self.parameters['w'] = w
        self.parameters['b'] = b

    def sigmoid(self, z):
        """
        Sigmoid function
        --------------------
        Input(s):
            z -
        --------------------
        Output(s):
            s - sigmoid(z)
        """
        s = 1 / (1 + np.exp(-z))
        return s

    def grad_cost(self, X_train, y_train, w, b):
        """
        Computes the cost function (negative log-likelihood) and
        partial derivatives of cost w.r.t the model parameters.
        --------------------
        Input(s):
            X_train - Data of size(n_samples, n_features)
            y_train - True labels of size (n_samples,1)
            w,b     - Model parameters
        --------------------
        Output(s):
            cost  - Negative log-likelihood cost
            grads - Gradients of loss function w.r.t model parameters (dw,db).
        """
        N = X_train.shape[0]

        y_train = y_train.reshape(-1,1)

        ### TODO - Change this part
        # Dummy variables currently
        A =  self.sigmoid(np.add(X_train @ w , b))
        cost = -1/N * sum([(y_train[n] * np.log(A[n]) + (1 - y_train[n])*np.log(1 - A[n])) for n in range(N)])
        dw = 1/N * X_train.T @ (A-y_train)
        db = 1/N * sum([(A[n] - y_train[n]) for n in range(N)])
        ###
        grads = {"dw": dw,
                 "db": db}
        return cost, grads

    def fit(self, X_train, y_train):
        """
        Optimize model parameters by running a gradient descent algorithm.
        --------------------
        Input(s):
            X_train - Data of size (n_samples, n_features)
            y_train - True labels of size (n_samples,1)
        --------------------
        Output(s)
        """
        n_features = X_train.shape[1]

        # Initialize parameters
        self.params = self.initialize_parameters(n_features)

        # Gradient descent
        w = self.parameters["w"]
        b = self.parameters["b"]
        for i in range(1,self.n_iterations+1):
            cost, grads = self.grad_cost(X_train, y_train, w, b)
            dw = grads["dw"]
            db = grads["db"]

            w -= self.learning_rate * dw
            b -= self.learning_rate * db

            if self.print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

        self.parameters = {"w": w,
                           "b": b}

        return self

    def predict(self, X_test):
        """
        Binary prediction of test data.
        --------------------
        Input(s):
            X   - Data of size (n_samples, n_features)
        --------------------
        Output(s)
            y_pred - Predicted labels (0/1) of size (n_samples,)
        """

        w = self.parameters["w"]
        b = self.parameters["b"]

        N = X_test.shape[0]
        y_pred = np.zeros((N,1))

        ### TODO - Change this part
        A = self.sigmoid(X_test @ w + b)
        r=0.5
        y_pred = (A >= r).astype(int)
        ###

        return y_pred.squeeze(), A
    

# Parameters (feel free to change)
learning_rate = 0.01
n_iterations = 1000
print_cost = True

# Define and train the model
model = LogisticRegression(learning_rate, n_iterations, print_cost)
model.fit(X_train, y_train)

# Predictions
y_pred_train, _ = model.predict(X_train)
print("Train accuracy: %.3f %%" %(np.mean(y_pred_train == y_train)*100))
y_pred_test,y_pred_prob_test = model.predict(X_test)
#printing the entire probabilty. if we do 0.2f we got 0.56
print(f"Predicted test class: {y_pred_test}, {np.where(y_pred_prob_test < 0.5, 1-y_pred_prob_test, y_pred_prob_test)}")

# Plot decision-boundaries
fig, ax = plot_data(X_train, y_train, X_test, title = 'Lines showing $P(y_{n} = 1 | x_{n}, X, y)$')
fig, ax = decision_boundary(model, fig, ax, labels = True)