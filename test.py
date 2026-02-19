from sklearn.datasets import make_gaussian_quantiles
import numpy as np
from matplotlib import pyplot as plt
import math
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
  
class GaussianNaiveBayes():
    """
    Custom implementation of a Gaussian Naive Bayes classifier.
    The parameters are estimated using MLE.
    """

    def __init__(self):
        """
        Parameter(s):
        """
        self.mean = {}
        self.std = {}
        self.prior = {}
        self.unique_classes = 0

    def get_class_parameters(self, X_class):
        """
        Estimating the MLE of the parameters.
        --------------------
        Input(s):
            X_class - Data points corresponding to a single class
        --------------------
        Output(s):
            mean_MLE - MLE of mean
            std_MLE  - MLE of scale
        """

        mean_MLE = np.mean(X_class, axis=0)
        std_MLE = np.std(X_class, axis=0)

        return mean_MLE, std_MLE

    def fit(self, X_train, y_train):
        """
        Compute model parameters using maximum likelihood estimates and a class size prior.
        --------------------
        Input(s):
            X_train   - Data of size (n_samples, n_features)
            y_train   - True labels of size (n_samples,1)
        --------------------
        Output(s)
        """

        # Compute mean, variance and prior of each class
        self.unique_classes = np.unique(y_train)
        for uc in self.unique_classes:
            X_class = X_train[y_train == uc]
            c_mean, c_std = self.get_class_parameters(X_class)
            self.mean[uc] = c_mean
            self.std[uc] = c_std
            self.prior[uc] = X_class.shape[0]/X_train.shape[0]

    def gaussian_density(self, x, mu, std):
        """
        1-D Gaussian density function.
        --------------------
        Input(s):
            x   - Data point
            mu  - mean
            std - scale
        --------------------
        Output(s):-
            N(mu, std^2)

        """
        return 1/(std*np.sqrt(2*np.pi))*np.exp(-(1/2)*((x-mu)/std)**2)

    def predict(self, X_test):
        """
        Prediction of test data.
        --------------------
        Input(s):
            X_test   - Data of size (n_samples, n_features)
        --------------------
        Output(s)
            y_pred - Predicted labels of size (n_samples,)
            y_pred_prob - Probabilistic labels of size (n_samples,n_classes)
        """

        n_samples = X_test.shape[0]
        y_pred_prob = np.zeros(shape=(n_samples,len(self.unique_classes)), dtype=np.float32)

        ### TODO - Change this part
        for i in range(n_samples):
            # we need to output [1,0] [0,1] [0.5,0.5]
            y_pred_prob[i,:] = [0] * len(self.unique_classes)
            for c in self.unique_classes:
              sample = X_test[i]
              top = math.prod([self.gaussian_density(xnd,self.mean[c][d],self.std[c][d]) for d,xnd in enumerate(sample)]) * self.prior[c]
              bottom = sum(
                math.prod([self.gaussian_density(xnd, self.mean[cPrime][d], self.std[cPrime][d]) for d,xnd in enumerate(sample)]) * self.prior[cPrime]
                          for cPrime in self.unique_classes)
              p = top/bottom
              y_pred_prob[i,c] = p
            #y_pred_prob[i,:] = [1,0] # Always predicts class 0 currently

        y_pred = np.argmax(y_pred_prob, axis=-1)
        return y_pred, y_pred_prob

# Define and train the model
model = GaussianNaiveBayes()
model.fit(X_train, y_train)

# Prediction
y_pred_train, _ = model.predict(X_train)
print("Train accuracy: %.3f %%" %(np.mean(y_pred_train == y_train)*100))
y_pred_test,y_pred_prob_test = model.predict(X_test)
print(f"Predicted test class: ${y_pred_test[0]}, (prob ${y_pred_prob_test[0,y_pred_test]})")

# Plot decision-boundaries
fig, ax = plot_data(X_train, y_train, X_test, title = 'Lines showing $P(y_{n} = 0 | x_{n}, X, y)$')
fig, ax = decision_boundary(model, fig, ax, labels = True)