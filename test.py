# Make it possible to show plots in the notebooks.
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
np.set_printoptions(precision=3)

def fit_linear_with_regularization(X, y, alpha):
  # X = 422 entries with 10 features per entry
  # add 1 to the first columns
  X = np.insert(X, 0, 1, axis=1)

  #w* = ((XTX + Ia)^-1) (Xty) from question c)
  XTX=np.matmul(np.transpose(X),X)
  Ia = np.identity(XTX.shape[0]) * alpha
  inverse = np.linalg.inv(XTX + Ia)
  XTy = np.matmul(np.transpose(X), y)
  w_star = np.matmul(inverse, XTy)
  return w_star

def predict(X_test, w):
  X_test = np.insert(X_test, 0, 1, axis=1)
  return np.matmul(X_test, w)

def plot_prediction(X_test, y_test, y_pred):
  """TODO"""
  
  # Scatter plot the first feature of X_test (x-axis) and y_test (y-axis)
  plt.plot(X_test) # TODO: Change me
  fig, ax = plt.subplots()
  # ax.plot(X_test,y_test)
  # TODO: Plot y_pred using the first feature of X_test as x-axis
  
  # TODO: Compute the mean squared error
  mean_squared_error = ((y_pred - y_test)**2).mean()
  return mean_squared_error

# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True)

# Split the dataset into training and test set
num_test_elements = 20

X_train = X[:-num_test_elements]
X_test = X[-num_test_elements:]

y_train = y[:-num_test_elements]
y_test = y[-num_test_elements:]

# Set alpha
alpha = 0.01

# Train using linear regression with regularization and find optimal model
w = fit_linear_with_regularization(X_train, y_train, alpha)

# Make predictions using the testing set X_test
y_pred = predict(X_test, w)
# Plots and mean squared error
error = plot_prediction(X_test, y_test, y_pred)
print('Mean Squared error is ', error)

# Show the plot
plt.show()