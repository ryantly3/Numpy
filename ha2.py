import numpy as np
from numpy.linalg import inv

import matplotlib
import sys
if sys.platform == 'darwin':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from GD import gradientDescent



# NOTE: Modify the codes between "START TODO" and "END TODO"
# There are two PLACEHODERS IN THIS SCRIPT


################ START TODO ################
# Sub-task 4)
# Test multiple learning rates and report their convergence curves.
ALPHA = 0.5
MAX_ITER = 500
################ END TODO ##################
print('ALPHA = {}'.format(ALPHA))
print('MAX_ITER = {}'.format(MAX_ITER))


# Load data and divide it into training set and testing set
data = np.loadtxt(open('sat_gpa.csv'), delimiter=',')
print('Shape of original data:', data.shape) # Check if data is 105 by 3

# Normalize data
data_norm = data / data.max(axis=0)


# training data
data_train = data_norm[0:60, :]
# testing data
data_test = data_norm[60:,:]


# Sub-task 1)
# Obtain model parameters using normal equation

# Get matrix X, adding a column of 1s to X, and vector 1
X = np.ones_like(data_train)
X[:, 1:3] = data_train[:, 0:2]
y = data_train[:, -1]

################ START TODO ################
# Hint: theta = (X^T X)^{-1} X^T y
X_t = np.transpose(X)
theta_method1 = np.dot(inv(X_t.dot(X)),(X_t.dot(y)))
################ END TODO ##################
print('theta_method1:', theta_method1)

# Compute the final residuls of using theta_method1
print('Residual sum of squares (RSS) of method 1: ', np.sum((X.dot(theta_method1) - y)**2))


# Sub-task 2)
# Learn model paramters from Gradient Descent

# theta is a vector of length 3, [theta_0, theta_1, theta_2]
# Initialize theta to [0,0,0]
theta = np.zeros(3)
# call the gradient descent function
# NOTE: There are TODOs in the body of gradientDescent function in GD.py
theta_method2, cost_array = gradientDescent(X, y, theta, ALPHA, MAX_ITER)

print('theta_method2:', theta_method2)
# Compute the final residuls of using theta_method2
print('Residual sum of squares (RSS) of method 2: ', np.sum((X.dot(theta_method2) - y)**2))

# Plot cost against iteration number
plt.plot(range(0,len(cost_array)), cost_array);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {}, resulting theta = {}'.format(ALPHA, theta_method2))
# plt.show()
plt.savefig('cost_iter_alpha{}.pdf'.format(ALPHA))


# Sub-task 3)
# Evaluate on testing data and compute mean square errors
X_test = np.ones_like(data_test)
X_test[:, 1:3] = data_test[:, 0:2]
y_test = data_test[:, -1]

################ START TODO ################
# Hint: make predictions using X_test and theta_method1
yhat_test_1 = X_test.dot(theta_method1)
mse_1 = np.mean((yhat_test_1 - y_test)**2)
print('MSE of theta_method1 on testing data: ', mse_1)

# Hint: make predictions using X_test and theta_method2
yhat_test_2 = X_test.dot(theta_method2)
mse_2 = np.mean((yhat_test_2 - y_test)**2)
print('MSE of theta_method2 on testing data: ', mse_2)
################ END TODO ##################
