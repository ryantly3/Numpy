import numpy as np
# X          - vector
# y          - vector
# theta      - vector
# alpha      - scalar
# iterations - scalar

def gradientDescent(X, y, theta, alpha, num_iters):
    '''
    This function returns a tuple (theta, cost_array)
    '''
    m = len(y)
    cost_array =[]

    for i in range(0, num_iters):
        ################ START TODO #################
        # Make predictions
        # Hint: y_hat = theta_0 + (theta_1 * x_1) + (theta_2 * x_2)
        # Shape of y_hat: m by 1
        y_hat = X.dot(theta)


        # Compute the difference between predictions and true values
        # Shape of residuals: m by 1
        residuals = y_hat - y

        # Calculate the current cost
        cost = (1/(2*m))*np.sum(residuals**2)
        cost_array.append(cost)

        # Compute gradients
        # Shape of gradients: 3 by 1, i.e., same as theta
        X_t = np.transpose(X)
        gradients = (1/m)*np.dot(X_t,residuals)

        # Update theta
        theta = theta - alpha*gradients
        ################ END TODO ##################

    return theta, cost_array
