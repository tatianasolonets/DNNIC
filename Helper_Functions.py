import numpy as np
import copy

"""
Sigmoid of x
Arguments: x - a scalar or numpy array
Returns: sigmoid of x
"""
def sigmoid(x):
     return 1 / (1 + np.exp(-x))

"""
Init Logistic Regression Parameters with zeros
z = wT * x + b
Arguments: n - size of the w vector (number of parameters)
Returns:  
w -- initialized vector of shape (n, 1)
b -- initialized scalar (bias) of type float
"""
def init_with_zeros(n):
    w = np.zeros((n, 1))
    b = 0.0
    return w, b

"""
Forward and backward propagation steps
Arguments:
    w - weights, a numpy array of size (num_px * num_px * 3, 1)
    b - bias, a scalar
    X - data of size (num_px * num_px * 3, number of examples)
    Y - "label" vector (containing 0 if no match, 1 if match) of size (1, number of examples)
Returns: 
grads - dictionary containing the gradients of the weights and bias
            (dw -- gradient of the loss with respect to w, thus same shape as w)
            (db -- gradient of the loss with respect to b, thus same shape as b)
cost - negative log-likelihood cost for logistic regression
"""
def propagate(w, b, X, Y):
     m = X.shape[1]
     
     # FORWARD PROPAGATION (FROM X TO COST)
     A = sigmoid(np.dot(w.T, X) + b) 
     cost = - 1/ m * (np.dot(Y, np.log(A).T) + np.dot((1 - Y), np.log(1 - A).T))
     
     # BACKWARD PROPAGATION (TO FIND GRAD)
     dw = 1/m * np.dot(X, (A - Y).T)
     db = 1/m * np.sum((A - Y))
     cost = np.squeeze(np.array(cost))

     grads = {"dw": dw,
             "db": db}
     return grads, cost

"""
This function optimizes w and b by running a gradient descent algorithm.
 1) Calculate the cost and the gradient for the current parameters. Use forward_prop().
 2) Update the parameters using gradient descent rule for w and b.

Arguments:
    w - weights, a numpy array of size (num_px * num_px * 3, 1)
    b - bias, a scalar
    X - data of shape (num_px * num_px * 3, number of examples)
    Y - true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations - number of iterations of the optimization loop
    learning_rate - learning rate of the gradient descent update rule
    print_cost - True to print the loss every 100 steps
    
    Returns:
    params - dictionary containing the weights w and bias b
    grads - dictionary containing the gradients of the weights and bias with respect to the cost function
    costs - list of all the costs computed during the optimization, this will be used to plot the learning curve.
"""
def run_gradient_descent(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
          # Cost and gradient calculation 
           grads, cost = propagate(w, b, X, Y)
           
           # Retrieve derivatives from grads
           dw = grads["dw"]
           db = grads["db"]
           
           # update rule
           w = w - np.dot(learning_rate, dw)
           b = b - np.dot(learning_rate, db)

           # Record the costs
           if i % 100 == 0:
            costs.append(cost)
        
            # Print the cost every 100 training iterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

"""
Predict whether the label is 0 or 1 using learned logistic egression parameters (w, b)

Arguments:
    w - weights, a numpy array of size (num_px * num_px * 3, 1)
    b - bias, a scalar
    X - data of size (num_px * num_px * 3, number of examples)

Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
"""
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X)+ b)

    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] > 0.5 :
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    return Y_prediction
