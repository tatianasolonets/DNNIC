import numpy as np
import copy
from enum import Enum

class InitMethod(Enum):
    ZEROS = 1
    RANDOM = 2
    HE = 3

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

"""
    Arguments:
    layers - list containing the size of each layer.
    init_method - initalize method type
    
    Returns:
    dictionary containing parameters "W1", "b1", ..., "WL", "bL":
        W1 - weight matrix of shape (layers_dims[1], layers_dims[0])
        b1 - bias vector of shape (layers_dims[1], 1)
        ...
        WL - weight matrix of shape (layers_dims[L], layers_dims[L-1])
        bL - bias vector of shape (layers_dims[L], 1)
"""
def init_params(layers, init_method = InitMethod.HE):
    # Number of layers in the network.
    result_parameters = {}
    # Number of layers in the network.
    L = len(layers)

    match init_method:
        case InitMethod.ZEROS:
            for l in range(1, L):
                result_parameters['W' + str(l)] = np.zeros((layers[l], layers[l-1]))
                result_parameters['b' + str(l)] = np.zeros((layers[l], 1))
            return
        case InitMethod.RANDOM:
            # Set the random seed to 3
            rng = np.random.default_rng(seed=3)
            for l in range(1, L):
                result_parameters['W' + str(l)] = rng.standard_normal(layers[l], layers[l-1]) * 10
                result_parameters['b' + str(l)] = np.zeros((layers[l], 1))
            return
        case InitMethod.HE:
            for l in range(1, L + 1):
                result_parameters['W' + str(l)] = rng.standard_normal(layers[l], layers[l-1]) * np.sqrt(2./layers[l-1])
                result_parameters['b' + str(l)] = np.zeros((layers[l], 1))
            return
    return result_parameters

"""
Forward propagation : LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
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
def forward_propagation(X, parameters):
     m = X.shape[1]
     
     # FORWARD PROPAGATION (FROM X TO COST)
     A = sigmoid(np.dot(w.T, X) + b) 
     cost = - 1/ m * (np.dot(Y, np.log(A).T) + np.dot((1 - Y), np.log(1 - A).T))
     cost = np.squeeze(np.array(cost))
     return cost

"""
   Forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    np.random.seed(1)
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    #(≈ 4 lines of code)         # Steps 1-4 below correspond to the Steps 1-4 described above. 
    # D1 =                                           # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    # D1 =                                           # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    # A1 =                                           # Step 3: shut down some neurons of A1
    # A1 =                                           # Step 4: scale the value of neurons that haven't been shut down
    # YOUR CODE STARTS HERE
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = (D1 < keep_prob).astype(int)
    A1 = np.multiply(A1, D1)
    A1 = A1 / keep_prob
    # YOUR CODE ENDS HERE
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    #(≈ 4 lines of code)
    # D2 =                                           # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    # D2 =                                           # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    # A2 =                                           # Step 3: shut down some neurons of A2
    # A2 =                                           # Step 4: scale the value of neurons that haven't been shut down
    # YOUR CODE STARTS HERE
    D2 =  np.random.rand(A2.shape[0], A2.shape[1])
    D2 = (D2 < keep_prob).astype(int)
    A2 = np.multiply(A2, D2)
    A2 = A2 / keep_prob
    # YOUR CODE ENDS HERE
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache