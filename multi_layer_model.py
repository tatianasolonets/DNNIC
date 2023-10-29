from helper_functions import *

"""
    Three-layer neural network: 
    LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X - input data, of shape (2, number of examples)
    Y - true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate - learning rate for gradient descent 
    num_iterations - number of iterations to run gradient descent
    print_cost - if True, print the cost every 1000 iterations
    init_method - initialization method
    lambda_regularization - regularization hyperparameter, scalar
    keep_prob_for_dropout - probability of keeping a neuron active during drop-out, scalar.
    
    Returns:
    parameters -- parameters learnt by the model
    """

def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, init_method = InitMethod.HE, lambda_regularization = 0, keep_prob_for_dropout = 1):
    grads = {}
    costs = [] 
    # Number of examples
    m = X.shape[1]
    layers = [X.shape[0], 20, 3, 1]

    # Initialize parameters dictionary.
    parameters = init_params(layers, init_method)

    # Gradient descent.
    for i in range(0, num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob_for_dropout == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob_for_dropout < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        
        # Cost function
        if lambda_regularization == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambda_regularization)
            
        # Backward propagation.
        assert (lambda_regularization == 0 or keep_prob_for_dropout == 1)   # it is possible to use both L2 regularization and dropout, 
                                                # but this assignment will only explore one at a time
        if lambda_regularization == 0 and keep_prob_for_dropout == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambda_regularization != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambda_regularization)
        elif keep_prob_for_dropout < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob_for_dropout)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
    
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters