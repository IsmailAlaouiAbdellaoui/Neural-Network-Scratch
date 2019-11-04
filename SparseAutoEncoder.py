import numpy as np 

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return np.exp(x)/np.square((np.exp(x)+1))

def cost(prediction,actual):
#    return 0.5*np.square(prediction-actual)
    return prediction-actual

def error(target,prediction):
    return 0.5*np.sum(np.square(prediction-target))

epsilon = 0.01
mean_normal = 0
variance_normal = epsilon**2

shape_input = (1,8)
shape_w1 = (8,3)

shape_z1 = (1,3)
shape_w2 = (3,8)
bias1 = np.random.normal(loc=mean_normal,scale=variance_normal)
bias2 = np.random.normal(loc=mean_normal,scale=variance_normal)
learning_rate = 0.01
lambd = 0.001
weights1 = np.random.normal(loc=mean_normal,scale=variance_normal,size=shape_w1)
weights2 = np.random.normal(loc=mean_normal,scale=variance_normal,size=shape_w2)

#weights1 = np.random.uniform(-0.01, 0.01, shape_w1)
#weights2 = np.random.uniform(-0.01, 0.01, shape_w2)

def temp():
    #min_range_weights = -0.1
    #max_range_weights = 0.1
    #weights1 = np.random.uniform(low= min_range_weights,high = max_range_weights,size=shape_w1)
    weights1 = np.random.normal(loc=mean_normal,scale=variance_normal,size=shape_w1)
    print(weights1)
    #print(weights1)
    total_input = np.eye(shape_input[1])
    random_index = np.random.randint(0,shape_input[1])
    x_input = total_input[random_index].reshape(shape_input)
    print(x_input.shape)
    print(weights1.shape)
    z2 = np.matmul(x_input,weights1) + bias1
    print(z2.shape)
    a2 = sigmoid(z2)
    #weights2 = np.random.uniform(low= min_range_weights,high = max_range_weights,size=shape_w2)
    weights2 = np.random.normal(loc=mean_normal,scale=variance_normal,size=shape_w2)
    z3 = np.matmul(a2,weights2) + bias2
    prediction = sigmoid(z3)
    prediction_prime = sigmoid_prime(z3)
    delta = (prediction - x_input) * prediction_prime #f(y) - T
    print(delta.shape)
    test = np.matmul(delta,weights2.T)
    print(test.shape)
    
#temp()



def forward_propagation(x_input,weights1,weights2):
    #Activation of first layer
    h = np.matmul(x_input,weights1) + bias1
    f_h = sigmoid(h)
    f_h_prime = sigmoid_prime(h)
    
    #Activation of second layer
    y = np.matmul(f_h,weights2) + bias2
    f_y = sigmoid(y)
    f_y_prime = sigmoid_prime(y)
    return f_h,f_h_prime,f_y,f_y_prime

def backpropagation(x_input,f_h,f_h_prime,f_y,f_y_prime):
    global weights1
    global weights2
    global bias1
    global bias2
    #Update of weights1
    target = x_input
    dError_df_y = (f_y-target) # is the subtraction
    dError_dy = f_y_prime * dError_df_y
    delta1 =  np.matmul(dError_dy,weights2.T) * np.matmul(x_input.T,f_h_prime)
    #delta1 of the form
    #[dW1  dW3],
    #[dW2  dW4]    
#    weights1 = weights1 - learning_rate*delta1
    weights1 = weights1 - learning_rate*(delta1 + lambd*weights1)
    
    #Update of weights2
#    delta2 = dError_df_y * np.matmul(f_y_prime.T,f_h)
    delta2 = np.multiply(dError_df_y,np.matmul(f_y_prime.T,f_h).T)
    #delta2 of the form
    #[dW5  dW7],
    #[dW6  dW8]
#    weights2 = weights2 - learning_rate*delta2
    weights2 = weights2 - learning_rate*(delta2 + lambd*weights2)
    
    #Update of bias 1
#    dError_dbias1 = np.matmul(dError_dy,weights2.T)*np.matmul(np.ones(x_input.shape).T,f_h_prime)
    dError_dbias1 = np.matmul(dError_dy,weights2.T) * f_h_prime
    bias1 = bias1 - learning_rate * dError_dbias1[0][1]
    
    #Update of bias 2
    dError_dbias2 = np.multiply(dError_df_y,np.matmul(f_y_prime.T,np.ones(f_h.shape)).T)
    bias2 = bias2 - learning_rate * dError_dbias2[0][0]
    
def training(number_iterations):
    global weights1
    global weights2
    error_list = []
    rmse_list = []
    total_input = np.eye(shape_input[1])
    for i in range(number_iterations):
        random_index = np.random.randint(0,shape_input[1])
        x_input = total_input[random_index].reshape(shape_input)
        target = x_input
        f_h,f_h_prime,f_y,f_y_prime = forward_propagation(x_input,weights1,weights2)
        backpropagation(x_input,f_h,f_h_prime,f_y,f_y_prime)
        if(i%50 == 0):
            difference_error = error(f_y,target)
#            print(er)
            error_list.append(difference_error)
            rmse = np.sqrt(np.mean(np.power(f_y-target,2)))
            rmse_list.append(rmse)
#    fig = plt.plot(error_list)
    plt.plot(rmse_list)
    plt.show()



training(1000)


    