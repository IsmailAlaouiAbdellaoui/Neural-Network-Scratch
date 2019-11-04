# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 19:58:14 2019

@author: Smail
"""

import numpy as np 
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return np.exp(x)/np.square((np.exp(x)+1))


def error(target,prediction):
    return 0.5*np.sum(np.square(prediction-target))

shape_input = (1,2)
shape_w1 = (2,2)

shape_z1 = (1,2)
shape_w2 = (2,2)
bias1 = 0.35
bias2 = 0.6
learning_rate = 0.5

w1 = 0.15
w2 = 0.2
w3 = 0.25
w4 = 0.3

w5 = 0.4
w6 = 0.45
w7 = 0.5
w8 = 0.55

t1 = 0.01
t2 = 0.99

x1 = 0.05
x2 = 0.1

x_input = np.array([[x1,x2]])

target = np.array([[t1,t2]])

weights1 = np.array([[w1,w3],
                     [w2,w4]])
    
weights2 = np.array([[w5,w7],
                     [w6,w8]])
   

lambd = 0.0001


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
    
    
    
def backpropagation(f_h,f_h_prime,f_y,f_y_prime):
    global weights1
    global weights2
    global bias1
    global bias2
    
    #Update of weights1
    dError_df_y = (f_y-target) # is the subtraction
    dError_dy = f_y_prime * dError_df_y
    delta1 =  np.matmul(dError_dy,weights2.T) * np.matmul(x_input.T,f_h_prime)
    #delta1 of the form
    #[dW1  dW3],
    #[dW2  dW4]    
#    weights1 = weights1 - learning_rate*delta1
    weights1 = weights1 - learning_rate*(delta1 + lambd*weights1)
    
    #Update of weights2
#    print(dError_df_y.shape)
#    print(np.matmul(f_y_prime.T,f_h).shape)
#    delta2 = dError_df_y * np.matmul(f_y_prime.T,f_h)
    delta2 = np.multiply(dError_df_y,np.matmul(f_y_prime.T,f_h))
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
    
#def training(number_iterations, initial_weights1, initial_weights2):
def training(number_iterations):
    global weights1
    global weights2
    error_list = []
    for i in range(number_iterations):
        f_h,f_h_prime,f_y,f_y_prime = forward_propagation(x_input,weights1,weights2)
        backpropagation(f_h,f_h_prime,f_y,f_y_prime)
        if(i%10 == 0):
            error_list.append(error(f_y,target))
    plt.plot(error_list,'x')
    plt.show()
        

def testing():
    global weights1
    global weights2
    f_h,f_h_prime,f_y,f_y_prime = forward_propagation(x_input,weights1,weights2)
    
print("Weights before backpropagation:")
print("Weights1:")
print(weights1)
print("Weights2:")
print(weights2)
#training(1)
print()
print("Weights after backpropagation:")
print("Weights1:")
print(weights1)
print("Weights2:")
print(weights2)
print("bias1",bias1)

shape_input = (1,8)
augmented_input_shape = (1,9)
total_input = np.eye(shape_input[1])
random_index = np.random.randint(0,shape_input[1])
x_input = total_input[random_index].reshape(shape_input)
print(x_input)
#print(np.append(x_input,1))
test = np.append(x_input,1).reshape(augmented_input_shape)
print(test)


    
    
    
    
    
    
    
    
    
    
    

    
