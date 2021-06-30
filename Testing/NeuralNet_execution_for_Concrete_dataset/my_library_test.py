#####################################################################
# Name :  Sushanth Keshav
# Topic : Implementing the learning process through my_neural_network
# Programming Language : Python
# Last Updated : 04.04.2020
#####################################################################

import numpy as np 
import matplotlib.pyplot as plt 
from neural_net_library import my_neural_network
import time

#The input for hidden_layer_dims must be a list containing number of neurons in each hiden layer
hidden_layer_dims = [5,5]
print('The Hidden Layer Dimensions : ',hidden_layer_dims)

#Enter the file name. Has to be a string with extension.
file_name = 'Concrete_Data.txt'
print("The file name is : ", file_name)

#Enter the hyper-parameters. Note that all the hyper-parameters are Non-Negative
epochs = 10000
lr = 0.01
# Mandatory for Momentum and Adam range: [0,1)
b1 = 0.9 
# Mandatory for RMSProp and Adam range: [0,1)
b2 = 0.999
activate = 'sigmoid'
optim = 'Momentum'



print("The learning rate is {}, beta1 = {}, beta2 = {}:".format(lr,b1,b2))
print("The activation function is :", activate)
print("The Optimizer function is :", optim)

#Start Timer
start_time = time.clock()

#Execution of the Neural Network Model
model = my_neural_network(hidden_layer_dims)
model.load_dataset(file_name)
model.test_train_split()
model.NN_model(epochs, lr,beta1=b1,beta2=b2,activation_function = activate,regularization = 0.01, batching = False, batch_size= 128,optimizer = optim,error_method = 'MSE',early_stop=True,tolerance=1e-4,learning_rate_decay=0.5,print_cost = False,plot=True)

#Stop timer
stop_time = time.clock()

#Evaluate the model on testing data
model.evaluate()

#Time Elapsed
total_time = stop_time - start_time
print("The total time taken for learning is : %.2f seconds" %(total_time))


           