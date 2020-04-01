#!/usr/bin/env python
# coding: utf-8
#Date: 22.03.2020

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#from tqdm import tqdm

# In[2]:


class my_neural_network:
    
    def __init__(self, hidden_layer_plan):
        
        '''
        Initialization function.
        
        input: 
        layer_plan : data_type = list
        
        List cointaing the number of neurons in every layers for the neural network. The first and last elements of
        layer_plan corresponds to number of input features and the output value. 
        The rest of the elements between them corresponds to the neurons in hidden layers.
        More the Hidden Layers implies deeper the network!
        '''
        
        #dataset
        '''Object to store the name of the dataset'''
        self.dataset = None
        
        #input and output data
        '''Objects to store the input and the coutput data. An attribute is additionally set to store the calculated output from the neural network'''
        self.input_data = None
        self.output_data = None
        self.output_data_noise = None
        self.predicted_output =None
        
        #train and test data
        '''Objects to store the Training data and Testing data. This is doen during the training and test data split'''
        self.training_data = None
        self.testing_data = None
        self.validation_data = None
        #layer details
        '''The attributes defining the deep network that is the number of hidden layers and its respective neurons '''
        self.hidden_layer_plan = hidden_layer_plan
        self.layer_dimensions = None
        self.num_of_layers = None
        self.num_of_examples = None
        self.epochs_number = None
        
        #Batching
        ''' List to store the mini batches'''
        self.mini_batches = []
        
        #Forward Activation
        ''' Objects to utilize during Forward Propagation'''
        self.activation_function_name = "sigmoid"
        self.activations, self.Z = [], [] # Lists to store the Activation outputs and their respective Linear Functions
        
        #Backward Activation
        self.dActivations = []
        
        #Error_method and loss
        self.error_method = "MSE"
        self.cost_derivative = None
        self.cost = []
        self.cost_value = None
        self.CV_cost = []
        self.regularization = 0.7
        
        #Network parameters
        self.network_weights, self.gradient_weights,self.momentum_weights, self.rms_weights = [],[],[],[]
        self.network_bias, self.gradient_bias,self.momentum_bias, self.rms_bias = [],[],[],[]
        
        #optimizer
        self.optimizer = None
        self.hyperparameters = []

    def __repr__(self):
        ''' Representative function to welcome the user for collaborating with the new AI trainer'''
        return 'Hello there!! I am your new AI trainer......!'
    
    
    def load_dataset(self, file, noise=False):
        ''' 
        Description:
        To load dataset 
        The current dataset is divided in such a way that only the last column is treated as output 
        while the others as input
        ...............................................................
        * Input Arguments: *
        1. file : data_type: string
           File name consisting the inputs and outputs. The last column in the file is the output.

        * Returns/Results: *
        1. self. dataset 
        2. self.input data
        3. self.output_data
        ..................................................................
        '''
        self.dataset = np.loadtxt(file)
        self.input_data = self.dataset[:,:-1]
        self.output_data = self.dataset[:,-1]
        self.output_data_noise = np.copy(self.output_data)
        #Adding noise to output_data
        if noise:
            np.random.seed(1)
            num_points = self.input_data.shape[0]
            noise = np.random.normal(loc=0, scale=1.0, size=num_points)
            self.output_data_noise += noise
         


        #Normalization
        '''
        Normalization helps to modulate the data such that the values lies along 0 and 1.
        This is helpful when using sigmoid functions or any activation functions
        '''
        max_inputs = np.amax(self.input_data, axis=0)
        min_inputs = np.amin(self.input_data, axis=0)
        diff_inputs = max_inputs - min_inputs
        self.input_data = np.divide((self.input_data - min_inputs),diff_inputs)
        
        max_outputs = np.amax(self.output_data, axis=0)
        min_outputs = np.amin(self.output_data, axis=0)
        diff_outputs = max_outputs - min_outputs
        self.output_data = np.divide((self.output_data - min_outputs), diff_outputs)

        max_outputs_noise = np.amax(self.output_data_noise, axis=0)
        min_outputs_noise = np.amin(self.output_data_noise, axis=0)
        diff_outputs_noise = max_outputs_noise - min_outputs_noise
        self.output_data_noise = np.divide((self.output_data_noise - min_outputs_noise), diff_outputs_noise)
        
        #Layer planning
        self.layer_dimensions = [self.input_data.shape[1]]+self.hidden_layer_plan+[1] #1 corresponds to columns in O/P
        self.num_of_layers = len(self.layer_dimensions)
        
    def test_train_split(self, split = 0.7, masking=True):
        '''
        Description:

        This function is utilized to segragate the complete data into training and testing data.
            - Helps in validating the result

        .........................................................................
        * Input Arguments : *
        
        1. split : Fraction of the entire dataset for Training
        2. Masking : helps in shuffling the data 
                     Default value = True
        3. self.... includes the input data and output data

        * Returns/Results : *
        1. Training Dataset : tuple containing training input and training output
        2. Testing Dataset : tuple containing testing input and testing output
        .........................................................................
        '''
        np.random.seed(1)
        n_total = int(self.dataset.shape[0])
        n_train = int(split*n_total)
        n_val_train = int(1.1*split*n_train)
        if masking:
            mask = np.zeros((n_total), dtype=bool)
            mask[:n_train] = True

            np.random.shuffle(mask)

            X_inp_train = self.input_data[mask]
            Y_out_train = self.output_data[mask]
            Y_out_train_noise = self.output_data_noise[mask]

            mask1 = np.zeros((n_train),dtype=bool)
            mask1[:n_val_train] = True

            np.random.shuffle(mask1)

            X_train = X_inp_train[mask1]
            Y_train = Y_out_train[mask1]
            Y_train_noise = Y_out_train_noise[mask1]

            X_val = X_inp_train[~mask1]
            Y_val = Y_out_train[~mask1]
            Y_val_noise = Y_out_train_noise[~mask1]

            X_test = self.input_data[~mask]
            Y_test = self.output_data[~mask]
            Y_test_noise = self.output_data_noise[~mask]
        else:
            X_train = self.input_data[:n_train,:]
            Y_train = self.output_data[:n_train]
            Y_train_noise = self.output_data_noise[:n_train]

            X_test = self.input_data[n_train:,:]
            Y_test = self.output_data[n_train:]
            Y_test_noise = self.output_data_noise[n_train :]

        self.training_data = (X_train.transpose(), Y_train_noise.transpose(),Y_train.transpose())
        self.validation_data = (X_val.transpose(), Y_val_noise.transpose(),Y_val.transpose())
        self.testing_data = (X_test.transpose(), Y_test_noise.transpose(), Y_test.transpose())
        self.num_of_examples = self.training_data[0].shape[1]
        
    def network_parameters_initialization(self):
        ''' 
        Description :

        Helps to initialize the network parameters i.e, Weights and Biases

        Shape of Weights = (Number of nodes in current layer, Number of nodes in previous layer)
        Shape of Biases = (Number of nodes in current layer, 1)

        Random seed(1) is used to initialize the same weights

        ..............................................................................
        * Input Arguments : *
        1. self ... includes the self. layer_dimensions

        * Returns/Results : *
        Network weights and network biases, weights and biases for Gradient descent, momentum, RMSProp 
        ................................................................................        
        '''
        
        #Random seed Generator
        np.random.seed(0)

        for i in range(1,self.num_of_layers):

            self.network_weights.append(np.random.randn(self.layer_dimensions[i], 
                                                                         self.layer_dimensions[i-1]))#*np.sqrt(4/self.num_of_layers))
            self.network_bias.append(np.random.randn(self.layer_dimensions[i],1))
            
            assert(self.network_weights[i-1].shape == (self.layer_dimensions[i], 
                                                                         self.layer_dimensions[i-1]))
            assert(self.network_bias[i-1].shape == (self.layer_dimensions[i],1))
            
            self.gradient_weights.append(np.zeros_like(self.network_weights[i-1])) #np.zeros((self.layer_dimensions[i], self.layer_dimensions[i-1]))
                                                                         #
            self.gradient_bias.append(np.zeros_like(self.network_bias[i-1]))#np.zeros((self.layer_dimensions[i],1))
            
            self.momentum_weights.append(np.zeros_like(self.network_weights[i-1]))
            
            self.momentum_bias.append(np.zeros_like(self.network_bias[i-1]))
            
            self.rms_weights.append(np.zeros_like(self.network_weights[i-1]))
            
            self.rms_bias.append(np.zeros_like(self.network_bias[i-1]))

    def batching(self, batching = False, batch_size = None):
        '''
        This is useful for Stochasting Batching and Mini Batching of teh training output

        Size is always in powers of 2 (32,64,128,...)
        '''
        #self.num_of_examples = self.training_data[0].shape[1]
              
        
        if batching:
            
            
            training_input = self.training_data[0]
            training_output = self.training_data[1]
            
            ##Shuffling the inputs
            #np.random.seed(0)
            #shuffle_series = np.arange(self.num_of_examples)
            #np.random.shuffle(shuffle_series)
#
            #shuffled_training_input = 
            #mini_batches = []
            
            number_of_batches = int(self.num_of_examples/batch_size)
            
            for j in range(0,number_of_batches):
                mini_train_input = training_input[:,(j*batch_size):((j+1)*batch_size)]
                mini_train_output = training_output[(j*batch_size):((j+1)*batch_size)]
                self.mini_batches.append((mini_train_input,mini_train_output))
                
            if self.num_of_examples % batch_size != 0:
                mini_train_input = training_input[:,(number_of_batches*batch_size):]
                mini_train_output = training_output[(number_of_batches*batch_size):]
                self.mini_batches.append((mini_train_input,mini_train_output))
        else:
            
            self.mini_batches = [self.training_data[:2]]

    def activation_function(self,Z):
        '''
        In artificial neural networks, the activation function of a node defines the output of that node given an input or set of inputs. [Wikipedia]
        '''
        
        if self.activation_function_name == "sigmoid":
            #print("Activating SIGMOID")
            return self.sigmoid(Z)

        elif self.activation_function_name == "relu":
            #print("Activating RELU")
            return self.relu(Z)

        elif self.activation_function_name == "tanh":
            #print("Activating Tanh")
            return self.tanh(Z)
        
    def activation_function_derivative(self,Z):
        ''' 
        Functions to call the corresponding derivatives of the activation functions
        '''
        if self.activation_function_name == "sigmoid":
            return self.sigmoid_derivative(Z)
        elif self.activation_function_name == "relu":
            return self.relu_derivative(Z)
        elif self.activation_function_name == "tanh":
            return self.tanh_derivative(Z)
        

    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))

    def sigmoid_derivative(self,Z):
        return self.sigmoid(Z)*(1-self.sigmoid(Z))
    
    def relu(self,Z):
        return np.maximum(0,Z)
    
    def relu_derivative(self,Z):
        dz = np.ones_like(Z)
        dz[Z<=0] = 0
        return dz

    def tanh(self,Z):
        return np.tanh(Z)
    
    def tanh_derivative(self,Z):
        return np.reciprocal(np.square(np.cosh(Z))) #sech^2(Z)

    def forward_propagation(self, X, check_w = None, check_b = None):
        
        self.activations = [X]
        self.Z, self.dActivations = [],[]

        #This is useful during gradient checking
        if check_w == None and check_b == None:
            weights = self.network_weights
            bias = self.network_bias
        else:
            weights = check_w
            bias = check_b
            
            
        for l in range(self.num_of_layers-1):

            Z = np.dot(weights[l], self.activations[l]) + bias[l]
            #print(" The Z.shape: ", Z.shape)
            activated_Z = self.activation_function(Z)
            #print(" The Activated Z.shape: ", activated_Z.shape)
            self.Z.append(Z)
            self.activations.append(activated_Z)

    def cost_function(self,Y, error_method = 'MSE', regularization=0.7,check_w =None):
        ''' 
        Description:

        - Using cost function helps to determine the amount of loss created using the parameters in forward propagation. 
        - Minimizing the cost function is the main objective

        Inorder to avoid Overfitting or Underfitting, L2 regularization technique is used.

            lambd: Regularization based hyper parameter (Penalty term)
            Higher Lamd ---> penalizes weights for Overfitting 
            Lower Lamd ---> penalizes weights for Underfitting
        
        ................................................................................
        * Input Arguments : *

        1. Y = Actual Outputs
        2. error_method : There are two most wdely used error methods:
                            a. MSE : Mean Squared Error ----> Regression 
                            b. Cross Entropy ------> Classification 
        3. regularization : True or False (Default: True)
                            if True imples apply regularization
        4. self ....includes the final activated output. Incase of regularization, network weights are used
        
        * Optional: *
        4. check_w : list containing network weights (useful for testing)

        * Returns/ Results: *
        cost : Scalar 
        ................................................................................
        '''
        self.error_method = error_method
        self.regularization = regularization 

        #lambd = 0.7 #Higher Lambda ensure to control the Overfitting

        m = Y.shape[0]
        #if test:
        #    y_o = self.testing_data[1]
        #else:
        #    y_o = Y
        L2_regularization_cost = 0.0
        if check_w == None:
            for lr in self.network_weights:
                L2_regularization_cost += (self.regularization/(2*m))*np.sum(np.square(lr))
        
        else:#elif check_w != None:
            for lr in check_w:
                L2_regularization_cost += (self.regularization/(2*m))*np.sum(np.square(lr))

        if self.error_method == "MSE":
            
            diff = self.activations[-1] - Y
            
            cost = 0.5*np.sum(np.square(diff))#*(1/m) #MSE Calculation
            
            #self.cost_value = np.squeeze(cost) ------------ to be added at the end of the mini batching
        elif self.error_method == "Cross Entropy":

            cost = np.sum((np.multiply(-Y,np.log(self.activations[-1]))-np.multiply((1-Y),np.log(1-self.activations[-1]))))
        
        cost = np.squeeze(cost)+np.squeeze(L2_regularization_cost)

        return np.squeeze(cost)

    def cost_function_derivative(self,y_o, test=False):
        
        """
        Description:

        Implements the derivative of the cost function w.r.t to the final activation output
        There are two kinds of error methods utilised:
            1. MSE : Mean Squared Error method ---> Useful for regression based 
            2. Cross Entropy : ----> Useful for CLassification problems

        ...........................................................................
        * Input Arguments : *

        1. y_o : Actual outputs 
        2. self... includes the final activated output, error_method opted

        * Returns/ Results : *
        the derivative of cost w.r.t to final activation is stored as an attribute under self.cost_derivatve
        ...........................................................................
        """    
        if self.error_method == 'MSE':
            delC_delA = (self.activations[-1] - y_o)#*(1/y_o.shape[0])

        elif self.error_method == "Cross Entropy":
            delC_delA = -np.divide(y_o,self.activations[-1])+np.divide((1-y_o),(1-self.activations[-1]))

        self.cost_derivative = delC_delA

    def back_propagation(self,X,Y, check_w = None, check_b = None):
        ''' 
        Description :

        To find the gradients, Back propagation is used. Gradients help to find the flatness i.e, minima

            - Gradients meaning the derivative of COst function w.r.t weights and biases
            - Chain rule of differentiation is implemented

        .................................................................................
        * Input Arguments : *

        1. X : The training input data 
        2. Y : The corresponding training output data
        3. self... here includes the network weights and biases, activations (A= G(Z)) and linear function(Z=WX+B)

        optional arguments: (useful during testing )
        4. check_w : list containing the network weights
        5. check_b : list containing the network biases

        * Returns : *

        Gradient Weights and Gradient Biases : List of all layer based gradients

        * Note: *
        The shape of the gradients is same  as their corresponding network parameters
        .................................................................................

        
        '''
        #if self.regularization:
        #    lambd = 0.7
        #else:
        #    lambd = 0.0

        #This is useful during gradient checking
        if check_w == None and check_b == None:
            weights = self.network_weights
            bias = self.network_bias
        else:
            weights = check_w
            bias = check_b

        delC_delA = self.cost_derivative#*(1.0/X.shape[1])

        #For Last Layer

        #delC _delZ = delC_delA * delA_delZ
        delC_delZ = np.multiply(delC_delA,self.activation_function_derivative(self.Z[-1]))###########*(1.0/X.shape[1])
        #delC_delB =  delC_delA * delA_delZ * delZ_delB
        delC_delB = np.sum(delC_delZ, axis=1,keepdims = True)#*(1.0/X.shape[1]) # Changed mean to SUM
        #delC_delW =  delC_delA * delA_delZ * delZ_delW
        delC_delW = np.dot(delC_delZ,self.activations[-2].transpose())#*(1.0/X.shape[1])

        self.gradient_weights[-1] = delC_delW + (self.regularization/X.shape[1])*weights[-1]#*self.network_weights[-1]
        assert ( self.gradient_weights[-1].shape == self.network_weights[-1].shape)
        self.gradient_bias[-1] = delC_delB#.reshape(self.network_bias[-1].shape)
        assert ( self.gradient_bias[-1].shape == self.network_bias[-1].shape)

        for i in reversed(range(2,self.num_of_layers)):
            #delC_delA = np.dot(self.network_weights[i-1].transpose(),delC_delZ)#################*(1.0/X.shape[1])
            delC_delA = np.dot(weights[i-1].transpose(),delC_delZ)
            delC_delZ = np.multiply(delC_delA , self.activation_function_derivative(self.Z[i-2]))
            delC_delB = np.sum(delC_delZ, axis=1, keepdims=True)#*(1.0/X.shape[1]) # Changed mean to SUM
            delC_delW = np.dot(delC_delZ,self.activations[i-2].transpose())#*(1.0/X.shape[1])#*

            self.gradient_weights[i-2] = delC_delW + (self.regularization/X.shape[1])*weights[i-2]#self.network_weights[i-2]
            assert ( self.gradient_weights[i-2].shape == self.network_weights[i-2].shape)
            self.gradient_bias[i-2] = delC_delB.reshape(self.network_bias[i-2].shape)
            assert ( self.gradient_bias[i-2].shape == self.network_bias[i-2].shape)

    def vector_to_list(self,array):
        
        weight_shapes = [arr_w.shape for arr_w in self.network_weights]
        bias_shapes = [arr_b.shape for arr_b in self.network_bias]
        
        t=0
        w,b = [],[]
        for m,n in zip(weight_shapes,bias_shapes):
            
            ini = t
            t = t+(m[0]*m[1])
            assert(t-ini == m[0]*m[1])
            w.append(array[ini : t,0].reshape(m))
            
            ini = t
            t = t+(n[0]*n[1])
            assert(t-ini == n[0]*n[1])
            b.append(array[ini : t,0].reshape(n))
            
        return w,b

    def gradient_checking(self,X,Y,error_method='MSE',regularization=0.7):
        """Implementation of gradient checking"""
        
        eps = 1e-7
        
        conglomerate_network_array = np.array([]).reshape(-1,1)
        conglomerate_gradient_array = np.array([]).reshape(-1,1)
        
        for p,r in zip(self.network_weights, self.network_bias):
            conglomerate_network_array = np.concatenate((conglomerate_network_array,p.reshape(-1,1),r.reshape(-1,1)))
            
        
        for dw,db in zip(self.gradient_weights,self.gradient_bias):
            conglomerate_gradient_array = np.concatenate((conglomerate_gradient_array,dw.reshape(-1,1),db.reshape(-1,1)))
        
        #central difference
        
        func_plus = np.zeros_like(conglomerate_network_array)
        func_minus = np.zeros_like(conglomerate_network_array)
        
        approx_graident = np.zeros_like(conglomerate_gradient_array)
        
        for i in range(func_plus.shape[0]):
            
            x_plus_eps = np.copy(conglomerate_network_array)
            
            x_plus_eps[i,0] = x_plus_eps[i,0]+eps
            
            weights_plus, bias_plus = self.vector_to_list(x_plus_eps)
            
            self.forward_propagation(X,check_w = weights_plus,check_b = bias_plus)
            
            cost_p = self.cost_function(Y,error_method, check_w=weights_plus)
            
            func_plus[i,0] = cost_p
            
            #########################################################################################
            x_minus_eps = np.copy(conglomerate_network_array)
            
            x_minus_eps[i,0] = x_minus_eps[i,0]-eps
            
            weights_minus, bias_minus = self.vector_to_list(x_minus_eps)
            
            self.forward_propagation(X, check_w=weights_minus, check_b=bias_minus)
            
            cost_m = self.cost_function(Y,error_method, check_w=weights_minus)
            
            func_minus[i,0] = cost_m
            
            ###############################################################################################
            approx_graident[i,0] = (func_plus[i,0] - func_minus[i,0])/(2*eps)
            ###############################################################################################
            #pint('the grad difference is ', approx_graident[i,0] - conglomerate_gradient_array[i,0])
        assert(approx_graident.shape == conglomerate_gradient_array.shape)    
        
        numerator= np.linalg.norm(approx_graident-conglomerate_gradient_array)
        denominator = np.linalg.norm(approx_graident)+np.linalg.norm(conglomerate_gradient_array)
        
        difference = numerator/denominator
        print('\n Gradient Checking difference : ', difference)
        if difference > 1e-4:
            raise ValueError("The Gradient Difference is abnormal. Kindly Check the Backpropagation!!")

    def update_parameters_GD(self,learning_rate):
            ''' 
            Description:

            Implementation of gradient descent method 
            .......................................................................
            * Input arguments: *

            1. Network weights and Network biases : list containing layer based weights and biases
            2. Gradient weights and gradient biases : list containing layer based gradients computed from Back Propogation
            3. learning_rate 
            

            * Returns : *

            Updated network weights and biases : List containing layer based updated Weights and Biases

            ................................................................................       

            '''
            for p in range(1,self.num_of_layers):

                self.network_weights[p-1] -= learning_rate*self.gradient_weights[p-1]

                self.network_bias[p-1] -= learning_rate*self.gradient_bias[p-1]

    
    def update_parameters_Momentum(self,learning_rate,beta1):
        '''
        Description :

        Implementation of Gradient descent with Momentum Method
        
        .............................................................................
        * Input arguments: *

        1. Network weights and Network biases : list containing layer based weights and biases
        2. Gradient weights and gradient biases : list containing layer based gradients computed from Back Propogation
        3. learning_rate 
        4. beta1 : Exponential Decay Hyper-parameter utilized in estimation of first moments

        * Returns : *

        Updated network weights and biases : List containing layer based updated Weights and Biases

        ................................................................................       
        
        '''
        
        for p in range(1,self.num_of_layers):
            
            self.momentum_weights[p-1] = beta1*self.momentum_weights[p-1] + (1-beta1)*self.gradient_weights[p-1]
            self.momentum_bias[p-1] = beta1*self.momentum_bias[p-1] + (1-beta1)*self.gradient_bias[p-1]
            
            self.network_weights[p-1] -= learning_rate*self.momentum_weights[p-1]
            self.network_bias[p-1] -= learning_rate*self.momentum_bias[p-1]
            
    def update_parameters_RMSProp(self, learning_rate,beta2):
        
        '''
        Description :

        Implementation of RMS Prop optimization
        
        .............................................................................
        * Input arguments: *

        1. Network weights and Network biases : list containing layer based weights and biases
        2. Gradient weights and gradient biases : list containing layer based gradients computed from Back Propogation
        3. learning_rate 
        4. beta2 : Exponential Decay Hyper-parameter utilized in estimation of second moments

        * Returns : *

        Updated network weights and biases : List containing layer based updated Weights and Biases

        ................................................................................       
        
        '''
        epsilon=1e-8 

        for p in range(1,self.num_of_layers):
            
            self.rms_weights[p-1]= beta2*self.rms_weights[p-1] + (1-beta2)*(self.gradient_weights[p-1])**2
            self.rms_bias[p-1] = beta2*self.rms_bias[p-1] + (1-beta2)*(self.gradient_bias[p-1])**2
            
            self.network_weights[p-1] -= (learning_rate*self.gradient_weights[p-1])/(np.sqrt(self.rms_weights[p-1])+epsilon)
            self.network_bias[p-1] -= (learning_rate*self.gradient_bias[p-1])/(np.sqrt(self.rms_bias[p-1])+epsilon)
        
    def update_parameters_Adam(self,learning_rate,beta1, beta2, t):
        '''Implementation of Adaptive Moment Estimation(Adam) optimization'''
        epsilon=1e-8
        temp_momentum_weights, temp_momentum_bias, temp_rms_weights, temp_rms_bias = [],[],[],[]
        
        for p in range(1,self.num_of_layers):
            
            self.momentum_weights[p-1] = beta1*self.momentum_weights[p-1] + (1-beta1)*self.gradient_weights[p-1]
            self.momentum_bias[p-1] = beta1*self.momentum_bias[p-1] + (1-beta1)*self.gradient_bias[p-1]
            
            temp_momentum_weights.append(self.momentum_weights[p-1]/(1-beta1**t))
            temp_momentum_bias.append(self.momentum_bias[p-1]/(1-beta1**t))
            
            self.rms_weights[p-1]= beta2*self.rms_weights[p-1] + (1-beta2)*(self.gradient_weights[p-1])**2
            self.rms_bias[p-1] = beta2*self.rms_bias[p-1] + (1-beta2)*(self.gradient_bias[p-1])**2
            
            temp_rms_weights.append(self.rms_weights[p-1]/(1-beta2**t))
            temp_rms_bias.append(self.rms_bias[p-1]/(1-beta2**t))
            
            self.network_weights[p-1] -= (learning_rate*temp_momentum_weights[p-1])/(np.sqrt(temp_rms_weights[p-1])+epsilon)
            self.network_bias[p-1] -= (learning_rate*temp_momentum_bias[p-1])/(np.sqrt(temp_rms_bias[p-1])+epsilon)

    def accuracy(self, testing = False):
        if testing:
            Y = self.testing_data[1]
            string = "Testing"
        else:
            Y = self.training_data[1]
            string = "Training"
        #assert(np.size(self.activations[-1]) == np.size(Y))
        u = ((Y- self.activations[-1])**2).sum()
        v = ((Y-Y.mean())**2).sum()
        error_percent = u/v#np.absolute(np.mean(np.divide(np.mean(self.activations[-1] - Y),np.mean(Y))))
        print('The R^2 score of {} is :'.format(string), (1-error_percent))

    def evaluate(self, plot=True):
        
        prediction_test = self.testing_data[0]

        self.forward_propagation(prediction_test)
        
        cost_test = self.cost_function(self.testing_data[1], self.error_method)
        print('The cost in Testing is: ', cost_test/(self.testing_data[0].shape[1]))
        
        self.accuracy(testing=True)

        if plot:
            fig,ax = plt.subplots(nrows = 2, ncols= 2, figsize=(20,8))
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
            plt.suptitle('TestPlot_Layer Plan: {}, Acti func: {}, Optimizer: {}, learning: {}, beta1: {}'.format( str(self.layer_dimensions), self.activation_function_name, self.optimizer,self.hyperparameters[0], self.hyperparameters[1]))

            Y= self.testing_data[2] #Without noise
            ax[0,0].plot(Y, 'r--', label='Truth')
            ax[0,0].plot(self.activations[-1].reshape(Y.shape), 'o' ,label='Predicted')
            ax[0,0].set_title('Output convergence during Testing')#, fontsize=15)
            ax[0,0].set_xlabel('Data Points')
            ax[0,0].set_ylabel('Output Value i.e, Yield')
            ax[0,0].legend()

            Y_n = self.testing_data[1]
            ax[0,1].plot(Y,Y, 'r--', label='Regression Line')
            ax[0,1].scatter(Y_n, self.activations[-1].reshape(Y.shape), label='Predicted Values')
            ax[0,1].set_title("Truth v/s Predicted")
            ax[0,1].set_xlabel("Actual Output")
            ax[0,1].set_ylabel("Predicted Output")
            ax[0,1].legend()

            final_weights, final_bias =[],[]
            for i,j in zip(self.network_weights, self.network_bias):
                final_weights.append(i.flatten())
                final_bias.append(j.flatten())
            X_bar_w = np.linspace(1,len(np.concatenate(final_weights).ravel()),len(np.concatenate(final_weights).ravel()))
            X_bar_b = np.linspace(1,len(np.concatenate(final_bias).ravel()),len(np.concatenate(final_bias).ravel()))
            ax[1,0].bar(X_bar_w, np.concatenate(final_weights).ravel(), label='Weights')
            ax[1,1].bar(X_bar_b, np.concatenate(final_bias).ravel(), label='Bias')
            ax[1,0].set_title("Weights")
            ax[1,0].legend()
            ax[1,1].set_title("Biases")
            ax[1,1].legend()

            plt.show()
    
    def plotting(self):
        fig, ax = plt.subplots(nrows = 2 , ncols=2, figsize=(30,10))
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
        plt.suptitle('Epochs: {}, Layer Plan: {}, Acti func: {}, Optimizer: {}, learning: {}, beta1: {}'.format( self.epochs_number,str(self.layer_dimensions), self.activation_function_name, self.optimizer,self.hyperparameters[0], self.hyperparameters[1]))
        
        ax[0,0].semilogy(np.array(self.cost), label = "Training Cost")
        ax[0,0].plot(np.array(self.CV_cost), "r--", label="Validation")
        ax[0,0].set_title('Cost Variance', fontsize=15)
        ax[0,0].set_xlabel('Iterations')
        ax[0,0].set_ylabel('Log(Cost Value)')
        ax[0,0].legend()

        Y= self.training_data[2]
        ax[0,1].plot(Y, 'r--', label='Truth')
        ax[0,1].plot(self.activations[-1].reshape(Y.shape), 'o' ,label='Predicted')
        ax[0,1].set_title('Output convergence')#, fontsize=15)
        ax[0,1].set_xlabel('Data Points')
        ax[0,1].set_ylabel('Output Value i.e, Yield')
        ax[0,1].legend()

        Y_n = self.training_data[1]
        ax[1,0].plot(Y,Y, 'r--', label='Regression Line')
        ax[1,0].scatter(Y_n, self.activations[-1].reshape(Y.shape), label='Predicted Values')
        ax[1,0].set_title("Truth v/s Predicted")
        ax[1,0].set_xlabel("Actual Output")
        ax[1,0].set_ylabel("Predicted Output")
        ax[1,0].legend()

        final_weights, final_bias =[],[]
        for i,j in zip(self.network_weights, self.network_bias):
            final_weights.append(i.flatten())
            final_bias.append(j.flatten())
        X_bar_w = np.linspace(1,len(np.concatenate(final_weights).ravel()),len(np.concatenate(final_weights).ravel()))
        ax[1,1].bar(X_bar_w, np.concatenate(final_weights).ravel(), label='Weights')
        ax[1,1].plot(np.concatenate(final_bias).ravel(), 'r--', label='Bias')
        ax[1,1].set_title("Weights and Biases")
        ax[1,1].legend()
        
        #plt.tight_layout()
        fig.savefig("tested_images/Image_{}_{}_{}_{}.png".format(self.epochs_number, str(self.layer_dimensions), self.activation_function_name, self.optimizer))
        plt.show()


    def NN_model(self, epochs, learning_rate, beta1=None, beta2=None, regularization = 0.7,activation_function = "sigmoid", batching=False, batch_size = None,  error_method = 'MSE', optimizer='GD', tolerance=1e-4, early_stop = False):
        
        ''' Deep neural network model'''
        
        
        self.activation_function_name = activation_function
        self.optimizer = optimizer
        self.epochs_number = epochs
        self.regularization = regularization
        #number_of_samples = self.training_data[0].shape[1]
        
        #Initialize the Weights and Biases
        self.network_parameters_initialization()
        count = 0
        time_step = 0
        for iteration in range(epochs):#, desc='Training Epochs'):
            
            #Batching
            self.batching(batching, batch_size)
            
            cost_measured = 0.0

            #Traversing through Mini Batches:
            for mini_batch in self.mini_batches:

                mini_batch_X, mini_batch_Y = mini_batch

                #Forward Prop
                self.forward_propagation(mini_batch_X)
                
                #Loss calculation
                cost_measured += self.cost_function(mini_batch_Y,error_method)
                self.cost_function_derivative(mini_batch_Y)

                #Back Prop
                self.back_propagation(mini_batch_X,mini_batch_Y)

                #Gradient Checking
                if iteration%1000 == 0:
                    #print("iteration%1000 : ", iteration, iteration%1000)
                    self.gradient_checking(mini_batch_X, mini_batch_Y,error_method)

                if self.optimizer == 'GD':
                    self.hyperparameters = [learning_rate,0.0]
                 #Updating parameters with Gradient Descent
                    self.update_parameters_GD(learning_rate)
             
                elif self.optimizer == 'Momentum':
                    self.hyperparameters = [learning_rate, beta1]
                #Updating parameters with Steepest Gradient
                    self.update_parameters_Momentum(learning_rate, beta1)
               
                elif self.optimizer == 'RMSProp':
                    self.hyperparameters = [learning_rate,beta2]
                    self.update_parameters_RMSProp(learning_rate, beta2)
            
                elif self.optimizer == 'adam':
                    time_step = time_step+1
                    self.hyperparameters = [learning_rate, beta1, beta2]
              #Updating parameters with Adam optimization
                    self.update_parameters_Adam(learning_rate, beta1, beta2, time_step)

            Average_cost = cost_measured/len(self.mini_batches)

            if iteration%1000 == 0:
                print ("Cost after epoch %i: %f" %(iteration, Average_cost))

            #if iteration%100 == 0:
            self.cost.append(Average_cost)

            #Cross Validation:
            cross_validation_input = self.validation_data[0]
            self.forward_propagation(cross_validation_input)
            #
            ##Cross Validation Cost computation
            CV_cost1 = self.cost_function(self.validation_data[1], self.error_method)#, test=False)
            self.CV_cost.append(CV_cost1)

            #Early Stopping
            #
            if early_stop and iteration>20:
                #tol=tolerance
                if(abs(np.array(self.cost[-11:-1])- np.array(self.cost[-10:])) < tolerance*(np.ones((10)))).all():
                    #count = count+1
                    #if count>1 :
                    print("The Training is paused at the Iteration: "+str(iteration)+'as it no more learning beyond the tol '+str(tolerance))
                    #print(self.cost[-20:])
                    print("\n The Current Training Cost is :",self.cost[-1]/self.num_of_examples)
                    break
                elif ((np.array(self.CV_cost[-10:])- np.array(self.CV_cost[-11:-1])) > np.zeros((10))).all():
                    count = count+1
                    if count>5 :
                        print("The Training is paused at the Iteration: "+str(iteration))#+'as it no more learning beyond the tol '+str(tolerance))
                        #print(self.CV_cost[-20:])
                        print("\n The Current Training Cost is :",self.cost[-1]/self.validation_data[0].shape[1])
                        break

            ### 
            self.epochs_number = iteration

        #prediction
        prediction_train = self.training_data[0]

        self.forward_propagation(prediction_train)

        self.accuracy()

        self.plotting()
        
        #Evaluation
        self.evaluate(plot=True)

        #self.accuracy()
        #self.plotting()


# In[3]:


#hidden_layer_dims = [20,10,5]
#model = my_neural_network(hidden_layer_dims)
#file_name = 'output_bigoni_martin_ellipse.txt'
#model.load_dataset(file_name)#, noise=True)
#model.test_train_split()
#model.NN_model(10001, 0.1,beta1=0.9,beta2=0.999,activation_function = "sigmoid", batching = False, batch_size= 128,optimizer = 'adam',early_stop=True)
#
