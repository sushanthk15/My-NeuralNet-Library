import numpy as np

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
        self.dataset = None
        
        #input and output data
        self.input_data = None
        self.output_data = None
        self.predicted_output =None
        
        #train and test data
        self.training_data = None
        self.testing_data = None
        
        #layer details
        self.hidden_layer_plan = hidden_layer_plan
        self.layer_dimensions = None
        self.num_of_layers = None
        
        #Batching
        self.mini_batches = []
        
        #Forward Activation
        self.activations, self.Z = [], []
        
        #Backward Activation
        self.dActivations = []
        
        #Error_method and loss
        self.error_method = None
        self.loss = None
        self.cost = []
        
        #Network parameters
        self.network_parameters, self.gradient_network_parameters = {}, {}
        
        
    def __repr__(self):
        ''' Representative function to welcome the user for collaborating with the new AI trainer'''
        return 'Hello there!! I am your new AI trainer......!'
    
    
    def load_dataset(self, file):
        ''' 
        To load dataset 
        
        input:
        file : data_type: string
        File name consisting the inputs and outputs. The last column in the file is the output.
        '''
        self.dataset = np.loadtxt(file)
        self.input_data = self.dataset[:,:-1]
        self.output_data = self.dataset[:,-1]
        
        #Normalization
        #max_inputs = np.amax(self.input_data, axis=0)
        #self.input_data = np.divide(self.input_data, max_inputs)
        
        #max_outputs = np.amax(self.output_data, axis=0)
        #self.output_data = np.divide(self.output_data, max_outputs)
        
        #Layer planning
        self.layer_dimensions = [self.input_data.shape[1]]+self.hidden_layer_plan+[1] #1 corresponds to columns in O/P
        self.num_of_layers = len(self.layer_dimensions)
        
    def test_train_split(self, split = 0.7):
        '''
        This function is utilized to segragate the complete data into training and testing data.
        '''
        
        n_total = int(self.dataset.shape[0])
        n_train = int(split*n_total)
        
        #mask = np.zeros((n_total), dtype=bool)
        #mask[:n_train] = True
        
        #np.random.shuffle(mask)
        
        X_train = self.input_data[:n_train,:]
        Y_train = self.output_data[:n_train]
        
        X_test = self.input_data[n_train:,:]
        Y_test = self.output_data[n_train:]
        
        self.training_data = (X_train.transpose(), Y_train)
        self.testing_data = (X_test.transpose(), Y_test)
    
    def network_parameters_initialization(self):
        for i in range(1,self.num_of_layers):
            self.network_parameters['Weights'+str(i)] = np.random.randn(self.layer_dimensions[i], 
                                                                         self.layer_dimensions[i-1])
            self.network_parameters['bias'+str(i)] = np.random.randn(self.layer_dimensions[i],1)
            
            assert(self.network_parameters['Weights'+str(i)].shape == (self.layer_dimensions[i], 
                                                                         self.layer_dimensions[i-1]))
            assert(self.network_parameters['bias'+str(i)].shape == (self.layer_dimensions[i],1))
            
            self.gradient_network_parameters['dWeights'+str(i)] = np.zeros((self.layer_dimensions[i], self.layer_dimensions[i-1]))
                                                                         #
            self.gradient_network_parameters['dbias'+str(i)] = np.zeros((self.layer_dimensions[i],1))
            
     
    def batching(self, batching = False, batch_size = None):
        '''
        The NN_gradient_descent function helps to build and get the deep network working. 
        
        inputs:
        learning_rate : floating number
        This is an hyper parameter which is used in gradient descent
        '''
        num_examples = self.training_data[0].shape[1]
              
        
        if batching:
            
            training_input = self.training_data[0]
            training_output = self.training_data[1]
            
            #mini_batches = []
            
            number_of_batches = int(num_examples/batch_size)
            
            for j in range(0,number_of_batches):
                mini_train_input = training_input[:,(j*batch_size):((j+1)*batch_size)]
                mini_train_output = training_output[:,(j*batch_size):((j+1)*batch_size)]
                self.mini_batches.append((mini_train_input,mini_train_output))
                
            if num_examples % batch_size != 0:
                mini_train_input = training_input[:,(number_of_batches*batch_size):]
                mini_train_output = training_output[:,(number_of_batches*batch_size):]
                self.mini_batches.append((mini_train_input,mini_train_output))
        else:
            
            self.mini_batches = [self.training_data]




    def sigmoid(self,Z):
        return 1./(1+np.exp(-Z))

    def sigmoid_derivative(self,Z):
        return self.sigmoid(Z)(1-self.sigmoid(Z))

    def forward_propagation(self, X):
        self.activations = [X]
        for l in range(self.num_of_layers):

            Z = np.dot(self.network_parameters['Weights'+str(l+1)], self.activations[0])+self.network_parameters['bias'+str(l+1)]

            activated_Z = self.sigmoid(Z)

            self.Z.append(Z)
            self.activations.append(activated_Z)

    def cost_function(self, error_method = 'MSE'):
        self.error_method = error_method

        if error_method == 'MSE':
            cost = 0.5*np.mean(np.square(self.activations[-1] - self.training_data[1]))
            cost = np.squeeze(cost)

    def cost_function_derivative(self):

        if self.error_method == 'MSE':
            delC_delA = self.activations[-1] - self.training_data[1]

        self.loss = delC_delA

    def back_propagation(self, X,Y):
        delC_delA = self.loss

        #For Last Layer

        #delC _delZ = delC_delA * delA_delZ
        delC_delZ = delC_delA*self.sigmoid_derivative(self.Z[-1])
        #delC_delB =  delC_delA * delA_delZ * delZ_delB
        delC_delB = np.mean(delC_delZ, axis=1)
        #delC_delW =  delC_delA * delA_delZ * delZ_delW
        delC_delW = np.dot(delC_delZ,self.activations[-2].transpose())

        self.gradient_network_parameters['dWeights'+str(self.num_of_layers-1)] = delC_delW
        assert ( self.gradient_network_parameters['dWeights'+str(self.num_of_layers-1)].shape == self.network_parameters['Weights'+str(self.num_of_layers-1)].shape )

        self.gradient_network_parameters['dbias'+str(self.num_of_layers-1)] = delC_delB
        assert ( self.gradient_network_parameters['dbias'+str(self.num_of_layers-1)].shape == self.network_parameters['bias'+str(self.num_of_layers-1)].shape)

        for i in reversed(range(2,self.num_of_layers)):
            delC_delA = np.dot(self.network_parameters['Weights'+str(i)].transpose(),delC_delZ)
            delC_delZ = delC_delA*self.sigmoid_derivative(self.Z[i-2])
            delC_delB = np.mean(delC_delZ, axis=1)
            delC_delW = (1.0/X.shape[1])*np.dot(delC_delZ,self.activations[i-2].transpose())

            self.gradient_network_parameters['dWeights'+str(i-1)] = delC_delW
            assert ( self.gradient_network_parameters['dWeights'+str(i-1)].shape == self.network_parameters['Weights'+str(i-1)].shape )

            self.gradient_network_parameters['dbias'+str(i-1)] = delC_delB
            assert ( self.gradient_network_parameters['dbias'+str(i-1)].shape == self.network_parameters['bias'+str(i-1)].shape)

    def update_parameters_GD(self,learning_rate):
            ''' Implementation of gradient descent method '''

        for p in range(1,self.num_of_layers):

            self.network_parameters['Weights'+str(p)] -= learning_rate*self.gradient_network_parameters['dWeights'+str(p)]

            self.network_parameters['bias'+str(p)] -= learning_rate*self.gradient_network_parameters['dbias'+str(p)]

    def NN_model(self, epochs, learning_rate, batching=False, batch_size = None,  error_method = 'MSE'):
        
        ''' Deep neural network model'''
        
        self.network_parameters_initialization()
        
        for iteration in range(epochs):
            
            #Batching
            self.batching(batching=False, batch_size=None)
            
            #Traversing through Mini Batches:
            for mini_batch in self.mini_batches:
                
                mini_batch_X, mini_batch_Y = mini_batch
                
                #Forward Prop
                self.forward_propagation(mini_batch_X)
                
                #Loss calculation
                self.cost_function(error_method)
                self.cost_function_derivative()

                if iteration%(epochs/10) == 0:
                    self.cost.append(np.squeeze(np.mean(self.loss)))
                    print('The cost after iteration: ', iteration, 'is :', np.squeeze(np.mean(self.loss)))
                #Back Prop
                self.back_propagation()
                
                #Updating parameters with Gradient Descent
                self.update_parameters_GD(learning_rate)
                
        #prediction
        prediction_train = self.training_data[0]
        self.forward_propagation(prediction_train)