import numpy as np:

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
    assert ( self.gradient_network_parameters['dWeights'+str(self.num_of_layers-1)].shape == self.network_parameters['Weights'+str(self.num_of_layers-1)].shape)
    self.gradient_network_parameters['dbias'+str(self.num_of_layers-1)] = delC_delB
    assert ( self.gradient_network_parameters['dbias'+str(self.num_of_layers-1)].shape == self.network_parameters['bias'+str(self.num_of_layers-1)].shape)

    for i in reversed(range(2,self.num_of_layers)):
        delC_delA = np.dot(self.network_parameters['Weights'+str(i)].transpose(),delC_delZ)
        delC_delZ = delC_delA*self.sigmoid_derivative(self.Z[i-2])
        delC_delB = np.mean(delC_delZ, axis=1)
        delC_delW = (1.0/X.shape[1])*np.dot(delC_delZ,self.activations[i-2].transpose())

        self.gradient_network_parameters['dWeights'+str(i-1)] = delC_delW
        assert ( self.gradient_network_parameters['dWeights'+str(i-1)].shape == self.network_parameters['Weights'+str(i-1)].shape)
        self.gradient_network_parameters['dbias'+str(i-1)] = delC_delB
        assert ( self.gradient_network_parameters['dbias'+str(i-1)].shape == self.network_parameters['bias'+str(i-1)].shape)

def update_parameters_GD(self,learning_rate):
        ''' Implementation of gradient descent method '''
        
    for p in range(1,self.num_of_layers):
            
        self.network_parameters['Weights'+str(p)] -= learning_rate*self.gradient_network_parameters['dWeights'+str(p)]
                                                                                                         
        self.network_parameters['bias'+str(p)] -= learning_rate*self.gradient_network_parameters['dbias'+str(p)]