#####################################################################
# Name :  Sushanth Keshav | Matrikel Nr: 63944
# Topic : Unit testing/Functional Testing of my_neural_network
# Programming Language : Python
# Last Updated : 04.04.2020
#####################################################################

import unittest
import numpy as np
from neural_net_library import my_neural_network

class Testing_my_Neural_Net(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('SetUpClass')

    @classmethod
    def tearDownClass(cls):
        print("TearDownClass")

    def setUp(self):
        print('SetUp')
        self.model1 = my_neural_network([1])
        self.model2 = my_neural_network([])
        self.model3 = my_neural_network([3,4,5])

        filename = 'log.txt'
        filename2 = 'XOR.txt'

        self.model1.load_dataset(filename)
        self.model2.load_dataset(filename2)
        self.model3.load_dataset(filename)

    def tearDown(self):
        print("TearDown \n")

    
    def test_load_dataset(self):
        ''' Testing the load Data Set Function '''

        print("Testing the Load Dataset function")
        self.assertEqual(self.model1.input_data.shape , (1000,2))
        self.assertEqual(self.model1.output_data.shape , (1000,))
        
        #checking if data is normalized
        print("Testing Normalization")
        self.assertTrue((np.amax(self.model1.input_data, axis=0) == np.ones((2,),dtype=float)).all())
        self.assertTrue((np.amin(self.model1.input_data, axis=0)== np.zeros((2,),dtype=float)).all())

        self.assertEqual(self.model1.layer_dimensions,[2,1,1])
        self.assertEqual(self.model1.num_of_layers ,3)

    def test_test_train_split(self):
        print('Test case for Test Train Split')
        self.model1.test_train_split()

        self.assertEqual(self.model1.training_data[0].shape , (2,539))
        self.assertEqual(self.model1.training_data[1].shape , (539,))

        self.assertEqual(self.model1.validation_data[0].shape , (2,161))
        self.assertEqual(self.model1.validation_data[1].shape , (161,))

        self.assertEqual(self.model1.testing_data[0].shape , (2,300))
        self.assertEqual(self.model1.testing_data[1].shape , (300,))

    def test_batching(self):
        print("Testing the Batching Operation")
        self.model1.test_train_split()
        
        self.model1.batching(batching=True, batch_size=100)
        self.assertEqual(len(self.model1.mini_batches),6)

        self.model1.batching(batching=False, batch_size=100)
        self.assertEqual(len(self.model1.mini_batches),1)

        self.model1.batching(batching=True, batch_size=1)
        self.assertEqual(len(self.model1.mini_batches),540)
    
    def test_network_parameters_initialization(self):
        print("Testing the Initialization function")
        np.random.seed(0)

        #Initialize Network parameters
        self.model1.network_parameters_initialization()
        self.model2.network_parameters_initialization()
        self.model3.network_parameters_initialization()

        #Testing network parameters
        W1 = self.model2.network_weights[0]
        B1 = self.model2.network_bias[0]
        
        W1_check = np.array([[ 1.76405235,  0.40015721]])
        B1_check = np.array([[ 0.97873798]])
        self.assertTrue(np.allclose(W1,W1_check))
        self.assertTrue(np.allclose(np.squeeze(B1),np.squeeze(B1_check)))

        #Testing initial gradient weights and biases
        dW2 = self.model3.gradient_weights[-2]
        dW2_check = np.zeros((5,4))
        dB2 = self.model3.gradient_bias[-3]
        dB2_check = np.zeros((4,1))
        self.assertTrue((dW2==dW2_check).all())
        self.assertTrue((dB2==dB2_check).all())

        #Testing initial momentum weights and biases
        dW2_mom = self.model1.momentum_weights[-1]
        dW2_check_mom = np.zeros((1,1))
        db2_mom = self.model1.momentum_bias[-1]
        db2_check_mom = np.zeros((1,1))
        self.assertTrue((dW2_mom==dW2_check_mom).all())
        self.assertTrue((db2_mom==db2_check_mom).all())

        #Testing initial rms weights and biases
        dW3_rms = self.model1.rms_weights[-1]
        dW3_check_rms = np.zeros((1,1))
        db3_rms = self.model1.rms_bias[-1]
        db3_check_rms = np.zeros((1,1))
        self.assertTrue((dW3_rms==dW3_check_rms).all())
        self.assertTrue((db3_rms==db3_check_rms).all())


    def test_sigmoid(self):
        print("Testing Sigmoid function")

        z1 = self.model1.sigmoid(np.array([0 , -np.inf , np.inf]))
        z2 = np.array([0.5, 0. , 1. ])
        self.assertTrue((z1==z2).all())

    def test_relu(self):
        print("Testing RELU function")
        
        self.assertEqual(self.model1.relu(np.array([-21.])), np.array([0.]))
        self.assertEqual(self.model1.relu(np.array([0.])), np.array([0.]))
        self.assertEqual(self.model1.relu(np.array([5])), np.array([5.]))
        self.assertTrue((self.model1.relu(np.array([-11 , -56.25 , 41.23 ])) == np.array([0 , 0. , 41.23])).all())
        
    def test_tanh(self):
        print("Testing Tanh function")

        z1 = self.model1.tanh(np.array([0 , -np.inf , np.inf]))
        z2 = np.array([ 0., -1.,  1.])
        self.assertTrue((z1==z2).all())

    def test_sigmoid_derivative(self):
        print("Testing Sigmoid Derivative function")
        
        z1 = self.model1.sigmoid_derivative(np.array([0 , -np.inf , np.inf]))
        z2 = np.array([0.25, 0. , 0. ])
        self.assertTrue((z1==z2).all())

    def test_relu_derivative(self):
        print("Testing the Relu derivative function")
        
        z1 = self.model1.relu_derivative(np.array([0 , -np.inf , np.inf]))
        z2 = np.array([0., 0. , 1. ])
        self.assertTrue((z1==z2).all())

    def test_tanh_derivative(self):
        print("Testing Tanh Derivative function")
        
        z1 = self.model1.tanh_derivative(np.array([0 , -np.inf , np.inf]))
        z2 = np.array([ 1. , 0. , 0. ])
        self.assertTrue((z1==z2).all())

    def test_forward_propagation(self):

        print("Testing Forward Prop and Cost Function")

        X = self.model2.input_data.transpose()
        Y = self.model2.output_data.transpose()
        self.model2.network_parameters_initialization()
        
        wts = [np.array([1,2])]
        bias = [np.array([0.]).reshape(-1,1)]
        self.model2.forward_propagation(X, wts, bias)

        z1 = self.model2.Z[0]
        
        A0 = self.model2.activations[0]
        A1 = self.model2.activations[1]
        
        z1_check = np.array([0,2,1,3,0,2,1,3,0,2,1,3]).reshape(1,12)
        self.assertTrue((z1==z1_check).all())
        A1_check = np.array([0.5 , 0.88079708, 0.73105858, 0.95257413, 0.5 , 0.88079708, 0.73105858, 0.95257413, 0.5 , 0.88079708, 0.73105858, 0.95257413])
        self.assertTrue((A0==X).all())
        
        self.assertTrue(np.allclose(A1, A1_check))
        print("Forward Prop is Successful")
        
        #######################################################################

        print("Testing COST function ")
        cost1 = self.model2.cost_function(Y,error_method='MSE',regularization=0.0)
        
        self.assertTrue(np.isclose(cost1,1.8659044))
        print("Cost Computation is Successful")

        ###########################################################################
        
        print("Testing COST_derivative function")
        self.model2.cost_function_derivative(Y)
        dcost_expected = self.model2.activations[1] - Y
        dcost_actual = self.model2.cost_derivative
        self.assertTrue((dcost_actual == dcost_expected).all())
        print("Derivative of Cost Computation is Successful")

        #############################################################################
        
        print("Testing Backward Propagation")
        self.model2.back_propagation(X,Y,wts, bias)
        delC_delW = self.model2.gradient_weights[-1]
        delC_delB = self.model2.gradient_bias[-1]
        dw = np.array([-0.029528927, 0.091555725])
        db = np.array([0.307924447])
        self.assertTrue(np.allclose(np.squeeze(delC_delB), np.squeeze(db)))
        print("Backward Prop is Successful")

        
    

if __name__ == '__main__':
    unittest.main()
    