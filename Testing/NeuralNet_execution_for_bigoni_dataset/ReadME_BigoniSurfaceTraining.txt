#####################################################################
# Name :  Sushanth Keshav
# Aim : Modelling the yield surfaces asper Bigoni Piccolroaz Model
#         and training the model to the generated Bigoni Surface
# Programming Language : Python
# Last Updated : 04.04.2020
#####################################################################

Steps:

1. Ensure that the latest version of python is installed with the packages such as numpy and matplotlib


2. The user has to ensure that the data file, neural_net_library.py and my_library_test.py files are all located in the same directory.

#The input for hidden_layer_dims must be a list containing number of neurons in each hiden layer
hidden_layer_dims = [10,10,10]
#Enter the file name. Has to be a string with extension.
file_name = 'Bigoni_yield_data.txt'
#Enter the test file name
file_name_t = 'output_bigoni_ellipse_check1'
#Enter the hyper-parameters. Note that all the hyper-parameters are Non-Negative
epochs = 10000
lr = 0.01
# Mandatory for Momentum and Adam range: [0,1)
b1 = 0.9 
activate = 'sigmoid'
optim = 'Momentum'

3. Open the "my_library_test.py" file and enter the text file name containing the data to which your model has to be trained

4. In the my_library_test.py file you will find few variables to enter the inputs. These input variables are 
   mandatory hyper-paramters for the model to be executed. Incase you need to tune other parameters 
   the function is called with all the default variables required. User can manually change themas required.

5. Execution of the Neural Network Model:
$ python3 my_library_test.py >log.txt
This shall help the user to create a log file with all the necessary print statements. Note that this gets updated after every run,
So the user incase has to store the data, he/she needs to change the name of log.txt located in teh same directory.


6. Incase the user would like to print the weights and biases he could do so by using the following command
$print(model.network_weights)
$print(model.network_bias)
#################################################################

Expected R^2 score is 0.94
