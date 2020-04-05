#####################################################################
# Name :  Sushanth Keshav | Matrikel Nr: 63944
# Topic : Modelling the yield surfaces asper Bigoni Piccolroaz Model
#         and training the model to the generated Bigoni Surface
# Programming Language : Python
# Last Updated : 04.04.2020
#####################################################################

Steps:

1. Ensure that the latest version of python is installed with the packages such as numpy and matplotlib

2. Open the Bigoni_Piccolroaz_YieldCriterion.py file to generate the yield surface for any kind of material based on the parameters

3. Refer report or literature Bigoni et.al (2004) for clear picture on what the variables mean. Please note all the parameters are non -negative in nature

4. run the file using the command : 'python3 Bigoni_Piccolroaz_YieldCriterion.py'

5. On succcessful execution, Bigoni_yield_data.txt file is created with the invariants Bigoni_yield_data

6. The user has to ensure that the data file, neural_net_library.py and my_library_test.py files are all located in the same directory.

7. Open the "my_library_test.py" file and enter the text file name containing the data to which your model has to be trained

8. In the my_library_test.py file you will find few variables to enter the inputs. These input variables are 
   mandatory hyper-paramters for the model to be executed. Incase you need to tune other parameters 
   the function is called with all the default variables required. User can manually change themas required.

9. Execution of the Neural Network Model:
$ python3 my_library_test.py >log.txt
This shall help the user to create a log file with all the necessary print statements. Note that this gets updated after every run,
So the user incase has to store the data, he/she needs to change the name of log.txt located in teh same directory.

