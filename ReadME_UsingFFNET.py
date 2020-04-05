#####################################################################
# Name :  Sushanth Keshav | Matrikel Nr: 63944
# Topic : Verification of the results by using FFNET package
# Programming Language : Python
# Last Updated : 04.04.2020
#####################################################################

Steps:

1. Ensure that the latest version of python is installed with the packages such as numpy, matplotlib
FFNET 0.8.3 (latest so far). 

2. Ensure that the dataset text file, UsingFFNET.py are located in the same directory.

3. Open the "UsingFFNET.py" file and enter the text file name containing the data to which your model has to be trained

4. In the UsingFFNET.py file you will find few variables to enter the inputs. These input variables are 
   mandatory hyper-paramters for the model to be executed. 

5. If the user would liek to train the model with any other optimization algorithm, please refer to the 
API docs of FFNET http://ffnet.sourceforge.net/index.html

6. Execution of the Neural Network Model:
$ python3 UsingFFNET.py >log.txt
This shall help the user to create a log file with all the necessary print statements. Note that this gets updated after every run,
So the user incase has to store the data, he/she needs to change the name of log.txt located in teh same directory.

7. Execution of the file also creates plots, if the user would like to save them : Once the figure appears
At the left bottom corner, click on the save button and save them in your local directory. *This is Optional*
