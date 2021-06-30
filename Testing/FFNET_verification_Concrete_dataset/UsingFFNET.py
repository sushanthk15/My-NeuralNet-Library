####################################################################################
#!/usr/bin/env python
# coding: utf-8
# Name : Sushanth Keshav
# Topic : Testing using FFNET python package and verifying custom built Feed Forward Neural Network Library for Compresive strength of Concrete
# Last Update: 04.04.2020
#####################################################################################

import numpy as np
import matplotlib.pyplot as plt 
import time 
from ffnet import ffnet, mlgraph, savenet, loadnet, exportnet

#Input from the user
file_name = 'Concrete_Data.txt'
optimizer = "Momentum"
#Hyper-parameter for the Momentum
beta1 = 0.9 # 0 for Gradient Descent else it is for Momentum
#Defining the Network Plan
conec = mlgraph((8,5,5,1))

#Start Timer
start_time = time.clock()

#Loading the Data
data = np.loadtxt(file_name)#output_bigoni_martin_ellipse
X = data[:,:-1]
y = data[:,-1]


#NORMALIZATION
max_inputs = np.amax(X, axis=0)
min_inputs = np.amin(X, axis=0)
diff_inputs = max_inputs - min_inputs
X = np.divide((X - min_inputs),diff_inputs)

max_outputs = np.amax(y, axis=0)
min_outputs = np.amin(y, axis=0)
diff_outputs = max_outputs - min_outputs
y = np.divide((y - min_outputs), diff_outputs)


net = ffnet(conec)

#Test train data split
n_tot=X.shape[0]
split = 0.7
n_train = int(split*n_tot)

mask = np.zeros((n_tot), dtype=bool)
mask[:n_train] = True
np.random.seed(0)
np.random.shuffle(mask)

X_train = X[mask]
Y_train = y[mask]

X_test = X[~mask]
Y_test = y[~mask]

#Training the network using momentum optimization
if optimizer == "Momentum" :
    net.train_momentum(X_train, Y_train, eta=0.01, momentum=beta1, maxiter=10000, disp=0)
elif optimizer == "BFGS":
    net.train_bfgs(X_train, Y_train)

#Evaluating the Data
output, regress = net.test(X_test, Y_test)

#saving network weights
savenet(net, "log.net")

#Accuracy calculations
accuracy_percent = abs(np.mean(np.divide((output - Y_test).mean(),Y_test.mean())))
print('Accuracy of testing: ',(1-accuracy_percent)*100)

#Stop timer
stop_time = time.clock()

#Plotting 
fig , ax = plt.subplots(ncols=2, figsize=(10,6))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
ax[0].plot(net.weights , label='Weights')
ax[0].set_title('Network Weights')
ax[0].legend()
ax[1].plot(Y_test, 'r--', label='Yield Surface')
ax[1].plot(output, label='Predicted')
ax[1].legend()
plt.show()

#Plotting the actual v/s predicted output
fig,ax=plt.subplots(figsize=(10,6))
ax.scatter (Y_test, output, edgecolors=(0,1,0))
ax.plot([0, 1], [0, 1], '--k', lw=4)
plt.xlabel("Actual Output", fontsize=15)
plt.ylabel("Predicted Output", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.text(0.075,0.95, r'Cross validated predictions using FFNET',fontsize=15)
plt.text(0.075,0.87, r'Optimizer: {}'.format(optimizer),fontsize=15)
plt.text(0.08,0.80, r'$R^2$=%.2f' % (regress[0][2]),fontsize=15)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.show()

#Time Elapsed
total_time = stop_time - start_time
print("The total time taken for learning is : %.2f seconds" %(total_time))

