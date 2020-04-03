import numpy as np
import matplotlib.pyplot as plt 
from ffnet import ffnet, mlgraph, savenet, loadnet, exportnet

#Defining the Network Plan
conec = mlgraph((2,20,20,20,1))
net = ffnet(conec)

#Loading the Data
data = np.loadtxt('output_bigoni_steel.txt')
X = data[:,:2]
y = data[:,2:]
#noise = np.random.normal(loc=0, scale=1.0, size=len(y))
#noise1 = noise.reshape(y.shape)
#y_n = y #+ noise1

#Test train data split
n_tot=X.shape[0]
split = 0.7
n_train = int(split*n_tot)

mask = np.zeros((n_tot), dtype=bool)
mask[:n_train] = True

np.random.shuffle(mask)

X_train = X[mask]
Y_train = y[mask]

X_test = X[~mask]
Y_test = y[~mask]

#Training the network using momentum optimization
net.train_momentum(X_train, Y_train, eta=0.01, momentum=0.9, maxiter=10000, disp=0)

#Evaluating the Data
output, regress = net.test(X_test, Y_test)

#saving network weights
savenet(net, "log.net")

#Accuracy calculations
accuracy_percent = abs(np.mean(np.divide((output - Y_test),Y_test)))
print('Accuracy of testing: ',(1-accuracy_percent)*100)

#Plotting 
fig , ax = plt.subplots(ncols=3, figsize=(20,8))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
ax[0].plot(net.weights , label='Weights')
ax[0].set_title('Network Weights')

ax[1].scatter(Y_test ,output[:,0], label='Predicted')
ax[1].plot(Y_test ,Y_test, 'r--',label='Truth')
ax[1].legend()
ax[1].set_title('Output convergence during Testing')
#ax[1].set_aspect(aspect='equal')
ax[2].plot(Y_test, 'r--')
ax[2].plot(output)
ax[1].legend()
plt.show()