import numpy as np
import matplotlib.pyplot as plt 
from ffnet import ffnet, mlgraph, savenet, loadnet, exportnet
conec = mlgraph( (2,5,5,1) )
net = ffnet(conec)
data = np.loadtxt('log.txt')
X = data[:,:2]
y = data[:,2:]

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
#net.train_tnc(input_data, output_data, maxfun = 1000)
net.train_momentum(X_train, Y_train, eta=0.01, momentum=0.9, maxiter=10000, disp=0)
output, regress = net.test(X_test, Y_test)
savenet(net, "log.net")
accuracy_percent = abs(np.mean(np.divide((output - Y_test),Y_test)))
print('Accuracy of testing: ',(1-accuracy_percent)*100)
fig , ax = plt.subplots(ncols=3, figsize=(20,8))
ax[0].plot(net.weights, 'o')
ax[0].
#print(output)
#print(regress)
#plt.plot(output_data[:,0],label='Truth')
ax[1].scatter(Y_test ,output[:,0], label='Predicted')
ax[1].plot(Y_test ,Y_test, 'r--',label='Truth')
ax[1].set_aspect(aspect='equal')
ax[2].plot(Y_test, 'r--')
ax[2].plot(output)
ax[1].legend()
plt.show()