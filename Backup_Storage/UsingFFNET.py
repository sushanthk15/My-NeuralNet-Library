import numpy as np
import matplotlib.pyplot as plt 
from ffnet import ffnet, mlgraph, savenet, loadnet, exportnet
conec = mlgraph( (2,15,15,1) )
net = ffnet(conec)
data = np.loadtxt('Bigoni_resultant.txt')
input_data = data[:,:-1]
output_data = data[:,-1]
#net.train_tnc(input_data, output_data, maxfun = 1000)
net.train_momentum(input_data, output_data, eta=0.01, momentum=0.8, maxiter=60000, disp=0)
output, regress = net.test(input_data, output_data)
accuracy_percent = np.mean(np.divide((output - output_data),output_data))
print(accuracy_percent)
#print(net.weights)
#print(output)
#print(regress)
plt.plot(output_data,label='Truth')
plt.plot(output, label='Predicted')
plt.legend()
plt.show()