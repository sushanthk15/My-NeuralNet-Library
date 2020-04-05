#!/usr/bin/env python
# coding: utf-8

# ## Loading the Libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from neural_net_library import *
from neural_net_library import my_neural_network


# ## Training the Model

# In[ ]:


hidden_layer_dims = [10,10,10]
model = my_neural_network(hidden_layer_dims)
file_name = 'output_bigoni_martin_ellipse.txt'
model.load_dataset(file_name)
model.test_train_split()
optim = 'Momentum'
model.NN_model(10000, 0.01,beta1=0.9,beta2=0.999,regularization = 0.7,activation_function = "sigmoid", batching = False, batch_size= 128,optimizer = optim,early_stop=False,learning_rate_decay=0.5)


### Testing the Model

# In[ ]:


#Load the unseen data or the testing data
model_t = my_neural_network(hidden_layer_dims)
file_namet= "output_bigoni_ellipse_check1.txt"
model_t.load_dataset(file_namet)

#Retrieve the Normalized inputs and outputs
X_t = model_t.input_data
X_t = X_t.T
Y_t=model_t.output_data

#Predict the output using the Forward Propagation
model.forward_propagation(X_t)


# In[ ]:


#Collect the predicted values and reshape it according  to the actual one. Helps in consistency
Y_pred = model.activations[-1]
Y_pred = Y_pred.reshape(Y_t.shape)


# In[ ]:


#R^2 Score Calculation
u = ((Y_t - Y_pred)**2).sum()
v = ((Y_t - Y_t.mean())**2).sum()
error_percent = u/v#np.absolute(np.mean(np.divide(np.mean(self.activations[-1] - Y),np.mean(Y))))
print('The R^2 score of testing:', (1-error_percent))


# In[ ]:


#Plot the Actual v/s Predicted

fig,ax = plt.subplots(figsize=(10,8))
ax.scatter(Y_t,Y_pred,edgecolors=(1,0,1))
ax.plot([Y_t.min(), Y_t.max()], [Y_t.min(), Y_t.max()], 'k--', lw=4)
plt.grid(False)
plt.xlabel("Actual Output", fontsize=15)
plt.ylabel("Predicted Output", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.text(0.065,0.95, r'Cross Validated predictions', fontsize=15)
plt.text(0.075,0.87, r'Optimizer = {}'.format(optim), fontsize=15)
plt.text(0.08,0.80, r'$R^2$= %.2f' %((1-error_percent)), fontsize=15)
ax.set_xlim([0,1])
ax.set_ylim([0,1])

plt.show()


# In[ ]:


## Plotting the Costs 
fig,ax2 = plt.subplots(figsize=(10,6))
ax2.semilogy(np.array(model.cost)  ,label='Training Cost')
ax2.semilogy(np.array(model.CV_cost), label='Validation cost')
ax2.set_xlabel('Epochs', fontsize=20)
ax2.set_ylabel('log(Cost)', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(prop={"size":15})
plt.show()


# ## Assinging the Input to variables, helps in visualization 
# ### The U refers to the hydrostatic pressure
# ### The V refers to LodeAngle

# In[ ]:


U=X_t[0,:] #Hydrostatic Pressure p
V=X_t[1,:] #Lode Angle Theta


# ## Plotting the Actual and Predicted yield Surfaces

# In[ ]:


from matplotlib import cm
fig1 = plt.figure(figsize=(10,8))
ax1 = plt.axes(projection='3d')

ax1.plot_wireframe(U.reshape((25,10)),V.reshape((25,10)),Y_t.reshape((25,10)) ,color='green' , label='Actual Yield Surface')#
ax1.plot_wireframe(U.reshape((25,10)),V.reshape((25,10)),Y_pred.reshape((25,10)), color='red',  label='Predicted Yield Surface')
ax1.set_xlabel('p', fontsize=20)
ax1.set_ylabel('$\Theta$', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax1.set_zlabel('q', fontsize=20)
plt.legend(prop={"size":15})
plt.show()

