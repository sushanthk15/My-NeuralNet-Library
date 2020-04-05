#####################################################################
# Name :  Sushanth Keshav | Matrikel Nr: 63944
# Topic : Modelling the yield surfaces asper Bigoni Piccolroaz Model
# Programming Language : Python
# Last Updated : 04.04.2020
# Credits: This program was mentored by Dr.Ing. Martin Abendroth
#####################################################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from datetime import date
import time

fc=1 # uniaxial compression yield strength [0:inf]
ft=1 # uniaxial tension yield strength [0:inf]
r=fc/ft	# compression tension ratio

'''Defining paramter values for Meridian function '''
pc=1    #	hydrostatic compression strength [0:inf]
c=1	# hydrostatic tension strength [0:inf]
M=0.5 # pressure sensitivity [0:inf]
m=2 # distorsion of meridian function (rounding) [1:inf]
a=1 #	distorsion of meridian function (skewness) [0:2]

''' Defining parameter values for Deviatoric function '''
b= 1 # shape of deviatoric section () [0:2]
g= 0.95 # shape of deviatoric section () [0:1]

fac=M*pc

''' Formulating functions for finding the yield surface '''
def phi(u): #Pressure sensitivity
    return (u+c)/(pc+c)
	
def Meridian(u): #Meridian FUnction u = 'p' -----> Hydrostatic Part of stress i.e, -(1/3)*trace(sigma)
    return -M*pc*np.sqrt((phi(u)-phi(u)**m)*(2.*(1.-a)*phi(u)+a))
	
def deviatoric(v): #Deviatoric Function v = 'theta'------> Lode Angle dependence basically g(theta) in Bigoni Piccol. Formula
    return 1./np.cos(b*np.pi/6.-1./3.*np.arccos(g*np.cos(3.*v)))

def yield_surface(u,v): #This is the 'q' part ------> Since at the yield point the function = 0 and so q = -Meridian * Deviatoric
    return -Meridian(u)*deviatoric(v)

# p,theta and q are scaled by fac
def x(u,v):
    return -1./fac*(1./3.*u+2./3.*yield_surface(u,v)*np.cos(v))

def y(u,v):
    return -1./fac*(1./3.*u+2./3.*yield_surface(u,v)*np.cos(v-2./3.*np.pi))

def z(u,v):
    return -1./fac*(1./3.*u+2./3.*yield_surface(u,v)*np.cos(v+2./3.*np.pi))

def radius(x,y,z):
    return np.sqrt(x**2+y**2+z**2) 

#Determining the invariant q

hydrostatic_pressure = []
lode_angle = []
yield_potential = []

# condition p is in [-c,pc] and \theta is in [0, pi/3]

for u in np.linspace(-c,pc,25):
    for v in np.linspace(0,np.pi/3.,40):
        hydrostatic_pressure.append(u)
        lode_angle.append(v)
        yield_potential.append(yield_surface(u,v))


#Converting lists into arrays -- handy for plotting
hydrostatic_pressure = np.array(hydrostatic_pressure)
lode_angle = np.array(lode_angle)
yield_potential = np.array(yield_potential)

#3D plots of the yield surface
fig1 = plt.figure(figsize=(10,6))
ax1 = plt.axes(projection='3d')
ax1.plot_wireframe((hydrostatic_pressure/pc).reshape(25,40),lode_angle.reshape(25,40),(yield_potential/pc).reshape(25,40))
ax1.set_xlabel('p/pc', fontsize=15)
ax1.set_ylabel('$\Theta$', fontsize=15)
ax1.set_zlabel('q/pc', fontsize=15)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


