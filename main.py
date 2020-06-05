#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""  This script is to simulate a example of the MIT rule """

__author__ = '{Miguel Angel Pimentel Vallejo}'
__email__ = '{miguel.pimentel@umich.mx}'
__date__= '{27/may/2020}'

#Import the modules needed to run the script
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

#Transform the transder function to space states
k = 1
k0 = 2
A,B,C,D = signal.tf2ss([k],[1,1])
AG,BG,CG,DG = signal.tf2ss([k0],[1,1])

#Function of the model with adaptative control
def model (x,t,gamma):

    #Vector of derivatives
    xdot = [0,0,0]

    #Comand signal 
    uc = np.sin(t)

    #Reference model
    xdot[1] = AG[0,0]*x[1] + BG[0,0]*uc

    #Error
    e = C*x[0] - CG*x[1]

    #Parameter thetha
    xdot[2] = -gamma*e*CG*x[1]

    #Control input
    u = x[2]*uc

    #Original system
    xdot[0] = A*x[0] + B*u

    return xdot


# initial condition
x0 = [0,0,0]

# time points
t = np.linspace(0,30,500)

#gamma value
gamma = 1/2

# solve ODE
x1 = odeint(model,x0,t,args=(gamma,))

#plot results
labels = ['$x$','$x_m$','$\\theta$','$u_c$']
plt.figure( )
plt.title('System')
x_plot = plt.plot(t,x1)
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend(x_plot ,labels)

plt.figure()
plt.plot(t,x1[:,2]*np.sin(t),label='$u$')
plt.plot(t,np.sin(t),"--",label='$u_c$')
plt.title('Inputs')
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend()

#gamma value
gamma = 1

# solve ODE
x2 = odeint(model,x0,t,args=(gamma,))

#plot results
labels = ['$x$','$x_m$','$\\theta$','$u_c$']
plt.figure( )
plt.title('System')
x_plot = plt.plot(t,x2)
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend(x_plot ,labels)

plt.figure()
plt.plot(t,x2[:,2]*np.sin(t),label='$u$')
plt.plot(t,np.sin(t),"--",label='$u_c$')
plt.title('Inputs')
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend()

#gamma value
gamma = 2

# solve ODE
x3 = odeint(model,x0,t,args=(gamma,))

#plot results
labels = ['$x$','$x_m$','$\\theta$','$u_c$']
plt.figure( )
plt.title('System')
x_plot = plt.plot(t,x3)
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend(x_plot ,labels)

plt.figure()
plt.plot(t,x3[:,2]*np.sin(t),label='$u$')
plt.plot(t,np.sin(t),"--",label='$u_c$')
plt.title('Inputs')
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend()


plt.figure()
gamma1_plot = plt.plot(t,x1[:,2])
gamma2_plot = plt.plot(t,x2[:,2])
gamma3_plot = plt.plot(t,x3[:,2])
plt.title('Parameter $\\theta$')
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
labels = ['$\\gamma = 0.5$','$\\gamma = 1$','$\\gamma = 2$']
plt.legend(gamma1_plot + gamma2_plot + gamma3_plot ,labels)

plt.show()