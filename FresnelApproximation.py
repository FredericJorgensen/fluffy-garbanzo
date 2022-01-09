#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 17:56:01 2021

@author: fredericjoergensen
"""
from numpy import *
import numpy as np
from scipy.integrate import nquad

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy
from numpy.random import randn, shuffle



#Define parameters

lam = 655e-9
k = 2 * pi / lam
R = 6e-5

def U(x, y, z):
    h0 = k* 1j/ (2 * pi * z) * exp(-1j * k * z)


    Utemp = lambda r, the: r * exp(-1j * k /(2 * z) * ((r* cos(the) -x)**2  + (r * sin(the) -y)**2 ))
    #U1 = lambda r: quadrature(lambda theta: Utemp(r, theta), 0, 2 * pi)[0]
    #plt.figure()
    #X = linspace(-1,1,100)
    #print((U1(X)))
    #plt.plot(X, U1(X))
    return nquad(Utemp, [[0,R], [0, 2 * pi]])[0] #quadrature(U1, 0, R)[0] * h0



plt.figure()

z = 0.1
N = 30
S = linspace(-0.01, 0.01, N)

x = S
y = S

X, Y = np.meshgrid(x, y)
Z = zeros((N, N))

for (i,x1) in enumerate(S):
    print(i)
    for (j,x2) in enumerate(S):
        Z[i,j]= abs(U(x1, x2, z))**2


print(shape(Z))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)

#title = ax.set_title("plot_surface: given X, Y and Z as 2D:")
#title.set_y(1.01)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()
fig.savefig('3D-PlotFresnelApproximation.pdf')




#for i in range(0, 100):
#    fs[i] = U(x, Y[i], z)




#plt.plot(Y, abs(fs))
#plt.grid(True)




