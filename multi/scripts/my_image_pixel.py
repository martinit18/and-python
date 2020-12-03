#!/usr/bin/env python
import sys
import math
import string
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#for arg in sys.argv:
#  print arg

f = open(sys.argv[1],'r')
#f = open('eigenstate.dat','r')
str1=f.readline()
str2=f.readline()
n1=int(str1.lstrip('#'))
n2=int(str2.lstrip('#'))
#print sys.argv,len(sys.argv)
arr=np.loadtxt(sys.argv[1],comments='#').reshape(n1,n2,-1)
#print arr
if len(sys.argv)==3:
  column=int(sys.argv[2])
else:
  column=-1 
Z=arr[:,:,column]
#delta = 0.025
#x = y = np.arange(-3.0, 3.0, delta)
#X, Y = np.meshgrid(x, y)
#Z = np.exp(-(X*X+Y*Y))
print('Maximum value = ',Z.max())
print('Minimum value = ',Z.min())
im = plt.imshow(Z,origin='lower',interpolation='nearest')

plt.show()

