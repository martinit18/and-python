#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:14:45 2020

@author: delande
"""
import os

option = '-n 2000 -d 0.02 -m 0.1 -i 1.0 '
#option = ''
for i in range(1,50):
  print(i)
  file_name = 'density_momentum_intermediate_'+str(i)+'.dat'
  os.system("python /users/champ/delande/git/and-python/scripts/my_extract_radial.py "+option+file_name)
os.system("python /users/champ/delande/git/and-python/scripts/my_extract_radial.py "+option+"density_momentum_final.dat")
