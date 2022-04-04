#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:14:45 2020

@author: delande
"""
import os

for i in range(1,60):
  print(i)
  file_name = 'density_momentum_'+"{:03d}".format(i)+'.dat'
  os.system("python /users/champ/delande/git/and-python/scripts/my_extract_radial.py "+file_name)
