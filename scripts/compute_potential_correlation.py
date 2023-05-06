#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dominique Delande"
__copyright__ = "Copyright (C) 2020 Dominique Delande"
__license__ = "GPL version 2 or later"
__version__ = "1.0"
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
# ____________________________________________________________________
#
# compute_prop.py
# Author: Dominique Delande
# Release date: April, 27, 2020
# License: GPL2 or later
"""
Created on Fri Aug 16 17:05:10 2019

@author: delande

Disordered system in any dimension
Discretization in configuration space
3-point discretization of the Laplace operator along each direction

This program computes the temporal propagation of any initial state

V0 (=disorder_strength) is the square root of the variance of the disorder
sigma (=correlation_length) is the correlation length of the disorder
Various disorder types can be used, defined by the disorder_type variable:
  anderson gaussian: Usual Anderson model (spatially uncorrelated disorder) with Gaussian on-site distribution of the disorder
  regensburg: Gaussian on-site distribution with spatial correlation function <V(r)V(r+delta)>=V_0^2 exp(-0.5*delta^2/sigma^2)
  singapore: Gaussian on-site distribution with spatial correlation function <V(r)V(r+delta)>=V_0^2 sinc(delta/sigma)
  konstanz: Rayleigh on-site distribution (average V_0, variance V_0^2) with spatial correlation function <V(r)V(r+delta)>=V_0^2 exp(-0.5*delta^2/sigma^2)
  speckle: Rayleigh on-site distribution (average V_0, variance V_0^2) with spatial correlation function <V(r)V(r+delta)>=V_0^2 (1+sinc^2(delta/sigma))

  Internally, the wavefunction can be stored using two different layouts.
  This does NOT affect the storage of the wavefunctions used for measurements, which is always 'complex'
  This affecty only the vector used in the guts of the propagation algorithm.
  'real' is usually a bit faster.
  For data_layout == 'complex':
        wfc is assumed to be in format where
        wfc[0:2*dim_x:2] contains the real part of the wavefunction and
        wfc[1:2*dim_x:2] contains the imag part of the wavefunction.
      For data_layout == 'real':
        wfc is assumed to be in format where
        wfc[0:dim_x] contains the real part of the wavefunction and
        wfc[dim_x:2*dim_x] contains the imag part of the wavefunction.

"""

import os
import time
import numpy as np
import getpass
import sys
import argparse
sys.path.append('/users/champ/delande/git/and-python')
sys.path.append('/home/lkb/delande/git/and-python')
sys.path.append('/home/delande/git/and-python')
import anderson



def main():
  parser = argparse.ArgumentParser(description='Compute the potential correlation function')
  parser.add_argument('filename', type=argparse.FileType('r'), help='name of the file containing parameters of the calculation')
  args = parser.parse_args()
  parameter_file = args.filename.name

# Determine is the script is ran inside MPI
# If yes, set the mpi_version to True, the  MPI communicator to comm, the number of
# MPI processes to nprocs, the rank of the current process to rank, and
# set mpi_string to something containing minimal MPI information
# If not run inside MPI, nprocs=1 and rank=0
  mpi_version = False
  nprocs = 1
  rank = 0
  comm = None
  environment_string='Script ran by '+getpass.getuser()+' on machine '+os.uname()[1]+'\n'\
             +'Name of python script:  {}'.format(os.path.abspath( __file__ ))+'\n'\
             +'Name of parameter file: {}'.format(os.path.abspath(parameter_file))+'\n'

  initial_time=time.asctime()
#    hostname = os.uname()[1].split('.')[0]
  print("Python script runs on machine : "+os.uname()[1])
  print("Name of python script:  {}".format(os.path.abspath( __file__ )))
  print("Name of parameter file: {}".format(os.path.abspath(parameter_file)))
  print()
  print("Python script started on: {}".format(initial_time))
  print()

# Parse parameter file and prepare the useful objects:
# H for the Hamiltonian of the system
# initial_state for the initial state
# propagation for the propagation scheme
# measurement for the measurement scheme
# measurement_global is used to gather (average) the results for several disorder configurations
# my_list_of_sections is the list of sections needed for this particular calculation
# Can be in any order
# The list determines the various structures returned by the routine
# Must be consistent otherwise disaster guaranted
  my_list_of_sections = []
  geometry, H, _, n_config = anderson.io.parse_parameter_file(mpi_version,comm,nprocs,rank,parameter_file,my_list_of_sections)
#  propagation.chebyshev_propagation = anderson.propagation.chebyshev_propagation_generic
#  propagation.chebyshev_step = eval("anderson.propagation.chebyshev_step_generic_"+propagation.data_layout)
#  H.apply_h = H.apply_h_generic
#  print(initial_state.randomize_initial_state)
  t1=time.perf_counter()
  my_timing=anderson.timing.Timing()

#  if rank==0:

# Print various things for the initial state
# At this point, it it not yet known whether there is a C implementation available
  header_string = environment_string+anderson.io.output_string(H,n_config,nprocs)
#  print(header_string)
# Print the initial density and wavefunction
#  anderson.io.print_measurements_initial(measurement_global,initial_state,header_string=header_string)
# Here starts the loop over disorder configurations
  for i in range(n_config):
# The following lines just for generating and printing a single realization of disorder
    H.generate_disorder(i+rank*n_config+1234)
#   print(H.disorder)
    if i==0:
      anderson.io.my_save_routine('potential.dat',H.disorder-H.diagonal,header=header_string)
# The following lines for computing the potential correlation function
      pot_correl = np.real(anderson.compute_correlation(H.disorder-H.diagonal,H.disorder-H.diagonal,shift_center=True))
    else:
      pot_correl += np.real(anderson.compute_correlation(H.disorder-H.diagonal,H.disorder-H.diagonal,shift_center=True))
  pot_correl /= n_config
#  np.savetxt('potential_correlation.dat',pot_correl,header=header_string)
  anderson.io.my_save_routine('potential_correlation.dat',pot_correl,header=header_string)


# Calculation is essentially finished
# It remains to output the results
  t2 = time.perf_counter()
  my_timing.TOTAL_TIME = t2-t1


  final_time = time.asctime()
  print("Python script ended on: {}".format(final_time))
  print("Wallclock time {0:.3f} seconds".format(t2-t1))
  print()
  print("Total CPU time       = {0:.3f}".format(my_timing.TOTAL_TIME))

if __name__ == "__main__":
  main()
