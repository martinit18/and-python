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
# compute_spectral_function.py
# Author: Dominique Delande
# Release date: April, 27, 2020
# License: GPL2 or later
"""
Created on Fri Aug 16 17:05:10 2019

@author: delande

Disordered many-dimensional system
Discretization in configuration space
3-point discretization of the Laplace operator (in each direction)

This program computes the spectral function (using Fourier transform of the temporal propagation)

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
        wfc[0:2*ntot:2] contains the real part of the wavefunction and
        wfc[1:2*ntot:2] contains the imag part of the wavefunction.
      For data_layout == 'real':
        wfc is assumed to be in format where
        wfc[0:ntot] contains the real part of the wavefunction and
        wfc[ntot:2*ntot] contains the imag part of the wavefunction.

"""

import os
import time
import math
import numpy as np
import getpass
import copy
import configparser
import sys
import socket
import argparse
sys.path.append('/users/champ/delande/git/and-python/multi')
sys.path.append('/home/lkb/delande/git/and-python/multi')
import anderson

def main():
  parser = argparse.ArgumentParser(description='Compute spectral function using the Gross-Pitaevskii or Schoedinger equation')
  parser.add_argument('filename', type=argparse.FileType('r'), help='name of the file containing parameters of the calculation')
  args = parser.parse_args()
  parameter_file = args.filename.name

# Determine is the script is ran inside MPI
# If yes, set the mpi_version to True, the  MPI communicator to comm, the number of
# MPI processes to nprocs, the rank of the current process to rank, and
# set mpi_string to something containing minimal MPI information
# If not run inside MPI, nprocs=1 and rank=0
  mpi_version, comm, nprocs, rank, mpi_string = anderson.determine_if_launched_by_mpi()
  environment_string='Script ran by '+getpass.getuser()+' on machine '+socket.getfqdn()+'\n'\
             +'Name of python script:  {}'.format(os.path.abspath( __file__ ))+'\n'\
             +'Name of parameter file: {}'.format(os.path.abspath(parameter_file))+'\n'\
             +mpi_string+'\n'

  if rank==0:
    initial_time=time.asctime()
#    hostname = os.uname()[1].split('.')[0]
    print("Python script runs on machine : "+socket.getfqdn())
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
# Must be consistent otherwise disaster guaranteed
  my_list_of_sections = ['Wavefunction','Nonlinearity','Propagation','Spectral','Spin']
  geometry, H, initial_state,  propagation, spectral_function, measurement, measurement_global, n_config = anderson.io.parse_parameter_file(mpi_version,comm,nprocs,rank,parameter_file,my_list_of_sections)

  t1=time.perf_counter()
  my_timing=anderson.timing.Timing()
# The following line is a temporary fix
  i_tab_0 = 0

  if rank==0:
    header_string = environment_string+anderson.io.output_string(H,n_config,nprocs,initial_state=initial_state,propagation=propagation,measurement=measurement_global,spectral_function=spectral_function)

#  print(header_string)

# Here starts the loop over disorder configurations
  for i in range(n_config):
 #   print(measurement.tab_spectrum)
    measurement.tab_spectrum = anderson.propagation.compute_spectral_function(i+rank*n_config, geometry, initial_state, H, propagation, measurement, spectral_function, my_timing)
 #   print(measurement.tab_spectrum)
    measurement_global.merge_measurement(measurement)
#  print(measurement_global.tab_position)
#
  if mpi_version:
    measurement_global.mpi_merge_measurement(comm,my_timing)
  t2 = time.perf_counter()
  my_timing.TOTAL_TIME = t2-t1
  if mpi_version:
    my_timing.mpi_merge(comm)
#    print('Before: ',rank,measurement_global.tab_autocorrelation[-1])
#    toto = np.empty_like(measurement_global.tab_autocorrelation)
#    comm.Reduce(measurement_global.tab_autocorrelation,toto,op=MPI.SUM)
#    measurement_global.tab_autocorrelation = toto
#    timing.CHE_TIME = comm.reduce(timing.CHE_TIME)
#    global_timing=comm.reduce(timing)
#    print(rank,global_timing.CHE_TIME)
#    print('After: ',rank,measurement_global.tab_autocorrelation[-1])
  if rank==0:
    environment_string+='Calculation   ended on: {}'.format(time.asctime())+'\n\n'
    measurement_global.normalize(n_config*nprocs)
    header_string = environment_string+anderson.io.output_string(H,n_config,nprocs,initial_state=initial_state,propagation=propagation,measurement=measurement_global,spectral_function=spectral_function, timing=my_timing)
    anderson.io.print_measurements_final(measurement_global,initial_state=initial_state,header_string=header_string)

    """
  i_tab_0 = propagation.first_measurement_autocorr

  header_string=environment_string\
             +params_string\
             +'Temporal autocorrelation function\n'\
             +'Column 1: Time\n'\
             +'Column 2: Real(<psi(0)|psi(t)>)\n'\
             +'Column 3: Imag(<psi(0)|psi(t)>)\n'\
             +'\n'
  np.savetxt('temporal_autocorrelation.dat',np.column_stack([propagation.tab_t_measurement[i_tab_0:]-propagation.tab_t_measurement[i_tab_0],np.mean(np.real(tab_autocorrelation[:,:number_of_measurements-i_tab_0]),0),np.mean(np.imag(tab_autocorrelation[:,:number_of_measurements-i_tab_0]),0)]),header=header_string)
    """

    final_time = time.asctime()
    print("Python script ended on: {}".format(final_time))
    print("Wallclock time {0:.3f} seconds".format(t2-t1))
    print()
    if (propagation.method=='ode'):
      print("GPE time             = {0:.3f}".format(my_timing.GPE_TIME))
      print("Number of time steps =",my_timing.N_SOLOUT)
    else:
      print("CHE time             = {0:.3f}".format(my_timing.CHE_TIME))
      print("Max nonlinear phase  = {0:.3f}".format(my_timing.MAX_NONLINEAR_PHASE))
      print("Max order            =",my_timing.MAX_CHE_ORDER)
    print("Expect time          = {0:.3f}".format(my_timing.EXPECT_TIME))
    if mpi_version:
      print("MPI time             = {0:.3f}".format(my_timing.MPI_TIME))
    print("Dummy time           = {0:.3f}".format(my_timing.DUMMY_TIME))
    print("Number of ops        = {0:.4e}".format(my_timing.NUMBER_OF_OPS))
    print("Total_CPU time       = {0:.3f}".format(my_timing.TOTAL_TIME))

if __name__ == "__main__":
  main()
