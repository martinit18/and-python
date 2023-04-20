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

"""

import os
import time
import numpy as np
import getpass
import sys
import timeit
import argparse
sys.path.append('/users/champ/delande/git/and-python')
sys.path.append('/home/lkb/delande/git/and-python')
sys.path.append('/home/delande/git/and-python')
import anderson

#import matplotlib.pyplot as plt

def main():
  parser = argparse.ArgumentParser(description='Compute the Lyapounov (inverse of localization length) vs. energy')
  parser.add_argument('filename', type=argparse.FileType('r'), help='name of the file containing parameters of the calculation')
  args = parser.parse_args()
  parameter_file = args.filename.name

# Determine is the script is ran inside MPI
# If yes, set the mpi_version to True, the  MPI communicator to comm, the number of
# MPI processes to nprocs, the rank of the current process to rank, and
# set mpi_string to something containing minimal MPI information
# If not run inside MPI, nprocs=1 and rank=0
  mpi_version, comm, nprocs, rank, mpi_string = anderson.determine_if_launched_by_mpi()
  environment_string='Script ran by '+getpass.getuser()+' on machine '+os.uname()[1]+'\n'\
             +'Name of python script:  {}'.format(os.path.abspath( __file__ ))+'\n'\
             +'Name of parameter file: {}'.format(os.path.abspath(parameter_file))+'\n'\
             +mpi_string+'\n'

  if rank==0:
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
# my_list_of_sections is the list of sections needed for this particular calculation
# Can be in any order
# The list determines the various structures returned by the routine
# Must be consistent otherwise disaster guaranted
  my_list_of_sections = ['Lyapounov']
  geometry, H, _, lyapounov, n_config = anderson.io.parse_parameter_file(mpi_version,comm,nprocs,rank,parameter_file,my_list_of_sections)

  t1=time.perf_counter()
  timing=anderson.timing.Timing()

  if rank==0: header_string = environment_string+anderson.io.output_string(H,n_config,nprocs,lyapounov=lyapounov)
  number_of_e_steps = lyapounov.number_of_e_steps
#  tab_mean_integrated_dos = np.zeros(number_of_k_steps+1)
#  tab_std_integrated_dos = np.zeros(number_of_k_steps+1)
#  tab_mean_lyapounov = np.zeros(number_of_e_steps+1)
#  tab_std_lyapounov = np.zeros(number_of_e_steps+1)
  tab_lyapounov = np.zeros(number_of_e_steps+1)
  tab_global_lyapounov = np.zeros((2,number_of_e_steps+1))
  debug = True
#  debug = False
  if mpi_version and debug:
    debug = False
    if rank==0:
      print('Debugging is not supported in the MPI version, I switch it off\n')
 # Here starts the loop over disorder configurations
  for i in range(n_config):
    if debug:
      tab_lyapounov, tab_x, tab_log_trans = lyapounov.compute_lyapounov(i+rank*n_config, H, timing, debug=True)
      if i==0:
        tab_global_log_trans = np.zeros_like(tab_log_trans)
      tab_global_log_trans += tab_log_trans
    else:
      tab_lyapounov = lyapounov.compute_lyapounov(i+rank*n_config, H, timing)
    tab_global_lyapounov[0] += tab_lyapounov
    tab_global_lyapounov[1] += tab_lyapounov**2
  if mpi_version:
    start_mpi_time = timeit.default_timer()
    tab_global_lyapounov_glob = np.empty_like(tab_global_lyapounov)
    comm.Reduce(tab_global_lyapounov,tab_global_lyapounov_glob)
    tab_global_lyapounov = np.copy(tab_global_lyapounov_glob)
    timing.MPI_TIME+=(timeit.default_timer() - start_mpi_time)
  t2 = time.perf_counter()
  timing.TOTAL_TIME = t2-t1
  if mpi_version:
    timing.mpi_merge(comm)

  if debug:
    np.savetxt('log_trans.dat',np.column_stack((tab_x,tab_global_log_trans/n_config)))
# Compute mean value and standard deviation
  if rank==0:
    tab_global_lyapounov /= n_config*nprocs
    if n_config*nprocs>1:
      tab_global_lyapounov[1] = np.sqrt(np.abs(tab_global_lyapounov[1]-tab_global_lyapounov[0]**2)/(n_config*nprocs-1))
    else:
      tab_global_lyapounov[1] = 0.0
#    print(tab_global_lyapounov[0])
    anderson.io.output_density('lyapounov_vs_energy.dat',tab_global_lyapounov,H,header_string=header_string,tab_abscissa=lyapounov.tab_energy,data_type='lyapounov')

    """

# Other code to compute an histogram of the lyapounov exponent at a given energy
  lyapounov = anderson.lyapounov.Lyapounov(e_histogram,e_histogram,0)
  header_string = environment_string+anderson.io.output_string(H,n_config,nprocs,lyapounov=lyapounov)
  tab_lyapounov = np.zeros(n_config)
  for i in range(n_config):
    tab_lyapounov[i] = lyapounov.compute_lyapounov(i, H)
  tab_histogram, bin_edges = np.histogram(tab_lyapounov, bins=number_of_bins, range=(lyapounov_min,lyapounov_max), density=True)
  anderson.io.output_density('histogram_lyapounov.dat',bin_edges[1:],tab_histogram,header_string,print_type='histogram_lyapounov')
    """


    final_time = time.asctime()
    print("Python script ended on: {}".format(final_time))
    print("Wallclock time {0:.3f} seconds".format(t2-t1))
    print()
    print("Lyapounov Matrix Factorization time = {0:.3f}".format(timing.LYAPOUNOV_MATRIX_FACTORIZATION_TIME))
    print("Number of ops                       = {0:.4e}".format(timing.LYAPOUNOV_MATRIX_FACTORIZATION_NOPS))
    print("Lyapounov Matrix Solution time      = {0:.3f}".format(timing.LYAPOUNOV_MATRIX_SOLUTION_TIME))
    print("Number of ops                       = {0:.4e}".format(timing.LYAPOUNOV_MATRIX_SOLUTION_NOPS))
    print("Lyapounov scalar time               = {0:.3f}".format(timing.LYAPOUNOV_SCALAR_TIME))
    print("Number of ops                       = {0:.4e}".format(timing.LYAPOUNOV_SCALAR_NOPS))
    print("Total Lyapounov time                = {0:.3f}".format(timing.LYAPOUNOV_TIME))
    print("Total Lyapounov ops                 = {0:.4e}".format(timing.LYAPOUNOV_NOPS))
#    print("Number of ops        = {0:.4e}".format(timing.LYAPOUNOV_NOPS))
    if mpi_version:
      print("MPI time                            = {0:.3f}".format(timing.MPI_TIME))
    print()
    print("Total time                          = {0:.3f}".format(timing.TOTAL_TIME))

if __name__ == "__main__":
  main()
