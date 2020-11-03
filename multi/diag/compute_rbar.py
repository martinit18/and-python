#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:05:10 2019

@author: delande

"""
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
# compute_IPR.py
# Author: Dominique Delande
# Release date: April, 27, 2020
# License: GPL2 or later

import os
import time
import math
import numpy as np
import copy
import getpass
import configparser
import timeit
import sys
import socket
import argparse
sys.path.append('../')
sys.path.append('/users/champ/delande/git/and-python/multi')
import anderson


def main():
  parser = argparse.ArgumentParser(description='Compute rbar vs. energy')
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
# propagation for the propagation scheme
# measurement for the measurement scheme
# measurement_global is used to gather (average) the results for several disorder configurations
  H, diagonalization, n_config = anderson.io.parse_parameter_file(mpi_version,comm,nprocs,rank,parameter_file,['Diagonalization'])
# Force lapack diagonalization
  diagonalization.method = 'lapack'
  diagonalization.number_of_eigenvalues = H.ntot
  t1=time.perf_counter()
  timing=anderson.Timing()

  if rank==0:

# Print various things for the initial state
# At this point, it it not yet known whether there is a C implementation available
    header_string = environment_string+anderson.io.output_string(H,n_config,nprocs,diagonalization=diagonalization)

  tab_r = np.zeros(H.ntot-2)
  tab_energy = np.zeros(H.ntot-2)
  emin=0.0
#  emax=2.0
  emax=4.0
  nsteps=50
  header_string += 'emin                            = '+str(emin)+'\nemax                            = '+str(emax)+'\nnumber of energy steps          = '+str(nsteps)+'\n\n'
  estep = (emax-emin)/nsteps
  tab_num = np.zeros(nsteps,dtype=int)
  tab_hist_r = np.zeros(nsteps)
  tab_middle_energy = np.arange(start=emin,stop=emax,step=estep)+0.5*estep

  # Here starts the loop over disorder configurations
  for i in range(n_config):
    tab_energy, tab_r = diagonalization.compute_tab_r(i+rank*n_config, H)
# accumulate r values in an energy-dependent array
    for j in range(H.ntot-2):
      k = int((tab_energy[j]-emin)/estep)
      k = max(k,0)
      k = min(k,nsteps-1)
      tab_num[k]+=1
      tab_hist_r[k]+=tab_r[j]
  for k in range(nsteps):
    tab_hist_r[k]/=tab_num[k]
#    H.generate_full_matrix()
#    print(H.generate_full_complex_matrix(1.0j))
  if mpi_version:
    start_mpi_time = timeit.default_timer()
    tab_hist_r_glob = np.empty_like(tab_hist_r)
    comm.Reduce(tab_hist_r,tab_hist_r_glob)
    tab_hist_r_glob/=nprocs
    timing.MPI_TIME+=(timeit.default_timer() - start_mpi_time)
  else:
    tab_hist_r_glob = tab_hist_r
  t2=time.perf_counter()
  timing.TOTAL_TIME = t2-t1
  if mpi_version:
    timing.mpi_merge(comm)
  if rank==0:
#    print(tab_energy_glob)
#    print(tab_IPR_glob)
#    print(tab_IPR_glob.shape)
    anderson.io.output_density('tab_rbar.dat',tab_hist_r_glob,H,header_string=header_string,tab_abscissa=tab_middle_energy,data_type='rbar')
    final_time = time.asctime()
    print("Python script ended on: {}".format(final_time))
    print("Wallclock time {0:.3f} seconds".format(t2-t1))
    print()
    if mpi_version:
      print("MPI time             = {0:.3f}".format(timing.MPI_TIME))
    print("Total_CPU time       = {0:.3f}".format(timing.TOTAL_TIME))

if __name__ == "__main__":
  main()
