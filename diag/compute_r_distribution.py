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
import numpy as np
import getpass
import timeit
import sys
import argparse
sys.path.append('/users/champ/delande/git/and-python')
sys.path.append('/home/lkb/delande/git/and-python')
import anderson

def main():
  parser = argparse.ArgumentParser(description='Compute r distribution at a given energy')
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
# propagation for the propagation scheme
# measurement for the measurement scheme
# measurement_global is used to gather (average) the results for several disorder configurations
  geometry, H, diagonalization, n_config = anderson.io.parse_parameter_file(mpi_version,comm,nprocs,rank,parameter_file,['Diagonalization','Spin'])
  t1=time.perf_counter()
  my_timing=anderson.timing.Timing()

  if rank==0:
    header_string = environment_string+anderson.io.output_string(H,n_config,nprocs,diagonalization=diagonalization)

  tab_r = np.zeros((diagonalization.number_of_eigenvalues-2)*n_config)
  tab_energy = np.zeros((diagonalization.number_of_eigenvalues-2)*n_config)

  # Here starts the loop over disorder configurations
  for i in range(n_config):
    tab_energy[i*(diagonalization.number_of_eigenvalues-2):(i+1)*(diagonalization.number_of_eigenvalues-2)], tab_r[i*(diagonalization.number_of_eigenvalues-2):(i+1)*(diagonalization.number_of_eigenvalues-2)] = diagonalization.compute_tab_r(i+rank*n_config, H)
  energy_min = np.zeros(1)
  energy_min[0] = np.min(tab_energy)
  energy_max = np.zeros(1)
  energy_max[0] = np.max(tab_energy)
#  print(energy_min,energy_max)
  if mpi_version:
    start_mpi_time = timeit.default_timer()
    tab_r_glob = np.zeros((diagonalization.number_of_eigenvalues-2)*n_config*nprocs)
    comm.Gather(tab_r,tab_r_glob)
    from mpi4py import MPI
    energy_glob=np.zeros(1)
    comm.Reduce(energy_min,energy_glob,op=MPI.MIN)
    energy_min[0] = energy_glob[0]
    comm.Reduce(energy_max,energy_glob,op=MPI.MAX)
    energy_max[0] = energy_glob[0]
 #    comm.Allreduce(MPI.IN_PLACE,energy_max,op=MPI.MAX)
    my_timing.MPI_TIME+=(timeit.default_timer() - start_mpi_time)
  else:
    tab_r_glob = tab_r
  t2=time.perf_counter()
  my_timing.TOTAL_TIME = t2-t1
  if mpi_version:
    my_timing.mpi_merge(comm)
  if rank==0:
#    print(energy_min[0],energy_max[0])
    header_string +='Minimum energy = '+str(energy_min[0])+'\nMaximum energy = '+str(energy_max[0])+'\n'
    tab_histogram, bin_edges = np.histogram(tab_r_glob, bins=diagonalization.number_of_bins, range=(0.,1.0), density=True)
    anderson.io.output_density('histogram_r.dat',tab_histogram,H,header_string=header_string,tab_abscissa=bin_edges[1:],data_type='histogram_r')
    final_time = time.asctime()
    print("Python script ended on: {}".format(final_time))
    print("Wallclock time {0:.3f} seconds".format(t2-t1))
    print()
    if mpi_version:
      print("MPI time             = {0:.3f}".format(my_timing.MPI_TIME))
    print("Total_CPU time       = {0:.3f}".format(my_timing.TOTAL_TIME))

if __name__ == "__main__":
  main()
