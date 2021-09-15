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
# compute_landscape.py
# Author: Dominique Delande
# Release date: April, 27, 2020
# License: GPL2 or later

import os
import time
import math
import numpy as np
import getpass
import configparser
import timeit
import sys
#sys.path.append('../')
sys.path.append('/users/champ/delande/git/and-python/')
import anderson
import mkl
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import rc

def compute_wigner_function(H,tab_wfc):
  number_of_eigenvalues = tab_wfc.shape[1]
  dim_x = tab_wfc.shape[0]
  tab_wigner = np.zeros((dim_x//2,dim_x,number_of_eigenvalues))
  tab_wigner = np.zeros((dim_x,dim_x,number_of_eigenvalues))
  tab_intermediate1 = np.zeros(dim_x,dtype=np.complex128)
  tab_intermediate2 = np.zeros(dim_x,dtype=np.complex128)
#  np.savetxt('orig_x.dat',np.column_stack((np.real(tab_wfc[:,0]),np.imag(tab_wfc[:,0]))))
  for i in range(number_of_eigenvalues):
    for j in range(dim_x):
      tab_intermediate1[0:dim_x-j]=tab_wfc[j:dim_x,i]
      tab_intermediate1[dim_x-j:dim_x]=tab_wfc[0:j,i]
      tab_intermediate2[0:j+1]=tab_wfc[j::-1,i]
      tab_intermediate2[j+1:dim_x]=tab_wfc[:j:-1,i]
#     tab_wigner[:,j,i] = np.real(np.fft.fft(tab_intermediate1*np.conj(tab_intermediate2)))
#      if j==50:
#        np.savetxt('orig_x1.dat',np.column_stack((np.real(tab_intermediate1),np.imag(tab_intermediate1))))
#        np.savetxt('orig_x2.dat',np.column_stack((np.real(tab_intermediate2),np.imag(tab_intermediate2))))
#        np.savetxt('znort_x.dat',np.column_stack((np.real(tab_intermediate1*np.conj(tab_intermediate2)),np.imag(tab_intermediate1*np.conj(tab_intermediate2)))))
#        np.savetxt('znort_p.dat',np.column_stack((np.real(np.fft.fft(tab_intermediate1*np.conj(tab_intermediate2))),np.imag(np.fft.fft(tab_intermediate1*np.conj(tab_intermediate2))))))
      tab_after_fft = np.real(np.fft.fftshift(np.fft.fft(tab_intermediate1*np.conj(tab_intermediate2))))
# The following line gives the "raw" Wigner function, which has a "ghost" shifted by L/2
#      tab_wigner[0:dim_x,j,i] = H.delta_x*tab_after_fft[0:dim_x]/np.pi
# In order to hide the ghost, one averages over consecutive p values in a symmetric way
# p=0 is in line dim_x//2
      tab_wigner[1:dim_x-1,j,i] = H.delta_x*(tab_after_fft[1:dim_x-1]+0.5*tab_after_fft[0:dim_x-2]+0.5*tab_after_fft[2:dim_x])/(2.0*np.pi)
      tab_wigner[0,j,i] = H.delta_x*(tab_after_fft[0]+0.5*tab_after_fft[dim_x-1]+0.5*tab_after_fft[1])/(2.0*np.pi)
      tab_wigner[dim_x-1,j,i] = H.delta_x*(tab_after_fft[dim_x-1]+0.5*tab_after_fft[dim_x-2]+0.5*tab_after_fft[0])/(2.0*np.pi)
  return tab_wigner


if __name__ == "__main__":
  environment_string='Script ran by '+getpass.getuser()+' on machine '+os.uname()[1]+'\n'\
             +'Name of python script: {}'.format(os.path.abspath( __file__ ))+'\n'\
             +'Started on: {}'.format(time.asctime())+'\n'
  try:
# First try to detect if the python script is launched by mpiexec/mpirun
# It can be done by looking at an environment variable
# Unfortunaltely, this variable depends on the MPI implementation
# For MPICH and IntelMPI, MPI_LOCALNRANKS can be checked for existence
#   os.environ['MPI_LOCALNRANKS']
# For OpenMPI, it is OMPI_COMM_WORLD_SIZE
#   os.environ['OMPI_COMM_WORLD_SIZE']
# In any case, when importing the module mpi4py, the MPI implementation for which
# the module was created is unknown. Thus, no portable way...
# The following line is for OpenMPI
    os.environ['OMPI_COMM_WORLD_SIZE']
# If no KeyError raised, the script has been launched by MPI,
# I must thus import the mpi4py module
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    mpi_version = True
    environment_string += 'MPI version ran on '+str(nprocs)+' processes\n\n'
  except KeyError:
# Not launched by MPI, use sequential code
    mpi_version = False
    nprocs = 1
    rank = 0
    environment_string += 'Single processor version\n\n'
  except ImportError:
# Launched by MPI, but no mpi4py module available. Abort the calculation.
    exit('mpi4py module not available! I stop!')

  if (rank==0):
    initial_time=time.asctime()
    hostname = os.uname()[1].split('.')[0]
    print("Python script started on: {}".format(initial_time))
    print("{:>24}: {}".format('from',hostname))
    print("Name of python script: {}".format(os.path.abspath( __file__ )))
#    print("Number of available threads: {}".format(multiprocessing.cpu_count()))
#    print("Number of disorder realizations: {}".format(n_config))

    timing=anderson.Timing()
    t1=time.perf_counter()

    config = configparser.ConfigParser()
    config.read('params.dat')

    Averaging = config['Averaging']
    n_config = Averaging.getint('n_config',1)
    n_config = (n_config+nprocs-1)//nprocs
    print("Total number of disorder realizations: {}".format(n_config*nprocs))
    print("Number of processes: {}".format(nprocs))

    System = config['System']
    system_size = System.getfloat('size')
    delta_x = System.getfloat('delta_x')
    boundary_condition = System.get('boundary_condition','periodic')

    Wavefunction = config['Wavefunction']
    initial_state_type = Wavefunction.get('initial_state')
    k_0 = 2.0*math.pi*Wavefunction.getfloat('k_0_over_2_pi')
    sigma_0 = Wavefunction.getfloat('sigma_0')

    Disorder = config['Disorder']
    disorder_type = Disorder.get('type','anderson_gaussian')
    print(disorder_type)
    correlation_length = Disorder.getfloat('sigma',0.0)
    V0 = Disorder.getfloat('V0',0.0)
    disorder_strength = V0

    Diagonalization = config['Diagonalization']
    diagonalization_method = Diagonalization.get('method','sparse')
    targeted_energy = Diagonalization.getfloat('targeted_energy')
    number_of_energy_levels = Diagonalization.getint('number_of_energy_levels',1)
    pivot_real = Diagonalization.getfloat('pivot_real')
    pivot_imag = Diagonalization.getfloat('pivot_imag')
    pivot = pivot_real+1j*pivot_imag
    IPR_min = Diagonalization.getfloat('IPR_min',0.0)
    IPR_max = Diagonalization.getfloat('IPR_max')
    number_of_bins = Diagonalization.getint('number_of_bins')

    Wigner = config['Wigner']
    p_max = Wigner.getfloat('p_max')
    v_min = Wigner.getfloat('v_min')
    v_max = Wigner.getfloat('v_max')
    multiplicative_factor_for_wfc = Wigner.getfloat('multiplicative_factor_for_wfc',1.0)
    multiplicative_factor_for_density = Wigner.getfloat('multiplicative_factor_for_density',0.0)
    if multiplicative_factor_for_density==0.0:
      multiplicative_factor_for_density = multiplicative_factor_for_wfc**2
    plot_individuals = Wigner.getboolean('plot_individuals',True)
    plot_global = Wigner.getboolean('plot_global',True)

  else:
    n_config = None
    system_size = None
    delta_x = None
    boundary_condition = None
    disorder_type = None
    correlation_length = None
    disorder_strength = None
    diagonalization_method = None
    targeted_energy = None
 #  n_config = comm.bcast(n_config,root=0)

  if mpi_version:
    n_config, system_size, delta_x,boundary_condition  = comm.bcast((n_config, system_size,delta_x,boundary_condition ))
    disorder_type, correlation_length, disorder_strength = comm.bcast((disorder_type, correlation_length, disorder_strength))
    diagonalization_method, targeted_energy = comm.bcast((diagonalization_method, targeted_energy))

  timing=anderson.Timing()
  t1=time.perf_counter()
#  matplotlib.use('pgf')
#  rc('text',usetex=True)
#  rc('text.latex', preamble=r'\usepackage{color}')
#  delta_x = comm.bcast(delta_x)
#  print(rank,n_config,system_size,delta_x)
    # Number of sites
  dim_x = int(system_size/delta_x+0.5)
    # Renormalize delta_x so that the system size is exactly what is wanted and split in an integer number of sites
  delta_x = system_size/dim_x
    #V0=0.025
    #disorder_strength = np.sqrt(V0)
  mkl.set_num_threads(1)
  os.environ["MKL_NUM_THREADS"] = "1"

  assert boundary_condition in ['periodic','open'], "Boundary condition must be either 'periodic' or 'open'"

  position = 0.5*delta_x*np.arange(1-dim_x,dim_x+1,2)
  # Prepare Hamiltonian structure (the disorder is NOT computed, as it is specific to each realization)
  H = anderson.Hamiltonian(dim_x, delta_x, boundary_condition=boundary_condition, disorder_type=disorder_type, correlation_length=correlation_length, disorder_strength=disorder_strength, interaction=0.0)
  initial_state = anderson.Wavefunction(dim_x,delta_x)
  initial_state.type = initial_state_type
  anderson.Wavefunction.plane_wave(initial_state,k_0)
  initial_state.wfc*=np.sqrt(system_size)
  #print(initial_state.wfc)
  diagonalization = anderson.diag.Diagonalization(targeted_energy,diagonalization_method)
   #comm.Bcast(H)
  header_string = environment_string+anderson.io.output_string(H,n_config,nprocs,diagonalization=diagonalization)

  tab_spectrum = diagonalization.compute_full_spectrum(0, H)
  anderson.io.output_density('spectrum.dat',np.arange(dim_x),tab_spectrum,header_string,print_type='density')
  energy = np.zeros(number_of_energy_levels)
  tab_wfc = np.zeros((dim_x,number_of_energy_levels))
  energy, tab_wfc = diagonalization.compute_wavefunction(0, H, k=number_of_energy_levels)
  print('Energy = ',energy)
#  tab_wfc2 = np.zeros((dim_x,1),dtype=np.complex128)
#  tab_wfc2[:,0]=initial_state.wfc
#  np.savetxt('toto.dat',np.abs(tab_wfc2[:,0])**2)
  tab_wigner = compute_wigner_function(H,tab_wfc)
  anderson.io.output_density('potential.dat',position,H.disorder-2.0*H.tunneling,header_string,print_type='potential')
  landscape = diagonalization.compute_landscape(0,H,initial_state,0.0)
  anderson.io.output_density('landscape.dat',position,landscape,header_string,print_type='wavefunction')
  delta_p=np.pi/system_size
  tab_p = np.arange(-delta_p*(dim_x//2),delta_p*(dim_x//2),step=delta_p)
  i_max = np.argmax(tab_p>p_max)
  i_min = np.argmax(tab_p>-p_max)
  tab_kinetic_energy = (1.0-np.cos(tab_p*delta_x))/delta_x**2
#  print(i_min,tab_p[i_min],tab_kinetic_energy[i_min],i_max,tab_p[i_max-1],tab_kinetic_energy[i_max-1],delta_p)
  number_of_levels = 10
  level_max = 0.5*p_max**2+2*V0
  raw_levels = np.linspace(level_max/number_of_levels,level_max,num=number_of_levels)
  if plot_individuals:
    for i in range(number_of_energy_levels):
      if np.amax(tab_wfc[:,i])<-np.amin(tab_wfc[:,i]):
        tab_wfc[:,i]=-tab_wfc[:,i]
      np.savetxt('wigner'+str(i+1)+'.dat',tab_wigner[:,:,i])
  #    np.savetxt('wigner'+str(i)+'_x.dat',np.column_stack([initial_state.position,np.sum(tab_wigner[:,:,i],axis=0)/dim_x]))
   #   print(i,np.sum(np.abs(tab_wfc[:,i])**2)*delta_x,np.sum(tab_wigner[:,:,i])*delta_x*delta_p)
  #    np.savetxt('pipo.dat',tab_wigner[:,100,i])
  #    np.savetxt('pipo2.dat',tab_wigner[:,600,i])
  #    x_wfc = np.sum(np.abs(tab_wfc[:,i])**2*initial_state.position*delta_x)
  #    x_wigner = np.sum(np.sum(tab_wigner[:,:,i],axis=0)*initial_state.position*delta_x/dim_x)
  #    print(i,x_wfc,x_wigner)
      energy_potential = np.sum(np.sum(tab_wigner[:,:,i],axis=0)*(H.disorder-2.0*H.tunneling))*delta_x*delta_p
 #     energy_kinetic = energy[i]-energy_potential
      energy_landscape = np.sum(np.sum(tab_wigner[:,:,i],axis=0)*landscape)*delta_x*delta_p
      energy_kinetic   = np.sum(np.sum(tab_wigner[:,:,i],axis=1)*tab_kinetic_energy)*delta_x*delta_p     
      wigner_times_kinetic = (tab_wigner[:,:,i].T*tab_kinetic_energy).T
#      toto_kinetic=np.sum(wigner_times_kinetic)*delta_x*delta_p
 #     wigner_times_potential = tab_wigner[:,:,i]*(H.disorder-2.0*H.tunneling)
#      toto_potential=np.sum(wigner_times_potential)*delta_x*delta_p
 #     wigner_times_landscape = tab_wigner[:,:,i]*landscape
 #     toto_landscape=np.sum(wigner_times_landscape)*delta_x*delta_p
      crossed_landscape=np.sum(np.sum(wigner_times_kinetic[:,:],axis=0)*landscape)*delta_x*delta_p 
      crossed_potential=np.sum(np.sum(wigner_times_kinetic[:,:],axis=0)*(H.disorder-2.0*H.tunneling))*delta_x*delta_p 
 #     crossed_2=np.sum(np.sum(wigner_times_landscape[:,:],axis=1)*tab_kinetic_energy*delta_x*delta_p) 
      kinetic_square = np.sum(np.sum(wigner_times_kinetic[:,:],axis=1)*tab_kinetic_energy)*delta_x*delta_p 
 #     kinetic_square_2 = np.sum(np.sum(tab_wigner[:,:,i],axis=1)*delta_x*delta_p*tab_kinetic_energy**2) 
 #     potential_square = np.sum(np.sum(wigner_times_potential[:,:],axis=0)*(H.disorder-2.0*H.tunneling)*delta_x*delta_p) 
      potential_square = np.sum(np.sum(tab_wigner[:,:,i],axis=0)*(H.disorder-2.0*H.tunneling)**2)*delta_x*delta_p
 #     landscape_square = np.sum(np.sum(wigner_times_landscape[:,:],axis=0)*landscape*delta_x*delta_p) 
      landscape_square = np.sum(np.sum(tab_wigner[:,:,i],axis=0)*landscape**2)*delta_x*delta_p
 #     print('energies',energy_kinetic,energy_kinetic_landscape,energy_potential,energy_landscape)
 #     print('toto',toto_kinetic,toto_potential,toto_landscape)
 #     print('kinetic_square',kinetic_square,kinetic_square_2)
 #     print('potential_square',potential_square,potential_square_2)
 #     print('landscape_square',landscape_square,landscape_square_2)
 #     print('crossed',crossed_1,crossed_2)
      average_extension_potential = energy_kinetic+energy_potential-energy[i]
      average_extension_landscape = energy_kinetic+energy_landscape-energy[i]
      average_dispersion_potential =  kinetic_square+potential_square+energy[i]**2-2.0*energy[i]*energy_potential-2.0*energy[i]*energy_kinetic+2.0*crossed_potential
      average_dispersion_landscape =  kinetic_square+landscape_square+energy[i]**2-2.0*energy[i]*energy_landscape-2.0*energy[i]*energy_kinetic+2.0*crossed_landscape
      print('average extension',average_extension_potential,average_extension_landscape)
      print('average dispersion',average_dispersion_potential,average_dispersion_landscape)
      
  #    if abs(energy_kinetic_landscape/energy_kinetic-1.0)>0.01:
  #      print('Warning, problem with kinetic energy!')
  #      print('Kinetic energy from potential = ',energy_kinetic)
  #      print('Kinetic energy from Wigner    = ',energy_kinetic_landscape)
  #    print(i,energy_landscape,energy_kinetic_landscape,energy_landscape+energy_kinetic_landscape)
  #    fig, axs = plt.subplots(2,sharex=True,constrained_layout=True)
  #    plt.figure(dpi=300)
      fig, axs = plt.subplots(2,sharex=True)
  #    fig.subplots_adjust(hspace=0.5)
      axs[0].plot(initial_state.position,landscape,color='blue')
      axs[0].plot(initial_state.position,energy[i]+multiplicative_factor_for_wfc*tab_wfc[:,i],color='red')
      axs[0].plot(initial_state.position,H.disorder-2.0*H.tunneling,color='grey',linewidth=0.5)
      axs[0].text(-0.7*system_size,v_min,'Landscape',color='blue',rotation='vertical')
      axs[0].text(-0.75*system_size,v_min,'Wavefunction',color='red',rotation='vertical')
      axs[0].text(-0.65*system_size,v_min,'Potential',color='grey',rotation='vertical')
      axs[0].axis([-0.5*system_size,0.5*system_size,v_min,v_max])
      my_string='From true potential: E='+"{:.4f}".format(energy[i])+' E_pot='+"{:.4f}".format(energy_potential)+' E_kin='+"{:.4f}".format(energy_kinetic)
      axs[0].text(-0.5*system_size,1.3*v_max-0.3*v_min,my_string)
      my_string='From landscape:                       E_pot='+"{:.4f}".format(energy_landscape)+' E_kin='+"{:.4f}".format(energy[i]-energy_landscape)
      axs[0].text(-0.5*system_size,1.1*v_max-0.1*v_min,my_string,color='blue')
      w_max = np.amax(tab_wigner[i_min:i_max,:,i])
  #    print('w_max=',w_max)
      xlist = initial_state.position
      ylist = tab_p[i_min:i_max]
      Z = np.zeros((ylist.size,xlist.size))
      for j in range(xlist.size):
        Z[:,j] = tab_kinetic_energy[i_min:i_max]+landscape[j]
      insert_position = np.argmax(raw_levels>energy[i])
  #    print(raw_levels)
  #    print(insert_position)
      levels = np.insert(raw_levels,insert_position,energy[i])
  #    print(levels)
      linewidths=0.4*np.ones(number_of_levels+1)
      linewidths[insert_position]=1.0
      linestyles=["dotted" for j in range(number_of_levels+1)]
      linestyles[insert_position]="solid"
  #    print(linestyles)
      axs[1].contour(xlist,ylist,Z,colors='black',linewidths=linewidths,linestyles=linestyles,levels=levels)
      im=axs[1].imshow(tab_wigner[i_min:i_max,:,i],origin='lower',interpolation='nearest',aspect='auto',extent=[-0.5*system_size,0.5*system_size,-p_max,p_max],cmap='seismic',vmin=-w_max,vmax=w_max)
      axs[1].set_xlabel('Position')
      axs[1].set_ylabel('Momentum')
      plt.title('Wigner function and contours of landscape+p**2/2')
      cbaxes = fig.add_axes([0.87, 0.12, 0.03, 0.32])
      cb = plt.colorbar(im, cax = cbaxes)
  #    fig.colorbar(im)
      fig.subplots_adjust(hspace=0.4)
      fig.subplots_adjust(bottom=0.12, left=0.18, right=0.85, top=0.85)
      """
      plt.subplot(221)
      plt.plot(initial_state.position,landscape,color='blue')
      plt.plot(initial_state.position,energy[i]+multiplicative_factor_for_wfc*tab_wfc[:,i],color='red')
      plt.text(-0.6*system_size,v_min,'Landscape',color='blue',rotation='vertical')
      plt.text(-0.65*system_size,v_min,'Wavefunction',color='red',rotation='vertical')
      plt.axis([-0.5*system_size,0.5*system_size,v_min,v_max])
      w_max = np.amax(tab_wigner[i_min:i_max,:,i])
      print('w_max=',w_max)
      plt.subplot(223)
      im=plt.imshow(tab_wigner[i_min:i_max,:,i],origin='lower',interpolation='nearest',aspect='auto',extent=[-0.5*system_size,0.5*system_size,-p_max,p_max],cmap='seismic',vmin=-w_max,vmax=w_max)
      plt.subplots_adjust(bottom=0.0, left=0.0, right=1.0, top=1.0)
      plt.xlabel('Position')
      plt.ylabel('Momentum')
      plt.subplot(224)
      plt.colorbar()
      """

      plt.savefig('figure'+str(i+1)+'.png', dpi=300)
      plt.show()
  if plot_global:
# Sum all Wigner functions
    global_wigner = np.sum(tab_wigner,axis=2)/number_of_energy_levels
    global_density = np.sum(tab_wfc**2,axis=1)/number_of_energy_levels
    np.savetxt('global_wigner.dat',global_wigner)
    fig, axs = plt.subplots(2,sharex=True)
#    fig.subplots_adjust(hspace=0.5)
    axs[0].plot(initial_state.position,landscape,color='blue')
    axs[0].plot(initial_state.position,targeted_energy+multiplicative_factor_for_density*global_density,color='red')
    axs[0].plot(initial_state.position,H.disorder-2.0*H.tunneling,color='grey',linewidth=0.5)
    axs[0].text(-0.7*system_size,v_min,'Landscape',color='blue',rotation='vertical')
    axs[0].text(-0.75*system_size,v_min,'Density',color='red',rotation='vertical')
    axs[0].text(-0.65*system_size,v_min,'Potential',color='grey',rotation='vertical')
    my_string="Targeted energy = {:.4f}  V0 = {:.4f}".format(targeted_energy,V0)
    axs[0].text(-0.25*system_size,1.3*v_max-0.3*v_min,my_string)
    my_string="Number of energy levels = {}".format(number_of_energy_levels)
    axs[0].text(-0.25*system_size,1.1*v_max-0.1*v_min,my_string)
    axs[0].axis([-0.5*system_size,0.5*system_size,v_min,v_max])
    w_max = np.amax(global_wigner[i_min:i_max,:])
    print('w_max=',w_max)
    xlist = initial_state.position
    ylist = tab_p[i_min:i_max]
    Z = np.zeros((ylist.size,xlist.size))

    for j in range(xlist.size):
      Z[:,j] = tab_kinetic_energy[i_min:i_max]+landscape[j]
    insert_position = np.argmax(raw_levels>targeted_energy)
#    print(raw_levels)
#    print(insert_position)
    levels = np.insert(raw_levels,insert_position,targeted_energy)
#    print(levels)
    linewidths=0.4*np.ones(number_of_levels+1)
    linewidths[insert_position]=1.0
    linestyles=["dotted" for j in range(number_of_levels+1)]
    linestyles[insert_position]="solid"
#    print(linestyles)
    axs[1].contour(xlist,ylist,Z,colors='black',linewidths=linewidths,linestyles=linestyles,levels=levels)
    im=axs[1].imshow(global_wigner[i_min:i_max,:],origin='lower',interpolation='nearest',aspect='auto',extent=[-0.5*system_size,0.5*system_size,-p_max,p_max],cmap='seismic',vmin=-w_max,vmax=w_max)
    axs[1].set_xlabel('Position')
    axs[1].set_ylabel('Momentum')
    plt.title('Average Wigner function and contours of landscape+p**2/2',fontsize=10)
    cbaxes = fig.add_axes([0.87, 0.12, 0.03, 0.32])
    cb = plt.colorbar(im, cax = cbaxes)
#    fig.colorbar(im)
    fig.subplots_adjust(hspace=0.4)
    fig.subplots_adjust(bottom=0.12, left=0.18, right=0.85, top=0.85)
    plt.savefig('global_landscape_figure.png', dpi=300)
    plt.show()

    fig, axs = plt.subplots(2,sharex=True)
#    fig.subplots_adjust(hspace=0.5)
    axs[0].plot(initial_state.position,landscape,color='blue')
    axs[0].plot(initial_state.position,targeted_energy+multiplicative_factor_for_density*global_density,color='red')
    axs[0].plot(initial_state.position,H.disorder-2.0*H.tunneling,color='grey',linewidth=0.5)
    axs[0].text(-0.7*system_size,v_min,'Landscape',color='blue',rotation='vertical')
    axs[0].text(-0.75*system_size,v_min,'Density',color='red',rotation='vertical')
    axs[0].text(-0.65*system_size,v_min,'Potential',color='grey',rotation='vertical')
    my_string="Targeted energy = {:.4f}  V0 = {:.4f}".format(targeted_energy,V0)
    axs[0].text(-0.25*system_size,1.3*v_max-0.3*v_min,my_string)
    my_string="Number of energy levels = {}".format(number_of_energy_levels)
    axs[0].text(-0.25*system_size,1.1*v_max-0.1*v_min,my_string)
    axs[0].axis([-0.5*system_size,0.5*system_size,v_min,v_max])
    w_max = np.amax(global_wigner[i_min:i_max,:])
#    print('w_max=',w_max)
    xlist = initial_state.position
    ylist = tab_p[i_min:i_max]
    for j in range(xlist.size):
      Z[:,j] = tab_kinetic_energy[i_min:i_max]+H.disorder[j]-2.0*H.tunneling
    insert_position = np.argmax(raw_levels>targeted_energy)
#    print(raw_levels)
#    print(insert_position)
    levels = np.insert(raw_levels,insert_position,targeted_energy)
#    print(levels)
    linewidths=0.4*np.ones(number_of_levels+1)
    linewidths[insert_position]=1.0
    linestyles=["dotted" for j in range(number_of_levels+1)]
    linestyles[insert_position]="solid"
#    print(linestyles)
    axs[1].contour(xlist,ylist,Z,colors='black',linewidths=linewidths,linestyles=linestyles,levels=levels)
    im=axs[1].imshow(global_wigner[i_min:i_max,:],origin='lower',interpolation='nearest',aspect='auto',extent=[-0.5*system_size,0.5*system_size,-p_max,p_max],cmap='seismic',vmin=-w_max,vmax=w_max)
    axs[1].set_xlabel('Position')
    axs[1].set_ylabel('Momentum')
    plt.title('Average Wigner function and contours of potential+p**2/2',fontsize=10)
    cbaxes = fig.add_axes([0.87, 0.12, 0.03, 0.32])
    cb = plt.colorbar(im, cax = cbaxes)
#    fig.colorbar(im)
    fig.subplots_adjust(hspace=0.4)
    fig.subplots_adjust(bottom=0.12, left=0.18, right=0.85, top=0.85)
    plt.savefig('global_potential_figure.png', dpi=300)
    plt.show()


  t2=time.perf_counter()
  timing.TOTAL_TIME = t2-t1
#  if mpi_version:
#    timing.mpi_merge(comm)
  if rank==0:
#    print(tab_IPR_glob.shape)
    anderson.io.output_density('wfc.dat',position,np.real(tab_wfc),header_string+'Energy = '+str(energy)+'\n',print_type='wavefunction_eigenstate')

    final_time = time.asctime()
    print("Python script ended on: {}".format(final_time))
    print("Wallclock time {0:.3f} seconds".format(t2-t1))
    print()
    if mpi_version:
      print("MPI time             = {0:.3f}".format(timing.MPI_TIME))
    print("Total_CPU time       = {0:.3f}".format(timing.TOTAL_TIME))


