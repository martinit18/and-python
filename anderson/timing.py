#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:25:02 2020

@author: delande
"""

class Timing:
  def __init__(self):
    self.GPE_TIME=0.0
    self.CHE_TIME=0.0
    self.EXPECT_TIME=0.0
    self.ODE_TIME=0.0
    self.GPE_NOPS=0.0
    self.CHE_NOPS=0.0
    self.EXPECT_NOPS=0.0
    self.ODE_NOPS=0.0
    self.TOTAL_TIME=0.0
    self.TOTAL_NOPS=0.0
    self.N_SOLOUT=0
    self.MAX_CHE_ORDER=0
    self.DUMMY_TIME=0.0
    self.LYAPOUNOV_TIME=0.0
    self.LYAPOUNOV_NOPS=0
    self.MAX_NONLINEAR_PHASE=0.0
    self.MPI_TIME=0.0
    self.KPM_TIME=0.0
    self.KPM_NOPS=0.0
    self.SPECTRUM_TIME=0.0
    self.SPECTRUM_NOPS=0.0
    return

  def mpi_merge(self,comm):
    try:
      from mpi4py import MPI
    except ImportError:
      print("mpi4py is not found!")
      return
    self.GPE_TIME       = comm.reduce(self.GPE_TIME)
    self.GPE_NOPS       = comm.reduce(self.GPE_NOPS)
    self.CHE_TIME       = comm.reduce(self.CHE_TIME)
    self.CHE_NOPS       = comm.reduce(self.CHE_NOPS)
    self.EXPECT_TIME    = comm.reduce(self.EXPECT_TIME)
    self.EXPECT_NOPS    = comm.reduce(self.EXPECT_NOPS)
    self.ODE_TIME       = comm.reduce(self.ODE_TIME)
    self.ODE_NOPS       = comm.reduce(self.ODE_NOPS)
    self.TOTAL_TIME     = comm.reduce(self.TOTAL_TIME)
    self.TOTAL_NOPS     = comm.reduce(self.TOTAL_NOPS)
    self.N_SOLOUT       = comm.reduce(self.N_SOLOUT)
    self.MAX_CHE_ORDER  = comm.reduce(self.MAX_CHE_ORDER,op=MPI.MAX)
    self.DUMMY_TIME     = comm.reduce(self.DUMMY_TIME)
    self.LYAPOUNOV_TIME = comm.reduce(self.LYAPOUNOV_TIME)
    self.LYAPOUNOV_NOPS = comm.reduce(self.LYAPOUNOV_NOPS)
    self.MAX_NONLINEAR_PHASE = comm.reduce(self.MAX_NONLINEAR_PHASE,op=MPI.MAX)
    self.MPI_TIME       = comm.reduce(self.MPI_TIME)
    self.KPM_TIME       = comm.reduce(self.KPM_TIME)
    self.KPM_NOPS       = comm.reduce(self.KPM_NOPS)
    self.SPECTRUM_TIME  = comm.reduce(self.SPECTRUM_TIME)
    self.SPECTRUM_NOPS  = comm.reduce(self.SPECTRUM_NOPS)
    return
