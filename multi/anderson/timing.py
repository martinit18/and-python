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
    self.TOTAL_TIME=0.0
    self.NUMBER_OF_OPS=0.0
    self.N_SOLOUT=0
    self.MAX_CHE_ORDER=0
    self.DUMMY_TIME=0.0
    self.LYAPOUNOV_TIME=0.0
    self.LYAPOUNOV_NOPS=0
    self.MAX_NONLINEAR_PHASE=0.0
    self.MPI_TIME=0.0
    return

  def mpi_merge(self,comm):
    try:
      from mpi4py import MPI
    except ImportError:
      print("mpi4py is not found!")
      return
    self.GPE_TIME       = comm.reduce(self.GPE_TIME)
    self.CHE_TIME       = comm.reduce(self.CHE_TIME)
    self.EXPECT_TIME    = comm.reduce(self.EXPECT_TIME)
    self.ODE_TIME       = comm.reduce(self.ODE_TIME)
    self.TOTAL_TIME     = comm.reduce(self.TOTAL_TIME)
    self.NUMBER_OF_OPS  = comm.reduce(self.NUMBER_OF_OPS)
    self.N_SOLOUT       = comm.reduce(self.N_SOLOUT)
    self.MAX_CHE_ORDER  = comm.reduce(self.MAX_CHE_ORDER,op=MPI.MAX)
    self.DUMMY_TIME     = comm.reduce(self.DUMMY_TIME)
    self.LYAPOUNOV_TIME = comm.reduce(self.LYAPOUNOV_TIME)
    self.LYAPOUNOV_NOPS = comm.reduce(self.LYAPOUNOV_NOPS)
    self.MAX_NONLINEAR_PHASE = comm.reduce(self.MAX_NONLINEAR_PHASE,op=MPI.MAX)
    self.MPI_TIME       = comm.reduce(self.MPI_TIME)
    return
