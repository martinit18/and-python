#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:00:38 2019

@author: delande
"""

from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef("double core_lyapounov_cffi(const int dim_x, const int loop_step, const double * disorder, const double energy, const double inv_tunneling);")

ffibuilder.set_source("_lyapounov",
r"""
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef __ICC
#include <mathimf.h>
#else
#include <math.h>
#endif

#define min(x,y) ((x>y) ? y : x)
    
double core_lyapounov_cffi(const int dim_x, const int loop_step, const double * disorder, const double energy, const double inv_tunneling)
{
  double gamma=0.0;	 
  int i, j, jmax;
  double psi_new, psi_cur=1.0, psi_old=0.0;
  for (i=0;i<dim_x;i+=loop_step) {
    jmax=min(i+loop_step,dim_x);
    for(j=i;j<jmax;j++) {
      psi_new=psi_cur*(inv_tunneling*(disorder[j]-energy))-psi_old;
      psi_old=psi_cur;
      psi_cur=psi_new;
    }  
    gamma+=log(fabs(psi_cur));
    psi_old/=psi_cur;
    psi_cur=1.0;
  }  
  gamma+=log(fabs(psi_cur));
//  printf("gamma = %lf\n",gamma);
  return(gamma);
}
""")

if __name__ == "__main__":
  ffibuilder.compile(verbose=True)
