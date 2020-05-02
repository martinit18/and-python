#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:00:38 2019

@author: delande
"""

from cffi import FFI
ffibuilder = FFI()

#ffibuilder.cdef("void elementary_clenshaw_step_real(const int dim_x, const char * restrict boundary_condition, const double * restrict wfc, const double * restrict psi, double * restrict psi_old, const double c_coef, const double one_or_two, const int add_real, const double tunneling, const double * restrict disorder);")
#ffibuilder.cdef("void elementary_clenshaw_step_complex(const int dim_x, const char * restrict boundary_condition, const double _Complex * restrict wfc, const double _Complex * restrict psi, double _Complex * restrict psi_old, const double c_coef, const double one_or_two, const int add_real, const double tunneling, const double * restrict disorder);")
ffibuilder.cdef("void chebyshev_clenshaw_real(const int dim_x, const int max_order, const char * restrict boundary_condition, double * wfc, double * psi, double * psi_old, const double tunneling, const double * disorder, const double *tab_coef, const double g_times_delta_t, const double e0_times_delta_t, double * nonlinear_phase);")
ffibuilder.cdef("void chebyshev_clenshaw_complex(const int dim_x, const int max_order, const char * restrict boundary_condition, double _Complex * wfc, double _Complex * psi, double _Complex * psi_old, const double tunneling, const double * disorder, const double *tab_coef, const double g_times_delta_t, const double e0_times_delta_t, double * nonlinear_phase);")


ffibuilder.set_source("_chebyshev",
r"""
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef __ICC
#include <mathimf.h>
#else
#include <math.h>
#endif
#include <complex.h>

void __inline elementary_clenshaw_step_real(const int dim_x, const char * restrict boundary_condition, const double * restrict wfc, const double * restrict psi, double * restrict psi_old, const double c_coef, const double one_or_two, const int add_real, const double tunneling, const double * restrict disorder)
{
  int i;
  if (add_real) {
    if (strcmp(boundary_condition,"periodic") == 0) {
      psi_old[0]=one_or_two*(disorder[0]*psi[0]-tunneling*(psi[1]+psi[dim_x-1]))+c_coef*wfc[0]-psi_old[0];
      psi_old[dim_x]=one_or_two*(disorder[0]*psi[dim_x]-tunneling*(psi[dim_x+1]+psi[2*dim_x-1]))+c_coef*wfc[dim_x]-psi_old[dim_x];
      psi_old[dim_x-1]=one_or_two*(disorder[dim_x-1]*psi[dim_x-1]-tunneling*(psi[0]+psi[dim_x-2]))+c_coef*wfc[dim_x-1]-psi_old[dim_x-1] ;
      psi_old[2*dim_x-1]=one_or_two*(disorder[dim_x-1]*psi[2*dim_x-1]-tunneling*(psi[dim_x]+psi[2*dim_x-2]))+c_coef*wfc[2*dim_x-1]-psi_old[2*dim_x-1];
    } else {
      psi_old[0]=one_or_two*(disorder[0]*psi[0]-tunneling*psi[1])+c_coef*wfc[0]-psi_old[0];
      psi_old[dim_x]=one_or_two*(disorder[0]*psi[dim_x]-tunneling*psi[dim_x+1])+c_coef*wfc[dim_x]-psi_old[dim_x];
      psi_old[dim_x-1]=one_or_two*(disorder[dim_x-1]*psi[dim_x-1]-tunneling*psi[dim_x-2])+c_coef*wfc[dim_x-1]-psi_old[dim_x-1] ;
      psi_old[2*dim_x-1]=one_or_two*(disorder[dim_x-1]*psi[2*dim_x-1]-tunneling*psi[2*dim_x-2])+c_coef*wfc[2*dim_x-1]-psi_old[2*dim_x-1];
    }
#ifdef __clang__
#else
#pragma GCC ivdep
#ifdef __ICC
#pragma distribute_point
#endif
#endif
    for (i=1;i<dim_x-1;i++) {
      psi_old[i]=one_or_two*(disorder[i]*psi[i]-tunneling*(psi[i+1]+psi[i-1]))+c_coef*wfc[i]-psi_old[i];
    }
#ifdef __clang__
#else
#pragma GCC ivdep
#ifdef __ICC
#pragma distribute_point
#endif
#endif
    for (i=dim_x+1;i<2*dim_x-1;i++) {
      psi_old[i]=one_or_two*(disorder[i-dim_x]*psi[i]-tunneling*(psi[i+1]+psi[i-1]))+c_coef*wfc[i]-psi_old[i];
    }
  } else {
    if (strcmp(boundary_condition,"periodic") == 0) {
      psi_old[0]=one_or_two*(disorder[0]*psi[0]-tunneling*(psi[1]+psi[dim_x-1]))-c_coef*wfc[dim_x]-psi_old[0];
      psi_old[dim_x]=one_or_two*(disorder[0]*psi[dim_x]-tunneling*(psi[dim_x+1]+psi[2*dim_x-1]))+c_coef*wfc[0]-psi_old[dim_x];
      psi_old[dim_x-1]=one_or_two*(disorder[dim_x-1]*psi[dim_x-1]-tunneling*(psi[0]+psi[dim_x-2]))-c_coef*wfc[2*dim_x-1]-psi_old[dim_x-1] ;
      psi_old[2*dim_x-1]=one_or_two*(disorder[dim_x-1]*psi[2*dim_x-1]-tunneling*(psi[dim_x]+psi[2*dim_x-2]))+c_coef*wfc[dim_x-1]-psi_old[2*dim_x-1];
    } else {
      psi_old[0]=one_or_two*(disorder[0]*psi[0]-tunneling*psi[1])-c_coef*wfc[dim_x]-psi_old[0];
      psi_old[dim_x]=one_or_two*(disorder[0]*psi[dim_x]-tunneling*psi[dim_x+1])+c_coef*wfc[0]-psi_old[dim_x];
      psi_old[dim_x-1]=one_or_two*(disorder[dim_x-1]*psi[dim_x-1]-tunneling*psi[dim_x-2])-c_coef*wfc[2*dim_x-1]-psi_old[dim_x-1] ;
      psi_old[2*dim_x-1]=one_or_two*(disorder[dim_x-1]*psi[2*dim_x-1]-tunneling*psi[2*dim_x-2])+c_coef*wfc[dim_x-1]-psi_old[2*dim_x-1];
    }
#ifdef __clang__
#else
#pragma GCC ivdep
#ifdef __ICC
#pragma distribute_point
#endif
#endif
    for (i=1;i<dim_x-1;i++) {
      psi_old[i]=one_or_two*(disorder[i]*psi[i]-tunneling*(psi[i+1]+psi[i-1]))-c_coef*wfc[i+dim_x]-psi_old[i];
    }
#ifdef __clang__
#else
#pragma GCC ivdep
#ifdef __ICC
#pragma distribute_point
#endif
#endif
    for (i=dim_x+1;i<2*dim_x-1;i++) {
      psi_old[i]=one_or_two*(disorder[i-dim_x]*psi[i]-tunneling*(psi[i+1]+psi[i-1]))+c_coef*wfc[i-dim_x]-psi_old[i];
    }
  }
  return;
}

void __inline elementary_clenshaw_step_complex(const int dim_x, const char * restrict boundary_condition, const double complex * restrict wfc, const double complex * restrict psi, double complex * restrict psi_old, const double c_coef, const double one_or_two, const int add_real, const double tunneling, const double * restrict disorder)
{
  int i;
  if (add_real) {
    if (strcmp(boundary_condition,"periodic") == 0) {
      psi_old[0]=one_or_two*(disorder[0]*psi[0]-tunneling*(psi[1]+psi[dim_x-1]))+c_coef*wfc[0]-psi_old[0];
      psi_old[dim_x-1]=one_or_two*(disorder[dim_x-1]*psi[dim_x-1]-tunneling*(psi[0]+psi[dim_x-2]))+c_coef*wfc[dim_x-1]-psi_old[dim_x-1];
    } else {
      psi_old[0]=one_or_two*(disorder[0]*psi[0]-tunneling*psi[1])+c_coef*wfc[0]-psi_old[0];
      psi_old[dim_x-1]=one_or_two*(disorder[dim_x-1]*psi[dim_x-1]-tunneling*psi[dim_x-2])+c_coef*wfc[dim_x-1]-psi_old[dim_x-1];
    }
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
    for (i=1;i<dim_x-1;i++) {
      psi_old[i]=one_or_two*(disorder[i]*psi[i]-tunneling*(psi[i+1]+psi[i-1]))+c_coef*wfc[i]-psi_old[i];
    }
  } else {
    if (strcmp(boundary_condition,"periodic") == 0) {
      psi_old[0]=one_or_two*(disorder[0]*psi[0]-tunneling*(psi[1]+psi[dim_x-1]))+I*c_coef*wfc[0]-psi_old[0];
      psi_old[dim_x-1]=one_or_two*(disorder[dim_x-1]*psi[dim_x-1]-tunneling*(psi[0]+psi[dim_x-2]))+I*c_coef*wfc[dim_x-1]-psi_old[dim_x-1] ;
    } else {
      psi_old[0]=one_or_two*(disorder[0]*psi[0]-tunneling*psi[1])+I*c_coef*wfc[0]-psi_old[0];
      psi_old[dim_x-1]=one_or_two*(disorder[dim_x-1]*psi[dim_x-1]-tunneling*psi[dim_x-2])+I*c_coef*wfc[dim_x-1]-psi_old[dim_x-1] ;
    }
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
   for (i=1;i<dim_x-1;i++) {
      psi_old[i]=one_or_two*(disorder[i]*psi[i]-tunneling*(psi[i+1]+psi[i-1]))+I*c_coef*wfc[i]-psi_old[i];
    }
  }
  return;
}

void chebyshev_clenshaw_real(const int dim_x, const int max_order,  const char * restrict boundary_condition, double * restrict wfc, double * restrict psi, double * restrict psi_old, const double tunneling, const double * restrict disorder, const double * restrict tab_coef, const double g_times_delta_t, const double e0_times_delta_t, double *nonlinear_phase)
{
  int i, order;
  double argument;
  double complex exp_i_argument;
  double phase;
  double cos_phase;
  double sin_phase;
  for (i=0;i<2*dim_x;i++) {
    psi[i] = tab_coef[max_order] * wfc[i];
  }
  elementary_clenshaw_step_real(dim_x, boundary_condition, wfc, psi, psi_old, tab_coef[max_order-1], 2.0, 0, tunneling, disorder);
// WARNING: max_order MUST be an even number, otherwise disaster guaranteed
  for (order=max_order-2;order>1;order-=2) {
//    printf("order = %d %d %lf %lf\n",order,(order+1)%2,tab_coef[order],tunneling);
    elementary_clenshaw_step_real(dim_x, boundary_condition, wfc, psi_old, psi, tab_coef[order], 2.0, 1, tunneling, disorder);
    elementary_clenshaw_step_real(dim_x, boundary_condition, wfc, psi, psi_old, tab_coef[order-1], 2.0, 0, tunneling, disorder);
  }
  elementary_clenshaw_step_real(dim_x, boundary_condition, wfc, psi_old, psi, tab_coef[0], 1.0, 1, tunneling, disorder);
// apply nonlinear shift
  if (g_times_delta_t==0.0) {
    cos_phase=cos(e0_times_delta_t);
    sin_phase=sin(e0_times_delta_t);
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
    for (i=0;i<dim_x;i++) {
      wfc[i] = psi[i]*cos_phase+psi[i+dim_x]*sin_phase;
      wfc[i+dim_x] = psi[i+dim_x]*cos_phase-psi[i]*sin_phase;
   }
  } else {
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
    for (i=0;i<dim_x;i++) {
      phase = g_times_delta_t*(psi[i]*psi[i]+psi[i+dim_x]*psi[i+dim_x]);
      *nonlinear_phase = (*nonlinear_phase > phase) ? *nonlinear_phase : phase;
      argument =  e0_times_delta_t+phase;
      wfc[i] = psi[i]*cos(argument)+psi[i+dim_x]*sin(argument);
      wfc[i+dim_x] = psi[i+dim_x]*cos(argument)-psi[i]*sin(argument);
    }
  }
  return;
}

void chebyshev_clenshaw_complex(const int dim_x, const int max_order, const char * restrict boundary_condition, double complex * restrict wfc, double  complex * restrict psi, double complex * restrict psi_old, const double tunneling, const double * restrict disorder, const double * restrict tab_coef, const double g_times_delta_t, const double e0_times_delta_t, double *nonlinear_phase)
{
  int i, order;
  double argument;
  double complex complex_argument;
  double phase;
//  printf("max order = %d\n",max_order);
  for (i=0;i<dim_x;i++) {
    psi[i] = tab_coef[max_order] * wfc[i];
  }
  elementary_clenshaw_step_complex(dim_x, boundary_condition, wfc, psi, psi_old, tab_coef[max_order-1], 2.0, 0, tunneling, disorder);
// WARNING: max_order MUST be an even number, otherwise disaster guaranteed
  for (order=max_order-2;order>1;order-=2) {
//    printf("order = %d %d %lf %lf\n",order,(order+1)%2,tab_coef[order],tunneling);
    elementary_clenshaw_step_complex(dim_x, boundary_condition, wfc, psi_old, psi, tab_coef[order], 2.0, 1, tunneling, disorder);
    elementary_clenshaw_step_complex(dim_x, boundary_condition, wfc, psi, psi_old, tab_coef[order-1], 2.0, 0, tunneling, disorder);
  }
  elementary_clenshaw_step_complex(dim_x, boundary_condition, wfc, psi_old, psi, tab_coef[0], 1.0, 1, tunneling, disorder);
// apply nonlinear shift
  if (g_times_delta_t==0.0) {
    complex_argument=cos(e0_times_delta_t)-I*sin(e0_times_delta_t);
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
    for (i=0;i<dim_x;i++) {
      wfc[i] = psi[i]*complex_argument;
    }
  } else {
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
    for (i=0;i<dim_x;i++) {
      phase = g_times_delta_t*(creal(psi[i])*creal(psi[i])+cimag(psi[i])*cimag(psi[i]));
      *nonlinear_phase = (*nonlinear_phase > phase) ? *nonlinear_phase : phase;
      argument =  e0_times_delta_t+phase;
      wfc[i] = psi[i]*(cos(argument)-I*sin(argument));
    }
  }
  return;
}
""")

if __name__ == "__main__":
  ffibuilder.compile(verbose=True)
