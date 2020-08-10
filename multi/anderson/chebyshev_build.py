#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:00:38 2019

@author: delande
"""

from cffi import FFI
ffibuilder = FFI()


ffibuilder.cdef("void elementary_clenshaw_step_real_real_coef_1d(const int dim_x, const int boundary_condition, const double * restrict wfc, const double * restrict psi, double * restrict psi_old, const double * restrict disorder, const double c_coef, const double c1, const double c2, const double c3);")

ffibuilder.cdef("void elementary_clenshaw_step_real_imag_coef_1d(const int dim_x, const int boundary_condition, const double * restrict wfc, const double * restrict psi, double * restrict psi_old, const double * restrict disorder, const double c_coef, const double c1, const double c2, const double c3);")

ffibuilder.cdef("void chebyshev_real(const int dimension, const int * restrict tab_dim, const int max_order,  const int * restrict tab_boundary_condition, double * restrict wfc, double * restrict psi, double * restrict psi_old,  const double * restrict disorder, const double * restrict tab_coef, const double * tab_tunneling, const double two_over_delta_e, const double two_e0_over_delta_e, const double g_times_delta_t, const double e0_times_delta_t, double * restrict nonlinear_phase);")

ffibuilder.cdef("static void elementary_clenshaw_step_complex_real_coef_1d(const int dim_x, const int boundary_condition, const double _Complex * restrict wfc, const double _Complex * restrict psi, double _Complex * restrict psi_old, const double * restrict disorder, const double c_coef, const double c1, const double c2, const double c3);")

ffibuilder.cdef("static void elementary_clenshaw_step_complex_imag_coef_1d(const int dim_x, const int boundary_condition, const double _Complex * restrict wfc, const double _Complex * restrict psi, double _Complex * restrict psi_old, const double * restrict disorder, const double c_coef, const double c1, const double c2, const double c3);")

ffibuilder.cdef("void chebyshev_complex(const int dimension, const int * restrict tab_dim, const int max_order,  const int * restrict tab_boundary_condition, double _Complex * restrict wfc, double _Complex * restrict psi, double _Complex * restrict psi_old,  const double * restrict disorder, const double * restrict tab_coef, const double * tab_tunneling, const double two_over_delta_e, const double two_e0_over_delta_e, const double g_times_delta_t, const double e0_times_delta_t, double * restrict nonlinear_phase);")

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

// DO NOT TRY to inline the static routines as it badly fails with the Intel 20 compiler for unknown reason

static void elementary_clenshaw_step_real_real_coef_1d(const int dim_x, const int boundary_condition, const double * restrict wfc, const double * restrict psi, double * restrict psi_old, const double * restrict disorder, const double c_coef, const double c1, const double c2, const double c3)
{
  int i;
// boundary_condition=1 is periodic
// boundary_condition=0 is open
  if (boundary_condition) {
//  if (strcmp(boundary_condition,"periodic") == 0) {
    psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]+psi[dim_x-1]) + c_coef*wfc[0] - psi_old[0];
    psi_old[dim_x] = (c1*disorder[0]-c2)*psi[dim_x] - c3*(psi[dim_x+1]+psi[2*dim_x-1]) + c_coef*wfc[dim_x] - psi_old[dim_x];
    psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[0]+psi[dim_x-2]) + c_coef*wfc[dim_x-1] - psi_old[dim_x-1] ;
    psi_old[2*dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[2*dim_x-1] -c3*(psi[dim_x]+psi[2*dim_x-2]) + c_coef*wfc[2*dim_x-1] - psi_old[2*dim_x-1];
  } else {
    psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]) + c_coef*wfc[0] - psi_old[0];
    psi_old[dim_x] = (c1*disorder[0]-c2)*psi[dim_x] - c3*(psi[dim_x+1]) + c_coef*wfc[dim_x] - psi_old[dim_x];
    psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[dim_x-2]) + c_coef*wfc[dim_x-1] - psi_old[dim_x-1] ;
    psi_old[2*dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[2*dim_x-1] -c3*(psi[2*dim_x-2]) + c_coef*wfc[2*dim_x-1] - psi_old[2*dim_x-1];
  }
#ifdef __clang__
#else
#pragma GCC ivdep
#ifdef __ICC
#pragma distribute_point
#endif
#endif
  for (i=1;i<dim_x-1;i++) {
    psi_old[i]=(c1*disorder[i]-c2)*psi[i]-c3*(psi[i+1]+psi[i-1])+c_coef*wfc[i]-psi_old[i];
  }
#ifdef __clang__
#else
#pragma GCC ivdep
#ifdef __ICC
#pragma distribute_point
#endif
#endif
  for (i=dim_x+1;i<2*dim_x-1;i++) {
    psi_old[i]=(c1*disorder[i-dim_x]-c2)*psi[i]-c3*(psi[i+1]+psi[i-1])+c_coef*wfc[i]-psi_old[i];
  }
  return;
}

static void elementary_clenshaw_step_real_imag_coef_1d(const int dim_x, const int boundary_condition, const double * restrict wfc, const double * restrict psi, double * restrict psi_old, const double * restrict disorder, const double c_coef, const double c1, const double c2, const double c3)
{
  int i;
// boundary_condition=1 is periodic
// boundary_condition=0 is open
  if (boundary_condition) {
//  if (strcmp(boundary_condition,"periodic") == 0) {
    psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]+psi[dim_x-1]) - c_coef*wfc[dim_x] - psi_old[0];
    psi_old[dim_x] = (c1*disorder[0]-c2)*psi[dim_x] - c3*(psi[dim_x+1]+psi[2*dim_x-1]) + c_coef*wfc[0] - psi_old[dim_x];
    psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[0]+psi[dim_x-2]) - c_coef*wfc[2*dim_x-1] - psi_old[dim_x-1] ;
    psi_old[2*dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[2*dim_x-1] -c3*(psi[dim_x]+psi[2*dim_x-2]) + c_coef*wfc[dim_x-1] - psi_old[2*dim_x-1];
  } else {
    psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]) - c_coef*wfc[dim_x] - psi_old[0];
    psi_old[dim_x] = (c1*disorder[0]-c2)*psi[dim_x] - c3*(psi[dim_x+1]) + c_coef*wfc[0] - psi_old[dim_x];
    psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[dim_x-2]) - c_coef*wfc[2*dim_x-1] - psi_old[dim_x-1] ;
    psi_old[2*dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[2*dim_x-1] -c3*(psi[2*dim_x-2]) + c_coef*wfc[dim_x-1] - psi_old[2*dim_x-1];
  }
#ifdef __clang__
#else
#pragma GCC ivdep
#ifdef __ICC
#pragma distribute_point
#endif
#endif
  for (i=1;i<dim_x-1;i++) {
    psi_old[i]=(c1*disorder[i]-c2)*psi[i]-c3*(psi[i+1]+psi[i-1])-c_coef*wfc[i+dim_x]-psi_old[i];
  }
#ifdef __clang__
#else
#pragma GCC ivdep
#ifdef __ICC
#pragma distribute_point
#endif
#endif
  for (i=dim_x+1;i<2*dim_x-1;i++) {
    psi_old[i]=(c1*disorder[i-dim_x]-c2)*psi[i]-c3*(psi[i+1]+psi[i-1])+c_coef*wfc[i-dim_x]-psi_old[i];
  }
  return;
}

void chebyshev_real(const int dimension, const int * restrict tab_dim, const int max_order,  const int * restrict tab_boundary_condition, double * restrict wfc, double * restrict psi, double * restrict psi_old,  const double * restrict disorder, const double * restrict tab_coef, const double * tab_tunneling, const double two_over_delta_e, const double two_e0_over_delta_e, const double g_times_delta_t, const double e0_times_delta_t, double * restrict nonlinear_phase)
{
  int i, order;
  double argument;
//  double complex exp_i_argument;
  double phase;
  double cos_phase;
  double sin_phase;
  double c1, c2, c3;
  int ntot=1;
  for (i=0;i<dimension;i++) {
    ntot *= tab_dim[i];
  }
  for (i=0;i<2*ntot;i++) {
    psi[i] = tab_coef[max_order] * wfc[i];
  }
  c1 = 2.0*two_over_delta_e;
  c2 = 2.0*two_e0_over_delta_e;
  if (dimension==1) {
    int dim_x = tab_dim[0];
    int boundary_condition = tab_boundary_condition[0];
    c3 = 2.0*tab_tunneling[0]*two_over_delta_e;
    elementary_clenshaw_step_real_imag_coef_1d(dim_x, boundary_condition, wfc, psi, psi_old, disorder, tab_coef[max_order-1], c1, c2, c3);
// WARNING: max_order MUST be an even number, otherwise disaster guaranteed
    for (order=max_order-2;order>1;order-=2) {
      elementary_clenshaw_step_real_real_coef_1d(dim_x, boundary_condition, wfc, psi_old, psi, disorder, tab_coef[order], c1, c2, c3);
      elementary_clenshaw_step_real_imag_coef_1d(dim_x, boundary_condition, wfc, psi, psi_old, disorder, tab_coef[order-1], c1, c2, c3);
    }
    c1 = two_over_delta_e;
    c2 = two_e0_over_delta_e;
    c3 = tab_tunneling[0]*two_over_delta_e;
    elementary_clenshaw_step_real_real_coef_1d(dim_x, boundary_condition, wfc, psi_old, psi, disorder, tab_coef[0], c1, c2, c3);
  }
// apply nonlinear shift
  if (g_times_delta_t==0.0) {
    cos_phase=cos(e0_times_delta_t);
    sin_phase=sin(e0_times_delta_t);
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
    for (i=0;i<ntot;i++) {
      wfc[i] = psi[i]*cos_phase+psi[i+ntot]*sin_phase;
      wfc[i+ntot] = psi[i+ntot]*cos_phase-psi[i]*sin_phase;
   }
  } else {
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
    for (i=0;i<ntot;i++) {
      phase = g_times_delta_t*(psi[i]*psi[i]+psi[i+ntot]*psi[i+ntot]);
      *nonlinear_phase = (*nonlinear_phase > fabs(phase)) ? *nonlinear_phase : fabs(phase);
      argument =  e0_times_delta_t+phase;
      wfc[i] = psi[i]*cos(argument)+psi[i+ntot]*sin(argument);
      wfc[i+ntot] = psi[i+ntot]*cos(argument)-psi[i]*sin(argument);
    }
  }
//  printf("done\n");
  return;
}

static void elementary_clenshaw_step_complex_real_coef_1d(const int dim_x, const int boundary_condition, const double complex * restrict wfc, const double complex * restrict psi, double complex * restrict psi_old, const double * restrict disorder, const double c_coef, const double c1, const double c2, const double c3)
{
  int i;
//  printf("c_coef %f\n",c_coef);
// boundary_condition=1 is periodic
// boundary_condition=0 is open
  if (boundary_condition) {
//  if (strcmp(boundary_condition,"periodic") == 0) {
    psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]+psi[dim_x-1]) + c_coef*wfc[0] - psi_old[0];
    psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[0]+psi[dim_x-2]) + c_coef*wfc[dim_x-1] - psi_old[dim_x-1] ;
  } else {
    psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]) + c_coef*wfc[0] - psi_old[0];
    psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[dim_x-2]) + c_coef*wfc[dim_x-1] - psi_old[dim_x-1] ;
  }
#ifdef __clang__
#else
#pragma GCC ivdep
#ifdef __ICC
#pragma distribute_point
#endif
#endif
  for (i=1;i<dim_x-1;i++) {
//    printf("i %d\n",i);
    psi_old[i]=(c1*disorder[i]-c2)*psi[i]-c3*(psi[i+1]+psi[i-1])+c_coef*wfc[i]-psi_old[i];
  }
//  printf("%f %f %f %f %f %f\n",psi[0],psi_old[0],wfc[0]);
  return;
}

static void elementary_clenshaw_step_complex_imag_coef_1d(const int dim_x, const int boundary_condition, const double complex * restrict wfc, const double complex * restrict psi, double complex * restrict psi_old, const double * restrict disorder, const double c_coef, const double c1, const double c2, const double c3)
{
  int i;
//  printf("c_coef %f\n",c_coef);
// boundary_condition=1 is periodic
// boundary_condition=0 is open
  if (boundary_condition) {
//  if (strcmp(boundary_condition,"periodic") == 0) {
    psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]+psi[dim_x-1]) + I*c_coef*wfc[0] - psi_old[0];
    psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[0]+psi[dim_x-2]) + I*c_coef*wfc[dim_x-1] - psi_old[dim_x-1] ;
  } else {
    psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]) +  I*c_coef*wfc[0] - psi_old[0];
    psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[dim_x-2]) + I*c_coef*wfc[dim_x-1] - psi_old[dim_x-1] ;
  }
#ifdef __clang__
#else
#pragma GCC ivdep
#ifdef __ICC
#pragma distribute_point
#endif
#endif
  for (i=1;i<dim_x-1;i++) {
    psi_old[i]=(c1*disorder[i]-c2)*psi[i]-c3*(psi[i+1]+psi[i-1])+I*c_coef*wfc[i]-psi_old[i];
  }
//  printf("%f %f %f %f %f %f\n",psi[0],psi_old[0],wfc[0]);
  return;
}


void chebyshev_complex(const int dimension, const int * restrict tab_dim, const int max_order,  const int * restrict tab_boundary_condition, double complex * restrict wfc, double complex * restrict psi, double complex * restrict psi_old,  const double * restrict disorder, const double * restrict tab_coef, const double * tab_tunneling, const double two_over_delta_e, const double two_e0_over_delta_e, const double g_times_delta_t, const double e0_times_delta_t, double * restrict nonlinear_phase)
{
  int i, order;
  double argument;
  double complex complex_argument;
  double phase;
  double c1, c2, c3;
  int ntot=1;
  for (i=0;i<dimension;i++) {
    ntot *= tab_dim[i];
  }
  for (i=0;i<ntot;i++) {
    psi[i] = tab_coef[max_order] * wfc[i];
  }
  c1 = 2.0*two_over_delta_e;
  c2 = 2.0*two_e0_over_delta_e;
  if (dimension==1) {
    int dim_x = tab_dim[0];
    int boundary_condition = tab_boundary_condition[0];
    c3 = 2.0*tab_tunneling[0]*two_over_delta_e;
    elementary_clenshaw_step_complex_imag_coef_1d(dim_x, boundary_condition, wfc, psi, psi_old, disorder, tab_coef[max_order-1], c1, c2, c3);
// WARNING: max_order MUST be an even number, otherwise disaster guaranteed
    for (order=max_order-2;order>1;order-=2) {
//      printf("order %d %f\n",order,tab_coef[order]);
      elementary_clenshaw_step_complex_real_coef_1d(dim_x, boundary_condition, wfc, psi_old, psi, disorder, tab_coef[order], c1, c2, c3);
//      printf("order %d %f\n",order-1,tab_coef[order-1]);
      elementary_clenshaw_step_complex_imag_coef_1d(dim_x, boundary_condition, wfc, psi, psi_old, disorder, tab_coef[order-1], c1, c2, c3);
    }
    c1 = two_over_delta_e;
    c2 = two_e0_over_delta_e;
    c3 = tab_tunneling[0]*two_over_delta_e;
    elementary_clenshaw_step_complex_real_coef_1d(dim_x, boundary_condition, wfc, psi_old, psi, disorder, tab_coef[0], c1, c2, c3);
  }
//  printf("%f %f %f %f %f %f\n",psi[100],psi_old[100],wfc[100]);
// apply nonlinear shift
  if (g_times_delta_t==0.0) {
    complex_argument=cos(e0_times_delta_t)-I*sin(e0_times_delta_t);
//    complex_argument = 1.0;
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
    for (i=0;i<ntot;i++) {
      wfc[i] = psi[i]*complex_argument;
    }
  } else {
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
    for (i=0;i<ntot;i++) {
      phase = g_times_delta_t*(creal(psi[i])*creal(psi[i])+cimag(psi[i])*cimag(psi[i]));
      *nonlinear_phase = (*nonlinear_phase > fabs(phase)) ? *nonlinear_phase : fabs(phase);
      argument =  e0_times_delta_t+phase;
      wfc[i] = psi[i]*(cos(argument)-I*sin(argument));
    }
  }
//  printf("done\n");
//  printf("wfc %f %f %f %f %f %f\n",wfc[0],wfc[1],wfc[ntot-1]);
  return;
}


/*
inline static void elementary_clenshaw_step_complex_1d(const int dim_x, const char * restrict boundary_condition, const double complex * restrict wfc, const double complex * restrict psi, double complex * restrict psi_old, const double c_coef, const double one_or_two, const int add_real, const double tunneling, const double * restrict disorder)
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

void chebyshev_clenshaw_complex_1d(const int dim_x, const int max_order, const char * restrict boundary_condition, double complex * restrict wfc, double  complex * restrict psi, double complex * restrict psi_old, const double tunneling, const double * restrict disorder, const double * restrict tab_coef, const double g_times_delta_t, const double e0_times_delta_t, double * restrict nonlinear_phase)
{
  int i, order;
  double argument;
  double complex complex_argument;
  double phase;
//  printf("max order = %d\n",max_order);
  for (i=0;i<dim_x;i++) {
    psi[i] = tab_coef[max_order] * wfc[i];
  }
  elementary_clenshaw_step_complex_1d(dim_x, boundary_condition, wfc, psi, psi_old, tab_coef[max_order-1], 2.0, 0, tunneling, disorder);
// WARNING: max_order MUST be an even number, otherwise disaster guaranteed
  for (order=max_order-2;order>1;order-=2) {
//    printf("order = %d %d %lf %lf\n",order,(order+1)%2,tab_coef[order],tunneling);
    elementary_clenshaw_step_complex_1d(dim_x, boundary_condition, wfc, psi_old, psi, tab_coef[order], 2.0, 1, tunneling, disorder);
    elementary_clenshaw_step_complex_1d(dim_x, boundary_condition, wfc, psi, psi_old, tab_coef[order-1], 2.0, 0, tunneling, disorder);
  }
  elementary_clenshaw_step_complex_1d(dim_x, boundary_condition, wfc, psi_old, psi, tab_coef[0], 1.0, 1, tunneling, disorder);
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
      *nonlinear_phase = (*nonlinear_phase > fabs(phase)) ? *nonlinear_phase : fabs(phase);
      argument =  e0_times_delta_t+phase;
      wfc[i] = psi[i]*(cos(argument)-I*sin(argument));
    }
  }
  return;
}
*/
""")

if __name__ == "__main__":
  ffibuilder.compile(verbose=True)
