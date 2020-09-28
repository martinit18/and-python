#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:00:38 2019

@author: delande
"""

from cffi import FFI
ffibuilder = FFI()


ffibuilder.cdef("static void elementary_clenshaw_step_real_1d(const int dim_x, const int boundary_condition, const double * restrict wfc, const double * restrict psi, double * restrict psi_old, const double * restrict disorder, const double c_coef, const int add_real, const double c1, const double c2, const double c3);")

ffibuilder.cdef("static void elementary_clenshaw_step_real_2d(const int dim_x, const int dim_y, const int b_x, const int b_y, const double * restrict wfc, const double * restrict psi, double * restrict psi_old, const double * restrict disorder, const double c_coef, const int add_real, const double c1, const double c2, const double c3_x, const double c_3y);")

ffibuilder.cdef("void chebyshev_real(const int dimension, const int * restrict tab_dim, const int max_order,  const int * restrict tab_boundary_condition, double * restrict wfc, double * restrict psi, double * restrict psi_old,  const double * restrict disorder, const double * restrict tab_coef, const double * tab_tunneling, const double two_over_delta_e, const double two_e0_over_delta_e, const double g_times_delta_t, const double e0_times_delta_t, double * restrict nonlinear_phase);")

ffibuilder.cdef("static void elementary_clenshaw_step_complex_1d(const int dim_x, const int boundary_condition, const double _Complex * restrict wfc, const double _Complex * restrict psi, double _Complex * restrict psi_old, const double * restrict disorder, const double c_coef, const int add_real, const double c1, const double c2, const double c3);")

ffibuilder.cdef("static void elementary_clenshaw_step_complex_2d(const int dim_x, const int dim_y, const int b_x, const int b_y, const double _Complex * restrict wfc, const double _Complex * restrict psi, double _Complex * restrict psi_old, const double * restrict disorder, const double c_coef, const int add_real, const double c1, const double c2, const double c3_x, const double c_3y);")

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

#define TIMING
#define SIZE 64

uint64_t timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
{
  return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) -
           ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
}
// DO NOT TRY to inline the static routines as it badly fails with the Intel 20 compiler for unknown reason

static void elementary_clenshaw_step_real_1d(const int dim_x, const int boundary_condition, const double * restrict wfc, const double * restrict psi, double * restrict psi_old, const double * restrict disorder, const double c_coef, const int add_real, const double c1, const double c2, const double c3)
{
  int i;
// boundary_condition=1 is periodic
// boundary_condition=0 is open
  if (boundary_condition) {
    if (add_real) {
      psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]+psi[dim_x-1]) + c_coef*wfc[0] - psi_old[0];
      psi_old[dim_x] = (c1*disorder[0]-c2)*psi[dim_x] - c3*(psi[dim_x+1]+psi[2*dim_x-1]) + c_coef*wfc[dim_x] - psi_old[dim_x];
      psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[0]+psi[dim_x-2]) + c_coef*wfc[dim_x-1] - psi_old[dim_x-1] ;
      psi_old[2*dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[2*dim_x-1] -c3*(psi[dim_x]+psi[2*dim_x-2]) + c_coef*wfc[2*dim_x-1] - psi_old[2*dim_x-1];
    } else {
      psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]+psi[dim_x-1]) - c_coef*wfc[dim_x] - psi_old[0];
      psi_old[dim_x] = (c1*disorder[0]-c2)*psi[dim_x] - c3*(psi[dim_x+1]+psi[2*dim_x-1]) + c_coef*wfc[0] - psi_old[dim_x];
      psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[0]+psi[dim_x-2]) - c_coef*wfc[2*dim_x-1] - psi_old[dim_x-1] ;
      psi_old[2*dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[2*dim_x-1] -c3*(psi[dim_x]+psi[2*dim_x-2]) + c_coef*wfc[dim_x-1] - psi_old[2*dim_x-1];
    }
  } else {
    if (add_real) {
      psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]) + c_coef*wfc[0] - psi_old[0];
      psi_old[dim_x] = (c1*disorder[0]-c2)*psi[dim_x] - c3*(psi[dim_x+1]) + c_coef*wfc[dim_x] - psi_old[dim_x];
      psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[dim_x-2]) + c_coef*wfc[dim_x-1] - psi_old[dim_x-1] ;
      psi_old[2*dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[2*dim_x-1] -c3*(psi[2*dim_x-2]) + c_coef*wfc[2*dim_x-1] - psi_old[2*dim_x-1];
    } else {
      psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]) - c_coef*wfc[dim_x] - psi_old[0];
      psi_old[dim_x] = (c1*disorder[0]-c2)*psi[dim_x] - c3*(psi[dim_x+1]) + c_coef*wfc[0] - psi_old[dim_x];
      psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[dim_x-2]) - c_coef*wfc[2*dim_x-1] - psi_old[dim_x-1] ;
      psi_old[2*dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[2*dim_x-1] -c3*(psi[2*dim_x-2]) + c_coef*wfc[dim_x-1] - psi_old[2*dim_x-1];
    }
  }
  if (add_real) {
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
  } else {
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
  }
  return;
}

static void elementary_clenshaw_step_real_2d(const int dim_x, const int dim_y, const int b_x, const int b_y, const double * restrict wfc, const double * restrict psi, double * restrict psi_old, const double * restrict disorder, const double c_coef, const int add_real, const double c1, const double c2, const double c3_x, const double c3_y)
{
  int i,j,i_low;
  int ntot = dim_x*dim_y;
  double *p_old,*p_current,*p_new,*p_temp;
#ifdef __ICC
  p_old = (double *) _mm_malloc ((2*dim_y+4)*sizeof(double),SIZE);
  p_current = (double *) _mm_malloc ((2*dim_y+4)*sizeof(double),SIZE);
  p_new = (double *) _mm_malloc ((2*dim_y+4)*sizeof(double),SIZE);
#else
  p_old = (double *) malloc ((2*dim_y+4)*sizeof(double));
  p_current = (double *) malloc ((2*dim_y+4)*sizeof(double));
  p_new = (double *) malloc ((2*dim_y+4)*sizeof(double));
#endif

// If periodic boundary conditions along x, initialize p_current to the last row, otherwise 0
  if (b_x) {
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
    for (i=0;i<dim_y;i++) {
      p_current[i+1] = psi[i+(dim_x-1)*dim_y];
    }
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
    for (i=0;i<dim_y;i++) {
      p_current[dim_y+i+3] = psi[ntot+i+(dim_x-1)*dim_y];
    }
  } else {
    for (i=0;i<dim_y+4;i++) {
      p_current[i] = 0.0;
    }
 }
// Initialize the next row, which will become the current row in the first iteration of the loop
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
  for (i=0;i<dim_y;i++) {
    p_new[i+1] = psi[i];
  }
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
  for (i=0;i<dim_y;i++) {
    p_new[dim_y+i+3] = psi[ntot+i];
  }
// If periodic boundary condition along y, copy the first and last components
  if (b_y) {
    p_new[0]=p_new[dim_y];
    p_new[dim_y+1]=p_new[1];
    p_new[dim_y+2]=p_new[2*dim_y+2];
    p_new[2*dim_y+3]=p_new[dim_y+3];
  } else {
    p_new[0]=0.0;
    p_new[dim_y+1]=0.0;
    p_new[dim_y+2]=0.0;
    p_new[2*dim_y+3]=0.0;
  }
// Starts iteration along the rows
  for (i=0; i<dim_x; i++) {
//    printf("i %d\n",i);
    p_temp=p_old;
    p_old=p_current;
    p_current=p_new;
    p_new=p_temp;
    if (i<dim_x-1) {
// The generic row
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
      for (j=0; j<dim_y; j++) {
        p_new[j+1]=psi[j+(i+1)*dim_y];
      }
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
      for (j=0; j<dim_y; j++) {
        p_new[j+dim_y+3]=psi[ntot+j+(i+1)*dim_y];
      }
   } else {
// If in last row, put in p_new the first row if periodic along x, 0 otherwise )
      if (b_x) {
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
        for (j=0;j<dim_y;j++) {
          p_new[j+1] = psi[j];
        }
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
        for (j=0;j<dim_y;j++) {
          p_new[j+dim_y+3] = psi[ntot+j];
        }
      } else {
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
        for (j=0;j<dim_y;j++) {
          p_new[j+1] = 0.0;
        }
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
        for (j=0;j<dim_y;j++) {
          p_new[j+dim_y+3] = 0.0;
        }
      }
    }
// If periodic boundary condition along y, copy the first and last components
    if (b_y) {
      p_new[0]=p_new[dim_y];
      p_new[dim_y+1]=p_new[1];
      p_new[dim_y+2]=p_new[2*dim_y+2];
      p_new[2*dim_y+3]=p_new[dim_y+3];
    }
    i_low=i*dim_y;
// Ready to treat the current row
    if (add_real) {
#ifdef __clang__
#else
#pragma GCC ivdep
//#pragma vector aligned
#endif
      for (j=0; j<dim_y; j++) {
        psi_old[i_low] = (c1*disorder[i_low]-c2)*p_current[j+1] - c3_y*(p_current[j+2]+ p_current[j]) - c3_x*(p_old[j+1]+p_new[j+1]) + c_coef*wfc[i_low] - psi_old[i_low];
        i_low++;
      }
      i_low=ntot+i*dim_y;
#ifdef __clang__
#else
#pragma GCC ivdep
//#pragma vector aligned
#endif
      for (j=0; j<dim_y; j++) {
        psi_old[i_low] = (c1*disorder[i_low-ntot]-c2)*p_current[j+dim_y+3] - c3_y*(p_current[j+dim_y+4]+ p_current[j+dim_y+2]) - c3_x*(p_old[j+dim_y+3]+p_new[j+dim_y+3]) + c_coef*wfc[i_low] - psi_old[i_low];
        i_low++;
      }
    } else {
#ifdef __clang__
#else
#pragma GCC ivdep
//#pragma vector aligned
#endif
      for (j=0; j<dim_y; j++) {
        psi_old[i_low] = (c1*disorder[i_low]-c2)*p_current[j+1] - c3_y*(p_current[j+2]+ p_current[j]) - c3_x*(p_old[j+1]+p_new[j+1]) - c_coef*wfc[ntot+i_low] - psi_old[i_low];
        i_low++;
      }
      i_low=ntot+i*dim_y;
#ifdef __clang__
#else
#pragma GCC ivdep
//#pragma vector aligned
#endif
      for (j=0; j<dim_y; j++) {
        psi_old[i_low] = (c1*disorder[i_low-ntot]-c2)*p_current[j+dim_y+3] - c3_y*(p_current[j+dim_y+4]+ p_current[j+dim_y+2]) - c3_x*(p_old[j+dim_y+3]+p_new[j+dim_y+3]) + c_coef*wfc[i_low-ntot] - psi_old[i_low];
        i_low++;
      }
    }
  }
//  printf("out %f %f %f %f %f %f\n",creal(psi[0]),cimag(psi[0]),creal(psi_old[0]),cimag(psi_old[0]),creal(wfc[0]), cimag(wfc[0]));
#ifdef __ICC
  _mm_free(p_new);
  _mm_free(p_current);
  _mm_free(p_old);
#else
  free(p_new);
  free(p_current);
  free(p_old);
#endif
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
#ifdef TIMING
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
#endif
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
    elementary_clenshaw_step_real_1d(dim_x, boundary_condition, wfc, psi, psi_old, disorder, tab_coef[max_order-1], 0, c1, c2, c3);
// WARNING: max_order MUST be an even number, otherwise disaster guaranteed
    for (order=max_order-2;order>1;order-=2) {
      elementary_clenshaw_step_real_1d(dim_x, boundary_condition, wfc, psi_old, psi, disorder, tab_coef[order], 1, c1, c2, c3);
      elementary_clenshaw_step_real_1d(dim_x, boundary_condition, wfc, psi, psi_old, disorder, tab_coef[order-1], 0, c1, c2, c3);
    }
    c1 = two_over_delta_e;
    c2 = two_e0_over_delta_e;
    c3 = tab_tunneling[0]*two_over_delta_e;
    elementary_clenshaw_step_real_1d(dim_x, boundary_condition, wfc, psi_old, psi, disorder, tab_coef[0], 1, c1, c2, c3);
  }
  if (dimension==2) {
    int dim_x = tab_dim[0];
    int dim_y = tab_dim[1];
    int b_x = tab_boundary_condition[0];
    int b_y = tab_boundary_condition[1];
    double c3_x = 2.0*tab_tunneling[0]*two_over_delta_e;
    double c3_y = 2.0*tab_tunneling[1]*two_over_delta_e;
//    printf("%d %d %d %d %f %f %f %f\n",dim_x,dim_y,b_x,b_y,c1,c2,c3_x,c3_y);
    elementary_clenshaw_step_real_2d(dim_x, dim_y, b_x, b_y, wfc, psi, psi_old, disorder, tab_coef[max_order-1], 0, c1, c2, c3_x, c3_y);
// WARNING: max_order MUST be an even number, otherwise disaster guaranteed
    for (order=max_order-2;order>1;order-=2) {
//      printf("order %d %f\n",order,tab_coef[order]);
      elementary_clenshaw_step_real_2d(dim_x, dim_y, b_x, b_y, wfc, psi_old, psi, disorder, tab_coef[order], 1, c1, c2, c3_x, c3_y);
//      printf("order %d %f\n",order-1,tab_coef[order-1]);
      elementary_clenshaw_step_real_2d(dim_x, dim_y, b_x, b_y, wfc, psi, psi_old, disorder, tab_coef[order-1], 0, c1, c2, c3_x, c3_y);
    }
    c1 = two_over_delta_e;
    c2 = two_e0_over_delta_e;
    c3_x = tab_tunneling[0]*two_over_delta_e;
    c3_y = tab_tunneling[1]*two_over_delta_e;
    elementary_clenshaw_step_real_2d(dim_x, dim_y, b_x, b_y, wfc, psi_old, psi, disorder, tab_coef[0], 1, c1, c2, c3_x, c3_y);
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
//      phase=0.0;
     *nonlinear_phase = (*nonlinear_phase > fabs(phase)) ? *nonlinear_phase : fabs(phase);
      argument =  e0_times_delta_t+phase;
      wfc[i] = psi[i]*cos(argument)+psi[i+ntot]*sin(argument);
      wfc[i+ntot] = psi[i+ntot]*cos(argument)-psi[i]*sin(argument);
    }
  }
//  printf("done\n");
#ifdef TIMING
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("time in ns %ld\n",timespecDiff(&end, &start));
#endif
  return;
}

static void elementary_clenshaw_step_complex_1d(const int dim_x, const int boundary_condition, const double complex * restrict wfc, const double complex * restrict psi, double complex * restrict psi_old, const double * restrict disorder, const double c_coef, const int add_real, const double c1, const double c2, const double c3)
{
  int i;
//  printf("c_coef %f\n",c_coef);
// boundary_condition=1 is periodic
// boundary_condition=0 is open
  if (boundary_condition) {
    if (add_real) {
      psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]+psi[dim_x-1]) + c_coef*wfc[0] - psi_old[0];
      psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[0]+psi[dim_x-2]) + c_coef*wfc[dim_x-1] - psi_old[dim_x-1] ;
    } else {
      psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]+psi[dim_x-1]) + I*c_coef*wfc[0] - psi_old[0];
      psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[0]+psi[dim_x-2]) + I*c_coef*wfc[dim_x-1] - psi_old[dim_x-1] ;
    }
  } else {
    if (add_real) {
      psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]) + c_coef*wfc[0] - psi_old[0];
      psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[dim_x-2]) + c_coef*wfc[dim_x-1] - psi_old[dim_x-1] ;
    } else {
      psi_old[0] = (c1*disorder[0]-c2)*psi[0] - c3*(psi[1]) +  I*c_coef*wfc[0] - psi_old[0];
      psi_old[dim_x-1] = (c1*disorder[dim_x-1]-c2)*psi[dim_x-1] -c3*(psi[dim_x-2]) + I*c_coef*wfc[dim_x-1] - psi_old[dim_x-1] ;
    }
  }
  if (add_real) {
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
  } else {
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
  }
//  printf("%f %f %f %f %f %f\n",psi[0],psi_old[0],wfc[0]);
  return;
}

static void elementary_clenshaw_step_complex_2d(const int dim_x, const int dim_y, const int b_x, const int b_y, const double complex * restrict wfc, const double complex * restrict psi, double complex * restrict psi_old, const double * restrict disorder, const double c_coef, const int add_real, const double c1, const double c2, const double c3_x, const double c3_y)
{
  int i,j,i_low;
//  printf("in  %f %f %f %f %f %f\n",creal(psi[0]),cimag(psi[0]),creal(psi_old[0]),cimag(psi_old[0]),creal(wfc[0]), cimag(wfc[0]));
//  int ntot = dim_x*dim_y;
  double complex *p_old,*p_current,*p_new,*p_temp;
#ifdef __ICC
  p_old = (double complex *) _mm_malloc ((dim_y+2)*sizeof(double complex),SIZE);
  p_current = (double complex *) _mm_malloc ((dim_y+2)*sizeof(double complex),SIZE);
  p_new = (double complex *) _mm_malloc ((dim_y+2)*sizeof(double complex),SIZE);
#else
  p_old = (double complex *) malloc ((dim_y+2)*sizeof(double complex));
  p_current = (double complex *) malloc ((dim_y+2)*sizeof(double complex));
  p_new = (double complex *) malloc ((dim_y+2)*sizeof(double complex));
#endif

// If periodic boundary conditions along x, initialize p_current to the last row, otherwise 0
  if (b_x) {
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
    for (i=0;i<dim_y;i++) {
      p_current[i+1] = psi[i+(dim_x-1)*dim_y];
    }
  } else {
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
    for (i=0;i<dim_y+2;i++) {
      p_current[i] =0.0;
    }
  }
// Initialize the next row, which will become the current row in the first iteration of the loop
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
  for (i=0;i<dim_y;i++) {
    p_new[i+1] = psi[i];
  }
// If periodic boundary condition along y, copy the first and last components
  if (b_y) {
    p_new[0]=p_new[dim_y];
    p_new[dim_y+1]=p_new[1];
  } else {
    p_new[0]=0.0;
    p_new[dim_y+1]=0.0;
  }
// Starts iteration along the rows
  for (i=0; i<dim_x; i++) {
//    printf("i %d\n",i);
    p_temp=p_old;
    p_old=p_current;
    p_current=p_new;
    p_new=p_temp;
    if (i<dim_x-1) {
// The generic row
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
      for (j=0; j<dim_y; j++) {
        p_new[j+1]=psi[j+(i+1)*dim_y];
      }
    } else {
// If in last row, put in p_new the first row if periodic along x, 0 otherwise )
      if (b_x) {
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
        for (j=0;j<dim_y;j++) {
          p_new[j+1] = psi[j];
        }
      } else {
#ifdef __clang__
#else
#pragma GCC ivdep
#endif
        for (j=0;j<dim_y;j++) {
          p_new[j+1] = 0.0;
        }
      }
    }
// If periodic boundary condition along y, copy the first and last components
    if (b_y) {
      p_new[0]=p_new[dim_y];
      p_new[dim_y+1]=p_new[1];
    }
    i_low=i*dim_y;
// Ready to treat the current row
    if (add_real) {
#ifdef __clang__
#else
#pragma GCC ivdep
//#pragma vector aligned
#endif
      for (j=0; j<dim_y; j++) {
        psi_old[i_low] = (c1*disorder[i_low]-c2)*p_current[j+1] - c3_y*(p_current[j+2]+ p_current[j]) - c3_x*(p_old[j+1]+p_new[j+1]) + c_coef*wfc[i_low] - psi_old[i_low];
        i_low++;
      }
    } else {
#ifdef __clang__
#else
#pragma GCC ivdep
//#pragma vector aligned
#endif
      for (j=0; j<dim_y; j++) {
        psi_old[i_low] = (c1*disorder[i_low]-c2)*p_current[j+1] - c3_y*(p_current[j+2]+ p_current[j]) - c3_x*(p_old[j+1]+p_new[j+1]) + I*c_coef*wfc[i_low] - psi_old[i_low];
        i_low++;
      }
    }
  }
//  printf("out %f %f %f %f %f %f\n",creal(psi[0]),cimag(psi[0]),creal(psi_old[0]),cimag(psi_old[0]),creal(wfc[0]), cimag(wfc[0]));
#ifdef __ICC
  _mm_free(p_new);
  _mm_free(p_current);
  _mm_free(p_old);
#else
  free(p_new);
  free(p_current);
  free(p_old);
#endif
  return;
}

void chebyshev_complex(const int dimension, const int * restrict tab_dim, const int max_order,  const int * restrict tab_boundary_condition, double complex * restrict wfc, double complex * restrict psi, double complex * restrict psi_old,  const double * restrict disorder, const double * restrict tab_coef, const double * tab_tunneling, const double two_over_delta_e, const double two_e0_over_delta_e, const double g_times_delta_t, const double e0_times_delta_t, double * restrict nonlinear_phase)
{
  int i, order;
  double argument;
  double complex complex_argument;
  double phase;
  double c1, c2, c3;
#ifdef TIMING
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
#endif
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
    elementary_clenshaw_step_complex_1d(dim_x, boundary_condition, wfc, psi, psi_old, disorder, tab_coef[max_order-1], 0, c1, c2, c3);
// WARNING: max_order MUST be an even number, otherwise disaster guaranteed
    for (order=max_order-2;order>1;order-=2) {
//      printf("order %d %f\n",order,tab_coef[order]);
      elementary_clenshaw_step_complex_1d(dim_x, boundary_condition, wfc, psi_old, psi, disorder, tab_coef[order], 1, c1, c2, c3);
//      printf("order %d %f\n",order-1,tab_coef[order-1]);
      elementary_clenshaw_step_complex_1d(dim_x, boundary_condition, wfc, psi, psi_old, disorder, tab_coef[order-1], 0, c1, c2, c3);
    }
    c1 = two_over_delta_e;
    c2 = two_e0_over_delta_e;
    c3 = tab_tunneling[0]*two_over_delta_e;
    elementary_clenshaw_step_complex_1d(dim_x, boundary_condition, wfc, psi_old, psi, disorder, tab_coef[0], 1, c1, c2, c3);
  }
//  printf("%f %f %f %f %f %f\n",psi[100],psi_old[100],wfc[100]);
  if (dimension==2) {
    int dim_x = tab_dim[0];
    int dim_y = tab_dim[1];
    int b_x = tab_boundary_condition[0];
    int b_y = tab_boundary_condition[1];
    double c3_x = 2.0*tab_tunneling[0]*two_over_delta_e;
    double c3_y = 2.0*tab_tunneling[1]*two_over_delta_e;
//    printf("%d %d %d %d %f %f %f %f\n",dim_x,dim_y,b_x,b_y,c1,c2,c3_x,c3_y);
    elementary_clenshaw_step_complex_2d(dim_x, dim_y, b_x, b_y, wfc, psi, psi_old, disorder, tab_coef[max_order-1], 0, c1, c2, c3_x, c3_y);
// WARNING: max_order MUST be an even number, otherwise disaster guaranteed
    for (order=max_order-2;order>1;order-=2) {
//      printf("order %d %f\n",order,tab_coef[order]);
      elementary_clenshaw_step_complex_2d(dim_x, dim_y, b_x, b_y, wfc, psi_old, psi, disorder, tab_coef[order], 1, c1, c2, c3_x, c3_y);
//      printf("order %d %f\n",order-1,tab_coef[order-1]);
      elementary_clenshaw_step_complex_2d(dim_x, dim_y, b_x, b_y, wfc, psi, psi_old, disorder, tab_coef[order-1], 0, c1, c2, c3_x, c3_y);
    }
    c1 = two_over_delta_e;
    c2 = two_e0_over_delta_e;
    c3_x = tab_tunneling[0]*two_over_delta_e;
    c3_y = tab_tunneling[1]*two_over_delta_e;
    elementary_clenshaw_step_complex_2d(dim_x, dim_y, b_x, b_y, wfc, psi_old, psi, disorder, tab_coef[0], 1, c1, c2, c3_x, c3_y);
  }

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
#ifdef TIMING
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("time in ns %ld\n",timespecDiff(&end, &start));
#endif
//  printf("done\n");
//  printf("wfc %f %f %f %f %f %f\n",wfc[0],wfc[1],wfc[ntot-1]);
  return;
}

""")

if __name__ == "__main__":
  ffibuilder.compile(verbose=True)
