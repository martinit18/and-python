#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef __ICC
#include <mathimf.h>
#else
#include <math.h>
#endif
#include "mkl.h"
#include "mkl_lapacke.h"

#define min(x,y) ((x>y) ? y : x)
    
double core_lyapounov(const int dim_x, const int loop_step, const double * restrict disorder, const double energy, const double inv_tunneling)
{
  double gamma=0.0;	 
  int i, j, jmax;
  double psi_new, psi_cur=1.0, psi_old=M_PI/sqrt(13.0);
  for (i=0;i<dim_x;i+=loop_step) {
    jmax=min(i+loop_step,dim_x);
/*
#ifdef __clang__
#pragma clang loop vectorize(enable) interleave(enable)
#else
#ifdef __ICC
//#pragma vector aligned
#pragma vector always
#pragma ivdep
#else      
#pragma GCC ivdep
#endif
#endif
*/
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

double core_lyapounov_non_diagonal_disorder(const int dim_x, const int loop_step, const double * restrict disorder, const int b, const double * restrict non_diagonal_disorder, const double energy, const double tunneling)
{
  double gamma=0.0;	 
  int i, j, jmax;
  if (b==1) {
    double psi_new, psi_cur=1.0, psi_old=M_PI/sqrt(13.0);
    for (i=0;i<dim_x;i+=loop_step) {
      jmax=min(i+loop_step,dim_x);
/*
#ifdef __clang__
#pragma clang loop vectorize(enable) interleave(enable)
#else
#ifdef __ICC
//#pragma vector aligned
#pragma vector always
#pragma ivdep
#else      
#pragma GCC ivdep
#endif
#endif
*/
      for(j=i;j<jmax;j++) {
        psi_new=(psi_cur*(disorder[j]-energy)+psi_old*(non_diagonal_disorder[j]-tunneling))/(tunneling-non_diagonal_disorder[j+1]);
        psi_old=psi_cur;
        psi_cur=psi_new;
      }  
      gamma+=log(fabs(psi_cur));
      psi_old/=psi_cur;
      psi_cur=1.0;
    }  
  }
//  printf("gamma = %lf\n",gamma);
  return(gamma);
}

double core_lyapounov_2d_c(const int dim_x, const int dim_y, const double * restrict disorder, const double energy, const int nrescale, const int i0, double * Bn, double * Bn_old, double * g1n)
{
  int i, j, k;
  double x=0.0;
  double temp;
  int *ipiv;
  double small_b, small_b_square;
  int info;
  ipiv = (int *) calloc(dim_y,sizeof(int));  
  for (i=0; i<=((dim_x-1)/nrescale)*nrescale; i++) {
//    printf("i= %d \n",i);
    if (i%nrescale==100) {
      for (j=0; j<dim_y; j++) {
        Bn_old[j*(dim_y+1)] += (energy-disorder[i*dim_y+j]);
        Bn_old[j*dim_y+(j+1)%dim_y] -= 1.0;
        Bn_old[j*dim_y+(j+dim_y-1)%dim_y] -= 1.0;
      } 
    } else {
      for (j=0; j<dim_y; j++) {
        for (k=0; k<dim_y; k++) {
          Bn_old[k*dim_y+j] += (energy-disorder[i*dim_y+j])*Bn[k*dim_y+j] - Bn[k*dim_y+(j+1)%dim_y] - Bn[k*dim_y+(j+dim_y-1)%dim_y];
        }
      }
    }
    for (j=0; j<dim_y*dim_y; j++) {
      temp = Bn[j];
      Bn[j] = Bn_old[j];
      Bn_old[j] = -temp;
      printf("i= %d j= %d Bn= %lf Bn_old= %lf\n",i,j,Bn[j],Bn_old[j]);
    }
    if (i%nrescale==0) {
//      info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, dim_y, dim_y, Bn, dim_y, ipiv);
      dgetrf(&dim_y,&dim_y,Bn,&dim_y,ipiv,&info);
      for (j=0; j<dim_y; j++) {
        printf("i= %d j= %d ipiv= %d\n",i,j,ipiv[j]);
      }  
//      printf("info_dgetrf %d \n",info);  
      if (i>=i0) {
        LAPACKE_dgetrs(LAPACK_COL_MAJOR,'n',dim_y, dim_y, Bn, dim_y, ipiv, g1n, dim_y); 
        for (j=0;j<dim_y*dim_y;j++) {
          printf("i= %d j= %d Bn= %lf g1n= %lf\n",i,j,Bn[j],g1n[j]);       
        }
        small_b_square = cblas_ddot(dim_y*dim_y,g1n,1,g1n,1);
        small_b = sqrt(small_b_square);
        printf("i= %d small_b= %lf\n",i,small_b);
        for (j=0; j<dim_y*dim_y; j++) {
          g1n[j] *= 1.0/small_b;    
        }  
      }          
      if (i>i0) {
        x -= log(small_b);
      }  
//      printf("x = %lf\n",x);
      LAPACKE_dgetrs(LAPACK_COL_MAJOR,'n',dim_y, dim_y, Bn, dim_y, ipiv, Bn_old, dim_y);    
      for (j=0; j<dim_y*dim_y; j++) {
        Bn[j]=0.0;        
      }  
      for (j=0; j<dim_y; j++) {
        Bn[j*(dim_y+1)]=1.0;        
      }    
    }
  }     
  printf("xfinal = %lf\n",x);
  return(x);
}

void update_B_c(const int dim_x, const int dim_y, const double * restrict disorder, const double energy, const int nrescale, const int i, double *  Bn, double *  Bn_old)
{
  int j, k;
  if (i%nrescale==1) {
    for (j=0; j<dim_y; j++) {
      Bn_old[j*(dim_y+1)] += (energy-disorder[i*dim_y+j]);
      Bn_old[j*dim_y+(j+1)%dim_y] -= 1.0;
      Bn_old[j*dim_y+(j+dim_y-1)%dim_y] -= 1.0;
    } 
  } else {
    for (k=0; k<dim_y; k++) {
#pragma ivdep
#pragma unroll
      for (j=0; j<dim_y; j++) {
        Bn_old[k*dim_y+j] += (energy-disorder[i*dim_y+j])*Bn[k*dim_y+j] - Bn[k*dim_y+(j+1)%dim_y] - Bn[k*dim_y+(j+dim_y-1)%dim_y];
      }
    }
  }
  return;
}
