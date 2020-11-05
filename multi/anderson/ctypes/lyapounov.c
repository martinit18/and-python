#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef __ICC
#include <mathimf.h>
#else
#include <math.h>
#endif

#define min(x,y) ((x>y) ? y : x)
    
double core_lyapounov(const int dim_x, const int loop_step, const double * restrict disorder, const double energy, const double inv_tunneling)
{
  double gamma=0.0;	 
  int i, j, jmax;
  double psi_new, psi_cur=1.0, psi_old=M_PI/sqrt(13.0);
  for (i=0;i<dim_x;i+=loop_step) {
    jmax=min(i+loop_step,dim_x-1);
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

double core_lyapounov_non_diagonal_disorder(const int dim_x, const int loop_step, const double * restrict disorder, const double * restrict non_diagonal_disorder, const double energy, const double tunneling)
{
  double gamma=0.0;	 
  int i, j, jmax;
  double psi_new, psi_cur=1.0, psi_old=M_PI/sqrt(13.0);
  for (i=0;i<dim_x;i+=loop_step) {
    jmax=min(i+loop_step,dim_x-1);
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
      psi_new=(psi_cur*(disorder[j]-energy)-(tunneling-non_diagonal_disorder[j])*psi_old)/(tunneling-non_diagonal_disorder[j+1]);
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
