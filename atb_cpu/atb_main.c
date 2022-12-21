// Use "gcc -O3 -fopenmp atb_main.c atb_par.c " to compile 

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define NTrials (5)
#define threshold (0.0000001)

void atb_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk);

void atb_seq(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++)
   for (j = 0; j < Nj; j++)
    for (k = 0; k < Nk; k++)
// C[i][j] = C[i][j] + A[k][i]*B[k][j];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
}


int main(int argc, char *argv[]){
  double tstart,telapsed;

  int i,j,k,nt,trial,max_threads,num_cases;
  int nthr_32[9] = {1,2,4,8,10,12,14,15,31};
  int nthr_40[9] = {1,2,4,8,10,12,14,19,39};
  int nthr_48[9] = {1,2,4,8,10,15,20,23,47};
  int nthr_56[9] = {1,2,4,8,10,15,20,27,55};
  int nthr_64[9] = {1,2,4,8,10,15,20,31,63};
  int nthreads[9];
  double mint_par[9],maxt_par[9];
  double mint_seq,maxt_seq;

  double *A, *B, *C, *Cref;
  int Ni,Nj,Nk;

  printf("Specify Matrix dimension Ni, Nj, Nk: ");
  scanf("%d %d %d", &Ni,&Nj,&Nk);
  A = (double *) malloc(sizeof(double)*Ni*Nk);
  B = (double *) malloc(sizeof(double)*Nk*Nj);
  C = (double *) malloc(sizeof(double)*Ni*Nj);
  Cref = (double *) malloc(sizeof(double)*Ni*Nj);
  for (i=0; i<Ni; i++)
   for (k=0; k<Nk; k++)
    A[k*Ni+i] = k*Ni+i-1;
  for (k=0; k<Nk; k++)
   for (j=0; j<Nj; j++)
    B[k*Nj+j] = k*Nj+j+1;
  for (i=0; i<Ni; i++)
   for (j=0; j<Nj; j++) {
    C[i*Nj+j] = 0;
    Cref[i*Nj+j] = 0;}

  printf("Reference sequential code performance for ATB (in GFLOPS)");
  mint_seq = 1e9; maxt_seq = 0;
  for(trial=0;trial<NTrials;trial++)
  {
   for(i=0;i<Ni;i++) for(j=0;j<Nj;j++) Cref[i*Nj+j] = 0;
   tstart = omp_get_wtime();
   atb_seq(A,B,Cref,Ni,Nj,Nk);
   telapsed = omp_get_wtime()-tstart;
   if (telapsed < mint_seq) mint_seq=telapsed;
   if (telapsed > maxt_seq) maxt_seq=telapsed;
  }
  printf(" Min: %.2f; Max: %.2f\n",2.0e-9*Ni*Nj*Nk/maxt_seq,2.0e-9*Ni*Nj*Nk/mint_seq);


  max_threads = omp_get_max_threads();
  printf("Max Threads (from omp_get_max_threads) = %d\n",max_threads);
  switch (max_threads)
  {
          case 32: for(i=0;i<9;i++) nthreads[i] = nthr_32[i]; num_cases=9; break;
          case 40: for(i=0;i<9;i++) nthreads[i] = nthr_40[i]; num_cases=9; break;
          case 48: for(i=0;i<9;i++) nthreads[i] = nthr_48[i]; num_cases=9; break;
          case 56: for(i=0;i<9;i++) nthreads[i] = nthr_56[i]; num_cases=9; break;
          case 64: for(i=0;i<9;i++) nthreads[i] = nthr_64[i]; num_cases=9; break;
          default: {
                    nt = 1;i=0;
                    while (nt <= max_threads) {nthreads[i]=nt; i++; nt *=2;}
                    if (nthreads[i-1] < max_threads) {nthreads[i] = max_threads; i++;}
                    num_cases = i;
                    nthreads[num_cases-1]--;
                    nthreads[num_cases-2]--;
                   }
  }

  for (nt=0;nt<num_cases;nt ++)
  {
   omp_set_num_threads(nthreads[nt]);
   mint_par[nt] = 1e9; maxt_par[nt] = 0;
   for (trial=0;trial<NTrials;trial++)
   {
    for(i=0;i<Ni;i++) for(j=0;j<Nj;j++) C[i*Nj+j] = 0;
    tstart = omp_get_wtime();
    atb_par(A,B,C,Ni,Nj,Nk);
    telapsed = omp_get_wtime()-tstart;
    if (telapsed < mint_par[nt]) mint_par[nt]=telapsed;
    if (telapsed > maxt_par[nt]) maxt_par[nt]=telapsed;
for (int l = 0; l < Ni*Nj; l++) if (fabs((C[l] - Cref[l])/Cref[l])>threshold) {printf("Error: mismatch at linearized index %d, was: %f, should be: %f\n", l, C[l], Cref[l]); return -1;}
   }
  }
    printf("Performance (Best & Worst) of parallelized version: GFLOPS on ");
  for (nt=0;nt<num_cases-1;nt++) printf("%d/",nthreads[nt]);
  printf("%d threads\n",nthreads[num_cases-1]);
  printf("Best Performance (GFLOPS): ");
  for (nt=0;nt<num_cases;nt++) printf("%.2f ",2.0e-9*Ni*Nj*Nk/mint_par[nt]);
  printf("\n");
  printf("Worst Performance (GFLOPS): ");
  for (nt=0;nt<num_cases;nt++) printf("%.2f ",2.0e-9*Ni*Nj*Nk/maxt_par[nt]);
  printf("\n");
}

