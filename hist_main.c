// Use "gcc -O3 -fopenmp hist_main.c hist_seq.c hist_par.c" to compile

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#ifndef N
#define N (10000000)
#endif
#define NBINS 16
#define NTrials (5)

void compare(int nbins, int ref[], int test[], int numt);
void pa1_hist_seq(int nelts, int nbins, int *__restrict__ data, int *__restrict__ hist);
void pa1_hist_par(int nelts, int nbins, int *__restrict__ data, int *__restrict__ hist);

int D[N], B[NBINS], BB[NBINS];

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
  
  printf("Size of Data Array = %d\n",N);

  for(i=0;i<N;i++) D[i] = rand() % NBINS;

  
  printf("Reference sequential code performance in GigaOps");
  mint_seq = 1e9; maxt_seq = 0;
  for(trial=0;trial<NTrials;trial++)
  {
   tstart = omp_get_wtime();
   pa1_hist_seq(N, NBINS, D, B);
   telapsed = omp_get_wtime()-tstart;
   if (telapsed < mint_seq) mint_seq=telapsed;
   if (telapsed > maxt_seq) maxt_seq=telapsed;
  }
  printf(" Min: %.2f; Max: %.2f\n",1.0e-9*N/maxt_seq,1.0e-9*N/mint_seq);
  
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
    tstart = omp_get_wtime();
    pa1_hist_par(N, NBINS, D, BB);
    telapsed = omp_get_wtime()-tstart;
    if (telapsed < mint_par[nt]) mint_par[nt]=telapsed;
    if (telapsed > maxt_par[nt]) maxt_par[nt]=telapsed;
   }
    compare(NBINS,B,BB,nthreads[nt]);

  }
  printf("Performance (Best & Worst) of parallelized version: GFLOPS || Speedup on ");
  for (nt=0;nt<num_cases-1;nt++) printf("%d/",nthreads[nt]);
  printf("%d threads\n",nthreads[num_cases-1]);
  printf("Best Performance (GFLOPS || Speedup): ");
  for (nt=0;nt<num_cases;nt++) printf("%.2f ",1.0e-9*N/mint_par[nt]);
  printf(" || ");
  for (nt=0;nt<num_cases;nt++) printf("%.2f ",mint_seq/mint_par[nt]);
  printf("\n");
  printf("Worst Performance (GFLOPS || Speedup): ");
  for (nt=0;nt<num_cases;nt++) printf("%.2f ",1.0e-9*N/maxt_par[nt]);
  printf(" || ");
  for (nt=0;nt<num_cases;nt++) printf("%.2f ",mint_seq/maxt_par[nt]);
  printf("\n");
}

void compare(int nbins, int ref[], int test[], int numt)
{
  int numdiffs;
  int i;
  numdiffs = 0;
  for (i=0;i<nbins;i++) if ((ref[i]-test[i]) != 0) numdiffs++;
  if (numdiffs > 0)
   printf("Error when executing on %d threads; %d differences found\n",
           numt,numdiffs);
}

