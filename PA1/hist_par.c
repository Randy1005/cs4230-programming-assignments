#include <omp.h>
#include <stdio.h>

void pa1_hist_par(int nelts, int nbins, int *__restrict__ data, int *__restrict__ hist) 
{
#pragma omp parallel
 {
   // define a local sum[nbins]
   // for recording occurences of each number
   int sum[nbins];
   int i, id, nthreads;
	
   // get thread id
   id = omp_get_thread_num();
   nthreads = omp_get_num_threads();

   printf("thread %d\n", id);
   // for (i = 0; i < nelts; i++) hist[data[i]] += 1;
 }
}
