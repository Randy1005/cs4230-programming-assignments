#include <omp.h>
#include <stdio.h>

void pa1_hist_par(int nelts, int nbins, int *__restrict__ data, int *__restrict__ hist) 
{
  for (int i = 0; i < nbins; i++) hist[i] = 0;

  #pragma omp parallel
  {
    // define a local occurence[nbins]
    // for recording occurences of each number
    int occ[nbins];
    int i, id, nthreads;
    
    for (int i = 0; i < nbins; i++) occ[i] = 0;

    // get thread id
    id = omp_get_thread_num();
    nthreads = omp_get_num_threads();

    int start = id * (nelts / nthreads);
    int end = start + (nelts / nthreads);
    for (i = start; i < end; i++) {
      // update occ
      occ[data[i]] += 1;
    }
    
    // if nelts is not a multiplier of nthreads
    // each thread has to handle at most one more element
    int elts_left = nelts % nthreads;
    if (elts_left != 0 && id < elts_left) {
      occ[data[nelts - elts_left + id]] += 1;
    }

    // sequential version 
    // for (i = 0; i < nelts; i++) hist[data[i]] += 1;

    // add local occurences to global
    for (int i = 0; i < nbins; i++) {
      hist[i] += occ[i];
    }
  }

  /*
  printf("\nPar version:\n");
  for (int i = 0; i < nbins; i++) {
    printf("%d ", hist[i]);
  }
  printf("\n");
  */
	
	/*	
	int i;
	for (i=0;i<nbins;i++) hist[i]=0;
	
	#pragma omp parallel for
	for (i = 0; i < nelts; i++) 
	{
		#pragma omp atomic
		hist[data[i]] += 1;
	}
	*/

}
