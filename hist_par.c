void pa1_hist_par(int nelts, int nbins, int *__restrict__ data, int *__restrict__ hist) 
{
#pragma omp parallel
 {
#pragma omp master
  {

   int i;

   for (i=0;i<nbins;i++) hist[i]=0;
   for (i = 0; i < nelts; i++) hist[data[i]] += 1;
  }
 }
}
