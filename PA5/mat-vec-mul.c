#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#define N 8192 
#define Niter 10
#define threshold 0.0000001

void mvseq(int n, double m[][n], double x[n], double y[n]);
void mvpar(int n, double m[][n], double x[n], double y[n]);
void compare(int n, double wref[n], double w[n]);

double A[N][N], x[N],temp[N],xx[N],temp1[N];
int myid, nprocs;
int main(int argc, char *argv[]) {

double clkbegin, clkend;
double t, tmax, *tarr;
int i,j,it;

  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &myid );
  MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
  tarr = malloc(sizeof(double)*nprocs);

  for(i=0;i<N;i++)
   { 
     x[i] = xx[i] = sqrt(1.0*i);
     for(j=0;j<N;j++) A[i][j] = 2.0*((i+j) % N)/(1.0*N*(N-1));
   }

  if (myid == 0) 
  {
   clkbegin = MPI_Wtime();
   mvseq(N,A,x,temp);
   clkend = MPI_Wtime();
   t = clkend-clkbegin;
   printf("Repeated MV: Sequential Version: Matrix Size = %d; %.2f GFLOPS; Time = %.3f sec; \n",
           N,2.0*1e-9*N*N*Niter/t,t);

  }

  MPI_Barrier(MPI_COMM_WORLD);

  clkbegin = MPI_Wtime();
  mvpar(N,A,xx,temp1);
  clkend = MPI_Wtime();
  t = clkend-clkbegin;
  MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 
  if (myid == 0)
  {
   printf("Repeated MV: Parallel Version: Matrix Size = %d; %.2f GFLOPS; Time = %.3f sec; \n",
          N,2.0*1e-9*N*N*Niter/tmax,tmax);
   compare(N,x,xx);
  }
  MPI_Finalize();
}


void mvseq(int n, double m[][n], double x[n], double y[n])
{ int i,j,iter;
  
  for(i=0;i<n;i++) { y[i]=0.0; }
  for(iter=0;iter<Niter;iter++)
    {
      for(i=0;i<n;i++)
	for(j=0;j<n;j++)
	  y[i] = y[i] + m[i][j]*x[j];
      for (i=0; i<N; i++) x[i] = sqrt(y[i]);
    }
}

void mvpar(int n, double m[][n], double x[n], double y[n])
{ int i,j,iter;

  // n is a multiple of 2, 4, 8, 16
  int rows = n / nprocs;


  double sbuff[rows], rbuff[n];
  for(i = 0; i < rows; i++) { sbuff[i] = 0.0; }
  for(iter=0;iter<Niter;iter++)
  {
      int cnt = 0;
      for(i = myid * rows; i < (myid + 1) * rows; i++) {
	for(j = 0; j < n; j++) {
	  sbuff[cnt] = sbuff[cnt] + m[i][j]*x[j];
	}
	cnt++;
      }
     
      MPI_Allgather(&sbuff, rows, MPI_DOUBLE,
		    &rbuff, rows, MPI_DOUBLE, MPI_COMM_WORLD);
      /*
      printf("myid = %d\n", myid);
      for (int k = 0; k < n; k++) {
        printf("sbuff[%d] = %lf\n", k, sbuff[k]);
      }
      */


      for (i = 0; i < N; i++) x[i] = sqrt(rbuff[i]);
  }
}

void compare(int n, double wref[n], double w[n])
{
double maxdiff,this_diff;
double minw,maxw,minref,maxref;
int numdiffs;
int i;
  numdiffs = 0;
  maxdiff = 0;
  minw = minref = 1.0e9;
  maxw = maxref = -1.0;
  for (i=0;i<n;i++)
    {
     this_diff = wref[i]-w[i];
     if (w[i] < minw) minw = w[i];
     if (w[i] > maxw) maxw = w[i];
     if (wref[i] < minref) minref = wref[i];
     if (wref[i] > maxref) maxref = wref[i];
     if (this_diff < 0) this_diff = -1.0*this_diff;
     if (this_diff>threshold)
      { numdiffs++;
        if (this_diff > maxdiff) maxdiff=this_diff;
      }
    }
   if (numdiffs > 0)
      printf("%d Diffs found over threshold %f; Max Diff = %f\n",
               numdiffs,threshold,maxdiff);
   else
      printf("No differences found between base and test versions\n");
   printf("MinRef = %f; MinPar = %f; MaxRef = %f; MaxPar = %f\n",
          minref,minw,maxref,maxw);
}

