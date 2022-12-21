#include <stdio.h>
#include <time.h>
#include <iostream>
#define threshold 0.0000001

#define FIXME1 1
#define FIXME2 2
#define FIXME3 3
#define FIXME4 4


void checkCUDAError(const char *msg);

cudaEvent_t start, stop;
float tstart, elapsedTime;

__global__ void atb(const double *A, const double *B, double *C, int Ni, int Nj, int Nk);

int main(){

  double *h_A, *h_B, *h_C, *h_Cref, *d_A, *d_B, *d_C;
  int i,j,k,Ni,Nj,Nk;

  printf("Specify Matrix dimension Ni, Nj, Nk: ");
  scanf("%d %d %d", &Ni,&Nj,&Nk);
  h_A = (double *) malloc(sizeof(double)*Ni*Nk);
  h_B = (double *) malloc(sizeof(double)*Nk*Nj);
  h_C = (double *) malloc(sizeof(double)*Ni*Nj);
  h_Cref = (double *) malloc(sizeof(double)*Ni*Nj);
  for (i=0; i<Ni; i++)
   for (k=0; k<Nk; k++)
    h_A[k*Ni+i] = k*Ni+i-1;
  for (k=0; k<Nk; k++)
   for (j=0; j<Nj; j++)
    h_B[k*Nj+j] = k*Nj+j+1;
  for (i=0; i<Ni; i++)
   for (j=0; j<Nj; j++) {
    h_C[i*Nj+j] = 0;
    h_Cref[i*Nj+j] = 0;}

  for (i=0;i<Ni;i++)
   for (k=0;k<Nk;k++)
    for (j=0;j<Nj;j++)
  // h_Cref[i][j] += h_A[k][i]*h_B[k][j];
     h_Cref[i*Nj+j] += h_A[i+Ni*k]*h_B[k*Nj+j];
  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, Nk*Ni*sizeof(double));
  cudaMalloc(&d_B, Nj*Nk*sizeof(double));
  cudaMalloc(&d_C, Ni*Nj*sizeof(double));
  checkCUDAError("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, Nk*Ni*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, Nj*Nk*sizeof(double), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy H2D failure");

  int block_size = 32;
  dim3 block;
  block.x = block_size;
  block.y = block_size;
  dim3 grid;
  grid.x = (Ni + block.x - 1) / block.x;
  grid.y = (Nj + block.y - 1) / block.y;
  printf("bloack size x %d \n", block.x);
  printf("block size y %d \n", block.y);
  printf("grid size x %d \n", grid.x);
  printf("grid size y %d \n", grid.y);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for(int trial=0;trial<5;trial++)
  {
   cudaEventRecord(start);
   // Launch kernel
   std::cout << "Ni: " << Ni << std::endl;
   std::cout << "Nj: " << Nj << std::endl;
   std::cout << "Nk: " << Nk << std::endl;
   atb<<<grid, block>>>(d_A, d_B, d_C, Ni,Nj,Nk);
   cudaEventRecord(stop);
   checkCUDAError("kernel launch");
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsedTime, start,stop);
//   cudaDeviceSynchronize();
   // Copy results back to host
   cudaMemcpy(h_C, d_C, Ni*Nj*sizeof(double), cudaMemcpyDeviceToHost);
   checkCUDAError("cudaMemcpy D2H");
   for (int l = 0; l < Ni*Nj; l++) if (fabs((h_C[l] - h_Cref[l])/h_Cref[l])>threshold) {printf("Error: mismatch at linearized index %d, was: %f, should be: %f\n", l, h_C[l], h_Cref[l]); return -1;}
   printf("<Ni=%d,Nj=%d,Nk=%d>: Trial %d: GFLOPS: %.2f\n",Ni,Nj,Nk,trial,2.0e-6*Ni*Nj*Nk/elapsedTime);
  }
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}


