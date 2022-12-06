// For compiling, use "nvcc -O3"; upon logging on to a CHPC node, "module load cuda" is needed to load "nvcc"

#include <stdio.h>

void checkCUDAError(const char *msg);

#include <stdio.h>

#define RADIUS        3
#define BLOCK_SIZE    256
#define NUM_ELEMENTS  (4097*257)
// #define NUM_ELEMENTS 4096
#define FIXME1 32
#define FIXME2 32

// The FIXMEs indicate where code must replace the FIXMEs.
// The number of output elements is N, out[0:N-1]
// The number of input elements is N+2*RADIUS, IN[0:N+2*RADIUS-1]
// Each element of out holds the sum of a set of 2*RADIUS+1 contiguous elements from in
// The sum of contents in in[0:2*RADIUS] is placed in out[0], 
// sum of elements in in[1:2*RADIUS+1] is placed in out[1], etc.

__global__ void stencil_1d(int *in, int *out, int N) 
{
	__shared__ int tmp[512 + 2 * RADIUS];

	// g : linearized thread index across all threads
	int g = blockDim.x * blockIdx.x + threadIdx.x;
	
	// l : shared memory index
	int l = threadIdx.x + RADIUS;

	// read input into shared memory
	if (g < N) {
		if (threadIdx.x == 0) {
			tmp[l - RADIUS] = in[g];
			tmp[l - RADIUS + 1] = in[g + 1];
			tmp[l - RADIUS + 2] = in[g + 2];
		}

		if (threadIdx.x == 511 || g == N - 1) {
			tmp[l + 1] = in[g + RADIUS + 1];
			tmp[l + 2] = in[g + RADIUS + 2];
			tmp[l + 3] = in[g + RADIUS + 3];
		}
		
		tmp[l] = in[g + RADIUS];
	}

	__syncthreads();

	// calculate stencil
	int sum = 0;
	for (int r = -RADIUS; r <= RADIUS; r++) {
		sum += tmp[l + r];
	}

	out[g] = sum;
}

int main()
{
  int i,r;
  int *d_in, *d_out;
	int *h_in, *h_out, *h_ref;

	h_in = (int*)malloc(sizeof(int) * (NUM_ELEMENTS + 2 * RADIUS));
	h_out = (int*)malloc(sizeof(int) * NUM_ELEMENTS);
	h_ref = (int*)malloc(sizeof(int) * NUM_ELEMENTS);

  // Initialize host data
  for(i = 0; i < (NUM_ELEMENTS + 2*RADIUS); i++ )
    h_in[i] = i; 
  for(i = 0; i < NUM_ELEMENTS; i++)
    h_ref[i] = 0;

  for(i = 0; i < NUM_ELEMENTS; i++)
   for(r = -RADIUS; r <= RADIUS; r++)
    h_ref[i] += h_in[RADIUS+i+r];

  // Allocate space on the device
  cudaMalloc( &d_in, (NUM_ELEMENTS + 2*RADIUS) * sizeof(int));
  cudaMalloc( &d_out, NUM_ELEMENTS * sizeof(int));
  checkCUDAError("cudaMalloc");

  // Copy input data to device
	cudaMemcpy( d_in, h_in, (NUM_ELEMENTS + 2*RADIUS) * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy");

  // Fix the FIXME's
  
	int num_blk = (NUM_ELEMENTS + 512 - 1) / 512;
	stencil_1d<<<num_blk, 512>>> (d_in, d_out,NUM_ELEMENTS);
  checkCUDAError("Kernel Launch Error:");

  cudaMemcpy( h_out, d_out, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMalloc");

  for( i = 0; i < NUM_ELEMENTS; ++i )
    if (h_ref[i] != h_out[i])
    {
      printf("ERROR: Mismatch at index %d: expected %d but found %d\n",i,h_ref[i], h_out[i]);
      break;
    }

    if (i== NUM_ELEMENTS) printf("SUCCESS!\n");

  // Free out memory
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
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

