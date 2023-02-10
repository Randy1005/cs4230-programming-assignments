#include <cstdlib>
#include <iostream>
#include <chrono>


// thread blocks of 32 x 8
#define TILE_DIM 32
#define BLOCK_ROWS 8



// matrix transpose GPU kernel
__global__ void mattranspose(const int *in, int *out, int N) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  int width = gridDim.x * TILE_DIM;

  for (int k = 0; k < TILE_DIM; k+= BLOCK_ROWS) {
    out[x * width + (y+k)] = in[(y+k) * width + x];
  }

}


// initialize matrix with N * N random integers
void init_matrix(int *mat, int size) {
  for (int i = 0; i < size; i++) {
    mat[i] = rand() % 100;
  }
}

void checkCUDAError(const std::string& msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        std::cerr << "Cuda error: " 
          << msg << ", " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
}



int main(int argc, char* argv[]) {
 
  // matrix size = N * N
  int N = 1024;
  int bytes = N * N * sizeof(int);
  
  // input arrays
  int *h_a;

  // output array
  int *h_c, *h_cref;

  // device arrays
  int *d_a, *d_c;

  // allocate memory on host device
  h_a = new int[N * N];
  h_c = new int[N * N];
  h_cref = new int[N * N];

  // initialize c to all 0s
  for (int i = 0; i < N * N; i++) {
    h_cref[i] = 0;
    h_c[i] = 0;
  }

  // allocate memory on GPU device
  cudaMalloc(&d_a, bytes);  
  cudaMalloc(&d_c, bytes);
  checkCUDAError("cuda malloc failure");

  // initialize input matrices
  init_matrix(h_a, N * N);

  // copy arrays to GPU device
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  checkCUDAError("cuda memcpy H2D failure");
  
  dim3 block(TILE_DIM, BLOCK_ROWS);
  // each thread block doesn't have to be all occupied
  // in this case, thread block is 32 x 8
  // but grid dimension is 32 x 32
  dim3 grid(N / TILE_DIM, N / TILE_DIM);
  
  auto beg_gpu = std::chrono::steady_clock::now(); 

  // launch matrix transpose kernel
  mattranspose<<<grid, block>>>(d_a, d_c, N);
 

  auto end_gpu = std::chrono::steady_clock::now(); 
  size_t gpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
    end_gpu - beg_gpu
  ).count();
  
  // sychronize device jobs
  cudaDeviceSynchronize();


  // copy result back to host
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  checkCUDAError("cuda memcpy D2H failure");

  auto beg_cpu = std::chrono::steady_clock::now();
  // sequential version for correctness check
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      h_cref[i*N + j] = h_a[j*N + i]; 
    }
  }
  auto end_cpu = std::chrono::steady_clock::now();
  size_t cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
    end_cpu - beg_cpu
  ).count();


  // check if GPU result matches with CPU result
  for (int i = 0; i < N * N; i++) {
    if (h_cref[i] != h_c[i]) {
      std::cerr << "result does not match\n";
      std::cerr << "h_cref " << i << " = " << h_cref[i] << "\n";
      std::cerr << "h_c " << i << " = " << h_c[i] << "\n";
      std::exit(EXIT_FAILURE); 
    }
  }


  std::cout << "GPU mat transpose runtime = " << gpu_time << " ns\n";
  std::cout << "CPU mat transpose runtime = " << cpu_time << " ns\n";
  std::cout << "speedup = " << cpu_time / gpu_time << "\n";


  cudaFree(d_a);
  cudaFree(d_c);

  free(h_a);
  free(h_c);

  return 0;
}





