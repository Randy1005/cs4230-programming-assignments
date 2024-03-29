#include <cstdlib>
#include <iostream>
#include <chrono>


#define BLOCK_SIZE 32
// matrix multiplication GPU kernel
__global__ void matmul(int *a, int *b, int *c, int N) {

  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if ((col < N) && (row < N)) {
    for (int k = 0; k < N; k++) {
      c[row * N + col] += a[row * N + k] * b[k * N + col];
    }
  }

}


// initialize matrix with N * N random integers
void init_matrix(int *mat, int size) {
  for (int i = 0; i < size; i++) {
    mat[i] = ::rand() % 100;
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
  int *h_a, *h_b;

  // output array
  int *h_c, *h_cref;

  // device arrays
  int *d_a, *d_b, *d_c;

  // allocate memory on host device
  h_a = new int[N * N];
  h_b = new int[N * N];
  h_c = new int[N * N];
  h_cref = new int[N * N];

  // initialize c to all 0s
  for (int i = 0; i < N * N; i++) {
    h_cref[i] = 0;
    h_c[i] = 0;
  }

  // allocate memory on GPU device
  cudaMalloc(&d_a, bytes);  
  cudaMalloc(&d_b, bytes);  
  cudaMalloc(&d_c, bytes);
  checkCUDAError("cuda malloc failure");

  // initialize input matrices
  init_matrix(h_a, N * N);
  init_matrix(h_b, N * N);

  // copy arrays to GPU device
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
  checkCUDAError("cuda memcpy H2D failure");
  
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(N / block.x, N / block.y);
  
  auto beg_gpu = std::chrono::steady_clock::now(); 

  // launch matrix multiplication kernel
  matmul<<<grid, block>>>(d_a, d_b, d_c, N);
 

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
      for (int k = 0; k < N; k++) {
        h_cref[i*N + j] += h_a[i*N + k] * h_b[k*N + j]; 
      }
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
      std::exit(EXIT_FAILURE); 
    }
  }


  std::cout << "GPU mat mul runtime = " << gpu_time << " ns\n";
  std::cout << "CPU mat mul runtime = " << cpu_time << " ns\n";
  std::cout << "speedup = " << cpu_time / gpu_time << "\n";


  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);




  return 0;
}





