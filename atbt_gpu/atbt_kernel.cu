// matrix multiply kernel: C = A^T * B^T
__global__ void atbt(const double *A, const double *B, double *C, int Ni, int Nj, int Nk) {
// Initially empty; clearly Will not pass correctness test
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if((row < Ni) && (col < Nj)) {
    double val = 0;
    for(int k = 0; k < Nk; ++k) {
      val += A[k*Ni+row] * B[col*Nk+k];
    }
    C[row*Nj+col] = val;
  }
}

