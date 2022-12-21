// matrix multiply kernel: C = A^T * B
__global__ void atb(const double *A, const double *B, double *C, int Ni, int Nj, int Nk) {
// Initially empty; will clearly not pass correctness
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // C[i][j] += A[k][i] * B[k][j];
  if((row < Ni) && (col < Nj)) {
    double val = 0;
    for(int k = 0; k < Nk; k++) {
      val += A[k*Ni+row] * B[k*Nj+col];
    }
    C[row*Nj+col] = val;
  }
}

