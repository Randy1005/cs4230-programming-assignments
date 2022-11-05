#include <omp.h>
void mv2_par(int n, double *__restrict__ m, double *__restrict__ x, double *__restrict__ y, double *__restrict__ z)
{
  int i, j;
	int rem = n % 4;

	
{
	#pragma omp parallel private(i, j)
  for (i = 0; i < n; i++) {
		#pragma omp for
		for (j = 0; j < n; j++) {
//    y[j] = y[j] + m[i][j] * x[i];
//    z[j] = z[j] + m[j][i] * x[i];
      y[j] = y[j] + m[i*n+j] * x[i];
		}

	}


	#pragma omp parallel private(i, j)
	{
		#pragma omp for
		for (j = 0; j < n; j++) {
			for (i = 0; i < n; i++) {
		//    y[j] = y[j] + m[i][j] * x[i];
		//    z[j] = z[j] + m[j][i] * x[i];
				z[j] = z[j] + m[j*n+i] * x[i];
			}
		}
	}

}


}
