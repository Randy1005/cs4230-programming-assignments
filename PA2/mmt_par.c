void mmt_par(int n, double *__restrict__ a, double *__restrict__ b,
                 double *__restrict__ c) {
int i, j, k;

int remainder = n % 4;

#pragma omp parallel private(i,j,k) 
{
	for(k = 0; k < n; k++)
	#pragma omp for schedule(static)
		for(i=0;i<n;i++) {
			
			for (j = 0; j < remainder; j++) {
				c[i*n+j]=c[i*n+j]+a[k*n+j]*b[k*n+i];
			}

			for(j = remainder; j < n; j += 4) {
//    c[i][j] = c[i][j] + a[k][j]*b[k][i];
				c[i*n+j]=c[i*n+j]+a[k*n+j]*b[k*n+i];
				c[i*n+(j + 1)]=c[i*n+ (j + 1)]+a[k*n+ (j + 1)]*b[k*n+i];
				c[i*n+ (j + 2)]=c[i*n+ (j + 2)]+a[k*n+(j + 2)]*b[k*n+i];
				c[i*n+ (j + 3)]=c[i*n+ (j + 3)]+a[k*n+(j + 3)]*b[k*n+i];
			}
		}

}

}
