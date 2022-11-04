void mmt_par(int n, double *__restrict__ a, double *__restrict__ b,
                 double *__restrict__ c) {
int i, j, k;

#pragma omp parallel private(i, j, k)
{
	for(k = 0; k < n; k++)
		#pragma omp for 
		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
				c[i*n+j]=c[i*n+j]+a[k*n+j]*b[k*n+i];

}

}
