void mmtu_par(int n, double *__restrict__ a, double *__restrict__ b,
                 double *__restrict__ c) {
int i, j, k;

#pragma omp parallel private(i,j,k) 
{
#pragma omp master
for(i=0;i<n;i++)
 for(j=i;j<n;j++)
  for(k=i;k<=j;k++)
//    c[i][j] = c[i][j] + a[k][j]*b[k][i];
  {
   c[i*n+j]=c[i*n+j]+a[k*n+j]*b[k*n+i];
  }
}
}