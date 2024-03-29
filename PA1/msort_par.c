#include "omp.h"
void msort_seq(int a[],int b[],int lo, int hi);
void merge(int a[],int b[],int lo,int mid,int hi);

void msort_par(int a[],int b[],int lo, int hi)
{
#pragma omp parallel
{
	#pragma omp single
	{
		int temp,mid;
		if (lo < hi) { 
			if (hi == lo+1) { 
				if (a[hi]<a[lo]) { temp=a[hi]; a[hi]=a[lo]; a[lo]=temp; }
			}
			else {     
				mid = (lo+hi)/2;
				#pragma omp task firstprivate (a, b, lo, mid)
				msort_seq(a,b,lo,mid);
				#pragma omp task firstprivate (a, b, mid, hi)
				msort_seq(a,b,mid+1,hi);
				#pragma omp taskwait	
				merge(a,b,lo,mid,hi);
			}
		}
	}
}

}

