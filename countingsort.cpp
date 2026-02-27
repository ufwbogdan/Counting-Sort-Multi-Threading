#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NPROC
#define NPROC 10
#endif

//A cache line is the unit of data transfer between the cache and main memory (needed for solving the task). Typically the cache line is 64 bytes.
#ifndef LEVEL1_DCACHE_LINESIZE
#define LEVEL1_DCACHE_LINESIZE 64 //bytes
#endif


#define CEILDIV(x,y) (((x)+(y)-1)/(y)) //return ceil(x/y)

// basically here we are computing the number of integers that fit in a cache line
// and also the padded size for an array of k integers
#define CLSIZE_INTS (LEVEL1_DCACHE_LINESIZE / sizeof(int))
#define PAD_SIZE(k) (CEILDIV((k), CLSIZE_INTS) * CLSIZE_INTS)

template <const int k> void seq_countingsort(int *out, int const *in, const int n) {
	int counters[k] = {}; // all zeros
	for (int i = 0; i < n; ++i)
		++counters[in[i]];
	int tmp, sum = 0;
	for (int i = 0; i < k; ++i) {
		tmp = counters[i];
		counters[i] = sum;
		sum += tmp;
	}
	for (int i = 0; i < n; ++i)
		out[counters[in[i]]++] = in[i];
}

template <const int k> void par_countingsort_padded(int *out, int const *in, const int n) {
	int padded_k = PAD_SIZE(k);
	alignas(LEVEL1_DCACHE_LINESIZE) int counters[NPROC][padded_k]; // aligined to cache line size

	for(int t = 0; t < NPROC; t++)
		for(int j = 0; j < padded_k; j++)
			counters[t][j] = 0;

	#pragma omp parallel num_threads(NPROC)
	{
		const int tid = omp_get_thread_num();
		#pragma omp for schedule(static)
		for(int i = 0; i < n; i++)
		{
			// initializing the input value
			int v = in[i];
			++counters[tid][v];
		}
	}

	int index = 0;
	for(int j = 0; j < k; ++j)
	{
		// here we are doing the prefix sum taking in account the padded size
		int col_total = 0;
		for(int t = 0; t < NPROC; ++t)
			col_total += counters[t][j];
		
		int offset = index;
		for(int t = 0; t < NPROC; ++t)
		{
			int cnt = counters[t][j];
			counters[t][j] = offset;
			offset += cnt;
		}
		index += col_total;
	}

	#pragma omp parallel num_threads(NPROC)
	{
		// here we are writing the output
		const int tid = omp_get_thread_num();
		#pragma omp for schedule(static)
		for(int i = 0; i < n; i++)
		{
			int v = in[i];
			int pos = counters[tid][v]++;
			out[pos] = v;
		}
	}
}
/*
	********************************************************************
	OBSERVATION BETWEEN THE TWO PARALLEL VERSIONS FOR DIFFERENT K VALUES:
	-> for small K values, for example 5 or 10, the padded version is much faster than the naive one, because in this case the false sharing is more impacting, in the 
	unpadded version multiple threads still being positioned on the same cache line, causing a lot of cache coherence (modifying and invalidating between the cores)
	-> once the k values get bigger, for example 100 or 500, the difference between the two is small to none, because in this case each thread's array is big enough and
	fits in multiple cache lines, so the false sharing effect for the unpadded version is not that visible anymore.
	********************************************************************
*/

template <const int k> void par_countingsort(int *out, int const *in, const int n) {
	// OLD NAIVE PARALLEL VERSION - FALSE CACHING 
	int counters[NPROC][k] = {}; // all zeros
	#pragma omp parallel num_threads(NPROC)
	{	
		int *thcounters = counters[omp_get_thread_num()];
		#pragma omp for
		for (int i = 0; i < n; ++i)
			++thcounters[in[i]];
		#pragma omp single
		{
			int tmp, sum = 0;
			for (int j = 0; j < k; ++j)
				for (int i = 0; i < NPROC; ++i) {
					tmp = counters[i][j];
					counters[i][j] = sum;
					sum += tmp;
				}
		}
		#pragma omp for
		for (int i = 0; i < n; ++i)
			out[thcounters[in[i]]++] = in[i];
	} 
}

bool checkreset(int *out, int const *in, const int n) {
	int insum = 0, outsum = 0, notsorted = 0;
	#pragma omp parallel for reduction(+:insum)
	for (int i = 0; i < n; ++i) insum += in[i];
	#pragma omp parallel for reduction(+:outsum)
	for (int i = 0; i < n; ++i) outsum += out[i];
	#pragma omp parallel for reduction(+:notsorted)
	for (int i = 1; i < n; ++i) notsorted += out[i-1]>out[i];
	if(insum!=outsum || notsorted) return false;
	#pragma omp parallel for
	for (int i = 0; i < n; ++i) out[i] = 0;
	return true;
}

#ifndef K
#define K 10
#endif

int main(int argc, char *argv[]) {

	//print some parameters
	printf("NPROC = %d\n", NPROC);
	printf("LEVEL1_DCACHE_LINESIZE = %d byte\n", LEVEL1_DCACHE_LINESIZE);
	printf("K = %d\n\n", K);

	//init input
	const int n = atoi(argv[1]);
	int* in = (int*)malloc(sizeof(int)*n);
	int* out = (int*)aligned_alloc(LEVEL1_DCACHE_LINESIZE, sizeof(int)*n);;
	for (int i = 0; i < n; ++i)
		in[i] = rand()%K;
	printf("n = %d\n", n);

	//tests
	double ts = omp_get_wtime();
	seq_countingsort<K>(out, in, n);
	ts = omp_get_wtime() - ts;

	printf("seq, elapsed time = %.3f seconds, check passed = %c\n", ts, checkreset(out,in,n)?'y':'n');
	
	double tp = omp_get_wtime();
	par_countingsort<K>(out, in, n);
	tp = omp_get_wtime() - tp;
	printf("par, elapsed time = %.3f seconds (%.1fx speedup), check passed = %c\n", tp, ts/tp, checkreset(out,in,n)?'y':'n');

	double tpp = omp_get_wtime();
	par_countingsort_padded<K>(out, in, n);
	tpp = omp_get_wtime() - tpp;
	printf("par padded, elapsed time = %.3f seconds (%.1fx speedup), check passed = %c\n", tpp, ts/tpp, checkreset(out,in,n)?'y':'n');

	//free mem
	free(in);
	free(out);

	return EXIT_SUCCESS;
}
