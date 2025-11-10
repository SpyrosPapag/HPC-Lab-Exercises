#include <stdio.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifndef WENOEPS
#define WENOEPS 1.e-6
#endif

#include "weno.h"

float * myalloc(const int NENTRIES, const int verbose )
{
	const int initialize = 1;
	enum { alignment_bytes = 64 } ; // changed the bytes to 64 from 32 for AVX-512
	float * tmp = NULL;
	
	const int result = posix_memalign((void **)&tmp, alignment_bytes, sizeof(float) * NENTRIES);
	assert(result == 0);
	
	if (initialize)
	{
		for(int i=0; i<NENTRIES; ++i)
			tmp[i] = drand48();

		if (verbose)
		{
			for(int i=0; i<NENTRIES; ++i)
				printf("tmp[%d] = %f\n", i, tmp[i]);
			printf("==============\n");
		}
	}
	return tmp;
}

double get_wtime()
{
	struct timeval t;
	gettimeofday(&t,  NULL);
	return t.tv_sec + t.tv_usec*1e-6;
}

void check_error(const double tol, float ref[], float val[], const int N)
{
	static const int verbose = 0;

	for(int i=0; i<N; ++i)
	{
		assert(!isnan(ref[i]));
		assert(!isnan(val[i]));

		const double err = ref[i] - val[i];
		const double relerr = err/fmaxf(FLT_EPSILON, fmaxf(fabs(val[i]), fabs(ref[i])));

		if (verbose) printf("+%1.1e,", relerr);

		if (fabs(relerr) >= tol && fabs(err) >= tol)
			printf("\n%d: %e %e -> %e %e\n", i, ref[i], val[i], err, relerr);

		assert(fabs(relerr) < tol || fabs(err) < tol);
	}

	if (verbose) printf("\t");
}

void benchmark(int argc, char *argv[], const int NENTRIES_, const int NTIMES, const int verbose, char *benchmark_name, const int optScenario)
{
	const int NENTRIES = 4 * (NENTRIES_ / 4);

	printf("nentries set to %e\n", (float)NENTRIES);
	
	float * const a = myalloc(NENTRIES, verbose);
	float * const b = myalloc(NENTRIES, verbose);
	float * const c = myalloc(NENTRIES, verbose);
	float * const d = myalloc(NENTRIES, verbose);
	float * const e = myalloc(NENTRIES, verbose);
	float * const f = myalloc(NENTRIES, verbose);
	float * const gold = myalloc(NENTRIES, verbose);
	float * const result = myalloc(NENTRIES, verbose);

	double t_start1 = get_wtime();
	weno_minus_reference(a, b, c, d, e, gold, NENTRIES, optScenario);
	double t_end1 = get_wtime();
	printf("\ttime for call 1: %e\n", t_end1 - t_start1);

	double t_start2 = get_wtime();
	weno_minus_reference(a, b, c, d, e, result, NENTRIES, optScenario);
	double t_end2 = get_wtime();
	printf("\ttime for call 2: %e\n", t_end2 - t_start2);

	const double tol = 1e-5;
	printf("minus: verifying accuracy with tolerance %.5e...", tol);
	check_error(tol, gold, result, NENTRIES);
	printf("passed!\n");

	free(a);
	free(b);
	free(c);
	free(d);
	free(e);
	free(gold);
	free(result);
}

int main (int argc, char *  argv[])
{
	printf("Hello, weno benchmark!\n");
	const int debug = 1;
	const int debugEntries = 32*32;//*32*32*32*5;
	const int verbose = 0;

	// Optimization Scenarios:
	// 0: no optimization
	// 1: only compiler options for auto vectorization
	// 2: only omp SIMD
	// 3: manual SIMD with AVX-512
	// DONT FORGET TO CHANGE COMPILER FLAGS IN THE MAKEFILE
	const int optScenario = 0;

	if (debug)
	{
		benchmark(argc, argv, debugEntries, 1, verbose, "debug", optScenario);
		return 0;
	}

	/* performance on cache hits */
	{
		const double desired_kb =  16 * 2 * 0.5; /* we want to fill 50% of the dcache */
		const int nentries =  1000 * 16 * (int)(pow(32 + 6, 2) * 4); //floor(desired_kb * 1024. / 7 / sizeof(float));
		const int ntimes = (int)floor(2. / (1e-7 * nentries));

		for(int i=0; i<4; ++i)
		{
			printf("*************** PEAK-LIKE BENCHMARK (RUN %d) **************************\n", i);
			benchmark(argc, argv, nentries, ntimes, 0, "cache", optScenario);
		}
	}
	/* performance on data streams */
	{
		const double desired_mb =  128 * 4;
		const int nentries =  (int)floor(desired_mb * 1024. * 1024. / 7 / sizeof(float));

		for(int i=0; i<4; ++i)
		{
			printf("*************** STREAM-LIKE BENCHMARK (RUN %d) **************************\n", i);
			benchmark(argc, argv, nentries, 1, 0, "stream", optScenario);
		}
	}

    return 0;
}
