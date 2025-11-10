#pragma once
#include <stdio.h>
#include <omp.h>
#include <immintrin.h>

float weno_minus_core(const float a, const float b, const float c, const float d, const float e)
{
	const float is0 = a*(a*(float)(4./3.)  - b*(float)(19./3.)  + c*(float)(11./3.)) + b*(b*(float)(25./3.)  - c*(float)(31./3.)) + c*c*(float)(10./3.);
	const float is1 = b*(b*(float)(4./3.)  - c*(float)(13./3.)  + d*(float)(5./3.))  + c*(c*(float)(13./3.)  - d*(float)(13./3.)) + d*d*(float)(4./3.);
	const float is2 = c*(c*(float)(10./3.) - d*(float)(31./3.)  + e*(float)(11./3.)) + d*(d*(float)(25./3.)  - e*(float)(19./3.)) + e*e*(float)(4./3.);

	const float is0plus = is0 + (float)WENOEPS;
	const float is1plus = is1 + (float)WENOEPS;
	const float is2plus = is2 + (float)WENOEPS;

	const float alpha0 = (float)(0.1)*((float)1/(is0plus*is0plus));
	const float alpha1 = (float)(0.6)*((float)1/(is1plus*is1plus));
	const float alpha2 = (float)(0.3)*((float)1/(is2plus*is2plus));
	const float alphasum = alpha0+alpha1+alpha2;
	const float inv_alpha = ((float)1)/alphasum;

	const float omega0 = alpha0 * inv_alpha;
	const float omega1 = alpha1 * inv_alpha;
	const float omega2 = 1-omega0-omega1;

	return omega0*((float)(1.0/3.)*a - (float)(7./6.)*b + (float)(11./6.)*c) +
		   omega1*(-(float)(1./6.)*b + (float)(5./6.)*c + (float)(1./3.)*d) +
		   omega2*((float)(1./3.)*c  + (float)(5./6.)*d - (float)(1./6.)*e);
}

// if compiled for avx only then use this function
// otherwise error will be thrown
#ifdef __AVX512F__
float* weno_minus_core_avx512(const float* a, const float* b, const float* c, const float* d, const float* e, float* out, const int NENTRIES)
{
	// make constants into SIMD vectors 
	__m512 _4_3 = _mm512_set1_ps(4.0f/3.0f);
	__m512 neg_19_3 = _mm512_set1_ps(-19.0f/3.0f);
	__m512 _11_3 = _mm512_set1_ps(11.0f/3.0f);
	__m512 _25_3 = _mm512_set1_ps(25.0f/3.0f);
	__m512 neg_31_3 = _mm512_set1_ps(-31.0f/3.0f);
	__m512 _10_3 = _mm512_set1_ps(10.0f/3.0f);
	__m512 _13_3 = _mm512_set1_ps(13.0f/3.0f);
	__m512 neg_13_3 = _mm512_set1_ps(-13.0f/3.0f); // NEED NEGATIVE VERSION FOR SUBTRACTION
	__m512 _5_3 = _mm512_set1_ps(5.0f/3.0f);
	__m512 _01 = _mm512_set1_ps(0.1f);
	__m512 _06 = _mm512_set1_ps(0.6f);
	__m512 _03 = _mm512_set1_ps(0.3f);
	__m512 _1_3 = _mm512_set1_ps(1.0f/3.0f);
	__m512 neg_7_6 = _mm512_set1_ps(-7.0f/6.0f);
	__m512 _11_6 = _mm512_set1_ps(11.0f/6.0f);
	__m512 neg_1_6 = _mm512_set1_ps(-1.0f/6.0f);
	__m512 _5_6 = _mm512_set1_ps(5.0f/6.0f);
	__m512 _WENOEPS = _mm512_set1_ps((float)WENOEPS);

	// calculate out using SIMD vectors 16 elements at a time
	float mod = NENTRIES % 16;
	
	int i = 0;
	for(; i < NENTRIES - mod; i+=16)
	{
		// load input values into SIMD vectors
		__m512 a_vec = _mm512_loadu_ps(&a[i]);
		__m512 b_vec = _mm512_loadu_ps(&b[i]);
		__m512 c_vec = _mm512_loadu_ps(&c[i]);
		__m512 d_vec = _mm512_loadu_ps(&d[i]);
		__m512 e_vec = _mm512_loadu_ps(&e[i]);

		// calculate smoothness indicators
		__m512 is0 = _mm512_fmadd_ps(a_vec, _mm512_fmadd_ps(a_vec, _4_3,_mm512_fmadd_ps(b_vec,neg_19_3,_mm512_mul_ps(c_vec,_11_3))), _mm512_fmadd_ps(b_vec,_mm512_fmadd_ps(b_vec,_25_3,_mm512_mul_ps(c_vec,neg_31_3)),_mm512_mul_ps(c_vec,_mm512_mul_ps(c_vec,_10_3))));
		__m512 is1 = _mm512_fmadd_ps(b_vec, _mm512_fmadd_ps(b_vec, _4_3,_mm512_fmadd_ps(c_vec,neg_13_3,_mm512_mul_ps(d_vec,_5_3))),  _mm512_fmadd_ps(c_vec,_mm512_fmadd_ps(c_vec,_13_3,_mm512_mul_ps(d_vec,neg_13_3)),_mm512_mul_ps(d_vec,_mm512_mul_ps(d_vec,_4_3))));
		__m512 is2 = _mm512_fmadd_ps(c_vec, _mm512_fmadd_ps(c_vec, _10_3,_mm512_fmadd_ps(d_vec,neg_31_3,_mm512_mul_ps(e_vec,_11_3))),_mm512_fmadd_ps(d_vec,_mm512_fmadd_ps(d_vec,_25_3,_mm512_mul_ps(e_vec,neg_19_3)),_mm512_mul_ps(e_vec,_mm512_mul_ps(e_vec,_4_3))));
		
		// normalize smoothness indicators
		__m512 is0plus = _mm512_add_ps(is0, _WENOEPS);
		__m512 is1plus = _mm512_add_ps(is1, _WENOEPS);
		__m512 is2plus = _mm512_add_ps(is2, _WENOEPS);

		// calculate nonlinear weights
		__m512 alpha0 = _mm512_mul_ps(_01,_mm512_rcp14_ps(_mm512_mul_ps(is0plus,is0plus)));
		__m512 alpha1 = _mm512_mul_ps(_06,_mm512_rcp14_ps(_mm512_mul_ps(is1plus,is1plus)));
		__m512 alpha2 = _mm512_mul_ps(_03,_mm512_rcp14_ps(_mm512_mul_ps(is2plus,is2plus)));

		// normalize nonlinear weights
		__m512 alphasum = _mm512_add_ps(alpha0,_mm512_add_ps(alpha1,alpha2));
		__m512 inv_alpha = _mm512_rcp14_ps(alphasum);

		__m512 omega0 = _mm512_mul_ps(alpha0, inv_alpha);
		__m512 omega1 = _mm512_mul_ps(alpha1, inv_alpha);
		__m512 omega2 = _mm512_sub_ps(_mm512_set1_ps(1.0f),_mm512_add_ps(omega0,omega1));

		// calculate result
		__m512 temp0 = _mm512_fmadd_ps(_1_3,a_vec,_mm512_fmadd_ps(neg_7_6,b_vec,_mm512_mul_ps(_11_6,c_vec)));
		__m512 temp1 = _mm512_fmadd_ps(neg_1_6,b_vec,_mm512_fmadd_ps(_5_6,c_vec,_mm512_mul_ps(_1_3,d_vec)));
		__m512 temp2 = _mm512_fmadd_ps(_1_3,c_vec,_mm512_fmadd_ps(_5_6,d_vec,_mm512_mul_ps(neg_1_6,e_vec)));

		__m512 res = _mm512_fmadd_ps(omega0,temp0,_mm512_fmadd_ps(omega1,temp1,_mm512_mul_ps(omega2,temp2)));
		
		// store the result in out
		_mm512_store_ps(out + i, res);
	}
	// remaining values (at most 15)
	// i stops at NENTRIES - mod in the previous for
	for(; i<NENTRIES; ++i)
	{
		out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
	}

	return out;
}
#endif

void weno_minus_reference(const float * restrict a, const float * restrict b, const float * restrict c,
			  const float * restrict d, const float * restrict e, float * restrict out,
			  const int NENTRIES, const int optScenario)
{
	switch(optScenario)
	{
		// no optimization
		case 0: printf("using scalar core without optimizations\n");
				for (int i=0; i<NENTRIES; ++i)
					out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
				break;
	
		// compiler options
		case 1: printf("using scalar core with compiler options\n");
				for (int i=0; i<NENTRIES; ++i)
					out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
				break;
		
		// omp SIMD
		case 2: printf("using scalar core with OpenMP SIMD\n");
				#pragma omp simd
				for (int i=0; i<NENTRIES; ++i)
					out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
				break;

		#ifdef __AVX512F__
		// SIMD with AVX-512
		case 3: if(NENTRIES < 16)
				{
					printf("using scalar core with avx compiler options because NENTRIES<16\n");
					for (int i=0; i<NENTRIES; ++i)
						out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
				}
				else
				{
					printf("using AVX512 core\n");
					weno_minus_core_avx512(a, b, c, d, e, out, NENTRIES);
				}
				break;
		#endif
	}
}


