#include "simd.hpp"
#include <stdio.h>

#include <immintrin.h>
#include <limits>
#include <functional>
#include <cmath>
#include <thread>
#include "hiprec.hpp"
#include <future>
#include <vector>
#include "polynomial.hpp"

#include <complex>
#include <valarray>

class timer {
	double start_time;
	double time;
public:
	inline timer() {
		time = 0.0;
	}
	inline void stop() {
		struct timespec res;
		clock_gettime(CLOCK_MONOTONIC, &res);
		const double stop_time = res.tv_sec + res.tv_nsec / 1e9;
		time += stop_time - start_time;
	}
	inline void start() {
		struct timespec res;
		clock_gettime(CLOCK_MONOTONIC, &res);
		start_time = res.tv_sec + res.tv_nsec / 1e9;
	}
	inline void reset() {
		time = 0.0;
	}
	inline double read() {
		return time;
	}
};

double rand1() {
	return (rand() + 0.5) / RAND_MAX;
}

struct test_result_t {
	double avg_err;
	double max_err;
	double speed;
};

template<class T, class V, class F1, class F2>
test_result_t test_function(const F1& ref, const F2& test, T a, T b, bool relerr) {
	constexpr int N = 1 << 23;
	constexpr int size = V::size();
	static T* xref;
	static V* xtest;
	static T* yref;
	static V* ytest;
	if (xref == nullptr) {
		posix_memalign((void**) &xref, 32, sizeof(T) * size * N);
		posix_memalign((void**) &xtest, 32, sizeof(V) * N);
		posix_memalign((void**) &yref, 32, sizeof(T) * size * N);
		posix_memalign((void**) &ytest, 32, sizeof(V) * N);
	}
	int nthreads = 1;
	for (int i = 0; i < size * N; i++) {
		xref[i] = rand1() * ((double) b - (double) a) + (double) a;
	}
	for (int i = 0; i < size * N; i += size) {
		for (int j = 0; j < size; j++) {
			xtest[i / size][j] = xref[i + j];
		}
	}
	timer tref, ttest;
	std::vector<std::future<void>> futs;
	futs.reserve(nthreads);
	tref.start();
	for (int i = 0; i < nthreads; i++) {
		futs.push_back(std::async([&]() {
			for (int i = 0; i < size * N; i++) {
				yref[i] = ref(xref[i]);
			}
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
	tref.stop();
	futs.resize(0);
	ttest.start();
	for (int i = 0; i < nthreads; i++) {
		futs.push_back(std::async([&]() {
			for (int i = 0; i < N; i++) {
				ytest[i] = test(xtest[i]);
			}
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
	ttest.stop();
	double ref_time = tref.read();
	double test_time = ttest.read();
	double speed = ref_time / test_time;
	double err = 0.0;
	double max_err = 0.0;
	for (int i = 0; i < size * N; i += size) {
		for (int j = 0; j < size; j++) {
			T a = yref[i + j];
			T b = ytest[i / size][j];
			double this_err = fabs((double) a - (double) b);
			if (relerr) {
				this_err /= fabs((double) a);
			}
			err += this_err;
			//printf( "%e %e %e\n", xref[i + j], a, b);
			max_err = std::max((double) max_err, (double) this_err);
		}
	}
	err /= N * size;
	test_result_t result;
	result.speed = speed;
	result.max_err = max_err / std::numeric_limits<T>::epsilon();
	result.avg_err = err / std::numeric_limits<T>::epsilon();
	return result;
}

template<class T, class V, class F1, class F2>
test_result_t test_function(const F1& ref, const F2& test, T a0, T b0, T a1, T b1, bool relerr) {
	constexpr int N = 1 << 23;
	constexpr int size = V::size();
	static T* xref;
	static V* xtest;
	static T* yref;
	static V* ytest;
	static T* zref;
	static V* ztest;
	if (xref == nullptr) {
		posix_memalign((void**) &xref, 32, sizeof(T) * size * N);
		posix_memalign((void**) &xtest, 32, sizeof(V) * N);
		posix_memalign((void**) &yref, 32, sizeof(T) * size * N);
		posix_memalign((void**) &ytest, 32, sizeof(V) * N);
		posix_memalign((void**) &zref, 32, sizeof(T) * size * N);
		posix_memalign((void**) &ztest, 32, sizeof(V) * N);
	}
	int nthreads = 1;
	for (int i = 0; i < size * N; i++) {
		xref[i] = rand1() * ((double) b0 - (double) a0) + (double) a0;
		yref[i] = rand1() * ((double) b1 - (double) a1) + (double) a1;
	}
	for (int i = 0; i < size * N; i += size) {
		for (int j = 0; j < size; j++) {
			xtest[i / size][j] = xref[i + j];
			ytest[i / size][j] = yref[i + j];
		}
	}
	timer tref, ttest;
	std::vector<std::future<void>> futs;
	futs.reserve(nthreads);
	tref.start();
	for (int i = 0; i < nthreads; i++) {
		futs.push_back(std::async([&]() {
			for (int i = 0; i < size * N; i++) {
				zref[i] = ref(xref[i], yref[i]);
			}
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
	tref.stop();
	futs.resize(0);
	ttest.start();
	for (int i = 0; i < nthreads; i++) {
		futs.push_back(std::async([&]() {
			for (int i = 0; i < N; i++) {
				ztest[i] = test(xtest[i], ytest[i]);
			}
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
	ttest.stop();
	double ref_time = tref.read();
	double test_time = ttest.read();
	double speed = ref_time / test_time;
	double err = 0.0;
	double max_err = 0.0;
	for (int i = 0; i < size * N; i += size) {
		for (int j = 0; j < size; j++) {
			T a = zref[i + j];
			T b = ztest[i / size][j];
			double this_err = fabs((double) a - (double) b);
			if (relerr) {
				this_err /= fabs((double) a);
			}
			err += this_err;
			max_err = std::max((double) max_err, (double) this_err);
		}
	}
	err /= N * size;
	test_result_t result;
	result.speed = speed;
	result.max_err = max_err / std::numeric_limits<T>::epsilon();
	result.avg_err = err / std::numeric_limits<T>::epsilon();
	return result;
}

#define TEST1(type, vtype, name, Ref, Test, a, b, rel) \
{ \
	auto ref = [=](type x) { \
		return Ref(x); \
	}; \
	auto test = [=](vtype x) { \
		return Test(x); \
	}; \
	auto res = test_function<type, vtype, decltype(ref), decltype(test)>(ref, test, (a), (b), rel); \
	printf("%6s %12e %12e %12e\n", #name, res.speed, res.avg_err, res.max_err); \
}

#define TEST2(type, vtype, name, Ref, Test, a0, b0, a1, b1, rel) \
{ \
	auto ref = [=](type x, type y) { \
		return Ref(x, y); \
	}; \
	auto test = [=](vtype x, vtype y) { \
		return Test(x, y); \
	}; \
	auto res = test_function<type, vtype, decltype(ref), decltype(test)>(ref, test, (a0), (b0), (a1), (b1), rel); \
	printf("%6s %12e %12e %12e\n", #name, res.speed, res.avg_err, res.max_err); \
}

double erf_test(double x) {
	static constexpr double toler = std::numeric_limits<float>::epsilon() * 0.5;
	static constexpr int M = 4;
	static int N;
	static std::vector<double> coeffs[M];
	static const hiprec_real xmax(26);
	static std::once_flag flag;
	std::call_once(flag, []() {
		for( int m = 0; m < M; m++) {
			const hiprec_real a = hiprec_real(m) * xmax / hiprec_real(M);
			const hiprec_real b = hiprec_real(m + 1) * xmax / hiprec_real(M);
			std::function<hiprec_real(hiprec_real)> func = [a, b](hiprec_real x) {
				const hiprec_real half = hiprec_real(0.5);
				const hiprec_real z = half*(a + b + x * (b - a));
				return erfc(z);
			};
			coeffs[m] = ChebyCoeffs(func, toler);
			N = std::max((int) coeffs[m].size(), N);
			printf( "N = %i\n", N);
		}
	});
	for (int m = 0; m < M; m++) {
		coeffs[m].resize(N, 0.0);
	}
	const int m = x / xmax * M;
	x -= (m + 0.5) * xmax / M;
	x *= 2.0 * M / xmax;
	//printf( "%i %e %i\n", m, x, N);
	double y = coeffs[m][N - 1];
	for (int n = N - 2; n >= 0; n--) {
		y = fma(y, x, coeffs[m][n]);
	}
	return y;
}

static double t000;

namespace simd {

/*
simd_f32 exp(simd_f32 x) {
	simd_f32 x0, y, zero;
	simd_i32 i;
	x = max(simd_f32(-87), min(simd_f32(87), x));
	x0 = round(x * simd_f32(1.44269504088896338700e+00));
	x -= x0 * simd_f32(6.93147180559945286227e-01);
	y = simd_f32(2.48015873015873015658e-05);
	y = fma(x, y, simd_f32(1.98412698412698413e-04));
	y = fma(x, y, simd_f32(1.38888888888888894e-03));
	y = fma(x, y, simd_f32(8.33333333333333322e-03));
	y = fma(x, y, simd_f32(4.16666666666666644e-02));
	y = fma(x, y, simd_f32(1.66666666666666657e-01));
	y = fma(x, y, simd_f32(5.00000000000000000e-01));
	y = fma(x, y, simd_f32(1.00000000000000000e+00));
	y = fma(x, y, simd_f32(1.00000000000000000e+00));
	i = (simd_i32(x0) + simd_i32(127)) << int(23);
	y *= (simd_f32&) i;
	return y;
}
*/
double exp_test(double x1) {
	constexpr int N = 6;
	constexpr int M = 64;
	static std::once_flag once;
	static double coeff[N];
	static double base[M*2028];
	std::call_once(once, [](){
		int fac = 1;
		for( int n = 0; n < M * 2028; n++) {
			int e = n - M * 1023;
			base[n] = pow(hiprec_real(2), hiprec_real(e)/hiprec_real(M));
		}
		for( int n = 0; n < N; n++) {
			coeff[n] = hiprec_real(1) / hiprec_real(fac);
			fac *= n + 1;
		}
	});
	double x = x1;
	double x0 = std::round(x * (hiprec_real(M) / hiprec_real(M_LN2)));
	x -= x0 * (hiprec_real(M_LN2) / hiprec_real(M));

	double y = coeff[N-1];
	for( int n = N - 2; n >= 0; n--) {
		y = std::fma(x, y, coeff[n]);
	}
	y *= base[(int)(x0) + M * 1023];
	return y * (1 - 3.333333e-17 * x1);
}

}


int main() {
	using namespace simd;
	FILE* fp = fopen("test.txt", "wt");
	double max_err = 0.0;
	for (double r = -26.5; r < 26.5; r += .02) {
		double a = erfc(simd_f64(r))[0];
		double b = erfc(r);
		max_err = std::max(max_err, fabs((a - b) / a));
		fprintf(fp, "%.10e %.10e %.10e %.10e\n", r, a, b, (a - b) / a);
	}
	printf("%e %e\n", max_err / std::numeric_limits<double>::epsilon(), t000);
	fclose(fp);
//	return 0;
	for (double x = 1.0; x < 2.0; x += 0.01) {
//		printf("%e %e %e %e\n", x, gamma(x), std::tgamma(x), (gamma(x) -std::tgamma(x))/std::tgamma(x));
	}
	srand(1234);
	for (double x = -4.0; x < 4.0; x += 0.1) {
		//	printf( "%e %e %e\n", x, atan(simd_f32(x))[0], std::atan(x));
	}
	const auto cvt32_test = [](simd_f32 x) {
		simd_i32 i = simd_i32(x);
		auto y = simd_f32(i);
		return y;
	};
	const auto cvt32_ref = [](float x) {
		int i = int(x);
		return float(i);
	};

	const auto cvt64_test = [](simd_f64 x) {
		simd_i64 i = simd_i64(x);
		simd_f64 y = simd_f64(i);
		//printf( "--- %e %lli %e\n", x[0], i[0], y[0]);
			return y;
		};
	const auto cvt64_ref = [](double x) {
		int i = (long long)(x);
		return double(i);
	};

	TEST1(double, simd_f64, exp2, exp2, exp2, -1000.0, 1000.0, true);
	TEST1(double, simd_f64, exp, exp, exp, -600.0, 600.0, true);
	TEST1(double, simd_f64, erfc, erfc, erfc, -25.0, 25.0, true);
	printf("Testing SIMD Functions\n");
	printf("\nSingle Precision\n");
	printf("name   speed        avg err      max err\n");

	TEST1(float, simd_f32, erfc, erfcf, erfc, -8.9, 8.9, true);
	TEST1(float, simd_f32, asinh, asinhf, asinh, .001, 10, true);
	TEST1(float, simd_f32, acosh, acoshf, acosh, 1.001, 10.0, true);
	TEST1(float, simd_f32, atanh, atanhf, atanh, 0.001, 0.999, true);
	TEST1(float, simd_f32, expm1, expm1f, expm1, -2.0, 2.0, true);
	TEST1(float, simd_f32, cbrt, cbrtf, cbrt, 1.0 / 4000, 4000, true);
	TEST1(float, simd_f32, erf, erff, erf, -7, 7, true);
	TEST1(float, simd_f32, tgamma, tgammaf, tgamma, -.99, -0.01, true);
	TEST1(float, simd_f32, cosh, coshf, cosh, -10.0, 10.0, true);
	TEST1(float, simd_f32, sinh, sinhf, sinh, 0.01, 10.0, true);
	TEST1(float, simd_f32, tanh, tanhf, tanh, 0.01, 10.0, true);
	TEST1(float, simd_f32, atan, atanf, atan, -4.0, 4.0, false);
	TEST1(float, simd_f32, acos, acosf, acos, -0.999, 0.999, false);
	TEST1(float, simd_f32, asin, asinf, asin, -0.999, 0.999, false);
	TEST2(float, simd_f32, pow, powf, pow, 1e-3, 1e3, .01, 10, true);
	TEST1(float, simd_f32, log, logf, log, exp(-1), exp(40), true);
	TEST1(float, simd_f32, sqrt, sqrtf, sqrt, 0, std::numeric_limits<int>::max(), true);
	TEST1(float, simd_f32, cos, cosf, cos, -2.0 * M_PI, 2.0 * M_PI, false);
	TEST1(float, simd_f32, sin, sinf, sin, -2.0 * M_PI, 2.0 * M_PI, false);
	TEST1(float, simd_f32, tan, tanf, tan, -2.0 * M_PI, 2.0 * M_PI, true);
	TEST1(float, simd_f32, exp, expf, exp, -86.0, 86.0, true);
	TEST1(float, simd_f32, cvt, cvt32_ref, cvt32_test, 1, +10000000, true);

	printf("\nDouble Precision\n");
	printf("name   speed        avg err      max err\n");
	TEST1(double, simd_f64, acosh, acosh, acosh, 1.001, 10.0, true);
	TEST1(double, simd_f64, asinh, asinh, asinh, .001, 10, true);
	TEST1(double, simd_f64, atanh, atanh, atanh, 0.001, 0.999, true);
	TEST1(double, simd_f64, expm1, expm1, expm1, -2.0, 2.0, true);
	TEST1(double, simd_f64, cbrt, cbrt, cbrt, 1.0 / 4000, 4000, true);
	TEST1(double, simd_f64, erf, erf, erf, -9, 9, true);
	TEST1(double, simd_f64, tgamma, tgamma, tgamma, -.99, -0.01, true);
	TEST1(double, simd_f64, cosh, cosh, cosh, -10.0, 10.0, true);
	TEST1(double, simd_f64, sinh, sinh, sinh, 0.01, 10.0, true);
	TEST1(double, simd_f64, tanh, tanh, tanh, 0.01, 10.0, true);
	TEST1(double, simd_f64, atan, atan, atan, -4.0, 4.0, false);
	TEST1(double, simd_f64, acos, acos, acos, -0.999, 0.999, false);
	TEST1(double, simd_f64, asin, asin, asin, -0.999, 0.999, false);
	TEST2(double, simd_f64, pow, pow, pow, 1e-3, 1e3, .01, 10, true);
	TEST1(double, simd_f64, log, log, log, exp(-1), exp(40), true);
	TEST1(double, simd_f64, sqrt, sqrt, sqrt, 0, std::numeric_limits<long long>::max(), true);
	TEST1(double, simd_f64, cos, cos, cos, -2.0 * M_PI, 2.0 * M_PI, false);
	TEST1(double, simd_f64, sin, sin, sin, -2.0 * M_PI, 2.0 * M_PI, false);
	TEST1(double, simd_f64, tan, tan, tan, -2.0 * M_PI, 2.0 * M_PI, true);
	TEST1(double, simd_f64, cvt, cvt64_ref, cvt64_test, 1LL, +1000000000LL, true);

	return (0);
}

