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
	constexpr int N = 1 << 19;
	constexpr int size = V::size();
	static T xref[size * N];
	static V xtest[N];
	static T yref[size * N];
	static V ytest[N];
	int nthreads = std::thread::hardware_concurrency();
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
			for (int i = 0; i < size * N; i++) {
				yref[i] = ref(xref[i]);
			}
			for (int i = 0; i < size * N; i++) {
				yref[i] = ref(xref[i]);
			}
			for (int i = 0; i < size * N; i++) {
				yref[i] = ref(xref[i]);
			}
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
			for (int i = 0; i < N; i++) {
				ytest[i] = test(xtest[i]);
			}
			for (int i = 0; i < N; i++) {
				ytest[i] = test(xtest[i]);
			}
			for (int i = 0; i < N; i++) {
				ytest[i] = test(xtest[i]);
			}
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
			//	printf( "%e %e %e\n", xref[i + j], a, b);
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

hiprec_real Tn(int n, hiprec_real x) {
	if (n == 0) {
		return hiprec_real(1);
	} else if (n == 1) {
		return x;
	} else {
		static const hiprec_real two = hiprec_real(2);
		hiprec_real Tnm2 = hiprec_real(1);
		hiprec_real Tnm1 = x;
		hiprec_real T;
		for (int m = 2; m <= n; m++) {
			T = two * x * Tnm1 - Tnm2;
			Tnm2 = Tnm1;
			Tnm1 = T;
		}
		return T;
	}
}

template<class F, class T>
T integrate_(F&& f, T a, T b, int N) {
	static const T p0 = (T(1) - sqrt(T(3) / T(5))) / T(2);
	static const T p1 = T(1) / T(2);
	static const T p2 = (T(1) + sqrt(T(3) / T(5))) / T(2);
	static const T w0 = T(5) / T(18);
	static const T w1 = T(4) / T(9);
	static const T w2 = T(5) / T(18);
	const T dx = (b - a) / T(N);
	T sum = T(0);
	const T a0 = dx * p0;
	const T a1 = dx * p1;
	const T a2 = dx * p2;
	const T b0 = dx * w0;
	const T b1 = dx * w1;
	const T b2 = dx * w2;
	for (int i = 0; i < N; i++) {
		const T x = a + T(i) * dx;
		const T x0 = x + a0;
		const T x1 = x + a1;
		const T x2 = x + a2;
		const T y0 = f(x0);
		const T y1 = f(x1);
		const T y2 = f(x2);
		const T y = y0 * b0 + y1 * b1 + y2 * b2;
		sum += y;
	}
	return sum;
}

template<class F, class T>
T integrate(F&& f, T a, T b, T toler = 1e-1) {
	constexpr int N = 2;
	const static T _14 = T(T(1) / T(4));
	const static T _24 = T(T(2) / T(4));
	const static T _34 = T(T(3) / T(4));
	const T p0 = (a * _34 + b * _14);
	const T p1 = (a + b) * _24;
	const T p2 = (a * _14 + b * _34);
	const T dx = b - a;
	const T A = integrate_(f, a, p1, 2 * N);
	const T B = integrate_(f, p1, b, 2 * N);
	const T C = integrate_(f, a, b, N);
	const T D = A + B;
	if ((double) (abs(C - D) - abs(toler)) < 0.0) {
		return D;
	} else {
		return integrate(f, a, p1, toler) + integrate(f, p1, b, toler);
	}
}

int main() {
	using namespace simd;
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

	printf("Testing SIMD Functions\n");
	printf("\nSingle Precision\n");
	printf("name   speed        avg err      max err\n");
	TEST1(float, simd_f32, log2, log2f, log2, exp(-1), exp(40), true);
	TEST1(float, simd_f32, sqrt, sqrtf, sqrt, 0, std::numeric_limits<int>::max(), true);
	TEST1(float, simd_f32, cos, cosf, cos, -2.0*M_PI, 2.0*M_PI, false);
	TEST1(float, simd_f32, sin, sinf, sin, -2.0*M_PI, 2.0*M_PI, false);
	TEST1(float, simd_f32, tan, tanf, tan, -2.0*M_PI, 2.0*M_PI, true);
	TEST1(float, simd_f32, exp, expf, exp, -86.0, 86.0, true);
	TEST1(float, simd_f32, erfc, erfcf, erfc, -9.0, 9.0, true);
	TEST1(float, simd_f32, cvt, cvt32_ref, cvt32_test, 1, +10000000, true);

	printf("\nDouble Precision\n");
	printf("name   speed        avg err      max err\n");
	TEST1(double, simd_f64, log2, log2, log2, exp(-1), exp(40), true);
	TEST1(double, simd_f64, sqrt, sqrt, sqrt, 0, std::numeric_limits<long long>::max(), true);
	TEST1(double, simd_f64, cos, cos, cos, -2.0*M_PI, 2.0*M_PI, false);
	TEST1(double, simd_f64, sin, sin, sin, -2.0*M_PI, 2.0*M_PI, false);
	TEST1(double, simd_f64, tan, tan, tan, -2.0*M_PI, 2.0*M_PI, true);
	TEST1(double, simd_f64, exp, exp, exp, -700.0, 700.0, true);
	TEST1(double, simd_f64, erfc, erfc, erfc, -26.0, 26.0, true);
	TEST1(double, simd_f64, cvt, cvt64_ref, cvt64_test, 1LL, +1000000000LL, true);

	return (0);
}

