#include "simd.hpp"
#include <stdio.h>

#include <immintrin.h>
#include <limits>
#include <functional>
#include <cmath>
#include <thread>
#include <fenv.h>
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
	bool flag = true;
	if (xref == nullptr) {
		flag = (posix_memalign((void**) &xref, 32, sizeof(T) * size * N) != 0) && flag;
		flag = (posix_memalign((void**) &xtest, 32, sizeof(V) * N) != 0) && flag;
		flag = (posix_memalign((void**) &yref, 32, sizeof(T) * size * N) != 0) && flag;
		flag = (posix_memalign((void**) &ytest, 32, sizeof(V) * N) != 0) && flag;
	}

	int nthreads = 1;
	for (int i = 0; i < size * N; i++) {
		do {
			xref[i] = rand1() * ((double) b - (double) a) + (double) a;
		} while (xref[i] == 0.0);
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
				this_err /= fabs((double) a + 1e-300);
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
	bool flag = true;
	if (xref == nullptr) {
		flag = (posix_memalign((void**) &xref, 32, sizeof(T) * size * N) != 0) && flag;
		flag = (posix_memalign((void**) &xtest, 32, sizeof(V) * N) != 0) && flag;
		flag = (posix_memalign((void**) &yref, 32, sizeof(T) * size * N) != 0) && flag;
		flag = (posix_memalign((void**) &ytest, 32, sizeof(V) * N) != 0) && flag;
		flag = (posix_memalign((void**) &zref, 32, sizeof(T) * size * N) != 0) && flag;
		flag = (posix_memalign((void**) &ztest, 32, sizeof(V) * N) != 0) && flag;
	}
	int nthreads = 1;
	for (int i = 0; i < size * N; i++) {
		do {
			xref[i] = rand1() * ((double) b0 - (double) a0) + (double) a0;
		} while (xref[i] == 0.0);
		do {
			yref[i] = rand1() * ((double) b1 - (double) a1) + (double) a1;
		} while (yref[i] == 0.0);
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

namespace simd {
}

void cordic(double theta, double& sintheta, double& costheta) {
	constexpr int N = 28;
	static double dphi1[N];
	static double dphi2[N];
	static double dphi3[N];
	static double taneps[N];
	static double twopiinv;
	static double factor;
	static std::once_flag once;
	static double dtheta1;
	static double dtheta2;
	static double dtheta3;
	static hiprec_real pi_exact = hiprec_real(4) * atan(hiprec_real(1));
	static double dbin = pi_exact * pow(hiprec_real(2), -hiprec_real(N));
	std::call_once(once, []() {
		hiprec_real factor_ = 1.0;
		hiprec_real dtheta0 = hiprec_real(2) * pi_exact;
		dtheta1 = dtheta0;
		dtheta2 = dtheta0 - hiprec_real(dtheta1);
		dtheta3 = dtheta0 - hiprec_real(dtheta1) - hiprec_real(dtheta2);
		for( int n = 0; n < N; n++) {
			hiprec_real eps = pi_exact * pow(hiprec_real(2), -hiprec_real(n));
			taneps[n] = tan(eps);
			hiprec_real phi0 = eps;
			dphi1[n] = phi0;
			dphi2[n] = phi0 - hiprec_real(dphi1[n]);
			dphi3[n] = phi0 - hiprec_real(dphi1[n]) - hiprec_real(dphi2[n]);
			if( n != 1 ) {
				factor_ *= hiprec_real(1) / sqrt(hiprec_real(1) + tan(eps) * tan(eps));
			}
		}
		factor = factor_;
	});
	int index = std::round(theta / dtheta1);
	theta = std::fma(double(-index), dtheta3, std::fma(double(-index), dtheta2, std::fma(double(-index), dtheta1, theta)));
	double x = 1.0;
	double y = 0.0;
	double phi1 = 0.0;
	double phi2 = 0.0;
	double phi3 = 0.0;
	for (int n = 0; n < N; n++) {
		double eps = taneps[n];
		double x0 = x;
		double y0 = y;
		if (phi3 + phi2 + phi1 + dphi1[n] / 2.0 < theta) {
			if (n == 1) {
				std::swap(x, y);
				y = -y;
			} else {
				x = std::fma(eps, -y0, x);
				y = std::fma(eps, x0, y);
			}
			phi1 += dphi1[n];
			phi2 += dphi2[n];
			phi3 += dphi3[n];
		} else if (phi3 + phi2 + phi1 - dphi1[n] / 2.0 > theta) {
			if (n == 1) {
				std::swap(x, y);
				x = -x;
			} else {
				x = std::fma(eps, y0, x);
				y = std::fma(eps, -x0, y);
			}
			phi1 -= dphi1[n];
			phi2 -= dphi2[n];
			phi3 -= dphi3[n];
		}
	}
	x = (double) x * factor;
	y = (double) y * factor;
	double eps = ((theta - phi1) - phi2) - phi3;
	costheta = x;
	sintheta = y;
	sintheta = std::fma(eps, x, sintheta);
	costheta = std::fma(eps, -y, costheta);
}

double cos_test(double theta) {
	constexpr int N = 11;
	static std::once_flag once;
	static double coeff[N];
	static hiprec_real pi_exact = hiprec_real(4) * atan(hiprec_real(1));
	static double pi1 = pi_exact;
	static double pi2 = pi_exact - hiprec_real(pi1);
	std::call_once(once, []() {
		hiprec_real fac(-1);
		for( int n = 0; n < N; n++) {
			coeff[n] = hiprec_real(1) / fac;
			fac *= hiprec_real(2 * n + 3);
			fac *= -hiprec_real(2 * n + 2);
		}
	});
	theta = abs(theta);
	int i = std::floor(theta / M_PI_2);
	i |= 1;
	int sgn = -2 * (i / 2 - 2 * (i / 4)) + 1;
	double x = sgn * std::fma(-double(i), pi2 * 0.5, std::fma(-double(i), pi1 * 0.5, theta));
	double x2 = x * x;
	double y = coeff[N - 1];
	for (int n = N - 2; n >= 0; n--) {
		y = std::fma(x2, y, coeff[n]);
	}
	y *= x;
	return y;
}

float acos_test(float x) {
	constexpr int N = 18;
	constexpr float coeffs[] = { 0, 2., 0.3333333333, 0.08888888889, 0.02857142857, 0.01015873016, 0.003848003848, 0.001522287237, 0.0006216006216,
			0.0002600159463, 0.0001108489034, 0.00004798653828, 0.00002103757656, 9.321264693e-6, 4.167443738e-6, 1.877744765e-6, 8.517995405e-7, 3.886999686e-7 };
	float sgn = copysign(1.0f, x);
	x = abs(x);
	x = 1.f - x;
	float y = coeffs[N - 1];
	for (int n = N - 2; n >= 0; n--) {
		y = std::fma(x, y, coeffs[n]);
	}
	y = sqrt(std::max(y, 0.f));
	long double exact_pi = 4.0L * atanl(1.0L);
	float pi1 = exact_pi;
	float pi2 = exact_pi - (long double) pi1;
	if (sgn == -1.f) {
		y = pi1 - y;
		y += pi2;
	}
	return y;
}

float log2_test(float x) {
	constexpr int N = 5;
	int i = std::ilogbf(x * sqrtf(2));
	x = ldexp(x, -i);
	printf("%e\n", x - 1.0);
	float y = (x - 1.0) / (x + 1.0);
	float y2 = y * y;
	float z = 2.0 / (2 * (N - 1) + 1);
	for (int n = N - 2; n >= 0; n--) {
		z = fma(z, y2, 2.0 / (2 * n + 1) / log(2));
	}
	z *= y;
	return z + i;
}

hiprec_real factorial(int n) {
	if (n < 2) {
		return hiprec_real(1);
	} else {
		return hiprec_real(n) * factorial(n - 1);
	}
}

float pow_test(float x, float y) {
	constexpr int Nlogbits = 11;
	constexpr int Ntable = 1 << Nlogbits;
	static float log2hi[Ntable];
	static float log2lo[Ntable];
	static bool init = false;
	float lomax = 0.0;
	if (!init) {
		for (int n = 0; n < Ntable; n++) {
			int i = (n << (23 - Nlogbits)) | (127 << 23);
			float a = (float&) i;
			log2hi[n] = log2(hiprec_real(a));
			log2lo[n] = log2(hiprec_real(a)) - hiprec_real(log2hi[n]);
			lomax = std::max(lomax, (float) fabs(log2lo[n]));
		}
		init = true;
	}
	int i = (int&) x;
	int xi = (i >> 23) - 127;
	int index = (i & 0x7FF000) >> (23 - Nlogbits);
	//printf("%i\n", index);
	int j = (i & 0x7FF000) | (127 << 23);
	int k = (i & 0x7FFFFF) | (127 << 23);
	float a = (float&) j;
	float b = (float&) k;
	float eps = (b - a) / a;
	float z = -0.25f;
	z = fmaf(z, eps, 0.5f);
	z *= eps;
	float z2 = z * z;
	//= (4.121985831e-01);
	float log1peps = 2.885390082e+00;
	log1peps *= z;
	float loghi = log2hi[index];
	float loglo = log2lo[index];
	//loghi += log1peps;
	float arg1 = y * loghi;
	float arg1err = fmaf(y, loghi, -arg1);
	float arg2 = y * loglo;
	float pwr = exp2f(y);
	float theta = 1.f;
	float err = 0.0;
	for (int i = 0; i < 7; i++) {
		if (xi & 1) {
			theta *= pwr;
		}
		pwr *= pwr;
		xi >>= 1;
	}
//	printf( "%e %e %e \n", log2hi[index],log2lo[index], log1peps);
	//theta += err;
	x = logf(2) * (y * log1peps + (arg2 + arg1err));
	double p = 1.0f / 6.0f;
	p = fmaf(p, x, 0.5f);
	p = fmaf(p, x, 1.0f);
	p = fmaf(p, x, 1.0f);
	z = exp2f(arg1) * theta * p;
	return z;
}

struct double2 {
	double x;
	double y;
	double2 operator*(double2 other) const {
		double2 res;
		res.x = x * other.x;
		res.y = y * other.y + fma(x, other.x, -res.x);
		res.y += x * other.y + y * other.x;
		return res;
	}
};
struct double3 {
	double x, y, z;
};

double_2 sqr(double x) {
	double_2 p;
	p.x = x * x;
	p.y = fma(x, x, -p.x);
	return p;
}

double_2 sqrt(double_2 A) {
	double xn = 1.0 / sqrt(A.x);
	double yn = A.x * xn;
	double_2 ynsqr = sqr(yn);
	double diff = (A - ynsqr).x;
	double_2 prod;
	prod.x = xn * diff;
	prod.y = fma(xn, diff, -prod.x);
	prod.x *= 0.5;
	prod.y *= 0.5;
	prod.x += yn;
	return double_2::quick_two_sum(prod.x, prod.y);
}

namespace simd {

simd_f64 pow_precise(simd_f64 x, simd_f64 z) {
	constexpr int M = 10;
	static simd_f64_2 coeff[M];
	static const double log2 = std::log(2);
	static bool init = false;
	if (!init) {
		init = true;
		for (int m = 0; m < M; m++) {
			hiprec_real exact = (hiprec_real(2) / (hiprec_real(2 * m + 1))) / log(hiprec_real(2));
			double cox = exact;
			double coy = exact - hiprec_real(cox);
			coeff[m] = simd_f64_2::quick_two_sum(simd_f64(cox), simd_f64(coy));
		}
	}
	simd_f64 y, y2, x2, x0;
	simd_f64_2 X, Y, X2;
	simd_i64 i, j, k;
	x0 = x * simd_f64(M_SQRT2);
	j = ((simd_i64&) x0 & simd_i64(0x7FF0000000000000ULL));
	k = ((simd_i64&) x & simd_i64(0x7FF0000000000000ULL));
	j >>= simd_i64(52);
	k >>= simd_i64(52);
	j -= simd_i64(1023);
	k -= j;
	k <<= simd_i64(52);
	i = (simd_i64&) x;
	i = (i & simd_i64(0xFFFFFFFFFFFFFULL)) | k;
	X = (simd_f64&) i;
	X = (X - simd_f64(1)) / (X + simd_f64(1));
	x2 = X.x * X.x;
	y = coeff[M - 1].x;
	for (int m = M - 2; m >= 1; m--) {
		y = fma(y, x2, coeff[m].x);
	}
	Y = y;
	Y = X * (simd_f64_2::two_product(y, x2) + coeff[0]) + simd_f64(j);
	simd_f64_2 YLOG2X = z * Y;
	return (exp2(YLOG2X.x) * (simd_f64(1) + simd_f64(log2) * YLOG2X.y));
}
/*
simd_f64 pow(simd_f64 x1, simd_f64 y) {
	static std::once_flag once;
	static constexpr int Nbits = 14;
	static constexpr int N = 1 << Nbits;
	static double loghitable[N];
	static double loglotable[N];
	static double expm1table[N];
	static double LN2hi = log(hiprec_real(2));
	static double LN2lo = log(hiprec_real(2)) - hiprec_real(LN2hi);
	static simd_f64 ln2x;
	static simd_f64 ln2y;
	static simd_f64 ax;
	static simd_f64 ay;
	static bool init = false;
	if (!init) {
		for (int n = 0; n < N; n++) {
			std::uint64_t ai = ((std::uint64_t) n << (52 - Nbits)) | ((std::uint64_t)(1023) << 52);
			double x = (double&) ai;
			ai = ((std::uint64_t) n << (52 - 2 * Nbits)) | ((std::uint64_t)(1023) << 52);
			double y = (double&) ai;
			y -= 1.0;
			hiprec_real log_exact = log(hiprec_real(x));
			hiprec_real exp_exact = exp(hiprec_real(y)) - hiprec_real(1);
			loghitable[n] = log_exact;
			expm1table[n] = exp_exact;
			loglotable[n] = log_exact - hiprec_real(loghitable[n]);
			hiprec_real exact = log2(exp(hiprec_real(1)));
			ax = simd_f64(exact);
			ay = simd_f64(exact - hiprec_real((double) exact));
		}
		ln2x = LN2hi;
		ln2y = LN2lo;
		init = true;
	}
	simd_f64 x = x1;
	simd_f64 result, invy, ifloat, x0, dx, y1, y2, one, logx, logy, yt1, zx, zy, argx, argy, yt2, yp1, e0, ddx, ddy, y0, dy, tmp, e1x, e1y, e2x, e2y;
	simd_i64 I, index, n, m;
	simd_f64_2 X0, X, Y, Z, LN2, DX, DY, Y0, Y1, Y2, E0, E1, D, TMP, LOGX, YLOGX;
	LN2.x = ln2x;
	LN2.y = ln2y;
	simd_f64 ONE(1.0);
	simd_f64 NONE(-1.0);
	simd_f64 HALF(0.5);
	invy = y < simd_f64(0);
	y = abs(y);
	simd_f64 xo = x;
	x = frexp(x, &I);
	x *= simd_f64(2);
	I = I - simd_f64(1);
	n = (simd_i64&) x;
	n >>= 52 - Nbits;
	n <<= 52 - Nbits;
	X = x;
	X0.x = (simd_f64&) n;
	DX.x = (X.x - X0.x) / X0.x;
	DX.y = 0.0;
	index = (n & simd_f64(0xFFFFFFFFFFFFFULL)) >> (52 - Nbits);
	Y1.x = DX.x;
	Y1.y = 0.0;
	yp1 = Y1.x + simd_f64(1);
	n = ((simd_i64&) yp1) & simd_i64(0xFFFFFFFFFFFFFULL);
	m = simd_i64(0xFFFFFFFFFFFFFULL);
	m >>= 52 - Nbits;
	m <<= 52 - Nbits;
	n = n & ~m;
	n >>= 52 - 2 * Nbits;
	e0.gather(expm1table, n);
	n <<= 52 - 2 * Nbits;
	n |= simd_i64(1023ULL) << 52;
	Y0 = (simd_f64&) n;
	Y0.x = Y0.x - ONE;
	Y0.y = 0.0;
	DY = Y1 - Y0;
	E0 = e0;
	DY.x = DY.x * (DY.x * HALF + ONE);
	D.x = (E0.x + ONE) * (DY.x + ONE);
	DY.y = D.y = 0.0;
	Y1.x = Y1.x + (DX.x - E0.x - DY.x - E0.x * DY.x) / D.x;
	Y1.y = 0.0;
	yt1.gather(loghitable, index);
	yt2.gather(loglotable, index);
	Y2 = simd_f64_2::quick_two_sum(yt1, yt2);
	LOGX = Y1 + Y2;
	LOGX = LOGX + LN2 * simd_f64(I);
	YLOGX = LOGX * simd_f64(y);
	result = exp(YLOGX.x) * (simd_f64(1) + YLOGX.y);
	result = blend(result, simd_f64(1) / result, invy);
	return result;
}*/
}

double_2 exact_log(double x) {
	constexpr int Nbits = 14;
	constexpr int N = 1 << Nbits;
	constexpr std::uint64_t logmask1 = (0xFFFFFFFFFFFFFFFFULL >> (52 - Nbits)) << (52 - Nbits);
	constexpr std::uint64_t logmask2 = (0xFFFFFFFFFFFFFULL >> (52 - Nbits)) << (52 - Nbits);
	constexpr std::uint64_t expmask1 = ~logmask1;
	constexpr std::uint64_t expmask2 = (!logmask2 & ((0xFFFFFFFFFFFFFULL >> (52 - 2 * Nbits)) << (52 - 2 * Nbits)));
	static double loghitable[N];
	static double loglotable[N];
	static double expm1hitable[N];
	static bool init = false;
	if (!init) {
		init = true;
		for (int n = 0; n < N; n++) {
			std::uint64_t ai = ((std::uint64_t) n << (52 - Nbits)) | ((std::uint64_t)(1023) << 52);
			double x = (double&) ai;
			ai = ((std::uint64_t) n << (52 - 2 * Nbits)) | ((std::uint64_t)(1023) << 52);
			double y = (double&) ai;
			y -= 1.0;
			hiprec_real log_exact = log(hiprec_real(x));
			hiprec_real exp_exact = exp(hiprec_real(y)) - hiprec_real(1);
			loghitable[n] = log_exact;
			expm1hitable[n] = exp_exact;
			loglotable[n] = log_exact - hiprec_real(loghitable[n]);
		}
	}
	volatile double s;
	volatile double v;
	volatile double e;
	uint64_t index;
	uint64_t n = ((uint64_t&) x);
	n >>= 52 - Nbits;
	n <<= 52 - Nbits;
	double_2 X0, DX, Y, Y2, Z, DY;
	X0.x = (double&) n;
	DX.x = (x - X0.x) / X0.x;
	n = (n & 0xFFFFFFFFFFFFFULL) >> (52 - Nbits);
	index = n;
	constexpr int M = 0;
	Y = DX;
	double_2 D;
	double yp1 = Y.x + Y.y + 1.0;
	n = ((uint64_t&) yp1) & 0xFFFFFFFFFFFFFULL;
	uint64_t m = 0xFFFFFFFFFFFFFULL;
	m >>= 52 - Nbits;
	m <<= 52 - Nbits;
	n = n & ~m;
	n >>= 52 - 2 * Nbits;
	if (n >= (1 << Nbits)) {
		n = (1 << Nbits) - 1;
	}
	double expm1p1 = expm1hitable[n];
	n <<= 52 - 2 * Nbits;
	n |= (uint64_t) 1023 << 52;
	double y0 = (double&) n;
	y0 -= 1.0;
	e = -y0;
	s = Y.x + e;
	DY.x = s;
	DY.y = 0.0;
	double E0 = expm1p1;
	double_2 E2, E1;
	s = 1.0 + E0;
	v = s - 1.0;
	e = E0 - v;
	E1.x = s;
	E1.y = e;
	double tmp = DY.x * fma(0.5, DY.x, 1.0);
	s = 1.0 + tmp;
	v = s - 1.0;
	e = tmp - v;
	E2.x = s;
	E2.y = e;
	D.x = E1.x * E2.x;
	D.y = fma(E1.x, E2.x, -D.x);
	D.y = fma(E1.x, E2.y, D.y);
	D.y = fma(E1.y, E2.x, D.y);
	double OMD;
	s = 1.0 - D.x;
	OMD = s - D.y;
	Y.x = Y.x + (OMD + DX.x) / D.x;
	Y.y = 0.0;
	Y2.x = loghitable[index];
	Y2.y = loglotable[index];
	s = Y2.x + Y.x;
	v = s - Y2.x;
	e = Y.x - v;
	Y.x = s;
	Y.y = e + Y2.y;
	return Y;
}

double pow_test(double x, double y) {
	static double LN2hi = log(hiprec_real(2));
	static double LN2lo = log(hiprec_real(2)) - hiprec_real(LN2hi);
	double_2 ln2;
	double volatile s;
	double volatile e;
	double volatile v;
	bool invy = y < 0.0;
	y = abs(y);
	ln2.x = LN2hi;
	ln2.y = LN2lo;
	int I;
	x = frexp(x, &I);
	x *= 2.0;
	I--;
	double_2 arg;
	double_2 Z, Y, R;
	auto logx = exact_log(x);
	Z.x = ln2.x * I;
	Z.y = fma(ln2.x, I, -Z.x);
	Z.y += ln2.y * I;
	s = Z.x + logx.x;
	v = s - Z.x;
	e = logx.x - v;
	logx.x = s;
	logx.y = e + Z.y + logx.y;
	arg.x = y * logx.x;
	arg.y = fma(y, logx.x, -arg.x);
	arg.y += y * logx.y;
	double res = (1.0 + arg.y) * exp(arg.x);
	if (invy) {
		res = 1.0 / res;
	}
	return res;
}

int main() {
	double s, c;
	//feenableexcept(FE_DIVBYZERO);
//	feenableexcept(FE_OVERFLOW);
//	feenableexcept(FE_INVALID);
	using namespace simd;
	FILE* fp = fopen("test.txt", "wt");
	double max_err = 0.0;
	std::vector<double> errs;
	timer tm1, tm2;
	double b = pow_precise(simd_f64(3.0), simd_f64(42.0))[0];
	bool first = true;
	for (double x = .1; x < 10.0; x *= 1.1254) {
		for (double y = -300.0; y <= 300.0; y += 1.11) {
			if (!first) {
				tm1.start();
			}
			b = pow_precise(simd_f64(x), simd_f64(y))[0];
			if (!first) {
				tm1.stop();
			}
			tm2.start();
			double a = pow(hiprec_real(x), hiprec_real(y));
			tm2.stop();
			max_err = std::max(max_err, fabs((a - b) / a));
			errs.push_back(fabs((a - b) / a));
			fprintf(fp, "%.10e %.10e %.10e %.10e %.10e\n", x, y, a, b, (a - b) / a / std::numeric_limits<double>::epsilon());
			first = false;
		}
	}
	//printf("%e\n", tm1.read() / tm2.read());
	std::sort(errs.begin(), errs.end());
	printf("%e %e\n", max_err / std::numeric_limits<double>::epsilon(), errs[50 * errs.size() / 100] / std::numeric_limits<double>::epsilon());
	fclose(fp);/*
	 max_err = 0.0;
	 errs.resize(0);
	 for (double r = 1.0 + (0x1) * std::numeric_limits<double>::epsilon(); r < 2.0; r += 0.001) {
	 auto xxx = log2_precise(r);
	 hiprec_real a = hiprec_real(xxx.x[0]) + hiprec_real(xxx.y[0]);
	 hiprec_real b = log2(hiprec_real(r));
	 max_err = std::max(max_err, fabs((a - b) / a));
	 errs.push_back(abs((a - b) / a));

	 fprintf(fp, "%.10e %.10e %.10e %.10e\n", (double) r, (double) a, (double) (a - b),
	 (double) ((a - b) / a / hiprec_real(std::numeric_limits<double>::epsilon())));
	 }
	 std::sort(errs.begin(), errs.end());
	 printf("%e %e\n", (double) (max_err / std::numeric_limits<double>::epsilon()),
	 (double) (errs[50 * errs.size() / 100] / std::numeric_limits<double>::epsilon()));
	 fclose(fp);
	 return 0;*/
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
//	TEST2(float, simd_f32, pow, powf, pow, 1e-3, 1e3, .01, 10, true);
	//TEST1(double, simd_f64, log2, log2, log2_precise, .0001, 100000, true);
//	TEST1(double, simd_f64, exp, exp, exp, -600.0, 600.0, true);
	/*TEST1(float, simd_f32, tgamma, tgammaf, tgamma, 0.5, 5.0, true);
	 TEST1(double, simd_f64, tgamma, tgamma, tgamma, 0.5, 10.0, true);

	 printf("Testing SIMD Functions\n");
	 printf("\nSingle Precision\n");
	 printf("name   speed        avg err      max err\n");*/

	 TEST2(float, simd_f32, pow, pow, pow, .1, 10, -30, 30, true);
	 TEST1(float, simd_f32, exp, expf, exp, -86.0, 86.0, true);
	 TEST1(float, simd_f32, exp2, exp2f, exp2, -125.0, 125.0, true);
	 TEST1(float, simd_f32, expm1, expm1f, expm1, -2.0, 2.0, true);
	 TEST1(float, simd_f32, log, logf, log, exp(-1), exp(40), true);
	 TEST1(float, simd_f32, log2, log2f, log2, 0.00001, 100000, true);
	 TEST1(float, simd_f32, log1p, log1pf, log1p, exp(-3), exp(3), true);
	 TEST1(float, simd_f32, erf, erff, erf, -7, 7, true);
	 TEST1(float, simd_f32, erfc, erfcf, erfc, -8.9, 8.9, true);
	 TEST1(float, simd_f32, cosh, coshf, cosh, -10.0, 10.0, true);
	 TEST1(float, simd_f32, sinh, sinhf, sinh, -10.0, 10.0, true);
	 TEST1(float, simd_f32, tanh, tanhf, tanh, -10.0, 10.0, true);
	 TEST1(float, simd_f32, sin, sinf, sin, -2 * M_PI, 2 * M_PI, true);
	 TEST1(float, simd_f32, cos, cosf, cos, -2 * M_PI, 2 * M_PI, true);
	 TEST1(float, simd_f32, tan, tanf, tan, -2 * M_PI, 2 * M_PI, true);
	 TEST1(float, simd_f32, asin, asinf, asin, -1, 1, true);
	 TEST1(float, simd_f32, acos, acosf, acos, -1, 1, true);
	 TEST1(float, simd_f32, atan, atanf, atan, -10.0, 10.0, true);

	/*
	 TEST1(float, simd_f32, acosh, acoshf, acosh, 1.001, 10.0, true);
	 TEST1(float, simd_f32, asinh, asinhf, asinh, .001, 10, true);
	 TEST1(float, simd_f32, atanh, atanhf, atanh, 0.001, 0.999, true);

	 TEST1(float, simd_f32, cbrt, cbrtf, cbrt, 1.0 / 4000, 4000, true);
	 TEST1(float, simd_f32, sqrt, sqrtf, sqrt, 0, std::numeric_limits<int>::max(), true);
	 TEST1(float, simd_f32, cvt, cvt32_ref, cvt32_test, 1, +10000000, true);*/

	printf("\nDouble Precision\n");
	printf("name   speed        avg err      max err\n");

	TEST2(double, simd_f64, pow, pow, pow, .1, 10, -300, 300, true);
	TEST1(double, simd_f64, exp, exp, exp, -600.0, 600.0, true);
	TEST1(double, simd_f64, exp2, exp2, exp2, -1000.0, 1000.0, true);
	TEST1(double, simd_f64, expm1, expm1, expm1, -2.0, 2.0, true);
	TEST1(double, simd_f64, log, log, log, exp(-1), exp(40), true);
	TEST1(double, simd_f64, log2, log2, log2, .0001, 100000, true);
	TEST1(double, simd_f64, log1p, log1p, log1p, exp(-3), exp(3), true);
	TEST1(double, simd_f64, erf, erf, erf, -9, 9, true);
	TEST1(double, simd_f64, erfc, erfc, erfc, -25.0, 25.0, true);
	TEST1(double, simd_f64, cosh, cosh, cosh, -10.0, 10.0, true);
	TEST1(double, simd_f64, sinh, sinh, sinh, -10.0, 10.0, true);
	TEST1(double, simd_f64, tanh, tanh, tanh, -10.0, 10.0, true);
	TEST1(double, simd_f64, sin, sin, sin, -2.0 * M_PI, 2.0 * M_PI, true);
	TEST1(double, simd_f64, cos, cos, cos, -2.0 * M_PI, 2.0 * M_PI, true);
	TEST1(double, simd_f64, tan, tan, tan, -2.0 * M_PI, 2.0 * M_PI, true);
	TEST1(double, simd_f64, asin, asin, asin, -1, 1, true);
	TEST1(double, simd_f64, acos, acos, acos, -1 + 1e-6, 1 - 1e-6, true);
	TEST1(double, simd_f64, atan, atan, atan, -10.0, 10.0, true);

	/*	TEST1(double, simd_f64, exp2, exp2, exp2, -1000.0, 1000.0, true);
	 TEST1(double, simd_f64, acosh, acosh, acosh, 1.001, 10.0, true);
	 TEST1(double, simd_f64, asinh, asinh, asinh, .001, 10, true);
	 TEST1(double, simd_f64, atanh, atanh, atanh, 0.001, 0.999, true);
	 TEST1(double, simd_f64, cbrt, cbrt, cbrt, 1.0 / 4000, 4000, true);
	 TEST1(double, simd_f64, sqrt, sqrt, sqrt, 0, std::numeric_limits<long long>::max(), true);
	 TEST1(double, simd_f64, cvt, cvt64_ref, cvt64_test, 1LL, +1000000000LL, true);*/

	return (0);
}

