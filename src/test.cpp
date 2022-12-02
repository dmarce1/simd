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

struct double_2 {
	double x;
	double y;
	static inline double_2 quick_two_sum(double a_, double b_) {
		double_2 r;
		volatile double a = a_;
		volatile double b = b_;
		volatile double s = a + b;
		volatile double tmp = s - a;
		volatile double e = b - tmp;
		r.x = s;
		r.y = e;
		return r;
	}
	static inline double_2 two_sum(double a_, double b_) {
		double_2 r;
		volatile double a = a_;
		volatile double b = b_;
		volatile double s = a + b;
		volatile double v = s - a;
		volatile double tmp1 = s - v;
		volatile double tmp2 = a - tmp1;
		volatile double tmp3 = b - v;
		volatile double e = tmp2 + tmp3;
		r.x = s;
		r.y = e;
		return r;
	}
	static inline double_2 two_product(double a_, double b_) {
		double_2 r;
		const static double zero = 0.0;
		volatile double a = a_;
		volatile double b = b_;
		volatile double xx = a * b;
		volatile double yy = std::fma(a, b, -xx);
		r.x = xx;
		r.y = yy;
		return r;
	}
public:
	inline double_2& operator=(double a) {
		x = a;
		y = 0.0;
		return *this;
	}
	inline double_2(double a) {
		*this = a;
	}
	inline double_2() {
	}
	inline operator double() const {
		return x + y;
	}
	inline double_2 operator+(double_2 other) const {
		double_2 s, t;
		s = two_sum(x, other.x);
		t = two_sum(y, other.y);
		s.y += t.x;
		s = quick_two_sum(s.x, s.y);
		s.y += t.y;
		s = quick_two_sum(s.x, s.y);
		return s;
	}
	inline double_2 operator*(double_2 other) const {
		double_2 p;
		p = two_product(x, other.x);
		p.y += x * other.y;
		p.y += y * other.x;
		p = quick_two_sum(p.x, p.y);
		return p;
	}
	inline double_2 operator-() const {
		double_2 r;
		r.x = -x;
		r.y = -y;
		return r;
	}
	inline double_2 operator/(const double_2 A) const {
		double_2 result;
		double xn = double(1) / A.x;
		double yn = x * xn;
		double_2 diff = (*this - A * double_2(yn));
		double_2 prod = two_product(xn, diff);
		return double_2(yn) + prod;
	}
	inline double_2 operator-(double_2 other) const {
		return *this + -other;
	}

};

double_2 exact_exp(double x) {
	constexpr int M = 20;
	constexpr int P = 2;
	static bool init = false;
	static double_2 coeff[M];
	if (!init) {
		init = true;
		hiprec_real c0 = hiprec_real(1);
		for (int m = 0; m < M; m++) {
			coeff[m] = c0;
			coeff[m] = coeff[m] + (c0 - hiprec_real(coeff[m]));
			c0 /= hiprec_real(m + 1);
		}
	}
	double_2 Z, X, S, T;
	volatile double s;
	volatile double v;
	volatile double w;
	volatile double z;
	volatile double e;
	double t;
	Z.x = 0.0;
	Z.y = 0.0;
	for (int m = M - 1; m > P; m--) {
		Z.x = fma(Z.x, x, coeff[m].x);
	}
	X.x = x;
	X.y = 0.0;
	t = Z.x * X.x;
	Z.y = fma(Z.x, X.x, -t);
	Z.x = t;
	s = 0.5 + Z.x;
	v = s - 0.5;
	e = Z.x - v;
	S.x = s;
	S.y = e;
	S.y += Z.y;
	s = S.x + S.y;
	v = s - S.x;
	e = S.y - v;
	Z.x = s;
	Z.y = e;
	for (int n = 0; n < 2; n++) {
		t = Z.x * X.x;
		e = fma(Z.x, X.x, -t);
		e = fma(Z.y, X.x, e);
		Z.x = t;
		Z.y = e;
		s = Z.x + Z.y;
		v = s - Z.x;
		e = Z.y - v;
		Z.x = s;
		Z.y = e;
		s = 1.0 + Z.x;
		v = s - 1.0;
		e = Z.x - v;
		S.x = s;
		S.y = e;
		S.y += Z.y;
		s = S.x + S.y;
		v = s - S.x;
		e = S.y - v;
		Z.x = s;
		Z.y = e;
	}
	return Z;
}

double_2 exact_log(double x) {
	constexpr int Nbitslog2 = 11;
	constexpr int Nlog2 = 1 << Nbitslog2;
	constexpr std::uint64_t log2mask1 = (0xFFFFFFFFFFFFFFFFULL >> (52 - Nbitslog2)) << (52 - Nbitslog2);
	constexpr std::uint64_t log2mask2 = (0xFFFFFFFFFFFFFULL >> (52 - Nbitslog2)) << (52 - Nbitslog2);
	static double_2 logtable[Nlog2];
	static bool init = false;
	if (!init) {
		init = true;
		for (int n = 0; n < Nlog2; n++) {
			std::uint64_t ai = ((std::uint64_t) n << (52 - Nbitslog2)) | ((std::uint64_t)(1023) << 52);
			double a = (double&) ai;
			double_2& entry = logtable[n];
			hiprec_real exact = log(hiprec_real(a));
			entry = exact;
			entry = entry + double_2(double(exact - hiprec_real(entry)));
		}
	}

	volatile double s;
	volatile double v;
	volatile double w;
	volatile double z;
	volatile double s2;
	double t;
	double e;
	double a;
	double b;
	double x2;
	double x0;
	double_2 Z, X, p, q;
	std::uint64_t i;
	i = (std::uint64_t&) x;
	i &= log2mask1;
	x0 = ((double &) i);
	x -= x0;
	i &= log2mask2;
	i >>= 52 - Nbitslog2;
	x0 *= 2;
	s = x0 + x;
	v = s - x0;
	e = x - v;
	q.x = s;
	q.y = e;
	a = 1.0 / q.x;
	b = x * a;
	p.x = q.x * b;
	p.y = fma(q.x, b, -p.x);
	p.y = fma(b, q.y, p.y);
	s = p.x + p.y;
	v = s - p.x;
	e = p.y - v;
	p.x = -s;
	p.y = -e;
	s = x + p.x;
	q.x = p.y;
	v = s - x;
	t = s - v;
	w = x - t;
	z = p.x - v;
	e = w + z;
	p.x = s;
	p.y = e;
	p.y += q.x;
	s = p.x + p.y;
	v = s - p.x;
	e = p.y - v;
	p.y = e;
	p.x = s + p.y;
	p.x *= a;
	s = b + p.x;
	v = s - b;
	t = s - v;
	w = b - t;
	z = p.x - v;
	X.x = s;
	X.y = w + z;
	x2 = X.x * X.x;
	a = 2.0;
	b = x2 * ((2.0 / 3.0) + 0.4 * x2);
	s = a + b;
	p.x = s;
	v = s - a;
	p.y = b - v;
	Z.x = X.x * p.x;
	Z.y = fma(X.x, p.x, -Z.x);
	Z.y = fma(X.x, p.y, Z.y);
	Z.y = fma(X.y, p.x, Z.y);
	s = Z.x + Z.y;
	v = s - Z.x;
	X.y = Z.y - v;
	X.x = s;
	Z = X;
	X = logtable[i];
	s = X.x + Z.x;
	q.x = X.y + Z.y;
	v = s - X.x;
	t = s - v;
	w = X.x - t;
	z = Z.x - v;
	e = w + z;
	p.x = s;
	p.y = e;
	p.y += q.x;
	s = p.x + p.y;
	v = s - p.x;
	e = p.y - v;
	p.x = s;
	p.y = e;
	return p;

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
	/*for (double x = 1; x < 140.0; x *= (1.0 + 1 * rand1())) {
	 for (double y = -140.0; y <= 140.0; y += 1 * rand1()) {
	 double b = pow_test2(x, y);
	 double a = pow(x, y);
	 max_err = std::max(max_err, fabs((a - b) / a));
	 errs.push_back(fabs((a - b) / a));
	 fprintf(fp, "%.10e %.10e %.10e %.10e %.10e\n", x, y, a, b, (a - b) / a / std::numeric_limits<double>::epsilon());
	 }
	 }*/
//	printf("%e %e\n", max_err / std::numeric_limits<double>::epsilon(), errs[50 * errs.size() / 100] / std::numeric_limits<double>::epsilon());
//	fclose(fp);
	//	return 0;
	for (double r = 0.0; r < 1.0; r += 0.001) {
		auto xxx = exact_exp(r);
		hiprec_real a = hiprec_real(xxx.x) + hiprec_real(xxx.y);
		hiprec_real b = exp(hiprec_real(r));
		max_err = std::max(max_err, fabs((a - b) / a));
		errs.push_back(abs((a - b) / a));

		fprintf(fp, "%.10e %.10e %.10e %.10e\n", (double) r, (double) a, (double) b,
				(double) ((a - b) / a / hiprec_real(std::numeric_limits<double>::epsilon())));
	}
	std::sort(errs.begin(), errs.end());
	printf("%e %e\n", (double) (max_err / std::numeric_limits<double>::epsilon()),
			(double) (errs[50 * errs.size() / 100] / std::numeric_limits<double>::epsilon()));
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
//	TEST2(float, simd_f32, pow, powf, pow, 1e-3, 1e3, .01, 10, true);
	TEST2(double, simd_f64, pow, pow, pow, 1, 1e3, 1, 10, true);
	/*	TEST1(double, simd_f64, exp, exp, exp, -600.0, 600.0, true);
	 TEST1(float, simd_f32, tgamma, tgammaf, tgamma, 0.5, 5.0, true);
	 TEST1(double, simd_f64, tgamma, tgamma, tgamma, 0.5, 10.0, true);

	 printf("Testing SIMD Functions\n");
	 printf("\nSingle Precision\n");
	 printf("name   speed        avg err      max err\n");

	 TEST1(float, simd_f32, cosh, coshf, cosh, -10.0, 10.0, true);
	 TEST1(float, simd_f32, sinh, sinhf, sinh, -10.0, 10.0, true);
	 TEST1(float, simd_f32, tanh, tanhf, tanh, -10.0, 10.0, true);
	 TEST1(float, simd_f32, exp, expf, exp, -86.0, 86.0, true);
	 TEST1(float, simd_f32, exp2, exp2f, exp2, -125.0, 125.0, true);
	 TEST1(float, simd_f32, expm1, expm1f, expm1, -2.0, 2.0, true);
	 TEST1(float, simd_f32, log, logf, log, exp(-1), exp(40), true);
	 TEST1(float, simd_f32, log2, log2f, log2, 0.00001, 100000, true);
	 TEST1(float, simd_f32, log1p, log1pf, log1p, exp(-3), exp(3), true);
	 TEST1(float, simd_f32, erf, erff, erf, -7, 7, true);
	 TEST1(float, simd_f32, erfc, erfcf, erfc, -8.9, 8.9, true);

	 TEST1(float, simd_f32, sin, sinf, sin, -2 * M_PI, 2 * M_PI, true);
	 TEST1(float, simd_f32, cos, cosf, cos, -2 * M_PI, 2 * M_PI, true);
	 TEST1(float, simd_f32, tan, tanf, tan, -2 * M_PI, 2 * M_PI, true);

	 TEST1(float, simd_f32, asin, asinf, asin, -1, 1, true);
	 TEST1(float, simd_f32, acos, acosf, acos, -1, 1, true);
	 TEST1(float, simd_f32, atan, atanf, atan, -10.0, 10.0, true);
	 */
	/*
	 TEST1(float, simd_f32, acosh, acoshf, acosh, 1.001, 10.0, true);
	 TEST1(float, simd_f32, asinh, asinhf, asinh, .001, 10, true);
	 TEST1(float, simd_f32, atanh, atanhf, atanh, 0.001, 0.999, true);

	 TEST1(float, simd_f32, cbrt, cbrtf, cbrt, 1.0 / 4000, 4000, true);
	 TEST1(float, simd_f32, sqrt, sqrtf, sqrt, 0, std::numeric_limits<int>::max(), true);
	 TEST1(float, simd_f32, cvt, cvt32_ref, cvt32_test, 1, +10000000, true);*/

	printf("\nDouble Precision\n");
	printf("name   speed        avg err      max err\n");

	TEST1(double, simd_f64, cosh, cosh, cosh, -10.0, 10.0, true);
	TEST1(double, simd_f64, sinh, sinh, sinh, -10.0, 10.0, true);
	TEST1(double, simd_f64, tanh, tanh, tanh, -10.0, 10.0, true);
	TEST1(double, simd_f64, exp, exp, exp, -600.0, 600.0, true);
	TEST1(double, simd_f64, exp2, exp2, exp2, -1000.0, 1000.0, true);
	TEST1(double, simd_f64, expm1, expm1, expm1, -2.0, 2.0, true);
	TEST1(double, simd_f64, log, log, log, exp(-1), exp(40), true);
	TEST1(double, simd_f64, log2, log2, log2, .0001, 100000, true);
	TEST1(double, simd_f64, log1p, log1p, log1p, exp(-3), exp(3), true);
	TEST1(double, simd_f64, erf, erf, erf, -9, 9, true);
	TEST1(double, simd_f64, erfc, erfc, erfc, -25.0, 25.0, true);

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

