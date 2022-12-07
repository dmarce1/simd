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

std::vector<double> generate_gamma(int N, int M) {
	constexpr double toler = 0.5 * std::numeric_limits<double>::epsilon();
	hiprec_real a = hiprec_real(1);
	hiprec_real b = hiprec_real(2);
	std::function<hiprec_real(hiprec_real)> func = [a,b](hiprec_real x) {
		const auto sum = a + b;
		const auto dif = b - a;
		const auto half = hiprec_real(0.5);
		x = half*(sum + dif * x);
		return gamma(x);
	};
	std::vector<hiprec_real> co = ChebyCoeffs(func, toler, 0);
	co.resize(co.size() + N - 1, 0.0);
	for (int m = 0; m < co.size(); m++) {
		printf("%e\n", (double) co[m]);
		co[m] = pow(hiprec_real(2), hiprec_real(m)) * co[m];
	}
	std::valarray<hiprec_real> A(co.data(), co.size());
	auto B = A;
	for (int n = 0; n < N - 1; n++) {
		auto num = (hiprec_real(n + 1) + hiprec_real(1) / hiprec_real(2));
		for (int m = 0; m < B.size(); m++) {
			if (m > 0) {
				B[m] = num * A[m] + A[m - 1];
			} else {
				B[m] = num * A[m];
			}
		}
		A = B;
	}
	co.resize(0);
	for (int i = 0; i < std::min((int) A.size(), M); i++) {
		co.push_back(A[i]);
	}
	co.resize(M, 0.0);
	std::vector<double> res(co.size());
	printf("\n");
	for (int i = 0; i < co.size(); i++) {
		res[i] = co[i];
		printf("%e\n", res[i] / res[0]);
	}
	return res;
}

hiprec_real loggamma_deriv(int n, hiprec_real z) {
	if (n == 0) {
		return hiprec_real(1) / (gamma(z));
	} else {
		hiprec_real z0 = z;
		hiprec_real z1 = z0 * hiprec_real(1.0 + 10.0 * std::numeric_limits<float>::epsilon());
		hiprec_real dz = z1 - z0;
		z0 = loggamma_deriv(n - 1, z0);
		z1 = loggamma_deriv(n - 1, z1);
		return (z1 - z0) / dz;
	}
}
;
double factorial2(int n) {
	if (n < 0) {
		printf("%i\n", n);
		return 0.0;
	}
	if (n == 0) {
		return 1.0;
	}
	return factorial2(n - 1) * n;
}
;

double binomial(int n, int k) {
	if (n < k) {
		n++;
		return (n - k) / (double) n * binomial(n, k);
	}
	return factorial2(n) / factorial2(k) / factorial2(n - k);
}
;

float tgamma_test2(float x) {
	static bool init = false;
	static constexpr int N = 2;
	static constexpr int M = 18;
	static float co[N][M];
	if (!init) {
		init = true;
		for (int n = 0; n < N; n++) {
			co[n][0] = lgamma(n + 1);
			double sum = -0.57721566490153286060651209008240243104215933593992;
			for (int k = 1; k < n + 1; k++) {
				sum += 1.0 / (double) k;
			}
			co[n][1] = sum;
			//	printf( "%i %i %e\n", n, 0, co[n][1]);
			for (int m = 2; m < M; m++) {
				double sum = 0.0;
				for (int k = n + 1;; k++) {
					double prod = 1.0;
					double kpow = k;
					int bits = m;
					while (bits) {
						if (bits & 1) {
							prod *= kpow;
						}
						bits >>= 1;
						kpow *= kpow;
					}
					double dsum = 1.0 / prod;
					if (std::abs(dsum / sum) < 1e-14) {
						break;
					}
					sum += dsum;
				}
				co[n][m] = sum / m * pow(-1, m);
			}
		}
	}
	float x0 = std::floor(x);
	float x3 = x;
//	bool z2one = x0 == 0.0;
//	if (z2one) {
//		x0 += 1.0;
//		x += 1.0;
//	}
	float x1 = x - x0;
	int nn = x0 - 1;
	int n;
	if (x1 < 0.5) {
		n = 0;
	} else {
		n = 1;
		x1 -= 1.0;
	}
	float y = co[n][M - 1];
	for (int m = M - 2; m >= 0; m--) {
		y = fmaf(y, x1, co[n][m]);
	}
	y = expf(y);
	float xi = -nn;
	while (nn > 0) {
		y = y * (xi + x);
		xi++;
		nn--;
	}
	while (nn < 0) {
		xi--;
		y = y / (xi + x);
		nn++;
	}
	return y;
}

double zeta(int s) {
	double_2 sum = 0.0;
	int nmax = pow((1ULL << 54), 1.0 / s);
	for (int n = 1; n < nmax; n++) {
		double k = 1;
		double j = 1;
		for (int m = 0; m < s; m++) {
			k *= (double) n;
			j *= (double) n + 1;
		}
		double b = ((double) (n - s)) / (double) k;
		double a = (double) n / (double) j;
		sum.y += a - b;
		sum = double_2::quick_two_sum(sum.x, sum.y);
	}
	return sum.x / (s - 1.0);
}

double tgamma_test(double x_) {
	using namespace simd;
	static bool init = false;
	static constexpr int M = 23;
	static constexpr int N = 2 * M;
	static constexpr int Nsin = 10;
	static double coeffs[Nsin];
	static constexpr int Nsets = 13;
	static double coeff[M][Nsets];
	int xa = 7.5;
	double xb = 2.5;
	if (!init) {
		init = true;
		hiprec_real A[N];
		A[0] = hiprec_real(0.5) * sqrt(hiprec_real(2));
		for (int n = 1; n < N; n++) {
			hiprec_real sum = 0.0;
			for (int k = 1; k < n; k++) {
				sum += A[k] * A[n - k] / hiprec_real(k + 1);
			}
			sum = 1.0 / n * A[n - 1] - sum;
			sum /= A[0] * (hiprec_real(1) + hiprec_real(1) / (hiprec_real(n) + hiprec_real(1)));
			A[n] = sum;
		}
		for (int n = 0; n < M; n++) {
			coeff[n][12] = (A[2 * n] * sqrt(hiprec_real(2)) * gamma(hiprec_real(n) + hiprec_real(0.5)) / gamma(hiprec_real(0.5)));
		}
		for (int n = 0; n < Nsin; n++) {
			const hiprec_real pi = hiprec_real(4) * atan(hiprec_real(1));
			coeffs[n] = pow(hiprec_real(pi), hiprec_real(2 * n + 1)) * pow(hiprec_real(-1), hiprec_real(n)) / factorial(2 * n + 1) / pi;
		}
		for (int n = 0; n < 6; n++) {
			hiprec_real a = hiprec_real(5) / hiprec_real(4) * (hiprec_real(n + 2));
			hiprec_real b = hiprec_real(5) / hiprec_real(4) * (hiprec_real(n + 3));
			std::function<hiprec_real(hiprec_real)> func = [a,b](hiprec_real x) {
				const auto sum = a + b;
				const auto dif = b - a;
				const auto half = hiprec_real(0.5);
				x = half*(sum + dif * x);
				return gamma(x)*exp(x)*pow(x,-x)*sqrt(x / hiprec_real(8) / atan(hiprec_real(1)));
			};
			auto chebies = ChebyCoeffs(func, std::numeric_limits<double>::epsilon() * 0.005, 0);
			chebies.resize(M, 0.0);
			for (int m = 0; m < M; m++) {
				coeff[m][4 + n] = chebies[m];
			}
		}
		coeff[0][0] = 0.0;
		coeff[0][1] = logl(tgammal(1.5));
		coeff[0][2] = 0.0;
		coeff[0][3] = logl(tgammal(2.5));
		coeff[1][0] = -.57721566490153286060651209008240243104215933593992L;
		coeff[1][2] = -.57721566490153286060651209008240243104215933593992 + 1.0L;
		double_2 sum = -.57721566490153286060651209008240243104215933593992L - 2.0L / 3.0L;
		coeff[1][1] = 0.0364899739785765205590237;
		coeff[1][3] = 0.7031566406452431872256903336679110677;
		for (int n = 2; n < M; n++) {
			double z = zeta(n);
			coeff[n][0] = pow(-1.0, n) * z / n;
			coeff[n][1] = pow(-1.0, n) * z / n * ((1 << n) - 1);
			coeff[n][1] += pow(-1.0, n - 1) * pow(0.5, -n) / n;
			coeff[n][2] = coeff[n][0] - pow(-1.0, n) / n;
			coeff[n][3] = coeff[n][1] + pow(-1.0, n - 1) * pow(1.5, -n) / n;
		}
	}
	simd_f64 x(x_);
	simd_f64 y, z, x0, x1, nf, x2, c, xc, a, b, sgn;
	simd_i64 neg, inv, ni, xoa, xob, flag;

//	bool neg = false;
//	double y = 0.0;
//	double z, x0, x2;
	/*	if (x < 0.0) {
	 neg = true;
	 if (x < -1.0) {
	 x0 = abs(x);
	 } else {
	 x0 = x + 1.0;
	 }
	 } else {
	 x0 = x;
	 }*/
	neg = x < simd_f64(0);
	x0 = x + simd_f64(1);
	x0 = blend(abs(x), x0, x >= simd_f64(-1));
	x0 = blend(x, x0, neg);

	/*	bool inv = x0 < 1.0;
	 if (inv) {
	 x0 += 1.0;
	 }*/
	inv = x0 < simd_f64(1);
	x0 = blend(x0, x0 + simd_f64(1), inv);

	/*	int n;
	 if (x0 > xa) {
	 z = 1.0 / x0;
	 n = 12;
	 } else if (x0 > xb) {
	 n = floor(x0 / 1.25) + 2;
	 double xc = 1.25 * (n - 1.5);
	 z = 2.0 * (x0 - xc) / 1.25;
	 } else if (x0 >= 1.0) {
	 double x1 = round(2.0 * x0) * 0.5;
	 z = x0 - x1;
	 n = round(2 * x1 - 2);
	 }*/

	x1 = round(simd_f64(2) * x0) * simd_f64(0.5);
	xoa = x0 > simd_f64(7.5);
	xob = x0 > simd_f64(2.5);
	nf = round(simd_f64(2) * x1 - simd_f64(2));
	nf = blend(nf, floor(x0 * simd_f64(0.8)) + simd_i64(2), xob);
	nf = blend(nf, simd_f64(12), xoa);
	xc = simd_f64(1.25) * (simd_f64(nf) - simd_f64(1.5));
	z = x0 - x1;
	z = blend(z, simd_f64(1.6) * (x0 - xc), xob);
	z = blend(z, simd_f64(1) / x0, xoa);
	ni = simd_i64(round(nf));
	/*for (int k = M - 1; k >= 0; k--) {
	 y = fma(y, z, coeff[n][k]);
	 }*/
	y = simd_f64(0);
	for (int k = M - 1; k >= 0; k--) {
		y = fma(y, z, c.gather(coeff[k], ni));
	}

	/*if (n >= 4) {
	 double a = exp(-x0);
	 double b = pow(x0, 0.5 * x0);
	 a *= b;
	 a *= b;
	 y *= a * sqrt(2.0 * M_PIl / x0);
	 } else {
	 y = exp(y);
	 }*/
	flag = ni >= simd_i64(4);
	a = exp(blend(y, -x0, flag));
	b = pow(x0, simd_f64(0.5) * x0);
	a = blend(a, a * b, flag);
	a = blend(a, a * b, flag);
	a = blend(a, a * sqrt(simd_f64(2.0 * M_PI) / x0), flag);
	y = blend(a, a * y, flag);
	/*	if (inv) {
	 y = y / (abs(x));
	 }*/
	y = blend(y, y / abs(x), inv);

	/*if (neg) {
		if (inv) {
			y /= -(x + 1.0);
		} else {
			x0 = x - floor(x);
			int n = round(floor(x * 2.0) * 0.5);
			double sgn = -1.0;
			if (abs(n % 2) == 1) {
				sgn = -sgn;
			}
			if (x0 > 0.5) {
				x0 = 1.0 - (x - floor(x));
			}
			z = 0.0;
			x2 = x0 * x0;
			for (int k = Nsin - 1; k >= 0; k--) {
				z = fma(z, x2, coeffs[k]);
			}
			z *= x0 * sgn;
			y = 1.0 / (y * x * z);
		}
	}*/
	x2 = x - floor(x);
	x0 = blend(x2, simd_f64(1) - (x - floor(x)), x2 > simd_f64(0.5));
	sgn = simd_f64(1);
	nf = abs(floor(abs(x) * simd_f64(2))*simd_f64(0.5));
	ni = simd_i64(nf);
	sgn = blend(sgn, -sgn, (ni & simd_i64(1)) != simd_i64(0));
//	printf( "%f %e %e \n", nf[0], x0[0], sgn[0]);
//	printf( "%e %i\n", sgn[0], ni[0]);
	z = simd_f64(0);
	x2 = x0 * x0;
	for (int k = Nsin - 1; k >= 0; k--) {
		z = fma(z, x2, simd_f64(coeffs[k]));
	}
	z *= x0 * sgn;
	z = simd_f64(1) / (x * y * z);
	z = blend(z, y / -(simd_f64(x) + simd_f64(1)), inv);
	y = blend(y, z, neg);

	return y[0];
}

int main() {
	using namespace simd;
	double s, c;
	double maxe = 0.0;
	double avge = 0.0;
	int N = 0;
	double eps = std::numeric_limits<double>::epsilon();
	for (double x = -0.00001; x > -170.00; x *= 1.0 + 0.0001 * rand1()) {
		N++;
		double a = tgammal(x);
		double b = tgamma_test(x);
		double err = std::abs((a - b) / a);
		maxe = std::max(maxe, err);
		avge += err;
		printf("%e %e %e %e\n", x, b, a, err / eps);
	}
	avge /= N;
	printf("%e %e \n", maxe / eps, avge / eps);
	return 0;
//	TEST2(float, simd_f32, pow, powf, pow, 1e-3, 1e3, .01, 10, true);
//TEST1(double, simd_f64, log2, log2, log2_precise, .0001, 100000, true);
//	TEST1(double, simd_f64, exp, exp, exp, -600.0, 600.0, true);
	return 0;
	TEST1(double, simd_f64, tgamma, tgamma, tgamma, -0.999, 0.000, true);
	TEST1(float, simd_f32, tgamma, tgammaf, tgamma, 1.0, 33.0, true);
	/*
	 printf("Testing SIMD Functions\n");
	 printf("\nSingle Precision\n");
	 printf("name   speed        avg err      max err\n");

	 TEST1(float, simd_f32, acosh, acoshf, acosh, 1.001, 10.0, true);
	 TEST1(float, simd_f32, asinh, asinhf, asinh, .001, 10, true);
	 TEST1(float, simd_f32, atanh, atanhf, atanh, 0.001, 0.999, true);
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
	 TEST1(float, simd_f32, atan, atanf, atan, -10.0, 10.0, true);*/

	/*

	 TEST1(float, simd_f32, cbrt, cbrtf, cbrt, 1.0 / 4000, 4000, true);
	 TEST1(float, simd_f32, sqrt, sqrtf, sqrt, 0, std::numeric_limits<int>::max(), true);
	 TEST1(float, simd_f32, cvt, cvt32_ref, cvt32_test, 1, +10000000, true);*/

	printf("\nDouble Precision\n");
	printf("name   speed        avg err      max err\n");

	/*TEST1(double, simd_f64, acosh, acosh, acosh, 1.001, 10.0, true);
	 TEST1(double, simd_f64, asinh, asinh, asinh, .001, 10, true);
	 TEST1(double, simd_f64, atanh, atanh, atanh, 0.001, 0.999, true);
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
	 TEST1(double, simd_f64, atan, atan, atan, -10.0, 10.0, true);*/

	/*	TEST1(double, simd_f64, exp2, exp2, exp2, -1000.0, 1000.0, true);
	 TEST1(double, simd_f64, cbrt, cbrt, cbrt, 1.0 / 4000, 4000, true);
	 TEST1(double, simd_f64, sqrt, sqrt, sqrt, 0, std::numeric_limits<long long>::max(), true);
	 TEST1(double, simd_f64, cvt, cvt64_ref, cvt64_test, 1LL, +1000000000LL, true);*/

	return (0);
}

