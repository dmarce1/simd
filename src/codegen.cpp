#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "../include/simd.hpp"
#include "../include/hiprec.hpp"
#include "../include/polynomial.hpp"
#include <functional>
#include <valarray>
#include <complex>

#define SYSTEM(...) if( system(__VA_ARGS__) != 0 ) {printf( "SYSTEM error %s %i\n", __FILE__, __LINE__); abort(); }

template<class ...Args>
std::string print2str(const char* fstr, Args&&...args) {
	std::string result;
	char* str;
	if (!asprintf(&str, fstr, std::forward<Args>(args)...)) {
		printf("Error in %s on line %i\n", __FILE__, __LINE__);
		abort();
	}
	result = str;
	free(str);
	return result;
}
void fft(std::valarray<std::complex<hiprec_real>>& x) {
	const size_t N = x.size();
	if (N <= 1)
		return;
	std::valarray<std::complex<hiprec_real>> even = x[std::slice(0, N / 2, 2)];
	std::valarray<std::complex<hiprec_real>> odd = x[std::slice(1, N / 2, 2)];
	fft(even);
	fft(odd);
	for (size_t k = 0; k < N / 2; ++k) {
		static const auto one = hiprec_real(1);
		static const auto twopi = hiprec_real(8) * atan(hiprec_real(1));
		std::complex<hiprec_real> t = std::polar(one, -twopi * hiprec_real((int) k) / hiprec_real((int) N)) * odd[k];
		x[k] = even[k] + t;
		x[k + N / 2] = even[k] - t;
	}
}

void dct(std::valarray<hiprec_real>& x) {
	const auto zero = hiprec_real(0);
	std::valarray<std::complex<hiprec_real>> y;
	y.resize(x.size() * 4);
	for (int n = 0; n < 4 * x.size(); n++) {
		std::complex<hiprec_real> z;
		if (n % 2 == 0) {
			z = std::complex<hiprec_real>(zero, zero);
		} else if (n < 2 * x.size()) {
			z.real(x[n / 2]);
			z.imag(zero);
		} else {
			z.real(x[2 * x.size() - n / 2 - 1]);
			z.imag(zero);
		}
		y[n] = z;
	}
	fft(y);
	for (int n = 0; n < x.size(); n++) {
		x[n] = y[n].real() * (hiprec_real(2) - hiprec_real((int) (n == 0))) / hiprec_real(2 * (int) x.size());
	}
}

std::vector<double> ChebyCoeffs(std::function<hiprec_real(hiprec_real)> func, hiprec_real (toler)) {
	static const hiprec_real one(1);
	static const hiprec_real pi(hiprec_real(4) * atan(hiprec_real(1)));
	hiprec_real last, next;
	next = hiprec_real(99);
	int M = 2;
	int N;
	std::valarray<hiprec_real> X;
	hiprec_real eps;
	do {
		X.resize(M);
		last = next;
		for (int m = 0; m < M; m++) {
			X[m] = func(cos(pi * (hiprec_real(m) + hiprec_real(0.5)) / hiprec_real(M)));
		}
		dct(X);
		int n = 0;
		eps = toler;
		while (abs(X[n]) > eps && n < M - 1) {
			n++;
			if (n > 0) {
				eps /= hiprec_real(2);
			}
		}
		next = X[n];
		M *= 2;
		N = n;
	} while (abs(next - last) / abs(last) > eps);
	auto tn = Tns(N + 1);
	polynomial c0(N + 1, hiprec_real(0));
	for (int n = 0; n < N; n++) {
		c0 = poly_add(c0, poly_mul(poly_term(0, X[n]), tn[n]));
	}
	std::vector<double> coeff(N);
	for (int n = 0; n < N; n++) {
		coeff[n] = (double) c0[n];
	}
	return coeff;
}

void float_funcs(FILE* fp) {
	{
		std::function<hiprec_real(hiprec_real)> func = [](hiprec_real x) {
			static const auto threehalf = hiprec_real(3) / hiprec_real(2);
			static const auto half = hiprec_real(1) / hiprec_real(2);
			static const auto log2inv = hiprec_real(1) / log(hiprec_real(2));
			return log(threehalf + x * half) * log2inv;
		};
		auto coeff = ChebyCoeffs(func, std::numeric_limits<float>::epsilon() * 0.5);
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 log2(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 y, z;\n");
		fprintf(fp, "\tsimd_i32 i;\n");
		fprintf(fp, "\ti = ((simd_i32&) x & simd_i32(0x7F800000));\n");
		fprintf(fp, "\ti >>= int(23);\n");
		fprintf(fp, "\ti -= simd_i32(127);\n");
		fprintf(fp, "\tz = simd_f32(i);\n");
		fprintf(fp, "\ti = (simd_i32&) x;\n");
		fprintf(fp, "\ti &= simd_i32(0x7FFFFF);\n");
		fprintf(fp, "\ti |= simd_i32(0x%X);\n", 127 << 23);
		fprintf(fp, "\tx = (simd_f32&) i;\n");
		fprintf(fp, "\tx *= simd_f32(2);\n");
		fprintf(fp, "\tx -= simd_f32(3);\n");
		int N = coeff.size() - 1;
		fprintf(fp, "\ty = simd_f32(%.17e);\n", coeff[N]);
		for( int n = N - 1; n >= 0; n--){
			fprintf(fp, "\ty = fma(y, x, simd_f32(%.17e));\n", coeff[n]);
		}
		fprintf(fp, "\ty += z;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	{
		constexpr int N = 19;
		polynomial c0;
		double coeff[N];
		coeff[0] = jn(0, 1.0);
		double b = -2.0;
		auto Tn = Tns(2 * N);
		for (int n = 1; n < N; n++) {
			coeff[n] = b * jn(2 * n, 1.0);
			b = -b;
		}
		for (int n = 0; n < N; n++) {
			c0 = poly_add(c0, poly_mul(poly_term(0, coeff[n]), Tn[2 * n]));
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 cos(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 p, y;\n");
		fprintf(fp, "\tp = x * simd_f32(%.17e) - simd_f32(0.5);\n", 1.0 / (2.0 * M_PI));
		fprintf(fp, "\tp -= floor(p);\n");
		fprintf(fp, "\tp -= simd_f32(0.5);\n");
		fprintf(fp, "\tx = p * p * simd_f32(%.17e);\n", 4.0 * M_PI * M_PI);
		fprintf(fp, "\ty = simd_f32(%.17e);\n", double(c0[N - 1]));
		for (int n = N - 3; n >= 0; n -= 2) {
			fprintf(fp, "\ty = fma(x, y, simd_f32(%.17e));\n", double(c0[n]));
		}
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	{
		constexpr int N = 9;
		double c0[N];
		int nf = 1;
		for (int n = 0; n < N; n++) {
			c0[n] = 1.0 / nf;
			nf *= (n + 1);
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 exp(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 x0, y, zero;\n");
		fprintf(fp, "\tsimd_i32 i;\n");
		fprintf(fp, "\tx = max(simd_f32(-87), min(simd_f32(87), x));\n");
		fprintf(fp, "\tx0 = round(x * simd_f32(%.20e));\n", 1.0 / M_LN2);
		fprintf(fp, "\tx -= x0 * simd_f32(%.20e);\n", M_LN2);
		fprintf(fp, "\ty = simd_f32(%.20e);\n", c0[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x, y, simd_f32(%.17e));\n", c0[n]);
		}
		fprintf(fp, "\ti = (simd_i32(x0) + simd_i32(127)) << int(23);\n");
		fprintf(fp, "\ty *= (simd_f32&) i;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}

	{
		constexpr int N = 16;
		constexpr int M = 4;
		constexpr double xmax = 9.19;
		static double c0[N][M];
		double a0 = 1.00;
		double a1 = 3.0;
		double a2 = 6.0;
		double a3 = (8.0 + xmax) * 0.5;
		for (int m = 0; m < M; m++) {
			double a;
			if (m == 0) {
				a = a0;
			} else if (m == 1) {
				a = a1;
			} else if (m == 2) {
				a = a2;
			} else {
				a = a3;
			}
			c0[0][m] = exp(a * a) * erfc(a);
			c0[1][m] = 2.0 * (-1.0 / sqrt(M_PI) + a * c0[0][m]);
			for (int n = 2; n < N - 1; n++) {
				c0[n][m] = 2.0 / n * (a * c0[n - 1][m] + c0[n - 2][m]);
			}
			c0[N - 1][m] = -99.0;
		}
		fprintf(fp, "simd_f32 erfc(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 a, b;\n");
		fprintf(fp, "\terfcexp(x, &a, &b);\n");
		fprintf(fp, "\treturn a;\n");
		fprintf(fp, "}\n\n");
		fprintf(fp, "void erfcexp(simd_f32 x, simd_f32* erfcptr, simd_f32* expptr) {\n");
		fprintf(fp, "\tconst static simd_f32 x0 = {%24.17ef, %24.17ef, %24.17ef, %24.17ef, 99.0f, 99.0f, 99.0f, 99.0f};\n", a0, a1, a2, a3);
		fprintf(fp, "\tconst static simd_f32 coeff[%i] = {\n", N / 2);
		for (int n = 0; n < N; n += 2) {
			fprintf(fp, "\t\t{");
			for (int m = 0; m < M; m++) {
				fprintf(fp, "%24.17ef, ", c0[n][m]);
			}
			for (int m = 0; m < M; m++) {
				fprintf(fp, "%24.17ef", c0[n + 1][m]);
				if (m != M - 1) {
					fprintf(fp, ", ");
				}
			}
			fprintf(fp, "}");
			if (n != N - 2) {
				fprintf(fp, ", ");
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\t};\n");
		fprintf(fp, "\tsimd_f32 r, y, c0, c1, e, a, neg, rng;\n");
		fprintf(fp, "\tsimd_i32 i0, i1;\n");
		fprintf(fp, "\tneg = simd_f32(x < simd_f32(0));\n");
		fprintf(fp, "\trng = simd_f32(x < simd_f32(%.17e));\n", xmax);
		fprintf(fp, "\tx = min(abs(x), simd_f32(%.17e));\n", xmax);
		fprintf(fp, "\te = exp(-x * x);\n");
		fprintf(fp, "\ti0 = max(((((simd_i32&) x) & simd_i32(0x7F800000)) >> int(23)) - simd_i32(127), simd_i32(0));\n");
		fprintf(fp, "\ti1 = i0 + simd_i32(%i);\n", M);
		fprintf(fp, "\tx -= x0.permute(i0);\n");
		fprintf(fp, "\ty = coeff[%i].permute(i0);\n", N / 2 - 1);
		for (int n = N - 3; n >= 0; n -= 2) {
			fprintf(fp, "\tc1 = coeff[%i].permute(i1);\n", n / 2);
			fprintf(fp, "\tc0 = coeff[%i].permute(i0);\n", n / 2);
			fprintf(fp, "\ty = fma(y, x, c1);\n");
			fprintf(fp, "\ty = fma(y, x, c0);\n");
		}
		fprintf(fp, "\ty *= e * rng;\n");
		fprintf(fp, "\t*expptr = e;\n");
		fprintf(fp, "\t*erfcptr = y * (simd_f32(1) - neg) + (simd_f32(2) - y) * neg;\n");
		fprintf(fp, "}\n");

	}

}
void double_funcs(FILE* fp) {
	{
		std::function<hiprec_real(hiprec_real)> func = [](hiprec_real x) {
			static const auto threehalf = hiprec_real(3) / hiprec_real(2);
			static const auto half = hiprec_real(1) / hiprec_real(2);
			static const auto log2inv = hiprec_real(1) / log(hiprec_real(2));
			return log(threehalf + x * half) * log2inv;
		};
		auto coeff = ChebyCoeffs(func, std::numeric_limits<double>::epsilon() * 0.5);
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 log2(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 y, z;\n");
		fprintf(fp, "\tsimd_i64 i;\n");
		fprintf(fp, "\ti = ((simd_i64&) x & simd_i64(0x7FF0000000000000LL));\n");
		fprintf(fp, "\ti >>= (long long)(52);\n");
		fprintf(fp, "\ti -= simd_i64(1023LL);\n");
		fprintf(fp, "\tz = simd_f64(i);\n");
		fprintf(fp, "\ti = (simd_i64&) x;\n");
		fprintf(fp, "\ti &= simd_i64(0xFFFFFFFFFFFFFLL);\n");
		fprintf(fp, "\ti |= simd_i64(0x%llXLL);\n", (long long) 1023 << (long long) 52);
		fprintf(fp, "\tx = (simd_f64&) i;\n");
		fprintf(fp, "\tx *= simd_f64(2);\n");
		fprintf(fp, "\tx -= simd_f64(3);\n");
		int N = coeff.size() - 1;
		fprintf(fp, "\ty = simd_f64(%.17e);\n", coeff[N]);
		for( int n = N - 1; n >= 0; n--){
			fprintf(fp, "\ty = fma(y, x, simd_f64(%.17e));\n", coeff[n]);
		}
		fprintf(fp, "\ty += z;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	{
		constexpr int N = 27;
		polynomial c0;
		long double coeff[N];
		coeff[0] = jnl(0, 1.0L);
		long double b = -2.0L;
		auto Tn = Tns(2 * N);
		for (int n = 1; n < N; n++) {
			coeff[n] = b * jnl(2 * n, 1.0L);
			b = -b;
		}
		for (int n = 0; n < N; n++) {
			c0 = poly_add(c0, poly_mul(poly_term(0, hiprec_real(coeff[n])), Tn[2 * n]));
		}
		const auto pi = 4.0L * atanl(1.0L);
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 cos(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 p, y, sbig;\n");
		fprintf(fp, "\tp = x * simd_f64(%.17e) - simd_f64(0.5);\n", double(1.0L / (2.0L * pi)));
		fprintf(fp, "\tp -= floor(p);\n");
		fprintf(fp, "\tp -= simd_f64(0.5);\n");
		fprintf(fp, "\tx = p * p * simd_f64(%.17e);\n", double(4.0L * pi * pi));
		fprintf(fp, "\ty = simd_f64(%.17e);\n", double(c0[N - 1]));
		for (int n = N - 3; n >= 0; n -= 2) {
			fprintf(fp, "\ty = fma(x, y, simd_f64(%.17e));\n", double(c0[n]));
		}
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	{
		constexpr int N = 13;
		double c0[N];
		int nf = 1;
		for (int n = 0; n < N; n++) {
			c0[n] = 1.0 / nf;
			nf *= (n + 1);
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 exp(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 x0, y;\n");
		fprintf(fp, "\tsimd_i64 i;\n");
		fprintf(fp, "\tx = max(simd_f64(-709), min(simd_f64(710), x));\n");
		fprintf(fp, "\tx0 = round(x * simd_f64(%.20e));\n", 1.0 / M_LN2);
		fprintf(fp, "\tx -= x0 * simd_f64(%.20e);\n", M_LN2);
		fprintf(fp, "\ty = simd_f64(%.20e);\n", c0[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x, y, simd_f64(%.17e));\n", c0[n]);
		}
		fprintf(fp, "\ti = (simd_i64(x0) + simd_i64(1023)) << (long long)(52);\n");
		fprintf(fp, "\ty *= (simd_f64&) i;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}

	{

		constexpr int N = 16;
		constexpr int M = 20;
		const hiprec_real xmax = 26.6;
		static hiprec_real c0[N][M];
		hiprec_real a[M];
		for (int n = 0; n < M; n++) {
			hiprec_real xmin = pow(hiprec_real(2), hiprec_real(n) / hiprec_real(4)) - hiprec_real(1);
			hiprec_real xmax1 = pow(hiprec_real(2), hiprec_real(n + 1) / hiprec_real(4)) - hiprec_real(1);
			if (n != M - 1) {
				a[n] = (xmax1 + xmin) / hiprec_real(2);
			} else {
				a[n] = (xmax + xmin) / hiprec_real(2);
			}
		}
		const hiprec_real pi = hiprec_real(4) * atan(hiprec_real(1));
		for (int m = 0; m < M; m++) {
			c0[0][m] = exp(a[m] * a[m]) * erfc(a[m]);
			c0[1][m] = hiprec_real(2) * (hiprec_real(-1) / sqrt(pi) + a[m] * c0[0][m]);
			for (int n = 2; n < N; n++) {
				c0[n][m] = hiprec_real(2) / hiprec_real(n) * (a[m] * c0[n - 1][m] + c0[n - 2][m]);
			}
		}
		fprintf(fp, "simd_f64 erfc(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 a, b;\n");
		fprintf(fp, "\terfcexp(x, &a, &b);\n");
		fprintf(fp, "\treturn a;\n");
		fprintf(fp, "}\n\n");
		fprintf(fp, "void erfcexp(simd_f64 x, simd_f64* erfcptr, simd_f64* expptr) {\n");
		fprintf(fp, "\tconstexpr static double x0[] = {");
		for (int m = 0; m < M; m++) {
			fprintf(fp, "%24.17e", double(a[m]));
			if (m != M - 1) {
				fprintf(fp, ", ");
			}
		}
		fprintf(fp, "};\n");
		fprintf(fp, "\tconstexpr static double coeff[%i][%i] = {\n", N, M);
		for (int n = 0; n < N; n++) {
			fprintf(fp, "\t\t{");
			for (int m = 0; m < M; m++) {
				fprintf(fp, "%24.17e", double(c0[n][m]));
				if (m != M - 1) {
					fprintf(fp, ", ");
				}
			}
			fprintf(fp, "}");
			if (n != N - 1) {
				fprintf(fp, ", ");
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\t};\n");
		fprintf(fp, "\tsimd_f64 r, y, e, a, q, c, neg, rng;\n");
		fprintf(fp, "\tsimd_i64 i;\n");
		fprintf(fp, "\tneg = simd_f64(x < simd_f64(0));\n");
		fprintf(fp, "\trng = simd_f64(x < simd_f64(%.17e));\n", double(xmax));
		fprintf(fp, "\tx = min(abs(x), simd_f64(%.17e));\n", double(xmax));
		fprintf(fp, "\tq = x + simd_f64(1);\n");
		fprintf(fp, "\tx = min(x, simd_f64(27));\n");
		fprintf(fp, "\tq *= q;\n");
		fprintf(fp, "\tq *= q;\n");
		fprintf(fp, "\ti = ((((simd_i64&) q) & simd_i64(0x7FF0000000000000)) >> (long long)(52)) - simd_i64(1023);\n");
		fprintf(fp, "\te = exp(-x * x);\n");

		fprintf(fp, "\ta.gather(x0, i);\n");
		fprintf(fp, "\tx -= a;\n");
		fprintf(fp, "\ty = c.gather(coeff[%i], i);\n", N - 1);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, c.gather(coeff[%i], i));\n", n);
		}
		fprintf(fp, "\ty *= e * rng;\n");
		fprintf(fp, "\t*expptr = e;\n");
		fprintf(fp, "\t*erfcptr = y * (simd_f64(1) - neg) + (simd_f64(2) - y) * neg;\n");
		fprintf(fp, "}\n");

	}
}

int main() {
	system("mkdir -p ./generated_code\n");
	system("mkdir -p ./generated_code/src/\n");
	FILE* fp = fopen("./generated_code/src/math.cpp", "wt");
	fprintf(fp, "#include \"simd.hpp\"\n");
	fprintf(fp, "\nnamespace simd {\n\n");
	float_funcs(fp);
	double_funcs(fp);
	fprintf(fp, "\n}\n");
	fclose(fp);
	return 0;
}

