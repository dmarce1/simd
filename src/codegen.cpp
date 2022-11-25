#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "../include/simd.hpp"
#include "../include/hiprec.hpp"
#include "../include/polynomial.hpp"

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

hiprec_real factorial(int n) {
	if (n <= 1) {
		return hiprec_real(1);
	} else {
		return hiprec_real(n) * factorial(n - 1);
	}
}

void float_funcs(FILE* fp) {
	{
		constexpr int M = 3;
		static std::vector<double> co[M];
		const hiprec_real xmax(4.1);
		static constexpr double toler = std::numeric_limits<float>::epsilon() * 0.5;
		int N = 0;
		for (int i = 0; i < M; i++) {
			hiprec_real a = hiprec_real(i) * xmax / hiprec_real(M);
			hiprec_real b = hiprec_real(i + 1) * xmax / hiprec_real(M);
			std::function<hiprec_real(hiprec_real)> func = [a,b](hiprec_real x) {
				const auto sum = a + b;
				const auto dif = b - a;
				const auto half = hiprec_real(0.5);
				return erf(half*(sum + dif * x));
			};
			co[i] = ChebyCoeffs(func, toler, 0);
			N = std::max(N, (int) co[i].size());
		}
		for (int n = 0; n < N; n++) {
			for (int m = 0; m < M; m++) {
				if (co[m].size() < n + 1) {
					co[m].push_back(0.0);
				}
			}
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 erf(simd_f32 x) {\n");
		fprintf(fp, "\tstatic const simd_f32 co[] = {\n");
		for (int n = 0; n < N; n += 2) {
			fprintf(fp, "\t\t{");
			for (int i = n; i < n + 2; i++) {
				double c1, c2, c3, c4;
				if (i < N) {
					int m = i / 2;
					const auto pi = hiprec_real(4) * atan(hiprec_real(1));
					c1 = (i % 2 == 1) ? (double) (hiprec_real(2) / sqrt(pi) * hiprec_real(m % 2 == 0 ? 1 : -1) / factorial(m) / hiprec_real(2 * m + 1)) : 0.0;
					c2 = co[0][i];
					c3 = co[1][i];
					c4 = co[2][i];
					const double factor = (double) (pow(hiprec_real(2) * hiprec_real(M) / xmax, hiprec_real(i)));
//					c1 *= factor;// * ((1<<(i)) * 0.5);
					//	c1 *= 2.0;
					printf("%e\n", c1);
					c2 *= factor;
					c3 *= factor;
					c4 *= factor;
				} else {
					c1 = c3 = c4 = c2 = 99.0;
				}
				fprintf(fp, "%.17e, %.17e, %.17e, %.17e", c1, c2, c3, c4);
				if (i != n + 1) {
					fprintf(fp, ",");
				}
				fprintf(fp, " ");
			}
			fprintf(fp, "}");
			if (n + 2 < N) {
				fprintf(fp, ",");
			}
			fprintf(fp, "\n");
		}
		auto sqrtpi = sqrt(hiprec_real(4) * atan(hiprec_real(1)));
		fprintf(fp, "\t};\n");
		fprintf(fp, "\tsimd_f32 y, s, z, x2;\n");
		fprintf(fp, "\tsimd_i32 i0, i1, l;\n");
		fprintf(fp, "\ts = copysign(simd_f32(1), x);\n");
		fprintf(fp, "\tx = abs(x);\n");
		fprintf(fp, "\tx2 = x * x;\n");
		fprintf(fp, "\tl = x < simd_f32(0.15);\n");
		fprintf(fp, "\tz = simd_f32(1) - simd_f32(l);\n");
		fprintf(fp, "\tx = min(x, simd_f32(%.17e));\n", (double) xmax - 0.00001);
		fprintf(fp, "\ti0 = x * simd_f32(%.17e);\n", (double) (hiprec_real(M) / xmax));
		fprintf(fp, "\ti0 += simd_i32(1);\n");
		fprintf(fp, "\ti0 -= l;\n");
		fprintf(fp, "\ti1 = i0 + simd_i32(4);\n");
		fprintf(fp, "\tx -= z * simd_f32(%.17e) * i0;\n", (double) (xmax / hiprec_real(M)));
		fprintf(fp, "\tx -= z * simd_f32(%.17e);\n", (double) (xmax * hiprec_real(0.5) / hiprec_real(M)) - (double) (xmax / hiprec_real(M)));
//		fprintf(fp, "\tx *= simd_f32(%.17e);\n",);
//		fprintf(fp, "\tx = blend(x, x0, l);\n");
		fprintf(fp, "\ty = co[%i].permute(i%i);\n", (N - 1) / 2, (N - 1) % 2);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, co[%i].permute(i%i));\n", n / 2, n % 2);
		}
		fprintf(fp, "\treturn s * y;\n");
		fprintf(fp, "}\n");
	}
	{
		constexpr int M = 4;
		static std::vector<double> co[M];
		static constexpr double toler = std::numeric_limits<float>::epsilon() * 0.5;
		int N = 0;
		for (int i = 0; i < M; i++) {
			hiprec_real a = hiprec_real(i) / hiprec_real(M) + hiprec_real(1);
			hiprec_real b = hiprec_real(i + 1) / hiprec_real(M) + hiprec_real(1);
			std::function<hiprec_real(hiprec_real)> func = [a,b](hiprec_real x) {
				const auto sum = a + b;
				const auto dif = b - a;
				const auto half = hiprec_real(0.5);
				return gamma(half*(sum + dif * x));
			};
			co[i] = ChebyCoeffs(func, toler, 0);
			N = std::max(N, (int) co[i].size());
		}
		for (int n = 0; n < N; n++) {
			for (int m = 0; m < M; m++) {
				if (co[m].size() < n + 1) {
					co[m].push_back(0.0);
				}
			}
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 tgamma(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 y, z, x0, x1, s;\n");
		fprintf(fp, "\tsimd_i32 i0, i1, n, c;\n");
		fprintf(fp, "\tstatic const simd_f32 co[] = {\n");
		for (int n = 0; n < N; n += 2) {
			fprintf(fp, "\t\t{");
			for (int i = n; i < n + 2; i++) {
				double c1, c2, c3, c4;
				if (i < N) {
					c1 = co[0][i];
					c2 = co[1][i];
					c3 = co[2][i];
					c4 = co[3][i];
				} else {
					c1 = c3 = c4 = c2 = 99.0;
				}
				fprintf(fp, "%.17e, %.17e, %.17e, %.17e", c1, c2, c3, c4);
				if (i != n + 1) {
					fprintf(fp, ",");
				}
				fprintf(fp, " ");
			}
			fprintf(fp, "}");
			if (n + 2 < N) {
				fprintf(fp, ",");
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\t};\n");
		double fac = 1.0;
		fprintf(fp, "\tn = x < simd_f32(0);\n");
		fprintf(fp, "\tc = (simd_i32(1) - n) * (x < simd_f32(1));\n");
		fprintf(fp, "\tz = simd_f32(1) - x;\n");
		fprintf(fp, "\tx = blend(x, z, n);\n");
		fprintf(fp, "\tx += simd_f32(c);\n");
		fprintf(fp, "\tz = x;\n");
		fprintf(fp, "\ts = sin(simd_f32(M_PI) * x);\n");
		fprintf(fp, "\tx -= floor(x);\n");
		fprintf(fp, "\ti0 = x * simd_f32(4);\n");
		fprintf(fp, "\ti1 = i0 + simd_f32(4);\n");
		fprintf(fp, "\tx1 = (simd_f32(i0) * simd_f32(0.25) + simd_f32(0.125));\n");
		fprintf(fp, "\tx0 = simd_f32(8) * (x - x1);\n");
		fprintf(fp, "\ty = co[%i].permute(i%i);\n", (N - 1) / 2, (N - 1) % 2);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x0, co[%i].permute(i%i));\n", n / 2, n % 2);
		}
		fprintf(fp, "\tfor (int i = 0; i < 33; i++) {\n");
		fprintf(fp, "\t\tz = max(z - simd_f32(1), simd_f32(1));\n");
		fprintf(fp, "\t\ty *= z;\n");
		fprintf(fp, "\t}\n");
		fprintf(fp, "\ty = blend(y, y / x, c);\n");
		fprintf(fp, "\ty = blend(y, simd_f32(M_PI) / (s * y), n);\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}

	{
		static std::vector<double> co1;
		static std::vector<double> co2;
		static const hiprec_real z0(0.5);
		static constexpr double toler = std::numeric_limits<float>::epsilon() * 0.5;
		auto func1 = [](hiprec_real x) {
			return asin((hiprec_real(1) + x) * z0 / hiprec_real(2));
		};
		auto func3 = [](hiprec_real y) {
			const auto x = hiprec_real(1) - y * y;
			const auto a = asin(x);
			static const auto b = hiprec_real(2) * atan(hiprec_real(1));
			const auto c = sqrt(hiprec_real(1) - x);
			return (b - a)/c;
		};
		auto func2 = [func3](hiprec_real x) {
			const static auto z1 = sqrt(hiprec_real(1) - z0);
			return func3((z1 + x * hiprec_real(z1)) / hiprec_real(2));
		};
		co1 = ChebyCoeffs(func1, toler, 0);
		co2 = ChebyCoeffs(func2, toler, 0);
		int N = std::max(co1.size(), co2.size());
		for (int n = 0; n < N; n++) {
			if (co1.size() < n + 1) {
				co1.push_back(0.0);
			}
			if (co2.size() < n + 1) {
				co2.push_back(0.0);
			}
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 asin(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 y, s, z, x0, x1;\n");
		fprintf(fp, "\tsimd_i32 i0, i1, i2, i3;\n");
		fprintf(fp, "\tstatic const simd_f32 co[] = {\n");
		for (int n = 0; n < N; n += 4) {
			fprintf(fp, "\t\t{");
			for (int i = n; i < n + 4; i++) {
				double c1, c2;
				if (i < N) {
					c1 = co1[i];
					c2 = co2[i];
				} else {
					c1 = 99.0;
					c2 = 99.0;
				}
				fprintf(fp, "%.17e, %.17e", c1, c2);
				if (i != n + 3) {
					fprintf(fp, ",");
				}
				fprintf(fp, " ");
			}
			fprintf(fp, "}");
			if (n + 4 < N) {
				fprintf(fp, ",");
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\t};\n");
		fprintf(fp, "\ts = copysign(simd_f32(1), x);\n");
		fprintf(fp, "\tx = abs(x);\n");
		fprintf(fp, "\ti0 = x > simd_f32(%.17e);\n", (double) z0);
		fprintf(fp, "\tz = sqrt(simd_f32(1) - x);\n");
		fprintf(fp, "\tx0 = simd_f32(%.17e) * x - simd_f32(1);\n", (double) (hiprec_real(2) / z0));
		fprintf(fp, "\tx1 = simd_f32(%.17e) * z - simd_f32(1);\n", (double) (hiprec_real(2) / sqrt(hiprec_real(1) - z0)));
		fprintf(fp, "\tx = blend(x0, x1, i0);\n");
		fprintf(fp, "\ti1 = i0 + simd_i32(2);\n");
		fprintf(fp, "\ti2 = i0 + simd_i32(4);\n");
		fprintf(fp, "\ti3 = i0 + simd_i32(6);\n");
		fprintf(fp, "\ty = co[%i].permute(i%i);\n", (N - 1) / 4, (N - 1) % 4);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, co[%i].permute(i%i));\n", n / 4, n % 4);
		}
		fprintf(fp, "\tz = simd_f32(%.17e) - y * z;\n", (double) (hiprec_real(2) * atan(hiprec_real(1))));
		fprintf(fp, "\ty = blend(y, z, i0);\n");
		fprintf(fp, "\treturn s * y;\n");
		fprintf(fp, "}\n");
	}

	{
		std::function<hiprec_real(hiprec_real)> func = [](hiprec_real x) {
			static const auto c0 = hiprec_real(3) / (hiprec_real(2) * sqrt(hiprec_real(2)));
			static const auto c1 = hiprec_real(1) / (hiprec_real(2) * sqrt(hiprec_real(2)));
			static const auto log2inv = hiprec_real(1) / log(hiprec_real(2));
			return log(c0 + x * c1) * log2inv;
		};
		auto coeff = ChebyCoeffs(func, std::numeric_limits<float>::epsilon() * double(1 << 10));
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
		for (int n = N - 1; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, simd_f32(%.17e));\n", coeff[n]);
		}
		fprintf(fp, "\ty += z;\n");
		fprintf(fp, "\ty += simd_f32(0.5);\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 log(simd_f32 x) {\n");
		fprintf(fp, "\treturn log2(x) * simd_f32(%.17e);\n", (double) hiprec_real(1) / log2(exp(hiprec_real(1))));
		fprintf(fp, "}\n");
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 log10(simd_f32 x) {\n");
		fprintf(fp, "\treturn log2(x) * simd_f32(%.17e);\n", (double) hiprec_real(1) / log2(hiprec_real(10)));
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
		constexpr int M = 6;
		static std::vector<double> co[M];
		const hiprec_real xmax(6.5);
		static constexpr double toler = std::numeric_limits<double>::epsilon() * 0.5;
		int N = 0;
		for (int i = 0; i < M; i++) {
			hiprec_real a = hiprec_real(i) * xmax / hiprec_real(M);
			hiprec_real b = hiprec_real(i + 1) * xmax / hiprec_real(M);
			std::function<hiprec_real(hiprec_real)> func = [a,b](hiprec_real x) {
				const auto sum = a + b;
				const auto dif = b - a;
				const auto half = hiprec_real(0.5);
				return erf(half*(sum + dif * x));
			};
			co[i] = ChebyCoeffs(func, toler, 0);
			N = std::max(N, (int) co[i].size());
		}
		for (int n = 0; n < N; n++) {
			for (int m = 0; m < M; m++) {
				if (co[m].size() < n + 1) {
					co[m].push_back(0.0);
				}
			}
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 erf(simd_f64 x) {\n");
		fprintf(fp, "\tstatic const double co[][%i] = {\n", M + 1);
		for (int n = 0; n < N; n ++) {
			fprintf(fp, "\t\t{");
			double c1, c2, c3, c4, c5, c6, c7;
			if (n < N) {
				int m = n / 2;
				const auto pi = hiprec_real(4) * atan(hiprec_real(1));
				c1 = (n % 2 == 1) ? (double) (hiprec_real(2) / sqrt(pi) * hiprec_real(m % 2 == 0 ? 1 : -1) / factorial(m) / hiprec_real(2 * m + 1)) : 0.0;
				c2 = co[0][n];
				c3 = co[1][n];
				c4 = co[2][n];
				c5 = co[3][n];
				c6 = co[4][n];
				c7 = co[5][n];
				const double factor = (double) (pow(hiprec_real(2) * hiprec_real(M) / xmax, hiprec_real(n)));
				c2 *= factor;
				c3 *= factor;
				c4 *= factor;
				c5 *= factor;
				c6 *= factor;
				c7 *= factor;
			} else {
				c7 = c5 = c6 = c1 = c3 = c4 = c2 = 99.0;
			}
			fprintf(fp, "%24.17e, %24.17e, %24.17e, %24.17e, %24.17e, %24.17e, %24.17e", c1, c2, c3, c4, c5, c6, c7);
			fprintf(fp, "}");
			if (n + 1 < N) {
				fprintf(fp, ",");
			}
			fprintf(fp, "\n");
		}
		auto sqrtpi = sqrt(hiprec_real(4) * atan(hiprec_real(1)));
		fprintf(fp, "\t};\n");
		fprintf(fp, "\tsimd_f64 y, s, z, x2, c;\n");
		fprintf(fp, "\tsimd_i64 i0, i1, l;\n");
		fprintf(fp, "\ts = copysign(simd_f64(1), x);\n");
		fprintf(fp, "\tx = abs(x);\n");
		fprintf(fp, "\tx2 = x * x;\n");
		fprintf(fp, "\tl = x < simd_f64(0.4);\n");
		fprintf(fp, "\tz = simd_f64(1) - simd_f64(l);\n");
		fprintf(fp, "\tx = min(x, simd_f64(%.17e));\n", (double) xmax - 0.00001);
		fprintf(fp, "\ti0 = x * simd_f64(%.17e);\n", (double) (hiprec_real(M) / xmax));
		fprintf(fp, "\ti0 += simd_i64(1);\n");
		fprintf(fp, "\ti0 -= l;\n");
		fprintf(fp, "\tx -= z * simd_f64(%.17e) * i0;\n", (double) (xmax / hiprec_real(M)));
		fprintf(fp, "\tx -= z * simd_f64(%.17e);\n", (double) (xmax * hiprec_real(0.5) / hiprec_real(M)) - (double) (xmax / hiprec_real(M)));
		fprintf(fp, "\ty = c.gather(co[%i], i0);\n", (N - 1));
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, c.gather(co[%i], i0));\n", n );
		}
		fprintf(fp, "\treturn s * y;\n");
		fprintf(fp, "}\n");
	}
	{
		constexpr int M = 4;
		static std::vector<double> co[M];
		static constexpr double toler = std::numeric_limits<double>::epsilon() * 0.5;
		int N = 0;
		for (int i = 0; i < M; i++) {
			hiprec_real a = hiprec_real(i) / hiprec_real(M) + hiprec_real(1);
			hiprec_real b = hiprec_real(i + 1) / hiprec_real(M) + hiprec_real(1);
			std::function<hiprec_real(hiprec_real)> func = [a,b](hiprec_real x) {
				const auto sum = a + b;
				const auto dif = b - a;
				const auto half = hiprec_real(0.5);
				return gamma(half*(sum + dif * x));
			};
			co[i] = ChebyCoeffs(func, toler, 0);
			N = std::max(N, (int) co[i].size());
		}
		for (int n = 0; n < N; n++) {
			for (int m = 0; m < M; m++) {
				if (co[m].size() < n + 1) {
					co[m].push_back(0.0);
				}
			}
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 tgamma(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 y, z, x0, x1, s, c0;\n");
		fprintf(fp, "\tsimd_i64 i0, n, c;\n");
		fprintf(fp, "\tstatic constexpr double co[][4] = {\n");
		for (int n = 0; n < N; n++) {
			fprintf(fp, "\t\t{");
			for (int i = n; i < n + 1; i++) {
				double c1, c2, c3, c4;
				if (i < N) {
					c1 = co[0][i];
					c2 = co[1][i];
					c3 = co[2][i];
					c4 = co[3][i];
				} else {
					c1 = c3 = c4 = c2 = 99.0;
				}
				fprintf(fp, "%.17e, %.17e, %.17e, %.17e", c1, c2, c3, c4);
				if (i != n) {
					fprintf(fp, ",");
				}
				fprintf(fp, " ");
			}
			fprintf(fp, "}");
			if (n + 1 < N) {
				fprintf(fp, ",");
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\t};\n");
		double fac = 1.0;
		fprintf(fp, "\tn = x < simd_f64(0);\n");
		fprintf(fp, "\tc = (simd_i64(1) - n) && (x < simd_f64(1));\n");
		fprintf(fp, "\tz = simd_f64(1) - x;\n");
		fprintf(fp, "\tx = blend(x, z, n);\n");
		fprintf(fp, "\tx += simd_f64(c);\n");
		fprintf(fp, "\tz = x;\n");
		fprintf(fp, "\ts = sin(simd_f64(M_PI) * x);\n");
		fprintf(fp, "\tx -= floor(x);\n");
		fprintf(fp, "\ti0 = x * simd_f64(4);\n");
		fprintf(fp, "\tx1 = (simd_f64(i0) * simd_f64(0.25) + simd_f64(0.125));\n");
		fprintf(fp, "\tx0 = simd_f64(8) * (x - x1);\n");
		fprintf(fp, "\ty = c0.gather(co[%i], i0);\n", (N - 1));
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x0, c0.gather(co[%i], i0));\n", n);
		}
		fprintf(fp, "\tfor (int i = 0; i < 150; i++) {\n");
		fprintf(fp, "\t\tz = max(z - simd_f64(1), simd_f64(1));\n");
		fprintf(fp, "\t\ty *= z;\n");
		fprintf(fp, "\t}\n");
		fprintf(fp, "\ty = blend(y, y / x, c);\n");
		fprintf(fp, "\ty = blend(y, simd_f64(M_PI) / (s * y), n);\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	{
		static std::vector<double> co1;
		static std::vector<double> co2;
		static const hiprec_real z0(0.5);
		static constexpr double toler = std::numeric_limits<double>::epsilon() * 0.5;
		auto func1 = [](hiprec_real x) {
			return asin((hiprec_real(1) + x) * z0 / hiprec_real(2));
		};
		auto func3 = [](hiprec_real y) {
			const auto x = hiprec_real(1) - y * y;
			const auto a = asin(x);
			static const auto b = hiprec_real(2) * atan(hiprec_real(1));
			const auto c = sqrt(hiprec_real(1) - x);
			return (b - a)/c;
		};
		auto func2 = [func3](hiprec_real x) {
			const static auto z1 = sqrt(hiprec_real(1) - z0);
			return func3((z1 + x * hiprec_real(z1)) / hiprec_real(2));
		};
		co1 = ChebyCoeffs(func1, toler, 0);
		co2 = ChebyCoeffs(func2, toler, 0);
		int N = std::max(co1.size(), co2.size());
		for (int n = 0; n < N; n++) {
			if (co1.size() < n + 1) {
				co1.push_back(0.0);
			}
			if (co2.size() < n + 1) {
				co2.push_back(0.0);
			}
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 asin(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 y, s, z, w, x0, x1;\n");
		fprintf(fp, "\tsimd_i64 i;\n");
		fprintf(fp, "\tsize_t j;\n");
		fprintf(fp, "\tstatic const simd_f64 co[][%i] = {\n", N);
		for (int bits = 0; bits < 16; bits++) {
			fprintf(fp, "\t\t{\n");
			for (int n = 0; n < N; n++) {
				fprintf(fp, "\t\t\t{");
				for (int i = 0; i < 4; i++) {
					double c1;
					if ((bits >> i) & 0x1) {
						c1 = co2[n];
					} else {
						c1 = co1[n];
					}
					fprintf(fp, "%.17e", c1);
					if (i != 3) {
						fprintf(fp, ", ");
					}
				}
				fprintf(fp, "}");
				if (n + 1 < N) {
					fprintf(fp, ",");
				}
				fprintf(fp, "\n");
			}
			fprintf(fp, "\t\t}");
			if (bits + 1 < 16) {
				fprintf(fp, ",");
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\t};\n");
		fprintf(fp, "\ts = copysign(simd_f64(1), x);\n");
		fprintf(fp, "\tx = abs(x);\n");
		fprintf(fp, "\ti = x > simd_f64(%.17e);\n", (double) z0);
		fprintf(fp, "\tz = sqrt(simd_f64(1) - x);\n");
		fprintf(fp, "\tx0 = simd_f64(%.17e) * x - simd_f64(1);\n", (double) (hiprec_real(2) / z0));
		fprintf(fp, "\tx1 = simd_f64(%.17e) * z - simd_f64(1);\n", (double) (hiprec_real(2) / sqrt(hiprec_real(1) - z0)));
		fprintf(fp, "\tx = blend(x0, x1, i);\n");
		fprintf(fp, "\ti = -i;\n");
		fprintf(fp, "\tj =  _mm256_movemask_pd(((simd_f64&) i).v);\n");
		fprintf(fp, "\ty = co[j][%i];\n", (N - 1));
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, co[j][%i]);\n", n);
		}
		fprintf(fp, "\tz = simd_f64(%.17e) - y * z;\n", (double) (hiprec_real(2) * atan(hiprec_real(1))));
		fprintf(fp, "\ty = blend(y, z, -i);\n");
		fprintf(fp, "\treturn s * y;\n");
		fprintf(fp, "}\n");
	}
	{
		std::function<hiprec_real(hiprec_real)> func = [](hiprec_real x) {
			static const auto c0 = hiprec_real(3) / (hiprec_real(2) * sqrt(hiprec_real(2)));
			static const auto c1 = hiprec_real(1) / (hiprec_real(2) * sqrt(hiprec_real(2)));
			static const auto log2inv = hiprec_real(1) / log(hiprec_real(2));
			return log(c0 + x * c1) * log2inv;
		};
		auto coeff = ChebyCoeffs(func, std::numeric_limits<double>::epsilon() * double(1 << 21));
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
		for (int n = N - 1; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, simd_f64(%.17e));\n", coeff[n]);
		}
		fprintf(fp, "\ty += z;\n");
		fprintf(fp, "\ty += simd_f64(0.5);\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 log(simd_f64 x) {\n");
		fprintf(fp, "\treturn log2(x) * simd_f64(%.17e);\n", (double) hiprec_real(1) / log2(exp(hiprec_real(1))));
		fprintf(fp, "}\n");
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 log10(simd_f64 x) {\n");
		fprintf(fp, "\treturn log2(x) * simd_f64(%.17e);\n", (double) hiprec_real(1) / log2(hiprec_real(10)));
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

