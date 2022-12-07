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

double zeta(int s) {
	double_2 sum = 0.0;
	int nmax = pow((1ULL << 58), 1.0 / s);
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
	sum = sum / double_2(s - 1);
	return (double) sum.x + (double) sum.y;
}

hiprec_real factorial(int n) {
	if (n <= 1) {
		return hiprec_real(1);
	} else {
		return hiprec_real(n) * factorial(n - 1);
	}
}

void include(FILE* fp, std::string filename) {
	constexpr int N = 1024;
	char buffer[N];
	FILE* fp0 = fopen(filename.c_str(), "rt");
	if (!fp0) {
		printf("Unable to open %s\n", filename.c_str());
		abort();
	}
	while (!feof(fp0)) {
		if (!fgets(buffer, N, fp0)) {
			break;
		}
		fprintf(fp, "%s", buffer);
	}
	fclose(fp0);
}

void float_funcs(FILE* fp) {
	include(fp, "../include/code.hpp");

	/* cos */
	{
		constexpr int N = 6;
		float coeff[N];
		hiprec_real pi_exact = hiprec_real(4) * atan(hiprec_real(1));
		float pi1 = pi_exact;
		float pi2 = pi_exact - hiprec_real(pi1);
		hiprec_real fac(1);
		for (int n = 0; n < N; n++) {
			coeff[n] = hiprec_real(1) / fac;
			fac *= hiprec_real(2 * n + 2);
			fac *= -hiprec_real(2 * n + 3);
		}
		fprintf(fp, "\nsimd_f32 cos(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 x0, s, x2, y;\n");
		fprintf(fp, "\tx = abs(x);\n");
		fprintf(fp, "\tx0 = floor(x * simd_f32(%.9e));\n", 1.0 / M_PI);
		fprintf(fp, "\ts = simd_f32(2) * (x0 - simd_f32(2) * floor(x0 * simd_f32(0.5))) - simd_f32(1);\n");
		fprintf(fp, "\tx0 += simd_f32(0.5);\n");
		fprintf(fp, "\tx = s * fma(-x0, simd_f32(%.9e), fma(-x0, simd_f32(%.9e), x));\n", pi2, pi1);
		fprintf(fp, "\tx2 = x * x;\n");
		fprintf(fp, "\ty = simd_f32(%.9e);\n", coeff[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x2, y, simd_f32(%.9e));\n", coeff[n]);
		}
		fprintf(fp, "\ty *= x;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	/* sin */
	{
		constexpr int N = 6;
		float coeff[N];
		hiprec_real pi_exact = hiprec_real(4) * atan(hiprec_real(1));
		float pi1 = pi_exact;
		float pi2 = pi_exact - hiprec_real(pi1);
		hiprec_real fac(1);
		for (int n = 0; n < N; n++) {
			coeff[n] = hiprec_real(1) / fac;
			fac *= hiprec_real(2 * n + 2);
			fac *= -hiprec_real(2 * n + 3);
		}
		fprintf(fp, "\nsimd_f32 sin(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 x0, s, x2, y;\n");
		fprintf(fp, "\tx0 = round(x * simd_f32(%.9e));\n", 1.0 / M_PI);
		fprintf(fp, "\ts = -simd_f32(2) * (x0 - simd_f32(2) * floor(x0 * simd_f32(0.5))) + simd_f32(1);\n");
		fprintf(fp, "\tx = s * fma(-x0, simd_f32(%.9e), fma(-x0, simd_f32(%.9e), x));\n", pi2, pi1);
		fprintf(fp, "\tx2 = x * x;\n");
		fprintf(fp, "\ty = simd_f32(%.9e);\n", coeff[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x2, y, simd_f32(%.9e));\n", coeff[n]);
		}
		fprintf(fp, "\ty *= x;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}

	/* erf */
	{
		constexpr int M = 3;
		static std::vector<hiprec_real> co[M];
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
				fprintf(fp, "%.9e, %.9e, %.9e, %.9e", c1, c2, c3, c4);
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
		fprintf(fp, "\tx = fabs(x);\n");
		fprintf(fp, "\tx2 = x * x;\n");
		fprintf(fp, "\tl = x < simd_f32(0.15);\n");
		fprintf(fp, "\tz = simd_f32(1) - simd_f32(l);\n");
		fprintf(fp, "\tx = min(x, simd_f32(%.9e));\n", (double) xmax - 0.00001);
		fprintf(fp, "\ti0 = x * simd_f32(%.9e);\n", (double) (hiprec_real(M) / xmax));
		fprintf(fp, "\ti0 += simd_i32(1);\n");
		fprintf(fp, "\ti0 -= l;\n");
		fprintf(fp, "\ti1 = i0 + simd_i32(4);\n");
		fprintf(fp, "\tx -= z * simd_f32(%.9e) * i0;\n", (double) (xmax / hiprec_real(M)));
		fprintf(fp, "\tx -= z * simd_f32(%.9e);\n", (double) (xmax * hiprec_real(0.5) / hiprec_real(M)) - (double) (xmax / hiprec_real(M)));
//		fprintf(fp, "\tx *= simd_f32(%.9e);\n",);
//		fprintf(fp, "\tx = blend(x, x0, l);\n");
		fprintf(fp, "\ty = co[%i].permute(i%i);\n", (N - 1) / 2, (N - 1) % 2);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, co[%i].permute(i%i));\n", n / 2, n % 2);
		}
		fprintf(fp, "\treturn s * y;\n");
		fprintf(fp, "}\n");
	}
	/* tgamma */
	{
		static bool init = false;
		static constexpr int M = 13;
		static constexpr int N = 2 * M;
		static constexpr int Nsin = 7;
		static float coeffs[Nsin];
		static constexpr int Nsets = 11;
		static hiprec_real coeff[M][Nsets];
		static float coeff0[M][Nsets];
		static float ehi, elo;
		int xa = 7.5;
		float xb = 2.5;
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
				coeff[n][10] = (A[2 * n] * sqrt(hiprec_real(2)) * gamma(hiprec_real(n) + hiprec_real(0.5)) / gamma(hiprec_real(0.5)));
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
				auto chebies = ChebyCoeffs(func, std::numeric_limits<float>::epsilon() * 0.005, 0);
				chebies.resize(M, 0.0);
				for (int m = 0; m < M; m++) {
					coeff[m][4 + n] = chebies[m];
				}
			}
			ehi = exp(hiprec_real(-1));
			elo = exp(hiprec_real(-1)) - hiprec_real(ehi);
			coeff[0][0] = 0.0;
			coeff[0][1] = logl(tgammal(1.5L));
			coeff[0][2] = 0.0;
			coeff[0][3] = logl(tgammal(2.5L));
			coeff[1][0] = -.57721566490153286060651209008240243104215933593992L;
			coeff[1][2] = -.57721566490153286060651209008240243104215933593992L + 1.0L;
			double_2 sum = -.57721566490153286060651209008240243104215933593992L - 2.0L / 3.0L;
			coeff[1][1] = 0.0364899739785765205590237L;
			coeff[1][3] = 0.7031566406452431872256903336679110677L;
			for (int n = 2; n < M; n++) {
				long double z = zeta(n);
				hiprec_real sgn = n % 2 == 1 ? -hiprec_real(1) : hiprec_real(1);
				hiprec_real invn = hiprec_real(1) / hiprec_real(n);
				hiprec_real zon = hiprec_real(z) / hiprec_real(n);
				coeff[n][0] = sgn * zon;
				coeff[n][1] = sgn * zon * hiprec_real((1 << n) - 1);
				coeff[n][1] -= sgn * pow(hiprec_real(0.5), hiprec_real(-n)) * invn;
				coeff[n][2] = coeff[n][0] - hiprec_real(sgn / hiprec_real(n));
				coeff[n][3] = coeff[n][1] - sgn * pow(hiprec_real(3) / hiprec_real(2), hiprec_real(-n)) * invn;
			}
			for (int m = 0; m < M; m++) {
				for (int n = 0; n < Nsets; n++) {
					coeff0[m][n] = coeff[m][n];
				}
			}
		}
		fprintf(fp, "\nsimd_f32 tgamma(simd_f32 x) {\n");
		fprintf(fp, "\tstatic const simd_f32_2 E(%.9e, %.9e);\n", ehi, elo);
		fprintf(fp, "\tstatic constexpr float coeffs[][%i] = { \n", Nsets);
		for (int m = 0; m < M; m++) {
			fprintf(fp, "\t\t{ ");
			for (int n = 0; n < Nsets; n++) {
				fprintf(fp, "%15.9e%s ", coeff0[m][n], n == Nsets - 1 ? "" : ",");
			}
			fprintf(fp, " }%s\n", m == M - 1 ? "" : ",");
		}
		fprintf(fp, "\t};\n");
		fprintf(fp, "\tsimd_f32 y, z, x0, x1, nf, x2, c, xc, a, b, sgn;\n"
				"\tsimd_i32 neg, inv, ni, xoa, xob, flag;\n"
				"\tneg = x < simd_f32(0);\n"
				"\tx0 = x + simd_f32(1);\n"
				"\tx0 = blend(abs(x), x0, x >= simd_f32(-1));\n"
				"\tx0 = blend(x, x0, neg);\n"
				"\tinv = x0 < simd_f32(1);\n"
				"\tx0 = blend(x0, x0 + simd_f32(1), inv);\n"
				"\tx1 = round(simd_f32(2) * x0) * simd_f32(0.5);\n"
				"\txoa = x0 > simd_f32(10.0);\n"
				"\txob = x0 > simd_f32(2.5);\n"
				"\tnf = round(simd_f32(2) * x1 - simd_f32(2));\n"
				"\tnf = blend(nf, floor(x0 * simd_f32(0.8)) + simd_i32(2), xob);\n"
				"\tnf = blend(nf, simd_f32(10), xoa);\n"
				"\txc = simd_f32(1.25) * (simd_f32(nf) - simd_f32(1.5));\n"
				"\tz = x0 - x1;\n"
				"\tz = blend(z, simd_f32(1.6) * (x0 - xc), xob);\n"
				"\tz = blend(z, simd_f32(1) / x0, xoa);\n"
				"\tni = simd_i32(round(nf));\n"
				"");
		fprintf(fp, "\ty = c.gather(coeffs[%i], ni);\n", M - 1);
		for (int k = M - 1; k >= 0; k--) {
			fprintf(fp, "\ty = fma(y, z, c.gather(coeffs[%i], ni));\n", k);
		}
		fprintf(fp, "\tflag = ni >= simd_i32(4);\n"
				"\ta = exp(y);\n"
				"\tsimd_f32_2 A = simd_f32_2(x0) * E;\n"
				"\tb = pow(A.x, x0) * (simd_f32(1) + x0 * A.y / A.x);\n"
				"\tb = y * b * sqrt(simd_f32(2.0 * M_PI) / x0);\n"
				"\ty = blend(a, b, flag);\n"
				"\ty = blend(y, y / abs(x), inv);\n"
				"\tx2 = x - floor(x);\n"
				"\tx0 = blend(x2, simd_f32(1) - (x - floor(x)), x2 > simd_f32(0.5));\n"
				"\tsgn = simd_f32(1);\n"
				"\tnf = abs(floor(abs(x) * simd_f32(2)) * simd_f32(0.5));\n"
				"\tni = simd_i32(nf);\n"
				"\tsgn = blend(sgn, -sgn, (ni & simd_i32(1)) != simd_i32(0));\n"
				"\tx2 = x0 * x0;\n"
				"");
		fprintf(fp, "\tz = simd_f32(%.9e);\n", coeffs[Nsin - 1]);
		for (int k = Nsin - 1; k >= 0; k--) {
			fprintf(fp, "\tz = fma(z, x2, simd_f32(%.9e));\n", coeffs[k]);
		}
		fprintf(fp, "\tz *= x0 * sgn;\n");
		fprintf(fp, "\tz = simd_f32(1) / (x * y * z);\n");
		fprintf(fp, "\tz = blend(z, y / -(simd_f32(x) + simd_f32(1)), inv);\n");
		fprintf(fp, "\ty = blend(y, z, neg);\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");

	}

	/* asin */
	{
		static std::vector<hiprec_real> co1;
		static std::vector<hiprec_real> co2;
		static const hiprec_real z0(0.5);
		static constexpr double toler = std::numeric_limits<float>::epsilon() * 0.5;
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
		co2 = ChebyCoeffs(func2, toler, 0);
		do {
			int n = co1.size();
			co1.push_back(
					factorial(hiprec_real(2 * n))
							/ (pow(hiprec_real(2), hiprec_real(2 * n)) * factorial(hiprec_real(n)) * factorial(hiprec_real(n)) * (hiprec_real(2 * n + 1))));

		} while (std::abs(double(co1.back()) * pow(double(z0), 2 * co1.size() - 1)) > toler);
		auto tmp = co1;
		co1.resize(0);
		for (int i = 0; i < tmp.size(); i++) {
			co1.push_back(0.0);
			co1.push_back(tmp[i]);
		}
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
				fprintf(fp, "%.9e, %.9e", c1, c2);
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
		fprintf(fp, "\tx = fabs(x);\n");
		fprintf(fp, "\ti0 = x > simd_f32(%.9e);\n", (double) z0);
		fprintf(fp, "\tz = sqrt(simd_f32(1) - x);\n");
		fprintf(fp, "\tx0 = x;\n");
		fprintf(fp, "\tx1 = simd_f32(%.9e) * z - simd_f32(1);\n", (double) (hiprec_real(2) / sqrt(hiprec_real(1) - z0)));
		fprintf(fp, "\tx = blend(x0, x1, i0);\n");
		fprintf(fp, "\ti1 = i0 + simd_i32(2);\n");
		fprintf(fp, "\ti2 = i0 + simd_i32(4);\n");
		fprintf(fp, "\ti3 = i0 + simd_i32(6);\n");
		fprintf(fp, "\ty = co[%i].permute(i%i);\n", (N - 1) / 4, (N - 1) % 4);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, co[%i].permute(i%i));\n", n / 4, n % 4);
		}
		fprintf(fp, "\tz = simd_f32(%.9e) - y * z;\n", (double) (hiprec_real(2) * atan(hiprec_real(1))));
		fprintf(fp, "\ty = blend(y, z, i0);\n");
		fprintf(fp, "\treturn s * y;\n");
		fprintf(fp, "}\n");
	}
	/* acos */
#define List(...) {__VA_ARGS__}
	{
		constexpr int N = 18;
		constexpr float coeffs[] =
				{ 0, 2., 0.3333333333, 0.08888888889, 0.02857142857, 0.01015873016, 0.003848003848, 0.001522287237, 0.0006216006216, 0.0002600159463,
						0.0001108489034, 0.00004798653828, 0.00002103757656, 9.321264693e-6, 4.167443738e-6, 1.877744765e-6, 8.517995405e-7, 3.886999686e-7 };
		hiprec_real exact_pi = hiprec_real(4) * atan(hiprec_real(1));
		float pi1 = exact_pi;
		float pi2 = exact_pi - hiprec_real(pi1);
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 acos(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 y, s, z;\n");
		fprintf(fp, "\ts = copysign(simd_f32(1), x);\n");
		fprintf(fp, "\tx = simd_f32(1) - abs(x);\n");
		fprintf(fp, "\ty = simd_f32(%.9e);\n", coeffs[N - 1]);
		for (int n = N - 2; n >= 1; n--) {
			fprintf(fp, "\ty = fma(x, y, simd_f32(%.9e));\n", coeffs[n]);
		}
		fprintf(fp, "\ty *= x;\n");
		fprintf(fp, "\ty = sqrt(y);\n");
		fprintf(fp, "\tz = (simd_f32(%.9e) - y) + simd_f32(%.9e);\n", pi1, pi2);
		fprintf(fp, "\ty = blend(y, z, s < simd_f32(0));\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	/* log2 */
	{
		constexpr int N = 4;
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 log2(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 y, y2, z, x0;\n");
		fprintf(fp, "\tsimd_i32 i, j, k;\n");
		fprintf(fp, "\tx0 = x * simd_f32(M_SQRT2);\n");
		fprintf(fp, "\tj = ((simd_i32&) x0 & simd_i32(0x7F800000));\n");
		fprintf(fp, "\tk = ((simd_i32&) x & simd_i32(0x7F800000));\n");
		fprintf(fp, "\tj >>= simd_i32(23);\n");
		fprintf(fp, "\tk >>= simd_i32(23);\n");
		fprintf(fp, "\tj -= simd_i32(127);\n");
		fprintf(fp, "\tk -= j;\n");
		fprintf(fp, "\tk <<= simd_i32(23);\n");
		fprintf(fp, "\ti = (simd_i32&) x;\n");
		fprintf(fp, "\ti = (i & simd_i32(0x007FFFFF)) | k;\n");
		fprintf(fp, "\tx = (simd_f32&) i;\n");
		fprintf(fp, "\ty = (x - simd_f32(1)) / (x + simd_f32(1));\n");
		fprintf(fp, "\ty2 = y * y;\n");
		fprintf(fp, "\tz = simd_f32(%.9e);\n", 2.0 / (2 * (N - 1) + 1) / log(2));
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\tz = fma(z, y2, simd_f32(%.9e));\n", 2.0 / (2 * n + 1) / log(2));
		}
		fprintf(fp, "\tz *= y;\n");
		fprintf(fp, "\tz += simd_f32(j);\n");
		fprintf(fp, "\treturn z;\n");
		fprintf(fp, "}\n");
	}
	/* log */
	{
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 log(simd_f32 x) {\n");
		fprintf(fp, "\treturn log2(x) * simd_f32(%.9e);\n", 1.0 / log2(exp(1)));
		fprintf(fp, "}\n");
	}
	/* log10 */
	{
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 log10(simd_f32 x) {\n");
		fprintf(fp, "\treturn log2(x) * simd_f32(%.9e);\n", 1.0 / log2(10.0));
		fprintf(fp, "}\n");
	}

	/* expm1 */
	{
		constexpr int N = 8;
		int factorial[N];
		factorial[0] = 1;
		factorial[1] = 1;
		for (int n = 2; n < N; n++) {
			factorial[n] = factorial[n - 1] * n;
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 expm1(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 y;\n");
		fprintf(fp, "\tsimd_i32 l;\n");
		fprintf(fp, "\tl = abs(x) < simd_f32(1.0/3.0);\n");
		fprintf(fp, "\ty = simd_f32(%.9e);\n", 1.0 / factorial[N - 1]);
		for (int n = N - 2; n >= 1; n--) {
			fprintf(fp, "\ty = fma(y, x, simd_f32(%.9e));\n", 1.0 / factorial[n]);
		}
		fprintf(fp, "\ty *= x;\n");
		fprintf(fp, "\ty = blend(exp(x) - simd_f32(1), y, l);");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}
	/* log1p */
	{
		constexpr int N = 4;
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 log1p(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 y, z, z2;\n");
		fprintf(fp, "\tsimd_i32 l;\n");
		fprintf(fp, "\tl = abs(x) < simd_f32(1.0/3.0);\n");
		fprintf(fp, "\tz = x / (x + simd_f32(2));\n");
		fprintf(fp, "\tz2 = z * z;\n");
		fprintf(fp, "\ty = simd_f32(%.9e);\n", 2.0 / (2 * (N - 1) + 1));
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, z2, simd_f32(%.9e));\n", 2.0 / (2 * n + 1));
		}
		fprintf(fp, "\ty *= z;\n");
		fprintf(fp, "\ty = blend(log(x + simd_f32(1)), y, l);");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}

	/* exp */
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
		fprintf(fp, "\tx0 = round(x * simd_f32(%.9e));\n", 1.0 / M_LN2);
		fprintf(fp, "\tx -= x0 * simd_f32(%.9e);\n", M_LN2);
		fprintf(fp, "\ty = simd_f32(%.9e);\n", c0[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x, y, simd_f32(%.9e));\n", c0[n]);
		}
		fprintf(fp, "\ti = (simd_i32(x0) + simd_i32(127)) << int(23);\n");
		fprintf(fp, "\ty *= (simd_f32&) i;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}

	{/* exp2 */
		constexpr int N = 9;
		double c0[N];
		int nf = 1;
		for (int n = 0; n < N; n++) {
			c0[n] = 1.0 / nf;
			nf *= (n + 1);
		}
		double log2 = log(2);
		double factor = 1.0;
		for (int n = 0; n < N; n++) {
			c0[n] *= factor;
			factor *= log2;
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 exp2(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 x0, y, zero;\n");
		fprintf(fp, "\tsimd_i32 i;\n");
		fprintf(fp, "\tx = max(simd_f32(-127), min(simd_f32(127), x));\n");
		fprintf(fp, "\tx0 = round(x);\n");
		fprintf(fp, "\tx -= x0;\n");
		fprintf(fp, "\ty = simd_f32(%.9e);\n", c0[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x, y, simd_f32(%.9e));\n", c0[n]);
		}
		fprintf(fp, "\ti = (simd_i32(x0) + simd_i32(127)) << int(23);\n");
		fprintf(fp, "\ty *= (simd_f32&) i;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}

	/* erfc */
	{

		constexpr int N = 10;
		constexpr int M = 7;
		const hiprec_real xmax = 9.1;
		static hiprec_real c0[N][M];
		hiprec_real a[M];
		for (int n = 0; n < M; n++) {
			hiprec_real xmin = pow(hiprec_real(2), hiprec_real(n) / hiprec_real(2)) - hiprec_real(1);
			hiprec_real xmax1 = pow(hiprec_real(2), hiprec_real(n + 1) / hiprec_real(2)) - hiprec_real(1);
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
		fprintf(fp, "simd_f32 erfc(simd_f32 x) {\n");
		fprintf(fp, "\tconstexpr static float x0[] = {");
		for (int m = 0; m < M; m++) {
			fprintf(fp, "%24.17e", double(a[m]));
			if (m != M - 1) {
				fprintf(fp, ", ");
			}
		}
		fprintf(fp, "};\n");
		fprintf(fp, "\tconstexpr static float coeff[%i][%i] = {\n", N, M);
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
		fprintf(fp, "\tsimd_f32 r, z, y, e, a, q, c, neg, rng;\n");
		fprintf(fp, "\tsimd_i32 i;\n");
		fprintf(fp, "\tneg = simd_f32(x < simd_f32(0));\n");
		fprintf(fp, "\trng = simd_f32(x < simd_f32(%.9e));\n", double(xmax));
		fprintf(fp, "\tx = min(fabs(x), simd_f32(%.9e));\n", double(xmax));
		fprintf(fp, "\tq = x + simd_f32(1);\n");
		fprintf(fp, "\tx = min(x, simd_f32(%.9e));\n", double(xmax));
		fprintf(fp, "\tq *= q;\n");
		fprintf(fp, "\ti = ((((simd_i32&) q) & simd_i32(0x7F800000)) >> int(23)) - simd_i32(127);\n");
		fprintf(fp, "\ty = x * x ;\n");
		fprintf(fp, "\tz = fma(x, x, -y);\n");
		fprintf(fp, "\te = exp(-y) * (simd_f32(1) - z);\n");
		fprintf(fp, "\ta.gather(x0, i);\n");
		fprintf(fp, "\tx -= a;\n");
		fprintf(fp, "\ty = c.gather(coeff[%i], i);\n", N - 1);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, c.gather(coeff[%i], i));\n", n);
		}
		fprintf(fp, "\ty *= e * rng;\n");
		fprintf(fp, "\treturn fma(simd_f32(1) - neg, y, neg * (simd_f32(2) - y));\n");
		fprintf(fp, "}\n");

	}

	/* pow */
	{
		fprintf(fp, "\nsimd_f32 pow(simd_f32 x, simd_f32 y) {\n");
		constexpr int M = 5;
		constexpr int N = 1;
		float coeff[M];
		float coeffx[N];
		float coeffy[N];
		for (int m = N; m < M; m++) {
			coeff[m] = (hiprec_real(2) / (hiprec_real(2 * m + 1))) / log(hiprec_real(2));
		}
		for (int m = 0; m < N; m++) {
			auto exact = (hiprec_real(2) / (hiprec_real(2 * m + 1))) / log(hiprec_real(2));
			coeffx[m] = exact;
			coeffy[m] = exact - hiprec_real(coeffx[m]);
		}
		fprintf(fp, "\tsimd_f32 z, x2, x0;\n");
		fprintf(fp, "\tsimd_i32 i, j, k;\n");
		fprintf(fp, "\tsimd_f32_2 X2, X, Z;\n");
		fprintf(fp, "\tx0 = x * simd_f32(M_SQRT2);\n");
		fprintf(fp, "\tj = ((simd_i32&) x0 & simd_i32(0x7F800000));\n");
		fprintf(fp, "\tk = ((simd_i32&) x & simd_i32(0x7F800000));\n");
		fprintf(fp, "\tj >>= simd_i32(23);\n");
		fprintf(fp, "\tk >>= simd_i32(23);\n");
		fprintf(fp, "\tj -= simd_i32(127);\n");
		fprintf(fp, "\tk -= j;\n");
		fprintf(fp, "\tk <<= simd_i32(23);\n");
		fprintf(fp, "\ti = (simd_i32&) x;\n");
		fprintf(fp, "\ti = (i & simd_i32(0x7FFFFFULL)) | k;\n");
		fprintf(fp, "\tX = (simd_f32&) i;\n");
		fprintf(fp, "\tX = (X - simd_f32(1)) / (X + simd_f32(1));\n");
		fprintf(fp, "\tX2 = X * X;\n");
		fprintf(fp, "\tx2 = X2.x;\n");
		fprintf(fp, "\tz = simd_f32(%.9e);\n", coeff[M - 1]);
		for (int m = M - 2; m >= N; m--) {
			fprintf(fp, "\tz = fma(z, x2, simd_f32(%.9e));\n", coeff[m]);
		}
		fprintf(fp, "\tZ = simd_f32_2::two_product(z, x2) + simd_f32_2(%.9e, %.9e);\n", coeffx[N - 1], coeffy[N - 1]);
		for (int m = N - 2; m >= 0; m--) {
			fprintf(fp, "\tZ = Z * X2 + simd_f32_2(%.9e, %.9e);\n", coeffx[m], coeffy[m]);
		}
		fprintf(fp, "\tZ = y * (X * Z + simd_f32(j));\n");
		fprintf(fp, "\tz = exp2(Z.x) * (simd_f32(1) + simd_f32(M_LN2) * Z.y);\n");
		fprintf(fp, "\treturn z;\n");
		fprintf(fp, "}\n\n");
	}

}
void double_funcs(FILE* fp) {
	include(fp, "../include/code64.hpp");

	/* erf */
	{
		constexpr int M = 6;
		static std::vector<hiprec_real> co[M];
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
		for (int n = 0; n < N; n++) {
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
		fprintf(fp, "\tx = fabs(x);\n");
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
			fprintf(fp, "\ty = fma(y, x, c.gather(co[%i], i0));\n", n);
		}
		fprintf(fp, "\treturn s * y;\n");
		fprintf(fp, "}\n");
	}

	/* tgamma */
	{
		static bool init = false;
		static constexpr int M = 24;
		static constexpr int M0 = 20;
		static constexpr int N = 2 * M;
		static constexpr int Nsin = 11;
		static double coeffs[Nsin];
		static constexpr int Nsets = 11;
		static hiprec_real coeff[M][Nsets];
		static double coeff0[M][Nsets];
		static double ehi, elo;
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
				coeff[n][10] = (A[2 * n] * sqrt(hiprec_real(2)) * gamma(hiprec_real(n) + hiprec_real(0.5)) / gamma(hiprec_real(0.5)));
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
			ehi = exp(hiprec_real(-1));
			elo = exp(hiprec_real(-1)) - hiprec_real(ehi);
			coeff[0][0] = 0.0;
			coeff[0][1] = logl(tgammal(1.5L));
			coeff[0][2] = 0.0;
			coeff[0][3] = logl(tgammal(2.5L));
			coeff[1][0] = -.57721566490153286060651209008240243104215933593992L;
			coeff[1][2] = -.57721566490153286060651209008240243104215933593992L + 1.0L;
			double_2 sum = -.57721566490153286060651209008240243104215933593992L - 2.0L / 3.0L;
			coeff[1][1] = 0.0364899739785765205590237L;
			coeff[1][3] = 0.7031566406452431872256903336679110677L;
			for (int n = 2; n < M; n++) {
				long double z = zeta(n);
				hiprec_real sgn = n % 2 == 1 ? -hiprec_real(1) : hiprec_real(1);
				hiprec_real invn = hiprec_real(1) / hiprec_real(n);
				hiprec_real zon = hiprec_real(z) / hiprec_real(n);
				coeff[n][0] = sgn * zon;
				coeff[n][1] = sgn * zon * hiprec_real((1 << n) - 1);
				coeff[n][1] -= sgn * pow(hiprec_real(0.5), hiprec_real(-n)) * invn;
				coeff[n][2] = coeff[n][0] - hiprec_real(sgn / hiprec_real(n));
				coeff[n][3] = coeff[n][1] - sgn * pow(hiprec_real(3) / hiprec_real(2), hiprec_real(-n)) * invn;
			}
			for (int m = 0; m < M; m++) {
				for (int n = 0; n < Nsets; n++) {
					if( n >=4 && n != 10) {
						coeff[m][n] *= pow(hiprec_real(16)/hiprec_real(10), hiprec_real(m));
					} else if( n != 10 ) {
						coeff[m][n] /= log(hiprec_real(2));
					}
					if( n >= 4 ) {
						hiprec_real exact = sqrt(hiprec_real(8) * atan(hiprec_real(1)) * exp(hiprec_real(-1)));
						coeff[m][n] = coeff[m][n] * exact;
					}
					coeff0[m][n] = coeff[m][n];
				}
			}
		}
		fprintf(fp, "\nsimd_f64 tgamma(simd_f64 x) {\n");
		fprintf(fp, "\tstatic const simd_f64_2 E(%.17e, %.17e);\n", ehi, elo);
		fprintf(fp, "\tstatic constexpr double coeffs[][%i] = { \n", Nsets);
		for (int m = 0; m < M; m++) {
			fprintf(fp, "\t\t{ ");
			for (int n = 0; n < Nsets; n++) {
				fprintf(fp, "%25.17e%s ", coeff0[m][n], n == Nsets - 1 ? "" : ",");
			}
			fprintf(fp, " }%s\n", m == M - 1 ? "" : ",");
		}

		fprintf(fp, "\t};\n");
		fprintf(fp, "\tsimd_f64 y, z, x0, x1, nf, x2, c, xc, a, b, sgn;\n"
				"\tsimd_i64 neg, inv, ni, xoa, xob, flag;\n"
				"\tneg = x < simd_f64(0);\n"
				"\tx0 = x + simd_f64(1);\n"
				"\tx0 = blend(abs(x), x0, x >= simd_f64(-1));\n"
				"\tx0 = blend(x, x0, neg);\n"
				"\tinv = x0 < simd_f64(1);\n"
				"\tx0 = blend(x0, x0 + simd_f64(1), inv);\n"
				"\tx1 = round(simd_f64(2) * x0) * simd_f64(0.5);\n"
				"\txoa = x0 > simd_f64(10.0);\n"
				"\txob = x0 > simd_f64(2.5);\n"
				"\tnf = round(simd_f64(2) * x1 - simd_f64(2));\n"
				"\tnf = blend(nf, floor(x0 * simd_f64(0.8)) + simd_i64(2), xob);\n"
				"\tnf = blend(nf, simd_f64(10), xoa);\n"
				"\txc = simd_f64(1.25) * (simd_f64(nf) - simd_f64(1.5));\n"
				"\tz = x0 - x1;\n"
				"\tz = blend(z, (x0 - xc), xob);\n"
				"\tz = blend(z, simd_f64(1) / x0, xoa);\n"
				"\tni = simd_i64(round(nf));\n"
				"");
		fprintf(fp, "\ty = c.gather(coeffs[%i], ni);\n", M - 1);
		for (int k = M - 1; k >= 0; k--) {
			fprintf(fp, "\ty = fma(y, z, c.gather(coeffs[%i], ni));\n", k);
		}
		fprintf(fp, "\tflag = ni >= simd_i64(4);\n"
				"\ta = exp2(y);\n"
				"\tsimd_f64_2 A = simd_f64_2(x0) * E;\n"
				"\tb = pow(A.x, x0 - simd_f64(0.5)) * (simd_f64(1) + (x0 - simd_f64(0.5)) * A.y / A.x);\n"
				"\tb = y * b;\n");
		fprintf(fp, "\ty = blend(a, b, flag);\n"
				"\ty = blend(y, y / abs(x), inv);\n"
				"\tx2 = x - floor(x);\n"
				"\tx0 = blend(x2, simd_f64(1) - (x - floor(x)), x2 > simd_f64(0.5));\n"
				"\tsgn = simd_f64(1);\n"
				"\tnf = abs(floor(abs(x) * simd_f64(2)) * simd_f64(0.5));\n"
				"\tni = simd_i64(nf);\n"
				"\tsgn = blend(sgn, -sgn, (ni & simd_i64(1)) != simd_i64(0));\n"
				"\tx2 = x0 * x0;\n"
				"");
		fprintf(fp, "\tz = simd_f64(%.17e);\n", coeffs[Nsin - 1]);
		for (int k = Nsin - 1; k >= 0; k--) {
			fprintf(fp, "\tz = fma(z, x2, simd_f64(%.17e));\n", coeffs[k]);
		}
		fprintf(fp, "\tz *= x0 * sgn;\n");
		fprintf(fp, "\tz = simd_f64(1) / (x * y * z);\n");
		fprintf(fp, "\tz = blend(z, y / -(simd_f64(x) + simd_f64(1)), inv);\n");
		fprintf(fp, "\ty = blend(y, z, neg);\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");

	}
	/* asin */
	{
		static std::vector<hiprec_real> co1;
		static std::vector<hiprec_real> co2;
		static const hiprec_real z0(0.5);
		static constexpr double toler = std::numeric_limits<double>::epsilon() * 0.5;
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
		co2 = ChebyCoeffs(func2, toler, 0);
		do {
			int n = co1.size();
			co1.push_back(
					factorial(hiprec_real(2 * n))
							/ (pow(hiprec_real(2), hiprec_real(2 * n)) * factorial(hiprec_real(n)) * factorial(hiprec_real(n)) * (hiprec_real(2 * n + 1))));

		} while (std::abs((double) co1.back() * pow(double(z0), 2 * co1.size() - 1)) > toler);
		auto tmp = co1;
		co1.resize(0);
		for (int i = 0; i < tmp.size(); i++) {
			co1.push_back(0.0);
			co1.push_back(tmp[i]);
		}
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
		fprintf(fp, "\tx = fabs(x);\n");
		fprintf(fp, "\ti = x > simd_f64(%.17e);\n", (double) z0);
		fprintf(fp, "\tz = sqrt(simd_f64(1) - x);\n");
		fprintf(fp, "\tx0 = x;\n");
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
	/* acos */
	{

		constexpr int N = 47;
		constexpr double coeffs[] = { 0, 2., 0.33333333333333333333, 0.088888888888888888889, 0.028571428571428571429, 0.01015873015873015873,
				0.0038480038480038480038, 0.0015222872365729508587, 0.00062160062160062160062, 0.00026001594629045609438, 0.00011084890341856286129,
				0.000047986538276434139085, 0.000021037576563219314599, 9.321264692626404007e-6, 4.167443738237730892e-6, 1.8777447648151615054e-6,
				8.5179954049074866675e-7, 3.8869996856618833991e-7, 1.7830839827877528608e-7, 8.2179119548112649632e-8, 3.8034182252395726304e-8,
				1.7669771081252369944e-8, 8.2371765822751534304e-9, 3.8519743631122456622e-9, 1.8064667004311861306e-9, 8.4940801587621486629e-10,
				4.0036199843335919414e-10, 1.8912977703770147816e-10, 8.9529615234080764659e-11, 4.2462927007573696003e-11, 2.0175887917157897366e-11,
				9.6024849949455883817e-12, 4.5773750397533285887e-12, 2.1851897625675563985e-12, 1.0446319804372558902e-12, 5.0003915916582517976e-13,
				2.3965100546875424304e-13, 1.1498989377545557163e-13, 5.523549634336795704e-14, 2.6560125447826616705e-14, 1.2784161647514013927e-14,
				6.159186581157007613e-15, 2.9700495246485742391e-15, 1.4334247227031696191e-15, 6.9237259986367832437e-16, 3.3468997586419007141e-16,
				1.6190807480291086828e-16 };
		hiprec_real exact_pi = hiprec_real(4) * atan(hiprec_real(1));
		double pi1 = exact_pi;
		double pi2 = exact_pi - hiprec_real(pi1);
		fprintf(fp, "\n");

		fprintf(fp, "simd_f64 acos(simd_f64 x) {\n");

		fprintf(fp, "\tsimd_f64 y1, y2, y, s, z, x2;\n");
		fprintf(fp, "\ts = copysign(simd_f64(1), x);\n");
		fprintf(fp, "\tx = simd_f64(1) - abs(x);\n");
		fprintf(fp, "\ty = simd_f64(%.17e);\n", coeffs[N - 1]);
		for (int n = N - 2; n >= 1; n -= 1) {
			fprintf(fp, "\ty = fma(x, y, simd_f64(%.17e));\n", coeffs[n]);
		}
		fprintf(fp, "\ty = x * y;\n");
		fprintf(fp, "\ty = sqrt(y);\n");
		fprintf(fp, "\tz = (simd_f64(%.17e) - y) + simd_f64(%.17e);\n", pi1, pi2);
		fprintf(fp, "\ty = blend(y, z, s < simd_f64(0));\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	/*log*/
	{
		constexpr int N = 10;
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 log2(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 y, y2, z, x0;\n");
		fprintf(fp, "\tsimd_i64 i, j, k;\n");
		fprintf(fp, "\tx0 = x * simd_f64(M_SQRT2);\n");
		fprintf(fp, "\tj = ((simd_i64&) x0 & simd_i64(0x7FF0000000000000ULL));\n");
		fprintf(fp, "\tk = ((simd_i64&) x & simd_i64(0x7FF0000000000000ULL));\n");
		fprintf(fp, "\tj >>= simd_i64(52);\n");
		fprintf(fp, "\tk >>= simd_i64(52);\n");
		fprintf(fp, "\tj -= simd_i64(1023);\n");
		fprintf(fp, "\tk -= j;\n");
		fprintf(fp, "\tk <<= simd_i64(52);\n");
		fprintf(fp, "\ti = (simd_i64&) x;\n");
		fprintf(fp, "\ti = (i & simd_i64(0xFFFFFFFFFFFFFULL)) | k;\n");
		fprintf(fp, "\tx = (simd_f64&) i;\n");
		fprintf(fp, "\ty = (x - simd_f64(1)) / (x + simd_f64(1));\n");
		fprintf(fp, "\ty2 = y * y;\n");
		fprintf(fp, "\tz = simd_f64(%.17e);\n", (double) (2.0L / (long double) (2 * (N - 1) + 1) / logl(2)));
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\tz = fma(z, y2, simd_f64(%.17e));\n", (double) (2.0L / (long double) (2 * n + 1) / log(2)));
		}
		fprintf(fp, "\tz *= y;\n");
		fprintf(fp, "\tz += simd_f64(j);\n");
		fprintf(fp, "\treturn z;\n");
		fprintf(fp, "}\n");
	}

	/* log */
	{
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 log(simd_f64 x) {\n");
		fprintf(fp, "\treturn log2(x) * simd_f64(%.17e);\n", double(1.0L / log2l(expl(1.0L))));
		fprintf(fp, "}\n");
	}
	/* log10 */
	{
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 log10(simd_f64 x) {\n");
		fprintf(fp, "\treturn log2(x) * simd_f64(%.17e);\n", double(1.0L / log2l(10.0L)));
		fprintf(fp, "}\n");
	}
	/* cos */
	{
		constexpr int N = 11;
		double coeff[N];
		hiprec_real pi_exact = hiprec_real(4) * atan(hiprec_real(1));
		double pi1 = pi_exact;
		double pi2 = pi_exact - hiprec_real(pi1);
		hiprec_real fac(1);
		for (int n = 0; n < N; n++) {
			coeff[n] = hiprec_real(1) / fac;
			fac *= hiprec_real(2 * n + 2);
			fac *= -hiprec_real(2 * n + 3);
		}
		fprintf(fp, "\nsimd_f64 cos(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 x0, s, x2, y;\n");
		fprintf(fp, "\tx = abs(x);\n");
		fprintf(fp, "\tx0 = floor(x * simd_f64(%.17e));\n", 1.0 / M_PI);
		fprintf(fp, "\ts = simd_f64(2) * (x0 - simd_f64(2) * floor(x0 * simd_f64(0.5))) - simd_f64(1);\n");
		fprintf(fp, "\tx0 += simd_f64(0.5);\n");
		fprintf(fp, "\tx = s * fma(-x0, simd_f64(%.17e), fma(-x0, simd_f64(%.17e), x));\n", pi2, pi1);
		fprintf(fp, "\tx2 = x * x;\n");
		fprintf(fp, "\ty = simd_f64(%.17e);\n", coeff[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x2, y, simd_f64(%.17e));\n", coeff[n]);
		}
		fprintf(fp, "\ty *= x;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	/* sin */
	{
		constexpr int N = 11;
		double coeff[N];
		hiprec_real pi_exact = hiprec_real(4) * atan(hiprec_real(1));
		double pi1 = pi_exact;
		double pi2 = pi_exact - hiprec_real(pi1);
		hiprec_real fac(1);
		for (int n = 0; n < N; n++) {
			coeff[n] = hiprec_real(1) / fac;
			fac *= hiprec_real(2 * n + 2);
			fac *= -hiprec_real(2 * n + 3);
		}
		fprintf(fp, "\nsimd_f64 sin(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 x0, s, x2, y;\n");
		fprintf(fp, "\tx0 = round(x * simd_f64(%.17e));\n", 1.0 / M_PI);
		fprintf(fp, "\ts = -simd_f64(2) * (x0 - simd_f64(2) * floor(x0 * simd_f64(0.5))) + simd_f64(1);\n");
		fprintf(fp, "\tx = s * fma(-x0, simd_f64(%.17e), fma(-x0, simd_f64(%.17e), x));\n", pi2, pi1);
		fprintf(fp, "\tx2 = x * x;\n");
		fprintf(fp, "\ty = simd_f64(%.17e);\n", coeff[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x2, y, simd_f64(%.17e));\n", coeff[n]);
		}
		fprintf(fp, "\ty *= x;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	/* log1p */
	{
		constexpr int N = 9;
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 log1p(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 y, z, z2;\n");
		fprintf(fp, "\tsimd_i64 l;\n");
		fprintf(fp, "\tl = abs(x) < simd_f64(1.0/3.0);\n");
		fprintf(fp, "\tz = x / (x + simd_f64(2));\n");
		fprintf(fp, "\tz2 = z * z;\n");
		fprintf(fp, "\ty = simd_f64(%.17e);\n", 2.0 / (2 * (N - 1) + 1));
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, z2, simd_f64(%.17e));\n", 2.0 / (2 * n + 1));
		}
		fprintf(fp, "\ty *= z;\n");
		fprintf(fp, "\ty = blend(log(x + simd_f64(1)), y, l);");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}

	{
		/* expm1 */

		constexpr int N = 14;
		int factorial[N];
		factorial[0] = 1;
		factorial[1] = 1;
		for (int n = 2; n < N; n++) {
			factorial[n] = factorial[n - 1] * n;
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 expm1(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 y;\n");
		fprintf(fp, "\tsimd_i64 l;\n");
		fprintf(fp, "\tl = abs(x) < simd_f64(1.0/3.0);\n");
		fprintf(fp, "\ty = simd_f64(%.17e);\n", 1.0 / factorial[N - 1]);
		for (int n = N - 2; n >= 1; n--) {
			fprintf(fp, "\ty = fma(y, x, simd_f64(%.17e));\n", 1.0 / factorial[n]);
		}
		fprintf(fp, "\ty *= x;\n");
		fprintf(fp, "\ty = blend(exp(x) - simd_f64(1), y, l);");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}

	{
		constexpr int N = 13;
		hiprec_real c0[N];
		int nf = 1;
		for (int n = 0; n < N; n++) {
			c0[n] = hiprec_real(1) / hiprec_real(nf);
			c0[n] *= pow(log(hiprec_real(2)), hiprec_real(n));
			nf *= (n + 1);
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 exp2(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 x0, y;\n");
		fprintf(fp, "\tsimd_i64 i;\n");
		fprintf(fp, "\tx = max(simd_f64(-1000), min(simd_f64(1000), x));\n");
		fprintf(fp, "\tx0 = round(x);\n");
		fprintf(fp, "\tx -= x0;\n");
		fprintf(fp, "\ty = simd_f64(%.20e);\n", (double) c0[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x, y, simd_f64(%.17e));\n", (double) c0[n]);
		}
		fprintf(fp, "\ti = (simd_i64(x0) + simd_i64(1023)) << (long long)(52);\n");
		fprintf(fp, "\ty *= (simd_f64&) i;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
		fprintf(fp, "\n");
	}
	/*	{

	 constexpr int N = 15;
	 constexpr int M = 10;
	 static double coeffs[N];
	 static double epwr[M];
	 static std::once_flag once;
	 std::call_once(once, []() {
	 hiprec_real fac(1);
	 for( int n = 0; n < N; n++) {
	 coeffs[n] = (double)(hiprec_real(1) / fac);
	 fac *= hiprec_real(n+1);
	 }
	 hiprec_real ep = exp(hiprec_real(1));
	 for( int m = 0; m < M; m++) {
	 epwr[m] = ep;
	 ep *= ep;
	 }
	 });
	 fprintf(fp, "\nsimd_f64 exp(simd_f64 x) {\n");
	 fprintf(fp, "\tsimd_f64 x0, y;\n");
	 fprintf(fp, "\tsimd_i64 i, j;\n");
	 fprintf(fp, "\ti = x < simd_f64(0);\n");
	 fprintf(fp, "\tx = abs(x);\n");
	 fprintf(fp, "\tx0 = round(x);\n");
	 fprintf(fp, "\tx -= x0;\n");
	 fprintf(fp, "\ty = simd_f64(%.16e);\n", (double) coeffs[N - 1]);
	 for (int n = N - 2; n >= 0; n--) {
	 fprintf(fp, "\ty = fma(x, y, simd_f64(%.17e));\n", (double) coeffs[n]);
	 }
	 fprintf(fp, "\tj = simd_i64(x0);\n");
	 for (int m = 0; m < M; m++) {
	 fprintf(fp, "\ty = blend(y, y * simd_f64(%.17e), j & simd_i64(0x%x));\n", (double) epwr[m], 1 << m);
	 }
	 fprintf(fp, "\ty = blend(y, simd_f64(1) / y, i);\n");
	 fprintf(fp, "\treturn y;\n");
	 fprintf(fp, "}\n");
	 }*/
	{
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 exp(simd_f64 x) {\n");
		hiprec_real exact = log2(exp(hiprec_real(1)));
		double_2 log2e = double_2::quick_two_sum(double(exact), exact - hiprec_real(double(exact)));
		fprintf(fp, "\tstatic const simd_f64_2 LOG2E(%.17e, %.17e);\n", log2e.x, log2e.y);
		fprintf(fp, "\tsimd_f64 y;\n");
		fprintf(fp, "\tsimd_f64_2 XLOG2E = x * LOG2E;\n");
		fprintf(fp, "\ty = exp2(XLOG2E.x) * (simd_f64(1) + simd_f64(%.17e) * XLOG2E.y);\n", log(2));
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}
	/* erfc */
	{

		constexpr int N = 15;
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
		fprintf(fp, "\tsimd_f64_2 Z;\n");
		fprintf(fp, "\tsimd_i64 i;\n");
		fprintf(fp, "\tneg = simd_f64(x < simd_f64(0));\n");
		fprintf(fp, "\trng = simd_f64(x < simd_f64(%.17e));\n", double(xmax));
		fprintf(fp, "\tx = min(fabs(x), simd_f64(%.17e));\n", double(xmax));
		fprintf(fp, "\tq = x + simd_f64(1);\n");
		fprintf(fp, "\tx = min(x, simd_f64(%.17e));\n", double(xmax));
		fprintf(fp, "\tq *= q;\n");
		fprintf(fp, "\tq *= q;\n");
		fprintf(fp, "\ti = ((((simd_i64&) q) & simd_i64(0x7FF0000000000000)) >> (long long)(52)) - simd_i64(1023);\n");
		fprintf(fp, "\tZ = simd_f64_2::two_product(x, x) ;\n");
		fprintf(fp, "\te = exp(-Z.x) * (simd_f64(1) - Z.y);\n");
		fprintf(fp, "\ta.gather(x0, i);\n");
		fprintf(fp, "\tx -= a;\n");
		fprintf(fp, "\ty = c.gather(coeff[%i], i);\n", N - 1);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, c.gather(coeff[%i], i));\n", n);
		}
		fprintf(fp, "\ty *= e * rng;\n");
		fprintf(fp, "\treturn fma(simd_f64(1) - neg, y, neg * (simd_f64(2) - y));\n");
		fprintf(fp, "}\n");

	}

	/* pow */
	{
		fprintf(fp, "\nsimd_f64 pow(simd_f64 x, simd_f64 y) {\n");
		constexpr int M = 11;
		constexpr int N = 2;
		double coeff[M];
		double coeffx[N];
		double coeffy[N];
		double c0x;
		double c0y;
		for (int m = N; m < M; m++) {
			coeff[m] = (hiprec_real(2) / (hiprec_real(2 * m + 1))) / log(hiprec_real(2));
		}
		for (int m = 0; m < N; m++) {
			auto exact = (hiprec_real(2) / (hiprec_real(2 * m + 1))) / log(hiprec_real(2));
			coeffx[m] = exact;
			coeffy[m] = exact - hiprec_real(coeffx[m]);
		}
		fprintf(fp, "\tsimd_f64 z, x2, x0;\n");
		fprintf(fp, "\tsimd_i64 i, j, k;\n");
		fprintf(fp, "\tsimd_f64_2 X2, X, Z;\n");
		fprintf(fp, "\tx0 = x * simd_f64(M_SQRT2);\n");
		fprintf(fp, "\tj = ((simd_i64&) x0 & simd_i64(0x7FF0000000000000ULL));\n");
		fprintf(fp, "\tk = ((simd_i64&) x & simd_i64(0x7FF0000000000000ULL));\n");
		fprintf(fp, "\tj >>= simd_i64(52);\n");
		fprintf(fp, "\tk >>= simd_i64(52);\n");
		fprintf(fp, "\tj -= simd_i64(1023);\n");
		fprintf(fp, "\tk -= j;\n");
		fprintf(fp, "\tk <<= simd_i64(52);\n");
		fprintf(fp, "\ti = (simd_i64&) x;\n");
		fprintf(fp, "\ti = (i & simd_i64(0xFFFFFFFFFFFFFULL)) | k;\n");
		fprintf(fp, "\tX = (simd_f64&) i;\n");
		fprintf(fp, "\tX = (X - simd_f64(1)) / (X + simd_f64(1));\n");
		fprintf(fp, "\tX2 = X * X;\n");
		fprintf(fp, "\tx2 = X2.x;\n");
		fprintf(fp, "\tz = simd_f64(%.17e);\n", coeff[M - 1]);
		for (int m = M - 2; m >= N; m--) {
			fprintf(fp, "\tz = fma(z, x2, simd_f64(%.17e));\n", coeff[m]);
		}
		fprintf(fp, "\tZ = simd_f64_2::two_product(z, x2) + simd_f64_2(%.17e, %.17e);\n", coeffx[N - 1], coeffy[N - 1]);
		for (int m = N - 2; m >= 0; m--) {
			fprintf(fp, "\tZ = Z * X2 + simd_f64_2(%.17e, %.17e);\n", coeffx[m], coeffy[m]);
		}
		fprintf(fp, "\tZ = y * (X * Z + simd_f64(j));\n");
		fprintf(fp, "\tz = exp2(Z.x) * (simd_f64(1) + simd_f64(M_LN2) * Z.y);\n");
		fprintf(fp, "\treturn z;\n");
		fprintf(fp, "}\n");
	}

}

/*void two_sum(simd_f64* xptr, simd_f64* yptr, simd_f64 a, simd_f64 b) {
 auto& x = *xptr;
 auto& y = *yptr;
 x = a + b;
 const simd_f64 bvirtual = x - a;
 const simd_f64 avirtual = x - bvirtual;
 const simd_f64 broundoff = b - bvirtual;
 const simd_f64 aroundoff = a - avirtual;
 y = aroundoff + broundoff;
 }

 void two_product(simd_f64* xptr, simd_f64* yptr, simd_f64 a, simd_f64 b) {
 auto& x = *xptr;
 auto& y = *yptr;
 x = a * b;
 y = fma(a, b, -x);
 }*/

int main() {
	system("mkdir -p ./generated_code\n");
	system("mkdir -p ./generated_code/src/\n");
	FILE* fp = fopen("./generated_code/src/math.cpp", "wt");
	fprintf(fp, "#include \"simd.hpp\"\n");
	fprintf(fp, "#include <utility>\n");
	fprintf(fp, "\nnamespace simd {\n\n");
	float_funcs(fp);
	double_funcs(fp);
	fprintf(fp, "\n}\n");
	fclose(fp);
	return 0;
}

