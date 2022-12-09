/*
 * polynomial.hpp
 *
 *  Created on: Nov 22, 2022
 *      Author: dmarce1
 */

#ifndef POLYNOMIAL_HPP_
#define POLYNOMIAL_HPP_

#include <complex>
#include <functional>
#include <valarray>

using polynomial = std::vector<hiprec_real>;

inline polynomial poly_add(const polynomial& A, const polynomial& B) {
	polynomial C;
	if (A.size() >= B.size()) {
		C = A;
		for (int i = 0; i < B.size(); i++) {
			C[i] += B[i];
		}
		return C;
	} else {
		return poly_add(B, A);
	}

}

inline polynomial poly_sub(const polynomial& A, const polynomial& B) {
	polynomial C;
	if (A.size() >= B.size()) {
		C = A;
		for (int i = 0; i < B.size(); i++) {
			C[i] -= B[i];
		}
		return C;
	} else {
		for (int i = 0; i < B.size(); i++) {
			C[i] = -B[i];
		}
		for (int i = 0; i < A.size(); i++) {
			C[i] += A[i];
		}
		return C;
	}

}

inline polynomial poly_mul(const polynomial& A, const polynomial& B) {
	polynomial C;
	int sz = A.size() + B.size() - 1;
	C.resize(sz, 0.0);
	for (int i = 0; i < A.size(); i++) {
		for (int j = 0; j < B.size(); j++) {
			C[i + j] += A[i] * B[j];
		}
	}
	return C;
}

inline polynomial poly_term(int n, hiprec_real a) {
	polynomial A(n + 1, hiprec_real(0));
	A[n] = a;
	return A;
}

std::vector<polynomial> Tns(int NMAX) {
	std::vector<polynomial> Tn(NMAX);
	if (NMAX) {
		Tn[0] = poly_term(0, hiprec_real(1));
		if (NMAX > 1) {
			Tn[1] = poly_term(1, hiprec_real(1));
			if (NMAX > 2) {
				for (int n = 2; n < NMAX; n++) {
					Tn[n] = poly_sub(poly_mul(poly_mul(poly_term(0, hiprec_real(2)), poly_term(1, hiprec_real(1))), Tn[n - 1]), Tn[n - 2]);
				}
			}
		}
	}
	return Tn;
}

static void fft(std::valarray<std::complex<hiprec_real>>& x) {
	const size_t N = x.size();
	if (N <= 1)
		return;
	std::valarray < std::complex < hiprec_real >> even = x[std::slice(0, N / 2, 2)];
	std::valarray < std::complex < hiprec_real >> odd = x[std::slice(1, N / 2, 2)];
	fft (even);
	fft (odd);
	for (size_t k = 0; k < N / 2; ++k) {
		static const auto one = hiprec_real(1);
		static const auto twopi = hiprec_real(8) * atan(hiprec_real(1));
		std::complex<hiprec_real> t = std::polar(one, -twopi * hiprec_real((int) k) / hiprec_real((int) N)) * odd[k];
		x[k] = even[k] + t;
		x[k + N / 2] = even[k] - t;
	}
}

static void dct(std::valarray<hiprec_real>& x) {
	const auto zero = hiprec_real(0);
	std::valarray < std::complex < hiprec_real >> y;
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
	fft (y);
	for (int n = 0; n < x.size(); n++) {
		x[n] = y[n].real() * (hiprec_real(2) - hiprec_real((int) (n == 0))) / hiprec_real(2 * (int) x.size());
	}
}

static std::vector<hiprec_real> ChebyCoeffs(std::function<hiprec_real(hiprec_real)> func, hiprec_real (toler), int evenodd = 0) {
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
		n = evenodd == -1 ? 1 : 0;
		while (abs(X[n]) > eps && n < M - 1 - (evenodd == 1)) {
			//	printf( "%i %e\n", n, (double)X[n]);
			n += 1 + (evenodd != 0);
			if (n > 0) {
				eps /= hiprec_real(2);
			}
			if (evenodd != 0) {
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
	return c0;
}

static std::vector<hiprec_real> ChebyCoeffs2(std::function<hiprec_real(hiprec_real)> func, int num, int evenodd = 0) {
	static const hiprec_real one(1);
	static const hiprec_real pi(hiprec_real(4) * atan(hiprec_real(1)));
	hiprec_real last, next;
	next = hiprec_real(99);
	int M = 1 << (int(ceil(log2(num))) + 2);
	int N;
	std::valarray<hiprec_real> X;
	X.resize(M);
	last = next;
	for (int m = 0; m < M; m++) {
		X[m] = func(cos(pi * (hiprec_real(m) + hiprec_real(0.5)) / hiprec_real(M)));
	}
	dct(X);
	N = num;
	auto tn = Tns(N + 1);
	polynomial c0(N + 1, hiprec_real(0));
	for (int n = 0; n < N; n++) {
		c0 = poly_add(c0, poly_mul(poly_term(0, X[n]), tn[n]));
	}
	return c0;
}
#endif /* POLYNOMIAL_HPP_ */
