/*
 * polynomial.hpp
 *
 *  Created on: Nov 22, 2022
 *      Author: dmarce1
 */

#ifndef POLYNOMIAL_HPP_
#define POLYNOMIAL_HPP_


using polynomial = std::vector<hiprec_real>;

inline polynomial poly_add(const polynomial& A, const polynomial& B) {
	polynomial C;
	if (A.size() > B.size()) {
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
	if (A.size() > B.size()) {
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
	Tn[0] = poly_term(0, hiprec_real(1));
	Tn[1] = poly_term(1, hiprec_real(1));
	for (int n = 2; n < NMAX; n++) {
		Tn[n] = poly_sub(poly_mul(poly_mul(poly_term(0, hiprec_real(2)), poly_term(1, hiprec_real(1))), Tn[n - 1]), Tn[n - 2]);
	}
	return Tn;
}

#endif /* POLYNOMIAL_HPP_ */
