/*
 * hiprec.hpp
 *
 *  Created on: Nov 22, 2022
 *      Author: dmarce1
 */

#ifndef HIPREC_HPP_
#define HIPREC_HPP_

#include <gmp.h>
#include <mpfr.h>

class hiprec_real {
	mpfr_t q;
public:
	hiprec_real() {
		mpfr_init2(q, 1024);
	}
	~hiprec_real() {
		mpfr_clear(q);
	}
	hiprec_real(const hiprec_real& other) {
		mpfr_init2(q, 1024);
		*this = other;
	}
	hiprec_real(hiprec_real&& other) {
		mpfr_init2(q, 1024);
		*this = std::move(other);
	}
	hiprec_real& operator=(const hiprec_real& other) {
		mpfr_set(q, other.q, MPFR_RNDN);
		return *this;
	}
	hiprec_real& operator=(hiprec_real&& other) {
		mpfr_set(q, other.q, MPFR_RNDN);
		return *this;
	}
	hiprec_real(long double a) {
		mpfr_init2(q, 1024);
		mpfr_set_ld(q, a, MPFR_RNDN);
	}
	hiprec_real(double a) {
		mpfr_init2(q, 1024);
		mpfr_set_d(q, a, MPFR_RNDN);
	}
	hiprec_real(int a) {
		mpfr_init2(q, 1024);
		mpfr_set_si(q, a, MPFR_RNDN);
	}
	operator double() const {
		return mpfr_get_d(q, MPFR_RNDN);
	}
	hiprec_real operator+(const hiprec_real& other) const {
		hiprec_real a;
		mpfr_add(a.q, q, other.q, MPFR_RNDN);
		return a;
	}
	hiprec_real operator-(const hiprec_real& other) const {
		hiprec_real a;
		mpfr_sub(a.q, q, other.q, MPFR_RNDN);
		return a;
	}
	hiprec_real operator*(const hiprec_real& other) const {
		hiprec_real a;
		mpfr_mul(a.q, q, other.q, MPFR_RNDN);
		return a;
	}
	hiprec_real operator/(const hiprec_real& other) const {
		hiprec_real a;
		mpfr_div(a.q, q, other.q, MPFR_RNDN);
		return a;
	}
	hiprec_real& operator+=(const hiprec_real& other) {
		*this = *this + other;
		return *this;
	}
	hiprec_real& operator-=(const hiprec_real& other) {
		*this = *this - other;
		return *this;
	}
	hiprec_real& operator*=(const hiprec_real& other) {
		*this = *this * other;
		return *this;
	}
	hiprec_real& operator/=(const hiprec_real& other) {
		*this = *this / other;
		return *this;
	}
	hiprec_real operator-() const {
		hiprec_real a(*this);
		mpfr_neg(a.q, q, MPFR_RNDN);
		return a;
	}
	bool operator==(const hiprec_real& a) const {
		return mpfr_cmp(q, a.q) == 0;
	}
	bool operator!=(const hiprec_real& a) const {
		return mpfr_cmp(q, a.q) != 0;
	}
	bool operator<(const hiprec_real& a) const {
		return mpfr_cmp(q, a.q) < 0;
	}
	bool operator>(const hiprec_real& a) const {
		return mpfr_cmp(q, a.q) > 0;
	}
	bool operator>=(const hiprec_real& a) const {
		return mpfr_cmp(q, a.q) >= 0;
	}
	bool operator<=(const hiprec_real& a) const {
		return mpfr_cmp(q, a.q) <= 0;
	}
	friend hiprec_real atan(hiprec_real);
	friend hiprec_real erfc(hiprec_real);
	friend hiprec_real sqrt(hiprec_real);
	friend hiprec_real exp(hiprec_real);
	friend hiprec_real asin(hiprec_real);
	friend hiprec_real gamma(hiprec_real);
	friend hiprec_real log(hiprec_real);
	friend hiprec_real log2(hiprec_real);
	friend hiprec_real abs(hiprec_real);
	friend hiprec_real cos(hiprec_real);
	friend hiprec_real sin(hiprec_real);
	friend hiprec_real pow(hiprec_real, hiprec_real);
};

hiprec_real pow(hiprec_real a, hiprec_real b) {
	hiprec_real c;
	mpfr_pow(c.q, a.q, b.q, MPFR_RNDN);
	return c;
}

hiprec_real log(hiprec_real a) {
	hiprec_real c;
	mpfr_log(c.q, a.q, MPFR_RNDN);
	return c;
}

hiprec_real log2(hiprec_real a) {
	hiprec_real c;
	mpfr_log2(c.q, a.q, MPFR_RNDN);
	return c;
}

hiprec_real abs(hiprec_real a) {
	hiprec_real c;
	mpfr_abs(c.q, a.q, MPFR_RNDN);
	return c;
}

hiprec_real cos(hiprec_real a) {
	hiprec_real c;
	mpfr_cos(c.q, a.q, MPFR_RNDN);
	return c;
}

hiprec_real sin(hiprec_real a) {
	hiprec_real c;
	mpfr_sin(c.q, a.q, MPFR_RNDN);
	return c;
}

hiprec_real atan(hiprec_real a) {
	hiprec_real b;
	mpfr_atan(b.q, a.q, MPFR_RNDN);
	return b;
}

hiprec_real gamma(hiprec_real a) {
	hiprec_real b;
	mpfr_gamma(b.q, a.q, MPFR_RNDN);
	return b;
}

hiprec_real asin(hiprec_real a) {
	hiprec_real b;
	mpfr_asin(b.q, a.q, MPFR_RNDN);
	return b;
}

hiprec_real erfc(hiprec_real a) {
	hiprec_real b;
	mpfr_erfc(b.q, a.q, MPFR_RNDN);
	return b;
}

hiprec_real sqrt(hiprec_real a) {
	hiprec_real b;
	mpfr_sqrt(b.q, a.q, MPFR_RNDN);
	return b;
}

hiprec_real exp(hiprec_real a) {
	hiprec_real b;
	mpfr_exp(b.q, a.q, MPFR_RNDN);
	return b;
}

#endif /* HIPREC_HPP_ */
