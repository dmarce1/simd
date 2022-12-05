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
		mpfr_init2(q, 512);
	}
	~hiprec_real() {
		mpfr_clear(q);
	}
	hiprec_real(const hiprec_real& other) {
		mpfr_init2(q, 512);
		*this = other;
	}
	hiprec_real(hiprec_real&& other) {
		mpfr_init2(q, 512);
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
		mpfr_init2(q, 512);
		mpfr_set_ld(q, a, MPFR_RNDN);
	}
	hiprec_real(double a) {
		mpfr_init2(q, 512);
		mpfr_set_d(q, a, MPFR_RNDN);
	}
	hiprec_real(int a) {
		mpfr_init2(q, 512);
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
	friend hiprec_real tan(hiprec_real);
	friend hiprec_real atan(hiprec_real);
	friend hiprec_real erfc(hiprec_real);
	friend hiprec_real erf(hiprec_real);
	friend hiprec_real sqrt(hiprec_real);
	friend hiprec_real exp(hiprec_real);
	friend hiprec_real exp2(hiprec_real);
	friend hiprec_real acos(hiprec_real);
	friend hiprec_real asin(hiprec_real);
	friend hiprec_real gamma(hiprec_real);
	friend hiprec_real log1p(hiprec_real);
	friend hiprec_real log(hiprec_real);
	friend hiprec_real log2(hiprec_real);
	friend hiprec_real abs(hiprec_real);
	friend hiprec_real cos(hiprec_real);
	friend hiprec_real sin(hiprec_real);
	friend hiprec_real pow(hiprec_real, hiprec_real);
	friend hiprec_real copysign(hiprec_real, hiprec_real);
};

hiprec_real pow(hiprec_real a, hiprec_real b) {
	hiprec_real c;
	mpfr_pow(c.q, a.q, b.q, MPFR_RNDN);
	return c;
}

hiprec_real copysign(hiprec_real a, hiprec_real b) {
	hiprec_real c;
	mpfr_copysign(c.q, a.q, b.q, MPFR_RNDN);
	return c;
}

hiprec_real log(hiprec_real a) {
	hiprec_real c;
	mpfr_log(c.q, a.q, MPFR_RNDN);
	return c;
}

hiprec_real log1p(hiprec_real a) {
	hiprec_real c;
	mpfr_log1p(c.q, a.q, MPFR_RNDN);
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

hiprec_real tan(hiprec_real a) {
	hiprec_real b;
	mpfr_tan(b.q, a.q, MPFR_RNDN);
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

hiprec_real acos(hiprec_real a) {
	hiprec_real b;
	mpfr_acos(b.q, a.q, MPFR_RNDN);
	return b;
}

hiprec_real erf(hiprec_real a) {
	hiprec_real b;
	mpfr_erf(b.q, a.q, MPFR_RNDN);
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

hiprec_real exp2(hiprec_real a) {
	hiprec_real b;
	mpfr_exp2(b.q, a.q, MPFR_RNDN);
	return b;
}


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


#endif /* HIPREC_HPP_ */
