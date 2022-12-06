
simd_f64 cbrt(simd_f64 x) {
	simd_f64 y, s;
	simd_i64 e;
	s = copysign(simd_f64(1), x);
	x = fabs(x);
	frexp(x, &e);
	e = simd_i64(simd_f64(e) / simd_f64(3));
	e += e >= 0;
	y = ldexp(simd_f64(0.57), e);
	y += simd_f64(1.0 / 3.0) * (x / (y * y) - y);
	y += simd_f64(1.0 / 3.0) * (x / (y * y) - y);
	y += simd_f64(1.0 / 3.0) * (x / (y * y) - y);
	y += simd_f64(1.0 / 3.0) * (x / (y * y) - y);
	y += simd_f64(1.0 / 3.0) * (x / (y * y) - y);
	y += simd_f64(1.0 / 3.0) * (x / (y * y) - y);
	y += simd_f64(1.0 / 3.0) * (x / (y * y) - y);
	y += simd_f64(1.0 / 3.0) * (x / (y * y) - y);
	y *= s;
	return y;
}

simd_f64 atan(simd_f64 x) {
	simd_f64 inner, inv, s, x0, x1, x2, y, y1, y0, left;
	s = copysign(simd_f64(1), x);
	x = abs(x);
	left = -simd_f64(2) * s + simd_f64(1);
	inner = copysign(simd_f64(0.5), simd_f64(0.05) - x) + simd_f64(0.5);
	inv = copysign(simd_f64(0.5), x - simd_f64(1)) + simd_f64(0.5);
	x0 = x;
	x1 = simd_f64(1) / (x + simd_f64(1e-301));
	x = blend(x0, x1, simd_i64(inv));
	x0 = x / sqrt(simd_f64(1) + x * x);
	x2 = x * x;
	x1 = simd_f64(-63.0 / 256.0);
	x1 = fma(x1, x2, simd_f64(35.0 / 128.0));
	x1 = fma(x1, x2, simd_f64(-5.0 / 16.0));
	x1 = fma(x1, x2, simd_f64(3.0 / 8.0));
	x1 = fma(x1, x2, simd_f64(-0.5));
	x1 = fma(x1, x2, simd_f64(1.0));
	x1 *= x;
	x = blend(x0, x1, simd_i64(inner));
	y = s * asin(x);
	y0 = -simd_f64(M_PI_2) - y;
	y0 = blend(y, y0, inv);
	y1 = simd_f64(M_PI_2) - y;
	y1 = blend(y, y1, inv);
	y = blend(y1, y0, left);
	return y;
}
