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

