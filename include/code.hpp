
simd_f32 cbrt(simd_f32 x) {
	simd_f32 y, s;
	simd_i32 e;
	s = copysign(simd_f32(1), x);
	x = fabs(x);
	frexp(x, &e);
	e = simd_i32(simd_f32(e) / simd_f32(3));
	e += e >= 0;
	y = ldexp(simd_f32(0.57), e);
	y += simd_f32(1.0 / 3.0) * (x / (y * y) - y);
	y += simd_f32(1.0 / 3.0) * (x / (y * y) - y);
	y += simd_f32(1.0 / 3.0) * (x / (y * y) - y);
	y += simd_f32(1.0 / 3.0) * (x / (y * y) - y);
	y += simd_f32(1.0 / 3.0) * (x / (y * y) - y);
	y += simd_f32(1.0 / 3.0) * (x / (y * y) - y);
	y *= s;
	return y;
}

simd_f32 atan(simd_f32 x) {
	simd_f32 inner, inv, s, x0, x1, x2, y, y1, y0, left;
	s = copysign(simd_f32(1), x);
	x = abs(x);
	left = -simd_f32(2) * s + simd_f32(1);
	inner = copysign(simd_f32(0.5), simd_f32(0.1) - x) + simd_f32(0.5);
	inv = copysign(simd_f32(0.5), x - simd_f32(1)) + simd_f32(0.5);
	x0 = x;
	x1 = simd_f32(1) / (x + simd_f32(1e-37));
	x = blend(x0, x1, simd_i32(inv));
	x0 = x / sqrt(simd_f32(1) + x * x);
	x2 = x * x;
	x1 = simd_f32(3.0 / 8.0);
	x1 = fma(x1, x2, simd_f32(-0.5));
	x1 = fma(x1, x2, simd_f32(1.0));
	x1 *= x;
	x = blend(x0, x1, simd_i32(inner));
	y = s * asin(x);
	y0 = -simd_f32(M_PI_2) - y;
	y0 = blend(y, y0, inv);
	y1 = simd_f32(M_PI_2) - y;
	y1 = blend(y, y1, inv);
	y = blend(y1, y0, left);
	return y;
}

