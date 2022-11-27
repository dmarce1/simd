
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
