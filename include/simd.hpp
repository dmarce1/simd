#pragma once

#include <immintrin.h>
#include <limits>
#include <mutex>
#include <cmath>

#pragma once

#ifdef NDEBUG
#define CHECK_ALIGNMENT(ptr, sz)
#else
#define CHECK_ALIGNMENT(ptr, sz) \
	if( uintptr_t(ptr) % uintptr_t(sz) != 0 ) { \
		printf( "Alignment error in %s on line %i!\n", __FILE__, __LINE__ ); \
		abort(); \
	}
#endif

namespace simd {

class simd_i32;
class simd_f32;

class alignas(sizeof(__m256i)) simd_i32 {
	union {
		__m256i v;
		int w[8];
	};
public:
	simd_i32() = default;
	simd_i32(const simd_i32&) = default;
	simd_i32(simd_i32&&) = default;
	simd_i32& operator=(const simd_i32&) = default;
	simd_i32& operator=(simd_i32&&) = default;
	simd_i32(const simd_f32& other);
	int operator[](int i) const {
		CHECK_ALIGNMENT(this, 32);
		return w[i];
	}
	int& operator[](int i) {
		CHECK_ALIGNMENT(this, 32);
		return w[i];
	}
	inline simd_i32(int a) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_set1_epi32(a);
	}
	inline simd_i32(const std::initializer_list<int>& list) {
		CHECK_ALIGNMENT(this, 32);
		int i = 0;
		for (auto j = list.begin(); j != list.end(); j++) {
			w[i++] = *j;
		}
	}
	inline simd_i32& gather(const int* ptr, simd_i32 indices) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_i32gather_epi32(ptr, indices.v, sizeof(int));
		return *this;
	}
	inline simd_i32 permute(const simd_i32& indices) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_permutevar8x32_epi32(v, indices.v);
		return result;
	}
	inline simd_i32& operator+=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_add_epi32(v, other.v);
		return *this;
	}
	inline simd_i32& operator-=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_sub_epi32(v, other.v);
		return *this;
	}
	inline simd_i32& operator*=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_mul_epi32(v, other.v);
		return *this;
	}
	inline simd_i32& operator&=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_and_si256(v, other.v);
		return *this;
	}
	inline simd_i32& operator^=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_xor_si256(v, other.v);
		return *this;
	}
	inline simd_i32& operator|=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_or_si256(v, other.v);
		return *this;
	}
	inline simd_i32 operator&&(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result = (*this) & other;
		return result;
	}
	inline simd_i32 operator||(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result = *this | other;
		return result;
	}
	inline simd_i32 operator!() const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result = (*this) == simd_i32(0);
		return result;
	}
	inline simd_i32 operator+(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_add_epi32(v, other.v);
		return result;
	}
	inline simd_i32 operator-(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_sub_epi32(v, other.v);
		return result;
	}
	inline simd_i32 operator~() const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_andnot_si256(v, simd_i32(0xFFFFFFFF).v);
		return result;
	}
	inline simd_i32 operator*(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_mul_epi32(v, other.v);
		return result;
	}
	inline simd_i32 operator&(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_and_si256(v, other.v);
		return result;
	}
	inline simd_i32 operator^(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_xor_si256(v, other.v);
		return result;
	}
	inline simd_i32 operator|(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_or_si256(v, other.v);
		return result;
	}
	inline simd_i32 operator>>(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_srlv_epi32(v, other.v);
		return result;
	}
	inline simd_i32 operator<<(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_sllv_epi32(v, other.v);
		return result;
	}
	inline simd_i32& operator>>=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_srlv_epi32(v, other.v);
		return *this;
	}
	inline simd_i32& operator<<=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_sllv_epi32(v, other.v);
		return *this;
	}
	inline simd_i32 operator>>(int i) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_srli_epi32(v, i);
		return result;
	}
	inline simd_i32 operator<<(int i) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_slli_epi32(v, i);
		return result;
	}
	inline simd_i32& operator>>=(int i) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_srli_epi32(v, i);
		return *this;
	}
	inline simd_i32& operator<<=(int i) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_slli_epi32(v, i);
		return *this;
	}
	inline simd_i32 operator-() const {
		CHECK_ALIGNMENT(this, 32);
		return simd_i32(0) - *this;
	}
	inline simd_i32 operator+() const {
		CHECK_ALIGNMENT(this, 32);
		return *this;
	}
	inline simd_i32 operator==(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_cmpeq_epi32(v, other.v);
		return -result;
	}
	inline simd_i32 operator!=(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		return simd_i32(1) - (*this == other);
	}
	inline simd_i32 operator>(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_cmpgt_epi32(v, other.v);
		return -result;
	}
	inline simd_i32 operator>=(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		return ((*this == other) + (*this > other)) > simd_i32(0);
	}
	inline simd_i32 operator<(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		return simd_i32(1) - (*this >= other);
	}
	inline simd_i32 operator<=(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		return simd_i32(1) - (*this > other);
	}
	inline static constexpr size_t size() {
		return 8;
	}
	inline simd_i32& pad(int n) {
		CHECK_ALIGNMENT(this, 32);
		const int& e = size();
		for (int i = n; i < e; i++) {
			w[i] = w[0];
		}
		return *this;
	}
	static inline simd_i32 mask(int n) {
		simd_i32 mk;
		for (int i = 0; i < n; i++) {
			mk[i] = 1;
		}
		for (int i = n; i < size(); i++) {
			mk[i] = 0;
		}
		return mk;
	}
	inline void set_NaN() {
		CHECK_ALIGNMENT(this, 32);
		for (int i = 0; i < size(); i++) {
			w[i] = std::numeric_limits<int>::signaling_NaN();
		}
	}
	friend simd_i32 max(simd_i32, simd_i32);
	friend simd_i32 min(simd_i32, simd_i32);
	friend simd_f32 blend(simd_f32, simd_f32, simd_i32);
	friend simd_f32;
	};

	inline simd_i32 max(simd_i32 a, simd_i32 b) {
		a.v = _mm256_max_epi32(a.v, b.v);
		return a;
	}

	inline simd_i32 min(simd_i32 a, simd_i32 b) {
		a.v = _mm256_min_epi32(a.v, b.v);
		return a;
	}

	class alignas(sizeof(__m256)) simd_f32 {
		__m256 v;
	public:
		simd_f32() = default;
		simd_f32(const simd_f32&) = default;
		simd_f32(simd_f32&&) = default;
		simd_f32& operator=(const simd_f32&) = default;
		simd_f32& operator=(simd_f32&&) = default;
		inline float operator[](int i) const {
			CHECK_ALIGNMENT(this, 32);
			return v[i];
		}
		inline float& operator[](int i) {
			CHECK_ALIGNMENT(this, 32);
			return v[i];
		}
		inline simd_f32(float a) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_broadcast_ss(&a);
		}
		inline simd_f32(const std::initializer_list<float>& list) {
			CHECK_ALIGNMENT(this, 32);
			int i = 0;
			for (auto j = list.begin(); j != list.end(); j++) {
				v[i++] = *j;
			}
		}
		inline simd_f32(const simd_i32& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_cvtepi32_ps(other.v);
		}
		inline simd_f32& gather(const float* ptr, simd_i32 indices) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_i32gather_ps(ptr, indices.v, sizeof(float));
			return *this;
		}
		inline simd_f32 permute(const simd_i32& indices) const {
			CHECK_ALIGNMENT(this, 32);
			simd_f32 result;
			result.v = _mm256_permutevar8x32_ps(v, indices.v);
			return result;
		}
		inline simd_f32& operator+=(const simd_f32& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_add_ps(v, other.v);
			return *this;
		}
		inline simd_f32& operator-=(const simd_f32& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_sub_ps(v, other.v);
			return *this;
		}
		inline simd_f32& operator*=(const simd_f32& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_mul_ps(v, other.v);
			return *this;
		}
		inline simd_f32& operator/=(const simd_f32& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_div_ps(v, other.v);
			return *this;
		}
		inline simd_f32 operator+(const simd_f32& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_f32 result;
			result.v = _mm256_add_ps(v, other.v);
			return result;
		}
		inline simd_f32 operator-(const simd_f32& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_f32 result;
			result.v = _mm256_sub_ps(v, other.v);
			return result;
		}
		inline simd_f32 operator*(const simd_f32& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_f32 result;
			result.v = _mm256_mul_ps(v, other.v);
			return result;
		}
		inline simd_f32 operator/(const simd_f32& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_f32 result;
			result.v = _mm256_div_ps(v, other.v);
			return result;
		}
		inline simd_f32 operator-() const {
			CHECK_ALIGNMENT(this, 32);
			return simd_f32(0) - *this;
		}
		inline simd_f32 operator+() const {
			CHECK_ALIGNMENT(this, 32);
			return *this;
		}
		inline simd_i32 operator==(const simd_f32& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i32 result;
			result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_EQ_OS));
			return -result;
		}
		inline simd_i32 operator!=(const simd_f32& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i32 result;
			result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_NEQ_OS));
			return -result;
		}
		inline simd_i32 operator>(const simd_f32& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i32 result;
			result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_GT_OS));
			return -result;
		}
		inline simd_i32 operator>=(const simd_f32& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i32 result;
			result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_GE_OS));
			return -result;
		}
		inline simd_i32 operator<(const simd_f32& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i32 result;
			result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_LT_OS));
			return -result;
		}
		inline simd_i32 operator<=(const simd_f32& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i32 result;
			result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_LE_OS));
			return -result;
		}
		inline static constexpr size_t size() {
			return 8;
		}
		inline simd_f32& pad(int n) {
			CHECK_ALIGNMENT(this, 32);
			const int& e = size();
			for (int i = n; i < e; i++) {
				v[i] = v[0];
			}
			return *this;
		}
		static inline simd_f32 mask(int n) {
			simd_f32 mk;
			for (int i = 0; i < n; i++) {
				mk[i] = 1.f;
			}
			for (int i = n; i < size(); i++) {
				mk[i] = 0.f;
			}
			return mk;
		}
		inline void set_NaN() {
			CHECK_ALIGNMENT(this, 32);
			for (int i = 0; i < size(); i++) {
				v[i] = std::numeric_limits<float>::signaling_NaN();
			}
		}
		friend simd_f32 sqrt(simd_f32);
		friend simd_f32 rsqrt(simd_f32);
		friend simd_f32 fma(simd_f32, simd_f32, simd_f32);
		friend float reduce_sum(simd_f32);
		friend simd_f32 round(simd_f32);
		friend simd_f32 floor(simd_f32);
		friend simd_f32 ceil(simd_f32);
		friend simd_f32 max(simd_f32, simd_f32);
		friend simd_f32 min(simd_f32, simd_f32);
		friend simd_f32 blend(simd_f32, simd_f32, simd_i32);
		friend class simd_i32;
	};

	simd_f32 log10(simd_f32 x);
	simd_f32 log2(simd_f32 x);
	simd_f32 log(simd_f32 x);
	simd_f32 cos(simd_f32 x);
	simd_f32 exp(simd_f32 x);
	simd_f32 erfc(simd_f32 x);
	simd_f32 pow(simd_f32 y, simd_f32 x);
	simd_f32 asin(simd_f32 x);
	void erfcexp(simd_f32, simd_f32*, simd_f32*);

	inline simd_f32 acos(simd_f32 x) {
		return simd_f32(M_PI_2) - asin(x);
	}

	inline simd_f32 atan(simd_f32 x) {
		simd_f32 z;
		z = sqrt(simd_f32(1.f) + x * x);
		return asin(x / z);
	}

	inline simd_f32 blend(simd_f32 a, simd_f32 b, simd_i32 mask) {
		mask = -mask;
		a.v = _mm256_blendv_ps(a.v, b.v, ((simd_f32&) mask).v);
		return a;
	}

	inline simd_f32 atanh(simd_f32 x) {
		return simd_f32(0.5) * log((simd_f32(1) + x) / (simd_f32(1) - x));
	}


	inline simd_f32 pow(simd_f32 y, simd_f32 x) {
		return exp(x * log(y));
	}

	inline simd_f32 max(simd_f32 a, simd_f32 b) {
		a.v = _mm256_max_ps(a.v, b.v);
		return a;
	}

	inline simd_f32 min(simd_f32 a, simd_f32 b) {
		a.v = _mm256_min_ps(a.v, b.v);
		return a;
	}

	inline simd_f32 round(simd_f32 x) {
		simd_f32 result;
		result.v = _mm256_round_ps(x.v, _MM_FROUND_TO_NEAREST_INT);
		return result;
	}

	inline simd_f32 floor(simd_f32 x) {
		simd_f32 result;
		result.v = _mm256_floor_ps(x.v);
		return result;
	}

	inline simd_f32 ceil(simd_f32 x) {
		simd_f32 result;
		result.v = _mm256_ceil_ps(x.v);
		return result;
	}

	inline float reduce_sum(simd_f32 x) {
		__m128 a = *((__m128 *) &x.v);
		__m128 b = *(((__m128 *) &x.v) + 1);
		a = _mm_add_ps(a, b);
		return a[0] + a[1] + a[2] + a[3];
	}

	inline simd_f32 fma(simd_f32 a, simd_f32 b, simd_f32 c) {
		simd_f32 result;
		result.v = _mm256_fmadd_ps(a.v, b.v, c.v);
		return result;
	}

	inline simd_i32::simd_i32(const simd_f32& other) {
		v = _mm256_cvtps_epi32(_mm256_round_ps(other.v, _MM_FROUND_TO_ZERO));
	}

	inline simd_f32 sqrt(simd_f32 x) {
		x.v = _mm256_sqrt_ps(x.v);
		return x;
	}

	inline simd_f32 rsqrt(simd_f32 x) {
		x.v = _mm256_rsqrt_ps(x.v);
		return x;
	}

	inline simd_f32 abs(simd_f32 x) {
		simd_i32 i = (((simd_i32&) x) & simd_i32(0x7FFFFFFF));
		return (simd_f32&) i;
	}

	inline simd_f32 copysign(simd_f32 x, simd_f32 y) {
		simd_f32 result = abs(x);
		simd_i32 i = ((simd_i32&) result) | (simd_i32(0x80000000) & (simd_i32&) (y));
		result = (simd_f32&) i;
		return result;
	}

	inline simd_f32 sin(simd_f32 x) {
		return cos(x - simd_f32(M_PI / 2.0));
	}

	inline simd_f32 tan(simd_f32 x) {
		return sin(x) / cos(x);
	}

	inline void sincos(simd_f32 x, simd_f32* s, simd_f32* c) {
		*s = sin(x);
		*c = cos(x);
	}

	class simd_i64;
	class simd_f64;

	class alignas(sizeof(__m256i)) simd_i64 {
		union {
			__m256i v;
			int64_t w[4];
		};
	public:
		simd_i64() = default;
		simd_i64(const simd_i64&) = default;
		simd_i64(simd_i64&&) = default;
		simd_i64& operator=(const simd_i64&) = default;
		simd_i64& operator=(simd_i64&&) = default;
		simd_i64(const simd_f64& other);
		long long operator[](int i) const {
			CHECK_ALIGNMENT(this, 32);
			return w[i];
		}
		int64_t& operator[](int i) {
			CHECK_ALIGNMENT(this, 32);
			return w[i];
		}
		inline simd_i64(long long a) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_set_epi64x(a, a, a, a);
		}
		inline simd_i64(const std::initializer_list<long long>& list) {
			CHECK_ALIGNMENT(this, 32);
			int i = 0;
			for (auto j = list.begin(); j != list.end(); j++) {
				(*this)[i++] = *j;
			}
		}
		inline simd_i64& gather(const long long* ptr, simd_i64 indices) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_i64gather_epi64(ptr, indices.v, sizeof(long long));
			return *this;
		}
		inline simd_i64& operator+=(const simd_i64& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_add_epi64(v, other.v);
			return *this;
		}
		inline simd_i64& operator-=(const simd_i64& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_sub_epi64(v, other.v);
			return *this;
		}
		inline simd_i64& operator&=(const simd_i64& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_and_si256(v, other.v);
			return *this;
		}
		inline simd_i64& operator^=(const simd_i64& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_xor_si256(v, other.v);
			return *this;
		}
		inline simd_i64& operator|=(const simd_i64& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_or_si256(v, other.v);
			return *this;
		}
		inline simd_i64 operator&&(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result = (*this) & other;
			return result;
		}
		inline simd_i64 operator||(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result = (*this) | other;
			return result;
		}
		inline simd_i64 operator!() const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result = (*this) == simd_i64(0);
			return result;
		}
		inline simd_i64 operator+(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_add_epi64(v, other.v);
			return result;
		}
		inline simd_i64 operator-(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_sub_epi64(v, other.v);
			return result;
		}
		inline simd_i64 operator~() const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_andnot_si256(v, simd_i64(0xFFFFFFFFFFFFFFFFLL).v);
			return result;
		}
		inline simd_i64 operator&(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_and_si256(v, other.v);
			return result;
		}
		inline simd_i64 operator^(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_xor_si256(v, other.v);
			return result;
		}
		inline simd_i64 operator|(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_or_si256(v, other.v);
			return result;
		}
		inline simd_i64 operator>>(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_srlv_epi64(v, other.v);
			return result;
		}
		inline simd_i64 operator<<(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_sllv_epi64(v, other.v);
			return result;
		}
		inline simd_i64 operator>>(long long i) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_srli_epi64(v, i);
			return result;
		}
		inline simd_i64 operator<<(long long i) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_slli_epi64(v, i);
			return result;
		}
		inline simd_i64& operator>>=(const simd_i64& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_srlv_epi64(v, other.v);
			return *this;
		}
		inline simd_i64& operator<<=(const simd_i64& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_sllv_epi64(v, other.v);
			return *this;
		}
		inline simd_i64& operator>>=(long long i) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_srli_epi64(v, i);
			return *this;
		}
		inline simd_i64& operator<<=(long long i) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_slli_epi64(v, i);
			return *this;
		}
		inline simd_i64 operator-() const {
			CHECK_ALIGNMENT(this, 32);
			return simd_i64(0) - *this;
		}
		inline simd_i64 operator+() const {
			CHECK_ALIGNMENT(this, 32);
			return *this;
		}
		inline simd_i64 operator==(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_cmpeq_epi64(v, other.v);
			return -result;
		}
		inline simd_i64 operator!=(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			return simd_i64(1) - (*this == other);
		}
		inline simd_i64 operator>(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_cmpgt_epi64(v, other.v);
			return -result;
		}
		inline simd_i64 operator>=(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			return ((*this == other) + (*this > other)) > simd_i64(0);
		}
		inline simd_i64 operator<(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			return simd_i64(1) - (*this >= other);
		}
		inline simd_i64 operator<=(const simd_i64& other) const {
			CHECK_ALIGNMENT(this, 32);
			return simd_i64(1) - (*this > other);
		}
		inline static constexpr size_t size() {
			return 4;
		}
		inline simd_i64& pad(int n) {
			const int& e = size();
			for (int i = n; i < e; i++) {
				v[i] = v[0];
			}
			return *this;
		}
		static inline simd_i64 mask(int n) {
			simd_i64 mk;
			for (int i = 0; i < n; i++) {
				mk[i] = 1;
			}
			for (int i = n; i < size(); i++) {
				mk[i] = 0;
			}
			return mk;
		}
		inline void set_NaN() {
			CHECK_ALIGNMENT(this, 32);
			for (int i = 0; i < size(); i++) {
				v[i] = std::numeric_limits<int>::signaling_NaN();
			}
		}
		friend simd_f64 blend(simd_f64, simd_f64, simd_i64);
		friend simd_i64 max(simd_i64, simd_i64);
		friend simd_i64 min(simd_i64, simd_i64);
		friend simd_f64;
	};

	inline simd_i64 max(simd_i64 a, simd_i64 b) {
		a.v = _mm256_max_epi64(a.v, b.v);
		return a;
	}

	inline simd_i64 min(simd_i64 a, simd_i64 b) {
		a.v = _mm256_min_epi64(a.v, b.v);
		return a;
	}

	class alignas(sizeof(__m256d)) simd_f64 {
		__m256d v;
	public:
		simd_f64() = default;
		simd_f64(const simd_f64&) = default;
		simd_f64(simd_f64&&) = default;
		simd_f64& operator=(const simd_f64&) = default;
		simd_f64& operator=(simd_f64&&) = default;
		inline double operator[](int i) const {
			CHECK_ALIGNMENT(this, 32);
			return v[i];
		}
		inline double& operator[](int i) {
			CHECK_ALIGNMENT(this, 32);
			return v[i];
		}
		inline simd_f64(double a) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_broadcast_sd(&a);
		}
		inline simd_f64(const std::initializer_list<double>& list) {
			CHECK_ALIGNMENT(this, 32);
			int i = 0;
			for (auto j = list.begin(); j != list.end(); j++) {
				v[i++] = *j;
			}
		}
		inline simd_f64(const simd_i64& other) {
			CHECK_ALIGNMENT(this, 32);
			v[0] = (double) other[0];
			v[1] = (double) other[1];
			v[2] = (double) other[2];
			v[3] = (double) other[3];
		}
		inline simd_f64& gather(const double* ptr, simd_i64 indices) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_i64gather_pd(ptr, indices.v, sizeof(double));
			return *this;
		}
		inline simd_f64& operator+=(const simd_f64& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_add_pd(v, other.v);
			return *this;
		}
		inline simd_f64& operator-=(const simd_f64& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_sub_pd(v, other.v);
			return *this;
		}
		inline simd_f64& operator*=(const simd_f64& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_mul_pd(v, other.v);
			return *this;
		}
		inline simd_f64& operator/=(const simd_f64& other) {
			CHECK_ALIGNMENT(this, 32);
			v = _mm256_div_pd(v, other.v);
			return *this;
		}
		inline simd_f64 operator+(const simd_f64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_f64 result;
			result.v = _mm256_add_pd(v, other.v);
			return result;
		}
		inline simd_f64 operator-(const simd_f64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_f64 result;
			result.v = _mm256_sub_pd(v, other.v);
			return result;
		}
		inline simd_f64 operator*(const simd_f64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_f64 result;
			result.v = _mm256_mul_pd(v, other.v);
			return result;
		}
		inline simd_f64 operator/(const simd_f64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_f64 result;
			result.v = _mm256_div_pd(v, other.v);
			return result;
		}
		inline simd_f64 operator-() const {
			CHECK_ALIGNMENT(this, 32);
			return simd_f64(0) - *this;
		}
		inline simd_f64 operator+() const {
			CHECK_ALIGNMENT(this, 32);
			return *this;
		}
		inline simd_i64 operator==(const simd_f64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_EQ_OS));
			return -result;
		}
		inline simd_i64 operator!=(const simd_f64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_NEQ_OS));
			return -result;
		}
		inline simd_i64 operator>(const simd_f64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_GT_OS));
			return -result;
		}
		inline simd_i64 operator>=(const simd_f64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_GE_OS));
			return -result;
		}
		inline simd_i64 operator<(const simd_f64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_LT_OS));
			return -result;
		}
		inline simd_i64 operator<=(const simd_f64& other) const {
			CHECK_ALIGNMENT(this, 32);
			simd_i64 result;
			result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_LE_OS));
			return -result;
		}
		inline static constexpr size_t size() {
			return 4;
		}
		inline simd_f64& pad(int n) {
			CHECK_ALIGNMENT(this, 32);
			const int& e = size();
			for (int i = n; i < e; i++) {
				v[i] = v[0];
			}
			return *this;
		}
		static inline simd_f64 mask(int n) {
			simd_f64 mk;
			for (int i = 0; i < n; i++) {
				mk[i] = 1.f;
			}
			for (int i = n; i < size(); i++) {
				mk[i] = 0.f;
			}
			return mk;
		}
		inline void set_NaN() {
			CHECK_ALIGNMENT(this, 32);
			for (int i = 0; i < size(); i++) {
				v[i] = std::numeric_limits<double>::signaling_NaN();
			}
		}
		friend simd_f64 sqrt(simd_f64);
		friend simd_f64 rsqrt(simd_f64);
		friend simd_f64 fma(simd_f64, simd_f64, simd_f64);
		friend double reduce_sum(simd_f64);
		friend simd_f64 round(simd_f64);
		friend simd_f64 floor(simd_f64);
		friend simd_f64 ceil(simd_f64);
		friend simd_f64 max(simd_f64, simd_f64);
		friend simd_f64 min(simd_f64, simd_f64);
		friend simd_f64 blend(simd_f64, simd_f64, simd_i64);
		friend simd_f64 asin(simd_f64 x);
		friend class simd_i64;
	};

	simd_f64 pow(simd_f64 y, simd_f64 x);
	simd_f64 log(simd_f64 x);
	simd_f64 log10(simd_f64 x);
	simd_f64 log2(simd_f64 x);
	simd_f64 cos(simd_f64 x);
	simd_f64 asin(simd_f64 x);
	simd_f64 exp(simd_f64 x);
	simd_f64 erfc(simd_f64 x);
	void erfcexp(simd_f64, simd_f64*, simd_f64*);

	inline simd_f64 pow(simd_f64 y, simd_f64 x) {
		return exp(x * log(y));
	}

	inline simd_f64 acos(simd_f64 x) {
		return simd_f64(M_PI_2) - asin(x);
	}

	inline simd_f64 atan(simd_f64 x) {
		simd_f64 z;
		z = sqrt(simd_f64(1.0) + x * x);
		return asin(x / z);
	}

	inline simd_f64 blend(simd_f64 a, simd_f64 b, simd_i64 mask) {
		mask = -mask;
		a.v = _mm256_blendv_pd(a.v, b.v, ((simd_f64&) mask).v);
		return a;
	}

	inline simd_f64 atanh(simd_f64 x) {
		return simd_f64(0.5) * log((simd_f64(1) + x) / (simd_f64(1) - x));
	}

	inline simd_f64 max(simd_f64 a, simd_f64 b) {
		a.v = _mm256_max_pd(a.v, b.v);
		return a;
	}

	inline simd_f64 min(simd_f64 a, simd_f64 b) {
		a.v = _mm256_min_pd(a.v, b.v);
		return a;
	}

	inline simd_f64 round(simd_f64 x) {
		simd_f64 result;
		result.v = _mm256_round_pd(x.v, _MM_FROUND_TO_NEAREST_INT);
		return result;
	}

	inline simd_f64 floor(simd_f64 x) {
		simd_f64 result;
		result.v = _mm256_floor_pd(x.v);
		return result;
	}

	inline simd_f64 ceil(simd_f64 x) {
		simd_f64 result;
		result.v = _mm256_ceil_pd(x.v);
		return result;
	}

	inline double reduce_sum(simd_f64 x) {
		__m128d a = *((__m128d *) &x.v);
		__m128d b = *(((__m128d *) &x.v) + 1);
		a = _mm_add_pd(a, b);
		return a[0] + a[1];
	}

	inline simd_f64 fma(simd_f64 a, simd_f64 b, simd_f64 c) {
		simd_f64 result;
		result.v = _mm256_fmadd_pd(a.v, b.v, c.v);
		return result;
	}

	inline simd_i64::simd_i64(const simd_f64& other) {
		v[0] = (long long) (other[0]);
		v[1] = (long long) (other[1]);
		v[2] = (long long) (other[2]);
		v[3] = (long long) (other[3]);
	}

	inline simd_f64 sqrt(simd_f64 x) {
		x.v = _mm256_sqrt_pd(x.v);
		return x;
	}

	inline simd_f64 rsqrt(simd_f64 x) {
		return simd_f64(1) / sqrt(x);
	}

	inline simd_f64 abs(simd_f64 x) {
		simd_i64 i = (((simd_i64&) x) & simd_i64(0x7FFFFFFFFFFFFFFFLL));
		return (simd_f64&) i;
	}

	inline simd_f64 copysign(simd_f64 x, simd_f64 y) {
		simd_f64 result = abs(x);
		simd_i64 i = ((simd_i64&) result) | (simd_i64(0x8000000000000000LL) & (simd_i64&) (y));
		result = (simd_f64&) i;
		return result;
	}

	inline simd_f64 sin(simd_f64 x) {
		return cos(x - simd_f64(M_PI / 2.0));
	}

	inline simd_f64 tan(simd_f64 x) {
		return sin(x) / cos(x);
	}

	inline void sincos(simd_f64 x, simd_f64* s, simd_f64* c) {
		*s = sin(x);
		*c = cos(x);
	}

}

#include <vector>
