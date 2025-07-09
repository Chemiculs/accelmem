#ifndef HC128_HXX
#define HC128_HXX

#include "memory.hxx"

class hc128 {

private:

	std::uint32_t w[1280];

	std::uint32_t p[512];

	std::uint32_t q[512];

	std::uint32_t step_counter = 0;

	std::uint32_t keystream_byte_index = 0;

	std::uint32_t keystream_output = NULL;

#pragma Internal Cryptographic Subroutines

	static inline const std::uint32_t FASTCALL rotr(

		const std::uint32_t x,

		const std::uint32_t n

	) noexcept {

		return (x >> n) | (x << (32ui32 - n));
	}

	static inline const std::uint32_t FASTCALL rotl(
	
		const std::uint32_t x,

		const std::uint32_t n

	) noexcept {

		return (x << n) | (x >> (32 - n));
	}

	static inline const std::uint32_t FASTCALL f1(

		const std::uint32_t x

	) noexcept {

		return rotr(x, 7ui32) ^ rotr(x, 18ui32) ^ (x >> 3ui32);
	}

	static inline const std::uint32_t FASTCALL f2(

		const std::uint32_t x

	) noexcept {

		return rotr(x, 17ui32) ^ rotr(x, 19ui32) ^ (x >> 10ui32);
	}

	static inline const std::uint32_t FASTCALL g1(
	
		const std::uint32_t x,

		const std::uint32_t y,

		const std::uint32_t z

	) noexcept {

		return (rotr(x, 10ui32) ^ rotr(z, 23ui32)) + (y >> 8ui32);
	}

	static inline const std::uint32_t g2(
	
		const std::uint32_t x,

		const std::uint32_t y,

		const std::uint32_t z

	) noexcept {

		return (rotl(x, 10ui32) ^ rotl(z, 23ui32)) + (y << 8);
	}

	inline const std::uint32_t FASTCALL h1(
		
		const std::uint32_t x
	
	) const noexcept {

		const std::uint8_t xx[2]{
			
			static_cast<std::uint8_t>(x),

			static_cast<std::uint8_t>(x >> 16ui32)
		};

		return q[xx[0]] + q[xx[1] + 256ui32];
	}

	inline const std::uint32_t FASTCALL h2(

		const std::uint32_t x

	) const noexcept {

		const std::uint8_t xx[2]{

			static_cast<std::uint8_t>(x),

			static_cast<std::uint8_t>(x >> 16ui32)
		};

		return p[xx[0]] + p[xx[1] + 256ui32];
	}

	inline void STDCALL warmup() noexcept {

		for (std::uint32_t i = 0; i < 512ui32; ++i) {

			p[i] = (
				p[i]
				+
				g1(p[(i - 3) & 0x1FFui32], p[(i - 10) & 0x1FFui32], p[(i - 511ui32) & 0x1FFui32])
				)
				^
				h1(p[(i - 12) & 0x1FF]
			);

			q[i] = (
				q[i]
				+
				g2(q[(i - 3) & 0x1FF], q[(i - 10ui32) & 0x1FF], q[(i - 511ui32) & 0x1FF])
				)
				^
				h2(q[(i - 12ui32) & 0x1FF]
			);
		}
	}

	inline const std::uint32_t STDCALL step(std::uint32_t step_count) noexcept {

		std::uint32_t i = step_count + 1;

		std::uint32_t j = i & 0x1FFui32;

		if ((i & 0x3FF) < 512ui32) {

			p[j] =
				p[j]
				+
				g1(p[(j - 3ui32) & 0x1FFui32], p[(j - 10ui32) & 0x1FFui32], p[(j - 511ui32) & 0x1FF]);

			return h1(p[(j - 12ui32) & 0x1FF]) ^ p[j];
		}

		q[j] =
			q[j]
			+
			g2(q[(j - 3ui32) & 0x1FFui32], q[(j - 10ui32) & 0x1FFui32], q[(j - 511ui32) & 0x1FF]);
	}

	inline const std::uint32_t STDCALL step() noexcept {

		keystream_output = step(step_counter);

		++step_counter;

		return keystream_output;
	}

	inline void FASTCALL expand_w(

		void* key,

		void* iv,

		const bool destroy_key	= false,

		const bool destroy_iv	= false

	) noexcept {

		step_counter = 0;

		std::uint32_t* k = reinterpret_cast<std::uint32_t*>(key);

		std::uint32_t* _iv = reinterpret_cast<std::uint32_t*>(iv);

		for (std::uint32_t i = 0; i < 4ui32; ++i) {

			w[i] = k[i];

			w[i + 4ui32] = k[i];
		}

		for (std::uint32_t i = 0; i < 4ui32; ++i) {

			w[i + 8ui32] = _iv[i];

			w[i + 12ui32] = _iv[i];
		}

		for (std::uint32_t i = 16ui32; i < 1280ui32; ++i) {

			w[i] = f2(w[i - 2ui32]) + w[i - 7ui32] + f1(w[i - 15ui32]) + w[i - 16ui32] + i;
		}

		for (std::uint32_t i = 0; i < 512ui32; ++i) {

			p[i] = w[i + 256ui32];

			q[i] = w[i + 768ui32];
		}

		SECURE_ZERO_MEMORY(

			&w[0],

			sizeof(w)
		);

		if (destroy_key)
			SECURE_ZERO_MEMORY(

				&k,

				sizeof(k)
			);

		if(destroy_iv)
			SECURE_ZERO_MEMORY(

				&iv,

				sizeof(iv)
			);

		warmup();
	}

#pragma endregion

public:

	FORCE_INLINE void FASTCALL encrypt_range(

		std::uint8_t* dst,

		const std::uint8_t* src,

		const std::size_t offset,

		const std::size_t len,

		void* key				= nullptr,

		void* iv				= nullptr,

		const bool destroy_key	= false,

		const bool destroy_iv	= false

	) noexcept {

		if (!dst || !src || !len)
			return;

		if(!accelmem::g_is_simd_check)
			accelmem::cpu_check_simd();

		if(key && iv)
			expand_w(
			
				key,
			
				iv,
			
				destroy_key
			);

		std::size_t l_offset	= offset;

		std::size_t end			= offset + len;

		// indexof(keystream_output)
		std::size_t word_index	= offset / sizeof(std::uint32_t);

		// Offset within keystream_output
		std::size_t byte_offset = offset % sizeof(std::uint32_t);

		step_counter			= static_cast<std::uint32_t>(word_index);

		// Allign to next 32-bit Word if necessary
		if (byte_offset) {

			step();

			for (std::size_t j = byte_offset; j < sizeof(std::uint32_t) && l_offset < len; ++l_offset, ++j)
				dst[l_offset - offset] = src[l_offset] ^ static_cast<std::uint8_t>(keystream_output >> (8 * j));
		}

		if (accelmem::g_is_sse2 && len > (sizeof(__m128i))) {
			
			std::size_t block_ct = (end - l_offset) / sizeof(__m128i);

			alignas(0x10) std::uint32_t keystream_buffer_128[4];

			while (l_offset + sizeof(__m128i) <= end) {

				for (std::size_t i = 0; i < sizeof(keystream_buffer_128) / sizeof(std::uint32_t); ++i)
					keystream_buffer_128[i] = step();

				const __m128i keystream_output_128 = _mm_load_si128(reinterpret_cast<const __m128i*>(keystream_buffer_128));

				__m128i plaintext_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + l_offset));

				const __m128i ciphertext_128 = _mm_xor_si128(plaintext_128, keystream_output_128);

				_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + (l_offset - offset)), ciphertext_128);

				l_offset += sizeof(__m128i);
			}
		}

		while (l_offset + sizeof(std::uint32_t) <= end) {

			*reinterpret_cast<std::uint32_t*>(dst + (l_offset - offset))
			=
			*reinterpret_cast<const std::uint32_t*>(src + l_offset) ^ step();

			l_offset += sizeof(std::uint32_t);

		}

		while (l_offset <= end) {

			for (std::size_t j = 0; l_offset < len; ++l_offset, ++j)
				dst[l_offset - offset] = src[l_offset] ^ static_cast<std::uint8_t>(step() >> (8 * j));

			++l_offset;
		}
	}

	FORCE_INLINE void FASTCALL encrypt(

		std::uint8_t* dst,

		const std::uint8_t* src,

		const std::size_t len,

		void* key				= nullptr,

		void* iv				= nullptr,

		const bool destroy_key	= false,

		const bool destroy_iv	= false

	) noexcept {

		if (!dst || !src || !len)
			return;

		encrypt_range(

			dst,

			src,

			0,

			len,

			key,

			iv,

			destroy_key,

			destroy_iv
		);
	}

	FORCE_INLINE void FASTCALL decrypt_range(

		std::uint8_t* dst,

		const std::uint8_t* src,

		const std::size_t offset,

		const std::size_t len,

		void* key				= nullptr,

		void* iv				= nullptr,

		const bool destroy_key	= false,

		const bool destroy_iv	= false

	) noexcept {

		if (!dst || !src || !len)
			return;

		encrypt_range(
			
			dst,
			
			src,

			offset,
			
			len,
			
			key,
			
			iv,

			destroy_key,

			destroy_iv
		);
	}

	FORCE_INLINE void FASTCALL decrypt(
	
		std::uint8_t* dst,
		
		const std::uint8_t* src,

		const std::size_t len,

		void* key				= nullptr,

		void* iv				= nullptr,

		const bool destroy_key	= false,

		const bool destroy_iv	= false

	) noexcept {

		if (!dst || !src || !len)
			return;

		decrypt_range(
			
			dst,
			
			src,

			0,

			len,
			
			key,
			
			iv,

			destroy_key,

			destroy_iv
		);
	}

	FORCE_INLINE STDCALL ~hc128() noexcept {

		SECURE_ZERO_MEMORY(p, sizeof(p));

		SECURE_ZERO_MEMORY(q, sizeof(q));

		step_counter = NULL;

		keystream_byte_index = NULL;

		keystream_output = NULL;
	}
};

#endif