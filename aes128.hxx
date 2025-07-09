#ifndef AES128_HXX
#define AES128_HXX

#include "crc32.hxx"

#include <wmmintrin.h>

/*
	This Class Implements the AES-128(CTR Mode) Cipher, using the AES-NI Intrinsic Instruction Set ; If the CPU doesn't have this Instruction Set, the Algorithm will Return False
*/

class aes128_ctr {

private:

#pragma region Algorithm Constants

	// ShiftRows indices for AES
	static const constexpr uint8_t shift_rows[16] {
		0, 5, 10, 15,
		4, 9, 14, 3,
		8, 13, 2, 7,
		12, 1, 6, 11
	};

	static const constexpr std::uint32_t nr = 10;

#pragma endregion

#pragma region Globals

	bool is_key_expanded = false;

	__m128i key_schedule[11];

	std::uint64_t counter = 0;

#pragma endregion

	inline void FASTCALL expand_key(

		void* key,

		const bool destroy_key = false

	) noexcept {

		if (!key)
			return;

		if (!accelmem::g_is_simd_check)
			accelmem::cpu_check_simd();

		__m128i key_gen, key_xor;
		
		key_schedule[0] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(key));

		key_gen = key_schedule[0];                     
		
		// Cheesy Macro workaround for constexpr requirement of Operand 2 on Intrinsic _mm_aeskeygenassist_si128
#define AES_EXPAND_STEP(round)										\
	key_xor = _mm_aeskeygenassist_si128(key_gen, round);			\
	key_xor = _mm_shuffle_epi32(key_xor, _MM_SHUFFLE(3,3,3,3));		\
	key_gen = _mm_xor_si128(key_gen, _mm_slli_si128(key_gen, 4));	\
	key_gen = _mm_xor_si128(key_gen, _mm_slli_si128(key_gen, 4));	\
	key_gen = _mm_xor_si128(key_gen, _mm_slli_si128(key_gen, 4));	\
	key_gen = _mm_xor_si128(key_gen, key_xor);						\
	key_schedule[round] = key_gen;

		AES_EXPAND_STEP(1);
		AES_EXPAND_STEP(2);
		AES_EXPAND_STEP(3);
		AES_EXPAND_STEP(4);
		AES_EXPAND_STEP(5);
		AES_EXPAND_STEP(6);
		AES_EXPAND_STEP(7);
		AES_EXPAND_STEP(8);
		AES_EXPAND_STEP(9);
		AES_EXPAND_STEP(nr);

do_ret:

		is_key_expanded = true;

		counter = 0;

		if (destroy_key)
			SECURE_ZERO_MEMORY(

				key,

				sizeof(__m128i)
			);
	}

public:

	FORCE_INLINE const bool FASTCALL aes128_encrypt_range(
		
		std::uint8_t*			dst,

		const std::uint8_t*		src,

		const std::size_t		offset,

		const std::size_t		len,

		std::uint64_t&			iv,

		// Allow default arg for key, incase has already been passed in and expanded
		void*					key = nullptr,

		const bool				destroy_key = false,

		const bool				destroy_iv = false,

		const bool				disable_thr = false

	) noexcept {

		if(!src || !dst || !len)
			return false;

		if (key)
			expand_key(key, destroy_key);
		else if (!is_key_expanded && !key)
			return false;

		if (!accelmem::g_is_aes_ni)
			return false;

		// Halve the MT Threshhold due to Cumbersome Workload
		const bool thr = disable_thr ? false : (len >= (OMP_MEM_THR_THRESHHOLD / 2));

		std::uint64_t base_block_index = offset / sizeof(__m128i);

		std::size_t block_offset = offset % sizeof(__m128i);

		__m128i ctr_block, keystream_block;

		std::uint64_t local_counter = base_block_index;

		std::size_t iterator = 0;

		// Handle initial unaligned prefix (if any)
		if (block_offset > 0) {

			alignas(16) std::uint8_t keystream_buf[sizeof(__m128i)];

			ctr_block = _mm_set_epi64x(iv, local_counter);

			keystream_block = _mm_xor_si128(ctr_block, key_schedule[0]);

			for (std::size_t j = 1; j < nr; ++j)
				keystream_block = _mm_aesenc_si128(keystream_block, key_schedule[j]);

			keystream_block = _mm_aesenclast_si128(keystream_block, key_schedule[nr]);

			_mm_store_si128(reinterpret_cast<__m128i*>(keystream_buf), keystream_block);

			std::size_t chunk_size = sizeof(__m128i) - block_offset;

			for (std::size_t j = 0; j < chunk_size && j < len; ++j)
				dst[j] = src[offset + j] ^ keystream_buf[block_offset + j];

			iterator += chunk_size;

			++local_counter;
		}

		std::int64_t block_ct = (len - iterator) / sizeof(__m128i);
		
#pragma omp parallel for if(thr) 
		for (std::int64_t i = 0; i < block_ct; ++i) {

			// Localize Variables for Multi-thread Compatibility
			const __m128i _counter_block = _mm_set_epi64x(iv, local_counter + i);

			__m128i _keystream_block = _mm_xor_si128(_counter_block, key_schedule[0]);

			for (std::size_t j = 1; j < nr; ++j)
				_keystream_block = _mm_aesenc_si128(_keystream_block, key_schedule[j]);

			_keystream_block = _mm_aesenclast_si128(_keystream_block, key_schedule[nr]);

			const __m128i plaintext128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + offset + iterator + (i * sizeof(__m128i))));

			const __m128i ciphertext128 = _mm_xor_si128(plaintext128, _keystream_block);

			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + iterator + (i * sizeof(__m128i))), ciphertext128);
		}

		const std::size_t loop_progress = block_ct * sizeof(__m128i);

		iterator += loop_progress;

		local_counter += block_ct;

		// Handle tail (if any)
		if (iterator < len) {

			alignas(16) std::uint8_t keystream_buf[sizeof(__m128i)];

			ctr_block = _mm_set_epi64x(iv, local_counter);

			keystream_block = _mm_xor_si128(ctr_block, key_schedule[0]);

			for (std::size_t j = 1; j < nr; ++j)
				keystream_block = _mm_aesenc_si128(keystream_block, key_schedule[j]);

			keystream_block = _mm_aesenclast_si128(keystream_block, key_schedule[nr]);

			_mm_store_si128(reinterpret_cast<__m128i*>(keystream_buf), keystream_block);

			const std::size_t tail_size = len - iterator;

			for (std::size_t j = 0; j < tail_size; ++j)
				dst[iterator + j] = src[offset + iterator + j] ^ keystream_buf[j];
		}

		counter = local_counter;

		if(destroy_iv)
			SECURE_ZERO_MEMORY(

				&iv,

				sizeof(std::uint64_t)
			);

		return true;
	}

	FORCE_INLINE const bool FASTCALL aes128_decrypt_range(
		
		std::uint8_t*			dst,

		const std::uint8_t*		src,

		const std::size_t		offset,

		const std::size_t		len,

		std::uint64_t&			iv,

		// Allow default arg for key, incase has already been passed in and expanded
		void*					key = nullptr,

		const bool				destroy_key = false,

		const bool				destroy_iv	= false,

		const bool				disable_thr = false

	) noexcept {

		return aes128_encrypt_range(

			dst,

			src,

			offset,

			len,

			iv,

			key,

			destroy_key,
			
			destroy_iv,

			disable_thr
		);
	}

	FORCE_INLINE const bool FASTCALL aes128_encrypt(
	
		std::uint8_t*			dst,

		const std::uint8_t*		src,

		const std::size_t		len,

		std::uint64_t&			iv,

		// Allow default arg for key, incase has already been passed in and expanded
		void*					key = nullptr,

		const bool				destroy_key = false,

		const bool				destroy_iv = false,

		const bool				disable_thr = false
		
	) noexcept {

		return aes128_encrypt_range(

			dst,

			src,

			0,

			len,

			iv,

			key,

			destroy_key,
			
			destroy_iv,

			disable_thr
		);
	}

	FORCE_INLINE const bool FASTCALL aes128_decrypt(
	
		std::uint8_t*			dst,

		const std::uint8_t*		src,

		const std::size_t		len,

		std::uint64_t&			iv,

		// Allow default arg for key, incase has already been passed in and expanded
		void*					key = nullptr,

		const bool				destroy_key = false,

		const bool				destroy_iv = false,

		const bool				disable_thr = false
		
	) noexcept {

		return aes128_decrypt_range(

			dst,

			src,

			0,

			len,

			iv,

			key,

			destroy_key,
			
			destroy_iv,

			disable_thr
		);
	}

	FORCE_INLINE FASTCALL aes128_ctr(
		
		void* key, 
		
		const bool wipe_key = false
	
	) noexcept {

		expand_key(key);
	}
};

#endif