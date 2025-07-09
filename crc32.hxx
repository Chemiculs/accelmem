#ifndef CRC32_HXX
#define CRC32_HXX

#include "hc128.hxx"

#include <nmmintrin.h>

#define MM_DO_BLOCK_PREFETCH()													\
			if (!(iterator % 0x40)) {											\
				auto* ptr = data + iterator;									\
				for (std::size_t i = 0; i < std::min(16ui64, ((len - iterator) / 0x40) + (((len - iterator) % 0x40) ? 1 : 0)); ++i)							\
					_mm_prefetch((const char*)(ptr + (0x40 * i)), _MM_HINT_NTA);\
			}

class crc32 {

private:

	std::uint32_t crc = UINT32_MAX;

	static const constexpr std::uint32_t crc32c_polynomial = 0x82F63B78;

	static bool is_galois_field_initialized;

	static std::uint32_t crc32c_galois_field_table[256];

	static inline void STDCALL initialize_galois_field() noexcept {

		if (is_galois_field_initialized)
			return;

		for (std::uint32_t i = 0; i < 256; ++i) {

			std::uint32_t crc = i;

			for (std::uint32_t j = 0; j < 8; ++j)
				if (crc & 0b1ui32)
					crc = (crc >> 0b1ui32) ^ crc32c_polynomial;
				else
					crc >>= 0b1ui32;

			crc32c_galois_field_table[i] = crc;
		}

		is_galois_field_initialized = true;
	}

	inline std::uint32_t FASTCALL software_update_crc(
	
		const std::uint8_t* data,

		const std::size_t len
	
	) noexcept {

		if (!data || !len)
			return UINT32_MAX;

		if(!is_galois_field_initialized)
			initialize_galois_field();

		crc = ~crc;

		std::size_t iterator = 0;

#define CRC_PROCESS_BLOCK(BLOCK)													\
				for (std::uint32_t x = 0; x < sizeof(BLOCK); ++x) {				\
					std::uint8_t low8 = static_cast<std::uint8_t>(crc ^ BLOCK); \
					crc = crc32c_galois_field_table[low8] ^ (crc >> 8);			\
					BLOCK >>= 8;												\
				}

		if (accelmem::g_is_sse2) {

			while (iterator + sizeof(__m128i) < len) {

				MM_DO_BLOCK_PREFETCH();

				const __m128i data_block = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + iterator));

#ifdef _WIN64

				alignas(0x8) std::uint64_t block1 = _mm_extract_epi64(data_block, 0);

				CRC_PROCESS_BLOCK(block1);

				alignas(0x8) std::uint64_t block2 = _mm_extract_epi64(data_block, 1);

				CRC_PROCESS_BLOCK(block2);

#else

				alignas(0x4) std::uint32_t block1 = _mm_extract_epi32(data_block, 0);

				CRC_PROCESS_BLOCK(block1);

				alignas(0x4) std::uint32_t block2 = _mm_extract_epi32(data_block, 1);

				CRC_PROCESS_BLOCK(block2);

				alignas(0x4) std::uint32_t block3 = _mm_extract_epi32(data_block, 2);

				CRC_PROCESS_BLOCK(block3);

				alignas(0x4) std::uint32_t block4 = _mm_extract_epi32(data_block, 3);

				CRC_PROCESS_BLOCK(block4);

#endif

				iterator += sizeof(__m128i);
			}
		}

#ifdef  _WIN64

		alignas(0x8) std::uint64_t buffer64 = NULL;

		while (iterator + sizeof(std::uint64_t) < len) {

			MM_DO_BLOCK_PREFETCH();

			buffer64 = *reinterpret_cast<const std::uint64_t*>(data + iterator);

			CRC_PROCESS_BLOCK(buffer64);

			iterator += sizeof(std::uint64_t);
		}

#else

		alignas(0x4) std::uint32_t buffer32 = NULL;

		while (iterator + sizeof(std::uint32_t) < len) {

			MM_DO_BLOCK_PREFETCH();

			buffer32 = *reinterpret_cast<const std::uint32_t*>(data + iterator);

			CRC_PROCESS_BLOCK(buffer32);

			iterator += sizeof(std::uint32_t);
		}

#endif

		std::uint8_t buffer = NULL;

		while (iterator < len) {

			buffer = *(data + iterator);

			crc = (crc >> 8) ^ crc32c_galois_field_table[buffer ^ static_cast<std::uint8_t>(crc & 0xFF)];

			++iterator;
		}

		return crc = ~crc;
	}

public:

	inline const std::uint32_t STDCALL get_crc() const noexcept {

		return crc;
	}

	inline void STDCALL reset_crc() noexcept {

		crc = UINT32_MAX;
	}

	inline std::uint32_t FASTCALL update_crc(
	
		const std::uint8_t* data,

		const std::size_t len

	) noexcept {

		if (!data || !len)
			return UINT32_MAX;

		if(!accelmem::g_is_simd_check)
			accelmem::cpu_check_simd();

		if(!accelmem::g_is_sse4_2)
			return software_update_crc(data, len);

		crc = ~crc;

		std::size_t iterator = 0;

		std::uint8_t buffer8 = NULL;

		alignas(0x4) std::uint32_t buffer32 = NULL;

		if (accelmem::g_is_sse2) {

			while (iterator + sizeof(__m128i) <= len) {

				MM_DO_BLOCK_PREFETCH();

				const __m128i data_block = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + iterator));

				crc = _mm_crc32_u64(crc, _mm_extract_epi64(data_block, 0));

				crc = _mm_crc32_u64(crc, _mm_extract_epi64(data_block, 1));

				iterator += sizeof(__m128i);
			}
		}

#ifdef _WIN64

		alignas(0x8) std::uint64_t buffer64 = NULL;

		while (iterator + sizeof(std::uint64_t) <= len) {

			MM_DO_BLOCK_PREFETCH();

			buffer64 = *reinterpret_cast<const std::uint64_t*>(data + iterator);

			crc = _mm_crc32_u64(crc, buffer64);

			iterator += sizeof(std::uint64_t);
		}

		SECURE_ZERO_MEMORY(

			&buffer64,

			sizeof(std::uint64_t)
		);

#else

		while (iterator + sizeof(std::uint32_t) <= len) {

			MM_DO_BLOCK_PREFETCH();

			buffer32 = *reinterpret_cast<const std::uint32_t*>(data + iterator);

			crc = _mm_crc32_u32(crc, buffer32);

			iterator += sizeof(std::uint32_t);
		}

		SECURE_ZERO_MEMORY(

			&buffer32,

			sizeof(std::uint32_t)
		);

#endif

#undef MM_DO_BLOCK_PREFETCH

		while (iterator < len) {

			buffer8 = *(data + iterator);

			crc = _mm_crc32_u8(crc, buffer8);

			++iterator;
		}

		SECURE_ZERO_MEMORY(

			&buffer8,

			sizeof(std::uint8_t)
		);

		return crc = ~crc;
	}
};

bool crc32::is_galois_field_initialized = false;

std::uint32_t crc32::crc32c_galois_field_table[256];

#endif