#ifndef OMP_MEM_HXX
#define OMP_MEM_HXX

#pragma region Imports

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <bit>

#include <omp.h>

#include <immintrin.h>
#include <intrin.h>
	
#ifdef __clang__

#include <cpuid.h>
#include <x86intrin.h>

#endif

#pragma endregion

#pragma region Preprocessor

// Size Threshhold Dictating use of Multithreading
#define OMP_MEM_THR_THRESHHOLD 0x6400000

#define SECURE_ZERO_MEMORY(block, block_len) \
	accelmem::a_vol_memset((volatile void*)(block), (block_len), 0x0ui8)

#ifdef NDEBUG

#ifdef _MSC_VER

#pragma optimize("", on)
#pragma inline_depth(255)
#pragma inline_recursion(on) 

#endif

#endif

#ifndef __clang__

#define FORCE_INLINE __forceinline

#define FASTCALL

#define STDCALL

#else

#define FORCE_INLINE inline __attribute__((always_inline))

#define FASTCALL __fastcall

#define STDCALL __stdcall

#endif

#pragma endregion

/*
	SIMD + Multithread Optimized Memory Operations 

	* memmove Excluded due to Incompatibility with Multithreading 
*/

class accelmem {

public:

#pragma region Singleton

	static const constexpr std::size_t g_chunk_size = 0x400;

	static volatile bool g_is_sse2;

	static volatile bool g_is_sse4_2;

	static volatile bool g_is_aes_ni;

	static volatile bool g_is_avx;

	static volatile bool g_is_avx2;

	static volatile bool g_is_simd_check;

#pragma endregion

#pragma region Helper Functions

#pragma region Hardware Detection

	static FORCE_INLINE void STDCALL cpu_check_simd() noexcept {

		if (g_is_simd_check)
			return;

#pragma region SSE2 Check

		std::int32_t cpu_info[4] = {};

#ifndef __clang__

		__cpuid(cpu_info, 1);

#else

		__cpuid(1, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);

#endif

		g_is_sse2 = (cpu_info[3] & (1 << 26)) != 0;

		g_is_sse4_2 = (cpu_info[2] & (1 << 20)) != 0;

		g_is_aes_ni = (cpu_info[2] & (1 << 25)) != 0;

#pragma endregion

#pragma region AVX Check

		bool osxsave = (cpu_info[2] & (1 << 27)) != 0;

		g_is_avx = (cpu_info[2] & (1 << 28)) != 0;

		if (osxsave && g_is_avx) {

#ifndef __clang__

			std::uint64_t xcr0 = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);

#else
			std::uint32_t eax, edx;

			__asm__ volatile (
				"xgetbv"
				: "=a"(eax), "=d"(edx)
				: "c"(0)
				);

			std::uint64_t xcr0 = ((std::uint64_t)edx << 32) | eax;

#endif

			g_is_avx = (xcr0 & 0x6) == 0x6;
		}
		else {

			g_is_avx = false;
		}

#pragma endregion

#pragma region AVX2 Check

#ifndef __clang__

		__cpuidex(cpu_info, 7, 0);
#else

		__cpuid_count(7, 0, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#endif

		g_is_avx2 = (cpu_info[1] & (1 << 5)) != 0;

#pragma endregion

		g_is_simd_check = true;
	}

#pragma endregion

#pragma region Volatile Memory Alignment

	static FORCE_INLINE void FASTCALL vol_i_align_alloc_set(

		volatile void* dst,

		const std::uint8_t pattern,

		std::size_t len,

		std::size_t alignment,

		volatile void** dst_aligned_out,

		std::size_t* len_aligned_out

	) noexcept {

		std::uintptr_t misalign = reinterpret_cast<std::uintptr_t>(dst) & alignment;

		if (!misalign) {

			*dst_aligned_out = const_cast<void*>(dst);

			*len_aligned_out = len;

			return;
		}

		std::size_t fixup_bytes = (alignment + 1) - misalign;

		std::size_t iterator = 0;

		while (fixup_bytes - iterator) {

			*(reinterpret_cast<volatile std::uint8_t*>(dst) + iterator) = pattern;

			++iterator;
		}

		*dst_aligned_out = reinterpret_cast<void*>(

			reinterpret_cast<std::uintptr_t>(dst) + fixup_bytes
			);

		*len_aligned_out = len - fixup_bytes;
	}

#pragma endregion

#pragma region Memory Alignment

	static FORCE_INLINE void FASTCALL i_align_alloc_set(
		
		void* dst,

		const std::uint8_t pattern,

		std::size_t len,

		std::size_t alignment,

		void** dst_aligned_out,

		std::size_t* len_aligned_out

	) noexcept {

		std::uintptr_t misalign = reinterpret_cast<std::uintptr_t>(dst) & alignment;

		if (!misalign) {

			*dst_aligned_out = dst;

			*len_aligned_out = len;

			return;	
		}

		std::size_t fixup_bytes = (alignment + 1) - misalign;

		std::size_t iterator = 0;

		while (fixup_bytes - iterator) {

			*(reinterpret_cast<std::uint8_t*>(dst) + iterator) = pattern;

			++iterator;
		}
		

		*dst_aligned_out = reinterpret_cast<void*>(
			
			reinterpret_cast<std::uintptr_t>(dst) + fixup_bytes
		);

		*len_aligned_out = len - fixup_bytes;
	}

	static FORCE_INLINE void  FASTCALL i_align_alloc_copy(
		
		void* dst,

		const std::int64_t& src_offset,

		std::size_t len,

		std::size_t alignment,

		void** dst_aligned_out,

		std::size_t* len_aligned_out

	) noexcept {

		if (!dst || !len || !src_offset || !alignment || !dst_aligned_out || !len_aligned_out)
			return;

		std::uintptr_t misalign = reinterpret_cast<std::uintptr_t>(dst) & alignment;

		if (!misalign) {

			*dst_aligned_out = dst;

			*len_aligned_out = len;

			return;
		}

		std::size_t fixup_bytes = (alignment + 1) - misalign;

		std::size_t iterator = 0;

		while (fixup_bytes - iterator) {

			*(reinterpret_cast<std::uint8_t*>(dst) + iterator) = *((reinterpret_cast<std::uint8_t*>(dst) + src_offset) + iterator);
				
			++iterator;
		}

		*dst_aligned_out = reinterpret_cast<void*>(
				
			reinterpret_cast<std::uintptr_t>(dst) + fixup_bytes
		);

		*len_aligned_out = len - fixup_bytes;
	}

	static FORCE_INLINE std::size_t i_align_alloc_compare(
	
		const void* data2,

		const std::size_t len,

		const std::size_t alignment,

		const std::size_t offset,

		const void** aligned_dst_out,

		std::size_t* aligned_len_out

	) noexcept {

		if (!data2 || !offset || !len)
			return sizeof(std::size_t) == sizeof(std::uint32_t) ? UINT32_MAX : UINT64_MAX;

		const std::uintptr_t misalign = reinterpret_cast<std::uintptr_t>(data2) & alignment;

		if (!misalign) {

			*aligned_dst_out = const_cast<void*>(data2);

			*aligned_len_out = len;

			return 0;
		}

		std::size_t fixup_bytes = (alignment + 1) - misalign;

		std::size_t diff_counter = 0;

		std::size_t iterator = 0;

		// Since we want an accurate Differential-Byte count, we skip using GPR-size (uintptr_t) Comparisons
		while (fixup_bytes - iterator) {

			if (reinterpret_cast<const std::uint8_t*>(data2)[iterator] != reinterpret_cast<const std::uint8_t*>(data2)[iterator + offset])
				++diff_counter;

			++iterator;
		}

		*aligned_dst_out = reinterpret_cast<void*>(
			
			reinterpret_cast<std::uintptr_t>(data2) + fixup_bytes
		);

		*aligned_len_out = len - fixup_bytes;

		return diff_counter;
	}

#pragma endregion

#pragma region Other

	static FORCE_INLINE const std::uint32_t i_bit_mismatch(
		
		const std::int32_t mask,
		
		const bool is_avx_mask = false
	
	) noexcept {

		const std::uint32_t mask_all_matched = is_avx_mask ? UINT32_MAX : UINT16_MAX;

		const std::uint8_t max_matched = is_avx_mask ? 0x20 : 0x10;

		std::uint32_t c_mask = static_cast<std::uint32_t>(mask);

		if (c_mask == mask_all_matched)
			return 0;

		// countof(true bits)
		const std::uint32_t matched = std::popcount<std::uint32_t>(mask);

		// 0x10 = max_truebits - countof(truebits) = countof(false_bits)
		return max_matched - matched;
		
	}

#pragma endregion

#pragma endregion

#pragma region Volatile Memory Operations

	static FORCE_INLINE void FASTCALL a_vol_memset(

		volatile void* dst,

		const std::size_t	len,

		const std::uint8_t	pattern,

		const bool			use_simd = true,

		const bool			force_sse2 = false,

		const bool			disable_thr = false

	) noexcept {

		if (!dst)
			return;

		if (!g_is_simd_check)
			cpu_check_simd();

#define DECLARE_BASE_LOCALS(DST_NAME, ALIGNED_DESTINATION_T)													\
		const bool ignore_avx				= g_is_avx2 ? force_sse2 : false;						\
		const bool is_skip_simd = (!g_is_sse2 || !use_simd || len < 0x4F);							\
		const bool thr = disable_thr ? false : len >= OMP_MEM_THR_THRESHHOLD;						\
		ALIGNED_DESTINATION_T* aligned_dst = reinterpret_cast<ALIGNED_DESTINATION_T*>(DST_NAME);		\
		std::size_t aligned_len = len;

		DECLARE_BASE_LOCALS(dst, volatile std::uint8_t);

		std::uintptr_t gpr_pattern = NULL;

		for (int i = 0; i < sizeof(std::uintptr_t); ++i)
			*(reinterpret_cast<std::uint8_t*>(&gpr_pattern) + i) = pattern;

		if (!is_skip_simd) {

			const std::uintptr_t alignment = g_is_avx2 ? (force_sse2 ? 0xF : 0x1F) : 0xF;

			vol_i_align_alloc_set(

				dst,

				pattern,

				len,

				alignment,

				reinterpret_cast<volatile void**>(&aligned_dst),

				&aligned_len
			);

			const std::size_t chunk_count = aligned_len / g_chunk_size;

			if (g_is_sse2 && (!g_is_avx2 || ignore_avx)) {

				const __m128i xmm_pattern = _mm_set1_epi8((char)pattern);

#pragma omp parallel for if(thr)
				for (std::int64_t c = 0; c < static_cast<std::int64_t>(chunk_count); ++c) {

#define DO_VOL_MEMSET_LOOP(TYPE, STREAM_FN, PATTERN)																				\
					volatile TYPE* chunk_dst = reinterpret_cast<volatile TYPE*>(aligned_dst) + c * (g_chunk_size / sizeof(TYPE));	\
					for (std::size_t x = 0; x < g_chunk_size; x += (sizeof(TYPE) * 4)) {											\
						STREAM_FN(const_cast<TYPE*>(chunk_dst), PATTERN);															\
						STREAM_FN(const_cast<TYPE*>(chunk_dst) + 1, PATTERN);														\
						STREAM_FN(const_cast<TYPE*>(chunk_dst) + 2, PATTERN);														\
						STREAM_FN(const_cast<TYPE*>(chunk_dst) + 3, PATTERN);														\
						chunk_dst += 4;																								\
					}

					DO_VOL_MEMSET_LOOP(__m128i, _mm_stream_si128, xmm_pattern);
				}
			}
			else if (g_is_avx2 && !ignore_avx) {

				const __m256i ymm_pattern = _mm256_set1_epi8((char)pattern);

#pragma omp parallel for if(thr)
				for (std::int64_t c = 0; c < static_cast<std::int64_t>(chunk_count); ++c) {

					DO_VOL_MEMSET_LOOP(__m256i, _mm256_stream_si256, ymm_pattern);

#undef DO_VOL_MEMSET_LOOP
				}
			}

			aligned_dst += chunk_count * g_chunk_size;

			aligned_len -= chunk_count * g_chunk_size;
		}

		// Medium tail: GPR_X-byte chunks
		while (aligned_len >= sizeof(std::uintptr_t)) {

			*reinterpret_cast<volatile std::uintptr_t*>(aligned_dst) = gpr_pattern;

			aligned_dst += sizeof(std::uintptr_t);

			aligned_len -= sizeof(std::uintptr_t);
		}

		// Small tail: byte-byte Copy remainder
		while (aligned_len) {

			*aligned_dst = pattern;

			++aligned_dst;

			--aligned_len;
		}

		_mm_sfence();
	}

#pragma endregion

#pragma region General Memory Operations

	static FORCE_INLINE std::size_t FASTCALL a_memcmp(
		
		const void*			data,

		const void*			data2,

		const std::size_t	len,

		const bool			use_simd = true,

		const bool			force_sse2 = false,

		const bool			disable_thr = false

	) noexcept {

		if (!data || !data2 || !len)
			return sizeof(std::size_t) == sizeof(std::uint32_t) ? UINT32_MAX : UINT64_MAX;

		if (!g_is_simd_check)
			cpu_check_simd();

		DECLARE_BASE_LOCALS(data2, const std::uint8_t);

		std::size_t diff_counter			= 0;

		const std::int64_t offset			= reinterpret_cast<std::int64_t>(data) - reinterpret_cast<std::int64_t>(data2);

		if (!is_skip_simd) {

			const std::uintptr_t alignment	= g_is_avx2 ? (force_sse2 ? 0xF : 0x1F) : 0xF;

			diff_counter += i_align_alloc_compare(

				data2,

				len,

				alignment,

				offset,

				reinterpret_cast<const void**>(&aligned_dst),

				&aligned_len
			);

			const std::size_t chunk_count	= aligned_len / g_chunk_size;

			if (g_is_sse2 && (!g_is_avx2 || ignore_avx)) {

#pragma omp parallel for if(thr) reduction(+:diff_counter)
				for (std::int64_t c = 0; c < chunk_count; ++c) {

#define MM_DO_MEMCMP_PREFETCH(TYPE)\
					const TYPE* chunk_dst = reinterpret_cast<const TYPE*>(aligned_dst) + c * (g_chunk_size / sizeof(TYPE));\
					const TYPE* chunk_src = reinterpret_cast<const TYPE*>(reinterpret_cast<std::uintptr_t>(chunk_dst) + offset);\
					const TYPE* d_prefetch_block_ptr = chunk_dst;\
					const TYPE* s_prefetch_block_ptr = chunk_src;\
					for (int i = 0; i < g_chunk_size; i += 0x40) {\
						_mm_prefetch(\
							reinterpret_cast<const char*>(d_prefetch_block_ptr),\
							_MM_HINT_NTA\
						);\
						_mm_prefetch(\
							reinterpret_cast<const char*>(s_prefetch_block_ptr),\
							_MM_HINT_NTA\
						);\
						s_prefetch_block_ptr += (0x40 / sizeof(TYPE));\
						d_prefetch_block_ptr += (0x40 / sizeof(TYPE));\
					}

#define DO_MEMCMP_LOOP(TYPE, LOAD_ALIGNED, LOAD_UNALIGNED, CMP, MOVEMASK,  BITARG)		\
					for (std::size_t x = 0; x < g_chunk_size; x += (sizeof(TYPE) * 4)) {\
						const TYPE d_v1 = LOAD_ALIGNED(chunk_dst + 0);					\
						const TYPE s_v1 = LOAD_UNALIGNED(chunk_src + 0);				\
						const TYPE cmp1 = CMP(d_v1, s_v1);								\
						diff_counter += i_bit_mismatch(									\
							MOVEMASK(cmp1),												\
							BITARG														\
						);																\
						const TYPE d_v2 = LOAD_ALIGNED(chunk_dst + 1);					\
						const TYPE s_v2 = LOAD_UNALIGNED(chunk_src + 1);				\
						const TYPE cmp2 = CMP(d_v2, s_v2);								\
						diff_counter += i_bit_mismatch(									\
							MOVEMASK(cmp2),												\
							BITARG														\
						);																\
						const TYPE d_v3 = LOAD_ALIGNED(chunk_dst + 2);					\
						const TYPE s_v3 = LOAD_UNALIGNED(chunk_src + 2);				\
						const TYPE cmp3 = CMP(d_v3, s_v3);								\
						diff_counter += i_bit_mismatch(									\
							MOVEMASK(cmp3),												\
							BITARG														\
						);																\
						const TYPE d_v4 = LOAD_ALIGNED(chunk_dst + 3);					\
						const TYPE s_v4 = LOAD_UNALIGNED(chunk_src + 3);				\
						const TYPE cmp4 = CMP(d_v4, s_v4);								\
						diff_counter += i_bit_mismatch(									\
							MOVEMASK(cmp4),												\
							BITARG														\
						);																\
						chunk_dst += 4;													\
						chunk_src += 4;													\
					}


					MM_DO_MEMCMP_PREFETCH(__m128i);

					DO_MEMCMP_LOOP(__m128i, _mm_load_si128, _mm_loadu_si128, _mm_cmpeq_epi8, _mm_movemask_epi8, false);
				}
			}
			else if (g_is_avx2 && !ignore_avx) {

#pragma omp parallel for if(thr) reduction(+:diff_counter)
				for (std::int64_t c = 0; c < chunk_count; ++c) {

					MM_DO_MEMCMP_PREFETCH(__m256i);

					DO_MEMCMP_LOOP(__m256i, _mm256_load_si256, _mm256_loadu_si256, _mm256_cmpeq_epi8, _mm256_movemask_epi8, true);

#undef MM_DO_MEMCMP_PREFETCH

#undef DO_MEMCMP_LOOP

				}
			}

			aligned_dst += g_chunk_size * chunk_count;

			aligned_len -= g_chunk_size * chunk_count;
		}

		// Medium Tail Comparison
		while (aligned_len >= sizeof(std::uintptr_t)) {

			// Fallback to byte-byte Comparison to Quantify mismatch
			if (*reinterpret_cast<const std::uintptr_t*>(aligned_dst) != *reinterpret_cast<const std::uintptr_t*>(aligned_dst + offset))
				for (std::size_t i = 0; i < sizeof(std::uintptr_t); ++i)
					if (aligned_dst[i] != aligned_dst[i + offset])
						++diff_counter;

			aligned_dst += sizeof(std::uintptr_t);

			aligned_len -= sizeof(std::uintptr_t);
		}

		// Tail Comparison
		while (aligned_len) {

			if (*aligned_dst != aligned_dst[offset])
				++diff_counter;

			++aligned_dst;

			--aligned_len;
		}

		return diff_counter;
	}

	static FORCE_INLINE void FASTCALL a_memset(

		void* dst,

		const std::size_t	len,

		const std::uint8_t	pattern,

		const bool			use_simd = true,

		const bool			force_sse2 = false,

		const bool			disable_thr = false

	) noexcept {

		if (!dst)
			return;

		if (!g_is_simd_check)
			cpu_check_simd();

		DECLARE_BASE_LOCALS(dst, std::uint8_t);

		std::uintptr_t gpr_pattern = NULL;

		for (int i = 0; i < sizeof(std::uintptr_t); ++i)
			*(reinterpret_cast<std::uint8_t*>(&gpr_pattern) + i) = pattern;

		if (!is_skip_simd) {

			const std::uintptr_t alignment = g_is_avx2 ? (force_sse2 ? 0xF : 0x1F) : 0xF;

			i_align_alloc_set(

				dst,

				pattern,

				len,

				alignment,

				reinterpret_cast<void**>(&aligned_dst),

				&aligned_len
			);

			const std::size_t chunk_count = aligned_len / g_chunk_size;

			if (g_is_sse2 && (!g_is_avx2 || ignore_avx)) {

				const __m128i xmm_pattern = _mm_set1_epi8((char)pattern);

#pragma omp parallel for if(thr)
				for (std::int64_t c = 0; c < static_cast<std::int64_t>(chunk_count); ++c) {

#define DO_MEMSET_LOOP(TYPE, STREAM_FN, PATTERN)															\
					TYPE* chunk_dst = reinterpret_cast<TYPE*>(aligned_dst) + c * (g_chunk_size / sizeof(TYPE));	\
					for (std::size_t x = 0; x < g_chunk_size; x += (sizeof(TYPE) * 4)) {						\
						STREAM_FN(chunk_dst,	PATTERN);														\
						STREAM_FN(chunk_dst + 1, PATTERN);														\
						STREAM_FN(chunk_dst + 2, PATTERN);														\
						STREAM_FN(chunk_dst + 3, PATTERN);														\
						chunk_dst += 4;																			\
					}

					DO_MEMSET_LOOP(__m128i, _mm_stream_si128, xmm_pattern);
				}
			}
			else if (g_is_avx2 && !ignore_avx) {

				const __m256i ymm_pattern = _mm256_set1_epi8((char)pattern);

#pragma omp parallel for if(thr)
				for (std::int64_t c = 0; c < static_cast<std::int64_t>(chunk_count); ++c) {

					DO_MEMSET_LOOP(__m256i, _mm256_stream_si256, ymm_pattern);

#undef DO_MEMSET_LOOP
				}
			}

			aligned_dst += chunk_count * g_chunk_size;

			aligned_len -= chunk_count * g_chunk_size;
		}

		// Medium tail: GPR_X-byte chunks
		while (aligned_len >= sizeof(std::uintptr_t)) {

			*reinterpret_cast< std::uintptr_t*>(aligned_dst) = gpr_pattern;

			aligned_dst += sizeof(std::uintptr_t);

			aligned_len -= sizeof(std::uintptr_t);
		}

		// Small tail: byte-byte Copy remainder
		while (aligned_len) {

			*aligned_dst = pattern;

			++aligned_dst;

			--aligned_len;
		}

		_mm_sfence();
	}

	static FORCE_INLINE void FASTCALL a_memcpy(

		void*				dst,

		const void*			src,

		const std::size_t	len,

		const bool			use_simd = true,

		const bool			force_sse2 = false,

		const bool			disable_thr = false

	) noexcept {

		if (!src || !dst || !len)
			return;

		if (!g_is_simd_check)
			cpu_check_simd();

		DECLARE_BASE_LOCALS(dst, void);

#undef DECLARE_BASE_GLOBALS
		
		const std::int64_t offset = reinterpret_cast<std::int64_t>(src) - reinterpret_cast<std::int64_t>(dst);

		if (!is_skip_simd) {

			const std::uintptr_t alignment = g_is_avx2 ? (force_sse2 ? 0xF : 0x1F) : 0xF;

			i_align_alloc_copy(

				dst,

				offset,

				len,

				alignment,

				&aligned_dst,

				&aligned_len
			);

			const std::size_t chunk_count = aligned_len / g_chunk_size;

			if (g_is_sse2 && (!g_is_avx2 || ignore_avx)) {

#pragma omp parallel for if(thr)
				for (std::intptr_t c = 0; c < static_cast<std::intptr_t>(chunk_count); ++c) {

#define MM_DO_MEMCPY_PREFETCH(TYPE)																													\
					TYPE* chunk_dst = reinterpret_cast<TYPE*>(																				\
					reinterpret_cast<std::uint8_t*>(aligned_dst) + (c * g_chunk_size)																\
						);																															\
					TYPE* block_ptr = chunk_dst;																									\
					for (int i = 0; i < g_chunk_size; i += 0x40) {																					\
						_mm_prefetch(																												\
							reinterpret_cast<const char*>(reinterpret_cast<const std::uint8_t*>(block_ptr) + offset),								\
							_MM_HINT_NTA																											\
						);																															\
						block_ptr += (0x40 / sizeof(TYPE));																							\
					}

#define DO_MEMCPY_LOOP(TYPE, STREAM_FN, LOAD_FN)																										\
for (std::size_t x = 0; x < g_chunk_size; x += (sizeof(TYPE) * 4)) {																					\
					TYPE	v1 = LOAD_FN(reinterpret_cast<const TYPE*>(reinterpret_cast<const std::uint8_t*>(chunk_dst) + offset + (sizeof(TYPE) * 0)));\
					TYPE	v2 = LOAD_FN(reinterpret_cast<const TYPE*>(reinterpret_cast<const std::uint8_t*>(chunk_dst) + offset + (sizeof(TYPE) * 1)));\
					STREAM_FN(chunk_dst, v1);																											\
					STREAM_FN(chunk_dst + 1, v2);																										\
					TYPE	v3 = LOAD_FN(reinterpret_cast<const TYPE*>(reinterpret_cast<const std::uint8_t*>(chunk_dst) + offset + (sizeof(TYPE) * 2)));\
					TYPE	v4 = LOAD_FN(reinterpret_cast<const TYPE*>(reinterpret_cast<const std::uint8_t*>(chunk_dst) + offset + (sizeof(TYPE) * 3)));\
					STREAM_FN(chunk_dst + 2, v3);																										\
					STREAM_FN(chunk_dst + 3, v4);																										\
					chunk_dst += 4;																														\
}

					MM_DO_MEMCPY_PREFETCH(__m128i);

					DO_MEMCPY_LOOP(__m128i, _mm_stream_si128, _mm_loadu_si128);
				}
			}
			else if (g_is_avx2 && !ignore_avx) {

#pragma omp parallel for if(thr)
				for (std::int64_t c = 0; c < static_cast<std::int64_t>(chunk_count); ++c) {

					MM_DO_MEMCPY_PREFETCH(__m256i);

					DO_MEMCPY_LOOP(__m256i, _mm256_stream_si256, _mm256_loadu_si256);

#undef MM_DO_MEMCPY_PREFETCH

#undef DO_MEMCPY_LOOP
				}
			}

			aligned_dst = static_cast<void*>(

				reinterpret_cast<std::uint8_t*>(aligned_dst) + (chunk_count * (g_chunk_size))
			);

			aligned_len -= chunk_count * g_chunk_size;
		}

		std::uint8_t* byte_dst = reinterpret_cast<std::uint8_t*>(aligned_dst);

		// Medium tail: GPR_X-byte chunks
		while (aligned_len >= sizeof(std::uintptr_t)) {

			*reinterpret_cast<std::uintptr_t*>(byte_dst) = *reinterpret_cast<const std::uintptr_t*>(byte_dst + offset);

			byte_dst += sizeof(std::uintptr_t);

			aligned_len -= sizeof(std::uintptr_t);
		}

		// Small tail: byte-byte Copy remainder
		while (aligned_len) {

			*byte_dst = *reinterpret_cast<std::uint8_t*>(byte_dst + offset);

			++byte_dst;
			--aligned_len;
		}

		_mm_sfence();
	}

#pragma endregion

};

#pragma region Static Instantiators

volatile bool accelmem::g_is_sse2 = false;

volatile bool accelmem::g_is_sse4_2 = false;

volatile bool accelmem::g_is_aes_ni = false;

volatile bool accelmem::g_is_avx = false;

volatile bool accelmem::g_is_avx2 = false;

volatile bool accelmem::g_is_simd_check = false;

#pragma endregion

#endif