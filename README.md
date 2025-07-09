# accelmem  
**C++ Cross-Platform SSE / AVX Intrinsic‑Accelerated, Multi‑threaded & Inlined Memory Operations, Hashing, and Encryption**

---

## ▶️ Description
Header-only library providing highly optimized memory primitives (memset, memcpy, memmove), Data-Hashing algorithms, and Encryption utilities accelerated with x86 SIMD (SSE/AVX/AVX2). It supports Multi-threading via OpenMP and performs runtime CPU feature detection.  

The `main.cxx` file demonstrates example usage of the library, and is in no way necessary to include in your Project.

The Library will Query the Host Hardware for CPU Intrinsics, Specifically SSE2 - 4.2, AVX, AVX2, and AES-NI. 
These Intrinsics will be Used if Available, if not the Algorithms shall fall-back to another Highly-Optimized Implementation (Excluding AES-128 - Which REQUIRES AES-NI Intrinsics for now)

---

## ▶️ Features

<details>
<summary><strong>✅ SIMD-Accelerated Memory Operations</strong></summary>

- Block-level memory functions using SSE2, AVX, and AVX2 intrinsics.
- Runtime dispatch based on CPU capabilities and OS feature detection (CPUID + XGETBV).
- Clean fallback to scalar paths when SIMD is unavailable.
</details>

<details>
<summary><strong>🔐 Hashing</strong></summary>

- Implements a high-performance CRC32C Implementation Utilizing Hardware Intrinsics for Acceleration, Safely Falls back to Software Variants if Intrinsics are not Detected on Host CPU
- Optimized for throughput in large data blocks.
</details>

<details>
<summary><strong>🔏 Encryption</strong></summary>

- Intrinsic-accelerated routines for symmetric encryption primitives (AES-NI Intrinsic Accelerated AES-128 CTR-Mode Cipher, and an SSE2 128-bit Register-Optimized HC128 Implementation).
- Thread-safe, inlined for minimal call overhead.
</details>

<details>
<summary><strong>⚙️ Multi-threaded Engines</strong></summary>

- Parallel processing using OpenMP.
- User-defined chunking enables efficient utilization of multi-core systems.
</details>

<details>
<summary><strong>🧩 Utility & Compatibility Layer</strong></summary>

- Cross-platform abstractions for compiler intrinsics (`__m128i`, `__m256i`, `_xgetbv`, etc.).

</details>

## ▶️ Usage Notes

<details>

* AES-128 CTR Has no fallback if Intrinsic Instructions are Unavailable at the Moment, the Functions will simply return False in this Case

* CRC32C Does not Support Multi-Threading unfortunately, due to the Design of the Cycle-Redundancy-Check Algorithm, as well as the HC-128 Encryption Cipher (AES, and General Memory Operations DO Support MT)

* memmove Implementation was Excluded from this Project, as Multithreading is Incompatible with the Overlapping-Copy Mechanism it must Employ

* By Default, accelmem only begins Utilizing OpenMP for Multithreading when dealing with ~100mb Data Allocations, this Threshhold may be Modulated by re-defining this Macro :

'''cpp
#define OMP_MEM_THR_THRESHHOLD <Threshhold_in_Bytes>

</details>
