#include "accelmem.hxx"

#include <iostream>
#include <string>
#include <chrono>
#include <random>

std::mt19937_64 rng(static_cast<std::uint64_t>(std::time(nullptr)));

std::uint8_t block1[(UINT16_MAX + 1) * 250];

std::uint8_t block2[(UINT16_MAX + 1) * 250];

std::int32_t main() noexcept {

	std::uint64_t data_r = NULL;

	std::size_t iter = 0;

	do {

		data_r = rng();

		*reinterpret_cast<std::uint64_t*>(&block1[iter]) = data_r;

		iter += sizeof(std::uint64_t);

	} while (iter < sizeof(block1));

	// memcpy Example
	accelmem::a_memcpy(
		
		block2,

		block1,

		sizeof(block1)
	);

	std::size_t r_code = 0;

	if (r_code = accelmem::a_memcmp(block1, block2, sizeof(block1)))
		std::cout << "[-] Memory Integrity Check for block1 memcpy Operation Failed! Invalid Bytes: " << r_code << std::endl;

	accelmem::a_memset(
	
		block1,

		sizeof(block1),

		0xFFui8
	);

	accelmem::a_memset(
		
		block2,

		sizeof(block2),

		0xFFui8
	);

	if(r_code = accelmem::a_memcmp(block1, block2, sizeof(block1)))
		std::cout << "[-] Memory Integrity Check for block2 memcpy Operation Failed! Invalid Bytes: " << r_code << std::endl;

	crc32 crc;

	crc.update_crc(block1, sizeof(block1));

	std::cout << "[+] block1 CRC32 Hash: 0x" << std::hex << std::noshowbase << crc.get_crc() << std::endl;

	crc.reset_crc();

	crc.update_crc(block2, sizeof(block2));

	std::cout << "[+] block2 CRC32 Hash: 0x" << crc.get_crc() << std::endl;

	crc.reset_crc();

	std::uint32_t key[4]{0x12345678, 0xFFFFFFFF, 0xAAAAAAAA, 0xBBBBBBBB};

	std::uint32_t iv[4]{0x87654321, 0xEEEEEEEE, 0xDDDDDDDD, 0xCCCCCCCC};

	hc128 hc;

	hc.encrypt(
		
		block1,

		block1,

		sizeof(block1),

		key,

		iv
	);

	crc.update_crc(
		
		block1, 

		sizeof(block1)
	);

	std::cout << "[+] block1 hc128 Encrypted CRC32 Hash: 0x" << crc.get_crc() << std::endl;

	crc.reset_crc();

	hc.decrypt(
		
		block1,

		block1,

		sizeof(block1),

		key,

		iv
	);

	crc.update_crc(
		
		block1,

		sizeof(block1)
	);

	std::cout << "[+] block1 hc128 Decrypted / Original CRC32 Hash: 0x" << crc.get_crc() << std::endl;

	crc.reset_crc();

	std::uint64_t aes_iv = 0xFAFAFAFAFAFAFAFA;

	aes128_ctr aes(key);

	aes.aes128_encrypt(
		
		block1,

		block1,

		sizeof(block1),

		aes_iv
	);

	crc.update_crc(
		
		block1, 
		
		sizeof(block1)
	);

	std::cout << "[+] block1 aes128-ctr Encrypted CRC32 Hash: 0x" << crc.get_crc() << std::endl;

	crc.reset_crc();

	aes.aes128_decrypt(
		
		block1,

		block1,

		sizeof(block1),

		aes_iv
	);

	crc.update_crc(
		
		block1,

		sizeof(block1)
	);

	std::cout << "[+] block1 aes128-ctr Decrypted / Original CRC32 Hash: 0x" << crc.get_crc() << std::endl;

	SECURE_ZERO_MEMORY(key, sizeof(key));

	SECURE_ZERO_MEMORY(iv, sizeof(iv));

	aes_iv = NULL;

	return NULL;
}