#ifndef CUDA_ADAM_G1_CUH
#define CUDA_ADAM_G1_CUH

#include "cuda_adam_common.cuh"

/* =========================================================================
 * Group 1:
 *   0: BLAKE-512
 *   1: BMW-512
 *   2: Groestl-512
 *   3: JH-512
 * ========================================================================= */

/* ------------------- 0. BLAKE-512 ------------------- */

static __constant__ uint8_t c_adam_blake_sigma[16][16] = {
	{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
	{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
	{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
	{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
	{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
	{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
	{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
	{  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
	{ 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
	{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
	{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
	{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
	{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
	{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 }
};

static __constant__ uint64_t c_adam_blake_u512[16] = {
	0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL,
	0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL,
	0x452821E638D01377ULL, 0xBE5466CF34E90C6CULL,
	0xC0AC29B7C97C50DDULL, 0x3F84D5B5B5470917ULL,
	0x9216D5D98979FB1BULL, 0xD1310BA698DFB5ACULL,
	0x2FFD72DBD01ADFB7ULL, 0xB8E1AFED6A267E96ULL,
	0xBA7C9045F12C7F99ULL, 0x24A19947B3916CF7ULL,
	0x0801F2E2858EFC16ULL, 0x636920D871574E69ULL
};

#define ADAM_BLAKE_G(a, b, c, d, x) do { \
	uint32_t idx1 = c_adam_blake_sigma[r][x]; \
	uint32_t idx2 = c_adam_blake_sigma[r][(x) + 1]; \
	v[a] += (m[idx1] ^ c_adam_blake_u512[idx2]) + v[b]; \
	v[d] = adam_rotr64(v[d] ^ v[a], 32); \
	v[c] += v[d]; \
	v[b] = adam_rotr64(v[b] ^ v[c], 25); \
	v[a] += (m[idx2] ^ c_adam_blake_u512[idx1]) + v[b]; \
	v[d] = adam_rotr64(v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = adam_rotr64(v[b] ^ v[c], 11); \
} while (0)

__device__ __forceinline__ void adam_blake512_single_block(const uint64_t *block, uint64_t bitlen, uint64_t *out32)
{
	uint64_t h[8] = {
		0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
		0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
		0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
		0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
	};
	uint64_t v[16];
	uint64_t m[16];

	#pragma unroll
	for (int i = 0; i < 16; i++)
		m[i] = adam_bswap64(block[i]);

	#pragma unroll
	for (int i = 0; i < 8; i++)
		v[i] = h[i];

	v[ 8] = c_adam_blake_u512[0];
	v[ 9] = c_adam_blake_u512[1];
	v[10] = c_adam_blake_u512[2];
	v[11] = c_adam_blake_u512[3];
	v[12] = c_adam_blake_u512[4] ^ bitlen;
	v[13] = c_adam_blake_u512[5] ^ bitlen;
	v[14] = c_adam_blake_u512[6];
	v[15] = c_adam_blake_u512[7];

	for (int r = 0; r < 16; r++) {
		ADAM_BLAKE_G(0, 4,  8, 12,  0);
		ADAM_BLAKE_G(1, 5,  9, 13,  2);
		ADAM_BLAKE_G(2, 6, 10, 14,  4);
		ADAM_BLAKE_G(3, 7, 11, 15,  6);
		ADAM_BLAKE_G(0, 5, 10, 15,  8);
		ADAM_BLAKE_G(1, 6, 11, 12, 10);
		ADAM_BLAKE_G(2, 7,  8, 13, 12);
		ADAM_BLAKE_G(3, 4,  9, 14, 14);
	}

	#pragma unroll
	for (int i = 0; i < 4; i++) {
		uint64_t hw = h[i] ^ v[i] ^ v[i + 8];
		out32[i] = adam_bswap64(hw);
	}
}

#undef ADAM_BLAKE_G

__global__ void adam_gpu_blake512_70(uint32_t threads, uint32_t start_nonce, uint64_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint64_t block[16];
	adam_load_le64_70(block, start_nonce + tid);
	block[8] |= (0x80ULL << 48);
	block[9] = 0; block[10] = 0; block[11] = 0; block[12] = 0;
	block[13] = 0x0100000000000000ULL;
	block[14] = 0;
	block[15] = adam_bswap64(560ULL);
	uint64_t out32[4];
	adam_blake512_single_block(block, 560ULL, out32);
	adam_store_le64_32(g_hash, tid, out32);
}

__global__ void adam_gpu_blake512_32(uint32_t threads, uint64_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint64_t block[16];
	adam_load_le64_32(block, g_hash, tid);
	block[4] = 0x80ULL;
	block[5] = 0; block[6] = 0; block[7] = 0; block[8] = 0;
	block[9] = 0; block[10] = 0; block[11] = 0; block[12] = 0;
	block[13] = 0x0100000000000000ULL;
	block[14] = 0;
	block[15] = adam_bswap64(256ULL);
	uint64_t out32[4];
	adam_blake512_single_block(block, 256ULL, out32);
	adam_store_le64_32(g_hash, tid, out32);
}

/* ------------------- 1. BMW-512 ------------------- */

static __constant__ uint64_t c_adam_bmw_iv[16] = {
	0x8081828384858687ULL, 0x88898A8B8C8D8E8FULL,
	0x9091929394959697ULL, 0x98999A9B9C9D9E9FULL,
	0xA0A1A2A3A4A5A6A7ULL, 0xA8A9AAABACADAEAFULL,
	0xB0B1B2B3B4B5B6B7ULL, 0xB8B9BABBBCBDBEBFULL,
	0xC0C1C2C3C4C5C6C7ULL, 0xC8C9CACBCCCDCECFULL,
	0xD0D1D2D3D4D5D6D7ULL, 0xD8D9DADBDCDDDEDFULL,
	0xE0E1E2E3E4E5E6E7ULL, 0xE8E9EAEBECEDEEEFULL,
	0xF0F1F2F3F4F5F6F7ULL, 0xF8F9FAFBFCFDFEFFULL
};

#define ADAM_BMW_S0(x) (adam_rotr64(x, 1) ^ (x))
#define ADAM_BMW_S1(x) (((x) >> 1) ^ ((x) << 3) ^ adam_rotl64(x, 4) ^ adam_rotl64(x, 37))
#define ADAM_BMW_S2(x) (((x) >> 1) ^ ((x) << 2) ^ adam_rotl64(x, 13) ^ adam_rotl64(x, 43))
#define ADAM_BMW_S3(x) (((x) >> 2) ^ ((x) << 1) ^ adam_rotl64(x, 19) ^ adam_rotl64(x, 53))
#define ADAM_BMW_S4(x) (((x) >> 2) ^ ((x) << 2) ^ adam_rotl64(x, 28) ^ adam_rotl64(x, 59))
#define ADAM_BMW_S5(x) (((x) >> 1) ^ (x))

#define ADAM_BMW_EXP2 \
	q[i + 0] + adam_rotl64(q[i + 1], 5) + q[i + 2] + adam_rotl64(q[i + 3], 11) + \
	q[i + 4] + adam_rotl64(q[i + 5], 27) + q[i + 6] + adam_rotl64(q[i + 7], 32) + \
	q[i + 8] + adam_rotl64(q[i + 9], 37) + q[i + 10] + adam_rotl64(q[i + 11], 43) + \
	q[i + 12] + adam_rotl64(q[i + 13], 53) + (((q[i + 14]) >> 1) ^ q[i + 14]) + (((q[i + 15]) >> 2) ^ q[i + 15])

__device__ void adam_bmw512_compress(const uint64_t *msg, uint64_t *hash)
{
	uint64_t tmp;
	uint64_t q[32];

	tmp = (msg[ 5] ^ hash[ 5]) - (msg[ 7] ^ hash[ 7]) + (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]) + (msg[14] ^ hash[14]);
	q[ 0] = ADAM_BMW_S1(tmp) + hash[ 1];
	tmp = (msg[ 6] ^ hash[ 6]) - (msg[ 8] ^ hash[ 8]) + (msg[11] ^ hash[11]) + (msg[14] ^ hash[14]) - (msg[15] ^ hash[15]);
	q[ 1] = ADAM_BMW_S2(tmp) + hash[ 2];
	tmp = (msg[ 0] ^ hash[ 0]) + (msg[ 7] ^ hash[ 7]) + (msg[ 9] ^ hash[ 9]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
	q[ 2] = ADAM_BMW_S3(tmp) + hash[ 3];
	tmp = (msg[ 0] ^ hash[ 0]) - (msg[ 1] ^ hash[ 1]) + (msg[ 8] ^ hash[ 8]) - (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]);
	q[ 3] = ADAM_BMW_S4(tmp) + hash[ 4];
	tmp = (msg[ 1] ^ hash[ 1]) + (msg[ 2] ^ hash[ 2]) + (msg[ 9] ^ hash[ 9]) - (msg[11] ^ hash[11]) - (msg[14] ^ hash[14]);
	q[ 4] = ADAM_BMW_S5(tmp) + hash[ 5];
	tmp = (msg[ 3] ^ hash[ 3]) - (msg[ 2] ^ hash[ 2]) + (msg[10] ^ hash[10]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
	q[ 5] = ADAM_BMW_S1(tmp) + hash[ 6];
	tmp = (msg[ 4] ^ hash[ 4]) - (msg[ 0] ^ hash[ 0]) - (msg[ 3] ^ hash[ 3]) - (msg[11] ^ hash[11]) + (msg[13] ^ hash[13]);
	q[ 6] = ADAM_BMW_S2(tmp) + hash[ 7];
	tmp = (msg[ 1] ^ hash[ 1]) - (msg[ 4] ^ hash[ 4]) - (msg[ 5] ^ hash[ 5]) - (msg[12] ^ hash[12]) - (msg[14] ^ hash[14]);
	q[ 7] = ADAM_BMW_S3(tmp) + hash[ 8];
	tmp = (msg[ 2] ^ hash[ 2]) - (msg[ 5] ^ hash[ 5]) - (msg[ 6] ^ hash[ 6]) + (msg[13] ^ hash[13]) - (msg[15] ^ hash[15]);
	q[ 8] = ADAM_BMW_S4(tmp) + hash[ 9];
	tmp = (msg[ 0] ^ hash[ 0]) - (msg[ 3] ^ hash[ 3]) + (msg[ 6] ^ hash[ 6]) - (msg[ 7] ^ hash[ 7]) + (msg[14] ^ hash[14]);
	q[ 9] = ADAM_BMW_S5(tmp) + hash[10];
	tmp = (msg[ 8] ^ hash[ 8]) - (msg[ 1] ^ hash[ 1]) - (msg[ 4] ^ hash[ 4]) - (msg[ 7] ^ hash[ 7]) + (msg[15] ^ hash[15]);
	q[10] = ADAM_BMW_S1(tmp) + hash[11];
	tmp = (msg[ 8] ^ hash[ 8]) - (msg[ 0] ^ hash[ 0]) - (msg[ 2] ^ hash[ 2]) - (msg[ 5] ^ hash[ 5]) + (msg[ 9] ^ hash[ 9]);
	q[11] = ADAM_BMW_S2(tmp) + hash[12];
	tmp = (msg[ 1] ^ hash[ 1]) + (msg[ 3] ^ hash[ 3]) - (msg[ 6] ^ hash[ 6]) - (msg[ 9] ^ hash[ 9]) + (msg[10] ^ hash[10]);
	q[12] = ADAM_BMW_S3(tmp) + hash[13];
	tmp = (msg[ 2] ^ hash[ 2]) + (msg[ 4] ^ hash[ 4]) + (msg[ 7] ^ hash[ 7]) + (msg[10] ^ hash[10]) + (msg[11] ^ hash[11]);
	q[13] = ADAM_BMW_S4(tmp) + hash[14];
	tmp = (msg[ 3] ^ hash[ 3]) - (msg[ 5] ^ hash[ 5]) + (msg[ 8] ^ hash[ 8]) - (msg[11] ^ hash[11]) - (msg[12] ^ hash[12]);
	q[14] = ADAM_BMW_S5(tmp) + hash[15];
	tmp = (msg[12] ^ hash[12]) - (msg[ 4] ^ hash[ 4]) - (msg[ 6] ^ hash[ 6]) - (msg[ 9] ^ hash[ 9]) + (msg[13] ^ hash[13]);
	q[15] = ADAM_BMW_S1(tmp) + hash[ 0];

	for (int i = 0; i < 2; i++) {
		q[i + 16] =
			ADAM_BMW_S2(q[i +  0]) + ADAM_BMW_S3(q[i +  1]) + ADAM_BMW_S4(q[i +  2]) + ADAM_BMW_S1(q[i +  3]) +
			ADAM_BMW_S2(q[i +  4]) + ADAM_BMW_S3(q[i +  5]) + ADAM_BMW_S4(q[i +  6]) + ADAM_BMW_S1(q[i +  7]) +
			ADAM_BMW_S2(q[i +  8]) + ADAM_BMW_S3(q[i +  9]) + ADAM_BMW_S4(q[i + 10]) + ADAM_BMW_S1(q[i + 11]) +
			ADAM_BMW_S2(q[i + 12]) + ADAM_BMW_S3(q[i + 13]) + ADAM_BMW_S4(q[i + 14]) + ADAM_BMW_S1(q[i + 15]) +
			((((uint64_t)(i + 16) * 0x0555555555555555ULL) + adam_rotl64(msg[i], i + 1) +
			  adam_rotl64(msg[i + 3], i + 4) - adam_rotl64(msg[i + 10], i + 11)) ^ hash[i + 7]);
	}
	for (int i = 2; i < 6; i++) {
		q[i + 16] = ADAM_BMW_EXP2 +
			((((uint64_t)(i + 16) * 0x0555555555555555ULL) + adam_rotl64(msg[i], i + 1) +
			  adam_rotl64(msg[i + 3], i + 4) - adam_rotl64(msg[i + 10], i + 11)) ^ hash[i + 7]);
	}
	for (int i = 6; i < 9; i++) {
		q[i + 16] = ADAM_BMW_EXP2 +
			((((uint64_t)(i + 16) * 0x0555555555555555ULL) + adam_rotl64(msg[i], i + 1) +
			  adam_rotl64(msg[i + 3], i + 4) - adam_rotl64(msg[i - 6], (i - 6) + 1)) ^ hash[i + 7]);
	}
	for (int i = 9; i < 13; i++) {
		q[i + 16] = ADAM_BMW_EXP2 +
			((((uint64_t)(i + 16) * 0x0555555555555555ULL) + adam_rotl64(msg[i], i + 1) +
			  adam_rotl64(msg[i + 3], i + 4) - adam_rotl64(msg[i - 6], (i - 6) + 1)) ^ hash[i - 9]);
	}
	for (int i = 13; i < 16; i++) {
		q[i + 16] = ADAM_BMW_EXP2 +
			((((uint64_t)(i + 16) * 0x0555555555555555ULL) + adam_rotl64(msg[i], i + 1) +
			  adam_rotl64(msg[i - 13], (i - 13) + 1) - adam_rotl64(msg[i - 6], (i - 6) + 1)) ^ hash[i - 9]);
	}

	uint64_t XL = q[16] ^ q[17] ^ q[18] ^ q[19] ^ q[20] ^ q[21] ^ q[22] ^ q[23];
	uint64_t XH = XL ^ q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

	hash[0] = ((XH <<  5) ^ (q[16] >> 5) ^ msg[0]) + (XL ^ q[24] ^ q[0]);
	hash[1] = ((XH >>  7) ^ (q[17] << 8) ^ msg[1]) + (XL ^ q[25] ^ q[1]);
	hash[2] = ((XH >>  5) ^ (q[18] << 5) ^ msg[2]) + (XL ^ q[26] ^ q[2]);
	hash[3] = ((XH >>  1) ^ (q[19] << 5) ^ msg[3]) + (XL ^ q[27] ^ q[3]);
	hash[4] = ((XH >>  3) ^  q[20]       ^ msg[4]) + (XL ^ q[28] ^ q[4]);
	hash[5] = ((XH <<  6) ^ (q[21] >> 6) ^ msg[5]) + (XL ^ q[29] ^ q[5]);
	hash[6] = ((XH >>  4) ^ (q[22] << 6) ^ msg[6]) + (XL ^ q[30] ^ q[6]);
	hash[7] = ((XH >> 11) ^ (q[23] << 2) ^ msg[7]) + (XL ^ q[31] ^ q[7]);

	hash[ 8] = adam_rotl64(hash[4],  9) + (XH ^ q[24] ^ msg[ 8]) + ((XL << 8) ^ q[23] ^ q[ 8]);
	hash[ 9] = adam_rotl64(hash[5], 10) + (XH ^ q[25] ^ msg[ 9]) + ((XL >> 6) ^ q[16] ^ q[ 9]);
	hash[10] = adam_rotl64(hash[6], 11) + (XH ^ q[26] ^ msg[10]) + ((XL << 6) ^ q[17] ^ q[10]);
	hash[11] = adam_rotl64(hash[7], 12) + (XH ^ q[27] ^ msg[11]) + ((XL << 4) ^ q[18] ^ q[11]);
	hash[12] = adam_rotl64(hash[0], 13) + (XH ^ q[28] ^ msg[12]) + ((XL >> 3) ^ q[19] ^ q[12]);
	hash[13] = adam_rotl64(hash[1], 14) + (XH ^ q[29] ^ msg[13]) + ((XL >> 4) ^ q[20] ^ q[13]);
	hash[14] = adam_rotl64(hash[2], 15) + (XH ^ q[30] ^ msg[14]) + ((XL >> 7) ^ q[21] ^ q[14]);
	hash[15] = adam_rotl64(hash[3], 16) + (XH ^ q[31] ^ msg[15]) + ((XL >> 2) ^ q[22] ^ q[15]);
}

#undef ADAM_BMW_S0
#undef ADAM_BMW_S1
#undef ADAM_BMW_S2
#undef ADAM_BMW_S3
#undef ADAM_BMW_S4
#undef ADAM_BMW_S5
#undef ADAM_BMW_EXP2

__device__ __forceinline__ void adam_bmw512_single_block(const uint64_t *msg, uint64_t *out32)
{
	uint64_t h[16];
	#pragma unroll
	for (int i = 0; i < 16; i++)
		h[i] = c_adam_bmw_iv[i];

	adam_bmw512_compress(msg, h);

	uint64_t fin[16];
	#pragma unroll
	for (int i = 0; i < 16; i++)
		fin[i] = 0xAAAAAAAAAAAAAAA0ULL + (uint64_t)i;

	adam_bmw512_compress(h, fin);

	#pragma unroll
	for (int i = 0; i < 4; i++)
		out32[i] = fin[i + 8];
}

__global__ void adam_gpu_bmw512_70(uint32_t threads, uint32_t start_nonce, uint64_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint64_t msg[16];
	adam_load_le64_70(msg, start_nonce + tid);
	msg[8] |= (0x80ULL << 48);
	for (int i = 9; i < 15; i++)
		msg[i] = 0;
	msg[15] = 560ULL;
	uint64_t out32[4];
	adam_bmw512_single_block(msg, out32);
	adam_store_le64_32(g_hash, tid, out32);
}

__global__ void adam_gpu_bmw512_32(uint32_t threads, uint64_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint64_t msg[16];
	adam_load_le64_32(msg, g_hash, tid);
	msg[4] = 0x80ULL;
	for (int i = 5; i < 15; i++)
		msg[i] = 0;
	msg[15] = 256ULL;
	uint64_t out32[4];
	adam_bmw512_single_block(msg, out32);
	adam_store_le64_32(g_hash, tid, out32);
}

/* ------------------- 2. Groestl-512 ------------------- */

#define ADAM_C64E(x) ( \
	(((uint64_t)(x)) >> 56) | \
	((((uint64_t)(x)) >> 40) & 0x000000000000FF00ULL) | \
	((((uint64_t)(x)) >> 24) & 0x0000000000FF0000ULL) | \
	((((uint64_t)(x)) >>  8) & 0x00000000FF000000ULL) | \
	((((uint64_t)(x)) <<  8) & 0x000000FF00000000ULL) | \
	((((uint64_t)(x)) << 24) & 0x0000FF0000000000ULL) | \
	((((uint64_t)(x)) << 40) & 0x00FF000000000000ULL) | \
	((((uint64_t)(x)) << 56) & 0xFF00000000000000ULL))

static __constant__ uint64_t c_adam_groestl_T0[256] = {
	ADAM_C64E(0xc632f4a5f497a5c6ULL), ADAM_C64E(0xf86f978497eb84f8ULL),
	ADAM_C64E(0xee5eb099b0c799eeULL), ADAM_C64E(0xf67a8c8d8cf78df6ULL),
	ADAM_C64E(0xffe8170d17e50dffULL), ADAM_C64E(0xd60adcbddcb7bdd6ULL),
	ADAM_C64E(0xde16c8b1c8a7b1deULL), ADAM_C64E(0x916dfc54fc395491ULL),
	ADAM_C64E(0x6090f050f0c05060ULL), ADAM_C64E(0x0207050305040302ULL),
	ADAM_C64E(0xce2ee0a9e087a9ceULL), ADAM_C64E(0x56d1877d87ac7d56ULL),
	ADAM_C64E(0xe7cc2b192bd519e7ULL), ADAM_C64E(0xb513a662a67162b5ULL),
	ADAM_C64E(0x4d7c31e6319ae64dULL), ADAM_C64E(0xec59b59ab5c39aecULL),
	ADAM_C64E(0x8f40cf45cf05458fULL), ADAM_C64E(0x1fa3bc9dbc3e9d1fULL),
	ADAM_C64E(0x8949c040c0094089ULL), ADAM_C64E(0xfa68928792ef87faULL),
	ADAM_C64E(0xefd03f153fc515efULL), ADAM_C64E(0xb29426eb267febb2ULL),
	ADAM_C64E(0x8ece40c94007c98eULL), ADAM_C64E(0xfbe61d0b1ded0bfbULL),
	ADAM_C64E(0x416e2fec2f82ec41ULL), ADAM_C64E(0xb31aa967a97d67b3ULL),
	ADAM_C64E(0x5f431cfd1cbefd5fULL), ADAM_C64E(0x456025ea258aea45ULL),
	ADAM_C64E(0x23f9dabfda46bf23ULL), ADAM_C64E(0x535102f702a6f753ULL),
	ADAM_C64E(0xe445a196a1d396e4ULL), ADAM_C64E(0x9b76ed5bed2d5b9bULL),
	ADAM_C64E(0x75285dc25deac275ULL), ADAM_C64E(0xe1c5241c24d91ce1ULL),
	ADAM_C64E(0x3dd4e9aee97aae3dULL), ADAM_C64E(0x4cf2be6abe986a4cULL),
	ADAM_C64E(0x6c82ee5aeed85a6cULL), ADAM_C64E(0x7ebdc341c3fc417eULL),
	ADAM_C64E(0xf5f3060206f102f5ULL), ADAM_C64E(0x8352d14fd11d4f83ULL),
	ADAM_C64E(0x688ce45ce4d05c68ULL), ADAM_C64E(0x515607f407a2f451ULL),
	ADAM_C64E(0xd18d5c345cb934d1ULL), ADAM_C64E(0xf9e1180818e908f9ULL),
	ADAM_C64E(0xe24cae93aedf93e2ULL), ADAM_C64E(0xab3e9573954d73abULL),
	ADAM_C64E(0x6297f553f5c45362ULL), ADAM_C64E(0x2a6b413f41543f2aULL),
	ADAM_C64E(0x081c140c14100c08ULL), ADAM_C64E(0x9563f652f6315295ULL),
	ADAM_C64E(0x46e9af65af8c6546ULL), ADAM_C64E(0x9d7fe25ee2215e9dULL),
	ADAM_C64E(0x3048782878602830ULL), ADAM_C64E(0x37cff8a1f86ea137ULL),
	ADAM_C64E(0x0a1b110f11140f0aULL), ADAM_C64E(0x2febc4b5c45eb52fULL),
	ADAM_C64E(0x0e151b091b1c090eULL), ADAM_C64E(0x247e5a365a483624ULL),
	ADAM_C64E(0x1badb69bb6369b1bULL), ADAM_C64E(0xdf98473d47a53ddfULL),
	ADAM_C64E(0xcda76a266a8126cdULL), ADAM_C64E(0x4ef5bb69bb9c694eULL),
	ADAM_C64E(0x7f334ccd4cfecd7fULL), ADAM_C64E(0xea50ba9fbacf9feaULL),
	ADAM_C64E(0x123f2d1b2d241b12ULL), ADAM_C64E(0x1da4b99eb93a9e1dULL),
	ADAM_C64E(0x58c49c749cb07458ULL), ADAM_C64E(0x3446722e72682e34ULL),
	ADAM_C64E(0x3641772d776c2d36ULL), ADAM_C64E(0xdc11cdb2cda3b2dcULL),
	ADAM_C64E(0xb49d29ee2973eeb4ULL), ADAM_C64E(0x5b4d16fb16b6fb5bULL),
	ADAM_C64E(0xa4a501f60153f6a4ULL), ADAM_C64E(0x76a1d74dd7ec4d76ULL),
	ADAM_C64E(0xb714a361a37561b7ULL), ADAM_C64E(0x7d3449ce49face7dULL),
	ADAM_C64E(0x52df8d7b8da47b52ULL), ADAM_C64E(0xdd9f423e42a13eddULL),
	ADAM_C64E(0x5ecd937193bc715eULL), ADAM_C64E(0x13b1a297a2269713ULL),
	ADAM_C64E(0xa6a204f50457f5a6ULL), ADAM_C64E(0xb901b868b86968b9ULL),
	ADAM_C64E(0x0000000000000000ULL), ADAM_C64E(0xc1b5742c74992cc1ULL),
	ADAM_C64E(0x40e0a060a0806040ULL), ADAM_C64E(0xe3c2211f21dd1fe3ULL),
	ADAM_C64E(0x793a43c843f2c879ULL), ADAM_C64E(0xb69a2ced2c77edb6ULL),
	ADAM_C64E(0xd40dd9bed9b3bed4ULL), ADAM_C64E(0x8d47ca46ca01468dULL),
	ADAM_C64E(0x671770d970ced967ULL), ADAM_C64E(0x72afdd4bdde44b72ULL),
	ADAM_C64E(0x94ed79de7933de94ULL), ADAM_C64E(0x98ff67d4672bd498ULL),
	ADAM_C64E(0xb09323e8237be8b0ULL), ADAM_C64E(0x855bde4ade114a85ULL),
	ADAM_C64E(0xbb06bd6bbd6d6bbbULL), ADAM_C64E(0xc5bb7e2a7e912ac5ULL),
	ADAM_C64E(0x4f7b34e5349ee54fULL), ADAM_C64E(0xedd73a163ac116edULL),
	ADAM_C64E(0x86d254c55417c586ULL), ADAM_C64E(0x9af862d7622fd79aULL),
	ADAM_C64E(0x6699ff55ffcc5566ULL), ADAM_C64E(0x11b6a794a7229411ULL),
	ADAM_C64E(0x8ac04acf4a0fcf8aULL), ADAM_C64E(0xe9d9301030c910e9ULL),
	ADAM_C64E(0x040e0a060a080604ULL), ADAM_C64E(0xfe66988198e781feULL),
	ADAM_C64E(0xa0ab0bf00b5bf0a0ULL), ADAM_C64E(0x78b4cc44ccf04478ULL),
	ADAM_C64E(0x25f0d5bad54aba25ULL), ADAM_C64E(0x4b753ee33e96e34bULL),
	ADAM_C64E(0xa2ac0ef30e5ff3a2ULL), ADAM_C64E(0x5d4419fe19bafe5dULL),
	ADAM_C64E(0x80db5bc05b1bc080ULL), ADAM_C64E(0x0580858a850a8a05ULL),
	ADAM_C64E(0x3fd3ecadec7ead3fULL), ADAM_C64E(0x21fedfbcdf42bc21ULL),
	ADAM_C64E(0x70a8d848d8e04870ULL), ADAM_C64E(0xf1fd0c040cf904f1ULL),
	ADAM_C64E(0x63197adf7ac6df63ULL), ADAM_C64E(0x772f58c158eec177ULL),
	ADAM_C64E(0xaf309f759f4575afULL), ADAM_C64E(0x42e7a563a5846342ULL),
	ADAM_C64E(0x2070503050403020ULL), ADAM_C64E(0xe5cb2e1a2ed11ae5ULL),
	ADAM_C64E(0xfdef120e12e10efdULL), ADAM_C64E(0xbf08b76db7656dbfULL),
	ADAM_C64E(0x8155d44cd4194c81ULL), ADAM_C64E(0x18243c143c301418ULL),
	ADAM_C64E(0x26795f355f4c3526ULL), ADAM_C64E(0xc3b2712f719d2fc3ULL),
	ADAM_C64E(0xbe8638e13867e1beULL), ADAM_C64E(0x35c8fda2fd6aa235ULL),
	ADAM_C64E(0x88c74fcc4f0bcc88ULL), ADAM_C64E(0x2e654b394b5c392eULL),
	ADAM_C64E(0x936af957f93d5793ULL), ADAM_C64E(0x55580df20daaf255ULL),
	ADAM_C64E(0xfc619d829de382fcULL), ADAM_C64E(0x7ab3c947c9f4477aULL),
	ADAM_C64E(0xc827efacef8bacc8ULL), ADAM_C64E(0xba8832e7326fe7baULL),
	ADAM_C64E(0x324f7d2b7d642b32ULL), ADAM_C64E(0xe642a495a4d795e6ULL),
	ADAM_C64E(0xc03bfba0fb9ba0c0ULL), ADAM_C64E(0x19aab398b3329819ULL),
	ADAM_C64E(0x9ef668d16827d19eULL), ADAM_C64E(0xa322817f815d7fa3ULL),
	ADAM_C64E(0x44eeaa66aa886644ULL), ADAM_C64E(0x54d6827e82a87e54ULL),
	ADAM_C64E(0x3bdde6abe676ab3bULL), ADAM_C64E(0x0b959e839e16830bULL),
	ADAM_C64E(0x8cc945ca4503ca8cULL), ADAM_C64E(0xc7bc7b297b9529c7ULL),
	ADAM_C64E(0x6b056ed36ed6d36bULL), ADAM_C64E(0x286c443c44503c28ULL),
	ADAM_C64E(0xa72c8b798b5579a7ULL), ADAM_C64E(0xbc813de23d63e2bcULL),
	ADAM_C64E(0x1631271d272c1d16ULL), ADAM_C64E(0xad379a769a4176adULL),
	ADAM_C64E(0xdb964d3b4dad3bdbULL), ADAM_C64E(0x649efa56fac85664ULL),
	ADAM_C64E(0x74a6d24ed2e84e74ULL), ADAM_C64E(0x1436221e22281e14ULL),
	ADAM_C64E(0x92e476db763fdb92ULL), ADAM_C64E(0x0c121e0a1e180a0cULL),
	ADAM_C64E(0x48fcb46cb4906c48ULL), ADAM_C64E(0xb88f37e4376be4b8ULL),
	ADAM_C64E(0x9f78e75de7255d9fULL), ADAM_C64E(0xbd0fb26eb2616ebdULL),
	ADAM_C64E(0x43692aef2a86ef43ULL), ADAM_C64E(0xc435f1a6f193a6c4ULL),
	ADAM_C64E(0x39dae3a8e372a839ULL), ADAM_C64E(0x31c6f7a4f762a431ULL),
	ADAM_C64E(0xd38a593759bd37d3ULL), ADAM_C64E(0xf274868b86ff8bf2ULL),
	ADAM_C64E(0xd583563256b132d5ULL), ADAM_C64E(0x8b4ec543c50d438bULL),
	ADAM_C64E(0x6e85eb59ebdc596eULL), ADAM_C64E(0xda18c2b7c2afb7daULL),
	ADAM_C64E(0x018e8f8c8f028c01ULL), ADAM_C64E(0xb11dac64ac7964b1ULL),
	ADAM_C64E(0x9cf16dd26d23d29cULL), ADAM_C64E(0x49723be03b92e049ULL),
	ADAM_C64E(0xd81fc7b4c7abb4d8ULL), ADAM_C64E(0xacb915fa1543faacULL),
	ADAM_C64E(0xf3fa090709fd07f3ULL), ADAM_C64E(0xcfa06f256f8525cfULL),
	ADAM_C64E(0xca20eaafea8fafcaULL), ADAM_C64E(0xf47d898e89f38ef4ULL),
	ADAM_C64E(0x476720e9208ee947ULL), ADAM_C64E(0x1038281828201810ULL),
	ADAM_C64E(0x6f0b64d564ded56fULL), ADAM_C64E(0xf073838883fb88f0ULL),
	ADAM_C64E(0x4afbb16fb1946f4aULL), ADAM_C64E(0x5cca967296b8725cULL),
	ADAM_C64E(0x38546c246c702438ULL), ADAM_C64E(0x575f08f108aef157ULL),
	ADAM_C64E(0x732152c752e6c773ULL), ADAM_C64E(0x9764f351f3355197ULL),
	ADAM_C64E(0xcbae6523658d23cbULL), ADAM_C64E(0xa125847c84597ca1ULL),
	ADAM_C64E(0xe857bf9cbfcb9ce8ULL), ADAM_C64E(0x3e5d6321637c213eULL),
	ADAM_C64E(0x96ea7cdd7c37dd96ULL), ADAM_C64E(0x611e7fdc7fc2dc61ULL),
	ADAM_C64E(0x0d9c9186911a860dULL), ADAM_C64E(0x0f9b9485941e850fULL),
	ADAM_C64E(0xe04bab90abdb90e0ULL), ADAM_C64E(0x7cbac642c6f8427cULL),
	ADAM_C64E(0x712657c457e2c471ULL), ADAM_C64E(0xcc29e5aae583aaccULL),
	ADAM_C64E(0x90e373d8733bd890ULL), ADAM_C64E(0x06090f050f0c0506ULL),
	ADAM_C64E(0xf7f4030103f501f7ULL), ADAM_C64E(0x1c2a36123638121cULL),
	ADAM_C64E(0xc23cfea3fe9fa3c2ULL), ADAM_C64E(0x6a8be15fe1d45f6aULL),
	ADAM_C64E(0xaebe10f91047f9aeULL), ADAM_C64E(0x69026bd06bd2d069ULL),
	ADAM_C64E(0x17bfa891a82e9117ULL), ADAM_C64E(0x9971e858e8295899ULL),
	ADAM_C64E(0x3a5369276974273aULL), ADAM_C64E(0x27f7d0b9d04eb927ULL),
	ADAM_C64E(0xd991483848a938d9ULL), ADAM_C64E(0xebde351335cd13ebULL),
	ADAM_C64E(0x2be5ceb3ce56b32bULL), ADAM_C64E(0x2277553355443322ULL),
	ADAM_C64E(0xd204d6bbd6bfbbd2ULL), ADAM_C64E(0xa9399070904970a9ULL),
	ADAM_C64E(0x07878089800e8907ULL), ADAM_C64E(0x33c1f2a7f266a733ULL),
	ADAM_C64E(0x2decc1b6c15ab62dULL), ADAM_C64E(0x3c5a66226678223cULL),
	ADAM_C64E(0x15b8ad92ad2a9215ULL), ADAM_C64E(0xc9a96020608920c9ULL),
	ADAM_C64E(0x875cdb49db154987ULL), ADAM_C64E(0xaab01aff1a4fffaaULL),
	ADAM_C64E(0x50d8887888a07850ULL), ADAM_C64E(0xa52b8e7a8e517aa5ULL),
	ADAM_C64E(0x03898a8f8a068f03ULL), ADAM_C64E(0x594a13f813b2f859ULL),
	ADAM_C64E(0x09929b809b128009ULL), ADAM_C64E(0x1a2339173934171aULL),
	ADAM_C64E(0x651075da75cada65ULL), ADAM_C64E(0xd784533153b531d7ULL),
	ADAM_C64E(0x84d551c65113c684ULL), ADAM_C64E(0xd003d3b8d3bbb8d0ULL),
	ADAM_C64E(0x82dc5ec35e1fc382ULL), ADAM_C64E(0x29e2cbb0cb52b029ULL),
	ADAM_C64E(0x5ac3997799b4775aULL), ADAM_C64E(0x1e2d3311333c111eULL),
	ADAM_C64E(0x7b3d46cb46f6cb7bULL), ADAM_C64E(0xa8b71ffc1f4bfca8ULL),
	ADAM_C64E(0x6d0c61d661dad66dULL), ADAM_C64E(0x2c624e3a4e583a2cULL)
};

#undef ADAM_C64E

__device__ __forceinline__ uint64_t adam_groestl_rbtt(
	const uint64_t *a,
	int b0, int b1, int b2, int b3, int b4, int b5, int b6, int b7)
{
	return
		                 c_adam_groestl_T0[(a[b0]      ) & 0xFF]       ^
		adam_rotl64(     c_adam_groestl_T0[(a[b1] >>  8) & 0xFF],  8)  ^
		adam_rotl64(     c_adam_groestl_T0[(a[b2] >> 16) & 0xFF], 16)  ^
		adam_rotl64(     c_adam_groestl_T0[(a[b3] >> 24) & 0xFF], 24)  ^
		adam_rotl64(     c_adam_groestl_T0[(a[b4] >> 32) & 0xFF], 32)  ^
		adam_rotl64(     c_adam_groestl_T0[(a[b5] >> 40) & 0xFF], 40)  ^
		adam_rotl64(     c_adam_groestl_T0[(a[b6] >> 48) & 0xFF], 48)  ^
		adam_rotl64(     c_adam_groestl_T0[(a[b7] >> 56)       ], 56);
}

__device__ void adam_groestl_perm_p(uint64_t *a)
{
	uint64_t t[16];
	for (uint32_t r = 0; r < 14; r++) {
		#pragma unroll
		for (int j = 0; j < 16; j++)
			a[j] ^= (uint64_t)((j << 4) + r);
		#pragma unroll
		for (int u = 0; u < 16; u++) {
			t[u] = adam_groestl_rbtt(a,
				(u + 0) & 0xF, (u + 1) & 0xF, (u + 2) & 0xF, (u + 3) & 0xF,
				(u + 4) & 0xF, (u + 5) & 0xF, (u + 6) & 0xF, (u + 11) & 0xF);
		}
		#pragma unroll
		for (int u = 0; u < 16; u++)
			a[u] = t[u];
	}
}

__device__ void adam_groestl_perm_q(uint64_t *a)
{
	uint64_t t[16];
	for (uint32_t r = 0; r < 14; r++) {
		#pragma unroll
		for (int j = 0; j < 16; j++)
			a[j] ^= ((uint64_t)r << 56) ^ (~((uint64_t)(j << 4) << 56));
		#pragma unroll
		for (int u = 0; u < 16; u++) {
			t[u] = adam_groestl_rbtt(a,
				(u + 1) & 0xF, (u + 3) & 0xF, (u + 5) & 0xF, (u + 11) & 0xF,
				(u + 0) & 0xF, (u + 2) & 0xF, (u + 4) & 0xF, (u + 6) & 0xF);
		}
		#pragma unroll
		for (int u = 0; u < 16; u++)
			a[u] = t[u];
	}
}

__device__ __forceinline__ void adam_groestl512_single_block(const uint64_t *msg, uint64_t *out32)
{
	uint64_t H[16];
	uint64_t g[16];
	uint64_t m[16];

	#pragma unroll
	for (int i = 0; i < 15; i++)
		H[i] = 0;
	H[15] = 0x0002000000000000ULL; /* out_size = 512 (LE64 swapped BE16 at end) */

	#pragma unroll
	for (int i = 0; i < 16; i++) {
		m[i] = msg[i];
		g[i] = m[i] ^ H[i];
	}

	adam_groestl_perm_p(g);
	adam_groestl_perm_q(m);

	#pragma unroll
	for (int i = 0; i < 16; i++) {
		H[i] ^= g[i] ^ m[i];
		g[i] = H[i];
	}

	adam_groestl_perm_p(g);

	#pragma unroll
	for (int i = 0; i < 4; i++)
		out32[i] = H[i + 8] ^ g[i + 8];
}

__global__ void adam_gpu_groestl512_70(uint32_t threads, uint32_t start_nonce, uint64_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint64_t msg[16];
	adam_load_le64_70(msg, start_nonce + tid);
	msg[8] |= (0x80ULL << 48);
	for (int i = 9; i < 15; i++)
		msg[i] = 0;
	msg[15] = 0x0100000000000000ULL;
	uint64_t out32[4];
	adam_groestl512_single_block(msg, out32);
	adam_store_le64_32(g_hash, tid, out32);
}

__global__ void adam_gpu_groestl512_32(uint32_t threads, uint64_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint64_t msg[16];
	adam_load_le64_32(msg, g_hash, tid);
	msg[4] = 0x80ULL;
	for (int i = 5; i < 15; i++)
		msg[i] = 0;
	msg[15] = 0x0100000000000000ULL;
	uint64_t out32[4];
	adam_groestl512_single_block(msg, out32);
	adam_store_le64_32(g_hash, tid, out32);
}

/* ------------------- 3. JH-512 ------------------- */

static __constant__ __align__(16) uint32_t c_adam_jh_E8[42][8] = {
	{ 0xa2ded572U, 0x90d6ab81U, 0x67f815dfU, 0xf6875a4dU, 0x0a15847bU, 0xc54f9f4eU, 0x571523b7U, 0x402bd1c3U },
	{ 0xe03a98eaU, 0xb4960266U, 0x9cfa455cU, 0x8a53bbf2U, 0x99d2c503U, 0x1a1456b5U, 0x9a99b266U, 0x31a2db88U },
	{ 0x5c5aa303U, 0x8019051cU, 0xdb0e199aU, 0x1d959e84U, 0x0ab23f40U, 0xadeb336fU, 0x1044c187U, 0xdccde75eU },
	{ 0x9213ba10U, 0x39812c0aU, 0x416bbf02U, 0x5078aa37U, 0x156578dcU, 0xd2bf1a3fU, 0xd027bbf7U, 0xd3910041U },
	{ 0x0d5a2d42U, 0x0ba75c18U, 0x907eccf6U, 0xac442bc7U, 0x9c9f62ddU, 0xd665dfd1U, 0xce97c092U, 0x23fcc663U },
	{ 0x036c6e97U, 0xbb03f1eeU, 0x1ab8e09eU, 0xfa618e5dU, 0x7e450521U, 0xb29796fdU, 0xa8ec6c44U, 0x97818394U },
	{ 0x37858e4aU, 0x8173fe8aU, 0x2f3003dbU, 0x6c69b8f8U, 0x2d8d672aU, 0x4672c78aU, 0x956a9ffbU, 0x14427fc0U },
	{ 0x8f15f4c5U, 0xb775de52U, 0xc45ec7bdU, 0xbc88e4aeU, 0xa76f4475U, 0x1e00b882U, 0x80bb118fU, 0xf4a3a698U },
	{ 0x338ff48eU, 0x20edf1b6U, 0x1563a3a9U, 0xfde05a7cU, 0x24565faaU, 0x5ae9ca36U, 0x89f9b7d5U, 0x362c4206U },
	{ 0x433529ceU, 0x591ff5d0U, 0x3d98fe4eU, 0x86814e6fU, 0x74f93a53U, 0x81ad9d0eU, 0xa74b9a73U, 0x9f5ad8afU },
	{ 0x670605a7U, 0x26077447U, 0x6a6234eeU, 0x3f1080c6U, 0xbe280b8bU, 0x6f7ea0e0U, 0x2717b96eU, 0x7b487ec6U },
	{ 0xa50a550dU, 0x81727686U, 0xc0a4f84aU, 0xd48d6050U, 0x9fe7e391U, 0x415a9e7eU, 0x9ef18e97U, 0x62b0e5f3U },
	{ 0xec1f9ffcU, 0xf594d74fU, 0x7a205440U, 0xd895fa9dU, 0x001ae4e3U, 0x117e2e55U, 0x84c9f4ceU, 0xa554c324U },
	{ 0x2872df5bU, 0xef7c8905U, 0x286efebdU, 0x2ed349eeU, 0xe27ff578U, 0x85937e44U, 0xb2c4a50fU, 0x7f5928ebU },
	{ 0x37695f70U, 0x04771bc7U, 0x4a3124b3U, 0xe720b951U, 0xf128865eU, 0xe843fe74U, 0x65e4d61dU, 0x8a87d423U },
	{ 0xa3e8297dU, 0xfb301b1dU, 0xf2947692U, 0xe01bdc5bU, 0x097acbddU, 0x4f4924daU, 0xc1d9309bU, 0xbf829cf2U },
	{ 0x31bae7a4U, 0x32fcae3bU, 0xffbf70b4U, 0x39d3bb53U, 0x0544320dU, 0xc1c39f45U, 0x48bcf8deU, 0xa08b29e0U },
	{ 0xfd05c9e5U, 0x01b771a2U, 0x0f09aef7U, 0x95ed44e3U, 0x12347094U, 0x368e3be9U, 0x34f19042U, 0x4a982f4fU },
	{ 0x631d4088U, 0xf14abb7eU, 0x15f66ca0U, 0x30c60ae2U, 0x4b44c147U, 0xc5b67046U, 0xffaf5287U, 0xe68c6eccU },
	{ 0x56a4d5a4U, 0x45ce5773U, 0x00ca4fbdU, 0xadd16430U, 0x4b849ddaU, 0x68cea6e8U, 0xae183ec8U, 0x67255c14U },
	{ 0xf28cdaa3U, 0x20b2601fU, 0x16e10ecbU, 0x7b846fc2U, 0x5806e933U, 0x7facced1U, 0x9a99949aU, 0x1885d1a0U },
	{ 0xa15b5932U, 0x67633d9fU, 0xd319dd8dU, 0xba6b04e4U, 0xc01c9a50U, 0xab19caf6U, 0x46b4a5aaU, 0x7eee560bU },
	{ 0xea79b11fU, 0x5aac571dU, 0x742128a9U, 0x76d35075U, 0x35f7bde9U, 0xfec2463aU, 0xee51363bU, 0x01707da3U },
	{ 0xafc135f7U, 0x15638341U, 0x42d8a498U, 0xa8db3aeaU, 0x20eced78U, 0x4d3bc3faU, 0x79676b9eU, 0x832c8332U },
	{ 0x1f3b40a7U, 0x6c4e3ee7U, 0xf347271cU, 0xfd4f21d2U, 0x34f04059U, 0x398dfdb8U, 0x9a762db7U, 0xef5957dcU },
	{ 0x490c9b8dU, 0xd0ae3b7dU, 0xdaeb492bU, 0x84558d7aU, 0x49d7a25bU, 0xf0e9a5f5U, 0x0d70f368U, 0x658ef8e4U },
	{ 0xf4a2b8a0U, 0x92946891U, 0x533b1036U, 0x4f88e856U, 0x9e07a80cU, 0x555cb05bU, 0x5aec3e75U, 0x4cbcbaf8U },
	{ 0x993bbbe3U, 0x28acae64U, 0x7b9487f3U, 0x6db334dcU, 0xd6f4da75U, 0x50a5346cU, 0x5d1c6b72U, 0x71db28b8U },
	{ 0xf2e261f8U, 0xf1bcac1cU, 0x2a518d10U, 0xa23fce43U, 0x3364dbe3U, 0x3cd1bb67U, 0xfc75dd59U, 0xb043e802U },
	{ 0xca5b0a33U, 0xc3943b92U, 0x75a12988U, 0x1e4d790eU, 0x4d19347fU, 0xd7757479U, 0x5c5316b4U, 0x3fafeeb6U },
	{ 0xf7d4a8eaU, 0x5324a326U, 0x21391abeU, 0xd23c32baU, 0x097ef45cU, 0x4a17a344U, 0x5127234cU, 0xadd5a66dU },
	{ 0xa63e1db5U, 0xa17cf84cU, 0x08c9f2afU, 0x4d608672U, 0x983d5983U, 0xcc3ee246U, 0x563c6b91U, 0xf6c76e08U },
	{ 0xb333982fU, 0xe8b6f406U, 0x5e76bcb1U, 0x36d4c1beU, 0xa566d62bU, 0x1582ee74U, 0x2ae6c4efU, 0x6321efbcU },
	{ 0x0d4ec1fdU, 0x1614c17eU, 0x69c953f4U, 0x16fae006U, 0xc45a7da7U, 0x3daf907eU, 0x26585806U, 0x3f9d6328U },
	{ 0xe3f2c9d2U, 0x16512a74U, 0x0cd29b00U, 0x9832e0f2U, 0x30ceaa5fU, 0xd830eb0dU, 0x300cd4b7U, 0x9af8cee3U },
	{ 0x7b9ec54bU, 0x574d239bU, 0x9279f1b5U, 0x316796e6U, 0x6ee651ffU, 0xf3a6e6ccU, 0xd3688604U, 0x05750a17U },
	{ 0xd98176b1U, 0xb3cb2bf4U, 0xce6c3213U, 0x47154778U, 0x8452173cU, 0x825446ffU, 0x62a205f8U, 0x486a9323U },
	{ 0x0758df38U, 0x442e7031U, 0x65655e4eU, 0x86ca0bd0U, 0x897cfcf2U, 0xa20940f0U, 0x8e5086fcU, 0x4e477830U },
	{ 0x39eea065U, 0x26b29721U, 0x8338f7d1U, 0x6ff81301U, 0x37e95ef7U, 0xd1ed44a3U, 0xbd3a2ce4U, 0xe7de9fefU },
	{ 0x15dfa08bU, 0x7ceca7d8U, 0xd9922576U, 0x7eb027abU, 0xf6f7853cU, 0xda7d8d53U, 0xbe42dc12U, 0xdea83eaaU },
	{ 0x93ce25aaU, 0xdaef5fc0U, 0xd86902bdU, 0xa5194a17U, 0xfd43f65aU, 0x33664d97U, 0xf908731aU, 0x6a21fd4cU },
	{ 0x3198b435U, 0xa163d09aU, 0x701541dbU, 0x72409751U, 0xbb0f1eeaU, 0xbf9d75f6U, 0x9b54cdedU, 0xe26f4791U }
};

#define ADAM_JH_SWAP1(x) do { uint32_t _y = (x) & 0xAAAAAAAAU; (x) = (((x) ^ _y) << 1) | (_y >> 1); } while (0)
#define ADAM_JH_SWAP2(x) do { uint32_t _y = (x) & 0xCCCCCCCCU; (x) = (((x) ^ _y) << 2) | (_y >> 2); } while (0)
#define ADAM_JH_SWAP4(x) do { uint32_t _y = (x) & 0xF0F0F0F0U; (x) = (((x) ^ _y) << 4) | (_y >> 4); } while (0)
#define ADAM_JH_SWAP8(x)  ((x) = __byte_perm((x), 0, 0x2301))
#define ADAM_JH_SWAP16(x) ((x) = __byte_perm((x), 0, 0x1032))

__device__ __forceinline__ void adam_jh_sbox_mds(uint32_t x[8][4], int rnd)
{
	const uint2 *cc = (const uint2 *)&c_adam_jh_E8[rnd][0];
	#pragma unroll
	for (int i = 0; i < 4; i++, ++cc) {
		uint32_t m0 = x[0][i], m2 = x[2][i], m4 = x[4][i], m6 = x[6][i];
		uint32_t temp0;
		m6 = ~m6;
		m0 ^= (~m4) & cc->x;
		temp0 = cc->x ^ (m0 & m2);
		m0 ^= m4 & m6;
		m6 ^= (~m2) & m4;
		m2 ^= m0 & m4;
		m4 ^= m0 & (~m6);
		m0 ^= m2 | m6;
		m6 ^= m2 & m4;
		m2 ^= temp0 & m0;
		m4 ^= temp0;

		uint32_t m1 = x[1][i], m3 = x[3][i], m5 = x[5][i], m7 = x[7][i];
		m7 = ~m7;
		m1 ^= (~m5) & cc->y;
		temp0 = cc->y ^ (m1 & m3);
		m1 ^= m5 & m7;
		m7 ^= (~m3) & m5;
		m3 ^= m1 & m5;
		m5 ^= m1 & (~m7);
		m1 ^= m3 | m7;
		m7 ^= m3 & m5;
		m3 ^= temp0 & m1;
		m5 ^= temp0;

		m1 ^= m2; m3 ^= m4; m5 ^= m0 ^ m6; m7 ^= m0;
		m0 ^= m3; m2 ^= m5; m4 ^= m1 ^ m7; m6 ^= m1;

		x[0][i] = m0; x[2][i] = m2; x[4][i] = m4; x[6][i] = m6;
		x[1][i] = m1; x[3][i] = m3; x[5][i] = m5; x[7][i] = m7;
	}
}

__device__ void adam_jh_E8(uint32_t x[8][4])
{
	for (int r = 0; r < 42; r += 7) {
		adam_jh_sbox_mds(x, r + 0);
		for (int j = 1; j < 8; j += 2) { ADAM_JH_SWAP1(x[j][0]); ADAM_JH_SWAP1(x[j][1]); ADAM_JH_SWAP1(x[j][2]); ADAM_JH_SWAP1(x[j][3]); }
		adam_jh_sbox_mds(x, r + 1);
		for (int j = 1; j < 8; j += 2) { ADAM_JH_SWAP2(x[j][0]); ADAM_JH_SWAP2(x[j][1]); ADAM_JH_SWAP2(x[j][2]); ADAM_JH_SWAP2(x[j][3]); }
		adam_jh_sbox_mds(x, r + 2);
		for (int j = 1; j < 8; j += 2) { ADAM_JH_SWAP4(x[j][0]); ADAM_JH_SWAP4(x[j][1]); ADAM_JH_SWAP4(x[j][2]); ADAM_JH_SWAP4(x[j][3]); }
		adam_jh_sbox_mds(x, r + 3);
		for (int j = 1; j < 8; j += 2) { ADAM_JH_SWAP8(x[j][0]); ADAM_JH_SWAP8(x[j][1]); ADAM_JH_SWAP8(x[j][2]); ADAM_JH_SWAP8(x[j][3]); }
		adam_jh_sbox_mds(x, r + 4);
		for (int j = 1; j < 8; j += 2) { ADAM_JH_SWAP16(x[j][0]); ADAM_JH_SWAP16(x[j][1]); ADAM_JH_SWAP16(x[j][2]); ADAM_JH_SWAP16(x[j][3]); }
		adam_jh_sbox_mds(x, r + 5);
		for (int j = 1; j < 8; j += 2) {
			uint32_t t = x[j][0]; x[j][0] = x[j][1]; x[j][1] = t;
			t = x[j][2]; x[j][2] = x[j][3]; x[j][3] = t;
		}
		adam_jh_sbox_mds(x, r + 6);
		for (int j = 1; j < 8; j += 2) {
			uint32_t t = x[j][0]; x[j][0] = x[j][2]; x[j][2] = t;
			t = x[j][1]; x[j][1] = x[j][3]; x[j][3] = t;
		}
	}
}

#undef ADAM_JH_SWAP1
#undef ADAM_JH_SWAP2
#undef ADAM_JH_SWAP4
#undef ADAM_JH_SWAP8
#undef ADAM_JH_SWAP16

__device__ __forceinline__ void adam_jh512_init(uint32_t x[8][4])
{
	const uint32_t iv[8][4] = {
		{ 0x964bd16fU, 0x17aa003eU, 0x052e6a63U, 0x43d5157aU },
		{ 0x8d5e228aU, 0x0bef970cU, 0x591234e9U, 0x61c3b3f2U },
		{ 0xc1a01d89U, 0x1e806f53U, 0x6b05a92aU, 0x806d2beaU },
		{ 0xdbcc8e58U, 0xa6ba7520U, 0x763a0fa9U, 0xf73bf8baU },
		{ 0x05e66901U, 0x694ae341U, 0x8e8ab546U, 0x5ae66f2eU },
		{ 0xd0a74710U, 0x243c84c1U, 0xb1716e3bU, 0x99c15a2dU },
		{ 0xecf657cfU, 0x56f8b19dU, 0x7c8806a7U, 0x56b11657U },
		{ 0xdffcc2e3U, 0xfb1785e6U, 0x78465a54U, 0x4bdd8cccU }
	};
	#pragma unroll
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 4; j++)
			x[i][j] = iv[i][j];
}

__global__ void adam_gpu_jh512_70(uint32_t threads, uint32_t start_nonce, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t w[18];
	adam_load_le32_70(w, start_nonce + tid);

	uint32_t x[8][4];
	adam_jh512_init(x);

	/* Block 0: bytes 0..63 */
	#pragma unroll
	for (int i = 0; i < 16; i++)
		x[i / 4][i & 3] ^= w[i];
	adam_jh_E8(x);
	#pragma unroll
	for (int i = 0; i < 16; i++)
		x[(i + 16) / 4][(i + 16) & 3] ^= w[i];

	/* Block 1: bytes 64..69 + 0x80 at byte 70 */
	uint32_t b1_0 = w[16];
	uint32_t b1_1 = w[17] | 0x00800000U;
	x[0][0] ^= b1_0; x[0][1] ^= b1_1;
	adam_jh_E8(x);
	x[4][0] ^= b1_0; x[4][1] ^= b1_1;

	/* Block 2: zeros + BE128(560 = 0x0230) in last byte pair -> LE32 word 15 = 0x30020000 */
	x[3][3] ^= 0x30020000U;
	adam_jh_E8(x);
	x[7][3] ^= 0x30020000U;

	uint32_t out32[8];
	#pragma unroll
	for (int i = 0; i < 4; i++) {
		out32[i]     = x[4][i];
		out32[i + 4] = x[5][i];
	}
	adam_store_le32_32(g_hash, tid, out32);
}

__global__ void adam_gpu_jh512_32(uint32_t threads, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t w[8];
	adam_load_le32_32(w, g_hash, tid);

	uint32_t x[8][4];
	adam_jh512_init(x);

	/* Block 0: bytes 0..31 + 0x80 at byte 32 */
	#pragma unroll
	for (int i = 0; i < 8; i++)
		x[i / 4][i & 3] ^= w[i];
	x[2][0] ^= 0x00000080U;
	adam_jh_E8(x);
	#pragma unroll
	for (int i = 0; i < 8; i++)
		x[(i + 16) / 4][(i + 16) & 3] ^= w[i];
	x[6][0] ^= 0x00000080U;

	/* Block 1: zeros + BE128(256 = 0x0100) -> LE32 word 15 = 0x00010000 */
	x[3][3] ^= 0x00010000U;
	adam_jh_E8(x);
	x[7][3] ^= 0x00010000U;

	uint32_t out32[8];
	#pragma unroll
	for (int i = 0; i < 4; i++) {
		out32[i]     = x[4][i];
		out32[i + 4] = x[5][i];
	}
	adam_store_le32_32(g_hash, tid, out32);
}

#endif
