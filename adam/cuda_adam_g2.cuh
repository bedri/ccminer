#ifndef CUDA_ADAM_G2_CUH
#define CUDA_ADAM_G2_CUH

#include "cuda_adam_common.cuh"

/* =========================================================================
 * Group 2:
 *   4: Keccak-512
 *   5: Skein-512
 *   6: Luffa-512
 *   7: CubeHash-512
 * ========================================================================= */

/* ------------------- 4. Keccak-512 ------------------- */

static __constant__ uint64_t c_adam_keccak_rc[24] = {
	0x0000000000000001ULL, 0x0000000000008082ULL,
	0x800000000000808AULL, 0x8000000080008000ULL,
	0x000000000000808BULL, 0x0000000080000001ULL,
	0x8000000080008081ULL, 0x8000000000008009ULL,
	0x000000000000008AULL, 0x0000000000000088ULL,
	0x0000000080008009ULL, 0x000000008000000AULL,
	0x000000008000808BULL, 0x800000000000008BULL,
	0x8000000000008089ULL, 0x8000000000008003ULL,
	0x8000000000008002ULL, 0x8000000000000080ULL,
	0x000000000000800AULL, 0x800000008000000AULL,
	0x8000000080008081ULL, 0x8000000000008080ULL,
	0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ void adam_keccak_f1600(uint64_t s[25])
{
	uint64_t t[5], u[5], v, w;
	for (int i = 0; i < 24; i++) {
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		u[0] = t[4] ^ adam_rotl64(t[1], 1);
		u[1] = t[0] ^ adam_rotl64(t[2], 1);
		u[2] = t[1] ^ adam_rotl64(t[3], 1);
		u[3] = t[2] ^ adam_rotl64(t[4], 1);
		u[4] = t[3] ^ adam_rotl64(t[0], 1);

		#pragma unroll
		for (int j = 0; j < 25; j += 5) {
			s[j + 0] ^= u[0]; s[j + 1] ^= u[1]; s[j + 2] ^= u[2];
			s[j + 3] ^= u[3]; s[j + 4] ^= u[4];
		}

		v = s[1];
		s[ 1] = adam_rotl64(s[ 6], 44);
		s[ 6] = adam_rotl64(s[ 9], 20);
		s[ 9] = adam_rotl64(s[22], 61);
		s[22] = adam_rotl64(s[14], 39);
		s[14] = adam_rotl64(s[20], 18);
		s[20] = adam_rotl64(s[ 2], 62);
		s[ 2] = adam_rotl64(s[12], 43);
		s[12] = adam_rotl64(s[13], 25);
		s[13] = adam_rotl64(s[19],  8);
		s[19] = adam_rotl64(s[23], 56);
		s[23] = adam_rotl64(s[15], 41);
		s[15] = adam_rotl64(s[ 4], 27);
		s[ 4] = adam_rotl64(s[24], 14);
		s[24] = adam_rotl64(s[21],  2);
		s[21] = adam_rotl64(s[ 8], 55);
		s[ 8] = adam_rotl64(s[16], 45);
		s[16] = adam_rotl64(s[ 5], 36);
		s[ 5] = adam_rotl64(s[ 3], 28);
		s[ 3] = adam_rotl64(s[18], 21);
		s[18] = adam_rotl64(s[17], 15);
		s[17] = adam_rotl64(s[11], 10);
		s[11] = adam_rotl64(s[ 7],  6);
		s[ 7] = adam_rotl64(s[10],  3);
		s[10] = adam_rotl64(v,     1);

		#pragma unroll
		for (int j = 0; j < 25; j += 5) {
			v = s[j + 0]; w = s[j + 1];
			s[j + 0] ^= (~w) & s[j + 2];
			s[j + 1] ^= (~s[j + 2]) & s[j + 3];
			s[j + 2] ^= (~s[j + 3]) & s[j + 4];
			s[j + 3] ^= (~s[j + 4]) & v;
			s[j + 4] ^= (~v) & w;
		}
		s[0] ^= c_adam_keccak_rc[i];
	}
}

__global__ void adam_gpu_keccak512_70(uint32_t threads, uint32_t start_nonce, uint64_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint64_t w[9];
	adam_load_le64_70(w, start_nonce + tid);

	uint64_t s[25];
	#pragma unroll
	for (int i = 0; i < 8; i++)
		s[i] = w[i];
	s[8] = w[8] | (0x01ULL << 48) | 0x8000000000000000ULL;
	#pragma unroll
	for (int i = 9; i < 25; i++)
		s[i] = 0;

	adam_keccak_f1600(s);
	adam_store_le64_32(g_hash, tid, s);
}

__global__ void adam_gpu_keccak512_32(uint32_t threads, uint64_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint64_t w[4];
	adam_load_le64_32(w, g_hash, tid);

	uint64_t s[25];
	#pragma unroll
	for (int i = 0; i < 4; i++)
		s[i] = w[i];
	s[4] = 0x01ULL;
	s[5] = 0; s[6] = 0; s[7] = 0;
	s[8] = 0x8000000000000000ULL;
	#pragma unroll
	for (int i = 9; i < 25; i++)
		s[i] = 0;

	adam_keccak_f1600(s);
	adam_store_le64_32(g_hash, tid, s);
}

/* ------------------- 5. Skein-512 ------------------- */

static __constant__ uint64_t c_adam_skein_iv[8] = {
	0x4903ADFF749C51CEULL, 0x0D95DE399746DF03ULL,
	0x8FD1934127C79BCEULL, 0x9A255629FF352CB1ULL,
	0x5DB62599DF6CA7B0ULL, 0xEABE394CA9D5C3F4ULL,
	0x991112C71A75B523ULL, 0xAE18A40B660FCC33ULL
};

#define ADAM_SKEIN_MIX(x0, x1, rc) do { \
	(x0) += (x1); \
	(x1) = adam_rotl64((x1), (rc)) ^ (x0); \
} while (0)

#define ADAM_SKEIN_MIX8(w0, w1, w2, w3, w4, w5, w6, w7, r0, r1, r2, r3) do { \
	ADAM_SKEIN_MIX(w0, w1, r0); ADAM_SKEIN_MIX(w2, w3, r1); \
	ADAM_SKEIN_MIX(w4, w5, r2); ADAM_SKEIN_MIX(w6, w7, r3); \
} while (0)

__device__ void adam_skein_ubi(uint64_t h[8], const uint64_t m[8], uint64_t t0, uint64_t t1)
{
	uint64_t k[18];
	#pragma unroll
	for (int i = 0; i < 8; i++)
		k[i] = h[i];
	k[8] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7] ^ 0x1BD11BDAA9FC1A22ULL;
	#pragma unroll
	for (int i = 0; i < 9; i++)
		k[9 + i] = k[i];

	uint64_t t[3] = { t0, t1, t0 ^ t1 };
	uint64_t p0 = m[0], p1 = m[1], p2 = m[2], p3 = m[3];
	uint64_t p4 = m[4], p5 = m[5], p6 = m[6], p7 = m[7];

	for (uint32_t u = 0; u < 9; u++) {
		uint32_t s = u << 1;
		uint32_t ks = s % 9U;
		p0 += k[ks + 0]; p1 += k[ks + 1]; p2 += k[ks + 2]; p3 += k[ks + 3];
		p4 += k[ks + 4]; p5 += k[ks + 5] + t[0]; p6 += k[ks + 6] + t[1]; p7 += k[ks + 7] + (uint64_t)s;
		ADAM_SKEIN_MIX8(p0, p1, p2, p3, p4, p5, p6, p7, 46, 36, 19, 37);
		ADAM_SKEIN_MIX8(p2, p1, p4, p7, p6, p5, p0, p3, 33, 27, 14, 42);
		ADAM_SKEIN_MIX8(p4, p1, p6, p3, p0, p5, p2, p7, 17, 49, 36, 39);
		ADAM_SKEIN_MIX8(p6, p1, p0, p7, p2, p5, p4, p3, 44,  9, 54, 56);

		s += 1;
		ks = s % 9U;
		p0 += k[ks + 0]; p1 += k[ks + 1]; p2 += k[ks + 2]; p3 += k[ks + 3];
		p4 += k[ks + 4]; p5 += k[ks + 5] + t[1]; p6 += k[ks + 6] + t[2]; p7 += k[ks + 7] + (uint64_t)s;
		ADAM_SKEIN_MIX8(p0, p1, p2, p3, p4, p5, p6, p7, 39, 30, 34, 24);
		ADAM_SKEIN_MIX8(p2, p1, p4, p7, p6, p5, p0, p3, 13, 50, 10, 17);
		ADAM_SKEIN_MIX8(p4, p1, p6, p3, p0, p5, p2, p7, 25, 29, 39, 43);
		ADAM_SKEIN_MIX8(p6, p1, p0, p7, p2, p5, p4, p3,  8, 35, 56, 22);

		uint64_t tmp = t[2];
		t[2] = t[1];
		t[1] = t[0];
		t[0] = tmp;
	}

	p0 += k[0]; p1 += k[1]; p2 += k[2]; p3 += k[3];
	p4 += k[4]; p5 += k[5] + t[0]; p6 += k[6] + t[1]; p7 += k[7] + 18ULL;

	h[0] = m[0] ^ p0; h[1] = m[1] ^ p1; h[2] = m[2] ^ p2; h[3] = m[3] ^ p3;
	h[4] = m[4] ^ p4; h[5] = m[5] ^ p5; h[6] = m[6] ^ p6; h[7] = m[7] ^ p7;
}

#undef ADAM_SKEIN_MIX
#undef ADAM_SKEIN_MIX8

__global__ void adam_gpu_skein512_70(uint32_t threads, uint32_t start_nonce, uint64_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint64_t w[9];
	adam_load_le64_70(w, start_nonce + tid);

	uint64_t h[8];
	#pragma unroll
	for (int i = 0; i < 8; i++)
		h[i] = c_adam_skein_iv[i];

	adam_skein_ubi(h, w, 64ULL, 0x7000000000000000ULL); /* First | Msg */

	uint64_t b1[8] = { w[8], 0, 0, 0, 0, 0, 0, 0 };
	adam_skein_ubi(h, b1, 70ULL, 0xB000000000000000ULL); /* Final | Msg */

	uint64_t out_blk[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	adam_skein_ubi(h, out_blk, 8ULL, 0xFF00000000000000ULL); /* First | Final | Out */

	adam_store_le64_32(g_hash, tid, h);
}

__global__ void adam_gpu_skein512_32(uint32_t threads, uint64_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint64_t b0[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	adam_load_le64_32(b0, g_hash, tid);

	uint64_t h[8];
	#pragma unroll
	for (int i = 0; i < 8; i++)
		h[i] = c_adam_skein_iv[i];

	adam_skein_ubi(h, b0, 32ULL, 0xF000000000000000ULL); /* First | Final | Msg */

	uint64_t out_blk[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	adam_skein_ubi(h, out_blk, 8ULL, 0xFF00000000000000ULL); /* First | Final | Out */

	adam_store_le64_32(g_hash, tid, h);
}

/* ------------------- 6. Luffa-512 ------------------- */

static __constant__ uint32_t c_adam_luffa_iv[5][8] = {
	{ 0x6d251e69U, 0x44b051e0U, 0x4eaa6fb4U, 0xdbf78465U, 0x6e292011U, 0x90152df4U, 0xee058139U, 0xdef610bbU },
	{ 0xc3b44b95U, 0xd9d2f256U, 0x70eee9a0U, 0xde099fa3U, 0x5d9b0557U, 0x8fc944b3U, 0xcf1ccf0eU, 0x746cd581U },
	{ 0xf7efc89dU, 0x5dba5781U, 0x04016ce5U, 0xad659c05U, 0x0306194fU, 0x666d1836U, 0x24aa230aU, 0x8b264ae7U },
	{ 0x858075d5U, 0x36d79cceU, 0xe571f7d7U, 0x204b1f67U, 0x35870c6aU, 0x57e9e923U, 0x14bcb808U, 0x7cde72ceU },
	{ 0x6c68e9beU, 0x5ec41e22U, 0xc825b7c7U, 0xaffb4363U, 0xf5df3999U, 0x0fc688f1U, 0xb07224ccU, 0x03e86ceaU }
};

static __constant__ uint32_t c_adam_luffa_rc[5][2][8] = {
	{
		{ 0x303994a6U, 0xc0e65299U, 0x6cc33a12U, 0xdc56983eU, 0x1e00108fU, 0x7800423dU, 0x8f5b7882U, 0x96e1db12U },
		{ 0xe0337818U, 0x441ba90dU, 0x7f34d442U, 0x9389217fU, 0xe5a8bce6U, 0x5274baf4U, 0x26889ba7U, 0x9a226e9dU }
	}, {
		{ 0xb6de10edU, 0x70f47aaeU, 0x0707a3d4U, 0x1c1e8f51U, 0x707a3d45U, 0xaeb28562U, 0xbaca1589U, 0x40a46f3eU },
		{ 0x01685f3dU, 0x05a17cf4U, 0xbd09cacaU, 0xf4272b28U, 0x144ae5ccU, 0xfaa7ae2bU, 0x2e48f1c1U, 0xb923c704U }
	}, {
		{ 0xfc20d9d2U, 0x34552e25U, 0x7ad8818fU, 0x8438764aU, 0xbb6de032U, 0xedb780c8U, 0xd9847356U, 0xa2c78434U },
		{ 0xe25e72c1U, 0xe623bb72U, 0x5c58a4a4U, 0x1e38e2e7U, 0x78e38b9dU, 0x27586719U, 0x36eda57fU, 0x703aace7U }
	}, {
		{ 0xb213afa5U, 0xc84ebe95U, 0x4e608a22U, 0x56d858feU, 0x343b138fU, 0xd0ec4e3dU, 0x2ceb4882U, 0xb3ad2208U },
		{ 0xe028c9bfU, 0x44756f91U, 0x7e8fce32U, 0x956548beU, 0xfe191be2U, 0x3cb226e5U, 0x5944a28eU, 0xa1c4c355U }
	}, {
		{ 0xf0d2e9e3U, 0xac11d7faU, 0x1bcb66f2U, 0x6f2d9bc9U, 0x78602649U, 0x8edae952U, 0x3b6ba548U, 0xedae9520U },
		{ 0x5090d577U, 0x2d1925abU, 0xb46496acU, 0xd1925ab0U, 0x29131ab6U, 0x0fc053c3U, 0x3f014f0cU, 0xfc053c31U }
	}
};

__device__ __forceinline__ void adam_luffa_m2(uint32_t d[8], const uint32_t s[8])
{
	uint32_t tmp = s[7];
	d[7] = s[6];
	d[6] = s[5];
	d[5] = s[4];
	d[4] = s[3] ^ tmp;
	d[3] = s[2] ^ tmp;
	d[2] = s[1];
	d[1] = s[0] ^ tmp;
	d[0] = tmp;
}

__device__ __forceinline__ void adam_luffa_subcrumb(uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3)
{
	uint32_t tmp = a0;
	a0 |= a1;
	a2 ^= a3;
	a1 = ~a1;
	a0 ^= a3;
	a3 &= tmp;
	a1 ^= a3;
	a3 ^= a2;
	a2 &= a0;
	a0 = ~a0;
	a2 ^= a1;
	a1 |= a3;
	tmp ^= a1;
	a3 ^= a2;
	a2 &= a1;
	a1 ^= a0;
	a0 = tmp;
}

__device__ __forceinline__ void adam_luffa_mixword(uint32_t &u, uint32_t &v)
{
	v ^= u;
	u = adam_rotl32(u,  2) ^ v;
	v = adam_rotl32(v, 14) ^ u;
	u = adam_rotl32(u, 10) ^ v;
	v = adam_rotl32(v,  1);
}

__device__ void adam_luffa_step(uint32_t V[5][8], const uint32_t m_in[8])
{
	uint32_t M[8], a[8], b[8];
	#pragma unroll
	for (int j = 0; j < 8; j++) {
		M[j] = m_in[j];
		a[j] = V[0][j] ^ V[1][j] ^ V[2][j] ^ V[3][j] ^ V[4][j];
	}
	adam_luffa_m2(a, a);
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 8; j++)
			V[i][j] ^= a[j];

	adam_luffa_m2(b, V[0]); for (int j = 0; j < 8; j++) b[j] ^= V[1][j];
	adam_luffa_m2(V[1], V[1]); for (int j = 0; j < 8; j++) V[1][j] ^= V[2][j];
	adam_luffa_m2(V[2], V[2]); for (int j = 0; j < 8; j++) V[2][j] ^= V[3][j];
	adam_luffa_m2(V[3], V[3]); for (int j = 0; j < 8; j++) V[3][j] ^= V[4][j];
	adam_luffa_m2(V[4], V[4]); for (int j = 0; j < 8; j++) V[4][j] ^= V[0][j];

	adam_luffa_m2(V[0], b); for (int j = 0; j < 8; j++) V[0][j] ^= V[4][j];
	adam_luffa_m2(V[4], V[4]); for (int j = 0; j < 8; j++) V[4][j] ^= V[3][j];
	adam_luffa_m2(V[3], V[3]); for (int j = 0; j < 8; j++) V[3][j] ^= V[2][j];
	adam_luffa_m2(V[2], V[2]); for (int j = 0; j < 8; j++) V[2][j] ^= V[1][j];
	adam_luffa_m2(V[1], V[1]); for (int j = 0; j < 8; j++) V[1][j] ^= b[j];

	for (int j = 0; j < 8; j++) V[0][j] ^= M[j];
	adam_luffa_m2(M, M); for (int j = 0; j < 8; j++) V[1][j] ^= M[j];
	adam_luffa_m2(M, M); for (int j = 0; j < 8; j++) V[2][j] ^= M[j];
	adam_luffa_m2(M, M); for (int j = 0; j < 8; j++) V[3][j] ^= M[j];
	adam_luffa_m2(M, M); for (int j = 0; j < 8; j++) V[4][j] ^= M[j];

	/* TWEAK5 */
	for (int i = 1; i < 5; i++)
		for (int j = 4; j < 8; j++)
			V[i][j] = adam_rotl32(V[i][j], i);

	/* P5 */
	for (int i = 0; i < 5; i++) {
		for (int r = 0; r < 8; r++) {
			adam_luffa_subcrumb(V[i][0], V[i][1], V[i][2], V[i][3]);
			adam_luffa_subcrumb(V[i][5], V[i][6], V[i][7], V[i][4]);
			adam_luffa_mixword(V[i][0], V[i][4]);
			adam_luffa_mixword(V[i][1], V[i][5]);
			adam_luffa_mixword(V[i][2], V[i][6]);
			adam_luffa_mixword(V[i][3], V[i][7]);
			V[i][0] ^= c_adam_luffa_rc[i][0][r];
			V[i][4] ^= c_adam_luffa_rc[i][1][r];
		}
	}
}

__global__ void adam_gpu_luffa512_70(uint32_t threads, uint32_t start_nonce, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t w[18];
	adam_load_le32_70(w, start_nonce + tid);

	uint32_t V[5][8];
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 8; j++)
			V[i][j] = c_adam_luffa_iv[i][j];

	uint32_t M[8];
	for (int j = 0; j < 8; j++) M[j] = adam_bswap32(w[j]);
	adam_luffa_step(V, M);
	for (int j = 0; j < 8; j++) M[j] = adam_bswap32(w[8 + j]);
	adam_luffa_step(V, M);

	M[0] = adam_bswap32(w[16]);
	M[1] = adam_bswap32(w[17] | 0x00800000U);
	for (int j = 2; j < 8; j++) M[j] = 0;
	adam_luffa_step(V, M);

	for (int j = 0; j < 8; j++) M[j] = 0;
	adam_luffa_step(V, M);

	uint32_t out32[8];
	#pragma unroll
	for (int j = 0; j < 8; j++)
		out32[j] = adam_bswap32(V[0][j] ^ V[1][j] ^ V[2][j] ^ V[3][j] ^ V[4][j]);
	adam_store_le32_32(g_hash, tid, out32);
}

__global__ void adam_gpu_luffa512_32(uint32_t threads, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t w[8];
	adam_load_le32_32(w, g_hash, tid);

	uint32_t V[5][8];
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 8; j++)
			V[i][j] = c_adam_luffa_iv[i][j];

	uint32_t M[8];
	for (int j = 0; j < 8; j++) M[j] = adam_bswap32(w[j]);
	adam_luffa_step(V, M);

	M[0] = 0x80000000U;
	for (int j = 1; j < 8; j++) M[j] = 0;
	adam_luffa_step(V, M);

	for (int j = 0; j < 8; j++) M[j] = 0;
	adam_luffa_step(V, M);

	uint32_t out32[8];
	#pragma unroll
	for (int j = 0; j < 8; j++)
		out32[j] = adam_bswap32(V[0][j] ^ V[1][j] ^ V[2][j] ^ V[3][j] ^ V[4][j]);
	adam_store_le32_32(g_hash, tid, out32);
}

/* ------------------- 7. CubeHash-512 ------------------- */

static __constant__ uint32_t c_adam_cubehash_iv[32] = {
	0x2AEA2A61U, 0x50F494D4U, 0x2D538B8BU, 0x4167D83EU,
	0x3FEE2313U, 0xC701CF8CU, 0xCC39968EU, 0x50AC5695U,
	0x4D42C787U, 0xA647A8B3U, 0x97CF0BEFU, 0x825B4537U,
	0xEEF864D2U, 0xF22090C4U, 0xD0E5CD33U, 0xA23911AEU,
	0xFCD398D9U, 0x148FE485U, 0x1B017BEFU, 0xB6444532U,
	0x6A536159U, 0x2FF5781CU, 0x91FA7934U, 0x0DBADEA9U,
	0xD65C8A2BU, 0xA5A70E75U, 0xB1C62456U, 0xBC796576U,
	0x1921C8F7U, 0xE7989AF1U, 0x7795D246U, 0xD43E3B44U
};

__device__ void adam_cubehash_rrounds(uint32_t x[32])
{
	for (int r = 0; r < 16; r++) {
		#pragma unroll
		for (int i = 0; i < 16; i++)
			x[16 + i] += x[i];
		#pragma unroll
		for (int i = 0; i < 16; i++)
			x[i] = adam_rotl32(x[i], 7);
		#pragma unroll
		for (int i = 0; i < 8; i++) {
			uint32_t t = x[i]; x[i] = x[8 + i]; x[8 + i] = t;
		}
		#pragma unroll
		for (int i = 0; i < 16; i++)
			x[i] ^= x[16 + i];
		#pragma unroll
		for (int j = 0; j < 4; j++)
			for (int m = 0; m < 2; m++) {
				int a = 16 + j * 4 + m;
				uint32_t t = x[a]; x[a] = x[a + 2]; x[a + 2] = t;
			}
		#pragma unroll
		for (int i = 0; i < 16; i++)
			x[16 + i] += x[i];
		#pragma unroll
		for (int i = 0; i < 16; i++)
			x[i] = adam_rotl32(x[i], 11);
		#pragma unroll
		for (int j = 0; j < 2; j++)
			for (int l = 0; l < 4; l++) {
				int a = j * 8 + l;
				uint32_t t = x[a]; x[a] = x[a + 4]; x[a + 4] = t;
			}
		#pragma unroll
		for (int i = 0; i < 16; i++)
			x[i] ^= x[16 + i];
		#pragma unroll
		for (int i = 16; i < 32; i += 2) {
			uint32_t t = x[i]; x[i] = x[i + 1]; x[i + 1] = t;
		}
	}
}

__global__ void adam_gpu_cubehash512_70(uint32_t threads, uint32_t start_nonce, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t w[18];
	adam_load_le32_70(w, start_nonce + tid);

	uint32_t x[32];
	#pragma unroll
	for (int i = 0; i < 32; i++)
		x[i] = c_adam_cubehash_iv[i];

	for (int i = 0; i < 8; i++) x[i] ^= w[i];
	adam_cubehash_rrounds(x);
	for (int i = 0; i < 8; i++) x[i] ^= w[8 + i];
	adam_cubehash_rrounds(x);
	x[0] ^= w[16];
	x[1] ^= w[17] | 0x00800000U;
	adam_cubehash_rrounds(x);

	x[31] ^= 1U;
	for (int i = 0; i < 10; i++)
		adam_cubehash_rrounds(x);

	adam_store_le32_32(g_hash, tid, x);
}

__global__ void adam_gpu_cubehash512_32(uint32_t threads, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t w[8];
	adam_load_le32_32(w, g_hash, tid);

	uint32_t x[32];
	#pragma unroll
	for (int i = 0; i < 32; i++)
		x[i] = c_adam_cubehash_iv[i];

	for (int i = 0; i < 8; i++) x[i] ^= w[i];
	adam_cubehash_rrounds(x);
	x[0] ^= 0x00000080U;
	adam_cubehash_rrounds(x);

	x[31] ^= 1U;
	for (int i = 0; i < 10; i++)
		adam_cubehash_rrounds(x);

	adam_store_le32_32(g_hash, tid, x);
}

#endif
