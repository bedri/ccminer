#ifndef CUDA_ADAM_COMMON_CUH
#define CUDA_ADAM_COMMON_CUH

#include <stdint.h>
#include "cuda_helper.h"
#include "cuda_vectors.h"

/* KRISTA ADAM constants in GPU constant memory */
static __constant__ __align__(16) uint8_t  c_adam_prefix[68];
static __constant__ __align__(16) uint32_t c_adam_target[8];

__device__ __forceinline__ uint32_t adam_rotr32(uint32_t x, uint32_t n)
{
#if __CUDA_ARCH__ >= 350
	return __funnelshift_r(x, x, n);
#else
	return (x >> n) | (x << (32 - n));
#endif
}

__device__ __forceinline__ uint32_t adam_rotl32(uint32_t x, uint32_t n)
{
#if __CUDA_ARCH__ >= 350
	return __funnelshift_l(x, x, n);
#else
	return (x << n) | (x >> (32 - n));
#endif
}

__device__ __forceinline__ uint64_t adam_rotl64(uint64_t x, uint32_t n)
{
	return (n == 0) ? x : ((x << n) | (x >> (64 - n)));
}

__device__ __forceinline__ uint64_t adam_rotr64(uint64_t x, uint32_t n)
{
	return (n == 0) ? x : ((x >> n) | (x << (64 - n)));
}

__device__ __forceinline__ uint32_t adam_bswap32(uint32_t x)
{
#ifdef __CUDA_ARCH__
	return __byte_perm(x, 0, 0x0123);
#else
	return cuda_swab32(x);
#endif
}

__device__ __forceinline__ uint64_t adam_bswap64(uint64_t x)
{
	return cuda_swab64(x);
}

/* Load 70-byte V11 input (66B constant prefix + 4B LE nonce) as 18 LE32 words (trailing 2 bytes 0) */
__device__ __forceinline__ void adam_load_le32_70(uint32_t *w, uint32_t nonce)
{
	const uint32_t *p = (const uint32_t *)c_adam_prefix;
	#pragma unroll
	for (int i = 0; i < 16; i++)
		w[i] = p[i];
	uint32_t tail = p[16] & 0x0000FFFFu;
	w[16] = tail | ((nonce & 0xFFFFu) << 16);
	w[17] = nonce >> 16;
}

/* Load 70-byte V11 input (66B constant prefix + 4B LE nonce) as 9 LE64 words (trailing 2 bytes 0) */
__device__ __forceinline__ void adam_load_le64_70(uint64_t *w, uint32_t nonce)
{
	const uint32_t *p = (const uint32_t *)c_adam_prefix;
	#pragma unroll
	for (int i = 0; i < 8; i++)
		w[i] = (uint64_t)p[2 * i] | ((uint64_t)p[2 * i + 1] << 32);
	uint32_t tail = p[16] & 0x0000FFFFu;
	w[8] = (uint64_t)tail | ((uint64_t)nonce << 16);
}

/* Load 32-byte intermediate hash from d_hash[tid * 8 .. tid * 8 + 7] as 8 LE32 words */
__device__ __forceinline__ void adam_load_le32_32(uint32_t *w, const uint32_t *g_hash, uint32_t tid)
{
	const uint32_t *src = g_hash + ((size_t)tid * 8);
	#pragma unroll
	for (int i = 0; i < 8; i++)
		w[i] = src[i];
}

/* Load 32-byte intermediate hash from d_hash as 4 LE64 words */
__device__ __forceinline__ void adam_load_le64_32(uint64_t *w, const uint64_t *g_hash, uint32_t tid)
{
	const uint64_t *src = g_hash + ((size_t)tid * 4);
	#pragma unroll
	for (int i = 0; i < 4; i++)
		w[i] = src[i];
}

__device__ __forceinline__ void adam_store_le32_32(uint32_t *g_hash, uint32_t tid, const uint32_t *h)
{
	uint32_t *dst = g_hash + ((size_t)tid * 8);
	#pragma unroll
	for (int i = 0; i < 8; i++)
		dst[i] = h[i];
}

__device__ __forceinline__ void adam_store_le64_32(uint64_t *g_hash, uint32_t tid, const uint64_t *h)
{
	uint64_t *dst = g_hash + ((size_t)tid * 4);
	#pragma unroll
	for (int i = 0; i < 4; i++)
		dst[i] = h[i];
}

/* Compute (h * factor) mod 2^256 in-place (little-endian 256-bit arithmetic, matches Core ArithUint256) */
__global__ void adam_scale_hash_kernel(uint32_t threads, uint32_t *g_hash, uint32_t factor)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t *words = g_hash + ((size_t)tid * 8);
	uint64_t carry = 0;
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		uint64_t prod = (uint64_t)words[i] * (uint64_t)factor + carry;
		words[i] = (uint32_t)prod;
		carry = prod >> 32;
	}
}

/* Compare 256-bit LE hash against c_adam_target[8] and record smallest matching thread index */
__global__ void adam_check_target_kernel(uint32_t threads, const uint32_t *g_hash, uint32_t *d_resNonce)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	const uint32_t *h = g_hash + ((size_t)tid * 8);
	bool hit = true;
	#pragma unroll
	for (int i = 7; i >= 0; i--) {
		uint32_t hi = h[i];
		uint32_t ti = c_adam_target[i];
		if (hi < ti) {
			hit = true;
			break;
		}
		if (hi > ti) {
			hit = false;
			break;
		}
	}
	if (hit)
		atomicMin(d_resNonce, tid);
}

#endif
