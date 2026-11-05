#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <algorithm>
#include "miner.h"
#include "cuda_helper.h"
#include "x11/cuda_x11.h"

// External declarations of global variables set by ccminer.cpp
extern int adam_version;
extern int adam_algo_single;
extern int adam_algo1;
extern int adam_algo2;
extern int adam_algo3;
extern int adam_factor;

// Thread-local GPU hash pointer array
static uint32_t *d_hash[MAX_GPUS] = { 0 };

// GPU kernel declarations for the 18 algorithms
// 0: Blake
extern void quark_blake512_cpu_init(int thr_id, uint32_t threads);
extern void quark_blake512_cpu_setBlock_80(int thr_id, uint32_t *pdata);
extern void quark_blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// 1: BMW
extern void quark_bmw512_cpu_init(int thr_id, uint32_t threads);
extern void quark_bmw512_cpu_setBlock_80(void *pdata);
extern void quark_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order);
extern void quark_bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// 2: Groestl
extern void quark_groestl512_cpu_init(int thr_id, uint32_t threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// 3: JH
extern void quark_jh512_cpu_init(int thr_id, uint32_t threads);
extern void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// 4: Keccak
extern void quark_keccak512_cpu_init(int thr_id, uint32_t threads);
extern void quark_keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// 5: Skein
extern void quark_skein512_cpu_init(int thr_id, uint32_t threads);
extern void skein512_cpu_setBlock_80(void *pdata);
extern void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int swap);
extern void quark_skein512_cpu_hash_64(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// 6: Luffa
extern void x11_luffa512_cpu_init(int thr_id, uint32_t threads);
extern void x11_luffa512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// 7: Cubehash
extern void x11_cubehash512_cpu_init(int thr_id, uint32_t threads);
extern void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// 8: Shavite
extern void x11_shavite512_cpu_init(int thr_id, uint32_t threads);
extern void x11_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// 9: Simd
extern int  x11_simd512_cpu_init(int thr_id, uint32_t threads);
extern void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// 10: Echo
extern void x11_echo512_cpu_init(int thr_id, uint32_t threads);
extern void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// 13: Hamsi
extern void x13_hamsi512_cpu_init(int thr_id, uint32_t threads);
extern void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// 14: Fugue
extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// 15: Shabal
extern void x14_shabal512_cpu_init(int thr_id, uint32_t threads);
extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// 16: Whirlpool
extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int flag);
extern void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x15_whirlpool_cpu_free(int thr_id);

// 17: Haval
extern void x17_haval256_cpu_init(int thr_id, uint32_t threads);
extern void x17_haval256_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, const int outlen);

// CPU fallback and puzzle hash functions
extern "C" uint32_t run_cpu_fallback_mine(int thr_id, const uint8_t *input_template, uint32_t start_nonce, uint32_t throughput, const uint32_t *target, uint32_t *found_nonce);
extern "C" void cpu_adam_puzzle_hash(int algoIndex, const uint8_t* pbegin, size_t len, void* pBuffer);

// Helper for CPU scaling
static void cpu_scale_hash(uint32_t *hash, int i_factor) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t val = (uint64_t)hash[i] * i_factor + carry;
        hash[i] = (uint32_t)val;
        carry = val >> 32;
    }
}

// CUDA Kernel for modular arithmetic scaling (256-bit mult by i_factor)
__global__ void adam_scale_hash(uint32_t *d_hash, int i_factor, uint32_t threads)
{
    uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread < threads) {
        uint32_t *hash_ptr = &d_hash[thread * 16];
        uint64_t carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t val = (uint64_t)hash_ptr[i] * i_factor + carry;
            hash_ptr[i] = (uint32_t)val;
            carry = val >> 32;
        }
        for (int i = 8; i < 16; i++) {
            hash_ptr[i] = 0;
        }
    }
}

// Helper to launch intermediate hash steps
static void launch_intermediate_gpu_hash(int algoIndex, int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int order)
{
    switch (algoIndex) {
        case 0:  quark_blake512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 1:  quark_bmw512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 2:  quark_groestl512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 3:  quark_jh512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 4:  quark_keccak512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 5:  quark_skein512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 6:  x11_luffa512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 7:  x11_cubehash512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 8:  x11_shavite512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 9:  x11_simd512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 10: x11_echo512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 13: x13_hamsi512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 14: x13_fugue512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 15: x14_shabal512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 16: x15_whirlpool_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash, order); break;
        case 17: x17_haval256_cpu_hash_64(thr_id, threads, startNonce, d_hash, 256); break;
        default: break;
    }
}

// Static initialization flags
static bool init[MAX_GPUS] = { 0 };

extern "C" void free_adam(int thr_id)
{
    init[thr_id] = false;
    cudaSetDevice(device_map[thr_id]);
    x15_whirlpool_cpu_free(thr_id);
    cudaDeviceSynchronize();
}

extern "C" int scanhash_adam(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];

    uint32_t throughput = std::min<uint32_t>(32768U, max_nonce > first_nonce ? (max_nonce - first_nonce) : 0U);
    if (throughput == 0) {
        return 0;
    }

    // ADAM consensus hashes a 70-byte input (66B prefix + 4B nonce) and 32-byte truncated
    // intermediate hashes with 256-bit scalar multiplication. Use the consensus-exact
    // OpenMP 70->32->32 pipeline in bounded batches with work_restart checks.
    uint8_t input_template[70];
    uint32_t le_data[20];
    for (int k = 0; k < 20; k++) {
        le32enc(le_data + k, pdata[k]);
    }
    memcpy(input_template, le_data, 66);

    uint32_t found_nonce = UINT32_MAX;
    uint32_t scanned = run_cpu_fallback_mine(thr_id, input_template, first_nonce, throughput, ptarget, &found_nonce);

    *hashes_done += scanned;
    pdata[19] = first_nonce + scanned;

    if (found_nonce != UINT32_MAX) {
        work->nonces[0] = found_nonce;
        return 1;
    }
    return 0;
}
