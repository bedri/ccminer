#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <algorithm>
#include <openssl/sha.h>

#include "miner.h"

// Include all required CPU SPH hash headers in extern "C"
extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_skein.h"
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_haval.h"
}

static const unsigned int HASHX11KV_MIN_NUMBER_ITERATIONS = 2;
static const unsigned int HASHX11KV_MAX_NUMBER_ITERATIONS = 6;
static const unsigned int HASHX11KV_NUMBER_ALGOS = 11;

static void HashX11KV(const void* pbegin, size_t len, void* pBuffer) {
    uint8_t hash[64];
    static const unsigned char pblank[1] = {0};

    sph_blake512_context    ctx_blake;
    sph_bmw512_context      ctx_bmw;
    sph_groestl512_context  ctx_groestl;
    sph_jh512_context       ctx_jh;
    sph_keccak512_context   ctx_keccak;
    sph_skein512_context    ctx_skein;
    sph_luffa512_context    ctx_luffa;
    sph_cubehash512_context ctx_cubehash;
    sph_shavite512_context  ctx_shavite;
    sph_simd512_context     ctx_simd;
    sph_echo512_context     ctx_echo;

    // Iteration 0
    sph_blake512_init(&ctx_blake);
    sph_blake512(&ctx_blake, (len == 0 ? pblank : pbegin), len);
    sph_blake512_close(&ctx_blake, hash);

    int n = HASHX11KV_MIN_NUMBER_ITERATIONS + (hash[63] % (HASHX11KV_MAX_NUMBER_ITERATIONS - HASHX11KV_MIN_NUMBER_ITERATIONS + 1));

    for (int i = 1; i < n; i++) {
        switch (hash[i % 64] % HASHX11KV_NUMBER_ALGOS) {
        case 0:
            sph_blake512_init(&ctx_blake);
            sph_blake512(&ctx_blake, hash, 64);
            sph_blake512_close(&ctx_blake, hash);
            break;
        case 1:
            sph_bmw512_init(&ctx_bmw);
            sph_bmw512(&ctx_bmw, hash, 64);
            sph_bmw512_close(&ctx_bmw, hash);
            break;
        case 2:
            sph_groestl512_init(&ctx_groestl);
            sph_groestl512(&ctx_groestl, hash, 64);
            sph_groestl512_close(&ctx_groestl, hash);
            break;
        case 3:
            sph_skein512_init(&ctx_skein);
            sph_skein512(&ctx_skein, hash, 64);
            sph_skein512_close(&ctx_skein, hash);
            break;
        case 4:
            sph_jh512_init(&ctx_jh);
            sph_jh512(&ctx_jh, hash, 64);
            sph_jh512_close(&ctx_jh, hash);
            break;
        case 5:
            sph_keccak512_init(&ctx_keccak);
            sph_keccak512(&ctx_keccak, hash, 64);
            sph_keccak512_close(&ctx_keccak, hash);
            break;
        case 6:
            sph_luffa512_init(&ctx_luffa);
            sph_luffa512(&ctx_luffa, hash, 64);
            sph_luffa512_close(&ctx_luffa, hash);
            break;
        case 7:
            sph_cubehash512_init(&ctx_cubehash);
            sph_cubehash512(&ctx_cubehash, hash, 64);
            sph_cubehash512_close(&ctx_cubehash, hash);
            break;
        case 8:
            sph_shavite512_init(&ctx_shavite);
            sph_shavite512(&ctx_shavite, hash, 64);
            sph_shavite512_close(&ctx_shavite, hash);
            break;
        case 9:
            sph_simd512_init(&ctx_simd);
            sph_simd512(&ctx_simd, hash, 64);
            sph_simd512_close(&ctx_simd, hash);
            break;
        case 10:
            sph_echo512_init(&ctx_echo);
            sph_echo512(&ctx_echo, hash, 64);
            sph_echo512_close(&ctx_echo, hash);
            break;
        }
    }

    memcpy(pBuffer, hash, 32);
}

static void HashX11KVS(const uint8_t* pbegin, size_t len, void* pBuffer, unsigned int level = 7) {
    if (len < 80) {
        HashX11KV(pbegin, len, pBuffer);
        return;
    }
    
    uint8_t hash[32];
    HashX11KV(pbegin, len, hash);
    if (level == 1) {
        memcpy(pBuffer, hash, 32);
        return;
    }
    
    uint32_t nonce = le32dec(pbegin + 76);
    
    uint8_t nextheader1[80];
    uint8_t nextheader2[80];
    
    uint32_t nextnonce1 = nonce + (le32dec(hash + 24) % 0xFFFF);
    uint32_t nextnonce2 = nonce + (le32dec(hash + 28) % 0xFFFF);
    
    memcpy(nextheader1, pbegin, 76);
    le32enc(nextheader1 + 76, nextnonce1);
    
    memcpy(nextheader2, pbegin, 76);
    le32enc(nextheader2 + 76, nextnonce2);
    
    uint8_t hash1[32];
    uint8_t hash2[32];
    
    HashX11KVS(nextheader1, 80, hash1, level - 1);
    HashX11KVS(nextheader2, 80, hash2, level - 1);
    
    uint8_t concat[96];
    memcpy(concat, hash, 32);
    memcpy(concat + 32, hash1, 32);
    memcpy(concat + 64, hash2, 32);
    
    uint8_t sha_inner[32];
    SHA256(concat, 96, sha_inner);
    SHA256(sha_inner, 32, (uint8_t*)pBuffer);
}

extern "C" void cpu_adam_puzzle_hash(int algoIndex, const uint8_t* pbegin, size_t len, void* pBuffer) {
    uint32_t hash[16];
    switch (algoIndex) {
        case 0: {
            sph_blake512_context ctx;
            sph_blake512_init(&ctx);
            sph_blake512(&ctx, pbegin, len);
            sph_blake512_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 1: {
            sph_bmw512_context ctx;
            sph_bmw512_init(&ctx);
            sph_bmw512(&ctx, pbegin, len);
            sph_bmw512_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 2: {
            sph_groestl512_context ctx;
            sph_groestl512_init(&ctx);
            sph_groestl512(&ctx, pbegin, len);
            sph_groestl512_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 3: {
            sph_jh512_context ctx;
            sph_jh512_init(&ctx);
            sph_jh512(&ctx, pbegin, len);
            sph_jh512_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 4: {
            sph_keccak512_context ctx;
            sph_keccak512_init(&ctx);
            sph_keccak512(&ctx, pbegin, len);
            sph_keccak512_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 5: {
            sph_skein512_context ctx;
            sph_skein512_init(&ctx);
            sph_skein512(&ctx, pbegin, len);
            sph_skein512_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 6: {
            sph_luffa512_context ctx;
            sph_luffa512_init(&ctx);
            sph_luffa512(&ctx, pbegin, len);
            sph_luffa512_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 7: {
            sph_cubehash512_context ctx;
            sph_cubehash512_init(&ctx);
            sph_cubehash512(&ctx, pbegin, len);
            sph_cubehash512_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 8: {
            sph_shavite512_context ctx;
            sph_shavite512_init(&ctx);
            sph_shavite512(&ctx, pbegin, len);
            sph_shavite512_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 9: {
            sph_simd512_context ctx;
            sph_simd512_init(&ctx);
            sph_simd512(&ctx, pbegin, len);
            sph_simd512_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 10: {
            sph_echo512_context ctx;
            sph_echo512_init(&ctx);
            sph_echo512(&ctx, pbegin, len);
            sph_echo512_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 11: {
            HashX11KVS(pbegin, len, pBuffer);
            break;
        }
        case 12: {
            uint8_t sha_inner[32];
            SHA256(pbegin, len, sha_inner);
            SHA256(sha_inner, 32, (uint8_t*)pBuffer);
            break;
        }
        case 13: {
            sph_hamsi512_context ctx;
            sph_hamsi512_init(&ctx);
            sph_hamsi512(&ctx, pbegin, len);
            sph_hamsi512_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 14: {
            sph_fugue512_context ctx;
            sph_fugue512_init(&ctx);
            sph_fugue512(&ctx, pbegin, len);
            sph_fugue512_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 15: {
            sph_shabal512_context ctx;
            sph_shabal512_init(&ctx);
            sph_shabal512(&ctx, pbegin, len);
            sph_shabal512_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 16: {
            sph_whirlpool_context ctx;
            sph_whirlpool_init(&ctx);
            sph_whirlpool(&ctx, pbegin, len);
            sph_whirlpool_close(&ctx, hash);
            memcpy(pBuffer, hash, 32);
            break;
        }
        case 17: {
            sph_haval256_5_context ctx;
            sph_haval256_5_init(&ctx);
            sph_haval256_5(&ctx, pbegin, len);
            sph_haval256_5_close(&ctx, (uint32_t*)pBuffer);
            break;
        }
        default: {
            uint8_t sha_inner[32];
            SHA256(pbegin, len, sha_inner);
            SHA256(sha_inner, 32, (uint8_t*)pBuffer);
            break;
        }
    }
}

static void cpu_scale_hash(uint32_t *hash, int i_factor) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t val = (uint64_t)hash[i] * i_factor + carry;
        hash[i] = (uint32_t)val;
        carry = val >> 32;
    }
}

extern "C" uint32_t run_cpu_fallback_mine(int thr_id, const struct work *work, const uint8_t *input_template, uint32_t start_nonce, uint32_t throughput, const uint32_t *target, uint32_t *found_nonce) {
    const int w_version = work->adam_version ? work->adam_version : 11;
    const int w_algo_single = work->adam_algo_single;
    const int w_algo1 = work->adam_algo1;
    const int w_algo2 = work->adam_algo2;
    const int w_algo3 = work->adam_algo3;
    const int w_factor = work->adam_factor > 0 ? work->adam_factor : 1;

    uint32_t scanned = 0;
    #pragma omp parallel for reduction(+:scanned)
    for (uint32_t i = 0; i < throughput; i++) {
        if (*found_nonce != UINT32_MAX || work_restart[thr_id].restart) continue;
        scanned++;

        uint32_t nonce = start_nonce + i;
        uint8_t input[70];
        memcpy(input, input_template, 66);
        le32enc(input + 66, nonce);

        uint32_t puzzleHash[8];
        if (w_version == 12) {
            uint32_t hash3[8];
            cpu_adam_puzzle_hash(w_algo3, input, 70, hash3);
            cpu_scale_hash(hash3, w_factor);
            
            uint32_t hash2[8];
            cpu_adam_puzzle_hash(w_algo2, (uint8_t*)hash3, 32, hash2);
            cpu_scale_hash(hash2, w_factor);

            cpu_adam_puzzle_hash(w_algo1, (uint8_t*)hash2, 32, puzzleHash);
        } else {
            cpu_adam_puzzle_hash(w_algo_single, input, 70, puzzleHash);
        }

        if (fulltest(puzzleHash, target)) {
            #pragma omp critical
            {
                if (*found_nonce == UINT32_MAX || nonce < *found_nonce) {
                    *found_nonce = nonce;
                }
            }
        }
    }
    return scanned;
}
