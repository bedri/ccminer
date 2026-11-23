#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <algorithm>
#include <atomic>
#include <omp.h>
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

static const uint32_t HASHX11KVS_MAX_LEVEL = 7;
static const uint32_t HASHX11KVS_MIN_LEVEL = 1;
static const uint32_t HASHX11KVS_MAX_DRIFT = 0xFFFF;
static const uint32_t HASHX11KVS_CACHE_CHUNK = 33;
static const uint32_t HASHX11KVS_CACHE_POS_1 = 0xFFFF;
static const uint32_t HASHX11KVS_CACHE_POS_2 = 0xFFFF * 2;
static const uint32_t HASHX11KVS_CACHE_POS_3 = 0xFFFF * 3;
static const uint32_t HASHX11KVS_CACHE_POS_4 = 0xFFFF * 4;
static const uint32_t HASHX11KVS_CACHE_POS_5 = 0xFFFF * 5;
static const uint32_t HASHX11KVS_CACHE_POS_6 = 0xFFFF * 6;
static const uint32_t HASHX11KVS_CACHE_OFF_1 = 0xFFFF * 33;
static const uint32_t HASHX11KVS_CACHE_OFF_2 = HASHX11KVS_CACHE_OFF_1 + 0xFFFF * 33 * 2;
static const uint32_t HASHX11KVS_CACHE_OFF_3 = HASHX11KVS_CACHE_OFF_2 + 0xFFFF * 33 * 3;
static const uint32_t HASHX11KVS_CACHE_OFF_4 = HASHX11KVS_CACHE_OFF_3 + 0xFFFF * 33 * 4;
static const uint32_t HASHX11KVS_CACHE_OFF_5 = HASHX11KVS_CACHE_OFF_4 + 0xFFFF * 33 * 5;
static const uint32_t HASHX11KVS_CACHE_TOTAL = HASHX11KVS_CACHE_OFF_5 + 0xFFFF * 33 * 6;

static void HashX11KVS_recursive(const uint8_t* pbegin, size_t len, void* pBuffer, unsigned int level, uint32_t nonce, uint8_t* cache) {
    if (len < 80) {
        HashX11KV(pbegin, len, pBuffer);
        return;
    }

    if (cache) {
        if (level == HASHX11KVS_MAX_LEVEL - 1 && cache[(nonce % HASHX11KVS_CACHE_POS_1) * HASHX11KVS_CACHE_CHUNK] == 0xFF) {
            memcpy(pBuffer, cache + ((nonce % HASHX11KVS_CACHE_POS_1) * HASHX11KVS_CACHE_CHUNK) + 1, 32);
            return;
        }
        if (level == HASHX11KVS_MAX_LEVEL - 2 && cache[HASHX11KVS_CACHE_OFF_1 + (nonce % HASHX11KVS_CACHE_POS_2) * HASHX11KVS_CACHE_CHUNK] == 0xFF) {
            memcpy(pBuffer, cache + HASHX11KVS_CACHE_OFF_1 + ((nonce % HASHX11KVS_CACHE_POS_2) * HASHX11KVS_CACHE_CHUNK) + 1, 32);
            return;
        }
        if (level == HASHX11KVS_MAX_LEVEL - 3 && cache[HASHX11KVS_CACHE_OFF_2 + (nonce % HASHX11KVS_CACHE_POS_3) * HASHX11KVS_CACHE_CHUNK] == 0xFF) {
            memcpy(pBuffer, cache + HASHX11KVS_CACHE_OFF_2 + ((nonce % HASHX11KVS_CACHE_POS_3) * HASHX11KVS_CACHE_CHUNK) + 1, 32);
            return;
        }
        if (level == HASHX11KVS_MAX_LEVEL - 4 && cache[HASHX11KVS_CACHE_OFF_3 + (nonce % HASHX11KVS_CACHE_POS_4) * HASHX11KVS_CACHE_CHUNK] == 0xFF) {
            memcpy(pBuffer, cache + HASHX11KVS_CACHE_OFF_3 + ((nonce % HASHX11KVS_CACHE_POS_4) * HASHX11KVS_CACHE_CHUNK) + 1, 32);
            return;
        }
        if (level == HASHX11KVS_MAX_LEVEL - 5 && cache[HASHX11KVS_CACHE_OFF_4 + (nonce % HASHX11KVS_CACHE_POS_5) * HASHX11KVS_CACHE_CHUNK] == 0xFF) {
            memcpy(pBuffer, cache + HASHX11KVS_CACHE_OFF_4 + ((nonce % HASHX11KVS_CACHE_POS_5) * HASHX11KVS_CACHE_CHUNK) + 1, 32);
            return;
        }
        if (level == HASHX11KVS_MAX_LEVEL - 6 && cache[HASHX11KVS_CACHE_OFF_5 + (nonce % HASHX11KVS_CACHE_POS_6) * HASHX11KVS_CACHE_CHUNK] == 0xFF) {
            memcpy(pBuffer, cache + HASHX11KVS_CACHE_OFF_5 + ((nonce % HASHX11KVS_CACHE_POS_6) * HASHX11KVS_CACHE_CHUNK) + 1, 32);
            return;
        }
    }

    uint8_t hash[32];
    HashX11KV(pbegin, len, hash);
    if (level == HASHX11KVS_MIN_LEVEL) {
        memcpy(pBuffer, hash, 32);
        if (cache) {
            cache[HASHX11KVS_CACHE_OFF_5 + (nonce % HASHX11KVS_CACHE_POS_6) * HASHX11KVS_CACHE_CHUNK] = 0xFF;
            memcpy(cache + HASHX11KVS_CACHE_OFF_5 + ((nonce % HASHX11KVS_CACHE_POS_6) * HASHX11KVS_CACHE_CHUNK) + 1, pBuffer, 32);
        }
        return;
    }

    if (cache && level == HASHX11KVS_MAX_LEVEL && nonce > 0) {
        cache[((nonce - 1) % HASHX11KVS_CACHE_POS_1) * HASHX11KVS_CACHE_CHUNK] = 0x00;
        cache[HASHX11KVS_CACHE_OFF_1 + ((nonce - 1) % HASHX11KVS_CACHE_POS_2) * HASHX11KVS_CACHE_CHUNK] = 0x00;
        cache[HASHX11KVS_CACHE_OFF_2 + ((nonce - 1) % HASHX11KVS_CACHE_POS_3) * HASHX11KVS_CACHE_CHUNK] = 0x00;
        cache[HASHX11KVS_CACHE_OFF_3 + ((nonce - 1) % HASHX11KVS_CACHE_POS_4) * HASHX11KVS_CACHE_CHUNK] = 0x00;
        cache[HASHX11KVS_CACHE_OFF_4 + ((nonce - 1) % HASHX11KVS_CACHE_POS_5) * HASHX11KVS_CACHE_CHUNK] = 0x00;
        cache[HASHX11KVS_CACHE_OFF_5 + ((nonce - 1) % HASHX11KVS_CACHE_POS_6) * HASHX11KVS_CACHE_CHUNK] = 0x00;
    }

    uint8_t nextheader1[80];
    uint8_t nextheader2[80];

    uint32_t nextnonce1 = nonce + (le32dec(hash + 24) % HASHX11KVS_MAX_DRIFT);
    uint32_t nextnonce2 = nonce + (le32dec(hash + 28) % HASHX11KVS_MAX_DRIFT);

    memcpy(nextheader1, pbegin, 76);
    le32enc(nextheader1 + 76, nextnonce1);

    memcpy(nextheader2, pbegin, 76);
    le32enc(nextheader2 + 76, nextnonce2);

    uint8_t concat[96];
    memcpy(concat, hash, 32);
    HashX11KVS_recursive(nextheader1, 80, concat + 32, level - 1, nextnonce1, cache);
    HashX11KVS_recursive(nextheader2, 80, concat + 64, level - 1, nextnonce2, cache);

    uint8_t sha_inner[32];
    SHA256(concat, 96, sha_inner);
    SHA256(sha_inner, 32, (uint8_t*)pBuffer);

    if (cache && level == HASHX11KVS_MAX_LEVEL - 1) {
        cache[(nonce % HASHX11KVS_CACHE_POS_1) * HASHX11KVS_CACHE_CHUNK] = 0xFF;
        memcpy(cache + ((nonce % HASHX11KVS_CACHE_POS_1) * HASHX11KVS_CACHE_CHUNK) + 1, pBuffer, 32);
        return;
    }
    if (cache && level == HASHX11KVS_MAX_LEVEL - 2) {
        cache[HASHX11KVS_CACHE_OFF_1 + (nonce % HASHX11KVS_CACHE_POS_2) * HASHX11KVS_CACHE_CHUNK] = 0xFF;
        memcpy(cache + HASHX11KVS_CACHE_OFF_1 + ((nonce % HASHX11KVS_CACHE_POS_2) * HASHX11KVS_CACHE_CHUNK) + 1, pBuffer, 32);
        return;
    }
    if (cache && level == HASHX11KVS_MAX_LEVEL - 3) {
        cache[HASHX11KVS_CACHE_OFF_2 + (nonce % HASHX11KVS_CACHE_POS_3) * HASHX11KVS_CACHE_CHUNK] = 0xFF;
        memcpy(cache + HASHX11KVS_CACHE_OFF_2 + ((nonce % HASHX11KVS_CACHE_POS_3) * HASHX11KVS_CACHE_CHUNK) + 1, pBuffer, 32);
        return;
    }
    if (cache && level == HASHX11KVS_MAX_LEVEL - 4) {
        cache[HASHX11KVS_CACHE_OFF_3 + (nonce % HASHX11KVS_CACHE_POS_4) * HASHX11KVS_CACHE_CHUNK] = 0xFF;
        memcpy(cache + HASHX11KVS_CACHE_OFF_3 + ((nonce % HASHX11KVS_CACHE_POS_4) * HASHX11KVS_CACHE_CHUNK) + 1, pBuffer, 32);
        return;
    }
    if (cache && level == HASHX11KVS_MAX_LEVEL - 5) {
        cache[HASHX11KVS_CACHE_OFF_4 + (nonce % HASHX11KVS_CACHE_POS_5) * HASHX11KVS_CACHE_CHUNK] = 0xFF;
        memcpy(cache + HASHX11KVS_CACHE_OFF_4 + ((nonce % HASHX11KVS_CACHE_POS_5) * HASHX11KVS_CACHE_CHUNK) + 1, pBuffer, 32);
        return;
    }
}

static void HashX11KVS(const uint8_t* pbegin, size_t len, void* pBuffer, unsigned int level = 7) {
    if (len < 80) {
        HashX11KV(pbegin, len, pBuffer);
        return;
    }
    static thread_local uint8_t* tls_kvs_cache = nullptr;
    static thread_local uint8_t tls_kvs_hdr[76] = {0};
    static thread_local bool tls_kvs_valid = false;
    static thread_local uint32_t tls_kvs_last_nonce = 0;
    uint32_t nonce = le32dec(pbegin + 76);
    if (!tls_kvs_cache) {
        tls_kvs_cache = (uint8_t*)calloc(1, HASHX11KVS_CACHE_TOTAL);
        tls_kvs_valid = false;
    }
    if (!tls_kvs_valid || memcmp(tls_kvs_hdr, pbegin, 76) != 0 || nonce != tls_kvs_last_nonce + 1) {
        if (tls_kvs_cache)
            memset(tls_kvs_cache, 0x00, HASHX11KVS_CACHE_TOTAL);
        memcpy(tls_kvs_hdr, pbegin, 76);
        tls_kvs_valid = true;
    }
    HashX11KVS_recursive(pbegin, len, pBuffer, level, nonce, tls_kvs_cache);
    tls_kvs_last_nonce = nonce;
}

extern "C" void cpu_hash_x11kvs_80(const uint8_t *header80, void *out32) {
    HashX11KVS(header80, 80, out32, 7);
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
            applog(LOG_ERR, "ADAM CPU unknown algo index %d", algoIndex);
            memset(pBuffer, 0, 32);
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

extern "C" size_t krista_address_to_script(unsigned char *out, size_t outsz, const char *addr) {
    static const char *pszBase58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    if (!out || !addr || !addr[0])
        return 0;

    size_t addr_len = strlen(addr);
    if (addr_len < 26 || addr_len > 64)
        return 0;

    unsigned char b256[50] = {0};
    size_t b256_len = 0;
    size_t leading_ones = 0;
    const char *p = addr;
    while (*p == '1') {
        leading_ones++;
        p++;
    }

    for (; *p; p++) {
        const char *ch = strchr(pszBase58, *p);
        if (!ch)
            return 0;
        int carry = (int)(ch - pszBase58);
        for (size_t i = 0; i < sizeof(b256); i++) {
            size_t idx = sizeof(b256) - 1 - i;
            int val = carry + 58 * (int)b256[idx];
            b256[idx] = (unsigned char)(val & 0xFF);
            carry = val >> 8;
        }
        if (carry != 0)
            return 0;
    }

    size_t first_nz = 0;
    while (first_nz < sizeof(b256) && b256[first_nz] == 0)
        first_nz++;

    b256_len = leading_ones + (sizeof(b256) - first_nz);
    if (b256_len != 25 && b256_len != 26)
        return 0;

    unsigned char bin[32] = {0};
    size_t nz_count = sizeof(b256) - first_nz;
    memcpy(bin + leading_ones, b256 + first_nz, nz_count);

    unsigned char h1[32], h2[32];
    SHA256(bin, b256_len - 4, h1);
    SHA256(h1, 32, h2);
    if (memcmp(h2, bin + b256_len - 4, 4) != 0)
        return 0;

    if (b256_len == 26) {
        bool is_main_p2pkh = (bin[0] == 10 && bin[1] == 100);
        bool is_test_p2pkh = (bin[0] == 24 && bin[1] == 181);
        if (!is_main_p2pkh && !is_test_p2pkh)
            return 0;
        if (outsz < 25)
            return 0;
        out[0] = 0x76; /* OP_DUP */
        out[1] = 0xa9; /* OP_HASH160 */
        out[2] = 0x14; /* push 20 */
        memcpy(out + 3, bin + 2, 20);
        out[23] = 0x88; /* OP_EQUALVERIFY */
        out[24] = 0xac; /* OP_CHECKSIG */
        return 25;
    }

    /* b256_len == 25: KRISTA P2SH ([16] main, [19] test/regtest) */
    if (bin[0] != 16 && bin[0] != 19)
        return 0;
    if (outsz < 23)
        return 0;
    out[0] = 0xa9; /* OP_HASH160 */
    out[1] = 0x14; /* push 20 */
    memcpy(out + 2, bin + 1, 20);
    out[22] = 0x87; /* OP_EQUAL */
    return 23;
}

extern "C" uint32_t run_cpu_classic_x11kvs_mine(int thr_id, const struct work *work, const uint32_t *pdata, uint32_t start_nonce, uint32_t throughput, const uint32_t *target, uint32_t *found_nonce) {
    (void)work;
    if (!found_nonce || !pdata || !target)
        return 0;

    std::atomic<uint32_t> best_nonce(UINT32_MAX);
    std::atomic<uint32_t> scanned(0);
    int max_t = std::min<int>(16, omp_get_max_threads());
    if (max_t < 1) max_t = 1;

    #pragma omp parallel num_threads(max_t)
    {
        uint8_t hdr[80];
        memcpy(hdr, pdata, 76);

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < (int64_t)throughput; i++) {
            if (best_nonce.load(std::memory_order_relaxed) != UINT32_MAX || work_restart[thr_id].restart)
                continue;

            uint32_t nonce = start_nonce + (uint32_t)i;
            le32enc(hdr + 76, nonce);

            uint32_t hash32[8];
            cpu_hash_x11kvs_80(hdr, hash32);
            scanned.fetch_add(1, std::memory_order_relaxed);

            if (fulltest(hash32, target)) {
                uint32_t cur = best_nonce.load(std::memory_order_relaxed);
                while (nonce < cur && !best_nonce.compare_exchange_weak(cur, nonce, std::memory_order_relaxed)) {}
            }
        }
    }

    uint32_t winner = best_nonce.load(std::memory_order_relaxed);
    if (winner != UINT32_MAX && (*found_nonce == UINT32_MAX || winner < *found_nonce))
        *found_nonce = winner;
    return scanned.load(std::memory_order_relaxed);
}

extern "C" uint32_t run_cpu_fallback_mine(int thr_id, const struct work *work, const uint8_t *input_template, uint32_t start_nonce, uint32_t throughput, const uint32_t *target, uint32_t *found_nonce) {
    if (!found_nonce)
        return 0;

    const int w_version = work->adam_version ? work->adam_version : 11;
    const int w_algo_single = work->adam_algo_single;
    const int w_algo1 = work->adam_algo1;
    const int w_algo2 = work->adam_algo2;
    const int w_algo3 = work->adam_algo3;
    const int w_factor = work->adam_factor > 0 ? work->adam_factor : 1;

    std::atomic<uint32_t> best_nonce(UINT32_MAX);
    int64_t scanned = 0;
    #pragma omp parallel for reduction(+:scanned)
    for (int64_t i = 0; i < (int64_t)throughput; i++) {
        if (best_nonce.load(std::memory_order_relaxed) != UINT32_MAX || work_restart[thr_id].restart) continue;
        scanned++;

        uint32_t nonce = start_nonce + (uint32_t)i;
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
            uint32_t cur = best_nonce.load(std::memory_order_relaxed);
            while (nonce < cur && !best_nonce.compare_exchange_weak(cur, nonce, std::memory_order_relaxed)) {}
        }
    }

    uint32_t winner = best_nonce.load(std::memory_order_relaxed);
    if (winner != UINT32_MAX && (*found_nonce == UINT32_MAX || winner < *found_nonce))
        *found_nonce = winner;
    return (uint32_t)scanned;
}
