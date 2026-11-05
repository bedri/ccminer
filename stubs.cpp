#include <stdint.h>
#include "miner.h"

// Stubs for cross-referenced functions from removed source files that have C++ linkage (no extern "C" in .cu files)
void x16_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash) {}
void whirlpool_midstate(void *state, const void *input) {}
void lyra2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix) {}
void lyra2_cuda_hash_64(int thr_id, const uint32_t threads, uint64_t* d_hash_256, uint32_t* d_hash_512, bool gtx750ti) {}
void phi_echo512_cpu_hash_64_filtered(int thr_id, const uint32_t threads, uint32_t* g_hash, uint32_t* d_filter) {}

extern "C" {

// Stubs for heavy/bastion
void heavycoin_hash(unsigned char* output, const unsigned char* input, int len) {}
int scanhash_heavy(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done, uint32_t maxvote, int block_header_sz) { return 0; }
void free_heavy(int thr_id) {}

int scanhash_bastion(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done) { return 0; }
void free_bastion(int thr_id) {}

// Stubs for equihash
int scanhash_equihash(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done) { return 0; }
void free_equihash(int thr_id) {}

bool equi_stratum_notify(struct stratum_ctx *sctx, json_t *params) { return false; }
bool equi_stratum_set_target(struct stratum_ctx *sctx, json_t *params) { return false; }
bool equi_stratum_submit(struct pool_infos *pool, struct work *work) { return false; }
bool equi_stratum_show_message(struct stratum_ctx *sctx, json_t *id, json_t *params) { return false; }
void equi_work_set_target(struct work* work, double diff) {}
void equi_store_work_solution(struct work* work, uint32_t* hash, void* sol_data) {}
int equi_verify_sol(void * const hdr, void * const sol) { return 0; }
double equi_network_diff(struct work *work) { return 0.0; }

// Stubs for lyra2 variants and allium
int scanhash_lyra2(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done) { return 0; }
void free_lyra2(int thr_id) {}

int scanhash_lyra2v2(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done) { return 0; }
void free_lyra2v2(int thr_id) {}

int scanhash_lyra2v3(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done) { return 0; }
void free_lyra2v3(int thr_id) {}

int scanhash_lyra2Z(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done) { return 0; }
void free_lyra2Z(int thr_id) {}

int scanhash_allium(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done) { return 0; }
void free_allium(int thr_id) {}

// Stubs for neoscrypt
int scanhash_neoscrypt(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done) { return 0; }
void free_neoscrypt(int thr_id) {}

// Stubs for x16r and x16s
int scanhash_x16r(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done) { return 0; }
void free_x16r(int thr_id) {}

int scanhash_x16s(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done) { return 0; }
void free_x16s(int thr_id) {}

// Stubs for scrypt and scrypt-jane
int scanhash_scrypt(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done,
        unsigned char *scratchbuf, struct timeval *tv_start, struct timeval *tv_end) { return 0; }
int scanhash_scrypt_jane(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done,
        unsigned char *scratchbuf, struct timeval *tv_start, struct timeval *tv_end) { return 0; }
void free_scrypt(int thr_id) {}
void free_scrypt_jane(int thr_id) {}

// Stubs for fugue256
void fugue256_hash(unsigned char* output, const unsigned char* input, int len) {}
int scanhash_fugue256(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done) { return 0; }
void free_fugue256(int thr_id) {}

// Print hash test stubs
void allium_hash(void *state, const void *input) {}
void bastionhash(void* output, const unsigned char* input) {}
void lyra2re_hash(void *state, const void *input) {}
void lyra2v2_hash(void *state, const void *input) {}
void lyra2v3_hash(void *state, const void *input) {}
void lyra2Z_hash(void *state, const void *input) {}
void neoscrypt(unsigned char *output, const unsigned char *input, uint32_t profile) {}
void scrypthash(void* output, const void* input) {}
void scryptjane_hash(void* output, const void* input) {}
void x16r_hash(void *output, const void *input) {}
void x16s_hash(void *output, const void *input) {}

// LYRA2 has C linkage
int LYRA2(void *K, int64_t kLen, const void *pwd, int32_t pwdlen, const void *salt, int32_t saltlen, int64_t timeCost, const int16_t nRows, const int16_t nCols) { return 0; }

}
