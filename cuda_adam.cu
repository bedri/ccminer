#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>

#include "miner.h"
#include "cuda_helper.h"

#include "adam/cuda_adam_common.cuh"
#include "adam/cuda_adam_g1.cuh"
#include "adam/cuda_adam_g2.cuh"
#include "adam/cuda_adam_g3.cuh"
#include "adam/cuda_adam_g4.cuh"

extern int opt_scantime;

/* CPU fallback and puzzle hash functions (from adam_cpu.cpp) */
extern "C" uint32_t run_cpu_fallback_mine(int thr_id, const struct work *work, const uint8_t *input_template, uint32_t start_nonce, uint32_t throughput, const uint32_t *target, uint32_t *found_nonce);
extern "C" void cpu_adam_puzzle_hash(int algoIndex, const uint8_t* pbegin, size_t len, void* pBuffer);

/* Per-GPU device state */
static uint32_t *d_hash[MAX_GPUS] = { 0 };
static uint32_t *d_resNonce[MAX_GPUS] = { 0 };
static uint32_t d_alloc_threads[MAX_GPUS] = { 0 };
static bool init[MAX_GPUS] = { 0 };

static bool adam_is_gpu_algo(int algoIndex)
{
	return (algoIndex >= 0 && algoIndex <= 17 && algoIndex != 11);
}

static bool adam_ensure_gpu_init(int thr_id, uint32_t req_threads, bool log_intensity)
{
	if (thr_id < 0 || thr_id >= MAX_GPUS || active_gpus <= 0) {
		applog(LOG_ERR, "ADAM CUDA: no active CUDA device for thread %d", thr_id);
		return false;
	}

	int dev_id = device_map[thr_id];
	cudaError_t err = cudaSetDevice(dev_id);
	if (err != cudaSuccess) {
		cudaReportHardwareFailure(thr_id, err, __func__);
		return false;
	}

	if (!init[thr_id]) {
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		if (log_intensity) {
			gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
			       throughput2intensity(req_threads), req_threads);
		}
	}

	if (!d_resNonce[thr_id]) {
		err = cudaMalloc(&d_resNonce[thr_id], sizeof(uint32_t));
		if (err != cudaSuccess) {
			cudaReportHardwareFailure(thr_id, err, __func__);
			return false;
		}
	}

	if (!d_hash[thr_id] || req_threads > d_alloc_threads[thr_id]) {
		if (d_hash[thr_id]) {
			cudaFree(d_hash[thr_id]);
			d_hash[thr_id] = NULL;
			d_alloc_threads[thr_id] = 0;
		}
		err = cudaMalloc(&d_hash[thr_id], (size_t)req_threads * 32U);
		if (err != cudaSuccess) {
			cudaReportHardwareFailure(thr_id, err, __func__);
			return false;
		}
		d_alloc_threads[thr_id] = req_threads;
	}

	init[thr_id] = true;
	return true;
}

extern "C" void free_adam(int thr_id)
{
	if (thr_id < 0 || thr_id >= MAX_GPUS || !init[thr_id])
		return;

	init[thr_id] = false;
	d_alloc_threads[thr_id] = 0;
	cudaSetDevice(device_map[thr_id]);
	if (d_hash[thr_id]) {
		cudaFree(d_hash[thr_id]);
		d_hash[thr_id] = NULL;
	}
	if (d_resNonce[thr_id]) {
		cudaFree(d_resNonce[thr_id]);
		d_resNonce[thr_id] = NULL;
	}
	cudaDeviceSynchronize();
}

static bool launch_adam_70(int thr_id, int algoIndex, dim3 grid, dim3 block, uint32_t threads, uint32_t start_nonce, uint32_t *g_hash)
{
	switch (algoIndex) {
		case 0:  adam_gpu_blake512_70<<<grid, block>>>(threads, start_nonce, (uint64_t*)g_hash); break;
		case 1:  adam_gpu_bmw512_70<<<grid, block>>>(threads, start_nonce, (uint64_t*)g_hash); break;
		case 2:  adam_gpu_groestl512_70<<<grid, block>>>(threads, start_nonce, (uint64_t*)g_hash); break;
		case 3:  adam_gpu_jh512_70<<<grid, block>>>(threads, start_nonce, g_hash); break;
		case 4:  adam_gpu_keccak512_70<<<grid, block>>>(threads, start_nonce, (uint64_t*)g_hash); break;
		case 5:  adam_gpu_skein512_70<<<grid, block>>>(threads, start_nonce, (uint64_t*)g_hash); break;
		case 6:  adam_gpu_luffa512_70<<<grid, block>>>(threads, start_nonce, g_hash); break;
		case 7:  adam_gpu_cubehash512_70<<<grid, block>>>(threads, start_nonce, g_hash); break;
		case 8:  adam_gpu_shavite512_70<<<grid, block>>>(threads, start_nonce, g_hash); break;
		case 9:  adam_gpu_simd512_70<<<grid, block>>>(threads, start_nonce, g_hash); break;
		case 10: adam_gpu_echo512_70<<<grid, block>>>(threads, start_nonce, g_hash); break;
		case 12: adam_gpu_sha256d_70<<<grid, block>>>(threads, start_nonce, g_hash); break;
		case 13: adam_gpu_hamsi512_70<<<grid, block>>>(threads, start_nonce, g_hash); break;
		case 14: adam_gpu_fugue512_70<<<grid, block>>>(threads, start_nonce, g_hash); break;
		case 15: adam_gpu_shabal512_70<<<grid, block>>>(threads, start_nonce, g_hash); break;
		case 16: adam_gpu_whirlpool_70<<<grid, block>>>(threads, start_nonce, (uint64_t*)g_hash); break;
		case 17: adam_gpu_haval256_70<<<grid, block>>>(threads, start_nonce, g_hash); break;
		default:
			applog(LOG_ERR, "ADAM CUDA: unsupported 70B algo index %d", algoIndex);
			return false;
	}
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		cudaReportHardwareFailure(thr_id, err, __func__);
		return false;
	}
	return true;
}

static bool launch_adam_32(int thr_id, int algoIndex, dim3 grid, dim3 block, uint32_t threads, uint32_t *g_hash)
{
	switch (algoIndex) {
		case 0:  adam_gpu_blake512_32<<<grid, block>>>(threads, (uint64_t*)g_hash); break;
		case 1:  adam_gpu_bmw512_32<<<grid, block>>>(threads, (uint64_t*)g_hash); break;
		case 2:  adam_gpu_groestl512_32<<<grid, block>>>(threads, (uint64_t*)g_hash); break;
		case 3:  adam_gpu_jh512_32<<<grid, block>>>(threads, g_hash); break;
		case 4:  adam_gpu_keccak512_32<<<grid, block>>>(threads, (uint64_t*)g_hash); break;
		case 5:  adam_gpu_skein512_32<<<grid, block>>>(threads, (uint64_t*)g_hash); break;
		case 6:  adam_gpu_luffa512_32<<<grid, block>>>(threads, g_hash); break;
		case 7:  adam_gpu_cubehash512_32<<<grid, block>>>(threads, g_hash); break;
		case 8:  adam_gpu_shavite512_32<<<grid, block>>>(threads, g_hash); break;
		case 9:  adam_gpu_simd512_32<<<grid, block>>>(threads, g_hash); break;
		case 10: adam_gpu_echo512_32<<<grid, block>>>(threads, g_hash); break;
		case 12: adam_gpu_sha256d_32<<<grid, block>>>(threads, g_hash); break;
		case 13: adam_gpu_hamsi512_32<<<grid, block>>>(threads, g_hash); break;
		case 14: adam_gpu_fugue512_32<<<grid, block>>>(threads, g_hash); break;
		case 15: adam_gpu_shabal512_32<<<grid, block>>>(threads, g_hash); break;
		case 16: adam_gpu_whirlpool_32<<<grid, block>>>(threads, (uint64_t*)g_hash); break;
		case 17: adam_gpu_haval256_32<<<grid, block>>>(threads, g_hash); break;
		default:
			applog(LOG_ERR, "ADAM CUDA: unsupported 32B algo index %d", algoIndex);
			return false;
	}
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		cudaReportHardwareFailure(thr_id, err, __func__);
		return false;
	}
	return true;
}

static void cpu_scale_hash(uint32_t *hash, uint32_t i_factor)
{
	uint64_t carry = 0;
	for (int i = 0; i < 8; i++) {
		uint64_t val = (uint64_t)hash[i] * (uint64_t)i_factor + carry;
		hash[i] = (uint32_t)val;
		carry = val >> 32;
	}
}

static void adam_eval_cpu(int version, int algo_single, int algo1, int algo2, int algo3, uint32_t factor, const uint8_t *input70, uint32_t *out32)
{
	if (version == 12) {
		uint32_t hash3[8];
		cpu_adam_puzzle_hash(algo3, input70, 70, hash3);
		cpu_scale_hash(hash3, factor);

		uint32_t hash2[8];
		cpu_adam_puzzle_hash(algo2, (const uint8_t*)hash3, 32, hash2);
		cpu_scale_hash(hash2, factor);

		cpu_adam_puzzle_hash(algo1, (const uint8_t*)hash2, 32, out32);
	} else {
		cpu_adam_puzzle_hash(algo_single, input70, 70, out32);
	}
}

extern "C" int scanhash_adam(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const time_t scan_start = time(NULL);
	const int max_scan_sec = (opt_scantime > 0) ? opt_scantime : 3;

	/* Path 1: Classic X11KVS PoW (pre-UPGRADE_ADAM blocks 1..199) on CPU OpenMP */
	if (work->adam_is_classic) {
		do {
			uint64_t avail = (pdata[19] < max_nonce)
				? ((uint64_t)max_nonce - (uint64_t)pdata[19] + (max_nonce == UINT32_MAX ? 1ULL : 0ULL))
				: 0ULL;
			uint32_t batch = (uint32_t)std::min<uint64_t>(65536ULL, avail);
			if (batch == 0)
				break;

			uint32_t found_nonce = UINT32_MAX;
			uint32_t scanned = run_cpu_classic_x11kvs_mine(
				thr_id, work, pdata, pdata[19], batch, ptarget, &found_nonce);

			if (found_nonce != UINT32_MAX) {
				uint8_t hdr[80];
				for (int k = 0; k < 20; k++)
					le32enc(hdr + 4 * k, pdata[k]);
				le32enc(hdr + 76, found_nonce);

				uint32_t vhash[8];
				cpu_hash_x11kvs_80(hdr, vhash);
				if (fulltest(vhash, ptarget)) {
					work_set_target_ratio(work, vhash);
					work->nonces[0] = found_nonce;
					work->valid_nonces = 1;
					*hashes_done = (unsigned long)(found_nonce - first_nonce + 1);
					pdata[19] = (found_nonce == UINT32_MAX) ? UINT32_MAX : (found_nonce + 1);
					return 1;
				}
				applog(LOG_ERR, "Classic X11KVS CPU verify failed at nonce %08x", found_nonce);
			}

			uint64_t next_n = (uint64_t)pdata[19] + scanned;
			if (next_n >= max_nonce) {
				pdata[19] = max_nonce;
				*hashes_done = (unsigned long)(next_n - first_nonce);
				return 0;
			}
			pdata[19] = (uint32_t)next_n;

			if ((time(NULL) - scan_start) >= max_scan_sec)
				break;
		} while (!work_restart[thr_id].restart && !abort_flag);

		*hashes_done = (unsigned long)(pdata[19] - first_nonce);
		return 0;
	}

	/* Build 66-byte ADAM puzzle prefix (bytes 66..69 are LE32 nonce) */
	uint8_t input_template[70];
	uint32_t le_data[20];
	for (int k = 0; k < 20; k++)
		le32enc(le_data + k, pdata[k]);
	memcpy(input_template, le_data, 66);

	int w_version = work->adam_version ? work->adam_version : 11;
	int w_algo_single = work->adam_algo_single;
	int w_algo1 = work->adam_algo1;
	int w_algo2 = work->adam_algo2;
	int w_algo3 = work->adam_algo3;
	uint32_t w_factor = (work->adam_factor > 0) ? (uint32_t)work->adam_factor : 1U;

	if (opt_benchmark && work->adam_version == 0) {
		w_version = 11;
		w_algo_single = 0;
		ptarget[7] = 0x0000FFFFU;
	}

	bool use_cpu_fallback = false;
	bool valid_algos = false;
	if (w_version == 12) {
		if (w_algo1 == 11 || w_algo2 == 11 || w_algo3 == 11) {
			use_cpu_fallback = (w_algo1 >= 0 && w_algo1 <= 17 &&
			                    w_algo2 >= 0 && w_algo2 <= 17 &&
			                    w_algo3 >= 0 && w_algo3 <= 17);
			valid_algos = use_cpu_fallback;
		} else {
			valid_algos = adam_is_gpu_algo(w_algo1) &&
			              adam_is_gpu_algo(w_algo2) &&
			              adam_is_gpu_algo(w_algo3);
		}
	} else {
		if (w_algo_single == 11) {
			use_cpu_fallback = true;
			valid_algos = true;
		} else {
			valid_algos = adam_is_gpu_algo(w_algo_single);
		}
	}

	if (!valid_algos) {
		applog(LOG_ERR, "ADAM: invalid puzzle algorithm specification (ver=%d single=%d chain=%d+%d+%d)",
		       w_version, w_algo_single, w_algo1, w_algo2, w_algo3);
		*hashes_done = 0;
		return 0;
	}

	/* Path 2: Legacy X11KVS (algo 11) CPU OpenMP fallback */
	if (use_cpu_fallback) {
		do {
			uint64_t avail = (pdata[19] < max_nonce)
				? ((uint64_t)max_nonce - (uint64_t)pdata[19] + (max_nonce == UINT32_MAX ? 1ULL : 0ULL))
				: 0ULL;
			uint32_t batch = (uint32_t)std::min<uint64_t>(65536ULL, avail);
			if (batch == 0)
				break;

			uint32_t found_nonce = UINT32_MAX;
			uint32_t scanned = run_cpu_fallback_mine(
				thr_id, work, input_template, pdata[19], batch, ptarget, &found_nonce);

			if (found_nonce != UINT32_MAX) {
				uint8_t check_in[70];
				memcpy(check_in, input_template, 66);
				le32enc(check_in + 66, found_nonce);

				uint32_t vhash[8];
				adam_eval_cpu(w_version, w_algo_single, w_algo1, w_algo2, w_algo3, w_factor, check_in, vhash);
				if (fulltest(vhash, ptarget)) {
					work_set_target_ratio(work, vhash);
					work->nonces[0] = found_nonce;
					work->valid_nonces = 1;
					*hashes_done = (unsigned long)(found_nonce - first_nonce + 1);
					pdata[19] = (found_nonce == UINT32_MAX) ? UINT32_MAX : (found_nonce + 1);
					return 1;
				}
				applog(LOG_ERR, "ADAM fallback CPU verify failed at nonce %08x", found_nonce);
			}

			uint64_t next_n = (uint64_t)pdata[19] + scanned;
			if (next_n >= max_nonce) {
				pdata[19] = max_nonce;
				*hashes_done = (unsigned long)(next_n - first_nonce);
				return 0;
			}
			pdata[19] = (uint32_t)next_n;

			if ((time(NULL) - scan_start) >= max_scan_sec)
				break;
		} while (!work_restart[thr_id].restart && !abort_flag);

		*hashes_done = (unsigned long)(pdata[19] - first_nonce);
		return 0;
	}

	/* Path 3: Native 17-algorithm CUDA GPU V11 / V12 pipeline */
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 20);
	if (!adam_ensure_gpu_init(thr_id, throughput, true)) {
		*hashes_done = 0;
		return 0;
	}

	uint8_t prefix68[68];
	memset(prefix68, 0, sizeof(prefix68));
	memcpy(prefix68, input_template, 66);

	cudaError_t err = cudaMemcpyToSymbol(c_adam_prefix, prefix68, sizeof(prefix68));
	if (err != cudaSuccess) {
		cudaReportHardwareFailure(thr_id, err, __func__);
		*hashes_done = 0;
		return 0;
	}
	err = cudaMemcpyToSymbol(c_adam_target, ptarget, 32);
	if (err != cudaSuccess) {
		cudaReportHardwareFailure(thr_id, err, __func__);
		*hashes_done = 0;
		return 0;
	}

	const uint32_t TPB = 256;

	do {
		uint64_t avail = (pdata[19] < max_nonce)
			? ((uint64_t)max_nonce - (uint64_t)pdata[19] + (max_nonce == UINT32_MAX ? 1ULL : 0ULL))
			: 0ULL;
		uint32_t batch = (uint32_t)std::min<uint64_t>(throughput, avail);
		if (batch == 0)
			break;

		dim3 block(TPB);
		dim3 grid((batch + TPB - 1U) / TPB);

		err = cudaMemset(d_resNonce[thr_id], 0xFF, sizeof(uint32_t));
		if (err != cudaSuccess) {
			cudaReportHardwareFailure(thr_id, err, __func__);
			break;
		}

		if (w_version == 12) {
			if (!launch_adam_70(thr_id, w_algo3, grid, block, batch, pdata[19], d_hash[thr_id]))
				break;
			adam_scale_hash_kernel<<<grid, block>>>(batch, d_hash[thr_id], w_factor);
			err = cudaGetLastError();
			if (err != cudaSuccess) {
				cudaReportHardwareFailure(thr_id, err, __func__);
				break;
			}
			if (!launch_adam_32(thr_id, w_algo2, grid, block, batch, d_hash[thr_id]))
				break;
			adam_scale_hash_kernel<<<grid, block>>>(batch, d_hash[thr_id], w_factor);
			err = cudaGetLastError();
			if (err != cudaSuccess) {
				cudaReportHardwareFailure(thr_id, err, __func__);
				break;
			}
			if (!launch_adam_32(thr_id, w_algo1, grid, block, batch, d_hash[thr_id]))
				break;
		} else {
			if (!launch_adam_70(thr_id, w_algo_single, grid, block, batch, pdata[19], d_hash[thr_id]))
				break;
		}

		adam_check_target_kernel<<<grid, block>>>(batch, d_hash[thr_id], d_resNonce[thr_id]);
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			cudaReportHardwareFailure(thr_id, err, __func__);
			break;
		}

		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			cudaReportHardwareFailure(thr_id, err, __func__);
			break;
		}

		uint32_t res_tid = 0xFFFFFFFFU;
		err = cudaMemcpy(&res_tid, d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			cudaReportHardwareFailure(thr_id, err, __func__);
			break;
		}

		if (res_tid < batch) {
			uint32_t found_nonce = pdata[19] + res_tid;
			uint8_t check_in[70];
			memcpy(check_in, input_template, 66);
			le32enc(check_in + 66, found_nonce);

			uint32_t vhash[8];
			adam_eval_cpu(w_version, w_algo_single, w_algo1, w_algo2, w_algo3, w_factor, check_in, vhash);

			if (fulltest(vhash, ptarget)) {
				work_set_target_ratio(work, vhash);
				work->nonces[0] = found_nonce;
				work->valid_nonces = 1;
				*hashes_done = (unsigned long)(found_nonce - first_nonce + 1);
				pdata[19] = (found_nonce == UINT32_MAX) ? UINT32_MAX : (found_nonce + 1);
				return 1;
			}

			applog(LOG_ERR, "ADAM CUDA/CPU mismatch at nonce %08x (tid=%u, ver=%d); discarding GPU hit",
			       found_nonce, res_tid, w_version);
			if (found_nonce == UINT32_MAX || found_nonce >= max_nonce) {
				pdata[19] = max_nonce;
				break;
			}
			pdata[19] = found_nonce + 1;
			continue;
		}

		uint64_t next_n = (uint64_t)pdata[19] + batch;
		if (next_n >= max_nonce) {
			pdata[19] = max_nonce;
			*hashes_done = (unsigned long)(next_n - first_nonce);
			return 0;
		}
		pdata[19] = (uint32_t)next_n;

		if ((time(NULL) - scan_start) >= max_scan_sec)
			break;
	} while (!work_restart[thr_id].restart && !abort_flag);

	*hashes_done = (unsigned long)(pdata[19] - first_nonce);
	return 0;
}

static const char *adam_algo_label(int algoIndex)
{
	switch (algoIndex) {
		case 0:  return "blake";
		case 1:  return "bmw";
		case 2:  return "groestl";
		case 3:  return "jh";
		case 4:  return "keccak";
		case 5:  return "skein";
		case 6:  return "luffa";
		case 7:  return "cubehash";
		case 8:  return "shavite";
		case 9:  return "simd";
		case 10: return "echo";
		case 12: return "DoubleSHA256";
		case 13: return "hamsi";
		case 14: return "fugue";
		case 15: return "shabal";
		case 16: return "whirlpool";
		case 17: return "haval256_5";
		default: return "unknown";
	}
}

extern "C" bool cuda_adam_selftest(int thr_id)
{
	const int algos[17] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17 };
	const uint32_t TPB = 256;
	int checks = 0;
	int failures = 0;

	if (!adam_ensure_gpu_init(thr_id, 64, false)) {
		applog(LOG_ERR, "ADAM CUDA self-test: failed to initialize GPU %d", thr_id);
		return false;
	}

	uint8_t prefix68[68];
	memset(prefix68, 0, sizeof(prefix68));
	for (int i = 0; i < 66; i++)
		prefix68[i] = (uint8_t)((i * 37 + 11) & 0xFF);
	prefix68[32] = 0x21;

	if (cudaMemcpyToSymbol(c_adam_prefix, prefix68, sizeof(prefix68)) != cudaSuccess) {
		free_adam(thr_id);
		return false;
	}

	uint8_t gpu_out[32 * 32];
	uint8_t in32[32 * 32];
	uint8_t cpu_out[32];

	/* 1. Single-pass 70B & 32B parity across all 17 CUDA algorithms */
	for (int ai = 0; ai < 17; ai++) {
		int algo = algos[ai];
		const uint32_t N70 = 32;
		const uint32_t start70 = 0xFFFFFFF0U;

		if (!launch_adam_70(thr_id, algo, dim3(1), dim3(TPB), N70, start70, d_hash[thr_id]) ||
		    cudaDeviceSynchronize() != cudaSuccess ||
		    cudaMemcpy(gpu_out, d_hash[thr_id], N70 * 32, cudaMemcpyDeviceToHost) != cudaSuccess) {
			applog(LOG_ERR, "ADAM CUDA self-test: 70B kernel execution failed for algo %d (%s)",
			       algo, adam_algo_label(algo));
			failures++;
			continue;
		}

		int algo_fail_70 = 0;
		for (uint32_t i = 0; i < N70; i++) {
			uint8_t in70[70];
			memcpy(in70, prefix68, 66);
			le32enc(in70 + 66, start70 + i);
			cpu_adam_puzzle_hash(algo, in70, 70, cpu_out);
			checks++;
			if (memcmp(gpu_out + i * 32, cpu_out, 32) != 0) {
				if (algo_fail_70 == 0) {
					applog(LOG_ERR, "ADAM CUDA 70B mismatch on algo %d (%s) at tid=%u",
					       algo, adam_algo_label(algo), i);
				}
				algo_fail_70++;
				failures++;
			}
		}

		for (uint32_t i = 0; i < 32; i++) {
			for (int b = 0; b < 32; b++)
				in32[i * 32 + b] = (uint8_t)(((i + 1) * 53 + b * 19 + algo) & 0xFF);
		}
		if (cudaMemcpy(d_hash[thr_id], in32, 32 * 32, cudaMemcpyHostToDevice) != cudaSuccess ||
		    !launch_adam_32(thr_id, algo, dim3(1), dim3(TPB), 32, d_hash[thr_id]) ||
		    cudaDeviceSynchronize() != cudaSuccess ||
		    cudaMemcpy(gpu_out, d_hash[thr_id], 32 * 32, cudaMemcpyDeviceToHost) != cudaSuccess) {
			applog(LOG_ERR, "ADAM CUDA self-test: 32B kernel execution failed for algo %d (%s)",
			       algo, adam_algo_label(algo));
			failures++;
			continue;
		}

		int algo_fail_32 = 0;
		for (uint32_t i = 0; i < 32; i++) {
			cpu_adam_puzzle_hash(algo, in32 + i * 32, 32, cpu_out);
			checks++;
			if (memcmp(gpu_out + i * 32, cpu_out, 32) != 0) {
				if (algo_fail_32 == 0) {
					applog(LOG_ERR, "ADAM CUDA 32B mismatch on algo %d (%s) at tid=%u",
					       algo, adam_algo_label(algo), i);
				}
				algo_fail_32++;
				failures++;
			}
		}
	}

	/* 2. Full V11 pipeline across all 17 algos */
	for (int ai = 0; ai < 17; ai++) {
		int algo = algos[ai];
		const uint32_t N = 16;
		const uint32_t start = 0x12345670U;
		if (!launch_adam_70(thr_id, algo, dim3(1), dim3(TPB), N, start, d_hash[thr_id]) ||
		    cudaDeviceSynchronize() != cudaSuccess ||
		    cudaMemcpy(gpu_out, d_hash[thr_id], N * 32, cudaMemcpyDeviceToHost) != cudaSuccess) {
			failures++;
			continue;
		}
		for (uint32_t i = 0; i < N; i++) {
			uint8_t in70[70];
			memcpy(in70, prefix68, 66);
			le32enc(in70 + 66, start + i);
			adam_eval_cpu(11, algo, -1, -1, -1, 1, in70, (uint32_t*)cpu_out);
			checks++;
			if (memcmp(gpu_out + i * 32, cpu_out, 32) != 0) {
				applog(LOG_ERR, "ADAM CUDA V11 mismatch on algo %d (%s) tid=%u",
				       algo, adam_algo_label(algo), i);
				failures++;
			}
		}
	}

	/* 3. Full V12 3-permutation pipeline (minerIdx = 0, 10, 14 -> factors 1, 11, 15) */
	const int miner_idxs[3] = { 0, 10, 14 };
	for (int mi = 0; mi < 3; mi++) {
		uint32_t factor = (uint32_t)(miner_idxs[mi] + 1);
		for (int ti = 0; ti < 17; ti++) {
			int a3 = algos[ti];
			int a2 = algos[(ti * 5 + 3) % 17];
			int a1 = algos[(ti * 11 + 7) % 17];
			const uint32_t N = 16;
			const uint32_t start = 0x76543210U + (uint32_t)(mi * 17);

			if (!launch_adam_70(thr_id, a3, dim3(1), dim3(TPB), N, start, d_hash[thr_id])) { failures++; continue; }
			adam_scale_hash_kernel<<<dim3(1), dim3(TPB)>>>(N, d_hash[thr_id], factor);
			if (!launch_adam_32(thr_id, a2, dim3(1), dim3(TPB), N, d_hash[thr_id])) { failures++; continue; }
			adam_scale_hash_kernel<<<dim3(1), dim3(TPB)>>>(N, d_hash[thr_id], factor);
			if (!launch_adam_32(thr_id, a1, dim3(1), dim3(TPB), N, d_hash[thr_id])) { failures++; continue; }

			if (cudaDeviceSynchronize() != cudaSuccess ||
			    cudaMemcpy(gpu_out, d_hash[thr_id], N * 32, cudaMemcpyDeviceToHost) != cudaSuccess) {
				failures++;
				continue;
			}
			for (uint32_t i = 0; i < N; i++) {
				uint8_t in70[70];
				memcpy(in70, prefix68, 66);
				le32enc(in70 + 66, start + i);
				adam_eval_cpu(12, -1, a1, a2, a3, factor, in70, (uint32_t*)cpu_out);
				checks++;
				if (memcmp(gpu_out + i * 32, cpu_out, 32) != 0) {
					applog(LOG_ERR, "ADAM CUDA V12 mismatch (%s+%s+%s factor=%u tid=%u)",
					       adam_algo_label(a1), adam_algo_label(a2), adam_algo_label(a3), factor, i);
					failures++;
				}
			}
		}
	}

	/* 4. Target check kernel & relative-tid wrap around nonce 0xFFFFFFFFU */
	{
		const uint32_t N = 16;
		const uint32_t start = 0xFFFFFFF8U; /* tid=7 is nonce 0xFFFFFFFFU */
		if (launch_adam_70(thr_id, 0, dim3(1), dim3(TPB), N, start, d_hash[thr_id])) {
			adam_scale_hash_kernel<<<dim3(1), dim3(TPB)>>>(N, d_hash[thr_id], 15);
			launch_adam_32(thr_id, 5, dim3(1), dim3(TPB), N, d_hash[thr_id]);
			adam_scale_hash_kernel<<<dim3(1), dim3(TPB)>>>(N, d_hash[thr_id], 15);
			launch_adam_32(thr_id, 12, dim3(1), dim3(TPB), N, d_hash[thr_id]);
			cudaDeviceSynchronize();
			cudaMemcpy(gpu_out, d_hash[thr_id], N * 32, cudaMemcpyDeviceToHost);

			uint32_t exact_target[8];
			memcpy(exact_target, gpu_out + 7 * 32, 32);
			cudaMemcpyToSymbol(c_adam_target, exact_target, 32);
			cudaMemset(d_resNonce[thr_id], 0xFF, sizeof(uint32_t));
			adam_check_target_kernel<<<dim3(1), dim3(TPB)>>>(N, d_hash[thr_id], d_resNonce[thr_id]);
			cudaDeviceSynchronize();
			uint32_t hit_tid = 0xFFFFFFFFU;
			cudaMemcpy(&hit_tid, d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
			checks++;
			if (hit_tid > 7 || !fulltest((const uint32_t*)(gpu_out + hit_tid * 32), exact_target)) {
				applog(LOG_ERR, "ADAM CUDA target hit check failed (hit_tid=%u)", hit_tid);
				failures++;
			}

			uint32_t zero_target[8] = { 0 };
			cudaMemcpyToSymbol(c_adam_target, zero_target, 32);
			cudaMemset(d_resNonce[thr_id], 0xFF, sizeof(uint32_t));
			adam_check_target_kernel<<<dim3(1), dim3(TPB)>>>(N, d_hash[thr_id], d_resNonce[thr_id]);
			cudaDeviceSynchronize();
			hit_tid = 0xFFFFFFFFU;
			cudaMemcpy(&hit_tid, d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
			checks++;
			if (hit_tid != 0xFFFFFFFFU) {
				applog(LOG_ERR, "ADAM CUDA zero-target miss check failed (hit_tid=%u)", hit_tid);
				failures++;
			}
		} else {
			failures++;
		}
	}

	free_adam(thr_id);

	if (failures == 0) {
		applog(LOG_NOTICE, "ADAM CUDA self-test PASS (%d checks across 17 algos x 70B/32B, V11, V12 factor 1/11/15, target wrap)", checks);
	} else {
		applog(LOG_ERR, "ADAM CUDA self-test FAIL (%d/%d checks failed)", failures, checks);
	}
	return (failures == 0);
}
