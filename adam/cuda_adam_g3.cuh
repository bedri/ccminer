#ifndef CUDA_ADAM_G3_CUH
#define CUDA_ADAM_G3_CUH

#include "cuda_adam_common.cuh"

/* =========================================================================
 * Group 3:
 *   8:  SHAvite-512
 *   9:  SIMD-512
 *   10: ECHO-512
 *   12: Double-SHA256
 * ========================================================================= */

/* Shared LE AES T-table (AES1..3 derived by 8/16/24-bit left rotation) */
static __constant__ uint32_t c_adam_aes0[256] = {
	0xA56363C6U, 0x847C7CF8U, 0x997777EEU, 0x8D7B7BF6U,
	0x0DF2F2FFU, 0xBD6B6BD6U, 0xB16F6FDEU, 0x54C5C591U,
	0x50303060U, 0x03010102U, 0xA96767CEU, 0x7D2B2B56U,
	0x19FEFEE7U, 0x62D7D7B5U, 0xE6ABAB4DU, 0x9A7676ECU,
	0x45CACA8FU, 0x9D82821FU, 0x40C9C989U, 0x877D7DFAU,
	0x15FAFAEFU, 0xEB5959B2U, 0xC947478EU, 0x0BF0F0FBU,
	0xECADAD41U, 0x67D4D4B3U, 0xFDA2A25FU, 0xEAAFAF45U,
	0xBF9C9C23U, 0xF7A4A453U, 0x967272E4U, 0x5BC0C09BU,
	0xC2B7B775U, 0x1CFDFDE1U, 0xAE93933DU, 0x6A26264CU,
	0x5A36366CU, 0x413F3F7EU, 0x02F7F7F5U, 0x4FCCCC83U,
	0x5C343468U, 0xF4A5A551U, 0x34E5E5D1U, 0x08F1F1F9U,
	0x937171E2U, 0x73D8D8ABU, 0x53313162U, 0x3F15152AU,
	0x0C040408U, 0x52C7C795U, 0x65232346U, 0x5EC3C39DU,
	0x28181830U, 0xA1969637U, 0x0F05050AU, 0xB59A9A2FU,
	0x0907070EU, 0x36121224U, 0x9B80801BU, 0x3DE2E2DFU,
	0x26EBEBCDU, 0x6927274EU, 0xCDB2B27FU, 0x9F7575EAU,
	0x1B090912U, 0x9E83831DU, 0x742C2C58U, 0x2E1A1A34U,
	0x2D1B1B36U, 0xB26E6EDCU, 0xEE5A5AB4U, 0xFBA0A05BU,
	0xF65252A4U, 0x4D3B3B76U, 0x61D6D6B7U, 0xCEB3B37DU,
	0x7B292952U, 0x3EE3E3DDU, 0x712F2F5EU, 0x97848413U,
	0xF55353A6U, 0x68D1D1B9U, 0x00000000U, 0x2CEDEDC1U,
	0x60202040U, 0x1FFCFCE3U, 0xC8B1B179U, 0xED5B5BB6U,
	0xBE6A6AD4U, 0x46CBCB8DU, 0xD9BEBE67U, 0x4B393972U,
	0xDE4A4A94U, 0xD44C4C98U, 0xE85858B0U, 0x4ACFCF85U,
	0x6BD0D0BBU, 0x2AEFEFC5U, 0xE5AAAA4FU, 0x16FBFBEDU,
	0xC5434386U, 0xD74D4D9AU, 0x55333366U, 0x94858511U,
	0xCF45458AU, 0x10F9F9E9U, 0x06020204U, 0x817F7FFEU,
	0xF05050A0U, 0x443C3C78U, 0xBA9F9F25U, 0xE3A8A84BU,
	0xF35151A2U, 0xFEA3A35DU, 0xC0404080U, 0x8A8F8F05U,
	0xAD92923FU, 0xBC9D9D21U, 0x48383870U, 0x04F5F5F1U,
	0xDFBCBC63U, 0xC1B6B677U, 0x75DADAAFU, 0x63212142U,
	0x30101020U, 0x1AFFFFE5U, 0x0EF3F3FDU, 0x6DD2D2BFU,
	0x4CCDCD81U, 0x140C0C18U, 0x35131326U, 0x2FECECC3U,
	0xE15F5FBEU, 0xA2979735U, 0xCC444488U, 0x3917172EU,
	0x57C4C493U, 0xF2A7A755U, 0x827E7EFCU, 0x473D3D7AU,
	0xAC6464C8U, 0xE75D5DBAU, 0x2B191932U, 0x957373E6U,
	0xA06060C0U, 0x98818119U, 0xD14F4F9EU, 0x7FDCDCA3U,
	0x66222244U, 0x7E2A2A54U, 0xAB90903BU, 0x8388880BU,
	0xCA46468CU, 0x29EEEEC7U, 0xD3B8B86BU, 0x3C141428U,
	0x79DEDEA7U, 0xE25E5EBCU, 0x1D0B0B16U, 0x76DBDBADU,
	0x3BE0E0DBU, 0x56323264U, 0x4E3A3A74U, 0x1E0A0A14U,
	0xDB494992U, 0x0A06060CU, 0x6C242448U, 0xE45C5CB8U,
	0x5DC2C29FU, 0x6ED3D3BDU, 0xEFACAC43U, 0xA66262C4U,
	0xA8919139U, 0xA4959531U, 0x37E4E4D3U, 0x8B7979F2U,
	0x32E7E7D5U, 0x43C8C88BU, 0x5937376EU, 0xB76D6DDAU,
	0x8C8D8D01U, 0x64D5D5B1U, 0xD24E4E9CU, 0xE0A9A949U,
	0xB46C6CD8U, 0xFA5656ACU, 0x07F4F4F3U, 0x25EAEACFU,
	0xAF6565CAU, 0x8E7A7AF4U, 0xE9AEAE47U, 0x18080810U,
	0xD5BABA6FU, 0x887878F0U, 0x6F25254AU, 0x722E2E5CU,
	0x241C1C38U, 0xF1A6A657U, 0xC7B4B473U, 0x51C6C697U,
	0x23E8E8CBU, 0x7CDDDDA1U, 0x9C7474E8U, 0x211F1F3EU,
	0xDD4B4B96U, 0xDCBDBD61U, 0x868B8B0DU, 0x858A8A0FU,
	0x907070E0U, 0x423E3E7CU, 0xC4B5B571U, 0xAA6666CCU,
	0xD8484890U, 0x05030306U, 0x01F6F6F7U, 0x120E0E1CU,
	0xA36161C2U, 0x5F35356AU, 0xF95757AEU, 0xD0B9B969U,
	0x91868617U, 0x58C1C199U, 0x271D1D3AU, 0xB99E9E27U,
	0x38E1E1D9U, 0x13F8F8EBU, 0xB398982BU, 0x33111122U,
	0xBB6969D2U, 0x70D9D9A9U, 0x898E8E07U, 0xA7949433U,
	0xB69B9B2DU, 0x221E1E3CU, 0x92878715U, 0x20E9E9C9U,
	0x49CECE87U, 0xFF5555AAU, 0x78282850U, 0x7ADFDFA5U,
	0x8F8C8C03U, 0xF8A1A159U, 0x80898909U, 0x170D0D1AU,
	0xDABFBF65U, 0x31E6E6D7U, 0xC6424284U, 0xB86868D0U,
	0xC3414182U, 0xB0999929U, 0x772D2D5AU, 0x110F0F1EU,
	0xCBB0B07BU, 0xFC5454A8U, 0xD6BBBB6DU, 0x3A16162CU
};

__device__ __forceinline__ void adam_aes_round_le(
	uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
	uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3,
	uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3)
{
	y0 = c_adam_aes0[x0 & 0xFF]
		^ adam_rotl32(c_adam_aes0[(x1 >>  8) & 0xFF],  8)
		^ adam_rotl32(c_adam_aes0[(x2 >> 16) & 0xFF], 16)
		^ adam_rotl32(c_adam_aes0[(x3 >> 24) & 0xFF], 24) ^ k0;
	y1 = c_adam_aes0[x1 & 0xFF]
		^ adam_rotl32(c_adam_aes0[(x2 >>  8) & 0xFF],  8)
		^ adam_rotl32(c_adam_aes0[(x3 >> 16) & 0xFF], 16)
		^ adam_rotl32(c_adam_aes0[(x0 >> 24) & 0xFF], 24) ^ k1;
	y2 = c_adam_aes0[x2 & 0xFF]
		^ adam_rotl32(c_adam_aes0[(x3 >>  8) & 0xFF],  8)
		^ adam_rotl32(c_adam_aes0[(x0 >> 16) & 0xFF], 16)
		^ adam_rotl32(c_adam_aes0[(x1 >> 24) & 0xFF], 24) ^ k2;
	y3 = c_adam_aes0[x3 & 0xFF]
		^ adam_rotl32(c_adam_aes0[(x0 >>  8) & 0xFF],  8)
		^ adam_rotl32(c_adam_aes0[(x1 >> 16) & 0xFF], 16)
		^ adam_rotl32(c_adam_aes0[(x2 >> 24) & 0xFF], 24) ^ k3;
}

/* ------------------- 8. SHAvite-512 ------------------- */

static __constant__ uint32_t c_adam_shavite_iv[16] = {
	0x72FCCDD8U, 0x79CA4727U, 0x128A077BU, 0x40D55AECU,
	0xD1901A06U, 0x430AE307U, 0xB29F5CD1U, 0xDF07FBFCU,
	0x8E45D73DU, 0x681AB538U, 0xBDE86578U, 0xDD577E47U,
	0xE275EADEU, 0x502D9FCDU, 0xB9357178U, 0x022A4B9AU
};

__device__ __forceinline__ void adam_shavite_round(
	uint32_t rk[32],
	uint32_t &p0, uint32_t &p1, uint32_t &p2, uint32_t &p3,
	uint32_t &p4, uint32_t &p5, uint32_t &p6, uint32_t &p7,
	uint32_t &p8, uint32_t &p9, uint32_t &pA, uint32_t &pB,
	uint32_t &pC, uint32_t &pD, uint32_t &pE, uint32_t &pF)
{
	uint32_t x0, x1, x2, x3, t;
	x0 = p4 ^ rk[ 0]; x1 = p5 ^ rk[ 1]; x2 = p6 ^ rk[ 2]; x3 = p7 ^ rk[ 3];
	adam_aes_round_le(x0, x1, x2, x3, rk[ 4], rk[ 5], rk[ 6], rk[ 7], x0, x1, x2, x3);
	adam_aes_round_le(x0, x1, x2, x3, rk[ 8], rk[ 9], rk[10], rk[11], x0, x1, x2, x3);
	adam_aes_round_le(x0, x1, x2, x3, rk[12], rk[13], rk[14], rk[15], x0, x1, x2, x3);
	adam_aes_round_le(x0, x1, x2, x3, 0U, 0U, 0U, 0U, x0, x1, x2, x3);
	p0 ^= x0; p1 ^= x1; p2 ^= x2; p3 ^= x3;

	x0 = pC ^ rk[16]; x1 = pD ^ rk[17]; x2 = pE ^ rk[18]; x3 = pF ^ rk[19];
	adam_aes_round_le(x0, x1, x2, x3, rk[20], rk[21], rk[22], rk[23], x0, x1, x2, x3);
	adam_aes_round_le(x0, x1, x2, x3, rk[24], rk[25], rk[26], rk[27], x0, x1, x2, x3);
	adam_aes_round_le(x0, x1, x2, x3, rk[28], rk[29], rk[30], rk[31], x0, x1, x2, x3);
	adam_aes_round_le(x0, x1, x2, x3, 0U, 0U, 0U, 0U, x0, x1, x2, x3);
	p8 ^= x0; p9 ^= x1; pA ^= x2; pB ^= x3;

	t = pC; pC = p8; p8 = p4; p4 = p0; p0 = t;
	t = pD; pD = p9; p9 = p5; p5 = p1; p1 = t;
	t = pE; pE = pA; pA = p6; p6 = p2; p2 = t;
	t = pF; pF = pB; pB = p7; p7 = p3; p3 = t;
}

__device__ void adam_shavite512_compress(const uint32_t msg[32], uint32_t count0, uint32_t out32[8])
{
	uint32_t rk[32];
	#pragma unroll
	for (int i = 0; i < 32; i++)
		rk[i] = msg[i];

	uint32_t p0 = c_adam_shavite_iv[0x0], p1 = c_adam_shavite_iv[0x1];
	uint32_t p2 = c_adam_shavite_iv[0x2], p3 = c_adam_shavite_iv[0x3];
	uint32_t p4 = c_adam_shavite_iv[0x4], p5 = c_adam_shavite_iv[0x5];
	uint32_t p6 = c_adam_shavite_iv[0x6], p7 = c_adam_shavite_iv[0x7];
	uint32_t p8 = c_adam_shavite_iv[0x8], p9 = c_adam_shavite_iv[0x9];
	uint32_t pA = c_adam_shavite_iv[0xA], pB = c_adam_shavite_iv[0xB];
	uint32_t pC = c_adam_shavite_iv[0xC], pD = c_adam_shavite_iv[0xD];
	uint32_t pE = c_adam_shavite_iv[0xE], pF = c_adam_shavite_iv[0xF];

	uint32_t u = 32;
	for (int r = 0; r < 14; r++) {
		adam_shavite_round(rk, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, pA, pB, pC, pD, pE, pF);
		if (r == 13)
			break;
		if ((r & 1) == 0) {
			/* AES key expansion step generating next 32 words into rk[0..31] */
			for (int s = 0; s < 4; s++) {
				uint32_t idx = (uint32_t)(s * 8);
				uint32_t x0, x1, x2, x3;
				uint32_t y0, y1, y2, y3;
				x0 = rk[(idx + 1) & 31];
				x1 = rk[(idx + 2) & 31];
				x2 = rk[(idx + 3) & 31];
				x3 = rk[(idx + 0) & 31];
				adam_aes_round_le(x0, x1, x2, x3, 0U, 0U, 0U, 0U, y0, y1, y2, y3);
				rk[idx + 0] = y0 ^ rk[(idx + 28) & 31];
				rk[idx + 1] = y1 ^ rk[(idx + 29) & 31];
				rk[idx + 2] = y2 ^ rk[(idx + 30) & 31];
				rk[idx + 3] = y3 ^ rk[(idx + 31) & 31];
				if (u == 32) {
					rk[idx + 0] ^= count0;
					rk[idx + 3] ^= 0xFFFFFFFFU;
				} else if (u == 440) {
					rk[idx + 1] ^= count0;
					rk[idx + 3] ^= 0xFFFFFFFFU;
				}
				u += 4;

				idx += 4;
				x0 = rk[(idx + 1) & 31];
				x1 = rk[(idx + 2) & 31];
				x2 = rk[(idx + 3) & 31];
				x3 = rk[(idx + 0) & 31];
				adam_aes_round_le(x0, x1, x2, x3, 0U, 0U, 0U, 0U, y0, y1, y2, y3);
				rk[idx + 0] = y0 ^ rk[(idx - 4) & 31];
				rk[idx + 1] = y1 ^ rk[(idx - 3) & 31];
				rk[idx + 2] = y2 ^ rk[(idx - 2) & 31];
				rk[idx + 3] = y3 ^ rk[(idx - 1) & 31];
				if (u == 164) {
					rk[idx + 3] ^= ~count0;
				} else if (u == 316) {
					rk[idx + 2] ^= count0;
					rk[idx + 3] ^= 0xFFFFFFFFU;
				}
				u += 4;
			}
		} else {
			/* Linear key expansion step */
			for (int s = 0; s < 8; s++) {
				uint32_t idx = (uint32_t)(s * 4);
				rk[idx + 0] ^= rk[(idx + 25) & 31];
				rk[idx + 1] ^= rk[(idx + 26) & 31];
				rk[idx + 2] ^= rk[(idx + 27) & 31];
				rk[idx + 3] ^= rk[(idx + 28) & 31];
				u += 4;
			}
		}
	}

	out32[0] = c_adam_shavite_iv[0] ^ p0;
	out32[1] = c_adam_shavite_iv[1] ^ p1;
	out32[2] = c_adam_shavite_iv[2] ^ p2;
	out32[3] = c_adam_shavite_iv[3] ^ p3;
	out32[4] = c_adam_shavite_iv[4] ^ p4;
	out32[5] = c_adam_shavite_iv[5] ^ p5;
	out32[6] = c_adam_shavite_iv[6] ^ p6;
	out32[7] = c_adam_shavite_iv[7] ^ p7;
}

__global__ void adam_gpu_shavite512_70(uint32_t threads, uint32_t start_nonce, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t msg[32];
	adam_load_le32_70(msg, start_nonce + tid);
	msg[17] |= 0x00800000U;
	for (int i = 18; i < 32; i++) msg[i] = 0;
	msg[27] = 560U << 16;
	msg[31] = 0x02000000U;
	uint32_t out32[8];
	adam_shavite512_compress(msg, 560U, out32);
	adam_store_le32_32(g_hash, tid, out32);
}

__global__ void adam_gpu_shavite512_32(uint32_t threads, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t msg[32];
	adam_load_le32_32(msg, g_hash, tid);
	msg[8] = 0x00000080U;
	for (int i = 9; i < 32; i++) msg[i] = 0;
	msg[27] = 256U << 16;
	msg[31] = 0x02000000U;
	uint32_t out32[8];
	adam_shavite512_compress(msg, 256U, out32);
	adam_store_le32_32(g_hash, tid, out32);
}

/* ------------------- 9. SIMD-512 ------------------- */

static __constant__ uint16_t c_adam_simd_alpha[256] = {
	  1,  41, 139,  45,  46,  87, 226,  14,  60, 147, 116, 130,
	190,  80, 196,  69,   2,  82,  21,  90,  92, 174, 195,  28,
	120,  37, 232,   3, 123, 160, 135, 138,   4, 164,  42, 180,
	184,  91, 133,  56, 240,  74, 207,   6, 246,  63,  13,  19,
	  8,  71,  84, 103, 111, 182,   9, 112, 223, 148, 157,  12,
	235, 126,  26,  38,  16, 142, 168, 206, 222, 107,  18, 224,
	189,  39,  57,  24, 213, 252,  52,  76,  32,  27,  79, 155,
	187, 214,  36, 191, 121,  78, 114,  48, 169, 247, 104, 152,
	 64,  54, 158,  53, 117, 171,  72, 125, 242, 156, 228,  96,
	 81, 237, 208,  47, 128, 108,  59, 106, 234,  85, 144, 250,
	227,  55, 199, 192, 162, 217, 159,  94, 256, 216, 118, 212,
	211, 170,  31, 243, 197, 110, 141, 127,  67, 177,  61, 188,
	255, 175, 236, 167, 165,  83,  62, 229, 137, 220,  25, 254,
	134,  97, 122, 119, 253,  93, 215,  77,  73, 166, 124, 201,
	 17, 183,  50, 251,  11, 194, 244, 238, 249, 186, 173, 154,
	146,  75, 248, 145,  34, 109, 100, 245,  22, 131, 231, 219,
	241, 115,  89,  51,  35, 150, 239,  33,  68, 218, 200, 233,
	 44,   5, 205, 181, 225, 230, 178, 102,  70,  43, 221,  66,
	136, 179, 143, 209,  88,  10, 153, 105, 193, 203,  99, 204,
	140,  86, 185, 132,  15, 101,  29, 161, 176,  20,  49, 210,
	129, 149, 198, 151,  23, 172, 113,   7,  30, 202,  58,  65,
	 95,  40,  98, 163
};

static __constant__ uint16_t c_adam_simd_yoff_n[256] = {
	  1, 163,  98,  40,  95,  65,  58, 202,  30,   7, 113, 172,
	 23, 151, 198, 149, 129, 210,  49,  20, 176, 161,  29, 101,
	 15, 132, 185,  86, 140, 204,  99, 203, 193, 105, 153,  10,
	 88, 209, 143, 179, 136,  66, 221,  43,  70, 102, 178, 230,
	225, 181, 205,   5,  44, 233, 200, 218,  68,  33, 239, 150,
	 35,  51,  89, 115, 241, 219, 231, 131,  22, 245, 100, 109,
	 34, 145, 248,  75, 146, 154, 173, 186, 249, 238, 244, 194,
	 11, 251,  50, 183,  17, 201, 124, 166,  73,  77, 215,  93,
	253, 119, 122,  97, 134, 254,  25, 220, 137, 229,  62,  83,
	165, 167, 236, 175, 255, 188,  61, 177,  67, 127, 141, 110,
	197, 243,  31, 170, 211, 212, 118, 216, 256,  94, 159, 217,
	162, 192, 199,  55, 227, 250, 144,  85, 234, 106,  59, 108,
	128,  47, 208, 237,  81,  96, 228, 156, 242, 125,  72, 171,
	117,  53, 158,  54,  64, 152, 104, 247, 169,  48, 114,  78,
	121, 191,  36, 214, 187, 155,  79,  27,  32,  76,  52, 252,
	213,  24,  57,  39, 189, 224,  18, 107, 222, 206, 168, 142,
	 16,  38,  26, 126, 235,  12, 157, 148, 223, 112,   9, 182,
	111, 103,  84,  71,   8,  19,  13,  63, 246,   6, 207,  74,
	240,  56, 133,  91, 184, 180,  42, 164,   4, 138, 135, 160,
	123,   3, 232,  37, 120,  28, 195, 174,  92,  90,  21,  82,
	  2,  69, 196,  80, 190, 130, 116, 147,  60,  14, 226,  87,
	 46,  45, 139,  41
};

static __constant__ uint16_t c_adam_simd_yoff_f[256] = {
	  2, 203, 156,  47, 118, 214, 107, 106,  45,  93, 212,  20,
	111,  73, 162, 251,  97, 215, 249,  53, 211,  19,   3,  89,
	 49, 207, 101,  67, 151, 130, 223,  23, 189, 202, 178, 239,
	253, 127, 204,  49,  76, 236,  82, 137, 232, 157,  65,  79,
	 96, 161, 176, 130, 161,  30,  47,   9, 189, 247,  61, 226,
	248,  90, 107,  64,   0,  88, 131, 243, 133,  59, 113, 115,
	 17, 236,  33, 213,  12, 191, 111,  19, 251,  61, 103, 208,
	 57,  35, 148, 248,  47, 116,  65, 119, 249, 178, 143,  40,
	189, 129,   8, 163, 204, 227, 230, 196, 205, 122, 151,  45,
	187,  19, 227,  72, 247, 125, 111, 121, 140, 220,   6, 107,
	 77,  69,  10, 101,  21,  65, 149, 171, 255,  54, 101, 210,
	139,  43, 150, 151, 212, 164,  45, 237, 146, 184,  95,   6,
	160,  42,   8, 204,  46, 238, 254, 168, 208,  50, 156, 190,
	106, 127,  34, 234,  68,  55,  79,  18,   4, 130,  53, 208,
	181,  21, 175, 120,  25, 100, 192, 178, 161,  96,  81, 127,
	 96, 227, 210, 248,  68,  10, 196,  31,   9, 167, 150, 193,
	  0, 169, 126,  14, 124, 198, 144, 142, 240,  21, 224,  44,
	245,  66, 146, 238,   6, 196, 154,  49, 200, 222, 109,   9,
	210, 141, 192, 138,   8,  79, 114, 217,  68, 128, 249,  94,
	 53,  30,  27,  61,  52, 135, 106, 212,  70, 238,  30, 185,
	 10, 132, 146, 136, 117,  37, 251, 150, 180, 188, 247, 156,
	236, 192, 108,  86
};

static __constant__ uint32_t c_adam_simd_iv512[32] = {
	0x0BA16B95U, 0x72F999ADU, 0x9FECC2AEU, 0xBA3264FCU,
	0x5E894929U, 0x8E9F30E5U, 0x2F1DAA37U, 0xF0F2C558U,
	0xAC506643U, 0xA90635A5U, 0xE25B878BU, 0xAAB7878FU,
	0x88817F7AU, 0x0A02892BU, 0x559A7550U, 0x598F657EU,
	0x7EEF60A1U, 0x6B70E3E8U, 0x9C1714D1U, 0xB958E2A8U,
	0xAB02675EU, 0xED1C014FU, 0xCD8D65BBU, 0xFDB7A257U,
	0x09254899U, 0xD699C7BCU, 0x9019B6DCU, 0x2B9022E4U,
	0x8FA14956U, 0x21BF9BD3U, 0xB94D0943U, 0x6FFDDC22U
};

#define ADAM_SIMD_REDS1(x) (((x) & 0xFF) - ((x) >> 8))
#define ADAM_SIMD_REDS2(x) (((x) & 0xFFFF) + ((x) >> 16))
#define ADAM_SIMD_INNER(l, h, mm) (((uint32_t)((l) * (mm)) & 0xFFFFU) + ((uint32_t)((h) * (mm)) << 16))
#define ADAM_SIMD_IF(x, y, z)  ((((y) ^ (z)) & (x)) ^ (z))
#define ADAM_SIMD_MAJ(x, y, z) (((x) & (y)) | (((x) | (y)) & (z)))

__device__ __forceinline__ void adam_simd_fft8(const uint8_t *x, uint32_t xb, uint32_t xs, int32_t d[8])
{
	int32_t x0 = x[xb];
	int32_t x1 = x[xb + xs];
	int32_t x2 = x[xb + 2 * xs];
	int32_t x3 = x[xb + 3 * xs];
	int32_t a0 = x0 + x2;
	int32_t a1 = x0 + (x2 << 4);
	int32_t a2 = x0 - x2;
	int32_t a3 = x0 - (x2 << 4);
	int32_t b0 = x1 + x3;
	int32_t b1 = ADAM_SIMD_REDS1((x1 << 2) + (x3 << 6));
	int32_t b2 = (x1 << 4) - (x3 << 4);
	int32_t b3 = ADAM_SIMD_REDS1((x1 << 6) + (x3 << 2));
	d[0] = a0 + b0; d[1] = a1 + b1; d[2] = a2 + b2; d[3] = a3 + b3;
	d[4] = a0 - b0; d[5] = a1 - b1; d[6] = a2 - b2; d[7] = a3 - b3;
}

__device__ __forceinline__ void adam_simd_fft16(const uint8_t *x, uint32_t xb, uint32_t xs, int32_t *q)
{
	int32_t d1[8], d2[8];
	adam_simd_fft8(x, xb, xs << 1, d1);
	adam_simd_fft8(x, xb + xs, xs << 1, d2);
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		q[i]     = d1[i] + (d2[i] << i);
		q[8 + i] = d1[i] - (d2[i] << i);
	}
}

__device__ __forceinline__ void adam_simd_fft_loop(int32_t *q, uint32_t hk, uint32_t as)
{
	int32_t m0 = q[0];
	int32_t n0 = q[hk];
	q[0]  = m0 + n0;
	q[hk] = m0 - n0;
	for (uint32_t u = 1; u < hk; u++) {
		int32_t m = q[u];
		int32_t n = q[u + hk];
		int32_t t = ADAM_SIMD_REDS2(n * (int32_t)c_adam_simd_alpha[u * as]);
		q[u]      = m + t;
		q[u + hk] = m - t;
	}
}

__device__ void adam_simd_fft64(const uint8_t *x, uint32_t xb, uint32_t xs, int32_t *q)
{
	uint32_t xd = xs << 1;
	adam_simd_fft16(x, xb + 0 * xs, xd << 1, &q[0]);
	adam_simd_fft16(x, xb + 2 * xs, xd << 1, &q[16]);
	adam_simd_fft_loop(&q[0], 16, 8);
	adam_simd_fft16(x, xb + 1 * xs, xd << 1, &q[32]);
	adam_simd_fft16(x, xb + 3 * xs, xd << 1, &q[48]);
	adam_simd_fft_loop(&q[32], 16, 8);
	adam_simd_fft_loop(&q[0], 32, 4);
}

__device__ void adam_simd_fft256(const uint8_t *x, int32_t q[256])
{
	adam_simd_fft64(x, 0, 4, &q[0]);
	adam_simd_fft64(x, 2, 4, &q[64]);
	adam_simd_fft_loop(&q[0], 64, 2);
	adam_simd_fft64(x, 1, 4, &q[128]);
	adam_simd_fft64(x, 3, 4, &q[192]);
	adam_simd_fft_loop(&q[128], 64, 2);
	adam_simd_fft_loop(&q[0], 128, 1);
}

__device__ __forceinline__ void adam_simd_step2_big(
	uint32_t state[32], const uint32_t w[8], bool use_maj, int r, int s, uint32_t ppb)
{
	uint32_t tA[8];
	#pragma unroll
	for (int n = 0; n < 8; n++)
		tA[n] = adam_rotl32(state[n], r);
	#pragma unroll
	for (int n = 0; n < 8; n++) {
		uint32_t f = use_maj ? ADAM_SIMD_MAJ(state[n], state[8 + n], state[16 + n])
		                     : ADAM_SIMD_IF(state[n], state[8 + n], state[16 + n]);
		uint32_t tt = state[24 + n] + w[n] + f;
		state[n]      = adam_rotl32(tt, s) + tA[ppb ^ n];
		state[24 + n] = state[16 + n];
		state[16 + n] = state[8 + n];
		state[8 + n]  = tA[n];
	}
}

__device__ void adam_simd_compress_big(uint32_t sc_state[32], const uint32_t buf_u32[32], int last)
{
	int32_t q[256];
	adam_simd_fft256((const uint8_t *)buf_u32, q);
	for (int i = 0; i < 256; i++) {
		int32_t tq = q[i] + (int32_t)(last ? c_adam_simd_yoff_f[i] : c_adam_simd_yoff_n[i]);
		tq = ADAM_SIMD_REDS2(tq);
		tq = ADAM_SIMD_REDS1(tq);
		tq = ADAM_SIMD_REDS1(tq);
		q[i] = (tq <= 128) ? tq : (tq - 257);
	}

	uint32_t state[32];
	#pragma unroll
	for (int i = 0; i < 32; i++)
		state[i] = sc_state[i] ^ buf_u32[i];

	static const uint8_t wbp[32] = {
		 4,  6,  0,  2,  7,  5,  3,  1,
		15, 11, 12,  8,  9, 13, 10, 14,
		17, 18, 23, 20, 22, 21, 16, 19,
		30, 24, 25, 31, 27, 29, 28, 26
	};
	static const uint8_t pp8k[11] = { 1, 6, 2, 3, 5, 7, 4, 1, 6, 2, 3 };
	static const int rot_p[4][4] = {
		{  3, 23, 17, 27 },
		{ 28, 19, 22,  7 },
		{ 29,  9, 15,  5 },
		{  4, 13, 10, 25 }
	};
	static const int32_t o1_tab[4] = { 0, 0, -256, -383 };
	static const int32_t o2_tab[4] = { 1, 1, -128, -255 };
	static const int32_t mm_tab[4] = { 185, 185, 233, 233 };

	for (int rnd = 0; rnd < 4; rnd++) {
		uint32_t w[64];
		int sb = rnd * 8;
		int32_t o1 = o1_tab[rnd], o2 = o2_tab[rnd], mm = mm_tab[rnd];
		for (int g = 0; g < 8; g++) {
			int v = (int)wbp[sb + g] << 4;
			#pragma unroll
			for (int k = 0; k < 8; k++)
				w[g * 8 + k] = ADAM_SIMD_INNER(q[v + 2 * k + o1], q[v + 2 * k + o2], mm);
		}
		adam_simd_step2_big(state, &w[ 0], false, rot_p[rnd][0], rot_p[rnd][1], pp8k[rnd + 0]);
		adam_simd_step2_big(state, &w[ 8], false, rot_p[rnd][1], rot_p[rnd][2], pp8k[rnd + 1]);
		adam_simd_step2_big(state, &w[16], false, rot_p[rnd][2], rot_p[rnd][3], pp8k[rnd + 2]);
		adam_simd_step2_big(state, &w[24], false, rot_p[rnd][3], rot_p[rnd][0], pp8k[rnd + 3]);
		adam_simd_step2_big(state, &w[32], true,  rot_p[rnd][0], rot_p[rnd][1], pp8k[rnd + 4]);
		adam_simd_step2_big(state, &w[40], true,  rot_p[rnd][1], rot_p[rnd][2], pp8k[rnd + 5]);
		adam_simd_step2_big(state, &w[48], true,  rot_p[rnd][2], rot_p[rnd][3], pp8k[rnd + 6]);
		adam_simd_step2_big(state, &w[56], true,  rot_p[rnd][3], rot_p[rnd][0], pp8k[rnd + 7]);
	}

	adam_simd_step2_big(state, &sc_state[ 0], false,  4, 13, 5);
	adam_simd_step2_big(state, &sc_state[ 8], false, 13, 10, 7);
	adam_simd_step2_big(state, &sc_state[16], false, 10, 25, 4);
	adam_simd_step2_big(state, &sc_state[24], false, 25,  4, 1);

	#pragma unroll
	for (int i = 0; i < 32; i++)
		sc_state[i] = state[i];
}

#undef ADAM_SIMD_REDS1
#undef ADAM_SIMD_REDS2
#undef ADAM_SIMD_INNER
#undef ADAM_SIMD_IF
#undef ADAM_SIMD_MAJ

__global__ void adam_gpu_simd512_70(uint32_t threads, uint32_t start_nonce, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t buf[32];
	adam_load_le32_70(buf, start_nonce + tid);
	for (int i = 18; i < 32; i++) buf[i] = 0;

	uint32_t sc_state[32];
	#pragma unroll
	for (int i = 0; i < 32; i++)
		sc_state[i] = c_adam_simd_iv512[i];

	adam_simd_compress_big(sc_state, buf, 0);

	for (int i = 0; i < 32; i++) buf[i] = 0;
	buf[0] = 560U;
	adam_simd_compress_big(sc_state, buf, 1);

	adam_store_le32_32(g_hash, tid, sc_state);
}

__global__ void adam_gpu_simd512_32(uint32_t threads, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t buf[32];
	adam_load_le32_32(buf, g_hash, tid);
	for (int i = 8; i < 32; i++) buf[i] = 0;

	uint32_t sc_state[32];
	#pragma unroll
	for (int i = 0; i < 32; i++)
		sc_state[i] = c_adam_simd_iv512[i];

	adam_simd_compress_big(sc_state, buf, 0);

	for (int i = 0; i < 32; i++) buf[i] = 0;
	buf[0] = 256U;
	adam_simd_compress_big(sc_state, buf, 1);

	adam_store_le32_32(g_hash, tid, sc_state);
}

/* ------------------- 10. ECHO-512 ------------------- */

__device__ __forceinline__ void adam_echo_mix_column(uint32_t W[16][4], int ia, int ib, int ic, int id)
{
	#pragma unroll
	for (int n = 0; n < 4; n++) {
		uint32_t a = W[ia][n], b = W[ib][n], c = W[ic][n], d = W[id][n];
		uint32_t ab = a ^ b, bc = b ^ c, cd = c ^ d;
		uint32_t abx = ((ab & 0x80808080U) >> 7) * 27U ^ ((ab & 0x7F7F7F7FU) << 1);
		uint32_t bcx = ((bc & 0x80808080U) >> 7) * 27U ^ ((bc & 0x7F7F7F7FU) << 1);
		uint32_t cdx = ((cd & 0x80808080U) >> 7) * 27U ^ ((cd & 0x7F7F7F7FU) << 1);
		W[ia][n] = abx ^ bc ^ d;
		W[ib][n] = bcx ^ a ^ cd;
		W[ic][n] = cdx ^ ab ^ d;
		W[id][n] = abx ^ bcx ^ cdx ^ ab ^ c;
	}
}

__device__ void adam_echo512_compress(const uint32_t buf[32], uint32_t count0, uint32_t out32[8])
{
	uint32_t Vs[8][4];
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		Vs[i][0] = 512U;
		Vs[i][1] = 0U;
		Vs[i][2] = 0U;
		Vs[i][3] = 0U;
	}

	uint32_t W[16][4];
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 4; j++)
			W[i][j] = Vs[i][j];
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 4; j++)
			W[8 + i][j] = buf[i * 4 + j];

	uint32_t K0 = count0, K1 = 0U, K2 = 0U, K3 = 0U;
	for (int u = 0; u < 10; u++) {
		for (int n = 0; n < 16; n++) {
			uint32_t y0, y1, y2, y3;
			adam_aes_round_le(W[n][0], W[n][1], W[n][2], W[n][3], K0, K1, K2, K3, y0, y1, y2, y3);
			adam_aes_round_le(y0, y1, y2, y3, 0U, 0U, 0U, 0U, W[n][0], W[n][1], W[n][2], W[n][3]);
			if (++K0 == 0U)
				if (++K1 == 0U)
					if (++K2 == 0U)
						++K3;
		}
		#pragma unroll
		for (int n = 0; n < 4; n++) {
			uint32_t t = W[1][n]; W[1][n] = W[5][n]; W[5][n] = W[9][n]; W[9][n] = W[13][n]; W[13][n] = t;
			uint32_t t0 = W[2][n]; W[2][n] = W[10][n]; W[10][n] = t0;
			uint32_t t1 = W[6][n]; W[6][n] = W[14][n]; W[14][n] = t1;
			t = W[15][n]; W[15][n] = W[11][n]; W[11][n] = W[7][n]; W[7][n] = W[3][n]; W[3][n] = t;
		}
		adam_echo_mix_column(W,  0,  1,  2,  3);
		adam_echo_mix_column(W,  4,  5,  6,  7);
		adam_echo_mix_column(W,  8,  9, 10, 11);
		adam_echo_mix_column(W, 12, 13, 14, 15);
	}

	#pragma unroll
	for (int i = 0; i < 4; i++) {
		out32[i]     = Vs[0][i] ^ buf[i]     ^ W[0][i] ^ W[8][i];
		out32[4 + i] = Vs[1][i] ^ buf[4 + i] ^ W[1][i] ^ W[9][i];
	}
}

__global__ void adam_gpu_echo512_70(uint32_t threads, uint32_t start_nonce, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t buf[32];
	adam_load_le32_70(buf, start_nonce + tid);
	buf[17] |= 0x00800000U;
	for (int i = 18; i < 32; i++) buf[i] = 0;
	buf[27] = 0x02000000U;
	buf[28] = 560U;
	uint32_t out32[8];
	adam_echo512_compress(buf, 560U, out32);
	adam_store_le32_32(g_hash, tid, out32);
}

__global__ void adam_gpu_echo512_32(uint32_t threads, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t buf[32];
	adam_load_le32_32(buf, g_hash, tid);
	buf[8] = 0x00000080U;
	for (int i = 9; i < 32; i++) buf[i] = 0;
	buf[27] = 0x02000000U;
	buf[28] = 256U;
	uint32_t out32[8];
	adam_echo512_compress(buf, 256U, out32);
	adam_store_le32_32(g_hash, tid, out32);
}

/* ------------------- 12. Double-SHA256 ------------------- */

static __constant__ uint32_t c_adam_sha256_k[64] = {
	0x428A2F98U, 0x71374491U, 0xB5C0FBCFU, 0xE9B5DBA5U, 0x3956C25BU, 0x59F111F1U, 0x923F82A4U, 0xAB1C5ED5U,
	0xD807AA98U, 0x12835B01U, 0x243185BEU, 0x550C7DC3U, 0x72BE5D74U, 0x80DEB1FEU, 0x9BDC06A7U, 0xC19BF174U,
	0xE49B69C1U, 0xEFBE4786U, 0x0FC19DC6U, 0x240CA1CCU, 0x2DE92C6FU, 0x4A7484AAU, 0x5CB0A9DCU, 0x76F988DAU,
	0x983E5152U, 0xA831C66DU, 0xB00327C8U, 0xBF597FC7U, 0xC6E00BF3U, 0xD5A79147U, 0x06CA6351U, 0x14292967U,
	0x27B70A85U, 0x2E1B2138U, 0x4D2C6DFCU, 0x53380D13U, 0x650A7354U, 0x766A0ABBU, 0x81C2C92EU, 0x92722C85U,
	0xA2BFE8A1U, 0xA81A664BU, 0xC24B8B70U, 0xC76C51A3U, 0xD192E819U, 0xD6990624U, 0xF40E3585U, 0x106AA070U,
	0x19A4C116U, 0x1E376C08U, 0x2748774CU, 0x34B0BCB5U, 0x391C0CB3U, 0x4ED8AA4AU, 0x5B9CCA4FU, 0x682E6FF3U,
	0x748F82EEU, 0x78A5636FU, 0x84C87814U, 0x8CC70208U, 0x90BEFFFAU, 0xA4506CEBU, 0xBEF9A3F7U, 0xC67178F2U
};

__device__ void adam_sha256_block(uint32_t state[8], const uint32_t block_be[16])
{
	uint32_t w[16];
	#pragma unroll
	for (int i = 0; i < 16; i++)
		w[i] = block_be[i];

	uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
	uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

	for (int i = 0; i < 64; i++) {
		uint32_t wi;
		if (i < 16) {
			wi = w[i];
		} else {
			uint32_t i15 = w[(i - 15) & 15];
			uint32_t i2  = w[(i -  2) & 15];
			uint32_t s0  = adam_rotr32(i15, 7) ^ adam_rotr32(i15, 18) ^ (i15 >> 3);
			uint32_t s1  = adam_rotr32(i2, 17) ^ adam_rotr32(i2,  19) ^ (i2  >> 10);
			wi = w[i & 15] = w[i & 15] + s0 + w[(i - 7) & 15] + s1;
		}
		uint32_t S1  = adam_rotr32(e, 6) ^ adam_rotr32(e, 11) ^ adam_rotr32(e, 25);
		uint32_t ch  = (e & f) ^ ((~e) & g);
		uint32_t t1  = h + S1 + ch + c_adam_sha256_k[i] + wi;
		uint32_t S0  = adam_rotr32(a, 2) ^ adam_rotr32(a, 13) ^ adam_rotr32(a, 22);
		uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
		uint32_t t2  = S0 + maj;

		h = g; g = f; f = e; e = d + t1;
		d = c; c = b; b = a; a = t1 + t2;
	}

	state[0] += a; state[1] += b; state[2] += c; state[3] += d;
	state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ __forceinline__ void adam_sha256_init(uint32_t s[8])
{
	s[0] = 0x6A09E667U; s[1] = 0xBB67AE85U; s[2] = 0x3C6EF372U; s[3] = 0xA54FF53AU;
	s[4] = 0x510E527FU; s[5] = 0x9B05688CU; s[6] = 0x1F83D9ABU; s[7] = 0x5BE0CD19U;
}

__device__ void adam_sha256_second_pass(uint32_t state[8], uint32_t out32[8])
{
	uint32_t b[16];
	#pragma unroll
	for (int i = 0; i < 8; i++)
		b[i] = state[i];
	b[8] = 0x80000000U;
	for (int i = 9; i < 15; i++) b[i] = 0U;
	b[15] = 256U;

	adam_sha256_init(state);
	adam_sha256_block(state, b);

	#pragma unroll
	for (int i = 0; i < 8; i++)
		out32[i] = adam_bswap32(state[i]);
}

__global__ void adam_gpu_sha256d_70(uint32_t threads, uint32_t start_nonce, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t w[18];
	adam_load_le32_70(w, start_nonce + tid);

	uint32_t s[8];
	adam_sha256_init(s);

	uint32_t b0[16];
	#pragma unroll
	for (int i = 0; i < 16; i++)
		b0[i] = adam_bswap32(w[i]);
	adam_sha256_block(s, b0);

	uint32_t b1[16];
	b1[0] = adam_bswap32(w[16]);
	b1[1] = adam_bswap32(w[17] | 0x00800000U);
	for (int i = 2; i < 15; i++) b1[i] = 0U;
	b1[15] = 560U;
	adam_sha256_block(s, b1);

	uint32_t out32[8];
	adam_sha256_second_pass(s, out32);
	adam_store_le32_32(g_hash, tid, out32);
}

__global__ void adam_gpu_sha256d_32(uint32_t threads, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t w[8];
	adam_load_le32_32(w, g_hash, tid);

	uint32_t s[8];
	adam_sha256_init(s);

	uint32_t b0[16];
	#pragma unroll
	for (int i = 0; i < 8; i++)
		b0[i] = adam_bswap32(w[i]);
	b0[8] = 0x80000000U;
	for (int i = 9; i < 15; i++) b0[i] = 0U;
	b0[15] = 256U;
	adam_sha256_block(s, b0);

	uint32_t out32[8];
	adam_sha256_second_pass(s, out32);
	adam_store_le32_32(g_hash, tid, out32);
}

#endif
