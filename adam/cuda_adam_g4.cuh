#ifndef CUDA_ADAM_G4_CUH
#define CUDA_ADAM_G4_CUH

#include "cuda_adam_common.cuh"

/* =========================================================================
 * Group 4:
 *   13: Hamsi-512
 *   14: Fugue-512
 *   15: Shabal-512
 *   16: Whirlpool
 *   17: HAVAL-256/5
 * ========================================================================= */

/* ------------------- 13. Hamsi-512 ------------------- */

static __constant__ uint32_t c_adam_hamsi_alpha_n[32] = {
	0xff00f0f0U, 0xccccaaaaU, 0xf0f0ccccU, 0xff00aaaaU, 0xccccaaaaU, 0xf0f0ff00U, 0xaaaaccccU, 0xf0f0ff00U,
	0xf0f0ccccU, 0xaaaaff00U, 0xccccff00U, 0xaaaaf0f0U, 0xaaaaf0f0U, 0xff00ccccU, 0xccccf0f0U, 0xff00aaaaU,
	0xccccaaaaU, 0xff00f0f0U, 0xff00aaaaU, 0xf0f0ccccU, 0xf0f0ff00U, 0xccccaaaaU, 0xf0f0ff00U, 0xaaaaccccU,
	0xaaaaff00U, 0xf0f0ccccU, 0xaaaaf0f0U, 0xccccff00U, 0xff00ccccU, 0xaaaaf0f0U, 0xff00aaaaU, 0xccccf0f0U
};

static __constant__ uint32_t c_adam_hamsi_alpha_f[32] = {
	0xcaf9639cU, 0x0ff0f9c0U, 0x639c0ff0U, 0xcaf9f9c0U, 0x0ff0f9c0U, 0x639ccaf9U, 0xf9c00ff0U, 0x639ccaf9U,
	0x639c0ff0U, 0xf9c0caf9U, 0x0ff0caf9U, 0xf9c0639cU, 0xf9c0639cU, 0xcaf90ff0U, 0x0ff0639cU, 0xcaf9f9c0U,
	0x0ff0f9c0U, 0xcaf9639cU, 0xcaf9f9c0U, 0x639c0ff0U, 0x639ccaf9U, 0x0ff0f9c0U, 0x639ccaf9U, 0xf9c00ff0U,
	0xf9c0caf9U, 0x639c0ff0U, 0xf9c0639cU, 0x0ff0caf9U, 0xcaf90ff0U, 0xf9c0639cU, 0xcaf9f9c0U, 0x0ff0639cU
};

static __constant__ uint32_t c_adam_hamsi_T512[64][16] = {
{ 0xef0b0270U, 0x3afd0000U, 0x5dae0000U, 0x69490000U, 0x9b0f3c06U, 0x4405b5f9U, 0x66140a51U, 0x924f5d0aU,
  0xc96b0030U, 0xe7250000U, 0x2f840000U, 0x264f0000U, 0x08695bf9U, 0x6dfcf137U, 0x509f6984U, 0x9e69af68U },
{ 0xc96b0030U, 0xe7250000U, 0x2f840000U, 0x264f0000U, 0x08695bf9U, 0x6dfcf137U, 0x509f6984U, 0x9e69af68U,
  0x26600240U, 0xddd80000U, 0x722a0000U, 0x4f060000U, 0x936667ffU, 0x29f944ceU, 0x368b63d5U, 0x0c26f262U },
{ 0x145a3c00U, 0xb9e90000U, 0x61270000U, 0xf1610000U, 0xce613d6cU, 0xb0493d78U, 0x47a96720U, 0xe18e24c5U,
  0x23671400U, 0xc8b90000U, 0xf4c70000U, 0xfb750000U, 0x73cd2465U, 0xf8a6a549U, 0x02c40a3fU, 0xdc24e61fU },
{ 0x23671400U, 0xc8b90000U, 0xf4c70000U, 0xfb750000U, 0x73cd2465U, 0xf8a6a549U, 0x02c40a3fU, 0xdc24e61fU,
  0x373d2800U, 0x71500000U, 0x95e00000U, 0x0a140000U, 0xbdac1909U, 0x48ef9831U, 0x456d6d1fU, 0x3daac2daU },
{ 0x54285c00U, 0xeaed0000U, 0xc5d60000U, 0xa1c50000U, 0xb3a26770U, 0x94a5c4e1U, 0x6bb0419dU, 0x551b3782U,
  0x9cbb1800U, 0xb0d30000U, 0x92510000U, 0xed930000U, 0x593a4345U, 0xe114d5f4U, 0x430633daU, 0x78cace29U },
{ 0x9cbb1800U, 0xb0d30000U, 0x92510000U, 0xed930000U, 0x593a4345U, 0xe114d5f4U, 0x430633daU, 0x78cace29U,
  0xc8934400U, 0x5a3e0000U, 0x57870000U, 0x4c560000U, 0xea982435U, 0x75b11115U, 0x28b67247U, 0x2dd1f9abU },
{ 0x29449c00U, 0x64e70000U, 0xf24b0000U, 0xc2f30000U, 0x0ede4e8fU, 0x56c23745U, 0xf3e04259U, 0x8d0d9ec4U,
  0x466d0c00U, 0x08620000U, 0xdd5d0000U, 0xbadd0000U, 0x6a927942U, 0x441f2b93U, 0x218ace6fU, 0xbf2c0be2U },
{ 0x466d0c00U, 0x08620000U, 0xdd5d0000U, 0xbadd0000U, 0x6a927942U, 0x441f2b93U, 0x218ace6fU, 0xbf2c0be2U,
  0x6f299000U, 0x6c850000U, 0x2f160000U, 0x782e0000U, 0x644c37cdU, 0x12dd1cd6U, 0xd26a8c36U, 0x32219526U },
{ 0xf6800005U, 0x3443c000U, 0x24070000U, 0x8f3d0000U, 0x21373bfbU, 0x0ab8d5aeU, 0xcdc58b19U, 0xd795ba31U,
  0xa67f0001U, 0x71378000U, 0x19fc0000U, 0x96db0000U, 0x3a8b6dfdU, 0xebcaaef3U, 0x2c6d478fU, 0xac8e6c88U },
{ 0xa67f0001U, 0x71378000U, 0x19fc0000U, 0x96db0000U, 0x3a8b6dfdU, 0xebcaaef3U, 0x2c6d478fU, 0xac8e6c88U,
  0x50ff0004U, 0x45744000U, 0x3dfb0000U, 0x19e60000U, 0x1bbc5606U, 0xe1727b5dU, 0xe1a8cc96U, 0x7b1bd6b9U },
{ 0xf7750009U, 0xcf3cc000U, 0xc3d60000U, 0x04920000U, 0x029519a9U, 0xf8e836baU, 0x7a87f14eU, 0x9e16981aU,
  0xd46a0000U, 0x8dc8c000U, 0xa5af0000U, 0x4a290000U, 0xfc4e427aU, 0xc9b4866cU, 0x98369604U, 0xf746c320U },
{ 0xd46a0000U, 0x8dc8c000U, 0xa5af0000U, 0x4a290000U, 0xfc4e427aU, 0xc9b4866cU, 0x98369604U, 0xf746c320U,
  0x231f0009U, 0x42f40000U, 0x66790000U, 0x4ebb0000U, 0xfedb5bd3U, 0x315cb0d6U, 0xe2b1674aU, 0x69505b3aU },
{ 0x774400f0U, 0xf15a0000U, 0xf5b20000U, 0x34140000U, 0x89377e8cU, 0x5a8bec25U, 0x0bc3cd1eU, 0xcf3775cbU,
  0xf46c0050U, 0x96180000U, 0x14a50000U, 0x031f0000U, 0x42947eb8U, 0x66bf7e19U, 0x9ca470d2U, 0x8a341574U },
{ 0xf46c0050U, 0x96180000U, 0x14a50000U, 0x031f0000U, 0x42947eb8U, 0x66bf7e19U, 0x9ca470d2U, 0x8a341574U,
  0x832800a0U, 0x67420000U, 0xe1170000U, 0x370b0000U, 0xcba30034U, 0x3c34923cU, 0x9767bdccU, 0x450360bfU },
{ 0xe8870170U, 0x9d720000U, 0x12db0000U, 0xd4220000U, 0xf2886b27U, 0xa921e543U, 0x4ef8b518U, 0x618813b1U,
  0xb4370060U, 0x0c4c0000U, 0x56c20000U, 0x5cae0000U, 0x94541f3fU, 0x3b3ef825U, 0x1b365f3dU, 0xf3d45758U },
{ 0xb4370060U, 0x0c4c0000U, 0x56c20000U, 0x5cae0000U, 0x94541f3fU, 0x3b3ef825U, 0x1b365f3dU, 0xf3d45758U,
  0x5cb00110U, 0x913e0000U, 0x44190000U, 0x888c0000U, 0x66dc7418U, 0x921f1d66U, 0x55ceea25U, 0x925c44e9U },
{ 0x0c720000U, 0x49e50f00U, 0x42790000U, 0x5cea0000U, 0x33aa301aU, 0x15822514U, 0x95a34b7bU, 0xb44b0090U,
  0xfe220000U, 0xa7580500U, 0x25d10000U, 0xf7600000U, 0x893178daU, 0x1fd4f860U, 0x4ed0a315U, 0xa123ff9fU },
{ 0xfe220000U, 0xa7580500U, 0x25d10000U, 0xf7600000U, 0x893178daU, 0x1fd4f860U, 0x4ed0a315U, 0xa123ff9fU,
  0xf2500000U, 0xeebd0a00U, 0x67a80000U, 0xab8a0000U, 0xba9b48c0U, 0x0a56dd74U, 0xdb73e86eU, 0x1568ff0fU },
{ 0x45180000U, 0xa5b51700U, 0xf96a0000U, 0x3b480000U, 0x1ecc142cU, 0x231395d6U, 0x16bca6b0U, 0xdf33f4dfU,
  0xb83d0000U, 0x16710600U, 0x379a0000U, 0xf5b10000U, 0x228161acU, 0xae48f145U, 0x66241616U, 0xc5c1eb3eU },
{ 0xb83d0000U, 0x16710600U, 0x379a0000U, 0xf5b10000U, 0x228161acU, 0xae48f145U, 0x66241616U, 0xc5c1eb3eU,
  0xfd250000U, 0xb3c41100U, 0xcef00000U, 0xcef90000U, 0x3c4d7580U, 0x8d5b6493U, 0x7098b0a6U, 0x1af21fe1U },
{ 0x75a40000U, 0xc28b2700U, 0x94a40000U, 0x90f50000U, 0xfb7857e0U, 0x49ce0baeU, 0x1767c483U, 0xaedf667eU,
  0xd1660000U, 0x1bbc0300U, 0x9eec0000U, 0xf6940000U, 0x03024527U, 0xcf70fcf2U, 0xb4431b17U, 0x857f3c2bU },
{ 0xd1660000U, 0x1bbc0300U, 0x9eec0000U, 0xf6940000U, 0x03024527U, 0xcf70fcf2U, 0xb4431b17U, 0x857f3c2bU,
  0xa4c20000U, 0xd9372400U, 0x0a480000U, 0x66610000U, 0xf87a12c7U, 0x86bef75cU, 0xa324df94U, 0x2ba05a55U },
{ 0x75c90003U, 0x0e10c000U, 0xd1200000U, 0xbaea0000U, 0x8bc42f3eU, 0x8758b757U, 0xbb28761dU, 0x00b72e2bU,
  0xeecf0001U, 0x6f564000U, 0xf33e0000U, 0xa79e0000U, 0xbdb57219U, 0xb711ebc5U, 0x4a3b40baU, 0xfeabf254U },
{ 0xeecf0001U, 0x6f564000U, 0xf33e0000U, 0xa79e0000U, 0xbdb57219U, 0xb711ebc5U, 0x4a3b40baU, 0xfeabf254U,
  0x9b060002U, 0x61468000U, 0x221e0000U, 0x1d740000U, 0x36715d27U, 0x30495c92U, 0xf11336a7U, 0xfe1cdc7fU },
{ 0x86790000U, 0x3f390002U, 0xe19ae000U, 0x98560000U, 0x9565670eU, 0x4e88c8eaU, 0xd3dd4944U, 0x161ddab9U,
  0x30b70000U, 0xe5d00000U, 0xf4f46000U, 0x42c40000U, 0x63b83d6aU, 0x78ba9460U, 0x21afa1eaU, 0xb0a51834U },
{ 0x30b70000U, 0xe5d00000U, 0xf4f46000U, 0x42c40000U, 0x63b83d6aU, 0x78ba9460U, 0x21afa1eaU, 0xb0a51834U,
  0xb6ce0000U, 0xdae90002U, 0x156e8000U, 0xda920000U, 0xf6dd5a64U, 0x36325c8aU, 0xf272e8aeU, 0xa6b8c28dU },
{ 0x14190000U, 0x23ca003cU, 0x50df0000U, 0x44b60000U, 0x1b6c67b0U, 0x3cf3ac75U, 0x61e610b0U, 0xdbcadb80U,
  0xe3430000U, 0x3a4e0014U, 0xf2c60000U, 0xaa4e0000U, 0xdb1e42a6U, 0x256bbe15U, 0x123db156U, 0x3a4e99d7U },
{ 0xe3430000U, 0x3a4e0014U, 0xf2c60000U, 0xaa4e0000U, 0xdb1e42a6U, 0x256bbe15U, 0x123db156U, 0x3a4e99d7U,
  0xf75a0000U, 0x19840028U, 0xa2190000U, 0xeef80000U, 0xc0722516U, 0x19981260U, 0x73dba1e6U, 0xe1844257U },
{ 0x54500000U, 0x0671005cU, 0x25ae0000U, 0x6a1e0000U, 0x2ea54edfU, 0x664e8512U, 0xbfba18c3U, 0x7e715d17U,
  0xbc8d0000U, 0xfc3b0018U, 0x19830000U, 0xd10b0000U, 0xae1878c4U, 0x42a69856U, 0x0012da37U, 0x2c3b504eU },
{ 0xbc8d0000U, 0xfc3b0018U, 0x19830000U, 0xd10b0000U, 0xae1878c4U, 0x42a69856U, 0x0012da37U, 0x2c3b504eU,
  0xe8dd0000U, 0xfa4a0044U, 0x3c2d0000U, 0xbb150000U, 0x80bd361bU, 0x24e81d44U, 0xbfa8c2f4U, 0x524a0d59U },
{ 0x69510000U, 0xd4e1009cU, 0xc3230000U, 0xac2f0000U, 0xe4950baeU, 0xcea415dcU, 0x87ec287cU, 0xbce1a3ceU,
  0xc6730000U, 0xaf8d000cU, 0xa4c10000U, 0x218d0000U, 0x23111587U, 0x7913512fU, 0x1d28ac88U, 0x378dd173U },
{ 0xc6730000U, 0xaf8d000cU, 0xa4c10000U, 0x218d0000U, 0x23111587U, 0x7913512fU, 0x1d28ac88U, 0x378dd173U,
  0xaf220000U, 0x7b6c0090U, 0x67e20000U, 0x8da20000U, 0xc7841e29U, 0xb7b744f3U, 0x9ac484f4U, 0x8b6c72bdU },
{ 0xcc140000U, 0xa5630000U, 0x5ab90780U, 0x3b500000U, 0x4bd013ffU, 0x879b3418U, 0x694348c1U, 0xca5a87feU,
  0x819e0000U, 0xec570000U, 0x66320280U, 0x95f30000U, 0x5da92802U, 0x48f43cbcU, 0xe65aa22dU, 0x8e67b7faU },
{ 0x819e0000U, 0xec570000U, 0x66320280U, 0x95f30000U, 0x5da92802U, 0x48f43cbcU, 0xe65aa22dU, 0x8e67b7faU,
  0x4d8a0000U, 0x49340000U, 0x3c8b0500U, 0xaea30000U, 0x16793bfdU, 0xcf6f08a4U, 0x8f19eaecU, 0x443d3004U },
{ 0x78230000U, 0x12fc0000U, 0xa93a0b80U, 0x90a50000U, 0x713e2879U, 0x7ee98924U, 0xf08ca062U, 0x636f8babU,
  0x02af0000U, 0xb7280000U, 0xba1c0300U, 0x56980000U, 0xba8d45d3U, 0x8048c667U, 0xa95c149aU, 0xf4f6ea7bU },
{ 0x02af0000U, 0xb7280000U, 0xba1c0300U, 0x56980000U, 0xba8d45d3U, 0x8048c667U, 0xa95c149aU, 0xf4f6ea7bU,
  0x7a8c0000U, 0xa5d40000U, 0x13260880U, 0xc63d0000U, 0xcbb36daaU, 0xfea14f43U, 0x59d0b4f8U, 0x979961d0U },
{ 0xac480000U, 0x1ba60000U, 0x45fb1380U, 0x03430000U, 0x5a85316aU, 0x1fb250b6U, 0xfe72c7feU, 0x91e478f6U,
  0x1e4e0000U, 0xdecf0000U, 0x6df80180U, 0x77240000U, 0xec47079eU, 0xf4a0694eU, 0xcda31812U, 0x98aa496eU },
{ 0x1e4e0000U, 0xdecf0000U, 0x6df80180U, 0x77240000U, 0xec47079eU, 0xf4a0694eU, 0xcda31812U, 0x98aa496eU,
  0xb2060000U, 0xc5690000U, 0x28031200U, 0x74670000U, 0xb6c236f4U, 0xeb1239f8U, 0x33d1dfecU, 0x094e3198U },
{ 0xaec30000U, 0x9c4f0001U, 0x79d1e000U, 0x2c150000U, 0x45cc75b3U, 0x6650b736U, 0xab92f78fU, 0xa312567bU,
  0xdb250000U, 0x09290000U, 0x49aac000U, 0x81e10000U, 0xcafe6b59U, 0x42793431U, 0x43566b76U, 0xe86cba2eU },
{ 0xdb250000U, 0x09290000U, 0x49aac000U, 0x81e10000U, 0xcafe6b59U, 0x42793431U, 0x43566b76U, 0xe86cba2eU,
  0x75e60000U, 0x95660001U, 0x307b2000U, 0xadf40000U, 0x8f321eeaU, 0x24298307U, 0xe8c49cf9U, 0x4b7eec55U },
{ 0x58430000U, 0x807e0000U, 0x78330001U, 0xc66b3800U, 0xe7375cdcU, 0x79ad3fddU, 0xac73fe6fU, 0x3a4479b1U,
  0x1d5a0000U, 0x2b720000U, 0x488d0000U, 0xaf611800U, 0x25cb2ec5U, 0xc879bfd0U, 0x81a20429U, 0x1e7536a6U },
{ 0x1d5a0000U, 0x2b720000U, 0x488d0000U, 0xaf611800U, 0x25cb2ec5U, 0xc879bfd0U, 0x81a20429U, 0x1e7536a6U,
  0x45190000U, 0xab0c0000U, 0x30be0001U, 0x690a2000U, 0xc2fc7219U, 0xb1d4800dU, 0x2dd1fa46U, 0x24314f17U },
{ 0xa53b0000U, 0x14260000U, 0x4e30001eU, 0x7cae0000U, 0x8f9e0dd5U, 0x78dfaa3dU, 0xf73168d8U, 0x0b1b4946U,
  0x07ed0000U, 0xb2500000U, 0x8774000aU, 0x970d0000U, 0x437223aeU, 0x48c76ea4U, 0xf4786222U, 0x9075b1ceU },
{ 0x07ed0000U, 0xb2500000U, 0x8774000aU, 0x970d0000U, 0x437223aeU, 0x48c76ea4U, 0xf4786222U, 0x9075b1ceU,
  0xa2d60000U, 0xa6760000U, 0xc9440014U, 0xeba30000U, 0xccec2e7bU, 0x3018c499U, 0x03490afaU, 0x9b6ef888U },
{ 0x88980000U, 0x1f940000U, 0x7fcf002eU, 0xfb4e0000U, 0xf158079aU, 0x61ae9167U, 0xa895706cU, 0xe6107494U,
  0x0bc20000U, 0xdb630000U, 0x7e88000cU, 0x15860000U, 0x91fd48f3U, 0x7581bb43U, 0xf460449eU, 0xd8b61463U },
{ 0x0bc20000U, 0xdb630000U, 0x7e88000cU, 0x15860000U, 0x91fd48f3U, 0x7581bb43U, 0xf460449eU, 0xd8b61463U,
  0x835a0000U, 0xc4f70000U, 0x01470022U, 0xeec80000U, 0x60a54f69U, 0x142f2a24U, 0x5cf534f2U, 0x3ea660f7U },
{ 0x52500000U, 0x29540000U, 0x6a61004eU, 0xf0ff0000U, 0x9a317eecU, 0x452341ceU, 0xcf568fe5U, 0x5303130fU,
  0x538d0000U, 0xa9fc0000U, 0x9ef70006U, 0x56ff0000U, 0x0ae4004eU, 0x92c5cdf9U, 0xa9444018U, 0x7f975691U },
{ 0x538d0000U, 0xa9fc0000U, 0x9ef70006U, 0x56ff0000U, 0x0ae4004eU, 0x92c5cdf9U, 0xa9444018U, 0x7f975691U,
  0x01dd0000U, 0x80a80000U, 0xf4960048U, 0xa6000000U, 0x90d57ea2U, 0xd7e68c37U, 0x6612cffdU, 0x2c94459eU },
{ 0xe6280000U, 0x4c4b0000U, 0xa8550000U, 0xd3d002e0U, 0xd86130b8U, 0x98a7b0daU, 0x289506b4U, 0xd75a4897U,
  0xf0c50000U, 0x59230000U, 0x45820000U, 0xe18d00c0U, 0x3b6d0631U, 0xc2ed5699U, 0xcbe0fe1cU, 0x56a7b19fU },
{ 0xf0c50000U, 0x59230000U, 0x45820000U, 0xe18d00c0U, 0x3b6d0631U, 0xc2ed5699U, 0xcbe0fe1cU, 0x56a7b19fU,
  0x16ed0000U, 0x15680000U, 0xedd70000U, 0x325d0220U, 0xe30c3689U, 0x5a4ae643U, 0xe375f8a8U, 0x81fdf908U },
{ 0xb4310000U, 0x77330000U, 0xb15d0000U, 0x7fd004e0U, 0x78a26138U, 0xd116c35dU, 0xd256d489U, 0x4e6f74deU,
  0xe3060000U, 0xbdc10000U, 0x87130000U, 0xbff20060U, 0x2eba0a1aU, 0x8db53751U, 0x73c5ab06U, 0x5bd61539U },
{ 0xe3060000U, 0xbdc10000U, 0x87130000U, 0xbff20060U, 0x2eba0a1aU, 0x8db53751U, 0x73c5ab06U, 0x5bd61539U,
  0x57370000U, 0xcaf20000U, 0x364e0000U, 0xc0220480U, 0x56186b22U, 0x5ca3f40cU, 0xa1937f8fU, 0x15b961e7U },
{ 0x02f20000U, 0xa2810000U, 0x873f0000U, 0xe36c7800U, 0x1e1d74efU, 0x073d2bd6U, 0xc4c23237U, 0x7f32259eU,
  0xbadd0000U, 0x13ad0000U, 0xb7e70000U, 0xf7282800U, 0xdf45144dU, 0x361ac33aU, 0xea5a8d14U, 0x2a2c18f0U },
{ 0xbadd0000U, 0x13ad0000U, 0xb7e70000U, 0xf7282800U, 0xdf45144dU, 0x361ac33aU, 0xea5a8d14U, 0x2a2c18f0U,
  0xb82f0000U, 0xb12c0000U, 0x30d80000U, 0x14445000U, 0xc15860a2U, 0x3127e8ecU, 0x2e98bf23U, 0x551e3d6eU },
{ 0x1e6c0000U, 0xc4420000U, 0x8a2e0000U, 0xbcb6b800U, 0x2c4413b6U, 0x8bfdd3daU, 0x6a0c1bc8U, 0xb99dc2ebU,
  0x92560000U, 0x1eda0000U, 0xea510000U, 0xe8b13000U, 0xa93556a5U, 0xebfb6199U, 0xb15c2254U, 0x33c5244fU },
{ 0x92560000U, 0x1eda0000U, 0xea510000U, 0xe8b13000U, 0xa93556a5U, 0xebfb6199U, 0xb15c2254U, 0x33c5244fU,
  0x8c3a0000U, 0xda980000U, 0x607f0000U, 0x54078800U, 0x85714513U, 0x6006b243U, 0xdb50399cU, 0x8a58e6a4U },
{ 0x033d0000U, 0x08b30000U, 0xf33a0000U, 0x3ac20007U, 0x51298a50U, 0x6b6e661fU, 0x0ea5cfe3U, 0xe6da7ffeU,
  0xa8da0000U, 0x96be0000U, 0x5c1d0000U, 0x07da0002U, 0x7d669583U, 0x1f98708aU, 0xbb668808U, 0xda878000U },
{ 0xa8da0000U, 0x96be0000U, 0x5c1d0000U, 0x07da0002U, 0x7d669583U, 0x1f98708aU, 0xbb668808U, 0xda878000U,
  0xabe70000U, 0x9e0d0000U, 0xaf270000U, 0x3d180005U, 0x2c4f1fd3U, 0x74f61695U, 0xb5c347ebU, 0x3c5dfffeU },
{ 0x01930000U, 0xe7820000U, 0xedfb0000U, 0xcf0c000bU, 0x8dd08d58U, 0xbca3b42eU, 0x063661e1U, 0x536f9e7bU,
  0x92280000U, 0xdc850000U, 0x57fa0000U, 0x56dc0003U, 0xbae92316U, 0x5aefa30cU, 0x90cef752U, 0x7b1675d7U },
{ 0x92280000U, 0xdc850000U, 0x57fa0000U, 0x56dc0003U, 0xbae92316U, 0x5aefa30cU, 0x90cef752U, 0x7b1675d7U,
  0x93bb0000U, 0x3b070000U, 0xba010000U, 0x99d00008U, 0x3739ae4eU, 0xe64c1722U, 0x96f896b3U, 0x2879ebacU },
{ 0x5fa80000U, 0x56030000U, 0x43ae0000U, 0x64f30013U, 0x257e86bfU, 0x1311944eU, 0x541e95bfU, 0x8ea4db69U,
  0x00440000U, 0x7f480000U, 0xda7c0000U, 0x2a230001U, 0x3badc9ccU, 0xa9b69c87U, 0x030a9e60U, 0xbe0a679eU },
{ 0x00440000U, 0x7f480000U, 0xda7c0000U, 0x2a230001U, 0x3badc9ccU, 0xa9b69c87U, 0x030a9e60U, 0xbe0a679eU,
  0x5fec0000U, 0x294b0000U, 0x99d20000U, 0x4ed00012U, 0x1ed34f73U, 0xbaa708c9U, 0x57140bdfU, 0x30aebcf7U },
{ 0xee930000U, 0xd6070000U, 0x92c10000U, 0x2b9801e0U, 0x9451287cU, 0x3b6cfb57U, 0x45312374U, 0x201f6a64U,
  0x7b280000U, 0x57420000U, 0xa9e50000U, 0x634300a0U, 0x9edb442fU, 0x6d9995bbU, 0x27f83b03U, 0xc7ff60f0U },
{ 0x7b280000U, 0x57420000U, 0xa9e50000U, 0x634300a0U, 0x9edb442fU, 0x6d9995bbU, 0x27f83b03U, 0xc7ff60f0U,
  0x95bb0000U, 0x81450000U, 0x3b240000U, 0x48db0140U, 0x0a8a6c53U, 0x56f56eecU, 0x62c91877U, 0xe7e00a94U }
};

__device__ __forceinline__ void adam_hamsi_sbox(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d)
{
	uint32_t t = a;
	a &= c; a ^= d; c ^= b; c ^= a; d |= t; d ^= b; t ^= c; b = d;
	d |= t; d ^= a; a &= b; t ^= a; b ^= d; b ^= t;
	a = c; c = b; b = d; d = ~t;
}

__device__ __forceinline__ void adam_hamsi_L(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d)
{
	a = adam_rotl32(a, 13); c = adam_rotl32(c, 3);
	b ^= a ^ c; d ^= c ^ (a << 3);
	b = adam_rotl32(b, 1); d = adam_rotl32(d, 7);
	a ^= b ^ d; c ^= d ^ (b << 7);
	a = adam_rotl32(a, 5); c = adam_rotl32(c, 22);
}

__device__ void adam_hamsi_perm(uint32_t h[16], const uint32_t m[16], int rounds, const uint32_t *alpha)
{
	uint32_t s[32] = {
		m[0x0], m[0x1], h[0x0], h[0x1], m[0x2], m[0x3], h[0x2], h[0x3],
		h[0x4], h[0x5], m[0x4], m[0x5], h[0x6], h[0x7], m[0x6], m[0x7],
		m[0x8], m[0x9], h[0x8], h[0x9], m[0xA], m[0xB], h[0xA], h[0xB],
		h[0xC], h[0xD], m[0xC], m[0xD], h[0xE], h[0xF], m[0xE], m[0xF]
	};
	for (int r = 0; r < rounds; r++) {
		#pragma unroll
		for (int i = 0; i < 32; i++)
			s[i] ^= alpha[i];
		s[1] ^= (uint32_t)r;
		#pragma unroll
		for (int i = 0; i < 8; i++)
			adam_hamsi_sbox(s[i], s[i + 8], s[i + 16], s[i + 24]);
		adam_hamsi_L(s[0], s[ 9], s[18], s[27]);
		adam_hamsi_L(s[1], s[10], s[19], s[28]);
		adam_hamsi_L(s[2], s[11], s[20], s[29]);
		adam_hamsi_L(s[3], s[12], s[21], s[30]);
		adam_hamsi_L(s[4], s[13], s[22], s[31]);
		adam_hamsi_L(s[5], s[14], s[23], s[24]);
		adam_hamsi_L(s[6], s[15], s[16], s[25]);
		adam_hamsi_L(s[7], s[ 8], s[17], s[26]);
		adam_hamsi_L(s[0], s[ 2], s[ 5], s[ 7]);
		adam_hamsi_L(s[16], s[19], s[21], s[22]);
		adam_hamsi_L(s[9], s[11], s[12], s[14]);
		adam_hamsi_L(s[25], s[26], s[28], s[31]);
	}
	h[0x0] ^= s[0x00]; h[0x1] ^= s[0x01]; h[0x2] ^= s[0x02]; h[0x3] ^= s[0x03];
	h[0x4] ^= s[0x04]; h[0x5] ^= s[0x05]; h[0x6] ^= s[0x06]; h[0x7] ^= s[0x07];
	h[0x8] ^= s[0x10]; h[0x9] ^= s[0x11]; h[0xA] ^= s[0x12]; h[0xB] ^= s[0x13];
	h[0xC] ^= s[0x14]; h[0xD] ^= s[0x15]; h[0xE] ^= s[0x16]; h[0xF] ^= s[0x17];
}

__device__ void adam_hamsi_expand_chunk(const uint8_t chunk[8], uint32_t m[16])
{
	#pragma unroll
	for (int i = 0; i < 16; i++)
		m[i] = 0;
	for (int u = 0; u < 8; u++) {
		uint32_t db = chunk[u];
		for (int v = 0; v < 8; v++, db >>= 1) {
			uint32_t dm = -(db & 1U);
			const uint32_t *tp = &c_adam_hamsi_T512[u * 8 + v][0];
			#pragma unroll
			for (int k = 0; k < 16; k++)
				m[k] ^= dm & tp[k];
		}
	}
}

__device__ __forceinline__ void adam_hamsi_init(uint32_t h[16])
{
	h[0x0] = 0x73746565U; h[0x1] = 0x6c706172U; h[0x2] = 0x6b204172U; h[0x3] = 0x656e6265U;
	h[0x4] = 0x72672031U; h[0x5] = 0x302c2062U; h[0x6] = 0x75732032U; h[0x7] = 0x3434362cU;
	h[0x8] = 0x20422d33U; h[0x9] = 0x30303120U; h[0xA] = 0x4c657576U; h[0xB] = 0x656e2d48U;
	h[0xC] = 0x65766572U; h[0xD] = 0x6c65652cU; h[0xE] = 0x2042656cU; h[0xF] = 0x6769756dU;
}

__global__ void adam_gpu_hamsi512_70(uint32_t threads, uint32_t start_nonce, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t w[18];
	adam_load_le32_70(w, start_nonce + tid);

	uint32_t h[16], m[16];
	adam_hamsi_init(h);

	for (int i = 0; i < 8; i++) {
		adam_hamsi_expand_chunk((const uint8_t *)&w[i * 2], m);
		adam_hamsi_perm(h, m, 6, c_adam_hamsi_alpha_n);
	}
	uint32_t c8[2] = { w[16], (w[17] & 0x0000FFFFU) | 0x00800000U };
	adam_hamsi_expand_chunk((const uint8_t *)c8, m);
	adam_hamsi_perm(h, m, 6, c_adam_hamsi_alpha_n);

	uint32_t c9[2] = { 0U, 0x30020000U }; /* BE64(560 = 0x0230) in bytes 6..7 */
	adam_hamsi_expand_chunk((const uint8_t *)c9, m);
	adam_hamsi_perm(h, m, 12, c_adam_hamsi_alpha_f);

	uint32_t out32[8];
	#pragma unroll
	for (int i = 0; i < 8; i++)
		out32[i] = adam_bswap32(h[i]);
	adam_store_le32_32(g_hash, tid, out32);
}

__global__ void adam_gpu_hamsi512_32(uint32_t threads, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t w[8];
	adam_load_le32_32(w, g_hash, tid);

	uint32_t h[16], m[16];
	adam_hamsi_init(h);

	for (int i = 0; i < 4; i++) {
		adam_hamsi_expand_chunk((const uint8_t *)&w[i * 2], m);
		adam_hamsi_perm(h, m, 6, c_adam_hamsi_alpha_n);
	}
	uint32_t c4[2] = { 0x00000080U, 0U };
	adam_hamsi_expand_chunk((const uint8_t *)c4, m);
	adam_hamsi_perm(h, m, 6, c_adam_hamsi_alpha_n);

	uint32_t c5[2] = { 0U, 0x00010000U }; /* BE64(256 = 0x0100) in bytes 6..7 */
	adam_hamsi_expand_chunk((const uint8_t *)c5, m);
	adam_hamsi_perm(h, m, 12, c_adam_hamsi_alpha_f);

	uint32_t out32[8];
	#pragma unroll
	for (int i = 0; i < 8; i++)
		out32[i] = adam_bswap32(h[i]);
	adam_store_le32_32(g_hash, tid, out32);
}

/* ------------------- 14. Fugue-512 ------------------- */

static __constant__ uint32_t c_adam_fugue_mix0[256] = {
	0x63633297U, 0x7c7c6febU, 0x77775ec7U, 0x7b7b7af7U, 0xf2f2e8e5U, 0x6b6b0ab7U, 0x6f6f16a7U, 0xc5c56d39U,
	0x303090c0U, 0x01010704U, 0x67672e87U, 0x2b2bd1acU, 0xfefeccd5U, 0xd7d71371U, 0xabab7c9aU, 0x767659c3U,
	0xcaca4005U, 0x8282a33eU, 0xc9c94909U, 0x7d7d68efU, 0xfafad0c5U, 0x5959947fU, 0x4747ce07U, 0xf0f0e6edU,
	0xadad6e82U, 0xd4d41a7dU, 0xa2a243beU, 0xafaf608aU, 0x9c9cf946U, 0xa4a451a6U, 0x727245d3U, 0xc0c0762dU,
	0xb7b728eaU, 0xfdfdc5d9U, 0x9393d47aU, 0x2626f298U, 0x363682d8U, 0x3f3fbdfcU, 0xf7f7f3f1U, 0xcccc521dU,
	0x34348cd0U, 0xa5a556a2U, 0xe5e58db9U, 0xf1f1e1e9U, 0x71714cdfU, 0xd8d83e4dU, 0x313197c4U, 0x15156b54U,
	0x04041c10U, 0xc7c76331U, 0x2323e98cU, 0xc3c37f21U, 0x18184860U, 0x9696cf6eU, 0x05051b14U, 0x9a9aeb5eU,
	0x0707151cU, 0x12127e48U, 0x8080ad36U, 0xe2e298a5U, 0xebeba781U, 0x2727f59cU, 0xb2b233feU, 0x757550cfU,
	0x09093f24U, 0x8383a43aU, 0x2c2cc4b0U, 0x1a1a4668U, 0x1b1b416cU, 0x6e6e11a3U, 0x5a5a9d73U, 0xa0a04db6U,
	0x5252a553U, 0x3b3ba1ecU, 0xd6d61475U, 0xb3b334faU, 0x2929dfa4U, 0xe3e39fa1U, 0x2f2fcdbcU, 0x8484b126U,
	0x5353a257U, 0xd1d10169U, 0x00000000U, 0xededb599U, 0x2020e080U, 0xfcfcc2ddU, 0xb1b13af2U, 0x5b5b9a77U,
	0x6a6a0db3U, 0xcbcb4701U, 0xbebe17ceU, 0x3939afe4U, 0x4a4aed33U, 0x4c4cff2bU, 0x5858937bU, 0xcfcf5b11U,
	0xd0d0066dU, 0xefefbb91U, 0xaaaa7b9eU, 0xfbfbd7c1U, 0x4343d217U, 0x4d4df82fU, 0x333399ccU, 0x8585b622U,
	0x4545c00fU, 0xf9f9d9c9U, 0x02020e08U, 0x7f7f66e7U, 0x5050ab5bU, 0x3c3cb4f0U, 0x9f9ff04aU, 0xa8a87596U,
	0x5151ac5fU, 0xa3a344baU, 0x4040db1bU, 0x8f8f800aU, 0x9292d37eU, 0x9d9dfe42U, 0x3838a8e0U, 0xf5f5fdf9U,
	0xbcbc19c6U, 0xb6b62feeU, 0xdada3045U, 0x2121e784U, 0x10107040U, 0xffffcbd1U, 0xf3f3efe1U, 0xd2d20865U,
	0xcdcd5519U, 0x0c0c2430U, 0x1313794cU, 0xececb29dU, 0x5f5f8667U, 0x9797c86aU, 0x4444c70bU, 0x1717655cU,
	0xc4c46a3dU, 0xa7a758aaU, 0x7e7e61e3U, 0x3d3db3f4U, 0x6464278bU, 0x5d5d886fU, 0x19194f64U, 0x737342d7U,
	0x60603b9bU, 0x8181aa32U, 0x4f4ff627U, 0xdcdc225dU, 0x2222ee88U, 0x2a2ad6a8U, 0x9090dd76U, 0x88889516U,
	0x4646c903U, 0xeeeebc95U, 0xb8b805d6U, 0x14146c50U, 0xdede2c55U, 0x5e5e8163U, 0x0b0b312cU, 0xdbdb3741U,
	0xe0e096adU, 0x32329ec8U, 0x3a3aa6e8U, 0x0a0a3628U, 0x4949e43fU, 0x06061218U, 0x2424fc90U, 0x5c5c8f6bU,
	0xc2c27825U, 0xd3d30f61U, 0xacac6986U, 0x62623593U, 0x9191da72U, 0x9595c662U, 0xe4e48abdU, 0x797974ffU,
	0xe7e783b1U, 0xc8c84e0dU, 0x373785dcU, 0x6d6d18afU, 0x8d8d8e02U, 0xd5d51d79U, 0x4e4ef123U, 0xa9a97292U,
	0x6c6c1fabU, 0x5656b943U, 0xf4f4fafdU, 0xeaeaa085U, 0x6565208fU, 0x7a7a7df3U, 0xaeae678eU, 0x08083820U,
	0xbaba0bdeU, 0x787873fbU, 0x2525fb94U, 0x2e2ecab8U, 0x1c1c5470U, 0xa6a65faeU, 0xb4b421e6U, 0xc6c66435U,
	0xe8e8ae8dU, 0xdddd2559U, 0x747457cbU, 0x1f1f5d7cU, 0x4b4bea37U, 0xbdbd1ec2U, 0x8b8b9c1aU, 0x8a8a9b1eU,
	0x70704bdbU, 0x3e3ebaf8U, 0xb5b526e2U, 0x66662983U, 0x4848e33bU, 0x0303090cU, 0xf6f6f4f5U, 0x0e0e2a38U,
	0x61613c9fU, 0x35358bd4U, 0x5757be47U, 0xb9b902d2U, 0x8686bf2eU, 0xc1c17129U, 0x1d1d5374U, 0x9e9ef74eU,
	0xe1e191a9U, 0xf8f8decdU, 0x9898e556U, 0x11117744U, 0x696904bfU, 0xd9d93949U, 0x8e8e870eU, 0x9494c166U,
	0x9b9bec5aU, 0x1e1e5a78U, 0x8787b82aU, 0xe9e9a989U, 0xcece5c15U, 0x5555b04fU, 0x2828d8a0U, 0xdfdf2b51U,
	0x8c8c8906U, 0xa1a14ab2U, 0x89899212U, 0x0d0d2334U, 0xbfbf10caU, 0xe6e684b5U, 0x4242d513U, 0x686803bbU,
	0x4141dc1fU, 0x9999e252U, 0x2d2dc3b4U, 0x0f0f2d3cU, 0xb0b03df6U, 0x5454b74bU, 0xbbbb0cdaU, 0x16166258U
};

__device__ __forceinline__ void adam_fugue_ror(uint32_t S[36], int n)
{
	uint32_t tmp[12];
	for (int i = 0; i < n; i++) tmp[i] = S[36 - n + i];
	for (int i = 35; i >= n; i--) S[i] = S[i - n];
	for (int i = 0; i < n; i++) S[i] = tmp[i];
}

__device__ __forceinline__ void adam_fugue_smix(uint32_t S[36])
{
	uint32_t x0 = S[0], x1 = S[1], x2 = S[2], x3 = S[3];
	uint32_t r0 = 0U, r1 = 0U, r2 = 0U, r3 = 0U, tmp;
	uint32_t c0, c1, c2, c3;

	c0 = c_adam_fugue_mix0[x0 >> 24];
	tmp = adam_rotr32(c_adam_fugue_mix0[(x0 >> 16) & 0xFF], 8);  c0 ^= tmp; r1 ^= tmp;
	tmp = adam_rotl32(c_adam_fugue_mix0[(x0 >>  8) & 0xFF], 16); c0 ^= tmp; r2 ^= tmp;
	tmp = adam_rotl32(c_adam_fugue_mix0[(x0      ) & 0xFF], 8);  c0 ^= tmp; r3 ^= tmp;

	c1 = c_adam_fugue_mix0[x1 >> 24]; r0 ^= c1;
	tmp = adam_rotr32(c_adam_fugue_mix0[(x1 >> 16) & 0xFF], 8);  c1 ^= tmp;
	tmp = adam_rotl32(c_adam_fugue_mix0[(x1 >>  8) & 0xFF], 16); c1 ^= tmp; r2 ^= tmp;
	tmp = adam_rotl32(c_adam_fugue_mix0[(x1      ) & 0xFF], 8);  c1 ^= tmp; r3 ^= tmp;

	c2 = c_adam_fugue_mix0[x2 >> 24]; r0 ^= c2;
	tmp = adam_rotr32(c_adam_fugue_mix0[(x2 >> 16) & 0xFF], 8);  c2 ^= tmp; r1 ^= tmp;
	tmp = adam_rotl32(c_adam_fugue_mix0[(x2 >>  8) & 0xFF], 16); c2 ^= tmp;
	tmp = adam_rotl32(c_adam_fugue_mix0[(x2      ) & 0xFF], 8);  c2 ^= tmp; r3 ^= tmp;

	c3 = c_adam_fugue_mix0[x3 >> 24]; r0 ^= c3;
	tmp = adam_rotr32(c_adam_fugue_mix0[(x3 >> 16) & 0xFF], 8);  c3 ^= tmp; r1 ^= tmp;
	tmp = adam_rotl32(c_adam_fugue_mix0[(x3 >>  8) & 0xFF], 16); c3 ^= tmp; r2 ^= tmp;
	tmp = adam_rotl32(c_adam_fugue_mix0[(x3      ) & 0xFF], 8);  c3 ^= tmp;

	S[0] = ((c0 ^ r0) & 0xFF000000U) | ((c1 ^ r1) & 0x00FF0000U) | ((c2 ^ r2) & 0x0000FF00U) | ((c3 ^ r3) & 0x000000FFU);
	S[1] = ((c1 ^ (r0 <<  8)) & 0xFF000000U) | ((c2 ^ (r1 <<  8)) & 0x00FF0000U) | ((c3 ^ (r2 <<  8)) & 0x0000FF00U) | ((c0 ^ (r3 >> 24)) & 0x000000FFU);
	S[2] = ((c2 ^ (r0 << 16)) & 0xFF000000U) | ((c3 ^ (r1 << 16)) & 0x00FF0000U) | ((c0 ^ (r2 >> 16)) & 0x0000FF00U) | ((c1 ^ (r3 >> 16)) & 0x000000FFU);
	S[3] = ((c3 ^ (r0 << 24)) & 0xFF000000U) | ((c0 ^ (r1 >>  8)) & 0x00FF0000U) | ((c1 ^ (r2 >>  8)) & 0x0000FF00U) | ((c2 ^ (r3 >>  8)) & 0x000000FFU);
}

__device__ __forceinline__ void adam_fugue_cmix36(uint32_t S[36])
{
	S[ 0] ^= S[4]; S[ 1] ^= S[5]; S[ 2] ^= S[6];
	S[18] ^= S[4]; S[19] ^= S[5]; S[20] ^= S[6];
}

__device__ void adam_fugue_word_step(uint32_t S[36], uint32_t q)
{
	S[22] ^= S[0]; S[0] = q; S[8] ^= q;
	S[ 1] ^= S[24]; S[4] ^= S[27]; S[7] ^= S[30];
	adam_fugue_ror(S, 3);
	adam_fugue_cmix36(S);
	adam_fugue_smix(S);
	adam_fugue_ror(S, 3);
	adam_fugue_cmix36(S);
	adam_fugue_smix(S);
	adam_fugue_ror(S, 3);
	adam_fugue_cmix36(S);
	adam_fugue_smix(S);
	adam_fugue_ror(S, 3);
	adam_fugue_cmix36(S);
	adam_fugue_smix(S);
}

__device__ void adam_fugue_finish(uint32_t S[36], uint32_t out32[8])
{
	for (int i = 0; i < 32; i++) {
		adam_fugue_ror(S, 3);
		adam_fugue_cmix36(S);
		adam_fugue_smix(S);
	}
	for (int i = 0; i < 13; i++) {
		S[4] ^= S[0]; S[ 9] ^= S[0]; S[18] ^= S[0]; S[27] ^= S[0];
		adam_fugue_ror(S, 9); adam_fugue_smix(S);
		S[4] ^= S[0]; S[10] ^= S[0]; S[18] ^= S[0]; S[27] ^= S[0];
		adam_fugue_ror(S, 9); adam_fugue_smix(S);
		S[4] ^= S[0]; S[10] ^= S[0]; S[19] ^= S[0]; S[27] ^= S[0];
		adam_fugue_ror(S, 9); adam_fugue_smix(S);
		S[4] ^= S[0]; S[10] ^= S[0]; S[19] ^= S[0]; S[28] ^= S[0];
		adam_fugue_ror(S, 8); adam_fugue_smix(S);
	}
	S[4] ^= S[0]; S[9] ^= S[0]; S[18] ^= S[0]; S[27] ^= S[0];

	out32[0] = adam_bswap32(S[ 1]); out32[1] = adam_bswap32(S[ 2]);
	out32[2] = adam_bswap32(S[ 3]); out32[3] = adam_bswap32(S[ 4]);
	out32[4] = adam_bswap32(S[ 9]); out32[5] = adam_bswap32(S[10]);
	out32[6] = adam_bswap32(S[11]); out32[7] = adam_bswap32(S[12]);
}

__device__ __forceinline__ void adam_fugue512_init(uint32_t S[36])
{
	for (int i = 0; i < 20; i++) S[i] = 0U;
	S[20] = 0x8807a57eU; S[21] = 0xe616af75U; S[22] = 0xc5d3e4dbU; S[23] = 0xac9ab027U;
	S[24] = 0xd915f117U; S[25] = 0xb6eecc54U; S[26] = 0x06e8020bU; S[27] = 0x4a92efd1U;
	S[28] = 0xaac6e2c9U; S[29] = 0xddb21398U; S[30] = 0xcae65838U; S[31] = 0x437f203fU;
	S[32] = 0x25ea78e7U; S[33] = 0x951fddd6U; S[34] = 0xda6ed11dU; S[35] = 0xe13e3567U;
}

__global__ void adam_gpu_fugue512_70(uint32_t threads, uint32_t start_nonce, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t w[18];
	adam_load_le32_70(w, start_nonce + tid);

	uint32_t S[36];
	adam_fugue512_init(S);

	for (int i = 0; i < 18; i++)
		adam_fugue_word_step(S, adam_bswap32(w[i]));
	adam_fugue_word_step(S, 0U);
	adam_fugue_word_step(S, 560U);

	uint32_t out32[8];
	adam_fugue_finish(S, out32);
	adam_store_le32_32(g_hash, tid, out32);
}

__global__ void adam_gpu_fugue512_32(uint32_t threads, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t w[8];
	adam_load_le32_32(w, g_hash, tid);

	uint32_t S[36];
	adam_fugue512_init(S);

	for (int i = 0; i < 8; i++)
		adam_fugue_word_step(S, adam_bswap32(w[i]));
	adam_fugue_word_step(S, 0U);
	adam_fugue_word_step(S, 256U);

	uint32_t out32[8];
	adam_fugue_finish(S, out32);
	adam_store_le32_32(g_hash, tid, out32);
}

/* ------------------- 15. Shabal-512 ------------------- */

static __constant__ uint32_t c_adam_shabal_A512[12] = {
	0x20728DFDU, 0x46C0BD53U, 0xE782B699U, 0x55304632U,
	0x71B4EF90U, 0x0EA9E82CU, 0xDBB930F1U, 0xFAD06B8BU,
	0xBE0CAE40U, 0x8BD14410U, 0x76D2ADACU, 0x28ACAB7FU
};
static __constant__ uint32_t c_adam_shabal_B512[16] = {
	0xC1099CB7U, 0x07B385F3U, 0xE7442C26U, 0xCC8AD640U,
	0xEB6F56C7U, 0x1EA81AA9U, 0x73B9D314U, 0x1DE85D08U,
	0x48910A5AU, 0x893B22DBU, 0xC5A0DF44U, 0xBBC4324EU,
	0x72D2F240U, 0x75941D99U, 0x6D8BDE82U, 0xA1A7502BU
};
static __constant__ uint32_t c_adam_shabal_C512[16] = {
	0xD9BF68D1U, 0x58BAD750U, 0x56028CB2U, 0x8134F359U,
	0xB5D469D8U, 0x941A8CC2U, 0x418B2A6EU, 0x04052780U,
	0x7F07D787U, 0x5194358FU, 0x3C60D665U, 0xBE97D79AU,
	0x950C3434U, 0xAED9A06DU, 0x2537DC8DU, 0x7CDB5969U
};

__device__ __forceinline__ void adam_shabal_apply_p(
	uint32_t A[12], uint32_t B[16], const uint32_t C[16], const uint32_t M[16])
{
	#pragma unroll
	for (int i = 0; i < 16; i++)
		B[i] = adam_rotl32(B[i], 17);

	uint32_t a_idx = 0;
	for (int step = 0; step < 3; step++) {
		#pragma unroll
		for (int i = 0; i < 16; i++) {
			uint32_t ap = (a_idx + 11) % 12;
			uint32_t c_idx = (8 - i + 16) & 15;
			uint32_t b1 = (i + 13) & 15;
			uint32_t b2 = (i +  9) & 15;
			uint32_t b3 = (i +  6) & 15;
			A[a_idx] = ((A[a_idx] ^ (adam_rotl32(A[ap], 15) * 5U) ^ C[c_idx]) * 3U)
			         ^ B[b1] ^ (B[b2] & ~B[b3]) ^ M[i];
			B[i] = ~(adam_rotl32(B[i], 1) ^ A[a_idx]);
			a_idx = (a_idx + 1) % 12;
		}
	}

	A[11] += C[ 6]; A[10] += C[ 5]; A[ 9] += C[ 4]; A[ 8] += C[ 3];
	A[ 7] += C[ 2]; A[ 6] += C[ 1]; A[ 5] += C[ 0]; A[ 4] += C[15];
	A[ 3] += C[14]; A[ 2] += C[13]; A[ 1] += C[12]; A[ 0] += C[11];
	A[11] += C[10]; A[10] += C[ 9]; A[ 9] += C[ 8]; A[ 8] += C[ 7];
	A[ 7] += C[ 6]; A[ 6] += C[ 5]; A[ 5] += C[ 4]; A[ 4] += C[ 3];
	A[ 3] += C[ 2]; A[ 2] += C[ 1]; A[ 1] += C[ 0]; A[ 0] += C[15];
	A[11] += C[14]; A[10] += C[13]; A[ 9] += C[12]; A[ 8] += C[11];
	A[ 7] += C[10]; A[ 6] += C[ 9]; A[ 5] += C[ 8]; A[ 4] += C[ 7];
	A[ 3] += C[ 6]; A[ 2] += C[ 5]; A[ 1] += C[ 4]; A[ 0] += C[ 3];
}

__device__ __forceinline__ void adam_shabal_swap_bc(uint32_t B[16], uint32_t C[16])
{
	#pragma unroll
	for (int i = 0; i < 16; i++) {
		uint32_t t = B[i]; B[i] = C[i]; C[i] = t;
	}
}

__device__ void adam_shabal_finish(uint32_t A[12], uint32_t B[16], uint32_t C[16], const uint32_t M[16], uint32_t Wlow, uint32_t out32[8])
{
	#pragma unroll
	for (int i = 0; i < 16; i++)
		B[i] += M[i];
	A[0] ^= Wlow;
	adam_shabal_apply_p(A, B, C, M);
	for (int r = 0; r < 3; r++) {
		adam_shabal_swap_bc(B, C);
		A[0] ^= Wlow;
		adam_shabal_apply_p(A, B, C, M);
	}
	#pragma unroll
	for (int i = 0; i < 8; i++)
		out32[i] = B[i];
}

__global__ void adam_gpu_shabal512_70(uint32_t threads, uint32_t start_nonce, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t w[18];
	adam_load_le32_70(w, start_nonce + tid);

	uint32_t A[12], B[16], C[16], M[16];
	for (int i = 0; i < 12; i++) A[i] = c_adam_shabal_A512[i];
	for (int i = 0; i < 16; i++) {
		B[i] = c_adam_shabal_B512[i];
		C[i] = c_adam_shabal_C512[i];
		M[i] = w[i];
	}

	/* Block 0 */
	for (int i = 0; i < 16; i++) B[i] += M[i];
	A[0] ^= 1U;
	adam_shabal_apply_p(A, B, C, M);
	for (int i = 0; i < 16; i++) C[i] -= M[i];
	adam_shabal_swap_bc(B, C);

	/* Final Block 1: 6 bytes + 0x80 */
	M[0] = w[16];
	M[1] = w[17] | 0x00800000U;
	for (int i = 2; i < 16; i++) M[i] = 0U;

	uint32_t out32[8];
	adam_shabal_finish(A, B, C, M, 2U, out32);
	adam_store_le32_32(g_hash, tid, out32);
}

__global__ void adam_gpu_shabal512_32(uint32_t threads, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t w[8];
	adam_load_le32_32(w, g_hash, tid);

	uint32_t A[12], B[16], C[16], M[16];
	for (int i = 0; i < 12; i++) A[i] = c_adam_shabal_A512[i];
	for (int i = 0; i < 16; i++) {
		B[i] = c_adam_shabal_B512[i];
		C[i] = c_adam_shabal_C512[i];
	}
	for (int i = 0; i < 8; i++) M[i] = w[i];
	M[8] = 0x00000080U;
	for (int i = 9; i < 16; i++) M[i] = 0U;

	uint32_t out32[8];
	adam_shabal_finish(A, B, C, M, 1U, out32);
	adam_store_le32_32(g_hash, tid, out32);
}

/* ------------------- 16. Whirlpool (plain) ------------------- */

static __constant__ uint64_t c_adam_whirlpool_T0[256] = {
	0xD83078C018601818ULL, 0x2646AF05238C2323ULL, 0xB891F97EC63FC6C6ULL, 0xFBCD6F13E887E8E8ULL,
	0xCB13A14C87268787ULL, 0x116D62A9B8DAB8B8ULL, 0x0902050801040101ULL, 0x0D9E6E424F214F4FULL,
	0x9B6CEEAD36D83636ULL, 0xFF510459A6A2A6A6ULL, 0x0CB9BDDED26FD2D2ULL, 0x0EF706FBF5F3F5F5ULL,
	0x96F280EF79F97979ULL, 0x30DECE5F6FA16F6FULL, 0x6D3FEFFC917E9191ULL, 0xF8A407AA52555252ULL,
	0x47C0FD27609D6060ULL, 0x35657689BCCABCBCULL, 0x372BCDAC9B569B9BULL, 0x8A018C048E028E8EULL,
	0xD25B1571A3B6A3A3ULL, 0x6C183C600C300C0CULL, 0x84F68AFF7BF17B7BULL, 0x806AE1B535D43535ULL,
	0xF53A69E81D741D1DULL, 0xB3DD4753E0A7E0E0ULL, 0x21B3ACF6D77BD7D7ULL, 0x9C99ED5EC22FC2C2ULL,
	0x435C966D2EB82E2EULL, 0x29967A624B314B4BULL, 0x5DE121A3FEDFFEFEULL, 0xD5AE168257415757ULL,
	0xBD2A41A815541515ULL, 0xE8EEB69F77C17777ULL, 0x926EEBA537DC3737ULL, 0x9ED7567BE5B3E5E5ULL,
	0x1323D98C9F469F9FULL, 0x23FD17D3F0E7F0F0ULL, 0x20947F6A4A354A4AULL, 0x44A9959EDA4FDADAULL,
	0xA2B025FA587D5858ULL, 0xCF8FCA06C903C9C9ULL, 0x7C528D5529A42929ULL, 0x5A1422500A280A0AULL,
	0x507F4FE1B1FEB1B1ULL, 0xC95D1A69A0BAA0A0ULL, 0x14D6DA7F6BB16B6BULL, 0xD917AB5C852E8585ULL,
	0x3C677381BDCEBDBDULL, 0x8FBA34D25D695D5DULL, 0x9020508010401010ULL, 0x07F503F3F4F7F4F4ULL,
	0xDD8BC016CB0BCBCBULL, 0xD37CC6ED3EF83E3EULL, 0x2D0A112805140505ULL, 0x78CEE61F67816767ULL,
	0x97D55373E4B7E4E4ULL, 0x024EBB25279C2727ULL, 0x7382583241194141ULL, 0xA70B9D2C8B168B8BULL,
	0xF6530151A7A6A7A7ULL, 0xB2FA94CF7DE97D7DULL, 0x4937FBDC956E9595ULL, 0x56AD9F8ED847D8D8ULL,
	0x70EB308BFBCBFBFBULL, 0xCDC17123EE9FEEEEULL, 0xBBF891C77CED7C7CULL, 0x71CCE31766856666ULL,
	0x7BA78EA6DD53DDDDULL, 0xAF2E4BB8175C1717ULL, 0x458E460247014747ULL, 0x1A21DC849E429E9EULL,
	0xD489C51ECA0FCACAULL, 0x585A99752DB42D2DULL, 0x2E637991BFC6BFBFULL, 0x3F0E1B38071C0707ULL,
	0xAC472301AD8EADADULL, 0xB0B42FEA5A755A5AULL, 0xEF1BB56C83368383ULL, 0xB666FF8533CC3333ULL,
	0x5CC6F23F63916363ULL, 0x12040A1002080202ULL, 0x93493839AA92AAAAULL, 0xDEE2A8AF71D97171ULL,
	0xC68DCF0EC807C8C8ULL, 0xD1327DC819641919ULL, 0x3B92707249394949ULL, 0x5FAF9A86D943D9D9ULL,
	0x31F91DC3F2EFF2F2ULL, 0xA8DB484BE3ABE3E3ULL, 0xB9B62AE25B715B5BULL, 0xBC0D9234881A8888ULL,
	0x3E29C8A49A529A9AULL, 0x0B4CBE2D26982626ULL, 0xBF64FA8D32C83232ULL, 0x597D4AE9B0FAB0B0ULL,
	0xF2CF6A1BE983E9E9ULL, 0x771E33780F3C0F0FULL, 0x33B7A6E6D573D5D5ULL, 0xF41DBA74803A8080ULL,
	0x27617C99BEC2BEBEULL, 0xEB87DE26CD13CDCDULL, 0x8968E4BD34D03434ULL, 0x3290757A483D4848ULL,
	0x54E324ABFFDBFFFFULL, 0x8DF48FF77AF57A7AULL, 0x643DEAF4907A9090ULL, 0x9DBE3EC25F615F5FULL,
	0x3D40A01D20802020ULL, 0x0FD0D56768BD6868ULL, 0xCA3472D01A681A1AULL, 0xB7412C19AE82AEAEULL,
	0x7D755EC9B4EAB4B4ULL, 0xCEA8199A544D5454ULL, 0x7F3BE5EC93769393ULL, 0x2F44AA0D22882222ULL,
	0x63C8E907648D6464ULL, 0x2AFF12DBF1E3F1F1ULL, 0xCCE6A2BF73D17373ULL, 0x82245A9012481212ULL,
	0x7A805D3A401D4040ULL, 0x4810284008200808ULL, 0x959BE856C32BC3C3ULL, 0xDFC57B33EC97ECECULL,
	0x4DAB9096DB4BDBDBULL, 0xC05F1F61A1BEA1A1ULL, 0x9107831C8D0E8D8DULL, 0xC87AC9F53DF43D3DULL,
	0x5B33F1CC97669797ULL, 0x0000000000000000ULL, 0xF983D436CF1BCFCFULL, 0x6E5687452BAC2B2BULL,
	0xE1ECB39776C57676ULL, 0xE619B06482328282ULL, 0x28B1A9FED67FD6D6ULL, 0xC33677D81B6C1B1BULL,
	0x74775BC1B5EEB5B5ULL, 0xBE432911AF86AFAFULL, 0x1DD4DF776AB56A6AULL, 0xEAA00DBA505D5050ULL,
	0x578A4C1245094545ULL, 0x38FB18CBF3EBF3F3ULL, 0xAD60F09D30C03030ULL, 0xC4C3742BEF9BEFEFULL,
	0xDA7EC3E53FFC3F3FULL, 0xC7AA1C9255495555ULL, 0xDB591079A2B2A2A2ULL, 0xE9C96503EA8FEAEAULL,
	0x6ACAEC0F65896565ULL, 0x036968B9BAD2BABAULL, 0x4A5E93652FBC2F2FULL, 0x8E9DE74EC027C0C0ULL,
	0x60A181BEDE5FDEDEULL, 0xFC386CE01C701C1CULL, 0x46E72EBBFDD3FDFDULL, 0x1F9A64524D294D4DULL,
	0x7639E0E492729292ULL, 0xFAEABC8F75C97575ULL, 0x360C1E3006180606ULL, 0xAE0998248A128A8AULL,
	0x4B7940F9B2F2B2B2ULL, 0x85D15963E6BFE6E6ULL, 0x7E1C36700E380E0EULL, 0xE73E63F81F7C1F1FULL,
	0x55C4F73762956262ULL, 0x3AB5A3EED477D4D4ULL, 0x814D3229A89AA8A8ULL, 0x5231F4C496629696ULL,
	0x62EF3A9BF9C3F9F9ULL, 0xA397F666C533C5C5ULL, 0x104AB13525942525ULL, 0xABB220F259795959ULL,
	0xD015AE54842A8484ULL, 0xC5E4A7B772D57272ULL, 0xEC72DDD539E43939ULL, 0x1698615A4C2D4C4CULL,
	0x94BC3BCA5E655E5EULL, 0x9FF085E778FD7878ULL, 0xE570D8DD38E03838ULL, 0x980586148C0A8C8CULL,
	0x17BFB2C6D163D1D1ULL, 0xE4570B41A5AEA5A5ULL, 0xA1D94D43E2AFE2E2ULL, 0x4EC2F82F61996161ULL,
	0x427B45F1B3F6B3B3ULL, 0x3442A51521842121ULL, 0x0825D6949C4A9C9CULL, 0xEE3C66F01E781E1EULL,
	0x6186522243114343ULL, 0xB193FC76C73BC7C7ULL, 0x4FE52BB3FCD7FCFCULL, 0x2408142004100404ULL,
	0xE3A208B251595151ULL, 0x252FC7BC995E9999ULL, 0x22DAC44F6DA96D6DULL, 0x651A39680D340D0DULL,
	0x79E93583FACFFAFAULL, 0x69A384B6DF5BDFDFULL, 0xA9FC9BD77EE57E7EULL, 0x1948B43D24902424ULL,
	0xFE76D7C53BEC3B3BULL, 0x9A4B3D31AB96ABABULL, 0xF081D13ECE1FCECEULL, 0x9922558811441111ULL,
	0x8303890C8F068F8FULL, 0x049C6B4A4E254E4EULL, 0x667351D1B7E6B7B7ULL, 0xE0CB600BEB8BEBEBULL,
	0xC178CCFD3CF03C3CULL, 0xFD1FBF7C813E8181ULL, 0x4035FED4946A9494ULL, 0x1CF30CEBF7FBF7F7ULL,
	0x186F67A1B9DEB9B9ULL, 0x8B265F98134C1313ULL, 0x51589C7D2CB02C2CULL, 0x05BBB8D6D36BD3D3ULL,
	0x8CD35C6BE7BBE7E7ULL, 0x39DCCB576EA56E6EULL, 0xAA95F36EC437C4C4ULL, 0x1B060F18030C0303ULL,
	0xDCAC138A56455656ULL, 0x5E88491A440D4444ULL, 0xA0FE9EDF7FE17F7FULL, 0x884F3721A99EA9A9ULL,
	0x6754824D2AA82A2AULL, 0x0A6B6DB1BBD6BBBBULL, 0x879FE246C123C1C1ULL, 0xF1A602A253515353ULL,
	0x72A58BAEDC57DCDCULL, 0x531627580B2C0B0BULL, 0x0127D39C9D4E9D9DULL, 0x2BD8C1476CAD6C6CULL,
	0xA462F59531C43131ULL, 0xF3E8B98774CD7474ULL, 0x15F109E3F6FFF6F6ULL, 0x4C8C430A46054646ULL,
	0xA5452609AC8AACACULL, 0xB50F973C891E8989ULL, 0xB42844A014501414ULL, 0xBADF425BE1A3E1E1ULL,
	0xA62C4EB016581616ULL, 0xF774D2CD3AE83A3AULL, 0x06D2D06F69B96969ULL, 0x41122D4809240909ULL,
	0xD7E0ADA770DD7070ULL, 0x6F7154D9B6E2B6B6ULL, 0x1EBDB7CED067D0D0ULL, 0xD6C77E3BED93EDEDULL,
	0xE285DB2ECC17CCCCULL, 0x6884572A42154242ULL, 0x2C2DC2B4985A9898ULL, 0xED550E49A4AAA4A4ULL,
	0x7550885D28A02828ULL, 0x86B831DA5C6D5C5CULL, 0x6BED3F93F8C7F8F8ULL, 0xC211A44486228686ULL
};

static __constant__ uint64_t c_adam_whirlpool_RC[10] = {
	0x4F01B887E8C62318ULL, 0x52916F79F5D2A636ULL, 0x357B0CA38E9BBC60ULL, 0x57FE4B2EC2D7E01DULL, 0xDA4AF09FE5377715ULL,
	0x856BA0B10A29C958ULL, 0x67053ECBF4105DBDULL, 0xD8957DA78B4127E4ULL, 0x9E4717DD667CEEFBULL, 0x33835AAD07BF2DCAULL
};

__device__ __forceinline__ uint64_t adam_whirlpool_elt(
	const uint64_t in[8],
	int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7)
{
	return
		                 c_adam_whirlpool_T0[(in[i0]      ) & 0xFF]       ^
		adam_rotl64(     c_adam_whirlpool_T0[(in[i1] >>  8) & 0xFF],  8)  ^
		adam_rotl64(     c_adam_whirlpool_T0[(in[i2] >> 16) & 0xFF], 16)  ^
		adam_rotl64(     c_adam_whirlpool_T0[(in[i3] >> 24) & 0xFF], 24)  ^
		adam_rotl64(     c_adam_whirlpool_T0[(in[i4] >> 32) & 0xFF], 32)  ^
		adam_rotl64(     c_adam_whirlpool_T0[(in[i5] >> 40) & 0xFF], 40)  ^
		adam_rotl64(     c_adam_whirlpool_T0[(in[i6] >> 48) & 0xFF], 48)  ^
		adam_rotl64(     c_adam_whirlpool_T0[(in[i7] >> 56)       ], 56);
}

__device__ void adam_whirlpool_round_func(const uint64_t src[8], uint64_t state[8])
{
	uint64_t n[8], h[8], tmp[8];
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		h[i] = state[i];
		n[i] = src[i] ^ h[i];
	}
	for (int r = 0; r < 10; r++) {
		tmp[0] = adam_whirlpool_elt(h, 0, 7, 6, 5, 4, 3, 2, 1) ^ c_adam_whirlpool_RC[r];
		tmp[1] = adam_whirlpool_elt(h, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp[2] = adam_whirlpool_elt(h, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp[3] = adam_whirlpool_elt(h, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp[4] = adam_whirlpool_elt(h, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp[5] = adam_whirlpool_elt(h, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp[6] = adam_whirlpool_elt(h, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp[7] = adam_whirlpool_elt(h, 7, 6, 5, 4, 3, 2, 1, 0);
		#pragma unroll
		for (int i = 0; i < 8; i++) h[i] = tmp[i];

		tmp[0] = adam_whirlpool_elt(n, 0, 7, 6, 5, 4, 3, 2, 1) ^ h[0];
		tmp[1] = adam_whirlpool_elt(n, 1, 0, 7, 6, 5, 4, 3, 2) ^ h[1];
		tmp[2] = adam_whirlpool_elt(n, 2, 1, 0, 7, 6, 5, 4, 3) ^ h[2];
		tmp[3] = adam_whirlpool_elt(n, 3, 2, 1, 0, 7, 6, 5, 4) ^ h[3];
		tmp[4] = adam_whirlpool_elt(n, 4, 3, 2, 1, 0, 7, 6, 5) ^ h[4];
		tmp[5] = adam_whirlpool_elt(n, 5, 4, 3, 2, 1, 0, 7, 6) ^ h[5];
		tmp[6] = adam_whirlpool_elt(n, 6, 5, 4, 3, 2, 1, 0, 7) ^ h[6];
		tmp[7] = adam_whirlpool_elt(n, 7, 6, 5, 4, 3, 2, 1, 0) ^ h[7];
		#pragma unroll
		for (int i = 0; i < 8; i++) n[i] = tmp[i];
	}
	#pragma unroll
	for (int i = 0; i < 8; i++)
		state[i] ^= n[i] ^ src[i];
}

__global__ void adam_gpu_whirlpool_70(uint32_t threads, uint32_t start_nonce, uint64_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint64_t w[9];
	adam_load_le64_70(w, start_nonce + tid);

	uint64_t state[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	adam_whirlpool_round_func(w, state);

	uint64_t b1[8] = {
		w[8] | (0x80ULL << 48), 0ULL, 0ULL, 0ULL,
		0ULL, 0ULL, 0ULL, adam_bswap64(560ULL)
	};
	adam_whirlpool_round_func(b1, state);
	adam_store_le64_32(g_hash, tid, state);
}

__global__ void adam_gpu_whirlpool_32(uint32_t threads, uint64_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint64_t b0[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	adam_load_le64_32(b0, g_hash, tid);
	b0[4] = 0x80ULL;

	uint64_t state[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	adam_whirlpool_round_func(b0, state);

	uint64_t b1[8] = { 0, 0, 0, 0, 0, 0, 0, adam_bswap64(256ULL) };
	adam_whirlpool_round_func(b1, state);
	adam_store_le64_32(g_hash, tid, state);
}

/* ------------------- 17. HAVAL-256/5 ------------------- */

#define ADAM_HAVAL_F1(x6, x5, x4, x3, x2, x1, x0) \
	(((x1) & ((x0) ^ (x4))) ^ ((x2) & (x5)) ^ ((x3) & (x6)) ^ (x0))
#define ADAM_HAVAL_F2(x6, x5, x4, x3, x2, x1, x0) \
	(((x2) & (((x1) & ~(x3)) ^ ((x4) & (x5)) ^ (x6) ^ (x0))) \
	 ^ ((x4) & ((x1) ^ (x5))) ^ (((x3) & (x5)) ^ (x0)))
#define ADAM_HAVAL_F3(x6, x5, x4, x3, x2, x1, x0) \
	(((x3) & (((x1) & (x2)) ^ (x6) ^ (x0))) \
	 ^ ((x1) & (x4)) ^ ((x2) & (x5)) ^ (x0))
#define ADAM_HAVAL_F4(x6, x5, x4, x3, x2, x1, x0) \
	(((x3) & (((x1) & (x2)) ^ ((x4) | (x6)) ^ (x5))) \
	 ^ ((x4) & ((~(x2) & (x5)) ^ (x1) ^ (x6) ^ (x0))) \
	 ^ ((x2) & (x6)) ^ (x0))
#define ADAM_HAVAL_F5(x6, x5, x4, x3, x2, x1, x0) \
	(((x0) & ~(((x1) & (x2) & (x3)) ^ (x5))) \
	 ^ ((x1) & (x4)) ^ ((x2) & (x5)) ^ ((x3) & (x6)))

#define ADAM_HAVAL_FP1(x6, x5, x4, x3, x2, x1, x0) ADAM_HAVAL_F1(x3, x4, x1, x0, x5, x2, x6)
#define ADAM_HAVAL_FP2(x6, x5, x4, x3, x2, x1, x0) ADAM_HAVAL_F2(x6, x2, x1, x0, x3, x4, x5)
#define ADAM_HAVAL_FP3(x6, x5, x4, x3, x2, x1, x0) ADAM_HAVAL_F3(x2, x6, x0, x4, x3, x1, x5)
#define ADAM_HAVAL_FP4(x6, x5, x4, x3, x2, x1, x0) ADAM_HAVAL_F4(x1, x5, x3, x2, x0, x4, x6)
#define ADAM_HAVAL_FP5(x6, x5, x4, x3, x2, x1, x0) ADAM_HAVAL_F5(x2, x5, x0, x6, x4, x3, x1)

#define ADAM_HAVAL_STEP(p, x7, x6, x5, x4, x3, x2, x1, x0, w, c) do { \
	uint32_t _t = ADAM_HAVAL_FP ## p(x6, x5, x4, x3, x2, x1, x0); \
	(x7) = adam_rotr32(_t, 7) + adam_rotr32((x7), 11) + (w) + (c); \
} while (0)

__device__ void adam_haval256_compress(const uint32_t in[32], uint32_t out32[8])
{
	uint32_t s0 = 0x243F6A88U, s1 = 0x85A308D3U, s2 = 0x13198A2EU, s3 = 0x03707344U;
	uint32_t s4 = 0xA4093822U, s5 = 0x299F31D0U, s6 = 0x082EFA98U, s7 = 0xEC4E6C89U;

	for (int i = 0; i < 32; i += 8) {
		ADAM_HAVAL_STEP(1, s7, s6, s5, s4, s3, s2, s1, s0, in[i + 0], 0U);
		ADAM_HAVAL_STEP(1, s6, s5, s4, s3, s2, s1, s0, s7, in[i + 1], 0U);
		ADAM_HAVAL_STEP(1, s5, s4, s3, s2, s1, s0, s7, s6, in[i + 2], 0U);
		ADAM_HAVAL_STEP(1, s4, s3, s2, s1, s0, s7, s6, s5, in[i + 3], 0U);
		ADAM_HAVAL_STEP(1, s3, s2, s1, s0, s7, s6, s5, s4, in[i + 4], 0U);
		ADAM_HAVAL_STEP(1, s2, s1, s0, s7, s6, s5, s4, s3, in[i + 5], 0U);
		ADAM_HAVAL_STEP(1, s1, s0, s7, s6, s5, s4, s3, s2, in[i + 6], 0U);
		ADAM_HAVAL_STEP(1, s0, s7, s6, s5, s4, s3, s2, s1, in[i + 7], 0U);
	}

	ADAM_HAVAL_STEP(2, s7, s6, s5, s4, s3, s2, s1, s0, in[ 5], 0x452821E6U);
	ADAM_HAVAL_STEP(2, s6, s5, s4, s3, s2, s1, s0, s7, in[14], 0x38D01377U);
	ADAM_HAVAL_STEP(2, s5, s4, s3, s2, s1, s0, s7, s6, in[26], 0xBE5466CFU);
	ADAM_HAVAL_STEP(2, s4, s3, s2, s1, s0, s7, s6, s5, in[18], 0x34E90C6CU);
	ADAM_HAVAL_STEP(2, s3, s2, s1, s0, s7, s6, s5, s4, in[11], 0xC0AC29B7U);
	ADAM_HAVAL_STEP(2, s2, s1, s0, s7, s6, s5, s4, s3, in[28], 0xC97C50DDU);
	ADAM_HAVAL_STEP(2, s1, s0, s7, s6, s5, s4, s3, s2, in[ 7], 0x3F84D5B5U);
	ADAM_HAVAL_STEP(2, s0, s7, s6, s5, s4, s3, s2, s1, in[16], 0xB5470917U);
	ADAM_HAVAL_STEP(2, s7, s6, s5, s4, s3, s2, s1, s0, in[ 0], 0x9216D5D9U);
	ADAM_HAVAL_STEP(2, s6, s5, s4, s3, s2, s1, s0, s7, in[23], 0x8979FB1BU);
	ADAM_HAVAL_STEP(2, s5, s4, s3, s2, s1, s0, s7, s6, in[20], 0xD1310BA6U);
	ADAM_HAVAL_STEP(2, s4, s3, s2, s1, s0, s7, s6, s5, in[22], 0x98DFB5ACU);
	ADAM_HAVAL_STEP(2, s3, s2, s1, s0, s7, s6, s5, s4, in[ 1], 0x2FFD72DBU);
	ADAM_HAVAL_STEP(2, s2, s1, s0, s7, s6, s5, s4, s3, in[10], 0xD01ADFB7U);
	ADAM_HAVAL_STEP(2, s1, s0, s7, s6, s5, s4, s3, s2, in[ 4], 0xB8E1AFEDU);
	ADAM_HAVAL_STEP(2, s0, s7, s6, s5, s4, s3, s2, s1, in[ 8], 0x6A267E96U);
	ADAM_HAVAL_STEP(2, s7, s6, s5, s4, s3, s2, s1, s0, in[30], 0xBA7C9045U);
	ADAM_HAVAL_STEP(2, s6, s5, s4, s3, s2, s1, s0, s7, in[ 3], 0xF12C7F99U);
	ADAM_HAVAL_STEP(2, s5, s4, s3, s2, s1, s0, s7, s6, in[21], 0x24A19947U);
	ADAM_HAVAL_STEP(2, s4, s3, s2, s1, s0, s7, s6, s5, in[ 9], 0xB3916CF7U);
	ADAM_HAVAL_STEP(2, s3, s2, s1, s0, s7, s6, s5, s4, in[17], 0x0801F2E2U);
	ADAM_HAVAL_STEP(2, s2, s1, s0, s7, s6, s5, s4, s3, in[24], 0x858EFC16U);
	ADAM_HAVAL_STEP(2, s1, s0, s7, s6, s5, s4, s3, s2, in[29], 0x636920D8U);
	ADAM_HAVAL_STEP(2, s0, s7, s6, s5, s4, s3, s2, s1, in[ 6], 0x71574E69U);
	ADAM_HAVAL_STEP(2, s7, s6, s5, s4, s3, s2, s1, s0, in[19], 0xA458FEA3U);
	ADAM_HAVAL_STEP(2, s6, s5, s4, s3, s2, s1, s0, s7, in[12], 0xF4933D7EU);
	ADAM_HAVAL_STEP(2, s5, s4, s3, s2, s1, s0, s7, s6, in[15], 0x0D95748FU);
	ADAM_HAVAL_STEP(2, s4, s3, s2, s1, s0, s7, s6, s5, in[13], 0x728EB658U);
	ADAM_HAVAL_STEP(2, s3, s2, s1, s0, s7, s6, s5, s4, in[ 2], 0x718BCD58U);
	ADAM_HAVAL_STEP(2, s2, s1, s0, s7, s6, s5, s4, s3, in[25], 0x82154AEEU);
	ADAM_HAVAL_STEP(2, s1, s0, s7, s6, s5, s4, s3, s2, in[31], 0x7B54A41DU);
	ADAM_HAVAL_STEP(2, s0, s7, s6, s5, s4, s3, s2, s1, in[27], 0xC25A59B5U);

	ADAM_HAVAL_STEP(3, s7, s6, s5, s4, s3, s2, s1, s0, in[19], 0x9C30D539U);
	ADAM_HAVAL_STEP(3, s6, s5, s4, s3, s2, s1, s0, s7, in[ 9], 0x2AF26013U);
	ADAM_HAVAL_STEP(3, s5, s4, s3, s2, s1, s0, s7, s6, in[ 4], 0xC5D1B023U);
	ADAM_HAVAL_STEP(3, s4, s3, s2, s1, s0, s7, s6, s5, in[20], 0x286085F0U);
	ADAM_HAVAL_STEP(3, s3, s2, s1, s0, s7, s6, s5, s4, in[28], 0xCA417918U);
	ADAM_HAVAL_STEP(3, s2, s1, s0, s7, s6, s5, s4, s3, in[17], 0xB8DB38EFU);
	ADAM_HAVAL_STEP(3, s1, s0, s7, s6, s5, s4, s3, s2, in[ 8], 0x8E79DCB0U);
	ADAM_HAVAL_STEP(3, s0, s7, s6, s5, s4, s3, s2, s1, in[22], 0x603A180EU);
	ADAM_HAVAL_STEP(3, s7, s6, s5, s4, s3, s2, s1, s0, in[29], 0x6C9E0E8BU);
	ADAM_HAVAL_STEP(3, s6, s5, s4, s3, s2, s1, s0, s7, in[14], 0xB01E8A3EU);
	ADAM_HAVAL_STEP(3, s5, s4, s3, s2, s1, s0, s7, s6, in[25], 0xD71577C1U);
	ADAM_HAVAL_STEP(3, s4, s3, s2, s1, s0, s7, s6, s5, in[12], 0xBD314B27U);
	ADAM_HAVAL_STEP(3, s3, s2, s1, s0, s7, s6, s5, s4, in[24], 0x78AF2FDAU);
	ADAM_HAVAL_STEP(3, s2, s1, s0, s7, s6, s5, s4, s3, in[30], 0x55605C60U);
	ADAM_HAVAL_STEP(3, s1, s0, s7, s6, s5, s4, s3, s2, in[16], 0xE65525F3U);
	ADAM_HAVAL_STEP(3, s0, s7, s6, s5, s4, s3, s2, s1, in[26], 0xAA55AB94U);
	ADAM_HAVAL_STEP(3, s7, s6, s5, s4, s3, s2, s1, s0, in[31], 0x57489862U);
	ADAM_HAVAL_STEP(3, s6, s5, s4, s3, s2, s1, s0, s7, in[15], 0x63E81440U);
	ADAM_HAVAL_STEP(3, s5, s4, s3, s2, s1, s0, s7, s6, in[ 7], 0x55CA396AU);
	ADAM_HAVAL_STEP(3, s4, s3, s2, s1, s0, s7, s6, s5, in[ 3], 0x2AAB10B6U);
	ADAM_HAVAL_STEP(3, s3, s2, s1, s0, s7, s6, s5, s4, in[ 1], 0xB4CC5C34U);
	ADAM_HAVAL_STEP(3, s2, s1, s0, s7, s6, s5, s4, s3, in[ 0], 0x1141E8CEU);
	ADAM_HAVAL_STEP(3, s1, s0, s7, s6, s5, s4, s3, s2, in[18], 0xA15486AFU);
	ADAM_HAVAL_STEP(3, s0, s7, s6, s5, s4, s3, s2, s1, in[27], 0x7C72E993U);
	ADAM_HAVAL_STEP(3, s7, s6, s5, s4, s3, s2, s1, s0, in[13], 0xB3EE1411U);
	ADAM_HAVAL_STEP(3, s6, s5, s4, s3, s2, s1, s0, s7, in[ 6], 0x636FBC2AU);
	ADAM_HAVAL_STEP(3, s5, s4, s3, s2, s1, s0, s7, s6, in[21], 0x2BA9C55DU);
	ADAM_HAVAL_STEP(3, s4, s3, s2, s1, s0, s7, s6, s5, in[10], 0x741831F6U);
	ADAM_HAVAL_STEP(3, s3, s2, s1, s0, s7, s6, s5, s4, in[23], 0xCE5C3E16U);
	ADAM_HAVAL_STEP(3, s2, s1, s0, s7, s6, s5, s4, s3, in[11], 0x9B87931EU);
	ADAM_HAVAL_STEP(3, s1, s0, s7, s6, s5, s4, s3, s2, in[ 5], 0xAFD6BA33U);
	ADAM_HAVAL_STEP(3, s0, s7, s6, s5, s4, s3, s2, s1, in[ 2], 0x6C24CF5CU);

	ADAM_HAVAL_STEP(4, s7, s6, s5, s4, s3, s2, s1, s0, in[24], 0x7A325381U);
	ADAM_HAVAL_STEP(4, s6, s5, s4, s3, s2, s1, s0, s7, in[ 4], 0x28958677U);
	ADAM_HAVAL_STEP(4, s5, s4, s3, s2, s1, s0, s7, s6, in[ 0], 0x3B8F4898U);
	ADAM_HAVAL_STEP(4, s4, s3, s2, s1, s0, s7, s6, s5, in[14], 0x6B4BB9AFU);
	ADAM_HAVAL_STEP(4, s3, s2, s1, s0, s7, s6, s5, s4, in[ 2], 0xC4BFE81BU);
	ADAM_HAVAL_STEP(4, s2, s1, s0, s7, s6, s5, s4, s3, in[ 7], 0x66282193U);
	ADAM_HAVAL_STEP(4, s1, s0, s7, s6, s5, s4, s3, s2, in[28], 0x61D809CCU);
	ADAM_HAVAL_STEP(4, s0, s7, s6, s5, s4, s3, s2, s1, in[23], 0xFB21A991U);
	ADAM_HAVAL_STEP(4, s7, s6, s5, s4, s3, s2, s1, s0, in[26], 0x487CAC60U);
	ADAM_HAVAL_STEP(4, s6, s5, s4, s3, s2, s1, s0, s7, in[ 6], 0x5DEC8032U);
	ADAM_HAVAL_STEP(4, s5, s4, s3, s2, s1, s0, s7, s6, in[30], 0xEF845D5DU);
	ADAM_HAVAL_STEP(4, s4, s3, s2, s1, s0, s7, s6, s5, in[20], 0xE98575B1U);
	ADAM_HAVAL_STEP(4, s3, s2, s1, s0, s7, s6, s5, s4, in[18], 0xDC262302U);
	ADAM_HAVAL_STEP(4, s2, s1, s0, s7, s6, s5, s4, s3, in[25], 0xEB651B88U);
	ADAM_HAVAL_STEP(4, s1, s0, s7, s6, s5, s4, s3, s2, in[19], 0x23893E81U);
	ADAM_HAVAL_STEP(4, s0, s7, s6, s5, s4, s3, s2, s1, in[ 3], 0xD396ACC5U);
	ADAM_HAVAL_STEP(4, s7, s6, s5, s4, s3, s2, s1, s0, in[22], 0x0F6D6FF3U);
	ADAM_HAVAL_STEP(4, s6, s5, s4, s3, s2, s1, s0, s7, in[11], 0x83F44239U);
	ADAM_HAVAL_STEP(4, s5, s4, s3, s2, s1, s0, s7, s6, in[31], 0x2E0B4482U);
	ADAM_HAVAL_STEP(4, s4, s3, s2, s1, s0, s7, s6, s5, in[21], 0xA4842004U);
	ADAM_HAVAL_STEP(4, s3, s2, s1, s0, s7, s6, s5, s4, in[ 8], 0x69C8F04AU);
	ADAM_HAVAL_STEP(4, s2, s1, s0, s7, s6, s5, s4, s3, in[27], 0x9E1F9B5EU);
	ADAM_HAVAL_STEP(4, s1, s0, s7, s6, s5, s4, s3, s2, in[12], 0x21C66842U);
	ADAM_HAVAL_STEP(4, s0, s7, s6, s5, s4, s3, s2, s1, in[ 9], 0xF6E96C9AU);
	ADAM_HAVAL_STEP(4, s7, s6, s5, s4, s3, s2, s1, s0, in[ 1], 0x670C9C61U);
	ADAM_HAVAL_STEP(4, s6, s5, s4, s3, s2, s1, s0, s7, in[29], 0xABD388F0U);
	ADAM_HAVAL_STEP(4, s5, s4, s3, s2, s1, s0, s7, s6, in[ 5], 0x6A51A0D2U);
	ADAM_HAVAL_STEP(4, s4, s3, s2, s1, s0, s7, s6, s5, in[15], 0xD8542F68U);
	ADAM_HAVAL_STEP(4, s3, s2, s1, s0, s7, s6, s5, s4, in[17], 0x960FA728U);
	ADAM_HAVAL_STEP(4, s2, s1, s0, s7, s6, s5, s4, s3, in[10], 0xAB5133A3U);
	ADAM_HAVAL_STEP(4, s1, s0, s7, s6, s5, s4, s3, s2, in[16], 0x6EEF0B6CU);
	ADAM_HAVAL_STEP(4, s0, s7, s6, s5, s4, s3, s2, s1, in[13], 0x137A3BE4U);

	ADAM_HAVAL_STEP(5, s7, s6, s5, s4, s3, s2, s1, s0, in[27], 0xBA3BF050U);
	ADAM_HAVAL_STEP(5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 3], 0x7EFB2A98U);
	ADAM_HAVAL_STEP(5, s5, s4, s3, s2, s1, s0, s7, s6, in[21], 0xA1F1651DU);
	ADAM_HAVAL_STEP(5, s4, s3, s2, s1, s0, s7, s6, s5, in[26], 0x39AF0176U);
	ADAM_HAVAL_STEP(5, s3, s2, s1, s0, s7, s6, s5, s4, in[17], 0x66CA593EU);
	ADAM_HAVAL_STEP(5, s2, s1, s0, s7, s6, s5, s4, s3, in[11], 0x82430E88U);
	ADAM_HAVAL_STEP(5, s1, s0, s7, s6, s5, s4, s3, s2, in[20], 0x8CEE8619U);
	ADAM_HAVAL_STEP(5, s0, s7, s6, s5, s4, s3, s2, s1, in[29], 0x456F9FB4U);
	ADAM_HAVAL_STEP(5, s7, s6, s5, s4, s3, s2, s1, s0, in[19], 0x7D84A5C3U);
	ADAM_HAVAL_STEP(5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 0], 0x3B8B5EBEU);
	ADAM_HAVAL_STEP(5, s5, s4, s3, s2, s1, s0, s7, s6, in[12], 0xE06F75D8U);
	ADAM_HAVAL_STEP(5, s4, s3, s2, s1, s0, s7, s6, s5, in[ 7], 0x85C12073U);
	ADAM_HAVAL_STEP(5, s3, s2, s1, s0, s7, s6, s5, s4, in[13], 0x401A449FU);
	ADAM_HAVAL_STEP(5, s2, s1, s0, s7, s6, s5, s4, s3, in[ 8], 0x56C16AA6U);
	ADAM_HAVAL_STEP(5, s1, s0, s7, s6, s5, s4, s3, s2, in[31], 0x4ED3AA62U);
	ADAM_HAVAL_STEP(5, s0, s7, s6, s5, s4, s3, s2, s1, in[10], 0x363F7706U);
	ADAM_HAVAL_STEP(5, s7, s6, s5, s4, s3, s2, s1, s0, in[ 5], 0x1BFEDF72U);
	ADAM_HAVAL_STEP(5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 9], 0x429B023DU);
	ADAM_HAVAL_STEP(5, s5, s4, s3, s2, s1, s0, s7, s6, in[14], 0x37D0D724U);
	ADAM_HAVAL_STEP(5, s4, s3, s2, s1, s0, s7, s6, s5, in[30], 0xD00A1248U);
	ADAM_HAVAL_STEP(5, s3, s2, s1, s0, s7, s6, s5, s4, in[18], 0xDB0FEAD3U);
	ADAM_HAVAL_STEP(5, s2, s1, s0, s7, s6, s5, s4, s3, in[ 6], 0x49F1C09BU);
	ADAM_HAVAL_STEP(5, s1, s0, s7, s6, s5, s4, s3, s2, in[28], 0x075372C9U);
	ADAM_HAVAL_STEP(5, s0, s7, s6, s5, s4, s3, s2, s1, in[24], 0x80991B7BU);
	ADAM_HAVAL_STEP(5, s7, s6, s5, s4, s3, s2, s1, s0, in[ 2], 0x25D479D8U);
	ADAM_HAVAL_STEP(5, s6, s5, s4, s3, s2, s1, s0, s7, in[23], 0xF6E8DEF7U);
	ADAM_HAVAL_STEP(5, s5, s4, s3, s2, s1, s0, s7, s6, in[16], 0xE3FE501AU);
	ADAM_HAVAL_STEP(5, s4, s3, s2, s1, s0, s7, s6, s5, in[22], 0xB6794C3BU);
	ADAM_HAVAL_STEP(5, s3, s2, s1, s0, s7, s6, s5, s4, in[ 4], 0x976CE0BDU);
	ADAM_HAVAL_STEP(5, s2, s1, s0, s7, s6, s5, s4, s3, in[ 1], 0x04C006BAU);
	ADAM_HAVAL_STEP(5, s1, s0, s7, s6, s5, s4, s3, s2, in[25], 0xC1A94FB6U);
	ADAM_HAVAL_STEP(5, s0, s7, s6, s5, s4, s3, s2, s1, in[15], 0x409F60C4U);

	out32[0] = s0 + 0x243F6A88U; out32[1] = s1 + 0x85A308D3U;
	out32[2] = s2 + 0x13198A2EU; out32[3] = s3 + 0x03707344U;
	out32[4] = s4 + 0xA4093822U; out32[5] = s5 + 0x299F31D0U;
	out32[6] = s6 + 0x082EFA98U; out32[7] = s7 + 0xEC4E6C89U;
}

#undef ADAM_HAVAL_F1
#undef ADAM_HAVAL_F2
#undef ADAM_HAVAL_F3
#undef ADAM_HAVAL_F4
#undef ADAM_HAVAL_F5
#undef ADAM_HAVAL_FP1
#undef ADAM_HAVAL_FP2
#undef ADAM_HAVAL_FP3
#undef ADAM_HAVAL_FP4
#undef ADAM_HAVAL_FP5
#undef ADAM_HAVAL_STEP

__global__ void adam_gpu_haval256_70(uint32_t threads, uint32_t start_nonce, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t buf[32];
	adam_load_le32_70(buf, start_nonce + tid);
	buf[17] |= 0x00010000U;
	for (int i = 18; i < 32; i++) buf[i] = 0U;
	buf[29] = 0x40290000U;
	buf[30] = 560U;
	uint32_t out32[8];
	adam_haval256_compress(buf, out32);
	adam_store_le32_32(g_hash, tid, out32);
}

__global__ void adam_gpu_haval256_32(uint32_t threads, uint32_t *g_hash)
{
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= threads)
		return;
	uint32_t buf[32];
	adam_load_le32_32(buf, g_hash, tid);
	buf[8] = 0x00000001U;
	for (int i = 9; i < 32; i++) buf[i] = 0U;
	buf[29] = 0x40290000U;
	buf[30] = 256U;
	uint32_t out32[8];
	adam_haval256_compress(buf, out32);
	adam_store_le32_32(g_hash, tid, out32);
}

#endif
