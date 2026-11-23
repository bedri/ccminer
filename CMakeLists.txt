cmake_minimum_required(VERSION 3.24)
project(ccminer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(jansson CONFIG REQUIRED)
find_package(PThreads4W REQUIRED)
find_package(OpenMP REQUIRED)

# Stage minimal Windows POSIX shims without adding compat/ itself to the include
# search path (compat/inttypes.h and compat/stdbool.h are legacy VS2010 stubs
# that break VS2022 C/C++ standard headers).
set(WIN_COMPAT_DIR "${CMAKE_BINARY_DIR}/win_compat")
file(MAKE_DIRECTORY "${WIN_COMPAT_DIR}/sys")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/compat/ccminer-config.h"
  "${WIN_COMPAT_DIR}/ccminer-config.h"
  COPYONLY
)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/compat/getopt/getopt.h"
  "${WIN_COMPAT_DIR}/getopt.h"
  COPYONLY
)
file(WRITE "${WIN_COMPAT_DIR}/unistd.h" "#pragma once\n#include \"compat/getopt/getopt.h\"\n")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/compat/sys/time.h" _compat_sys_time_h)
file(WRITE "${WIN_COMPAT_DIR}/sys/time.h" "#pragma once\n#include <winsock2.h>\n${_compat_sys_time_h}")

set(CCMINER_SOURCES
  crc32.c
  hefty1.c
  ccminer.cpp
  pools.cpp
  util.cpp
  bench.cpp
  bignum.cpp
  api.cpp
  hashlog.cpp
  nvml.cpp
  stats.cpp
  sysinfos.cpp
  cuda.cpp
  nvsettings.cpp
  stubs.cpp
  sph/fugue.c
  groestlcoin.cpp
  cuda_groestlcoin.cu
  myriadgroestl.cpp
  cuda_myriadgroestl.cu
  Algo256/cuda_bmw256.cu
  Algo256/cuda_cubehash256.cu
  Algo256/cuda_blake256.cu
  Algo256/cuda_keccak256_sm3.cu
  Algo256/cuda_keccak256.cu
  Algo256/cuda_skein256.cu
  Algo256/blake256.cu
  Algo256/decred.cu
  Algo256/vanilla.cu
  Algo256/keccak256.cu
  Algo256/blake2s.cu
  sph/blake2s.c
  Algo256/bmw.cu
  Algo256/cuda_bmw.cu
  blake2b.cu
  crypto/xmr-rpc.cpp
  crypto/wildkeccak-cpu.cpp
  crypto/wildkeccak.cu
  crypto/cryptolight.cu
  crypto/cryptolight-core.cu
  crypto/cryptolight-cpu.cpp
  crypto/cryptonight.cu
  crypto/cryptonight-core.cu
  crypto/cryptonight-extra.cu
  crypto/cryptonight-cpu.cpp
  crypto/oaes_lib.cpp
  crypto/aesb.cpp
  crypto/cpu/c_keccak.c
  JHA/jha.cu
  JHA/jackpotcoin.cu
  JHA/cuda_jha_keccak512.cu
  JHA/cuda_jha_compactionTest.cu
  cuda_checkhash.cu
  quark/cuda_jh512.cu
  quark/cuda_quark_blake512.cu
  quark/cuda_quark_groestl512.cu
  quark/cuda_skein512.cu
  quark/cuda_bmw512.cu
  quark/cuda_quark_keccak512.cu
  quark/nist5.cu
  quark/quarkcoin.cu
  quark/cuda_quark_compactionTest.cu
  pentablake.cu
  skein.cu
  cuda_skeincoin.cu
  skein2.cpp
  zr5.cu
  skunk/skunk.cu
  skunk/cuda_skunk.cu
  skunk/cuda_skunk_streebog.cu
  sha256/sha256d.cu
  sha256/cuda_sha256d.cu
  sha256/sha256t.cu
  sha256/cuda_sha256t.cu
  sha256/sha256q.cu
  sha256/cuda_sha256q.cu
  sia/sia.cu
  sia/sia-rpc.cpp
  sph/blake2b.c
  sph/bmw.c
  sph/blake.c
  sph/groestl.c
  sph/jh.c
  sph/keccak.c
  sph/skein.c
  sph/cubehash.c
  sph/echo.c
  sph/luffa.c
  sph/sha2.c
  sph/shavite.c
  sph/simd.c
  sph/hamsi.c
  sph/hamsi_helper.c
  sph/streebog.c
  sph/shabal.c
  sph/whirlpool.c
  sph/sha2big.c
  sph/haval.c
  sph/ripemd.c
  sph/sph_sha2.c
  polytimos.cu
  cuda_adam.cu
  adam_cpu.cpp
  lbry/lbry.cu
  lbry/cuda_sha256_lbry.cu
  lbry/cuda_sha512_lbry.cu
  lbry/cuda_lbry_merged.cu
  qubit/qubit.cu
  qubit/qubit_luffa512.cu
  qubit/deep.cu
  qubit/luffa.cu
  tribus/tribus.cu
  tribus/cuda_echo512_final.cu
  x11/x11.cu
  x12/x12.cu
  x11/fresh.cu
  x11/cuda_x11_luffa512.cu
  x11/cuda_x11_cubehash512.cu
  x11/cuda_x11_shavite512.cu
  x11/cuda_x11_simd512.cu
  x11/cuda_x11_echo.cu
  x11/exosis.cu
  x11/cuda_x11_luffa512_Cubehash.cu
  x11/x11evo.cu
  x11/timetravel.cu
  x11/bitcore.cu
  x13/x13.cu
  x13/cuda_x13_hamsi512.cu
  x13/cuda_x13_fugue512.cu
  x13/hsr.cu
  x13/cuda_hsr_sm3.cu
  x13/sm3.c
  x15/x14.cu
  x15/x15.cu
  x15/cuda_x14_shabal512.cu
  x15/cuda_x15_whirlpool.cu
  x15/whirlpool.cu
  x15/cuda_x15_whirlpool_sm3.cu
  x17/x17.cu
  x17/hmq17.cu
  x17/sonoa.cu
  x17/cuda_x17_haval256.cu
  x17/cuda_x17_sha512.cu
  phi/phi.cu
  phi/phi2.cu
  phi/cuda_phi2.cu
  phi/cuda_phi2_cubehash512.cu
  x11/cuda_streebog_maxwell.cu
  x11/c11.cu
  x11/s3.cu
  x11/sib.cu
  x11/veltor.cu
  x11/cuda_streebog.cu
  compat/getopt/getopt_long.c
  compat/gettimeofday.c
  compat/winansi.c
)

add_executable(ccminer ${CCMINER_SOURCES})

set_target_properties(ccminer PROPERTIES
  CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;120"
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(ccminer PRIVATE
  "${WIN_COMPAT_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CUDAToolkit_INCLUDE_DIRS}"
)

target_compile_definitions(ccminer PRIVATE
  WIN32
  WIN32_LEAN_AND_MEAN
  NOMINMAX
  _CRT_SECURE_NO_WARNINGS
  HAVE_STRUCT_TIMESPEC
  cudaThreadSynchronize=cudaDeviceSynchronize
)

target_compile_options(ccminer PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/openmp>
  $<$<COMPILE_LANGUAGE:C,CXX>:/MP>
)

set_source_files_properties(compat/gettimeofday.c PROPERTIES
  COMPILE_OPTIONS "/FIwinsock2.h"
)

# Per-file CUDA register caps matching Makefile.am (default 128 except overrides)
set(_cuda_no_cap
  crypto/cryptonight-extra.cu
  x11/cuda_x11_simd512.cu
)
set(_cuda_cap_64
  Algo256/blake256.cu
  crypto/cryptonight-core.cu
  skein.cu
)
set(_cuda_cap_72
  x13/cuda_x13_hamsi512.cu
)
set(_cuda_cap_76
  Algo256/cuda_bmw.cu
  x11/cuda_x11_luffa512_Cubehash.cu
)
set(_cuda_cap_80
  qubit/qubit_luffa512.cu
  x11/cuda_x11_luffa512.cu
  x17/cuda_x17_sha512.cu
  quark/cuda_quark_blake512.cu
  quark/cuda_jh512.cu
  quark/cuda_quark_compactionTest.cu
  JHA/cuda_jha_compactionTest.cu
)
set(_cuda_cap_88
  quark/cuda_quark_keccak512.cu
)

foreach(_src IN LISTS CCMINER_SOURCES)
  if(_src MATCHES "\\.cu$")
    if(_src IN_LIST _cuda_no_cap)
      # no --maxrregcount override
    elseif(_src IN_LIST _cuda_cap_64)
      set_source_files_properties("${_src}" PROPERTIES COMPILE_OPTIONS "--maxrregcount=64")
    elseif(_src IN_LIST _cuda_cap_72)
      set_source_files_properties("${_src}" PROPERTIES COMPILE_OPTIONS "--maxrregcount=72")
    elseif(_src IN_LIST _cuda_cap_76)
      set_source_files_properties("${_src}" PROPERTIES COMPILE_OPTIONS "--maxrregcount=76")
    elseif(_src IN_LIST _cuda_cap_80)
      set_source_files_properties("${_src}" PROPERTIES COMPILE_OPTIONS "--maxrregcount=80")
    elseif(_src IN_LIST _cuda_cap_88)
      set_source_files_properties("${_src}" PROPERTIES COMPILE_OPTIONS "--maxrregcount=88")
    else()
      set_source_files_properties("${_src}" PROPERTIES COMPILE_OPTIONS "--maxrregcount=128")
    endif()
  endif()
endforeach()

target_link_libraries(ccminer PRIVATE
  CUDA::cudart_static
  CURL::libcurl
  OpenSSL::SSL
  OpenSSL::Crypto
  ZLIB::ZLIB
  jansson::jansson
  PThreads4W::PThreads4W
  ws2_32
  wldap32
  crypt32
  normaliz
  winmm
  advapi32
  shell32
  user32
  gdi32
)
