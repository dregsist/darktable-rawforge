/*
 * Standalone AMaZE demosaic test harness.
 * Usage: test_amaze <w> <h> <filters_hex> <input.bin> <output.bin>
 */

/* Minimal stubs for C++ build (avoid pulling in stubs.h C-style inlines) */
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>

#ifndef MAX
#define MAX(a,b) ((a)>(b)?(a):(b))
#endif
#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

/* Bayer FC */
static inline int FC(const size_t row, const size_t col, const uint32_t filters)
{
  return (int)(filters >> (((row << 1 & 14) + (col & 1)) << 1) & 3);
}

static inline float sqrf(const float a) { return a*a; }
static inline float interpolatef(const float a, const float b, const float c)
{
  return a * (b - c) + c;
}

/* darktable compiler macros */
#define G_BEGIN_DECLS
#define G_END_DECLS
#define AMAZETS 160

/* OpenMP/SIMD macros: no-ops for single-thread test build */
#define DT_OMP_PRAGMA(...)
#define DT_OMP_DECLARE_SIMD(...)
#define DT_OMP_FOR(...)
#define DT_OMP_FOR_SIMD(...)

#include "../darktable/src/iop/demosaicing/amaze.cc"

int main(int argc, char *argv[])
{
  if(argc != 6)
  {
    std::fprintf(stderr, "Usage: %s <w> <h> <filters_hex> <input.bin> <output.bin>\n", argv[0]);
    return 1;
  }
  const int width = std::atoi(argv[1]);
  const int height = std::atoi(argv[2]);
  const uint32_t filters = (uint32_t)std::strtoul(argv[3], NULL, 16);
  const size_t npix = (size_t)width * height;

  float *in_buf = (float *)std::malloc(npix * sizeof(float));
  std::FILE *f = std::fopen(argv[4], "rb");
  if(!f || std::fread(in_buf, sizeof(float), npix, f) != npix) return 1;
  std::fclose(f);

  float *out_buf = (float *)std::calloc(npix * 4, sizeof(float));
  amaze_demosaic(in_buf, out_buf, width, height, filters, 1.0f);

  f = std::fopen(argv[5], "wb");
  if(!f || std::fwrite(out_buf, sizeof(float), npix*4, f) != npix*4) return 1;
  std::fclose(f);
  std::fprintf(stderr, "OK: AMaZE %dx%d -> %s\n", width, height, argv[5]);
  std::free(in_buf); std::free(out_buf);
  return 0;
}
