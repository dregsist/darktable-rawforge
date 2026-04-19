/*
 * Minimal darktable stubs for standalone Menon/ARI demosaic test harness.
 * Include this BEFORE including menon.c or ari.c.
 */
#pragma once

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

/* ---- GLib types ---- */
typedef int gboolean;
#define TRUE  1
#define FALSE 0

/* ---- SIMD alignment ---- */
#ifdef _MSC_VER
#define DT_ALIGNED_PIXEL __declspec(align(16))
#else
#define DT_ALIGNED_PIXEL __attribute__((aligned(16)))
#endif
typedef DT_ALIGNED_PIXEL float dt_aligned_pixel_t[4];

/* ---- Image copy helper ---- */
static inline void dt_iop_image_copy_by_size(float *const restrict out,
                                              const float *const restrict in,
                                              const int width, const int height,
                                              const int ch)
{
  memcpy(out, in, (size_t)width * height * ch * sizeof(float));
}

/* ---- Utility macros ---- */
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))

/* ---- Bayer pattern: FC macro (from darktable imageop_math.h) ---- */
static inline int FC(const size_t row, const size_t col, const uint32_t filters)
{
  return (int)(filters >> (((row << 1 & 14) + (col & 1)) << 1) & 3);
}

/* Channel indices (from darktable imageop_math.h) */
#define RED   0
#define GREEN 1
#define BLUE  2
#define ALPHA 3

/* xtrans stub: only Bayer (filters!=9) is tested, so this is never called */
static inline int FCNxtrans(const int row, const int col, const uint8_t (*const xtrans)[6])
{
  (void)row; (void)col; (void)xtrans;
  return 1; /* stub: return GREEN */
}

/* ---- Math helpers (from darktable common/math.h) ---- */
static inline float sqrf(const float a) { return a * a; }
/* interpolatef(a,b,c) = a*b + (1-a)*c */
static inline float interpolatef(const float a, const float b, const float c)
{
  return a * (b - c) + c;
}

/* ---- GCC branch-prediction hint: no-op on MSVC ---- */
#ifndef __builtin_expect
#define __builtin_expect(x, y) (x)
#endif

/* ---- Memory allocation (aligned, 64-byte) ---- */
#ifdef _WIN32
#include <malloc.h>
static inline void *_dt_alloc_aligned(const size_t size)
{
  return _aligned_malloc(size, 64);
}
static inline void _dt_free_align(void *ptr)
{
  _aligned_free(ptr);
}
#else
static inline void *_dt_alloc_aligned(const size_t size)
{
  void *ptr = NULL;
  posix_memalign(&ptr, 64, size);
  return ptr;
}
static inline void _dt_free_align(void *ptr)
{
  free(ptr);
}
#endif

static inline float *dt_alloc_align_float(const size_t nfloats)
{
  return (float *)_dt_alloc_aligned(nfloats * sizeof(float));
}
static inline float *dt_calloc_align_float(const size_t nfloats)
{
  float *buf = (float *)_dt_alloc_aligned(nfloats * sizeof(float));
  if(buf) memset(buf, 0, nfloats * sizeof(float));
  return buf;
}

/* RCD tile size */
#ifndef DT_RCD_TILESIZE
#define DT_RCD_TILESIZE 112
#endif
static inline uint8_t *dt_alloc_aligned_uint8(const size_t n)
{
  return (uint8_t *)_dt_alloc_aligned(n);
}
/* dt_alloc_aligned() signature: returns void* for given byte size */
static inline void *dt_alloc_aligned(const size_t size)
{
  return _dt_alloc_aligned(size);
}
#define dt_free_align(ptr) _dt_free_align(ptr)

/* ---- OpenMP: real pragmas when compiled with /openmp, no-op otherwise ---- */
#ifdef _OPENMP
  /* MSVC uses __pragma(); GCC/Clang use _Pragma(). */
  #if defined(_MSC_VER)
    #define DT_OMP_PRAGMA(ARGS)     __pragma(omp ARGS)
    #define DT_OMP_FOR(...)         __pragma(omp parallel for)
    #define DT_OMP_FOR_SIMD(...)    __pragma(omp parallel for)
  #else
    #define DT_OMP_PRAGMA(ARGS)     _Pragma(#ARGS)
    #define DT_OMP_FOR(...)         _Pragma("omp parallel for")
    #define DT_OMP_FOR_SIMD(...)    _Pragma("omp parallel for simd")
  #endif
#else
  #define DT_OMP_PRAGMA(...)        /* no-op: single-threaded */
  #define DT_OMP_FOR(...)           /* no-op */
  #define DT_OMP_FOR_SIMD(...)      /* no-op */
#endif
#define DT_OMP_DECLARE_SIMD(...)    /* no-op (auto-vectorizer hint only) */
#define dt_omp_firstprivate(...)    /* no-op (only used inside DT_OMP_PRAGMA) */
#define dt_omp_nontemporal(...)     /* no-op: nontemporal store hint */

/* ---- Pixel channel iteration (4-channel RGBA float output) ---- */
#define DT_PIXEL_SIMD_CHANNELS 4
#define for_each_channel(_var, ...) for (size_t _var = 0; _var < DT_PIXEL_SIMD_CHANNELS; _var++)
#define for_four_channels(_var, ...) for (size_t _var = 0; _var < 4; _var++)
#define for_three_channels(_var, ...) for (size_t _var = 0; _var < 3; _var++)

/* ---- VNG-specific stubs ---- */
/* fcol: Bayer (filters!=9) -> FC(); xtrans path never taken in tests */
static inline int fcol(const int row, const int col, const uint32_t filters,
                       const uint8_t (*const xtrans)[6])
{
  (void)xtrans;
  if(filters == 9u) return 1; /* unreachable in Bayer tests */
  return FC((size_t)row, (size_t)col, filters);
}

/* Bayer filters are never 4-bayer (CYGM/RGBE) in our tests */
#define FILTERS_ARE_4BAYER(f) (0)

/* SIMD vector helpers used by VNG linear interpolation */
static inline void dt_vector_clipneg(float *p)
{
  for(int c = 0; c < 4; c++) if(p[c] < 0.0f) p[c] = 0.0f;
}
static inline void dt_vector_max(float *to, const float *a, const float *b)
{
  for(int c = 0; c < 4; c++) to[c] = a[c] > b[c] ? a[c] : b[c];
}

/* dt_print stub (only hit on allocation failure, which won't happen in tests) */
#define DT_DEBUG_ALWAYS 0
static inline void dt_print(int level, const char *fmt, ...)
{
  (void)level; (void)fmt;
}

/* ---- LMMSE-specific stubs ---- */
#ifndef CLAMPF
#define CLAMPF(a, mn, mx) ((a) < (mn) ? (mn) : ((a) > (mx) ? (mx) : (a)))
#endif

/* GCC __attribute__ -> MSVC no-op on array declarations (alignment loss is harmless) */
#ifdef _MSC_VER
#ifndef __attribute__
#define __attribute__(x)
#endif
#endif

/* median-of-9 (sorts 9 floats and returns the 5th, 0-indexed 4th) */
static inline float median9f(const float *p)
{
  float a[9]; for(int i = 0; i < 9; i++) a[i] = p[i];
  for(int i = 0; i < 5; i++)
  {
    int m = i;
    for(int j = i + 1; j < 9; j++) if(a[j] < a[m]) m = j;
    float t = a[i]; a[i] = a[m]; a[m] = t;
  }
  return a[4];
}
