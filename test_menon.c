/*
 * Standalone Menon DDFAPD demosaic test harness.
 * Reads a binary float32 CFA file, runs menon_demosaic(), writes binary float32 RGBA output.
 *
 * Usage:
 *   test_menon <width> <height> <filters_hex> <input.bin> <output.bin>
 *
 * Input  (.bin): width*height float32 values, CFA mosaic
 * Output (.bin): width*height*4 float32 values, RGBA (R,G,B,0)
 *
 * Filters values (uint32 hex):
 *   RGGB = 0x94949494  GRBG = 0x49494949
 *   GBRG = 0x61616161  BGGR = 0x16161616
 */

#include "stubs.h"

/* Pull in the Menon implementation directly */
#include "../darktable/src/iop/demosaicing/menon.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_float_bin(const char *path, float **buf, size_t n)
{
  FILE *f = fopen(path, "rb");
  if(!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
  *buf = (float *)malloc(n * sizeof(float));
  if(!*buf) { fclose(f); return -1; }
  if(fread(*buf, sizeof(float), n, f) != n)
  {
    fprintf(stderr, "Read error %s\n", path);
    free(*buf); *buf = NULL; fclose(f); return -1;
  }
  fclose(f);
  return 0;
}

static int write_float_bin(const char *path, const float *buf, size_t n)
{
  FILE *f = fopen(path, "wb");
  if(!f) { fprintf(stderr, "Cannot open %s for write\n", path); return -1; }
  if(fwrite(buf, sizeof(float), n, f) != n)
  {
    fprintf(stderr, "Write error %s\n", path);
    fclose(f); return -1;
  }
  fclose(f);
  return 0;
}

int main(int argc, char *argv[])
{
  if(argc < 6 || argc > 7)
  {
    fprintf(stderr, "Usage: %s <width> <height> <filters_hex> <input.bin> <output.bin> [refining_step]\n", argv[0]);
    fprintf(stderr, "  filters: RGGB=0x94949494 GRBG=0x61616161 GBRG=0x49494949 BGGR=0x16161616\n");
    fprintf(stderr, "  refining_step: 0=off, 1=on (default: 1)\n");
    return 1;
  }

  const int width   = atoi(argv[1]);
  const int height  = atoi(argv[2]);
  const uint32_t filters = (uint32_t)strtoul(argv[3], NULL, 16);
  const char *in_path  = argv[4];
  const char *out_path = argv[5];
  const int refining_step = (argc >= 7) ? atoi(argv[6]) : 1;

  if(width <= 0 || height <= 0)
  {
    fprintf(stderr, "Invalid dimensions %dx%d\n", width, height);
    return 1;
  }

  const size_t npix = (size_t)width * height;

  /* Read CFA */
  float *in_buf = NULL;
  if(read_float_bin(in_path, &in_buf, npix) != 0) return 1;

  /* Allocate output (RGBA = 4 floats per pixel) */
  float *out_buf = (float *)calloc(npix * 4, sizeof(float));
  if(!out_buf) { fprintf(stderr, "OOM\n"); free(in_buf); return 1; }

  /* Run Menon demosaic */
  menon_demosaic(out_buf, in_buf, width, height, filters, refining_step);

  /* Write output */
  if(write_float_bin(out_path, out_buf, npix * 4) != 0)
  {
    free(in_buf); free(out_buf); return 1;
  }

  fprintf(stderr, "OK: %dx%d filters=0x%08X refining=%d -> %s\n", width, height, filters, refining_step, out_path);
  free(in_buf);
  free(out_buf);
  return 0;
}
