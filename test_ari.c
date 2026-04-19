/*
 * Standalone ARI demosaic test harness.
 * Reads float32 CFA bin, runs ari_demosaic(), writes float32 RGBA output.
 *
 * Usage:
 *   test_ari <w> <h> <filters_hex> <input.bin> <output.bin> [quality]
 *   quality: 1 (HA), 2 (MLRI+HA adaptive), 3 (MLRI+HA+RI adaptive). Default 2.
 */

#include "stubs.h"

#include "../darktable/src/iop/demosaicing/basics.c"
#include "../darktable/src/iop/demosaicing/ari.c"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  if(argc < 6 || argc > 7)
  {
    fprintf(stderr, "Usage: %s <w> <h> <filters_hex> <input.bin> <output.bin> [quality]\n", argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const uint32_t filters = (uint32_t)strtoul(argv[3], NULL, 16);
  const int quality = (argc == 7) ? atoi(argv[6]) : 2;
  const size_t npix = (size_t)width * height;

  float *in_buf = (float *)malloc(npix * sizeof(float));
  if(!in_buf) return 1;
  FILE *f = fopen(argv[4], "rb");
  if(!f || fread(in_buf, sizeof(float), npix, f) != npix) { fprintf(stderr, "read err\n"); return 1; }
  fclose(f);

  float *out_buf = (float *)calloc(npix * 4, sizeof(float));
  if(!out_buf) return 1;

  #include <omp.h>
  const double _t0 = omp_get_wtime();
  ari_demosaic(out_buf, in_buf, width, height, filters, quality);
  const double ari_secs = omp_get_wtime() - _t0;
  fprintf(stderr, "ari_demosaic: %.4f s (wall)\n", ari_secs);

  f = fopen(argv[5], "wb");
  if(!f || fwrite(out_buf, sizeof(float), npix * 4, f) != npix * 4) { fprintf(stderr, "write err\n"); return 1; }
  fclose(f);
  fprintf(stderr, "OK: %dx%d filters=0x%08X quality=%d -> %s\n", width, height, filters, quality, argv[5]);
  free(in_buf); free(out_buf);
  return 0;
}
