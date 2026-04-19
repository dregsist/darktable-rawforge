/*
 * Standalone VNG demosaic test harness.
 * Usage: test_vng <w> <h> <filters_hex> <input.bin> <output.bin>
 */
#include "stubs.h"
#include "../darktable/src/iop/demosaicing/basics.c"
#include "../darktable/src/iop/demosaicing/vng.c"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  if(argc < 6 || argc > 7)
  {
    fprintf(stderr, "Usage: %s <w> <h> <filters_hex> <input.bin> <output.bin> [only_linear]\n", argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]), height = atoi(argv[2]);
  const uint32_t filters = (uint32_t)strtoul(argv[3], NULL, 16);
  const int only_linear = (argc == 7) ? atoi(argv[6]) : 0;
  const size_t npix = (size_t)width * height;
  float *in_buf = (float *)malloc(npix * sizeof(float));
  FILE *f = fopen(argv[4], "rb");
  if(!f || fread(in_buf, sizeof(float), npix, f) != npix) return 1;
  fclose(f);
  float *out_buf = (float *)calloc(npix * 4, sizeof(float));
  vng_interpolate(out_buf, in_buf, width, height, filters, NULL, only_linear ? TRUE : FALSE);
  f = fopen(argv[5], "wb");
  if(!f || fwrite(out_buf, sizeof(float), npix*4, f) != npix*4) return 1;
  fclose(f);
  fprintf(stderr,"OK: VNG %dx%d -> %s\n", width, height, argv[5]);
  free(in_buf); free(out_buf);
  return 0;
}
