/*
 * Standalone PPG demosaic test harness.
 * Usage: test_ppg <w> <h> <filters_hex> <input.bin> <output.bin>
 */
#include "stubs.h"
#include "../darktable/src/iop/demosaicing/basics.c"
#include "../darktable/src/iop/demosaicing/ppg.c"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  if(argc != 6)
  {
    fprintf(stderr, "Usage: %s <w> <h> <filters_hex> <input.bin> <output.bin>\n", argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]), height = atoi(argv[2]);
  const uint32_t filters = (uint32_t)strtoul(argv[3], NULL, 16);
  const size_t npix = (size_t)width * height;
  float *in_buf = (float *)malloc(npix * sizeof(float));
  FILE *f = fopen(argv[4], "rb");
  if(!f || fread(in_buf, sizeof(float), npix, f) != npix) { fprintf(stderr,"read err\n"); return 1; }
  fclose(f);
  float *out_buf = (float *)calloc(npix * 4, sizeof(float));
  /* margin = 100000 => process full interior (matches demosaic.c single-tile call) */
  demosaic_ppg(out_buf, in_buf, width, height, filters, 0.0f, 100000);
  f = fopen(argv[5], "wb");
  if(!f || fwrite(out_buf, sizeof(float), npix*4, f) != npix*4) return 1;
  fclose(f);
  fprintf(stderr,"OK: PPG %dx%d -> %s\n", width, height, argv[5]);
  free(in_buf); free(out_buf);
  return 0;
}
