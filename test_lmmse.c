/*
 * Standalone LMMSE demosaic test harness.
 * Usage: test_lmmse <w> <h> <filters_hex> <input.bin> <output.bin> [mode]
 *   mode: 0=basic, 1=median (default), 2=3xmedian, 3=refine+medians, 4=2xrefine+medians
 */
#include "stubs.h"

#define DT_LMMSE_TILESIZE 136
typedef enum dt_iop_demosaic_lmmse_t {
  DT_LMMSE_REFINE_0 = 0,
  DT_LMMSE_REFINE_1 = 1,
  DT_LMMSE_REFINE_2 = 2,
  DT_LMMSE_REFINE_3 = 3,
  DT_LMMSE_REFINE_4 = 4,
} dt_iop_demosaic_lmmse_t;

#include "../darktable/src/iop/demosaicing/basics.c"
#include "../darktable/src/iop/demosaicing/ppg.c"
#include "../darktable/src/iop/demosaicing/lmmse.c"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  if(argc < 6 || argc > 7)
  {
    fprintf(stderr,"Usage: %s <w> <h> <filters_hex> <input.bin> <output.bin> [mode]\n", argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]), height = atoi(argv[2]);
  const uint32_t filters = (uint32_t)strtoul(argv[3], NULL, 16);
  const int mode = (argc == 7) ? atoi(argv[6]) : 1;
  const size_t npix = (size_t)width * height;
  float *in_buf = (float *)malloc(npix * sizeof(float));
  FILE *f = fopen(argv[4], "rb");
  if(!f || fread(in_buf, sizeof(float), npix, f) != npix) return 1;
  fclose(f);
  float *out_buf = (float *)calloc(npix * 4, sizeof(float));
  _init_lmmse_gamma();
  lmmse_demosaic(out_buf, in_buf, width, height, filters, (dt_iop_demosaic_lmmse_t)mode, 1.0f);
  _cleanup_lmmse_gamma();
  f = fopen(argv[5], "wb");
  if(!f || fwrite(out_buf, sizeof(float), npix*4, f) != npix*4) return 1;
  fclose(f);
  fprintf(stderr,"OK: LMMSE %dx%d mode=%d -> %s\n", width, height, mode, argv[5]);
  free(in_buf); free(out_buf);
  return 0;
}
