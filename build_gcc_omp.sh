#!/usr/bin/env bash
# Build all demosaic test harnesses with MinGW GCC + real OpenMP.
# Darktable is developed on Linux/GCC so this matches production.
set -e

export PATH=/c/tools/msys64/mingw64/bin:$PATH
GCC=gcc
GXX=g++
CD=$(cd "$(dirname "$0")" && pwd)
cd "$CD"

CFLAGS="-O3 -march=native -fopenmp -std=gnu11 -Wall -Wno-unused-variable -I."
CXXFLAGS="-O3 -march=native -fopenmp -std=gnu++17 -Wall -Wno-unused-variable -I. -Istub_headers"
LDFLAGS="-fopenmp -lm -static -static-libgcc -static-libstdc++"

for name in test_ari test_menon test_rcd test_ppg test_vng test_lmmse; do
  echo "Building $name.exe ..."
  "$GCC" $CFLAGS $name.c -o $name.exe $LDFLAGS
  echo "BUILD OK: $name.exe"
done

echo "Building test_amaze.exe ..."
"$GXX" $CXXFLAGS test_amaze.cc -o test_amaze.exe $LDFLAGS
echo "BUILD OK: test_amaze.exe"

echo "ALL BUILDS OK"
