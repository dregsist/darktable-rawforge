@echo off
set CL_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe
set MSVC_INC=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include
set MSVC_LIB=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\lib\x64
set SDK_UCRT_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt
set SDK_UM_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um
set SDK_SH_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared
set SDK_UCRT_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64
set SDK_UM_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64
set WORK=C:\Users\dregsist\darktable-rawforge\demosaic_test
cd /d %WORK%

echo Building test_menon.exe ...
"%CL_PATH%" /nologo /O2 /TC /W3 /I"%MSVC_INC%" /I"%SDK_UCRT_INC%" /I"%SDK_UM_INC%" /I"%SDK_SH_INC%" /I"%WORK%" test_menon.c /Fe:test_menon.exe /link /LIBPATH:"%MSVC_LIB%" /LIBPATH:"%SDK_UCRT_LIB%" /LIBPATH:"%SDK_UM_LIB%"

if errorlevel 1 (
  echo BUILD FAILED: test_menon.exe
  exit /b 1
)
echo BUILD OK: test_menon.exe

echo Building test_rcd.exe ...
"%CL_PATH%" /nologo /O2 /TC /W3 /I"%MSVC_INC%" /I"%SDK_UCRT_INC%" /I"%SDK_UM_INC%" /I"%SDK_SH_INC%" /I"%WORK%" test_rcd.c /Fe:test_rcd.exe /link /LIBPATH:"%MSVC_LIB%" /LIBPATH:"%SDK_UCRT_LIB%" /LIBPATH:"%SDK_UM_LIB%"

if errorlevel 1 (
  echo BUILD FAILED: test_rcd.exe
  exit /b 1
)
echo BUILD OK: test_rcd.exe

echo Building test_ari.exe ...
"%CL_PATH%" /nologo /O2 /TC /W3 /I"%MSVC_INC%" /I"%SDK_UCRT_INC%" /I"%SDK_UM_INC%" /I"%SDK_SH_INC%" /I"%WORK%" test_ari.c /Fe:test_ari.exe /link /LIBPATH:"%MSVC_LIB%" /LIBPATH:"%SDK_UCRT_LIB%" /LIBPATH:"%SDK_UM_LIB%"

if errorlevel 1 (
  echo BUILD FAILED: test_ari.exe
  exit /b 1
)
echo BUILD OK: test_ari.exe

echo Building test_ppg.exe ...
"%CL_PATH%" /nologo /O2 /TC /W3 /I"%MSVC_INC%" /I"%SDK_UCRT_INC%" /I"%SDK_UM_INC%" /I"%SDK_SH_INC%" /I"%WORK%" test_ppg.c /Fe:test_ppg.exe /link /LIBPATH:"%MSVC_LIB%" /LIBPATH:"%SDK_UCRT_LIB%" /LIBPATH:"%SDK_UM_LIB%"
if errorlevel 1 ( echo BUILD FAILED: test_ppg.exe & exit /b 1 )
echo BUILD OK: test_ppg.exe

echo Building test_vng.exe ...
"%CL_PATH%" /nologo /O2 /TC /W3 /I"%MSVC_INC%" /I"%SDK_UCRT_INC%" /I"%SDK_UM_INC%" /I"%SDK_SH_INC%" /I"%WORK%" test_vng.c /Fe:test_vng.exe /link /LIBPATH:"%MSVC_LIB%" /LIBPATH:"%SDK_UCRT_LIB%" /LIBPATH:"%SDK_UM_LIB%"
if errorlevel 1 ( echo BUILD FAILED: test_vng.exe & exit /b 1 )
echo BUILD OK: test_vng.exe

echo Building test_lmmse.exe ...
"%CL_PATH%" /nologo /O2 /TC /W3 /I"%MSVC_INC%" /I"%SDK_UCRT_INC%" /I"%SDK_UM_INC%" /I"%SDK_SH_INC%" /I"%WORK%" test_lmmse.c /Fe:test_lmmse.exe /link /LIBPATH:"%MSVC_LIB%" /LIBPATH:"%SDK_UCRT_LIB%" /LIBPATH:"%SDK_UM_LIB%"
if errorlevel 1 ( echo BUILD FAILED: test_lmmse.exe & exit /b 1 )
echo BUILD OK: test_lmmse.exe

echo Building test_amaze.exe ...
"%CL_PATH%" /nologo /O2 /TP /W3 /EHsc /I"%MSVC_INC%" /I"%SDK_UCRT_INC%" /I"%SDK_UM_INC%" /I"%SDK_SH_INC%" /I"%WORK%" /I"%WORK%\stub_headers" test_amaze.cc /Fe:test_amaze.exe /link /LIBPATH:"%MSVC_LIB%" /LIBPATH:"%SDK_UCRT_LIB%" /LIBPATH:"%SDK_UM_LIB%"
if errorlevel 1 ( echo BUILD FAILED: test_amaze.exe & exit /b 1 )
echo BUILD OK: test_amaze.exe
