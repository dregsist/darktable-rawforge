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

set COMMON=/nologo /O2 /openmp /TP /EHsc /W3 /I"%MSVC_INC%" /I"%SDK_UCRT_INC%" /I"%SDK_UM_INC%" /I"%SDK_SH_INC%" /I"%WORK%"
set LINKOPTS=/link /LIBPATH:"%MSVC_LIB%" /LIBPATH:"%SDK_UCRT_LIB%" /LIBPATH:"%SDK_UM_LIB%"

for %%n in (test_ari test_menon test_rcd test_ppg test_vng test_lmmse) do (
  echo Building %%n.exe ...
  "%CL_PATH%" %COMMON% %%n.c /Fe:%%n.exe %LINKOPTS%
  if errorlevel 1 ( echo BUILD FAILED: %%n.exe & exit /b 1 )
)

echo Building test_amaze.exe ...
"%CL_PATH%" /nologo /O2 /openmp /TP /W3 /EHsc /I"%MSVC_INC%" /I"%SDK_UCRT_INC%" /I"%SDK_UM_INC%" /I"%SDK_SH_INC%" /I"%WORK%" /I"%WORK%\stub_headers" test_amaze.cc /Fe:test_amaze.exe %LINKOPTS%
if errorlevel 1 ( echo BUILD FAILED: test_amaze.exe & exit /b 1 )
echo ALL BUILDS OK
