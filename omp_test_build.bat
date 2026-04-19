@echo off
set CL_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe
set MSVC_INC=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include
set MSVC_LIB=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\lib\x64
set SDK_UCRT_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt
set SDK_UM_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um
set SDK_UCRT_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64
set SDK_UM_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64
cd /d %~dp0
"%CL_PATH%" /nologo /O2 /openmp /TC /W3 /I"%MSVC_INC%" /I"%SDK_UCRT_INC%" /I"%SDK_UM_INC%" omp_test.c /Fe:omp_test.exe /link /LIBPATH:"%MSVC_LIB%" /LIBPATH:"%SDK_UCRT_LIB%" /LIBPATH:"%SDK_UM_LIB%"
