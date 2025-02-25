@echo off
echo Running NVIDIA nbody benchmark...

REM Pull and run the CUDA samples container with nbody benchmark
docker run --gpus all --rm nvidia/samples:nbody nbody -benchmark -numbodies=256000

echo.
echo Benchmark complete.
pause