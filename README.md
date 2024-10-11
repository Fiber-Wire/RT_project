# RT_project

A primitive ray-tracer written in CUDA.

## Current status

Forked from [Ray Tracing: The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html) 
with minimal modifications to provide a baseline.

## How-to

### Build
Use `CMake` and `vcpkg`.

### Run
Run with any arguments for `.ppm` output:
```shell
RT_project.exe XXX > render.ppm
```

Run directly for real-time display output:
```shell
RT_project.exe
```