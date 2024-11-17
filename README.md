# RT_project

A primitive ray-tracer written in CUDA.

## Current status

- Forked from [Ray Tracing: The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html) 
with minimal modifications to provide a baseline.
- Working CUDA implementation (except image texture) with recursive function call

![reference.png](reference.png)

## How-to

### Debug
```shell
compute-sanitizer.bat .\RT_project.exe
```

### Build
Use `CMake` and `vcpkg`.

### Run
Run with default arguments for real-time display output:
```shell
RT_project.exe
```

Run with arguments for real-time display output:
```shell
RT_project.exe --size <int> --depth <int> --samples <int> --frame <int>
```
