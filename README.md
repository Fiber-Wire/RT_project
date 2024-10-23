# RT_project

A primitive ray-tracer written in CUDA.

## Current status

- Forked from [Ray Tracing: The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html) 
with minimal modifications to provide a baseline.
- Working CUDA implementation with loop-based function calls

Reference render: 1600x1600 px, 1024 samples per pixel, depth of ray is 8.
![reference.png](reference.png)

## TO-DO
- ~~Trim register usage(ideally all <= 64): metal::scatter(): 62, dielectric::scatter(): 69, lambertian::scatter(): 54, 
sphere::hit(): 70, quad::hit(): 67 (numbers are for SM_89 only)~~ Mostly done by `--use_fast_math`
- Improve cache hit-rate

### Frame time

Controlled test scene (same as the reference render above), 400x400 px, 32 samples per pixel, depth of ray is 4.
Camera spins at 0.1rad/frame for a total of 62 frames.

- AMD Ryzen 9 7940HS 8c16t: \~456.5 ms.
- NVIDIA GeForce RTX 4060 Laptop GPU: \~49.9 ms.

## How-to

### Debug
```shell
compute-sanitizer.bat .\RT_project.exe
```

### Build
Use `CMake` and `vcpkg`.

### Run
Run with any arguments for `.ppm` output:
```shell
RT_project.exe --device default > render.ppm
```

Run with default arguments for real-time display output:
```shell
RT_project.exe
```
Run with arguments for real-time display output:
```shell
RT_project.exe --size <int> --depth <int> --samples <int> --device <string> --frame <int>
```