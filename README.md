# RT_project

A primitive ray-tracer written in CUDA.

## Current status

- Forked from [Ray Tracing: The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html) 
with minimal modifications to provide a baseline.
- Working CUDA implementation with intra-block ray reordering.

Reference render: 1600x1600 px, 1024 samples per pixel, depth of ray is 8.
![reference.png](reference.png)

## How-to

### Debug
```shell
compute-sanitizer.bat .\RT_project.exe
```

### Build
Only Windows is supported for now. 
Besides compiler toolchain, this project uses `CMake` and `vcpkg`.
Make sure to have `CMake` available on your system.

Initialize `vcpkg`:
```shell
git clone --recursive uri://link.to.this/repository.git
cd vcpkg
.\bootstrap-vcpkg.bat -disableMetrics
```

Initialize `CMake` configs and build. 
If you use MSVC, you need to run those commands inside the `Developer Command Prompt`:
```shell
cmake --preset=release
cmake --build ./cmake-build-release --target RT_project -j 4
```

The built program is found at `./cmake-build-release/RT_project.exe`

For optimal performance, you may need to tune the resource usage by changing `CUDA_MAXREG` in `CMakeLists.txt`.

The core code for ray tracing can be found at `camera::render_pixel_block<>()`.

Most performance-related configs can be found at `helpers.cuh`.

### Run

Run for real-time display output:
```shell
RT_project.exe
```