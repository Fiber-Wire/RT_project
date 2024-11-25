#ifndef RAY_H
#define RAY_H

#include "vec.cuh"


class ray {
  public:
    __host__ __device__ ray(): orig(), dir() {
    }

    __host__ __device__ ray(const point3& origin, const vec3& direction)
      : orig(origin), dir(direction) {}

    __host__ __device__ ray(const point3& origin, const NormVec3& direction)
      : orig(origin), dir(direction) {}

    __host__ __device__ const point3& origin() const  { return orig; }
    __host__ __device__ const NormVec3& direction() const { return dir; }

    __host__ __device__ point3 at(const float t) const {
        return orig + t*vec3(dir);
    }

  private:
    point3 orig;
    NormVec3 dir;
};


#endif
