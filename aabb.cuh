#ifndef AABB_H
#define AABB_H
#include "interval.cuh"
#include "vec.cuh"
#include "ray.cuh"

class aabb {
  public:
    interval x, y, z;

    __host__ __device__ aabb() {} // The default AABB is empty, since intervals are empty by default.

    __host__ __device__ aabb(const interval& x, const interval& y, const interval& z)
      : x(x), y(y), z(z)
    {
        pad_to_minimums();
    }

    __host__ __device__ aabb(const point3& a, const point3& b) {
        // Treat the two points a and b as extrema for the bounding box, so we don't require a
        // particular minimum/maximum coordinate order.

        x = (a[0] <= b[0]) ? interval(a[0], b[0]) : interval(b[0], a[0]);
        y = (a[1] <= b[1]) ? interval(a[1], b[1]) : interval(b[1], a[1]);
        z = (a[2] <= b[2]) ? interval(a[2], b[2]) : interval(b[2], a[2]);

        pad_to_minimums();
    }

    __host__ __device__ aabb(const aabb& box0, const aabb& box1) {
        x = interval(box0.x, box1.x);
        y = interval(box0.y, box1.y);
        z = interval(box0.z, box1.z);
    }

    __host__ __device__ const interval& axis_interval(const int n) const {
        if (n == 1) return y;
        if (n == 2) return z;
        return x;
    }

    __host__ __device__ bool hit(const ray& r, interval ray_t) const {
        const point3& ray_orig = r.origin();
        const auto ray_dir  = vec3(r.direction());
        #pragma unroll
        for (int axis = 0; axis < 3; axis++) {
            const interval& ax = axis_interval(axis);
            const float adinv = 1.0f / ray_dir[axis];

            const auto t0 = (ax.min - ray_orig[axis]) * adinv;
            const auto t1 = (ax.max - ray_orig[axis]) * adinv;

            ray_t.min = max(ray_t.min, min(t0, t1));
            ray_t.max = min(ray_t.max, max(t0, t1));
        }
        return ray_t.max > ray_t.min;
    }

    __host__ __device__ int longest_axis() const {
        // Returns the index of the longest axis of the bounding box.

        if (x.size() > y.size())
            return x.size() > z.size() ? 0 : 2;
        else
            return y.size() > z.size() ? 1 : 2;
    }

    __host__ __device__ point3 center() const{
        return {x.centre(), y.centre(), z.centre()};
    }

    __host__ __device__ static aabb empty() {
        return {interval::empty(),    interval::empty(),    interval::empty()};
    }
    __host__ __device__ static aabb universe() {
        return {interval::universe(),    interval::universe(),    interval::universe()};
    }

  private:

    __host__ __device__ void pad_to_minimums() {
        // Adjust the AABB so that no side is narrower than some delta, padding if necessary.

        constexpr float delta = 0.0001f;
        if (x.size() < delta) x = x.expand(delta);
        if (y.size() < delta) y = y.expand(delta);
        if (z.size() < delta) z = z.expand(delta);
    }
};


__host__ __device__ inline aabb operator+(const aabb& bbox, const vec3& offset) {
    return aabb(bbox.x + offset.x, bbox.y + offset.y, bbox.z + offset.z);
}

__host__ __device__ inline aabb operator+(const vec3& offset, const aabb& bbox) {
    return bbox + offset;
}
inline __host__ __device__ aabb merge(const aabb& lhs, const aabb& rhs)
{
    aabb merged;
    merged.x.max = ::fmaxf(lhs.x.max, rhs.x.max);
    merged.y.max = ::fmaxf(lhs.y.max, rhs.y.max);
    merged.z.max = ::fmaxf(lhs.z.max, rhs.z.max);
    merged.x.min = ::fminf(lhs.x.min, rhs.x.min);
    merged.y.min = ::fminf(lhs.y.min, rhs.y.min);
    merged.z.min = ::fminf(lhs.z.min, rhs.z.min);
    return merged;
}


#endif
