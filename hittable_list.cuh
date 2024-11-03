#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.cuh"
#include <span>
class hittable_list final : public hittable {
public:
    hittable** objects{};
    int count{};
    int capacity{};
    __host__ __device__ hittable_list();
    __host__ __device__ explicit hittable_list(const int capacity);

    __host__ __device__ explicit hittable_list(hittable* object);
    __host__ __device__ hittable_list(const hittable_list& other);
    __host__ __device__ hittable_list& operator=(const hittable_list& other);

    __host__ __device__ void clear();

    __host__ __device__ std::span<hittable*> get_objects();

    __host__ __device__ void add(hittable* object);

    __host__ __device__ void add(const hittable_list* list);

    __host__ __device__ bool hit(const ray& r, const interval ray_t, hit_record& rec) const override;

    __host__ __device__ aabb bounding_box() const override;

    __host__ __device__ ~hittable_list() override;

private:
    aabb bbox;
};



#endif
