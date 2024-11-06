//
// Created by JCW on 2024/11/3.
//

#ifndef GET_HIT_CUH
#define GET_HIT_CUH
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "bvh.cuh"
#include "geometry.cuh"
__host__ __device__ inline bool get_hit(const ray& r, const interval ray_t, hit_record& rec, const hittable* hit) {
    bool hit_anything = false;
    switch (hit->type) {
        case hit_type::eBVH: hit_anything = static_cast<const bvh_tree*>(hit)->hit(r, ray_t, rec); break;
        case hit_type::eTranslate: hit_anything = static_cast<const translate*>(hit)->hit(r, ray_t, rec); break;
        case hit_type::eRotate_y: hit_anything = static_cast<const rotate_y*>(hit)->hit(r, ray_t, rec); break;
        case hit_type::eList: hit_anything = static_cast<const hittable_list*>(hit)->hit(r, ray_t, rec); break;
        case hit_type::eSphere: hit_anything = static_cast<const sphere*>(hit)->hit(r, ray_t, rec); break;
        case hit_type::eQuad: hit_anything = static_cast<const quad*>(hit)->hit(r, ray_t, rec); break;
        default: break;
    }
    return hit_anything;
}
#endif //GET_HIT_CUH
