#ifndef BVH_H
#define BVH_H

#include "aabb.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"

#include <algorithm>


class bvh_node : public hittable {
  public:
    __host__ __device__ bvh_node(): bbox(aabb::empty()) {}
    __host__ __device__ explicit bvh_node(hittable_list list) : bvh_node(list.get_objects(), 0, list.count) {
        // There's a C++ subtlety here. This constructor (without span indices) creates an
        // implicit copy of the hittable list, which we will modify. The lifetime of the copied
        // list only extends until this constructor exits. That's OK, because we only need to
        // persist the resulting bounding volume hierarchy.
    }

    __host__ __device__ bvh_node(std::span<hittable*> objects, size_t start, size_t end) {
        // Build the bounding box of the span of source objects.
        bbox = aabb::empty();
        for (size_t object_index=start; object_index < end; object_index++)
            bbox = aabb(bbox, objects[object_index]->bounding_box());

        int axis = bbox.longest_axis();

        size_t object_span = end - start;
        // TODO: add bvh_node detection, or split scene and bvh altogether
        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            left = objects[start];
            right = objects[start+1];
        } else {
            {
#ifdef __CUDA_ARCH__
                // TODO: use a more efficient sorting method
                for (size_t comp_index=start; comp_index < end-1; comp_index++) {
                    for (size_t object_index=comp_index+1; object_index < end; object_index++) {
                        if (!box_compare(objects[comp_index], objects[object_index], axis)) {
                            const auto temp = objects[comp_index];
                            objects[comp_index] = objects[object_index];
                            objects[object_index] = temp;
                        }
                    }
                }
#else
                std::sort(objects.begin()+start, objects.begin()+end,
                          [axis, this](const hittable* lhs, const hittable* rhs)
                            {return box_compare(lhs, rhs, axis);});
#endif __CUDA_ARCH__
            }

            auto mid = start + object_span/2;
            left_bvh = true;
            right_bvh = true;
            left = new bvh_node{objects, start, mid};
            right = new bvh_node{objects, mid, end};
        }
    }

    __host__ __device__ ~bvh_node() override {
        if (left_bvh) {
            delete left;
            left_bvh = false;
        }
        if (right_bvh) {
            delete right;
            right_bvh = false;
        }
    }

    __host__ __device__ bvh_node(const bvh_node& other) {
        *this = other;
    }

    __host__ __device__ bvh_node& operator=(const bvh_node& other) {
        if (this != &other) {
            bbox = other.bbox;
            left = other.left;
            right = other.right;
            if (other.left_bvh) {
                left = new bvh_node{};
                *(bvh_node*)(left) = *(bvh_node*)(other.left);
                left_bvh = true;
            }
            if (other.right_bvh) {
                right = new bvh_node{};
                *(bvh_node*)(right) = *(bvh_node*)(other.right);
                right_bvh = true;
            }
        }
        return *this;
    }

    __host__ __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        if (!bbox.hit(r, ray_t))
            return false;

        bool hit_left = left->hit(r, ray_t, rec);
        bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

        return hit_left || hit_right;
    }

    __host__ __device__ aabb bounding_box() const override { return bbox; }

  private:
    hittable* left{};
    bool left_bvh{false};
    hittable* right{};
    bool right_bvh{false};
    aabb bbox;

    __host__ __device__ static bool box_compare(
        const hittable* a, const hittable* b, int axis_index
    ) {
        auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
        auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
        return a_axis_interval.min < b_axis_interval.min;
    }

    __host__ __device__ static bool box_x_compare (const hittable* a, const hittable* b) {
        return box_compare(a, b, 0);
    }

    __host__ __device__ static bool box_y_compare (const hittable* a, const hittable* b) {
        return box_compare(a, b, 1);
    }

    __host__ __device__ static bool box_z_compare (const hittable* a, const hittable* b) {
        return box_compare(a, b, 2);
    }
};


#endif BVH_H
