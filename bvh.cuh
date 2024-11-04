#ifndef BVH_H
#define BVH_H

#include "aabb.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"

#include <algorithm>

class bvh_node final : public hittable {
  public:
    __host__ __device__ bvh_node(): bbox(aabb::empty()) {
        type = hit_type::eBVH;
    }
    __host__ __device__ explicit bvh_node(hittable_list list) : bvh_node(list.get_objects(), 0, list.count) {
        // There's a C++ subtlety here. This constructor (without span indices) creates an
        // implicit copy of the hittable list, which we will modify. The lifetime of the copied
        // list only extends until this constructor exits. That's OK, because we only need to
        // persist the resulting bounding volume hierarchy.
        printf("build bvh with %i objs, depth %i\n", list.count, depth);
    }

    __host__ __device__ bvh_node(std::span<hittable*> objects, const size_t start, const size_t end) {
        type = hit_type::eBVH;
        if(end - start <= 0) {
            bbox = aabb::empty();
            return;
        }

        auto bvh_node_list = new bvh_node[glm::pow(2,glm::log2(end-start)+1)];
        bvh_node_construct_info info_stack[10];
        int stack_index = 0;
        info_stack[stack_index] = {this, start, end};
        while(stack_index>-1) {
            auto [node, start_t, end_t] = info_stack[stack_index];
            stack_index--;
            // Build the bounding box of the span of source objects.
            node->bbox = aabb::empty();
            for (size_t object_index=start_t; object_index < end_t; object_index++)
                node->bbox = aabb(node->bbox, objects[object_index]->bounding_box());

            int axis = node->bbox.longest_axis();
            const size_t object_span = end_t - start_t;

            // TODO: add bvh_node detection, or split scene and bvh altogether
            if (object_span == 1) {
                node->left = node->right = objects[start_t];
                node->depth = 0;
            } else if (object_span == 2) {
                node->left = objects[start_t];
                node->right = objects[start_t+1];
                node->depth = 0;
            } else {
                {
#ifdef __CUDA_ARCH__
                    // TODO: use a more efficient sorting method
                    for (size_t comp_index=start_t; comp_index < end_t-1; comp_index++) {
                        for (size_t object_index=comp_index+1; object_index < end_t; object_index++) {
                            if (!box_compare(objects[comp_index], objects[object_index], axis)) {
                                const auto temp = objects[comp_index];
                                objects[comp_index] = objects[object_index];
                                objects[object_index] = temp;
                            }
                        }
                    }
#else
                    std::sort(objects.begin()+start_t, objects.begin()+end_t,
                              [axis, this](const hittable* lhs, const hittable* rhs)
                                {return box_compare(lhs, rhs, axis);});
#endif
                }
                const auto mid = start_t + object_span/2;
                node->child_bvh = true;
                node->right = bvh_node_list + 1;
                node->left = bvh_node_list;
                info_stack[stack_index + 2] = {bvh_node_list, start_t, mid};//left
                info_stack[stack_index + 1] = {bvh_node_list + 1, mid, end_t};//right
                const int temp_depth = glm::log2(object_span);
                node->depth = glm::pow(2,temp_depth)==object_span? temp_depth-1:temp_depth;
                bvh_node_list += 2;
                stack_index += 2;
            }
        }
    }

    __host__ __device__ ~bvh_node() override {
        if (child_bvh) {
            delete left;
            delete right;
            child_bvh = false;
        }
    }

    __host__ __device__ bvh_node(const bvh_node& other) {
        *this = other;
    }

    __host__ __device__ bvh_node& operator=(const bvh_node& other) {
        if (this != &other) {
            type = other.type;
            bbox = other.bbox;
            left = other.left;
            right = other.right;
            depth = other.depth;
            if (other.child_bvh) {
                left = new bvh_node{};
                *static_cast<bvh_node *>(left) = *static_cast<bvh_node *>(other.left);
                right = new bvh_node{};
                *static_cast<bvh_node *>(right) = *static_cast<bvh_node *>(other.right);
                child_bvh = true;
            }
        }
        return *this;
    }

    __host__ __device__ bool hit(const ray& r, const interval ray_t, hit_record& rec) const {
        bvh_node const * bvh_stack[10];
        int stack_index = 0;
        bvh_stack[stack_index] = this;
        bool is_hit = false;
        auto max_t = ray_t.max;
        while (stack_index > -1) {
            const auto current_node = bvh_stack[stack_index];
            stack_index--;
            const bool current_hit = current_node->bbox.hit(r, interval(ray_t.min, max_t));
            if (current_hit) {
                if (current_node->child_bvh) {
                    stack_index+=1;
                    bvh_stack[stack_index] = static_cast<bvh_node const*>(current_node->right);
                    stack_index+=1;
                    bvh_stack[stack_index] = static_cast<bvh_node const*>(current_node->left);
                } else {
                    const bool hit_left = get_hit(r, interval(ray_t.min, max_t), rec, current_node->left);
                    max_t = hit_left ? rec.t : max_t;
                    is_hit = is_hit || hit_left;
                    if (current_node->right != current_node->left) {
                        const bool hit_right = get_hit(r, interval(ray_t.min, max_t), rec, current_node->right);
                        max_t = hit_right ? rec.t : max_t;
                        is_hit = is_hit || hit_right;
                    }
                }
            }
        }
        return is_hit;
    }

    __host__ __device__ aabb bounding_box() const override { return bbox; }

  private:
    struct bvh_node_construct_info {
        bvh_node* node;
        size_t start;
        size_t end;
    };
    hittable* left{};
    hittable* right{};
    aabb bbox;
    int depth{};
    bool child_bvh{false};

    __host__ __device__ static bool box_compare(
        const hittable* a, const hittable* b, const int axis_index
    ) {
        const auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
        const auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
        return a_axis_interval.min < b_axis_interval.min;
    }
};


#endif BVH_H
