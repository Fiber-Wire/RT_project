#ifndef BVH_H
#define BVH_H

#include "aabb.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"

#include <algorithm>

class bvh_tree final : public hittable {
  public:
    __host__ __device__ bvh_tree() {
        type = hit_type::eBVH;
        bbox_list = new aabb[1];
        root.bbox_id = 0;
        bbox_list[root.bbox_id] = aabb::empty();
    }
    __host__ __device__ explicit bvh_tree(hittable_list list) : bvh_tree(list.get_objects(), 0, list.count) {
        // There's a C++ subtlety here. This constructor (without span indices) creates an
        // implicit copy of the hittable list, which we will modify. The lifetime of the copied
        // list only extends until this constructor exits. That's OK, because we only need to
        // persist the resulting bounding volume hierarchy.
        printf("build bvh with %i objs, length %i\n", list.count, node_length);
    }



    /// [start, end)
    __host__ __device__ bvh_tree(const std::span<hittable*> objects, const int start, const int end) {
        type = hit_type::eBVH;
        if(end - start <= 0) {
            bbox_list = new aabb[1];
            root.bbox_id = 0;
            bbox_list[root.bbox_id] = aabb::empty();
            return;
        }

        count = end - start;
        node_length = glm::pow(2,glm::log2(end-start)+1)+1;
        leaf_list = new bvh_node[node_length];
        primitive_list = new hittable *[count];
        for(int i = 0; i < count; i++) {
            primitive_list[i] = objects[start+i];
        }
        bbox_list = new aabb[node_length];
        bvh_tree_construct_info info_stack[10];
        int stack_index = 0;
        int node_write_ptr = 1;
        info_stack[stack_index] = {0, 0, static_cast<unsigned int>(count)};
        while(stack_index>-1) {
            auto [node, start_t, end_t] = info_stack[stack_index];
            stack_index--;
            // Build the bounding box of the span of source objects.
            auto& current_node = leaf_list[node];
            current_node.bbox_id = node;
            auto& current_bbox = bbox_list[current_node.bbox_id];
            current_bbox = aabb::empty();
            for (int object_index=start_t; object_index < end_t; object_index++)
                current_bbox = aabb(current_bbox, primitive_list[object_index]->bounding_box());

            const int axis = current_bbox.longest_axis();
            const int object_span = end_t - start_t;

            // TODO: add bvh_node detection, or split scene and bvh altogether
            if (object_span == 1) {
                current_node.left = current_node.right = start_t;
            } else if (object_span == 2) {
                current_node.left = start_t;
                current_node.right = start_t+1;
            } else {
                sort_primitive(axis, start_t, end_t);
                const auto mid = start_t + object_span/2;
                current_node.child_bvh = true;
                current_node.left = node_write_ptr;
                node_write_ptr += 1;
                current_node.right = node_write_ptr;
                node_write_ptr += 1;
                info_stack[stack_index + 1] = {current_node.right, mid, end_t};//right
                stack_index += 1;
                info_stack[stack_index + 1] = {current_node.left, start_t, mid};//left
                stack_index += 1;
            }
        }
        root = leaf_list[0];
    }

    __host__ __device__ ~bvh_tree() override {
        delete[] bbox_list;
        bbox_list = nullptr;
        delete[] leaf_list;
        leaf_list = nullptr;
        delete[] primitive_list;
        primitive_list = nullptr;
    }

    __host__ __device__ bvh_tree(const bvh_tree& other) {
        *this = other;
    }

    __host__ __device__ bvh_tree& operator=(const bvh_tree& other) {
        if (this != &other) {
            type = other.type;
            root = other.root;
            count = other.count;
            node_length = other.node_length;
            delete [] primitive_list;
            if (count != 0) {
                primitive_list = new hittable*[count];
                for (auto i = 0; i < count; i++) {
                    primitive_list[i] = other.primitive_list[i];
                }
            } else {
                primitive_list = nullptr;
            }

            delete [] leaf_list;
            if (node_length != 0) {
                leaf_list = new bvh_node[node_length];
                for (auto i = 0; i < node_length; i++) {
                    leaf_list[i] = other.leaf_list[i];
                }
            } else {
                leaf_list = nullptr;
            }

            delete [] bbox_list;
            if (node_length != 0) {
                bbox_list = new aabb[node_length];
                for (auto i = 0; i < node_length; i++) {
                    bbox_list[i] = other.bbox_list[i];
                }
            } else {
                bbox_list = nullptr;
            }
        }
        return *this;
    }

    __host__ __device__ bool hit(const ray& r, const interval ray_t, hit_record& rec) const {
        int bvh_stack[10];
        int stack_index = 0;
        bvh_stack[stack_index] = 0;
        bool is_hit = false;
        auto max_t = ray_t.max;
        while (stack_index > -1) {
            const auto& current_node = leaf_list[bvh_stack[stack_index]];
            stack_index--;
            if (bbox_list[current_node.bbox_id].hit(r, interval(ray_t.min, max_t))) {
                if (current_node.child_bvh) {
                    stack_index+=1;
                    bvh_stack[stack_index] = current_node.right;
                    stack_index+=1;
                    bvh_stack[stack_index] = current_node.left;
                } else {
                    const bool hit_left = get_hit(r, interval(ray_t.min, max_t), rec,
                                                  primitive_list[current_node.left]);
                    max_t = hit_left ? rec.t : max_t;
                    is_hit = is_hit || hit_left;
                    if (current_node.right != current_node.left) {
                        const bool hit_right = get_hit(r, interval(ray_t.min, max_t), rec,
                                                       primitive_list[current_node.right]);
                        max_t = hit_right ? rec.t : max_t;
                        is_hit = is_hit || hit_right;
                    }
                }
            }
        }
        return is_hit;
    }

    __host__ __device__ aabb bounding_box() const override { return bbox_list[root.bbox_id]; }

  private:
    struct bvh_tree_construct_info {
        unsigned int node_id;
        unsigned int start;
        unsigned int end;
    };
    struct bvh_node {
        unsigned int left{};
        unsigned int right{};
        unsigned int bbox_id{};
        bool child_bvh{false};
    };
    bvh_node root{};
    int count{0};
    hittable** primitive_list{nullptr};
    int node_length{0};
    bvh_node* leaf_list{nullptr};
    aabb* bbox_list{nullptr};

    __host__ __device__ static bool box_compare(
        const hittable* a, const hittable* b, const int axis_index
    ) {
        const auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
        const auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
        return a_axis_interval.min < b_axis_interval.min;
    }

    __host__ __device__ void sort_primitive(const int axis, const int start, const int end) const {
#ifdef __CUDA_ARCH__
        // TODO: use a more efficient sorting method
        for (auto comp_index=start; comp_index < end-1; comp_index++) {
            for (auto object_index=comp_index+1; object_index < end; object_index++) {
                if (!box_compare(primitive_list[comp_index], primitive_list[object_index], axis)) {
                    const auto temp = primitive_list[comp_index];
                    primitive_list[comp_index] = primitive_list[object_index];
                    primitive_list[object_index] = temp;
                }
            }
        }
#else
        std::sort(primitive_list+start, primitive_list+end,
                  [axis, this](const hittable* lhs, const hittable* rhs)
                  {return box_compare(lhs, rhs, axis);});
#endif
    }
};


#endif BVH_H
