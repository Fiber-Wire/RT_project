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
        bbox_list[0] = aabb::empty();
    }

    __host__ __device__ explicit bvh_tree(hittable_list list) : bvh_tree(list.get_objects(), 0, list.count) {
        // There's a C++ subtlety here. This constructor (without span indices) creates an
        // implicit copy of the hittable list, which we will modify. The lifetime of the copied
        // list only extends until this constructor exits. That's OK, because we only need to
        // persist the resulting bounding volume hierarchy.
        printf("build bvh with %i objs, length %i\n", list.count, node_length);
    }

    /// [start, end)
    __host__ __device__ bvh_tree(const std::span<hittable *> objects, const int start, const int end) {
        type = hit_type::eBVH;
        if (end - start <= 0) {
            bbox_list = new aabb[1];
            bbox_list[0] = aabb::empty();
            return;
        }

        count = end - start;
        node_length = count * 2 - 1;
        leaf_list = new bvh_node[node_length];
        primitive_list = new hittable *[count];
        for (int i = 0; i < count; i++) {
            primitive_list[i] = objects[start + i];
        }
        bbox_list = new aabb[node_length];
        bvh_rebuild<<<1,1>>>(*this);
    }

    __host__ __device__ ~bvh_tree() override {
        delete[] bbox_list;
        bbox_list = nullptr;
        delete[] leaf_list;
        leaf_list = nullptr;
        delete[] primitive_list;
        primitive_list = nullptr;
    }

    __host__ __device__ bvh_tree(const bvh_tree &other) {
        *this = other;
    }

    __host__ __device__ bvh_tree &operator=(const bvh_tree &other) {
        if (this != &other) {
            type = other.type;
            count = other.count;
            node_length = other.node_length;
            delete [] primitive_list;
            if (count != 0) {
                primitive_list = new hittable *[count];
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

    friend __global__ void bvh_rebuild(bvh_tree &obj);

    __host__ __device__ bool hit(const ray &r, const interval ray_t, hit_record &rec) const {
        int bvh_stack[10];
        int stack_index = 0;
        bvh_stack[stack_index] = 0;
        bool is_hit = false;
        auto max_t = ray_t.max;
        while (stack_index > -1) {
            const auto &current_node = leaf_list[bvh_stack[stack_index]];
            stack_index--;
            if (bbox_list[bvh_stack[stack_index + 1]].hit(r, interval(ray_t.min, max_t))) {
                //if current_node is leaf node, left points to the obj_id, right is 0xFFFFFFFF
                if (current_node.right != 0xFFFFFFFF) {
                    stack_index += 1;
                    bvh_stack[stack_index] = current_node.right;
                    stack_index += 1;
                    bvh_stack[stack_index] = current_node.left;
                } else {
                    const bool hit_left = get_hit(r, interval(ray_t.min, max_t), rec,
                                                  primitive_list[current_node.left]);
                    max_t = hit_left ? rec.t : max_t;
                    is_hit = is_hit || hit_left;
                }
            }
        }
        return is_hit;
    }

    __host__ __device__ aabb bounding_box() const override { return bbox_list[0]; }

    struct bvh_node {
        unsigned int left{};
        unsigned int right{};
        unsigned int parent_id{};
    };

private:

    int count{0};
    hittable **primitive_list{nullptr};
    int node_length{0};
    bvh_node *leaf_list{nullptr};
    aabb *bbox_list{nullptr};
    __device__ static uint2 determine_range(const unsigned long long *object_mortons,
                                            const unsigned int num_objects, unsigned int node_idx) {
        if (node_idx == 0) {
            return make_uint2(0, num_objects - 1);
        }

        // determine direction of the range
        const unsigned long long self_code = object_mortons[node_idx];
        const int L_delta = common_upper_bits(self_code, object_mortons[node_idx - 1]);
        const int R_delta = common_upper_bits(self_code, object_mortons[node_idx + 1]);
        const int d = (R_delta > L_delta) ? 1 : -1;
        // Compute upper bound for the length of the range
        // find the range of nodes that have the maximum same bits at the beginning using binary search[first <<1 then >>1]
        const int delta_min = min(L_delta, R_delta);
        int l_max = 2;
        int delta = -1;
        int i_tmp = node_idx + d * l_max;
        if (0 <= i_tmp && i_tmp < num_objects) {
            delta = common_upper_bits(self_code, object_mortons[i_tmp]);
        }
        while (delta > delta_min) {
            l_max <<= 1;
            i_tmp = node_idx + d * l_max;
            delta = -1;
            if (0 <= i_tmp && i_tmp < num_objects) {
                delta = common_upper_bits(self_code, object_mortons[i_tmp]);
            }
        }

        // Find the other end by binary search
        int l = 0;
        int t = l_max >> 1;
        while (t > 0) {
            i_tmp = node_idx + (l + t) * d;
            delta = -1;
            if (0 <= i_tmp && i_tmp < num_objects) {
                delta = common_upper_bits(self_code, object_mortons[i_tmp]);
            }
            if (delta > delta_min) {
                l += t;
            }
            t >>= 1;
        }
        unsigned int jdx = node_idx + l * d;
        if (d < 0) {
            thrust::swap(node_idx, jdx); // make it sure that idx < jdx
        }

        return make_uint2(node_idx, jdx);
    }

    __device__ static unsigned int find_split(const unsigned long long *node_code,
                                              const unsigned int first, const unsigned int last) {
        const unsigned long long first_code = node_code[first];
        const unsigned long long last_code = node_code[last];
        if (first_code == last_code) {
            return (first + last) >> 1;
        }
        //first and last have the least common bits
        const int delta_node = common_upper_bits(first_code, last_code);

        // binary search
        int split = first;
        int stride = last - first;
        do {
            stride = (stride + 1) >> 1;
            const int middle = split + stride;
            if (middle < last) {
                const int delta = common_upper_bits(first_code, node_code[middle]);
                if (delta > delta_node) {
                    split = middle;
                }
            }
        } while (stride > 1);

        return split;
    }

    __device__ static void construct_internal_nodes(bvh_node *internal_nodes, const unsigned long long *object_mortons,
                                             const unsigned int num_objects) {
        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator<unsigned int>(0),
                         // number of internal nodes is one less than num_objects
                         thrust::make_counting_iterator<unsigned int>(num_objects - 1),
                         [=] __device__ (const unsigned int idx) {
                             const uint2 ij = determine_range(object_mortons, num_objects, idx);
                             // binary search from the range
                             const int gamma = find_split(object_mortons, ij.x, ij.y);

                             internal_nodes[idx].left = gamma;
                             internal_nodes[idx].right = gamma + 1;
                             //? range [low high]
                             if (min(ij.x, ij.y) == gamma) {
                                 // last num_objects record the objects
                                 internal_nodes[idx].left += num_objects - 1;
                             }
                             if (max(ij.x, ij.y) == gamma + 1) {
                                 internal_nodes[idx].right += num_objects - 1;
                             }
                             internal_nodes[internal_nodes[idx].left].parent_id = idx;
                             internal_nodes[internal_nodes[idx].right].parent_id = idx;
                         });
    }
};


__global__ inline void bvh_rebuild(bvh_tree &obj) {
    const unsigned int num_internal_nodes = obj.count - 1;
    //init bbox_list
    const auto default_aabb = aabb::empty();
    thrust::fill(
        thrust::device,
        obj.bbox_list,
        obj.bbox_list + num_internal_nodes,
        default_aabb);

    thrust::transform(thrust::device, obj.primitive_list, obj.primitive_list + obj.count,
                      obj.bbox_list + num_internal_nodes, []__device__(const hittable *object) {
                          return object->bounding_box();
                      });

    //get bbox for the whole tree
    const auto aabb_whole = thrust::reduce(thrust::device,
                                           obj.bbox_list + num_internal_nodes, obj.bbox_list + obj.node_length,
                                           default_aabb,
                                           []__device__ (const aabb &lhs, const aabb &rhs) {
                                               return merge(lhs, rhs);
                                           });

    //morton3d construct
    const auto morton = new unsigned int[obj.count];
    thrust::transform(thrust::device, obj.primitive_list, obj.primitive_list + obj.count, morton,
                      [=] __device__ (const hittable *object) {
                          return mortonEncode3D(object->bounding_box().center(), aabb_whole.x, aabb_whole.y,
                                                aabb_whole.z);
                      });

    // TODO: remove indices if possible
    auto indices = new unsigned int [obj.count];
    thrust::copy(thrust::device, thrust::make_counting_iterator<unsigned int>(0),
                 thrust::make_counting_iterator<unsigned int>(obj.count),
                 indices);

    // keep indices ascending order
    thrust::stable_sort_by_key(thrust::device, morton, morton + obj.count,
                               thrust::make_zip_iterator(
                                   thrust::make_tuple(obj.bbox_list + num_internal_nodes, indices)));

    //extend mortonCode from 32 to 64  64 = morton obj_id
    const auto morton64 = new unsigned long long int[obj.count];
    thrust::transform(thrust::device, morton, morton + obj.count, indices, morton64,
                      [] __device__ (const unsigned int m, const unsigned int idx) {
                          unsigned long long int m64 = m;
                          m64 <<= 32;
                          m64 |= idx;
                          return m64;
                      });

    // construct leaf nodes and aabbs
    constexpr bvh_tree::bvh_node default_node{
        .left = 0xFFFFFFFF,
        .right = 0xFFFFFFFF,
        .parent_id = 0xFFFFFFFF
    };

    //set default value for internal nodes
    thrust::fill(
        thrust::device,
        obj.leaf_list,
        obj.leaf_list + num_internal_nodes,
        default_node);

    //init leaf nodes
    thrust::transform(thrust::device, indices, indices + obj.count,
                      obj.leaf_list + num_internal_nodes,
                      [] __device__ (const unsigned int idx) {
                          //if current_node is leaf node, left points to the obj_id, right is 0xFFFFFFFF
                          return bvh_tree::bvh_node{
                              .left = idx,
                              .right = 0xFFFFFFFF,
                              .parent_id = 0xFFFFFFFF
                          };
                      });

    bvh_tree::construct_internal_nodes(obj.leaf_list, morton64, obj.count);

    const auto flags = new int[num_internal_nodes];
    const auto leaf_list = obj.leaf_list;
    const auto bbox_list = obj.bbox_list;
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<unsigned int>(num_internal_nodes),
                     thrust::make_counting_iterator<unsigned int>(obj.node_length),
                     [=] __device__ (const unsigned int idx) {
                         unsigned int parent = leaf_list[idx].parent_id;
                         while (parent != 0xFFFFFFFF) // means idx == 0
                         {
                             //lock in cuda
                             const int old = atomicCAS(flags + parent, 0, 1);
                             if (old == 0) {
                                 // this is the first thread entered here.
                                 // wait the other thread from the other child node.
                                 return;
                             }
                             // assert(old == 1);
                             // here, the flag has already been 1. it means that this
                             // thread is the 2nd thread. merge AABB of both children.

                             const auto lidx = leaf_list[parent].left;
                             const auto ridx = leaf_list[parent].right;
                             const auto lbox = bbox_list[lidx];
                             const auto rbox = bbox_list[ridx];
                             bbox_list[parent] = merge(lbox, rbox);

                             // look the next parent...
                             parent = leaf_list[parent].parent_id;
                         }
                     });

    delete[] morton;
    delete[] morton64;
    delete[] indices;
    delete[] flags;
}

#endif BVH_H
