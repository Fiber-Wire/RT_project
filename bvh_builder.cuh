#ifndef BVH_BUILDER_CUH
#define BVH_BUILDER_CUH
#include "bvh.cuh"
#include "geometry.cuh"

__global__ inline void bvh_tree_build(bvh_tree** tree, hittable_list **object_list){
    auto const tree_ptr = new bvh_tree(**object_list);
    *tree = tree_ptr;
}

__global__ inline void clear_bvh_builder(const bvh_tree::info* bvh_info,bvh_tree** box_ball) {
    delete [] bvh_info->d_temp_storage_reduce1;
    delete [] bvh_info->reduce_out1;
    delete [] bvh_info->morton;
    delete [] bvh_info->d_temp_storage_sort;
    delete [] bvh_info->morton_sorted;
    delete [] bvh_info->primitive_list_sorted;
    delete [] bvh_info->morton64;
    delete [] bvh_info->parent_ids;
    delete [] bvh_info->flags;

    delete *box_ball;
}

class bvh_builder {
    public:
    __host__  explicit bvh_builder(hittable_list **list, int num) : num(num) {
        bvh_tree_build<<<1,1>>>(tree.cudaPtr,list);
        initialize_info<<<1,1>>>(tree.cudaPtr,bvh_info.cudaPtr);
    }

    __host__ void update_serial() const {
        constexpr auto block_size = 1;
        constexpr auto grid_size = 1;
        float ms; // elapsed time in milliseconds
        cudaEvent_t startEvent, stopEvent;
        utils::cu_ensure(cudaEventCreate(&startEvent));
        utils::cu_ensure(cudaEventCreate(&stopEvent));
        utils::cu_ensure(cudaEventRecord(startEvent,0));

        aabb_init<<<grid_size,block_size>>>(tree.cudaPtr,bvh_info.cudaPtr);
        aabb_obj_init<<<grid_size,block_size>>>(tree.cudaPtr,bvh_info.cudaPtr);
        aabb_reduce_serial<<<1,1>>>(tree.cudaPtr,bvh_info.cudaPtr);
        morton_code<<<grid_size,block_size>>>(tree.cudaPtr,bvh_info.cudaPtr);
        morton_code_sort_serial<<<1,1>>>(tree.cudaPtr,bvh_info.cudaPtr);
        morton_code_extend_serial<<<grid_size,block_size>>>(tree.cudaPtr,bvh_info.cudaPtr);
        node_init<<<grid_size,block_size>>>(tree.cudaPtr,bvh_info.cudaPtr);
        construct_internal_nodes<<<grid_size,block_size>>>(tree.cudaPtr,bvh_info.cudaPtr);
        set_parent_serial<<<grid_size,block_size>>>(tree.cudaPtr,bvh_info.cudaPtr);

        utils::cu_ensure( cudaEventRecord(stopEvent, 0) );
        utils::cu_ensure( cudaEventSynchronize(stopEvent) );
        utils::cu_ensure( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
        utils::log("Serial bvh building time: " + std::to_string(ms) + " ms\n");
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    __host__ void update() const {
        constexpr auto block_size = 128;
        auto grid_size = std::max(num/block_size, 1);
        float ms; // elapsed time in milliseconds
        cudaEvent_t startEvent, stopEvent;
        utils::cu_ensure(cudaEventCreate(&startEvent));
        utils::cu_ensure(cudaEventCreate(&stopEvent));
        utils::cu_ensure(cudaEventRecord(startEvent,0));

        aabb_init<<<grid_size,block_size>>>(tree.cudaPtr,bvh_info.cudaPtr);
        aabb_obj_init<<<grid_size,block_size>>>(tree.cudaPtr,bvh_info.cudaPtr);
        aabb_reduce<<<1,1>>>(tree.cudaPtr,bvh_info.cudaPtr);
        morton_code<<<grid_size,block_size>>>(tree.cudaPtr,bvh_info.cudaPtr);
        morton_code_sort<<<1,1>>>(tree.cudaPtr,bvh_info.cudaPtr);
        morton_code_extend<<<grid_size,block_size>>>(tree.cudaPtr,bvh_info.cudaPtr);
        node_init<<<grid_size,block_size>>>(tree.cudaPtr,bvh_info.cudaPtr);
        construct_internal_nodes<<<grid_size,block_size>>>(tree.cudaPtr,bvh_info.cudaPtr);
        set_parent<<<grid_size,block_size>>>(tree.cudaPtr,bvh_info.cudaPtr);

        utils::cu_ensure( cudaEventRecord(stopEvent, 0) );
        utils::cu_ensure( cudaEventSynchronize(stopEvent) );
        utils::cu_ensure( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
        utils::log("Parallel bvh building time: " + std::to_string(ms) + " ms\n");
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }
    __host__ ~bvh_builder() {
        clear_bvh_builder<<<1,1>>>(bvh_info.cudaPtr,tree.cudaPtr);
    }

    __host__ bvh_tree** get_scene() const {
        return tree.cudaPtr;
    }
private:
    int num;
    utils::CuArrayRAII<bvh_tree*> tree{nullptr};
    utils::CuArrayRAII<bvh_tree::info> bvh_info{nullptr};
};


#endif //BVH_BUILDER_CUH
