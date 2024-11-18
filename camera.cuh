#ifndef CAMERA_H
#define CAMERA_H

#include "get_hit.cuh"
#include "material.cuh"
#include <span>
#include <utils.hpp>

using RayInfo = ray_info;
using BlockReduceBounds = cub::BlockReduce<RayInfo::bounds, blkx_t, cub::BLOCK_REDUCE_WARP_REDUCTIONS, blky_t, blkz_t>;
using BlockSortMorton = cub::BlockRadixSort<
        // KeyT, blkx, numPerThread, ValT
        unsigned int, blkx_t, numRays_t, RayInfo,
        // default params
        4, true, cub::BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeFourByte,
        // blky, blkz
        blky_t, blkz_t>;
using BlockReduceRayCount = cub::BlockReduce<int, blkx_t, cub::BLOCK_REDUCE_WARP_REDUCTIONS, blky_t, blkz_t>;
template <int samples=samplePPx_t>
struct block_ray_query_container {
    struct bounds_container {
        BlockReduceBounds::TempStorage bound_temp;
        RayInfo::bounds block_bounds_temp;
        __device__ bounds_container() {}
    };
    struct finished_container {
        BlockReduceRayCount::TempStorage finish_storage;
        bool finished{};
        __device__ finished_container() {}
    };
    union block_comms {
        BlockSortMorton::TempStorage morton_temp;
        finished_container finished_temp;
        bounds_container bounds_temp;
        __device__ block_comms() {}
    };
    block_comms comms;
    short num_samples[blky_t*blkz_t];
    __device__ block_ray_query_container() {}
    __device__ void reset() {
        for (auto &n : num_samples) {
            n = samples;
        }
    }

    __device__ short request_samples(int &req) {
        if constexpr (blkx_t > 0) {
            const auto mask = utils::tId_to_warp_mask<blkx_t>(utils::getBTId<3>());
            const auto bTId_offset = utils::getBTId<3>()-utils::getBTId<1>();
            short first = samples;
            __syncwarp(mask);
            auto& s = num_samples[threadIdx.y+threadIdx.z*blky_t];
            for (int i=0; i<blkx_t; i++) {
                int req_i = __shfl_sync(mask, req, (i+bTId_offset)%32);
                short first_i = samples-s;
                if (threadIdx.x == 0) {
                    if (req_i > s) {
                        req_i = s;
                        s = 0;
                    } else {
                        s -= req_i;
                    }
                }
                const int res_i = __shfl_sync(mask, req_i, bTId_offset%32);
                if (threadIdx.x == i) {
                    req = res_i;
                    first = first_i;
                }
            }
            return first;
        } else {
            auto& s = num_samples[threadIdx.y+threadIdx.z*blky_t];
            const short first = samples-s;
            req = min(s, req);
            s = max(s-req, 0);
            return first;
        }
    }

    __device__ bool block_finished(int valid_rays) {
        auto& finish_storage = comms.finished_temp.finish_storage;
        auto& finished = comms.finished_temp.finished;
        valid_rays = BlockReduceRayCount(finish_storage).Sum(valid_rays);
        if (utils::getBTId<3>()==0) {
            if (valid_rays > 0) {
                finished = false;
            } else {
                bool finished_i = true;
                for (const auto& s: num_samples) {
                    finished_i &= (s ==0);
                }
                finished = finished_i;
            }
        }
        __syncthreads();
        return finished;
    }
    __device__ RayInfo::bounds find_ray_bounds(const RayInfo (&rays)[numRays_t]) {
        auto& bound_temp = comms.bounds_temp.bound_temp;
        auto& block_bounds = comms.bounds_temp.block_bounds_temp;
        RayInfo::bounds result{};
        for (const auto& ray : rays) {
            if (ray.is_valid_ray()) {
                result.add_point(ray.r0);
            }
        }

        result = BlockReduceBounds(bound_temp).Reduce(
            result, []__device__ (const RayInfo::bounds &a, const RayInfo::bounds &b) {
            return RayInfo::bounds::merge_op(a, b);
        });
        block_bounds = result;
        __syncthreads();
        return block_bounds;
    }
    __device__ void block_morton_sort(RayInfo (&local_rays)[numRays_t], const RayInfo::bounds &bound) {
        unsigned int local_mortons[numRays_t];
        //auto bound = comms.bounds_temp.block_bounds_temp;
        for (int i = 0; i < numRays_t; i++) {
            local_mortons[i] = RayInfo::get_morton(local_rays[i], bound);
        }
        //__syncthreads();
        BlockSortMorton(comms.morton_temp).SortBlockedToStriped(local_mortons, local_rays);
    }

};


class camera {
  public:
    const bvh_tree *world;
    float aspect_ratio      = 1.0;  // Ratio of image width over height
    int    image_width       = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;   // Count of random samples for each pixel
    int    max_depth         = 10;   // Maximum number of ray bounces into scene
    color  background;               // Scene background color

    float vfov     = 90;              // Vertical view angle (field of view)
    point3 lookfrom = point3(0,0,0);   // Point camera is looking from
    point3 lookat   = point3(0,0,-1);  // Point camera is looking at
    vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction

    float focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

    template <int thread_per_pixel, int samples=samplePPx_t>
__device__ void render_pixel_block(const int row_offset, const int col_offset,
    block_ray_query_container<samples> *block_comms,
    color *output_samples,
    curandState *rnd) const {
    const auto col_id = col_offset + threadIdx.y+blockIdx.x*blky_t;
    const auto row_id = row_offset + threadIdx.z+blockIdx.y*blkz_t;
    const auto bTId = utils::getBTId<3>();
    int numSample = samples/thread_per_pixel;
    constexpr int numPx = blky_t*blkz_t;
    RayInfo local_rays[numRays_t];
    for (auto& r: local_rays) {
        r.retire();
    }
    // fill
    int requested_samples = numRays_t;
    auto filled_rays = fill_local_rays<samples>(block_comms, rnd, col_id, row_id, local_rays, requested_samples);
    do {
        // query
        for (auto &r_info : local_rays) {
            if (r_info.is_valid_ray()) {
                bool ray_complete;
                vec3 attenuation;
                std::tie(ray_complete, attenuation) = ray_propagate(r_info.r0, rnd);
                r_info.pixel *= attenuation;
                r_info.dep_smp.x -= 1;
                r_info.update_pixel_state(ray_complete);
            }
        }
        // shade
        requested_samples = 0;
        for (auto &r_info : local_rays) {
            if (!r_info.is_valid_ray()) {
                if (r_info.is_pixel_ready()) {
                    auto [x,y] = r_info.get_px_offset();
                    x += col_offset;
                    y += row_offset;
                    output_samples[r_info.dep_smp.y+(x+y*width_t)*samples] = r_info.pixel;
                    r_info.retire();
                }
                requested_samples += 1;
            }
        }
        // fill
        filled_rays = fill_local_rays<samples>(block_comms, rnd, col_id, row_id, local_rays, requested_samples);

        // sort
        const auto bound = block_comms->find_ray_bounds(local_rays);
        block_comms->block_morton_sort(local_rays, bound);
        //FIXME: number of valid rays will change after sort
    } while(!block_comms->block_finished(numRays_t-requested_samples+filled_rays));
}

    __host__ __device__ void initialize() {
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        // Determine viewport dimensions.
        const auto theta = degrees_to_radians(vfov);
        const auto h = std::tan(theta/2);
        const auto viewport_height = 2 * h * focus_dist;
        const auto viewport_width = viewport_height * (static_cast<float>(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        const vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        const vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / static_cast<float>(image_width);
        pixel_delta_v = viewport_v / static_cast<float>(image_height);

        // Calculate the location of the upper left pixel.
        const auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2.0f - viewport_v/2.0f;
        pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

    }


  private:
    int    image_height;         // Rendered image height
    float pixel_samples_scale;  // Color scale factor for a sum of pixel samples
    point3 center;               // Camera center
    point3 pixel00_loc;          // Location of pixel 0, 0
    vec3   pixel_delta_u;        // Offset to pixel to the right
    vec3   pixel_delta_v;        // Offset to pixel below
    vec3   u, v, w;              // Camera frame basis vectors

    template <int samples>
    __device__ int fill_local_rays(block_ray_query_container<samples> *block_comms, curandState *rnd,
        const unsigned col_id, const unsigned row_id, RayInfo (&local_rays)[numRays_t], int requested_samples) const {
        auto sid_first = block_comms->request_samples(requested_samples);
        const auto filled_samples = requested_samples;
        // if (sid_first+requested_samples> samples) {
        //     printf("Block {%d, %d} Thread (%d, %d, %d): allowed: %d samples, starting at %d\n",
        //     blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.z,
        //     requested_samples, sid_first);
        // }
        int ray_counter = 0;
        while (requested_samples > 0 && ray_counter < numRays_t) {
            if (auto &r_info = local_rays[ray_counter]; !r_info.is_valid_ray()) {
                r_info.pixel = color(1.0f, 1.0f, 1.0f);
                r_info.r0 = get_ray(col_id, row_id, rnd);
                r_info.set_px_offset(threadIdx.y+blockIdx.x*blky_t, threadIdx.z+blockIdx.y*blkz_t);
                r_info.dep_smp.x = max_depth;
                r_info.dep_smp.y = sid_first;
                requested_samples -= 1;
                sid_first += 1;
                // if (blockIdx.x==0&& blockIdx.y==0&& threadIdx.x==0&& threadIdx.y==0&& threadIdx.z==0) {
                //     printf("Block {%d, %d} Thread (%d, %d, %d): xy (%f, %f)\n",
                // blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.z,
                // r_info.r0.direction().get_compressed().x, r_info.r0.direction().get_compressed().y);
                // }
            }

            ray_counter++;
        }
        return filled_samples;
    }

    __host__ __device__ ray get_ray(const int col_id, const int row_id, curandState* rnd) const {
        // Construct a camera ray originating from the defocus disk and directed at a randomly
        // sampled point around the pixel location i, j.

        const auto offset = sample_square(rnd);
        const auto pixel_sample = pixel00_loc
                                  + ((col_id + offset.x) * pixel_delta_u)
                                  + ((row_id + offset.y) * pixel_delta_v);

        const auto ray_origin = center;
        const auto ray_direction = normalize(pixel_sample - ray_origin);

        return ray(ray_origin, ray_direction);
    }

    __host__ __device__ static vec3 sample_square(curandState* rnd) {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return vec3(random_float(rnd) - 0.5f, random_float(rnd) - 0.5f, 0.0f);
    }

    __host__ __device__ std::tuple<bool, color> ray_propagate(ray &cur_ray, curandState *rnd) const {
        bool end = false;
        hit_record rec;
        color attenuation;
        // If the ray hits nothing, return the background color.
        if (!world->hit(cur_ray, interval(0.001f, infinity), rec)) {
            end = true;
            attenuation = background;
        } else {
#ifdef __CUDA_ARCH__
            auto material_ptr = CUDA_MATERIALS;
#else
            auto material_ptr = HOST_MATERIALS;
#endif
            if (material_ptr[rec.mat_id]->will_scatter) {
                NormVec3 scattered_direction;
                attenuation = material_ptr[rec.mat_id]->scatter(cur_ray, rec, scattered_direction, rnd);
                cur_ray = ray(cur_ray.at(rec.t), scattered_direction);
            } else {
                end = true;
                attenuation = material_ptr[rec.mat_id]->emitted(rec.u, rec.v);
            }
        }
        return std::make_tuple(end, attenuation);
    }
};


#endif
