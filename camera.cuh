#ifndef CAMERA_H
#define CAMERA_H

#include "get_hit.cuh"
#include "material.cuh"
#include <span>
#include <utils.hpp>


class camera {
  public:
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

    void render(const bvh_tree& world) {
        initialize();

        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = 0; j < image_height; j++) {
            std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
            for (int i = 0; i < image_width; i++) {
                color pixel_color(0,0,0);
                for (int sample = 0; sample < samples_per_pixel; sample++) {
                    ray r = get_ray(i, j, nullptr);
                    pixel_color += ray_color(r, max_depth, &world, nullptr);
                }
                write_color(std::cout, pixel_samples_scale * pixel_color);
            }
        }

        std::clog << "\rDone.                 \n";
    }

    __host__ __device__ unsigned int render_pixel(const bvh_tree* world, int row_id, int col_id, curandState *rnd) const {
        color pixel_color(0,0,0);
        for (int sample = 0; sample < samples_per_pixel; sample++) {
            ray r = get_ray(col_id, row_id, rnd);
            pixel_color += ray_color(r, max_depth, world, rnd);
        }
        return pixel_from_color(pixel_samples_scale * pixel_color);
    }

    template <int thread_per_pixel>
    __device__ unsigned int render_pixel(const bvh_tree* world, const int row_id, const int col_id, curandState *rnd,
                                         const int thread_index) const {
        color pixel_color(0,0,0);
        for (int sample = thread_index; sample < samples_per_pixel; sample+=thread_per_pixel) {
            ray r = get_ray(col_id, row_id, rnd);
            pixel_color += ray_color(r, max_depth, world, rnd);
        }
        __syncwarp();
        for (int shfl_dist = thread_per_pixel/2; shfl_dist>0; shfl_dist/=2) {
            pixel_color.x += __shfl_xor_sync(utils::tId_to_warp_mask<thread_per_pixel>(threadIdx.x), pixel_color.x, shfl_dist);
            pixel_color.y += __shfl_xor_sync(utils::tId_to_warp_mask<thread_per_pixel>(threadIdx.x), pixel_color.y, shfl_dist);
            pixel_color.z += __shfl_xor_sync(utils::tId_to_warp_mask<thread_per_pixel>(threadIdx.x), pixel_color.z, shfl_dist);
        }
        return pixel_from_color(pixel_samples_scale * pixel_color);
    }

    void render(const bvh_tree& world, std::span<unsigned int> image) {
        initialize();

        for (int j = 0; j < image_height; j++) {
            utils::log<utils::LogLevel::eVerbose>(
                std::string{"Scanlines remaining: "} + std::to_string(image_height - j));
            for (int i = 0; i < image_width; i++) {
                image[i+j*image_width] = render_pixel(&world, j, i, nullptr);
            }
        }
    }

    void render_parallel(const bvh_tree& world, std::span<unsigned int> image) {
        initialize();
        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads{};
        for (auto tId = 0; tId < num_threads; tId++) {
            threads.emplace_back([tId, num_threads, this, &image, &world]() {
                for (int j = tId; j < image_height; j+=num_threads) {
                    for (int i = 0; i < image_width; i++) {
                        image[i+j*image_width] = render_pixel(&world, j, i, nullptr);
                    }
                }
            });
        }
        for (auto& t : threads) {
            t.join();
        }
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

    __host__ __device__ color ray_color(const ray& r, const int depth, const bvh_tree* world, curandState* rnd) const {
        ray cur_ray = r;
        auto cur_attenuation = vec3(1.0f,1.0f,1.0f);
        for(int i = 0; i < depth; i++) {
            hit_record rec;

            // If the ray hits nothing, return the background color.
            if (!world->hit(cur_ray, interval(0.001f, infinity), rec))
                return cur_attenuation * background;

            NormVec3 scattered_direction;
            color attenuation;

#ifdef __CUDA_ARCH__
            if (CUDA_MATERIALS[rec.mat_id]->will_scatter) {
                attenuation = CUDA_MATERIALS[rec.mat_id]->scatter(cur_ray, rec, scattered_direction, rnd);
            } else {
                color color_from_emission = CUDA_MATERIALS[rec.mat_id]->emitted(rec.u, rec.v);
                return cur_attenuation * color_from_emission;
            }
#else
            if (HOST_MATERIALS[rec.mat_id]->will_scatter) {
                attenuation = HOST_MATERIALS[rec.mat_id]->scatter(cur_ray, rec, scattered_direction, rnd);
            } else {
                color color_from_emission = HOST_MATERIALS[rec.mat_id]->emitted(rec.u, rec.v);
                return cur_attenuation * color_from_emission;
            }
#endif

            cur_attenuation *= attenuation;
            cur_ray = ray(cur_ray.at(rec.t), scattered_direction);
        }
        // If we've exceeded the ray bounce limit, no more light is gathered.
        return color(0,0,0);
    }
};


#endif
