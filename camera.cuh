#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.cuh"
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

    void render(const hittable& world) {
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

    __host__ __device__ unsigned int render_pixel(const hittable* world, int j, int i, curandState *rnd) {
        color pixel_color(0,0,0);
        for (int sample = 0; sample < samples_per_pixel; sample++) {
            ray r = get_ray(i, j, rnd);
            pixel_color += ray_color(r, max_depth, world, rnd);
        }
        return pixel_from_color(pixel_samples_scale * pixel_color);
    }

    template <int thread_per_pixel>
    __device__ unsigned int render_pixel(const hittable* world, const int j, const int i, curandState *rnd,
                                         const int thread_index) const {
        color pixel_color(0,0,0);
        for (int sample = thread_index; sample < samples_per_pixel; sample+=thread_per_pixel) {
            ray r = get_ray(i, j, rnd);
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

    void render(const hittable& world, std::span<unsigned int> image) {
        initialize();

        for (int j = 0; j < image_height; j++) {
            utils::log<utils::LogLevel::eVerbose>(
                std::string{"Scanlines remaining: "} + std::to_string(image_height - j));
            for (int i = 0; i < image_width; i++) {
                image[i+j*image_width] = render_pixel(&world, j, i, nullptr);
            }
        }
    }

    void render_parallel(const hittable& world, std::span<unsigned int> image) {
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
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta/2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (float(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / (float)image_width;
        pixel_delta_v = viewport_v / (float)image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2.0f - viewport_v/2.0f;
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



    __host__ __device__ ray get_ray(int i, int j, curandState* rnd) const {
        // Construct a camera ray originating from the defocus disk and directed at a randomly
        // sampled point around the pixel location i, j.

        auto offset = sample_square(rnd);
        auto pixel_sample = pixel00_loc
                          + ((i + offset.x) * pixel_delta_u)
                          + ((j + offset.y) * pixel_delta_v);

        auto ray_origin = center;
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    __host__ __device__ vec3 sample_square(curandState* rnd) const {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return vec3(random_float(rnd) - 0.5, random_float(rnd) - 0.5, 0);
    }

    __host__ __device__ color ray_color(const ray& r, int depth, const hittable* world, curandState* rnd) const {
        ray cur_ray = r;
        vec3 cur_attenuation = vec3(1.0f,1.0f,1.0f);
        for(int i = 0; i < depth; i++) {
            hit_record rec;

            // If the ray hits nothing, return the background color.
            if (!world->hit(cur_ray, interval(0.001, INFINITY), rec))
                return cur_attenuation * background;

            ray scattered;
            color attenuation;
            color color_from_emission = rec.mat->emitted(rec.u, rec.v, rec.p);

            if (!rec.mat->scatter(cur_ray, rec, attenuation, scattered, rnd))
                return cur_attenuation * color_from_emission;

            cur_attenuation *= attenuation;
            cur_ray = scattered;
        }
        // If we've exceeded the ray bounce limit, no more light is gathered.
        return color(0,0,0);
    }
};


#endif
