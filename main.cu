#include <future>
#include <semaphore>
#include <string>
#include <thread>
#include "globals.cuh"
#include "sdl_wrapper.hpp"
#include "helpers.cuh"
#include "vec.cuh"

#include "bvh.cuh"
#include "camera.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "material.cuh"
#include "geometry.cuh"
#include "texture.cuh"
#include "curand.h"
#include "curand_kernel.h"
#include "pseudo_rnd.hpp"

__host__ __device__ bvh_tree *final_scene_build(curandState *rnd, const image_record *image_rd) {
    hittable_list world{8};

    // Geometry primitives
    auto spheres = new utils::NaiveVector<sphere>{1029};
    auto quads = new utils::NaiveVector<quad>{2401};

    // Materials
    //   Textures needed by materials;
    auto image_texture_emat = new image_texture(image_rd[0]);
    auto material_handles = new material *[8];
#ifdef __CUDA_ARCH__
    CUDA_MATERIALS = material_handles;
#else
    HOST_MATERIALS = material_handles;
#endif
    material_handles[0] = new lambertian(color(0.48, 0.83, 0.53));
    material_handles[1] = new diffuse_light(color(7, 7, 7));
    material_handles[2] = new dielectric(1.5);
    material_handles[3] = new metal(color(0.8, 0.8, 0.9), 1.0);
    material_handles[4] = new dielectric(1.5);
    material_handles[5] = new lambertian(image_texture_emat);
    material_handles[6] = new lambertian(color(.73, .73, .73));
    material_handles[7] = new metal(color(212.0f / 256, 175.0f / 256, 55.0f / 256), 0.025);


    // 1st item
    auto ground = material_handles[0];
    int boxes_per_side = 20;
    hittable_list boxes1{boxes_per_side * boxes_per_side};
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0f;
            auto x0 = -1000.0f + i * w;
            auto z0 = -1000.0f + j * w;
            auto y0 = 0.0f;
            auto x1 = x0 + w - 0.1f;
            // Get identical scene between runs
            // compute-sanitizer does not like what we do here
#ifdef __CUDA_ARCH__
            auto y1 = random_float(1,101, rnd);
#else
            auto y1 = get_rnd(i * boxes_per_side + j) * 100 + 1;
#endif
            auto z1 = z0 + w - 0.1f;

            auto box3 = create_box(point3(x0, y0, z0), point3(x1, y1, z1), 0, quads);
            boxes1.add(static_cast<hittable *>(box3));
        }
    }
    auto bvh_node_boxes1 = new bvh_tree{boxes1};
    world.add(bvh_node_boxes1);

    // 2nd
    auto light = material_handles[1];
    quads->push({point3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), 1});
    world.add(quads->end() - 1);

    // 3rd
    auto dielectric_sphere = material_handles[2];
    spheres->push({point3(260, 150, 45), 50, 2});
    world.add(spheres->end() - 1);

    //4th
    auto metal_sphere = material_handles[3];
    spheres->push({point3(0, 150, 145), 50, 3});
    world.add(spheres->end() - 1);

    //5th
    auto dielectric_ground = material_handles[4];
    spheres->push({point3(360, 150, 145), 70, 4});
    world.add(spheres->end() - 1);

    //6th
    auto lambertian_emat = material_handles[5];
    spheres->push({point3(400, 200, 400), 100, 5});
    world.add(spheres->end() - 1);

    //7th
    int ns = 1000;
    auto boxes2 = hittable_list{ns};
    auto white = material_handles[6];
    for (int j = 0; j < ns; j++) {
        // compute-sanitizer does not like what we do here
#ifdef __CUDA_ARCH__
        auto center = random_in_cube(0,165, rnd);
#else
        auto center = get_rand_vec3(j);
#endif
        spheres->push({center, 10, 6});
        boxes2.add(spheres->end() - 1);
    }
    auto bvh_node_box = new bvh_tree(boxes2);
    //cudaDeviceSynchronize();
    auto bvh_node_box_rotate_y = new rotate_y(bvh_node_box, 15);
    auto bvh_node_box_translate = new translate(bvh_node_box_rotate_y, vec3(-100, 270, 395));
    world.add(bvh_node_box_translate);

    //8th
    auto metal_2 = material_handles[7];
    spheres->push({point3(240, 320, 400), 60, 7});
    world.add(spheres->end() - 1);

    auto tree = new bvh_tree{world};
    return tree;
}

__global__ void final_scene_build_cuda(bvh_tree **world_ptr, curandState *states, const image_record *image_rd) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        *world_ptr = final_scene_build(states, image_rd);
    }
}

__host__ __device__ camera final_camera(const int image_width, const int samples_per_pixel, const int max_depth) {
    camera cam;

    cam.aspect_ratio = 1.0;
    cam.image_width = image_width;
    cam.samples_per_pixel = samples_per_pixel;
    cam.max_depth = max_depth;
    cam.background = color(0, 0, 0);

    cam.vfov = 40;
    cam.lookfrom = point3(478, 278, -600);
    cam.lookat = point3(278, 278, 0);
    cam.vup = vec3(0, 1, 0);
    return cam;
}

__global__ void camera_init_cuda(camera *cam, bvh_tree **scene) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        cam->world = *scene;
        cam->initialize();
        const auto r = cam->lookfrom - cam->lookat;
        const auto tan_v = -normalize(cross(r, cam->vup));
        constexpr auto dtheta = 0.1f;
        cam->lookfrom += tan_v * length(r) * sinf(dtheta) - r * (1.0f - cosf(dtheta));
    }
}

template<int threadPerPixel = blkx_t>
__global__ void camera_render_cuda(const camera *cam, std::span<color> output_samples,
                                   curandState *devStates) {
    const auto tid = utils::getTId<3, 2>();
    const auto bTId = utils::getBTId<3>();
    __shared__ block_ray_query_container<> shared_comms;
    constexpr auto shared_size = sizeof(shared_comms);

    __syncthreads();

    auto devState = devStates[tid];
    constexpr auto img_width = width_t;
    constexpr auto img_height = height_t;

    for (int y = threadIdx.z + blockIdx.y * blockDim.z; y < img_height; y += blockDim.z * gridDim.y) {
        for (int x = threadIdx.y + blockIdx.x * blockDim.y; x < img_width; x += blockDim.y * gridDim.x) {
            if (bTId == 0) {
                shared_comms.reset();
            }
            __syncthreads();
            cam->render_pixel_block<threadPerPixel>(
                y, x,
                &shared_comms,
                output_samples.data(),
                &devState);
        }
    }
    devStates[tid] = devState;
}

__global__ void merge_samples_cuda(const camera *cam, std::span<color> input_samples,
                                   std::span<unsigned int> output_image) {
    constexpr auto threadPerPixel = 32;
    const auto width = cam->image_width;
    const auto height = output_image.size() / width;
    for (int y = blockIdx.y; y < height; y += gridDim.y) {
        for (int x = blockIdx.x; x < width; x += gridDim.x) {
            // Specialize WarpReduce for type int
            using WarpReduce = cub::WarpReduce<color>;

            // Allocate WarpReduce shared memory for 4 warps
            __shared__ typename WarpReduce::TempStorage temp_storage;

            // Return the warp-wide sums to each lane0 (threads 0, 32, 64, and 96)
            color aggregate = WarpReduce(temp_storage).Sum(
                input_samples[threadIdx.x + threadPerPixel * (x + y * width)]);
            if (threadIdx.x == 0) {
                output_image[x + y * width] = pixel_from_color(aggregate / static_cast<float>(cam->samples_per_pixel));
            }
            __syncthreads();
        }
    }
}

void render_thread_cuda(const camera &cam, camera *cam_cuda, bvh_tree **scene_cuda, std::span<unsigned int> image,
                        curandState *devStates) {
    const int height = static_cast<int>(cam.image_width / cam.aspect_ratio);
    const int width = cam.image_width;
    utils::CuArrayRAII<color> dev_samples{nullptr, image.size() * cam.samples_per_pixel};
    unsigned int *imageGpuPtr{};
    cudaMalloc(&imageGpuPtr, image.size() * sizeof(unsigned int));
    //utils::cu_ensure(cudaFuncSetCacheConfig(camera_render_cuda, cudaFuncCachePreferL1));
    const std::span imageGpu{imageGpuPtr, static_cast<std::span<unsigned>::size_type>(height * width)};
    while (!mainRendererComm.stop_render.load()) {
        if (mainRendererComm.frame_start_render.try_acquire()) {
            camera_init_cuda<<<1,1>>>(cam_cuda, scene_cuda);
            camera_render_cuda<<<GRIDDIMS,BLOCKDIMS>>>(cam_cuda, dev_samples.cudaView, devStates);
            merge_samples_cuda<<<GRIDDIMS, 32>>>(cam_cuda, dev_samples.cudaView, imageGpu);
            utils::cu_ensure(cudaMemcpy(image.data(), imageGpuPtr, image.size() * sizeof(unsigned int),
                                        cudaMemcpyDeviceToHost));
            mainRendererComm.frame_rendered.release();
        }
        std::this_thread::yield();
    }
    cudaFree(imageGpuPtr);
}

void render_scene_realtime_cuda(bvh_tree **scene, camera &cam, camera *cam_cuda, curandState *devStates,
                                const int &max_frame) {
    const int height = static_cast<int>(cam.image_width / cam.aspect_ratio);
    auto window = sdl_raii::Window{"RT_project", cam.image_width, height};
    auto renderer = sdl_raii::Renderer{window.get()};
    auto surface = sdl_raii::Surface{cam.image_width, height};
    const auto image = std::span{
        static_cast<unsigned int *>(surface.get()->pixels), static_cast<size_t>(cam.image_width) * height
    };
    std::promise<void> render_finished;
    const std::future<void> render_finished_future = render_finished.get_future();
    std::thread{
        [=, &render_finished, &cam, &scene] {
            render_thread_cuda(cam, cam_cuda, scene, image, devStates);
            render_finished.set_value_at_thread_exit();
        }
    }.detach();
    mainRendererComm.frame_start_render.release();
    size_t frames = 0;
    std::chrono::microseconds frame_times{};
    auto t0 = std::chrono::steady_clock::now();
    while (!want_exit_sdl() && ((frames < max_frame) || (max_frame < 0))) {
        if (mainRendererComm.frame_rendered.try_acquire()) {
            auto texture = sdl_raii::Texture{renderer.get(), surface.get()};
            SDL_RenderClear(renderer.get());
            SDL_RenderCopy(renderer.get(), texture.get(), nullptr, nullptr);
            SDL_RenderPresent(renderer.get());
            auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0);
            frames += 1;
            frame_times += frame_time;
            utils::log("Frame time: " + std::to_string(frame_time.count() / 1e3) + " ms");
            mainRendererComm.frame_start_render.release();
            t0 = std::chrono::steady_clock::now();
        }
        std::this_thread::yield();
    }
    utils::log("Total frames: " + std::to_string(frames) +
               ", avg. frame time: " + std::to_string(frame_times.count() / frames / 1e3) + " ms.");
    notify_renderer_exit();
    while (render_finished_future.wait_for(std::chrono::milliseconds{5}) == std::future_status::timeout) {
        want_exit_sdl();
    }
    render_finished_future.wait();
}

__global__ void initCurand(curandState *state, unsigned long seed) {
    const auto idx = utils::getTId<3, 2>();
    curand_init(seed, idx, 0, &state[idx]);
}

void parse_arguments(int argc, char **argv, int &size, int &samples, int &depth, int &frame) {
    for (int i = 1; i < argc; i += 2) {
        if (std::string(argv[i]) == "--depth") {
            depth = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--frame") {
            frame = std::atoi(argv[i + 1]);
        } else {
            std::cerr << "Usage: " << argv[0]
                    << "--depth <int> --frame <int>\n\n"
                    "       --depth: maximum depth for rays\n"
                    "       --frame: non-stop when set to negative\n" << std::endl;
        }
    }
}

int main(int argc, char *argv[]) {
    sdl_raii::SDL sdl{};
    initialize_main_sync_objs();

    int size = 512, samples = 32, depth = 4, frame = 62;
    parse_arguments(argc, argv, size, samples, depth, frame);
    size = width_t;
    samples = samplePPx_t;
    GRIDDIMS.x = size / BLOCKDIMS.y;

    GRIDDIMS.y = size / BLOCKDIMS.z;

    auto image_ld = image_loader("earthmap.jpg");
    auto cam = final_camera(size, samples, depth);

    auto rec_cuda = image_ld.get_record_cuda();
    const utils::CuArrayRAII image_rd{&rec_cuda};
    const auto numThreads = GRIDDIMS.x * GRIDDIMS.y * GRIDDIMS.z * BLOCKDIMS.x * BLOCKDIMS.y * BLOCKDIMS.z;
    const utils::CuArrayRAII<curandState> devStates{nullptr, numThreads};
    // Cherry-picked seed
    initCurand<<<GRIDDIMS,BLOCKDIMS>>>(devStates.cudaPtr, 1);
    const utils::CuArrayRAII<bvh_tree *> sceneGpuPtr{nullptr};
    const utils::CuArrayRAII camGpuPtr{&cam};
    final_scene_build_cuda<<<1,1>>>(sceneGpuPtr.cudaPtr, devStates.cudaPtr, image_rd.cudaPtr);
    utils::cu_ensure(cudaDeviceSynchronize());
    render_scene_realtime_cuda(sceneGpuPtr.cudaPtr, cam, camGpuPtr.cudaPtr, devStates.cudaPtr, frame);
    // we don't do the cleanup yet, but this will make compute-sanitizer unhappy
    //cudaDeviceReset();
    return 0;
}
