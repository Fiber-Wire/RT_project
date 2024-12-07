#include <future>
#include <semaphore>
#include <string>
#include <thread>
#include "globals.cuh"
#include "sdl_wrapper.hpp"
#include "helpers.cuh"
#include "vec.cuh"

#include "bvh.cuh"
#include "bvh_builder.cuh"
#include "camera.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "material.cuh"
#include "geometry.cuh"
#include "texture.cuh"
#include "curand.h"
#include "curand_kernel.h"
#include "pseudo_rnd.hpp"

__device__ hittable_list* bvh_scene_build(curandState* rnd, const int num, utils::NaiveVector<sphere>** spheres_ptr) {
    auto material_handles = new material*[2];
    CUDA_MATERIALS = material_handles;
    material_handles[0] = new lambertian(color(.73, .73, .73));
    material_handles[1] = new diffuse_light(color(7, 7, 7));

    auto quads = new utils::NaiveVector<quad>{1};

    int ns = num;
    auto spheres = new utils::NaiveVector<sphere>{ns};

    auto boxes2 = new hittable_list{ns + 1};
    for (int j = 0; j < ns; j++) {
        auto center = point3(278,278,0) + random_in_cube(-100,100,rnd);
        spheres->push({center, 5, 0});
        boxes2->add(spheres->end()-1);
    }

    quads->push({point3(278, 554, 0), vec3(300, 0, 0), vec3(0, 0, 265), 1});
    boxes2->add(quads->end()-1);
    *spheres_ptr = spheres;
    return boxes2;
}

__global__ void bvh_scene_build_cuda(hittable_list** world_ptr, curandState* states, const int num, utils::NaiveVector<sphere>** sphere_vector) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid==0) {
        *world_ptr = bvh_scene_build(states,num, sphere_vector);
    }
}

__host__ __device__ camera final_camera(const int image_width, const int samples_per_pixel, const int max_depth) {
    camera cam;

    cam.aspect_ratio      = 1.0;
    cam.image_width       = image_width;
    cam.samples_per_pixel = samples_per_pixel;
    cam.max_depth         = max_depth;
    cam.background        = color(0,0,0);

    cam.vfov     = 40;
    cam.lookfrom = point3(478, 278, -600);
    cam.lookat   = point3(278, 278, 0);
    cam.vup      = vec3(0,1,0);
    return cam;
}

__global__ void camera_init_cuda(camera* cam, bvh_tree** scene) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid==0) {
        cam->world = *scene;
        cam->initialize();
        const auto r = cam->lookfrom-cam->lookat;
        const auto tan_v = -normalize(cross(r, cam->vup));
        constexpr auto dtheta = 0.1f;
        cam->lookfrom += tan_v*length(r)*sinf(dtheta)-r*(1.0f-cosf(dtheta));
    }
}

__global__ void camera_render_cuda(const camera* cam, std::span<unsigned int> image, curandState* devStates) {
    const auto tid = utils::getTId<3, 2>();
    // only use 1, 2, 4, ..., 32
    constexpr auto threadPerPixel = 16;
    //static_assert(threadPerPixel<=BLOCKDIMS.x);

    auto devState = devStates[tid];
    const auto width = cam->image_width;
    const auto height = image.size()/width;

    for (int y = threadIdx.z + blockIdx.y*blockDim.z; y < height; y += blockDim.z*gridDim.y) {
        for (int x = threadIdx.y + blockIdx.x*blockDim.y; x < width; x += blockDim.y*gridDim.x) {
            auto pixel_sampled_result = cam->render_pixel_block<threadPerPixel>(y, x, &devState);
            __syncwarp(utils::tId_to_warp_mask<threadPerPixel>(tid));
            for (int shfl_dist = threadPerPixel/2; shfl_dist>0; shfl_dist/=2) {
                pixel_sampled_result.x += __shfl_xor_sync(utils::tId_to_warp_mask<threadPerPixel>(tid), pixel_sampled_result.x, shfl_dist);
                pixel_sampled_result.y += __shfl_xor_sync(utils::tId_to_warp_mask<threadPerPixel>(tid), pixel_sampled_result.y, shfl_dist);
                pixel_sampled_result.z += __shfl_xor_sync(utils::tId_to_warp_mask<threadPerPixel>(tid), pixel_sampled_result.z, shfl_dist);
            }
            if (threadIdx.x == 0) {
                image[x+y*width] = pixel_from_color(pixel_sampled_result);
            }
        }
    }

    devStates[tid] = devState;
}

__global__ void spheres_motion(utils::NaiveVector<sphere>** spheres_ptr, curandState* devStates) {
    const auto tid = utils::getTId<3, 2>();
    const auto gsize = utils::getGrid<3,2>();
    auto& spheres = *spheres_ptr;
    for (auto sid = tid; sid < spheres->count; sid +=gsize) {
        auto& s = spheres->data[sid];
        s = sphere{s.bounding_box().center()+random_in_cube(-5, 5, devStates+tid), s.bounding_box().x.size()/2, 0};
    }
}

void render_thread_cuda(const camera& cam, camera* cam_cuda, bvh_tree** scene_cuda, std::span<unsigned int> image, curandState* devStates,
    bvh_builder& builder, utils::CuArrayRAII<utils::NaiveVector<sphere>*>& spheres, const int parallel_build){
    const int height = static_cast<int>(cam.image_width / cam.aspect_ratio);
    const int width  = cam.image_width;
    unsigned int* imageGpuPtr{};
    cudaMalloc(&imageGpuPtr, image.size()*sizeof(unsigned int));
    //utils::cu_ensure(cudaFuncSetCacheConfig(camera_render_cuda, cudaFuncCachePreferL1));
    const std::span imageGpu{imageGpuPtr, static_cast<std::span<unsigned>::size_type>(height*width)};
    while (!mainRendererComm.stop_render.load()) {
        if (mainRendererComm.frame_start_render.try_acquire()) {
            camera_init_cuda<<<1,1>>>(cam_cuda, scene_cuda);
            spheres_motion<<<GRIDDIMS,BLOCKDIMS>>>(spheres.cudaPtr, devStates);
            if(parallel_build == 1)
                builder.update();
            else builder.update_serial();
            camera_render_cuda<<<GRIDDIMS,BLOCKDIMS>>>(cam_cuda, imageGpu, devStates);
            utils::cu_ensure(cudaMemcpy(image.data(), imageGpuPtr, image.size()*sizeof(unsigned int), cudaMemcpyDeviceToHost));
            mainRendererComm.frame_rendered.release();
        }
        std::this_thread::yield();
    }
    cudaFree(imageGpuPtr);
}

void render_scene_realtime_cuda(bvh_tree** scene, camera &cam, camera *cam_cuda, curandState* devStates,
    const int max_frame, const int parallel_build,
    bvh_builder& builder, utils::CuArrayRAII<utils::NaiveVector<sphere>*>& spheres) {
    const int height = static_cast<int>(cam.image_width / cam.aspect_ratio);
    auto window = sdl_raii::Window{"RT_project", cam.image_width, height};
    auto renderer = sdl_raii::Renderer{window.get()};
    auto surface = sdl_raii::Surface{cam.image_width, height};
    const auto image = std::span{static_cast<unsigned int *>(surface.get()->pixels), static_cast<size_t>(cam.image_width)*height};
    std::promise<void> render_finished;
    const std::future<void> render_finished_future = render_finished.get_future();
    std::thread{[=, &render_finished, &cam, &scene, &builder, &spheres] {
        render_thread_cuda(cam, cam_cuda, scene, image, devStates, builder, spheres, parallel_build);
        render_finished.set_value_at_thread_exit();
    }}.detach();
    mainRendererComm.frame_start_render.release();
    size_t frames = 0;
    std::chrono::microseconds frame_times{};
    auto t0 = std::chrono::steady_clock::now();
    while (!want_exit_sdl() && ((frames < max_frame) || (max_frame < 0)))
    {
        if (mainRendererComm.frame_rendered.try_acquire()) {
            auto texture = sdl_raii::Texture{renderer.get(), surface.get()};
            SDL_RenderClear(renderer.get());
            SDL_RenderCopy(renderer.get(),texture.get(), nullptr, nullptr);
            SDL_RenderPresent(renderer.get());
            auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-t0);
            frames += 1;
            frame_times += frame_time;
            utils::log("Frame time: "+std::to_string(frame_time.count()/1e3)+" ms");
            mainRendererComm.frame_start_render.release();
            t0 = std::chrono::steady_clock::now();
        }
        std::this_thread::yield();
    }
    utils::log("Total frames: "+std::to_string(frames)+
           ", avg. frame time: "+std::to_string(frame_times.count()/frames/1e3)+" ms.");
    notify_renderer_exit();
    while(render_finished_future.wait_for(std::chrono::milliseconds{5})==std::future_status::timeout) {
        want_exit_sdl();
    }
    render_finished_future.wait();
}

__global__ void initCurand(curandState *state, unsigned long seed){
    const auto idx = utils::getTId<3,2>();
    curand_init(seed, idx, 0, &state[idx]);
}

void parse_arguments(int argc, char** argv, int& size, int& samples, int& depth, int& frame, int& obj_num, int &parallel_build) {
    for (int i = 1; i < argc; i += 2) {
        if (std::string(argv[i]) == "--size") {
            size = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--samples") {
            samples = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--depth") {
            depth = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--frame") {
            frame = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--obj_num") {
            obj_num = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--parallel_build") {
            parallel_build = std::atoi(argv[i + 1]);
        } else {
            std::cerr << "Usage: " << argv[0]
                << " --size <int> --depth <int> --samples <int> --frame <int> --obj_num <int> --parallel_build <int>\n\n"
                   "       --size: width of image in px\n"
                   "       --depth: maximum depth for rays\n"
                   "       --samples: number of samples per pixel\n"
                   "       --frame: non-stop when set to negative\n"
                   "       --obj_num: number of spheres generated\n"
                   "       --parallel_build: 1 for parallel bvh building\n" << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    sdl_raii::SDL sdl{};
    initialize_main_sync_objs();

    int size = 512, samples = 32, depth = 4, frame = 62, obj_num = 1000, parallel_build = 1;
    parse_arguments(argc, argv, size, samples, depth, frame, obj_num, parallel_build);
    GRIDDIMS.x = size/BLOCKDIMS.y;
    GRIDDIMS.y = size/BLOCKDIMS.z;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024<<20);

    auto cam = final_camera(size, samples, depth);

    const auto numThreads = GRIDDIMS.x * GRIDDIMS.y * GRIDDIMS.z * BLOCKDIMS.x * BLOCKDIMS.y * BLOCKDIMS.z;
    const utils::CuArrayRAII<curandState> devStates{nullptr, numThreads};
    // Cherry-picked seed
    initCurand<<<GRIDDIMS,BLOCKDIMS>>>(devStates.cudaPtr, 1);
    const utils::CuArrayRAII camGpuPtr{&cam};
    const utils::CuArrayRAII<hittable_list*> listGpuPtr{nullptr};
    utils::CuArrayRAII<utils::NaiveVector<sphere>*> sphere_vector{nullptr};
    bvh_scene_build_cuda<<<1,1>>>(listGpuPtr.cudaPtr, devStates.cudaPtr,obj_num, sphere_vector.cudaPtr);
    utils::cu_ensure(cudaDeviceSynchronize());
    bvh_builder builder(listGpuPtr.cudaPtr, obj_num + 1);

    render_scene_realtime_cuda(builder.get_scene(), cam, camGpuPtr.cudaPtr, devStates.cudaPtr, frame, parallel_build, builder, sphere_vector);
    // we don't do the cleanup yet, but this will make compute-sanitizer unhappy
    //cudaDeviceReset();
    return 0;
}
