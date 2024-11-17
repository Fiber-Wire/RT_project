#include <future>
#include <semaphore>
#include <string>
#include <thread>
#include "sdl_wrapper.hpp"
#include "helpers.cuh"

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
#define BLOCKDIM_X 64
#define GRIDDIM_X (625*2)
struct MainRendererComm{
    std::binary_semaphore frame_start_render{0};
    std::binary_semaphore frame_rendered{0};
    std::atomic<bool> stop_render{false};
};
MainRendererComm mainRendererComm{};
void initialize_main_sync_objs(){
    mainRendererComm.frame_start_render.try_acquire();
    mainRendererComm.frame_rendered.try_acquire();
    mainRendererComm.stop_render.store(false);
}
void notify_renderer_exit(){
    mainRendererComm.stop_render.store(true);
}

hittable_list final_scene_build() {
    auto ground = new lambertian(color(0.48, 0.83, 0.53));

    int boxes_per_side = 20;
    hittable_list boxes1{boxes_per_side*boxes_per_side};
    //////// vector<hittable*> all_boxes3
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i*w;
            auto z0 = -1000.0 + j*w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_float(1,101, nullptr);
            auto z1 = z0 + w;

            auto box3 = create_box(point3(x0,y0,z0), point3(x1,y1,z1), ground);
            boxes1.add(static_cast<hittable *>(box3));
        }
    }

    hittable_list world{7};

    auto bvh_node_boxes1 = new bvh_node{boxes1};
    world.add(bvh_node_boxes1);

    auto light = new diffuse_light(color(7, 7, 7));
    auto quad_light_1 = new quad(point3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), light);
    world.add(quad_light_1);

    auto dielectric_sphere = new dielectric(1.5);
    auto dielectric_sphere_1 = new sphere(point3(260, 150, 45), 50,dielectric_sphere);
    world.add(dielectric_sphere_1);

    auto metal_sphere = new metal(color(0.8, 0.8, 0.9), 1.0);
    auto metal_sphere_1 = new sphere(point3(0, 150, 145), 50, metal_sphere);
    world.add(metal_sphere_1);

    auto dielectric_ground = new dielectric(1.5);
    auto dielectric_ground_1 = new sphere(point3(360, 150, 145), 70, dielectric_ground);
    world.add(dielectric_ground_1);

    auto image_ptr = new image_loader{"earthmap.jpg"};
    auto image_texture_emat = new image_texture(image_ptr->get_record());
    auto lambertian_emat = new lambertian(image_texture_emat);
    auto lambertian_emat_sphere_1 = new sphere(point3(400, 200, 400), 100, lambertian_emat);
    world.add(lambertian_emat_sphere_1);

    int ns = 1000;
    hittable_list boxes2{ns};
    auto white = new lambertian(color(.73, .73, .73));
    for (int j = 0; j < ns; j++) {
        auto boxes2_sphere = new sphere(random_in_cube(0, 165, nullptr), 10, white);
        boxes2.add(boxes2_sphere);
    }

    auto bvh_node_box = new bvh_node(boxes2);
    auto bvh_node_box_rotate_y = new rotate_y(bvh_node_box, 15);
    auto bvh_node_box_translate = new translate(bvh_node_box_rotate_y, vec3(-100,270,395));
    world.add(bvh_node_box_translate);
    return world;
}

__host__ __device__ hittable_list debug_scene_build(curandState* rnd) {
    hittable_list boxes1{4};
    auto ground = new lambertian(color(0.48, 0.83, 0.53));
    hittable_list world{3};

    int boxes_per_side = 2;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 800.0;
            auto x0 = -1000.0 + i*w;
            auto z0 = -1000.0 + j*w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            #ifdef __CUDA_ARCH__
            auto y1 = random_float(1,101, rnd);
            #else
            auto y1 = get_rnd(i*boxes_per_side+j)*100+1;
            #endif

            auto z1 = z0 + w;

            auto box3 = create_box(point3(x0,y0,z0), point3(x1,y1,z1), ground);
            boxes1.add(static_cast<hittable *>(box3));
        }
    }

    auto bvh_node_boxes1 = new bvh_node{boxes1};
    world.add(bvh_node_boxes1);

    auto light = new diffuse_light(color(7, 7, 7));
    world.add(new sphere{point3{123, 554, 147}, 100, light});

    auto dielectric_sphere = new dielectric(1.5);
    auto dielectric_sphere_1 = new sphere(point3(260, 150, 45), 50,dielectric_sphere);
    world.add(dielectric_sphere_1);

    hittable_list tree{1};
    tree.add(new bvh_node{world});
    return tree;
}

__host__ __device__ hittable_list debug_scene_build(curandState* rnd, image_record* image_rd) {
    hittable_list boxes1{64};
    auto ground = new lambertian(color(0.48, 0.83, 0.53));
    hittable_list world{5};

    int boxes_per_side = 8;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 200.0f;
            auto x0 = -1000.0f + i*w;
            auto z0 = -1000.0f + j*w;
            auto y0 = 0.0f;
            auto x1 = x0 + w-0.1f;
            // Get near-identical scene between CPU and CUDA
            #ifdef __CUDA_ARCH__
            auto y1 = random_float(1,101, rnd);
            #else
            auto y1 = get_rnd(i*boxes_per_side+j)*100+1;
            #endif
            auto z1 = z0 + w-0.1f;

            auto box3 = create_box(point3(x0,y0,z0), point3(x1,y1,z1), ground);
            boxes1.add(static_cast<hittable *>(box3));
        }
    }

    auto image_texture_emat = new image_texture(image_rd[0]);
    auto lambertian_emat = new lambertian(image_texture_emat);
    auto lambertian_emat_sphere_1 = new sphere(point3(400, 200, 400), 100, lambertian_emat);
    world.add(lambertian_emat_sphere_1);
    auto bvh_node_boxes1 = new bvh_node{boxes1};
    world.add(bvh_node_boxes1);

    auto light = new diffuse_light(color(7, 7, 7));
    world.add(new sphere{point3{123, 554, 147}, 100, light});

    auto dielectric_sphere = new dielectric(1.5);
    auto dielectric_sphere_1 = new sphere(point3(260, 150, 45), 50,dielectric_sphere);
    world.add(dielectric_sphere_1);

    int ns = 10;
    auto boxes2 = new hittable_list{ns};
    auto white = new lambertian(color(.73, .73, .73));
    for (int j = 0; j < ns; j++) {
        auto boxes2_sphere = new sphere(random_in_cube(0, 165, rnd), 10, white);
        boxes2->add(boxes2_sphere);
    }

    // FIXME: bvh_node(hittable_list) will cause stack overflow in CUDA
    auto bvh_node_box = new bvh_node(*boxes2);
    auto bvh_node_box_rotate_y = new rotate_y(bvh_node_box, 15);
    auto bvh_node_box_translate = new translate(bvh_node_box_rotate_y, vec3(-100,270,395));

    world.add(bvh_node_box_translate);

    hittable_list tree{1};
    tree.add(new bvh_node{world});
    return tree;
}

__global__ void debug_scene_build_cuda(hittable_list** world_ptr, curandState* states, image_record* image_rd) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid==0) {
        *world_ptr = new hittable_list{1};
        //**world_ptr = debug_scene_build(states,image_rd);
        **world_ptr = debug_scene_build(states);
    }
}

__host__ __device__ camera final_camera(int image_width, int samples_per_pixel, int max_depth) {
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

void render_scene(const hittable_list &scene, camera &cam) {
    cam.render(scene);
}

void render_thread(camera &cam, const hittable_list &scene, std::span<unsigned int> image) {
    while (!mainRendererComm.stop_render.load()) {
        if (mainRendererComm.frame_start_render.try_acquire()) {
            cam.render_parallel(scene, image);
            mainRendererComm.frame_rendered.release();
        }
        std::this_thread::yield();
    }
}

__global__ void camera_init_cuda(camera* cam) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid==0) {
        cam->initialize();
    }
}

__global__ void camera_render_cuda(camera* cam, hittable_list** scenepptr, std::span<unsigned int> image, curandState* devStates) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    // only use 1, 2, 4, ..., 32
    constexpr auto threadPerPixel = 32;
    static_assert(threadPerPixel<=BLOCKDIM_X);
    auto devState = devStates[tid];
    auto gridSize = blockDim.x * gridDim.x;
    auto width = cam->image_width;
    auto scene = *scenepptr;
    for (unsigned int pixelId = tid/threadPerPixel; pixelId < image.size(); pixelId+=gridSize/threadPerPixel) {
        auto pixel_result = cam->render_pixel<threadPerPixel>(scene, pixelId/width, pixelId%width, &devState, tid%threadPerPixel);
        if (tid%threadPerPixel == 0) {
            image[pixelId] = pixel_result;
        }
    }
    devStates[tid] = devState;
}

void render_thread_cuda(const camera& cam, camera* cam_cuda, hittable_list** scene_cuda, std::span<unsigned int> image, curandState* devStates) {
    int height = int(cam.image_width/cam.aspect_ratio);
    int width  = cam.image_width;
    unsigned int* imageGpuPtr{};
    cudaMalloc(&imageGpuPtr, image.size()*sizeof(unsigned int));
    std::span<unsigned int> imageGpu{imageGpuPtr, static_cast<std::span<unsigned>::size_type>(height*width)};
    while (!mainRendererComm.stop_render.load()) {
        if (mainRendererComm.frame_start_render.try_acquire()) {
            camera_init_cuda<<<1,1>>>(cam_cuda);
            camera_render_cuda<<<GRIDDIM_X,BLOCKDIM_X>>>(cam_cuda, scene_cuda, imageGpu, devStates);
            utils::cu_ensure(cudaMemcpy(image.data(), imageGpuPtr, image.size()*sizeof(unsigned int), cudaMemcpyDeviceToHost));
            mainRendererComm.frame_rendered.release();
        }
        std::this_thread::yield();
    }
    cudaFree(imageGpuPtr);
}

void render_scene_realtime(hittable_list &scene, camera &cam) {
    int height = int(cam.image_width/cam.aspect_ratio);
    auto window = sdl_raii::Window{"RT_project", cam.image_width, height};
    auto renderer = sdl_raii::Renderer{window.get()};
    auto surface = sdl_raii::Surface{cam.image_width, height};
    auto image = std::span{static_cast<unsigned int *>(surface.get()->pixels), static_cast<size_t>(cam.image_width)*height};
    std::promise<void> render_finished;
    std::future<void> render_finished_future = render_finished.get_future();
    std::thread{[=, &render_finished, &cam, &scene] {
        render_thread(cam, scene, image);
        render_finished.set_value_at_thread_exit();
    }}.detach();
    mainRendererComm.frame_start_render.release();
    size_t frames = 0;
    std::chrono::microseconds frame_times{};
    auto t0 = std::chrono::steady_clock::now();
    while (!want_exit_sdl())
    {
        if (mainRendererComm.frame_rendered.try_acquire_for(std::chrono::milliseconds{5})) {
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
    }
    utils::log("Total frames: "+std::to_string(frames)+
               ", avg. frame time: "+std::to_string(frame_times.count()/frames/1e3)+" ms.");
    notify_renderer_exit();
    while(render_finished_future.wait_for(std::chrono::milliseconds{5})==std::future_status::timeout) {
        want_exit_sdl();
    }
    render_finished_future.wait();
}

void render_scene_realtime_cuda(hittable_list** scene, camera &cam, camera *cam_cuda, curandState* devStates, const int& max_frame) {
    int height = int(cam.image_width/cam.aspect_ratio);
    auto window = sdl_raii::Window{"RT_project", cam.image_width, height};
    auto renderer = sdl_raii::Renderer{window.get()};
    auto surface = sdl_raii::Surface{cam.image_width, height};
    auto image = std::span{static_cast<unsigned int *>(surface.get()->pixels), static_cast<size_t>(cam.image_width)*height};
    std::promise<void> render_finished;
    std::future<void> render_finished_future = render_finished.get_future();
    std::thread{[=, &render_finished, &cam, &scene] {
        render_thread_cuda(cam, cam_cuda, scene, image, devStates);
        render_finished.set_value_at_thread_exit();
    }}.detach();
    mainRendererComm.frame_start_render.release();
    size_t frames = 0;
    std::chrono::microseconds frame_times{};
    auto t0 = std::chrono::steady_clock::now();
    while (!want_exit_sdl() && ((frames < max_frame) || (max_frame < 0)))
    {
        if (mainRendererComm.frame_rendered.try_acquire_for(std::chrono::milliseconds{5})) {
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
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

void parse_arguments(int argc, char** argv, int& size, int& samples, int& depth, int& frame) {
    for (int i = 1; i < argc; i += 2) {
        if (std::string(argv[i]) == "--size") {
            size = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--samples") {
            samples = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--depth") {
            depth = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--frame") {
            frame = std::atoi(argv[i + 1]);
        } else {
            std::cerr << "Usage: " << argv[0]
                << " --size <int> --depth <int> --samples <int> --device <string>\n\n"
                   "       --size: width of image in px\n"
                   "       --depth: maximum depth for rays\n"
                   "       --samples: number of samples per pixel\n"
                   "       --frame: non-stop when set to negative\n" << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    sdl_raii::SDL sdl{};
    initialize_main_sync_objs();

    int size = 400, samples = 32, depth = 3, frame = 62;
    parse_arguments(argc, argv, size, samples, depth, frame);

    auto image_ld = image_loader("earthmap.jpg");
    auto rec_cuda = image_ld.get_record_cuda();
    auto rec = image_ld.get_record();
    utils::CuArrayRAII image_rd{&rec_cuda};

    utils::CuArrayRAII<curandState> devStates{nullptr, GRIDDIM_X*BLOCKDIM_X};
    // Cherry-picked seed
    initCurand<<<GRIDDIM_X,BLOCKDIM_X>>>(devStates.cudaPtr, 5);
    utils::CuArrayRAII<hittable_list*> sceneGpuPtr{nullptr};

    auto cam = final_camera(size, samples, depth);
    utils::CuArrayRAII camGpuPtr{&cam};

    // auto scene = debug_scene_build(nullptr,&rec);
    // //auto scene = final_scene_build();
    // render_scene_realtime(scene, cam);
    debug_scene_build_cuda<<<1,1>>>(sceneGpuPtr.cudaPtr, devStates.cudaPtr, image_rd.cudaPtr);
    cudaDeviceSynchronize();
    utils::cu_check();
    render_scene_realtime_cuda(sceneGpuPtr.cudaPtr, cam, camGpuPtr.cudaPtr, devStates.cudaPtr, frame);

    // we don't do the cleanup yet, but this will make compute-sanitizer unhappy
    //cudaDeviceReset();
    return 0;
}
