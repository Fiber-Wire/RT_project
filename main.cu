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
#define GRIDDIM_X (400*400/BLOCKDIM_X)
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

void cornell_box() {
    hittable_list world{1000};
    std::vector<material*> objects;

    const auto red   = new lambertian(color(.65, .05, .05));
    const auto white = new lambertian(color(.73, .73, .73));
    const auto green = new lambertian(color(.12, .45, .15));
    const auto light = new diffuse_light(color(15, 15, 15));
    objects.push_back(red);
    objects.push_back(white);
    objects.push_back(green);
    objects.push_back(light);
    const auto quad_1 = new quad(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), green);
    const auto quad_2 = new quad(point3(0,0,0), vec3(0,555,0), vec3(0,0,555), red);
    const auto quad_3 = new quad(point3(343, 554, 332), vec3(-130,0,0), vec3(0,0,-105), light);
    const auto quad_4 = new quad(point3(0,0,0), vec3(555,0,0), vec3(0,0,555), white);
    const auto quad_5 = new quad(point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), white);
    const auto quad_6 = new quad(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), white);

    world.add(quad_1);
    world.add(quad_2);
    world.add(quad_3);
    world.add(quad_4);
    world.add(quad_5);
    world.add(quad_6);

    hittable_list* box1 = create_box(point3(0,0,0), point3(165,330,165), white);
    const auto box1_rotate = new rotate_y(box1,15);
    const auto box1_translate = new translate(box1_rotate, vec3(265,0,295));
    world.add(box1_translate);

    hittable_list* box2 = create_box(point3(0,0,0), point3(165,165,165), white);
    const auto box2_rotate = new rotate_y(box2, -18);
    const auto box2_translate = new translate(box2_rotate, vec3(130,0,65));
    world.add(box2_translate);

    camera cam;

    cam.aspect_ratio      = 1.0;
    cam.image_width       = 600;
    cam.samples_per_pixel = 200;
    cam.max_depth         = 50;
    cam.background        = color(0,0,0);

    cam.vfov     = 40;
    cam.lookfrom = point3(278, 278, -800);
    cam.lookat   = point3(278, 278, 0);
    cam.vup      = vec3(0,1,0);

    cam.render(world);
}

__host__ __device__ hittable_list final_scene_build(curandState* rnd, const image_record* image_rd) {
    hittable_list world{8};

    // 1st item
    auto ground = new lambertian(color(0.48, 0.83, 0.53));
    int boxes_per_side = 20;
    hittable_list boxes1{boxes_per_side*boxes_per_side};
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0f;
            auto x0 = -1000.0f + i*w;
            auto z0 = -1000.0f + j*w;
            auto y0 = 0.0f;
            auto x1 = x0 + w-0.1f;
            // Get identical scene between runs
            auto y1 = get_rnd(i*boxes_per_side+j)*100+1;
            auto z1 = z0 + w-0.1f;

            auto box3 = create_box(point3(x0,y0,z0), point3(x1,y1,z1), ground);
            boxes1.add(static_cast<hittable *>(box3));
        }
    }
    auto bvh_node_boxes1 = new bvh_node{boxes1};
    world.add(bvh_node_boxes1);

    // 2nd
    auto light = new diffuse_light(color(7, 7, 7));
    auto quad_light_1 = new quad(point3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), light);
    world.add(quad_light_1);

    // 3rd
    auto dielectric_sphere = new dielectric(1.5);
    auto dielectric_sphere_1 = new sphere(point3(260, 150, 45), 50,dielectric_sphere);
    world.add(dielectric_sphere_1);

    //4th
    auto metal_sphere = new metal(color(0.8, 0.8, 0.9), 1.0);
    auto metal_sphere_1 = new sphere(point3(0, 150, 145), 50, metal_sphere);
    world.add(metal_sphere_1);

    //5th
    auto dielectric_ground = new dielectric(1.5);
    auto dielectric_ground_1 = new sphere(point3(360, 150, 145), 70, dielectric_ground);
    world.add(dielectric_ground_1);

    //6th
    auto image_texture_emat = new image_texture(image_rd[0]);
    auto lambertian_emat = new lambertian(image_texture_emat);
    auto lambertian_emat_sphere_1 = new sphere(point3(400, 200, 400), 100, lambertian_emat);
    world.add(lambertian_emat_sphere_1);

    //7th
    int ns = 1000;
    auto boxes2 = hittable_list{ns};
    auto white = new lambertian(color(.73, .73, .73));
    for (int j = 0; j < ns; j++) {
        auto center = get_rand_vec3(j);
        auto boxes2_sphere = new sphere(center, 10, white);
        boxes2.add(boxes2_sphere);
    }
    auto bvh_node_box = new bvh_node(boxes2);
    auto bvh_node_box_rotate_y = new rotate_y(bvh_node_box, 15);
    auto bvh_node_box_translate = new translate(bvh_node_box_rotate_y, vec3(-100,270,395));
    world.add(bvh_node_box_translate);

    //8th
    auto metal_2 = new metal(color(212.0f/256, 175.0f/256, 55.0f/256), 0.025);
    auto metal_sphere_2 = new sphere(point3(240, 320, 400), 60, metal_2);
    world.add(metal_sphere_2);

    hittable_list tree{1};
    tree.add(new bvh_node{world});
    return tree;
}

__global__ void final_scene_build_cuda(bvh_node** world_ptr, curandState* states, image_record* image_rd) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid==0) {
        const auto temp = new hittable_list{1};
        *temp = final_scene_build(states,image_rd);
        *world_ptr = static_cast<bvh_node *>(temp->get_objects()[0]);
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

void render_scene(const hittable_list &scene, camera &cam) {
    cam.render(scene);
}

void render_thread(camera &cam, const hittable_list &scene, const std::span<unsigned int> image) {
    while (!mainRendererComm.stop_render.load()) {
        if (mainRendererComm.frame_start_render.try_acquire()) {
            cam.render_parallel(scene, image);
            mainRendererComm.frame_rendered.release();
        }
        std::this_thread::yield();
    }
}

__global__ void camera_render_cuda(camera* cam, bvh_node** scenepptr, std::span<unsigned int> image, curandState* devStates) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    // only use 1, 2, 4, ..., 32
    constexpr auto threadPerPixel = 32;
    static_assert(threadPerPixel<=BLOCKDIM_X);
    if (tid==0) {
        cam->initialize();
    }
    auto devState = devStates[tid];
    const auto gridSize = blockDim.x * gridDim.x;
    const auto width = cam->image_width;
    const auto scene = *scenepptr;
    for (unsigned int pixelId = tid/threadPerPixel; pixelId < image.size(); pixelId+=gridSize/threadPerPixel) {
        const auto pixel_result = cam->render_pixel<threadPerPixel>(scene, pixelId/width, pixelId%width, &devState, tid%threadPerPixel);
        if (tid%threadPerPixel == 0) {
            image[pixelId] = pixel_result;
        }
    }
    devStates[tid] = devState;
}

void render_thread_cuda(const camera& cam, camera* cam_cuda, bvh_node** scene_cuda, std::span<unsigned int> image, curandState* devStates) {
    const int height = static_cast<int>(cam.image_width / cam.aspect_ratio);
    const int width  = cam.image_width;
    unsigned int* imageGpuPtr{};
    cudaMalloc(&imageGpuPtr, image.size()*sizeof(unsigned int));
    const std::span<unsigned int> imageGpu{imageGpuPtr, static_cast<std::span<unsigned>::size_type>(height*width)};
    while (!mainRendererComm.stop_render.load()) {
        if (mainRendererComm.frame_start_render.try_acquire()) {
            camera_render_cuda<<<GRIDDIM_X,BLOCKDIM_X>>>(cam_cuda, scene_cuda, imageGpu, devStates);
            utils::cu_ensure(cudaMemcpy(image.data(), imageGpuPtr, image.size()*sizeof(unsigned int), cudaMemcpyDeviceToHost));
            mainRendererComm.frame_rendered.release();
        }
        std::this_thread::yield();
    }
    cudaFree(imageGpuPtr);
}

void render_scene_realtime(hittable_list &scene, camera &cam) {
    const int height = static_cast<int>(cam.image_width / cam.aspect_ratio);
    auto window = sdl_raii::Window{"RT_project", cam.image_width, height};
    auto renderer = sdl_raii::Renderer{window.get()};
    auto surface = sdl_raii::Surface{cam.image_width, height};
    const auto image = std::span{static_cast<unsigned int *>(surface.get()->pixels), static_cast<size_t>(cam.image_width)*height};
    std::promise<void> render_finished;
    const std::future<void> render_finished_future = render_finished.get_future();
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

void render_scene_realtime_cuda(bvh_node** scene, camera &cam, camera *cam_cuda, curandState* devStates) {
    const int height = static_cast<int>(cam.image_width / cam.aspect_ratio);
    auto window = sdl_raii::Window{"RT_project", cam.image_width, height};
    auto renderer = sdl_raii::Renderer{window.get()};
    auto surface = sdl_raii::Surface{cam.image_width, height};
    const auto image = std::span{static_cast<unsigned int *>(surface.get()->pixels), static_cast<size_t>(cam.image_width)*height};
    std::promise<void> render_finished;
    const std::future<void> render_finished_future = render_finished.get_future();
    std::thread{[=, &render_finished, &cam, &scene] {
        render_thread_cuda(cam, cam_cuda, scene, image, devStates);
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

__global__ void initCurand(curandState *state, unsigned long seed){
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

int main(int argc, char* argv[]) {
    sdl_raii::SDL sdl{};
    initialize_main_sync_objs();

    auto image_ld = image_loader("earthmap.jpg");
    auto rec_cuda = image_ld.get_record_cuda();
    const auto rec = image_ld.get_record();
    const utils::CuArrayRAII image_rd{&rec_cuda};

    const utils::CuArrayRAII<curandState> devStates{nullptr, GRIDDIM_X*BLOCKDIM_X};
    // Cherry-picked seed
    initCurand<<<GRIDDIM_X,BLOCKDIM_X>>>(devStates.cudaPtr, 1);
    const utils::CuArrayRAII<bvh_node*> sceneGpuPtr{nullptr};

    auto cam = final_camera(400, 32, 4);
    const utils::CuArrayRAII camGpuPtr{&cam};
    if (argc!=1) {
        const auto scene = final_scene_build(nullptr,&rec);
        render_scene(scene, cam);
    } else {
        if (false) {
            auto scene = final_scene_build(nullptr,&rec);
            render_scene_realtime(scene, cam);
        } else {
            final_scene_build_cuda<<<1,1>>>(sceneGpuPtr.cudaPtr, devStates.cudaPtr, image_rd.cudaPtr);
            cudaDeviceSynchronize();
            utils::cu_check();
            render_scene_realtime_cuda(sceneGpuPtr.cudaPtr, cam, camGpuPtr.cudaPtr, devStates.cudaPtr);
        }
    }
    // we don't do the cleanup yet, but this will make compute-sanitizer unhappy
    //cudaDeviceReset();
    return 0;
}
