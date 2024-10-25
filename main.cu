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

void cornell_box() {
    hittable_list world{1000};
    std::vector<material*> objects;
    auto quads_for_box = new utils::NaiveVector<quad>{12};

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

    hittable_list* box1 = create_box(point3(0,0,0), point3(165,330,165), white,quads_for_box);
    const auto box1_rotate = new rotate_y(box1,15);
    const auto box1_translate = new translate(box1_rotate, vec3(265,0,295));
    world.add(box1_translate);

    hittable_list* box2 = create_box(point3(0,0,0), point3(165,165,165), white, quads_for_box);
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

    const auto world_bvh = bvh_node(world);

    cam.render(world_bvh);
}

__host__ __device__ bvh_node* final_scene_build(curandState* rnd, const image_record* image_rd) {
    hittable_list world{8};

    // Geometry primitives
    auto spheres = new utils::NaiveVector<sphere>{1005};
    auto quads = new utils::NaiveVector<quad>{2401};

    // Materials
    //   Textures needed by materials;
    auto image_texture_emat = new image_texture(image_rd[0]);
    auto material_handles = new material*[8];
#ifdef __CUDA_ARCH__
    CUDA_MATERIALS = material_handles;
#endif
    material_handles[0] = new lambertian(color(0.48, 0.83, 0.53));
    material_handles[1] = new diffuse_light(color(7, 7, 7));
    material_handles[2] = new dielectric(1.5);
    material_handles[3] = new metal(color(0.8, 0.8, 0.9), 1.0);
    material_handles[4] = new dielectric(1.5);
    material_handles[5] = new lambertian(image_texture_emat);
    material_handles[6] = new lambertian(color(.73, .73, .73));
    material_handles[7] = new metal(color(212.0f/256, 175.0f/256, 55.0f/256), 0.025);


    // 1st item
    auto ground = material_handles[0];
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
            // compute-sanitizer does not like what we do here
#ifdef __CUDA_ARCH__
            auto y1 = random_float(1,101, rnd);
#else
            auto y1 = get_rnd(i*boxes_per_side+j)*100+1;
#endif
            auto z1 = z0 + w-0.1f;

            auto box3 = create_box(point3(x0,y0,z0), point3(x1,y1,z1), ground, quads);
            boxes1.add(static_cast<hittable *>(box3));
        }
    }
    auto bvh_node_boxes1 = new bvh_node{boxes1};
    world.add(bvh_node_boxes1);

    // 2nd
    auto light = material_handles[1];
    quads->push({point3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), light});
    world.add(quads->end());

    // 3rd
    auto dielectric_sphere = material_handles[2];
    spheres->push({point3(260, 150, 45), 50,dielectric_sphere});
    world.add(spheres->end());

    //4th
    auto metal_sphere = material_handles[3];
    spheres->push({point3(0, 150, 145), 50, metal_sphere});
    world.add(spheres->end());

    //5th
    auto dielectric_ground = material_handles[4];
    spheres->push({point3(360, 150, 145), 70, dielectric_ground});
    world.add(spheres->end());

    //6th
    auto lambertian_emat = material_handles[5];
    spheres->push({point3(400, 200, 400), 100, lambertian_emat});
    world.add(spheres->end());

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
        spheres->push({center, 10, white});
        boxes2.add(spheres->end());
    }
    auto bvh_node_box = new bvh_node(boxes2);
    auto bvh_node_box_rotate_y = new rotate_y(bvh_node_box, 15);
    auto bvh_node_box_translate = new translate(bvh_node_box_rotate_y, vec3(-100,270,395));
    world.add(bvh_node_box_translate);

    //8th
    auto metal_2 = material_handles[7];
    spheres->push({point3(240, 320, 400), 60, metal_2});
    world.add(spheres->end());

    auto tree = new bvh_node{world};
    return tree;
}

__global__ void final_scene_build_cuda(bvh_node** world_ptr, curandState* states, const image_record* image_rd) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid==0) {
        *world_ptr = final_scene_build(states,image_rd);
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

void render_scene(const bvh_node &scene, camera &cam) {
    cam.render(scene);
}

void render_thread(camera &cam, const bvh_node &scene, const std::span<unsigned int> image) {
    while (!mainRendererComm.stop_render.load()) {
        if (mainRendererComm.frame_start_render.try_acquire()) {
            const auto r = cam.lookfrom-cam.lookat;
            const auto tan_v = -normalize(cross(r, cam.vup));
            constexpr auto dtheta = 0.1f;
            cam.lookfrom += tan_v*length(r)*sinf(dtheta)-r*(1.0f-cosf(dtheta));
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
        const auto r = cam->lookfrom-cam->lookat;
        const auto tan_v = -normalize(cross(r, cam->vup));
        constexpr auto dtheta = 0.1f;
        cam->lookfrom += tan_v*length(r)*sinf(dtheta)-r*(1.0f-cosf(dtheta));
    }
}

__global__ void camera_render_cuda(camera* cam, bvh_node** scenepptr, std::span<unsigned int> image, curandState* devStates) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    // only use 1, 2, 4, ..., 32
    constexpr auto threadPerPixel = 32;
    static_assert(threadPerPixel<=BLOCKDIM_X);

    auto devState = devStates[tid];
    const auto gridSize = blockDim.x * gridDim.x;
    const auto width = cam->image_width;
    const auto scene = *scenepptr;

    for (unsigned int pixelId = tid/threadPerPixel; pixelId < image.size(); pixelId+=gridSize/threadPerPixel) {
        const auto pixel_result = cam->render_pixel<threadPerPixel>(
            scene,
            pixelId/width, pixelId%width, &devState,
            tid%threadPerPixel);
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
    //utils::cu_ensure(cudaFuncSetCacheConfig(camera_render_cuda, cudaFuncCachePreferL1));
    const std::span<unsigned int> imageGpu{imageGpuPtr, static_cast<std::span<unsigned>::size_type>(height*width)};
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

void render_scene_realtime(bvh_node &scene, camera &cam, const int &max_frame) {
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

void render_scene_realtime_cuda(bvh_node** scene, camera &cam, camera *cam_cuda, curandState* devStates, const int& max_frame) {
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
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

void parse_arguments(int argc, char** argv, int& size, int& samples, int& depth, std::string& device, int& frame) {
    for (int i = 1; i < argc; i += 2) {
        if (std::string(argv[i]) == "--size") {
            size = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--samples") {
            samples = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--depth") {
            depth = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--device") {
            device = argv[i + 1];
        }else if (std::string(argv[i]) == "--frame") {
            frame = std::atoi(argv[i + 1]);
        } else {
            std::cerr << "Usage: " << argv[0]
                << " --size <int> --depth <int> --samples <int> --device <string>\n\n"
                   "       --size: width of image in px\n"
                   "       --depth: maximum depth for rays\n"
                   "       --samples: number of samples per pixel\n"
                   "       --device: device to use (cpu, gpu, default)\n"
                   "       --frame: non-stop when set to negative\n" << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    sdl_raii::SDL sdl{};
    initialize_main_sync_objs();

    int size = 400, samples = 32, depth = 4, frame = 62;
    std::string device = "gpu";
    parse_arguments(argc, argv, size, samples, depth, device, frame);
    GRIDDIM_X = size*size/(BLOCKDIM_X/32);

    auto image_ld = image_loader("earthmap.jpg");
    const auto rec = image_ld.get_record();
    auto cam = final_camera(size, samples, depth);

    if (device == "default") {
        const auto scene = final_scene_build(nullptr,&rec);
        render_scene(*scene, cam);
    } else {
        if (device == "cpu") {
            auto scene = final_scene_build(nullptr,&rec);
            render_scene_realtime(*scene, cam, frame);
        } else {
            auto rec_cuda = image_ld.get_record_cuda();
            const utils::CuArrayRAII image_rd{&rec_cuda};
            const utils::CuArrayRAII<curandState> devStates{nullptr, static_cast<size_t>(GRIDDIM_X*BLOCKDIM_X)};
            // Cherry-picked seed
            initCurand<<<GRIDDIM_X,BLOCKDIM_X>>>(devStates.cudaPtr, 1);
            const utils::CuArrayRAII<bvh_node*> sceneGpuPtr{nullptr};
            const utils::CuArrayRAII camGpuPtr{&cam};
            final_scene_build_cuda<<<1,1>>>(sceneGpuPtr.cudaPtr, devStates.cudaPtr, image_rd.cudaPtr);
            cudaDeviceSynchronize();
            utils::cu_check();
            render_scene_realtime_cuda(sceneGpuPtr.cudaPtr, cam, camGpuPtr.cudaPtr, devStates.cudaPtr, frame);
        }
    }
    // we don't do the cleanup yet, but this will make compute-sanitizer unhappy
    //cudaDeviceReset();
    return 0;
}
