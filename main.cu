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

    auto red   = new lambertian(color(.65, .05, .05));
    auto white = new lambertian(color(.73, .73, .73));
    auto green = new lambertian(color(.12, .45, .15));
    auto light = new diffuse_light(color(15, 15, 15));
    objects.push_back(red);
    objects.push_back(white);
    objects.push_back(green);
    objects.push_back(light);
    auto quad_1 = new quad(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), green);
    auto quad_2 = new quad(point3(0,0,0), vec3(0,555,0), vec3(0,0,555), red);
    auto quad_3 = new quad(point3(343, 554, 332), vec3(-130,0,0), vec3(0,0,-105), light);
    auto quad_4 = new quad(point3(0,0,0), vec3(555,0,0), vec3(0,0,555), white);
    auto quad_5 = new quad(point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), white);
    auto quad_6 = new quad(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), white);

    world.add(quad_1);
    world.add(quad_2);
    world.add(quad_3);
    world.add(quad_4);
    world.add(quad_5);
    world.add(quad_6);

    hittable_list* box1 = create_box(point3(0,0,0), point3(165,330,165), white);
    auto box1_rotate = new rotate_y(box1,15);
    auto box1_translate = new translate(box1_rotate, vec3(265,0,295));
    world.add(box1_translate);

    hittable_list* box2 = create_box(point3(0,0,0), point3(165,165,165), white);
    auto box2_rotate = new rotate_y(box2, -18);
    auto box2_translate = new translate(box2_rotate, vec3(130,0,65));
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
            auto y1 = random_float(1,101);
            auto z1 = z0 + w;

            auto box3 = create_box(point3(x0,y0,z0), point3(x1,y1,z1), ground);
            boxes1.add(box3);
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

    auto image_texture_emat = new image_texture("earthmap.jpg") ;
    auto lambertian_emat = new lambertian(image_texture_emat);
    auto lambertian_emat_sphere_1 = new sphere(point3(400, 200, 400), 100, lambertian_emat);
    world.add(lambertian_emat_sphere_1);

    int ns = 1000;
    hittable_list boxes2{ns};
    auto white = new lambertian(color(.73, .73, .73));
    for (int j = 0; j < ns; j++) {
        auto boxes2_sphere = new sphere(random_in_cube(0, 165), 10, white);
        boxes2.add(boxes2_sphere);
    }

    auto bvh_node_box = new bvh_node(boxes2);
    auto bvh_node_box_rotate_y = new rotate_y(bvh_node_box, 15);
    auto bvh_node_box_translate = new translate(bvh_node_box_rotate_y, vec3(-100,270,395));
    world.add(bvh_node_box_translate);
    return world;
}

hittable_list debug_scene_build() {
    hittable_list boxes1{4};
    auto ground = new lambertian(color(0.48, 0.83, 0.53));

    int boxes_per_side = 2;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 800.0;
            auto x0 = -1000.0 + i*w;
            auto z0 = -1000.0 + j*w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_float(1,101);
            auto z1 = z0 + w;

            auto box3 = create_box(point3(x0,y0,z0), point3(x1,y1,z1), ground);
            boxes1.add(box3);
        }
    }

    hittable_list world{3};

    auto bvh_node_boxes1 = new bvh_node{boxes1};
    world.add(bvh_node_boxes1);

    auto light = new diffuse_light(color(7, 7, 7));
    world.add(new sphere{point3{123, 554, 147}, 100, light});

    auto dielectric_sphere = new dielectric(1.5);
    auto dielectric_sphere_1 = new sphere(point3(260, 150, 45), 50,dielectric_sphere);
    world.add(dielectric_sphere_1);

    return world;
}

camera final_camera(int image_width, int samples_per_pixel, int max_depth) {
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
        if (mainRendererComm.frame_start_render.try_acquire_for(std::chrono::milliseconds(5))) {
            cam.render_parallel(scene, image);
            mainRendererComm.frame_rendered.release();
        }
    }
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
    auto t0 = std::chrono::steady_clock::now();
    while (!want_exit_sdl())
    {
        if (mainRendererComm.frame_rendered.try_acquire_for(std::chrono::milliseconds{5})) {
            auto texture = sdl_raii::Texture{renderer.get(), surface.get()};
            SDL_RenderClear(renderer.get());
            SDL_RenderCopy(renderer.get(),texture.get(), nullptr, nullptr);
            SDL_RenderPresent(renderer.get());
            auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-t0);
            utils::log("Frame time: "+std::to_string(frame_time.count()/1e3)+" ms");
            mainRendererComm.frame_start_render.release();
            t0 = std::chrono::steady_clock::now();
        }
    }
    notify_renderer_exit();
    while(render_finished_future.wait_for(std::chrono::milliseconds{5})==std::future_status::timeout) {
        want_exit_sdl();
    }
    render_finished_future.wait();
}


int main(int argc, char* argv[]) {
    sdl_raii::SDL sdl{};
    initialize_main_sync_objs();
    auto scene = final_scene_build();
    auto cam = final_camera(400, 50, 4);
    if (argc!=1) {
        render_scene(scene, cam);
    } else {
        render_scene_realtime(scene, cam);
    }
    return 0;
}
