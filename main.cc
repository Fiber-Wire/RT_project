//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include <future>
#include <semaphore>
#include <string>
#include <thread>
#include "sdl_wrapper.hpp"
#include "rtweekend.h"

#include "bvh.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "quad.h"
#include "sphere.h"
#include "texture.h"

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
    hittable_list world;

    auto red   = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(15, 15, 15));

    world.add(make_shared<quad>(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), green));
    world.add(make_shared<quad>(point3(0,0,0), vec3(0,555,0), vec3(0,0,555), red));
    world.add(make_shared<quad>(point3(343, 554, 332), vec3(-130,0,0), vec3(0,0,-105), light));
    world.add(make_shared<quad>(point3(0,0,0), vec3(555,0,0), vec3(0,0,555), white));
    world.add(make_shared<quad>(point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), white));
    world.add(make_shared<quad>(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), white));

    shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, vec3(265,0,295));
    world.add(box1);

    shared_ptr<hittable> box2 = box(point3(0,0,0), point3(165,165,165), white);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, vec3(130,0,65));
    world.add(box2);

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
    hittable_list boxes1;
    auto ground = std::make_shared<lambertian>(color(0.48, 0.83, 0.53));

    int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i*w;
            auto z0 = -1000.0 + j*w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_double(1,101);
            auto z1 = z0 + w;

            boxes1.add(box(point3(x0,y0,z0), point3(x1,y1,z1), ground));
        }
    }

    hittable_list world;

    world.add(std::make_shared<bvh_node>(boxes1));

    auto light = std::make_shared<diffuse_light>(color(7, 7, 7));
    world.add(std::make_shared<quad>(point3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), light));

    world.add(std::make_shared<sphere>(point3(260, 150, 45), 50, std::make_shared<dielectric>(1.5)));
    world.add(std::make_shared<sphere>(
        point3(0, 150, 145), 50, std::make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)
    ));

    auto boundary = std::make_shared<sphere>(point3(360, 150, 145), 70, std::make_shared<dielectric>(1.5));
    world.add(boundary);

    auto emat = std::make_shared<lambertian>(std::make_shared<image_texture>("earthmap.jpg"));
    world.add(std::make_shared<sphere>(point3(400, 200, 400), 100, emat));

    hittable_list boxes2;
    auto white = std::make_shared<lambertian>(color(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2.add(std::make_shared<sphere>(point3::random(0, 165), 10, white));
    }

    world.add(std::make_shared<translate>(
        std::make_shared<rotate_y>(
                std::make_shared<bvh_node>(boxes2), 15),
            vec3(-100,270,395)
        )
    );
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

void render_scene(hittable_list &scene, camera &cam) {
    cam.render(scene);
}

void render_thread(camera &cam, const hittable_list &scene, std::span<unsigned int> image) {
    while (!mainRendererComm.stop_render.load()) {
        mainRendererComm.frame_start_render.acquire();
        cam.render(scene, image);
        mainRendererComm.frame_rendered.release();
    }
}

void render_scene_realtime(hittable_list &scene, camera &cam) {
    int height = int(cam.image_width/cam.aspect_ratio);
    auto window = sdl_raii::Window{"theNextWeek", cam.image_width, height};
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
