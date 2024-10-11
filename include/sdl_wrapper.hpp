//
// Created by JCW on 9/10/2024.
//

#ifndef RT_PROJECT_SDL_WRAPPER_HPP
#define RT_PROJECT_SDL_WRAPPER_HPP
#define WIN32_LEAN_AND_MEAN
#include "SDL2/SDL.h"
#include "utils.hpp"
namespace sdl_raii {
    class Texture: private utils::NonCopyable<Texture> {
    public:
        Texture(SDL_Renderer* renderer, SDL_Surface* surface) {
            texture_ = SDL_CreateTextureFromSurface(renderer, surface);
        }
        ~Texture() {
            SDL_DestroyTexture(texture_);
            texture_ = nullptr;
        }
        SDL_Texture* get() {
            return texture_;
        }
    private:
        SDL_Texture* texture_{};
    };
    class Surface: private utils::NonCopyable<Surface> {
    public:
        Surface(int width, int height) {
            surface_ = SDL_CreateRGBSurface(0, width, height, 32, 0, 0, 0, 0);
        }
        ~Surface() {
            SDL_FreeSurface(surface_);
            surface_ = nullptr;
        }
        SDL_Surface* get() {
            return surface_;
        }
    private:
        SDL_Surface* surface_{};
    };
    class Renderer: private utils::NonCopyable<Renderer> {
    public:
        Renderer(SDL_Window* window) {
            renderer_ = SDL_CreateRenderer(window, -1, 0);
        }
        ~Renderer() {
            SDL_DestroyRenderer(renderer_);
            renderer_ = nullptr;
        }
        SDL_Renderer* get() {
            return renderer_;
        }
    private:
        SDL_Renderer* renderer_{};
    };
    class Window: private utils::NonCopyable<Window> {
    public:
        Window(std::string_view title, int width, int height) {
            window_ = SDL_CreateWindow(
                    std::string{title}.c_str(),
                    SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                    width, height,
                    SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI);// | SDL_WINDOW_FULLSCREEN);
            SDL_SetWindowResizable(window_, SDL_FALSE);
        }
        ~Window() {
            SDL_DestroyWindow(window_);
            window_ = nullptr;
        }
        SDL_Window* get() {
            return window_;
        }
    private:
        SDL_Window* window_{};
    };
    class SDL: private utils::NonCopyable<SDL> {
    public:
        SDL() {
            SDL_SetMainReady();
            //SDL_Vulkan_LoadLibrary(nullptr);
            SDL_SetHint(SDL_HINT_VIDEO_HIGHDPI_DISABLED, "0");
            SDL_Init(SDL_INIT_VIDEO);
        }
        ~SDL() {
            //SDL_Vulkan_UnloadLibrary();
            SDL_Quit();
        }
    };
}

SDL_Window *init_sdl(int width, int height) {
    SDL_SetHint(SDL_HINT_VIDEO_HIGHDPI_DISABLED, "0");
    SDL_Init(SDL_INIT_VIDEO);
    //
    //SDL_Vulkan_LoadLibrary(nullptr);
    auto window = SDL_CreateWindow(
            "RT_project",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            width, height,
            SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI);// | SDL_WINDOW_FULLSCREEN);
    SDL_SetWindowResizable(window, SDL_FALSE);

    return window;
}

void clean_sdl(SDL_Window *&window) {
    SDL_DestroyWindow(window);
    window = nullptr;
    //SDL_Vulkan_UnloadLibrary();
    SDL_Quit();
}

bool want_exit_sdl() {
    SDL_Event sdlEvent;
    static bool bRet = false;

    while ((SDL_PollEvent(&sdlEvent) != 0) & (!bRet)) {
        if (sdlEvent.type == SDL_QUIT) {
            bRet = true;
        } else if (sdlEvent.type == SDL_KEYDOWN) {
            if (sdlEvent.key.keysym.sym == SDLK_ESCAPE
                || sdlEvent.key.keysym.sym == SDLK_q) {
                bRet = true;
            }
            if (sdlEvent.key.keysym.sym == SDLK_RETURN) {
                if (sdlEvent.key.keysym.mod & KMOD_ALT) {
                    // TODO: get full-screen mode working.
                    utils::log_and_pause("Full-screen mode alteration requested.", 0);
                }
            }
        }
    }
    return bRet;
}
#endif //RT_PROJECT_SDL_WRAPPER_HPP
