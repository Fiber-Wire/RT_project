cmake_minimum_required ( VERSION 3.28 )
include(cmake/AddFile.cmake)
project(RT_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set ( SOURCE_RT_PROJECT
        main.cc
        aabb.h
        bvh.h
        camera.h
        hittable.h
        hittable_list.h
        material.h
        quad.h
        sphere.h
        texture.h
        .manifest
        color.h
        interval.h
        ray.h
        rtw_stb_image.h
        rtweekend.h
        vec3.h
)
find_package(SDL2 CONFIG REQUIRED)
find_package(Stb REQUIRED)
add_library(commons_headers INTERFACE)
target_include_directories(commons_headers INTERFACE include)


add_executable(RT_project  ${SOURCE_RT_PROJECT})
target_link_libraries(RT_project PRIVATE commons_headers)
target_link_libraries(commons_headers
        INTERFACE
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)
target_include_directories(RT_project PRIVATE ${Stb_INCLUDE_DIR})

add_custom_file(RT_project images/earthmap.jpg images/earthmap.jpg)

if (0)
    if(MSVC)
        target_compile_options(RT_project PRIVATE /fsanitize=address)
        target_link_options(RT_project PRIVATE /fsanitize=address)
        # you need to copy those files from your compiler(cl.exe)'s directory to the thirdparty subdirectory
        add_custom_file(RT_project
                thirdparty/clang_rt.asan_dbg_dynamic-x86_64.dll clang_rt.asan_dbg_dynamic-x86_64.dll)
        add_custom_file(RT_project
                thirdparty/clang_rt.asan_dbg_dynamic-x86_64.pdb clang_rt.asan_dbg_dynamic-x86_64.pdb)
        add_custom_file(RT_project
                thirdparty/clang_rt.asan_dynamic-x86_64.dll clang_rt.asan_dynamic-x86_64.dll)
        add_custom_file(RT_project
                thirdparty/clang_rt.asan_dynamic-x86_64.pdb clang_rt.asan_dynamic-x86_64.pdb)
    else ()
        target_compile_options(RT_project PRIVATE -fsanitize=address)
        target_link_options(RT_project PRIVATE -fsanitize=address)
    endif ()
    #add_custom_file(RT_project ASAN.supp ASAN.supp)
endif ()
