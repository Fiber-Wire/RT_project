cmake_minimum_required ( VERSION 3.28 )
include(cmake/AddFile.cmake)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_ARCHITECTURES "50;52;53;60;61;62;70;72;75;80;86;87;89")
else()
    set(CMAKE_CUDA_ARCHITECTURES "75;89")
endif()
endif()
if (NOT DEFINED CUDA_MAXREG)
    set(CUDA_MAXREG 64)
endif ()
project(RT_project LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set ( SOURCE_RT_PROJECT
        main.cu
        image_loader.cu
        .manifest
)

find_package(CUDAToolkit COMPONENTS curand REQUIRED)
find_package(SDL2 CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(glm CONFIG REQUIRED)
add_library(commons_headers INTERFACE)
target_include_directories(commons_headers INTERFACE include)

add_executable(RT_project ${SOURCE_RT_PROJECT})
# Why does LTO make performance worse?
set_property(TARGET RT_project PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
target_include_directories(RT_project PRIVATE ${Stb_INCLUDE_DIR})
target_link_libraries(RT_project PRIVATE CUDA::curand)
target_link_libraries(RT_project PRIVATE glm::glm)
target_link_libraries(RT_project PRIVATE commons_headers)
target_link_libraries(commons_headers
        INTERFACE
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

add_custom_file(RT_project images/earthmap.jpg images/earthmap.jpg)
add_custom_file(RT_project test_script.py test_script.py)

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
if (CMAKE_CUDA_COMPILER)
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(RT_project PRIVATE --extended-lambda --expt-relaxed-constexpr --use_fast_math -maxrregcount ${CUDA_MAXREG})
    else()
        target_compile_options(RT_project PRIVATE
                --extended-lambda --expt-relaxed-constexpr --generate-line-info --use_fast_math -maxrregcount ${CUDA_MAXREG})
    endif()

endif ()
