//
// Created by JCW on 2024/11/2.
//
#include "image_loader.cuh"

// Disable strict warnings for this header from the Microsoft Visual C++ compiler.
#ifdef _MSC_VER
    #pragma warning (push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"
#include <utils.hpp>
#include <cstdlib>

__device__ __host__ int image_record::bytes_per_scanline() const {
    return bytes_per_pixel * image_width;
}


image_loader::image_loader() {}

image_loader::image_loader(const char* image_filename) {
    // Loads image data from the specified file.
    // If the image was not loaded successfully, width() and height() will return 0.

    const auto filename = std::string(image_filename);

    if (load("images/" + filename)) return;

    utils::log(std::string("ERROR: Could not load image file '") + image_filename);
}

image_loader::~image_loader() {
    delete[] bdata;
    if (bdata_cuda != nullptr) {
        cudaFree(bdata_cuda);
        bdata_cuda = nullptr;
    }
}

bool image_loader::load(const std::string& filename) {
    // Loads the linear (gamma=1) image data from the given file name. Returns true if the
    // load succeeded. The resulting data buffer contains the three [0.0, 1.0]
    // floating-point values for the first pixel (red, then green, then blue). Pixels are
    // contiguous, going left to right for the width of the image, followed by the next row
    // below, for the full height of the image.

    auto n = bytes_per_pixel; // Dummy out parameter: original components per pixel
    fdata = stbi_loadf(filename.c_str(), &image_width, &image_height, &n, bytes_per_pixel);
    if (fdata == nullptr) return false;

    convert_to_bytes();
    STBI_FREE(fdata);
    fdata = nullptr;
    return true;
}

int image_loader::width()  const { return (bdata == nullptr) ? 0 : image_width; }

int image_loader::height() const { return (bdata == nullptr) ? 0 : image_height; }

image_record image_loader::get_record() const {
    image_record record;
    record.image_data = bdata;
    record.image_width = image_width;
    record.image_height = image_height;
    record.bytes_per_pixel = bytes_per_pixel;
    return record;
}

image_record image_loader::get_record_cuda() {
    image_copy_to_cuda();
    image_record record_cuda;
    record_cuda.image_data = bdata_cuda;
    record_cuda.image_width = image_width;
    record_cuda.image_height = image_height;
    record_cuda.bytes_per_pixel = bytes_per_pixel;
    return record_cuda;
}

void image_loader::image_copy_to_cuda() {
    if (bdata_cuda == nullptr) {
        cudaMalloc(&bdata_cuda, image_width*image_height*bytes_per_pixel*sizeof(unsigned char));
        cudaMemcpy(bdata_cuda,bdata,image_width*image_height*bytes_per_pixel*sizeof(unsigned char),cudaMemcpyHostToDevice);
    }
}

int image_loader::clamp(const int x, const int low, const int high) {
    // Return the value clamped to the range [low, high).
    if (x < low) return low;
    if (x < high) return x;
    return high - 1;
}

unsigned char image_loader::float_to_byte(const float value) {
    if (value <= 0.0)
        return 0;
    if (1.0 <= value)
        return 255;
    return static_cast<unsigned char>(256.0 * value);
}

void image_loader::convert_to_bytes() {
    // Convert the linear floating point pixel data to bytes, storing the resulting byte
    // data in the `bdata` member.

    const int total_bytes = image_width * image_height * bytes_per_pixel;
    bdata = new unsigned char[total_bytes];

    // Iterate through all pixel components, converting from [0.0, 1.0] float values to
    // unsigned [0, 255] byte values.

    auto *bptr = bdata;
    auto *fptr = fdata;
    for (auto i=0; i < total_bytes; i++, fptr++, bptr++)
        *bptr = float_to_byte(*fptr);
}

// Restore MSVC compiler warnings
#ifdef _MSC_VER
    #pragma warning (pop)
#endif