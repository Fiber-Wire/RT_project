#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H
#include <string>

struct image_record {
    unsigned char* image_data{};
    int image_width = 0;
    int image_height = 0;
    int bytes_per_pixel = 3;
    __device__ __host__ int bytes_per_scanline() const {
        return bytes_per_pixel * image_width;
    }
};

class image_loader {
  public:
    image_loader();

    explicit image_loader(const char* image_filename);
    ~image_loader();

    bool load(const std::string& filename);

    int width()  const;
    int height() const;

    image_record get_record() const;

    image_record get_record_cuda();

  private:
    unsigned char* bdata_cuda{};
    const int      bytes_per_pixel = 3;
    float         *fdata = nullptr;         // Linear floating point pixel data
    unsigned char *bdata = nullptr;         // Linear 8-bit pixel data
    int            image_width = 0;         // Loaded image width
    int            image_height = 0;        // Loaded image height

    void image_copy_to_cuda();

    static int clamp(int x, int low, int high);

    static unsigned char float_to_byte(float value);

    void convert_to_bytes();
};





#endif
