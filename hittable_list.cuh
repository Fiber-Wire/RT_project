#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "aabb.cuh"
#include "hittable.cuh"

class hittable_list final : public hittable {
  public:
    hittable** objects{};
    int count{};
    int capacity{};

    __host__ __device__ explicit hittable_list() : hittable_list(1) {}

    __host__ __device__ explicit hittable_list(const int capacity) : capacity(capacity) {
        objects = new hittable*[capacity];
        type = hit_type::eList;
    }

    __host__ __device__ explicit hittable_list(hittable* object): hittable_list(1) { add(object); }
    __host__ __device__ hittable_list(const hittable_list& other) {
        *this = other;
    }
    __host__ __device__ hittable_list& operator=(const hittable_list& other) {
        if (this != &other) {
            type = other.type;
            capacity = other.capacity;
            delete[] objects;
            objects = new hittable*[capacity];
            count = other.count;
            for (int i = 0; i < count; i++) {
                objects[i] = other.objects[i];
            }
            bbox = other.bbox;
        }
        return *this;
    }

    __host__ __device__ void clear() {
        delete[] objects;
        count = 0;
        objects = new hittable*[capacity];
    }

    __host__ __device__ std::span<hittable*> get_objects() {
        return {objects, static_cast<size_t>(count)};
    }

    __host__ __device__ void add(hittable* object) {
        if (object == nullptr) {
            return;
        }
        if (count < capacity) {
            objects[count] = object;
            bbox = aabb(bbox, object->bounding_box());
            count++;
        }

    }

    __host__ __device__ void add(const hittable_list* list) {
        if (list == nullptr) {
            return;
        }
        // concatenate the incoming list
        if (count+list->count <= capacity) {
            for (int i = 0; i < list->count; i++) {
                add(list->objects[i]);
            }
        }

    }

    __host__ __device__ bool hit(const ray& r, const interval ray_t, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (auto i = 0; i < count; i++) {
            const auto& object = objects[i];
            if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    __host__ __device__ aabb bounding_box() const override { return bbox; }

    __host__ __device__ ~hittable_list() override {
        delete[] objects;
        capacity = 0;
        count = 0;
    }

  private:
    aabb bbox;
};


#endif
