#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "aabb.cuh"
#include "hittable.cuh"

class hittable_list : public hittable {
  public:
    hittable** objects{};
    int count{};
    int capacity{};

    __host__ __device__ explicit hittable_list(const int capacity) : capacity(capacity) {
        objects = new hittable*[capacity];
        are_hitlist = new bool[capacity];
    }

    __host__ __device__ explicit hittable_list(hittable* object) { add(object); }
    hittable_list(const hittable_list& other) {
        *this = other;
    }
    __host__ __device__ hittable_list& operator=(const hittable_list& other) {
        if (this != &other) {
            capacity = other.capacity;
            delete[] objects;
            delete[] are_hitlist;
            objects = new hittable*[capacity];
            are_hitlist = new bool[capacity];
            count = other.count;
            for (int i = 0; i < count; i++) {
                objects[i] = other.objects[i];
                are_hitlist[i] = other.are_hitlist[i];
            }
            bbox = other.bbox;
            for (int i = 0; i < count; i++) {
                if (are_hitlist[i]) {
                    objects[i] = new hittable_list{other.capacity};
                    *(hittable_list*)(objects[i]) = *(hittable_list*)(other.objects[i]);
                }
            }
        }
        return *this;
    }

    __host__ __device__ void clear() {
        delete[] objects;
        delete[] are_hitlist;
        count = 0;
        objects = new hittable*[capacity];
        are_hitlist = new bool[capacity];
    }

    __host__ __device__ std::span<hittable*> get_objects() {
        return {objects, static_cast<size_t>(count)};
    }

    __host__ __device__ void add(hittable* object) {
        if (count < capacity) {
            objects[count] = object;
            are_hitlist[count] = false;
            bbox = aabb(bbox, object->bounding_box());
            count++;
        }

    }

    __host__ __device__ void add(hittable_list* list) {
        if (count < capacity) {
            objects[count] = list;
            are_hitlist[count] = true;
            bbox = aabb(bbox, list->bounding_box());
            count++;
        }

    }

    __host__ __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
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

    __host__ __device__ ~hittable_list() {
        delete[] are_hitlist;
        delete[] objects;
        capacity = 0;
        count = 0;
    }

  private:
    aabb bbox;
    bool* are_hitlist{};
};


#endif
