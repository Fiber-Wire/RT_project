#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "aabb.cuh"
#include "hittable.cuh"

#include <vector>


class hittable_list : public hittable {
  public:
    std::vector<hittable*> objects;

    hittable_list() {}
    hittable_list(hittable* object) { add(object); }
    hittable_list(const hittable_list& other) {
        *this = other;
    }
    hittable_list& operator=(const hittable_list& other) {
        if (this != &other) {
            objects = other.objects;
            bbox = other.bbox;
            are_hitlist = other.are_hitlist;
            for (int i = 0; i < objects.size(); i++) {
                if (are_hitlist[i]) {
                    objects[i] = new hittable_list{};
                    *dynamic_cast<hittable_list*>(objects[i]) = *dynamic_cast<hittable_list*>(other.objects[i]);
                }
            }
        }
        return *this;
    }

    void clear() { objects.clear(); }

    void add(hittable* object) {
        objects.push_back(object);
        are_hitlist.push_back(false);
        bbox = aabb(bbox, object->bounding_box());
    }

    void add(hittable_list* list) {
        objects.push_back(list);
        are_hitlist.push_back(true);
        bbox = aabb(bbox, list->bounding_box());
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (const auto& object : objects) {
            if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    aabb bounding_box() const override { return bbox; }

  private:
    aabb bbox;
    std::vector<bool> are_hitlist{};
};


#endif