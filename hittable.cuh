#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.cuh"
#include "color.cuh"
#include "vec.cuh"
#include "ray.cuh"


class material;


class hit_record {
  public:
    float u;
    float v;

    NormVec3 normal;
    float t;
    short mat_id;
    bool front_face;

    __host__ __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(vec3(r.direction()), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

struct compressed_pxId {
    unsigned short pxId{};
    __host__ __device__ void set_pxId(const unsigned int col_id, const unsigned int row_id) {
        pxId = (col_id & 0xff) | ((row_id & 0xff) << 8);
    }
    __host__ __device__ std::tuple<int, int> get_pixel_block_xy() const {
        return std::make_tuple(pxId & 0xff, pxId >> 8);
    }
};

struct ray_info {
    ray r0{};
    color pixel{};
    short depth{};
    short sampleId{};
    compressed_pxId pxId{};

    struct bounds {
        __device__ explicit bounds(const ray_info& elem) {
            const auto& origin = elem.r0.origin();
            const auto& direction = elem.r0.direction().get_compressed();
            ox.min = ox.max = origin.x;
            oy.min = oy.max = origin.y;
            oz.min = oz.max = origin.z;
            dx.min = dx.max = direction.x;
            dy.min = dy.max = direction.y;
        }
        __device__ bounds():
        ox(interval::empty()), oy(interval::empty()), oz(interval::empty()),
        dx(interval::empty()), dy(interval::empty()) {}
        __device__ void add_point(const ray& ray) {
            const auto& origin = ray.origin();
            const auto& direction = ray.direction().get_compressed();
            ox.add_point(origin.x);
            oy.add_point(origin.y);
            oz.add_point(origin.z);
            dx.add_point(direction.x);
            dy.add_point(direction.y);
        }
        __device__ static bounds merge_op(const bounds& a, const bounds& b) {
            auto result = bounds{};
            result.ox = {a.ox, b.ox};
            result.oy = {a.oy, b.oy};
            result.oz = {a.oz, b.oz};
            result.dx = {a.dx, b.dx};
            result.dy = {a.dy, b.dy};
            return result;
        }
        interval ox, oy, oz;
        interval dx, dy;
    };
    __device__ bool is_valid_ray() const {
        return depth > 0;
    }
    __device__ bool is_pixel_ready() const {
        return depth == -1;
    }
    __device__ void update_pixel_state(const bool end) {
        if (depth == 0 || end) {
            depth = -1;
            pixel = end ? pixel : color{0.0f, 0.0f, 0.0f};
        }
    }
    __device__ void retire() {
        depth = -2;
    }
    // 6bits per component
    // MSB: 0 if valid
    // [0:29] dx dy z y x ...
    static __device__ unsigned int get_morton(const ray_info & key, const bounds &b) {
        const auto& origin = key.r0.origin();
        const auto& direction = key.r0.direction().get_compressed();
        const unsigned int r0x = static_cast<unsigned int>((origin.x-b.ox.min)/(b.ox.max-b.ox.min) * 63.0f);
        const unsigned int r0y = static_cast<unsigned int>((origin.y-b.oy.min)/(b.oy.max-b.oy.min) * 63.0f);
        const unsigned int r0z = static_cast<unsigned int>((origin.z-b.oz.min)/(b.oz.max-b.oz.min) * 63.0f);
        const unsigned int r0dx = static_cast<unsigned int>((direction.x-b.dx.min)/(b.dx.max-b.dx.min) * 63.0f);
        const unsigned int r0dy = static_cast<unsigned int>((direction.y-b.dy.min)/(b.dy.max-b.dy.min) * 63.0f);
        return (spreadBits5D(r0dx) << 4) | (spreadBits5D(r0dy) << 3) |
            (spreadBits5D(r0z) << 2) | (spreadBits5D(r0y) << 1) | (spreadBits5D(r0x)) | (key.is_valid_ray() ? 0x0 : 0x80000000);
    }
    static __device__ unsigned int spreadBits5D(unsigned int x) {
        x &= 0x3f;
        x = (x | (x << 16)) & 0x30000f;
        x = (x | (x << 8)) & 0x300c03;
        x = (x | (x << 4)) & 0x2108421;
        return x;
    }

};

enum class hit_type {
    eUndefined,
    eBVH,
    eSphere,
    eQuad,
    eTranslate,
    eRotate_y,
    eList
};

class hittable {
  public:
    hit_type type{hit_type::eUndefined};
    __host__ __device__ virtual ~hittable() {}

    __host__ __device__ virtual aabb bounding_box() const = 0;
};

__host__ __device__ inline bool get_hit(const ray& r, const interval ray_t, hit_record& rec, const hittable* hit);

class translate final : public hittable {
  public:
    __host__ __device__ translate(): object(nullptr), offset(0) {
        type=hit_type::eTranslate;
    }

    __host__ __device__ translate(hittable *object, const vec3& offset)
      : object(object), offset(offset) {
        type=hit_type::eTranslate;
    }

    __host__ __device__ bool hit(const ray& r, const interval ray_t, hit_record& rec) const {
        // Move the ray backwards by the offset
        const ray offset_r(r.origin() - offset, vec3(r.direction()));

        // Determine whether an intersection exists along the offset ray (and if so, where)
        if (!get_hit(offset_r, ray_t, rec, object))
            return false;

        // No need to move the intersection point forwards by the offset,
        // since the hit-point can be inferred by ray.at(rec.t)

        return true;
    }

    __host__ __device__ aabb bounding_box() const override { return object->bounding_box() + offset; }

  private:
    hittable* object;
    vec3 offset;
};


class rotate_y final : public hittable {
  public:
    __host__ __device__ rotate_y(): object(nullptr), sin_theta(0), cos_theta(0) {
        type = hit_type::eRotate_y;
    }
    __host__ __device__ rotate_y(hittable* object, const float angle) : object(object) {
        type = hit_type::eRotate_y;
        const auto radians = degrees_to_radians(angle);
        sin_theta = std::sin(radians);
        cos_theta = std::cos(radians);
        bbox = object->bounding_box();

        point3 min( infinity,  infinity,  infinity);
        point3 max(-infinity, -infinity, -infinity);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    const auto x = i*bbox.x.max + (1-i)*bbox.x.min;
                    const auto y = j*bbox.y.max + (1-j)*bbox.y.min;
                    const auto z = k*bbox.z.max + (1-k)*bbox.z.min;

                    const auto newx =  cos_theta*x + sin_theta*z;
                    const auto newz = -sin_theta*x + cos_theta*z;

                    vec3 tester(newx, y, newz);

                    for (int c = 0; c < 3; c++) {
                        min[c] = std::fmin(min[c], tester[c]);
                        max[c] = std::fmax(max[c], tester[c]);
                    }
                }
            }
        }

        bbox = aabb(min, max);
    }

    __host__ __device__ bool hit(const ray& r, const interval ray_t, hit_record& rec) const {

        // Transform the ray from world space to object space.

        const auto origin = point3(
            (cos_theta * r.origin().x) - (sin_theta * r.origin().z),
            r.origin().y,
            (sin_theta * r.origin().x) + (cos_theta * r.origin().z)
        );
        const auto decomp_normal = vec3(r.direction());
        const auto direction = vec3(
            (cos_theta * decomp_normal.x) - (sin_theta * decomp_normal.z),
            decomp_normal.y,
            (sin_theta * decomp_normal.x) + (cos_theta * decomp_normal.z)
        );

        const ray rotated_r(origin, direction);

        // Determine whether an intersection exists in object space (and if so, where).

        if (!get_hit(rotated_r, ray_t, rec, object))
            return false;

        // Transform the intersection from object space back to world space.
        // No need to worry about hit-point
        const auto decomp_rec_normal = vec3(rec.normal);
        rec.normal = vec3(
            (cos_theta * decomp_rec_normal.x) + (sin_theta * decomp_rec_normal.z),
            decomp_rec_normal.y,
            (-sin_theta * decomp_rec_normal.x) + (cos_theta * decomp_rec_normal.z)
        );

        return true;
    }

    __host__ __device__ aabb bounding_box() const override { return bbox; }

  private:
    hittable* object;
    float sin_theta;
    float cos_theta;
    aabb bbox;
};


#endif
