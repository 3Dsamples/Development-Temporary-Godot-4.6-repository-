// File 331: modules/wicked/src/collision/wicked_shape.h
// Complete collision shape library for the WickedEngine module.
// Provides support point queries, inertia tensors, and AABBs for all
// common shape types: sphere, box, capsule, cylinder, cone, convex hull,
// triangle mesh, heightfield, and compound. All methods are inline for
// maximum performance in the narrow‑phase GJK.

#ifndef WICKED_COLLISION_SHAPE_H
#define WICKED_COLLISION_SHAPE_H

#include "core/object/ref_counted.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/templates/local_vector.h"
#include "../core/wicked_types.h"
#include "../core/wicked_constants.h"

namespace wicked {

// ---------------------------------------------------------------------------
// Base class
// ---------------------------------------------------------------------------
class WickedShape : public RefCounted {
    GDCLASS(WickedShape, RefCounted);
public:
    WickedShape() {}
    virtual ~WickedShape() {}
    virtual ShapeType get_shape_type() const = 0;
    virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const = 0;
    virtual mat3 compute_inertia(real_t p_mass) const = 0;
    virtual aabb get_local_aabb() const = 0;
protected:
    static void _bind_methods() {}
};

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------
class WickedShapeSphere : public WickedShape {
    GDCLASS(WickedShapeSphere, WickedShape);
    real_t radius = 0.5;
public:
    WickedShapeSphere(real_t p_r = 0.5) : radius(MAX(p_r, 0.0)) {}
    virtual ShapeType get_shape_type() const override { return ShapeType::SPHERE; }
    virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
        vec3 local_dir = p_transform.basis.xform_inv(p_dir).normalized();
        return p_transform.xform(local_dir * radius);
    }
    virtual mat3 compute_inertia(real_t p_mass) const override {
        real_t I = (2.0f / 5.0f) * p_mass * radius * radius;
        return mat3().scaled(vec3(I, I, I));
    }
    virtual aabb get_local_aabb() const override {
        return aabb(vec3(-radius), vec3(radius * 2));
    }
    void set_radius(real_t p_r) { radius = MAX(p_r, 0.0); }
    real_t get_radius() const { return radius; }
protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_radius", "radius"), &WickedShapeSphere::set_radius);
        ClassDB::bind_method(D_METHOD("get_radius"), &WickedShapeSphere::get_radius);
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
    }
};

// ---------------------------------------------------------------------------
// Box
// ---------------------------------------------------------------------------
class WickedShapeBox : public WickedShape {
    GDCLASS(WickedShapeBox, WickedShape);
    vec3 half_extents = vec3(0.5, 0.5, 0.5);
public:
    WickedShapeBox(const vec3 &p_he = vec3(0.5,0.5,0.5)) : half_extents(p_he.abs()) {}
    virtual ShapeType get_shape_type() const override { return ShapeType::BOX; }
    virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
        vec3 local_dir = p_transform.basis.xform_inv(p_dir);
        return p_transform.xform(vec3(
            (local_dir.x >= 0) ? half_extents.x : -half_extents.x,
            (local_dir.y >= 0) ? half_extents.y : -half_extents.y,
            (local_dir.z >= 0) ? half_extents.z : -half_extents.z));
    }
    virtual mat3 compute_inertia(real_t p_mass) const override {
        real_t x = half_extents.x, y = half_extents.y, z = half_extents.z;
        return mat3().scaled(vec3(
            (1.0f/12.0f) * p_mass * (y*y + z*z),
            (1.0f/12.0f) * p_mass * (x*x + z*z),
            (1.0f/12.0f) * p_mass * (x*x + y*y)));
    }
    virtual aabb get_local_aabb() const override {
        return aabb(-half_extents, half_extents * 2.0f);
    }
    void set_half_extents(const vec3 &p_he) { half_extents = p_he.abs(); }
    const vec3 &get_half_extents() const { return half_extents; }
protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_half_extents", "extents"), &WickedShapeBox::set_half_extents);
        ClassDB::bind_method(D_METHOD("get_half_extents"), &WickedShapeBox::get_half_extents);
        ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "half_extents"), "set_half_extents", "get_half_extents");
    }
};

// ---------------------------------------------------------------------------
// Capsule (along Y axis)
// ---------------------------------------------------------------------------
class WickedShapeCapsule : public WickedShape {
    GDCLASS(WickedShapeCapsule, WickedShape);
    real_t radius = 0.5;
    real_t height = 1.0;
public:
    WickedShapeCapsule(real_t r=0.5, real_t h=1.0) : radius(MAX(r,0.0)), height(MAX(h,0.0)) {}
    virtual ShapeType get_shape_type() const override { return ShapeType::CAPSULE; }
    virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
        vec3 local_dir = p_transform.basis.xform_inv(p_dir).normalized();
        real_t half_h = MAX(height * 0.5f - radius, 0.0f);
        vec3 center(0,0,0);
        if (local_dir.y > 0.0f) center.y = half_h;
        else if (local_dir.y < 0.0f) center.y = -half_h;
        return p_transform.xform(center + local_dir * radius);
    }
    virtual mat3 compute_inertia(real_t p_mass) const override {
        real_t r2 = radius * radius, h2 = height * height;
        real_t I_xy = p_mass * (3.0f * r2 + h2) / 12.0f;
        real_t I_z  = p_mass * r2 * 0.5f;
        return mat3().scaled(vec3(I_xy, I_xy, I_z));
    }
    virtual aabb get_local_aabb() const override {
        real_t half_h = height * 0.5f;
        return aabb(vec3(-radius, -half_h, -radius), vec3(radius*2, height, radius*2));
    }
    void set_radius(real_t r) { radius = MAX(r,0.0); }
    real_t get_radius() const { return radius; }
    void set_height(real_t h) { height = MAX(h,0.0); }
    real_t get_height() const { return height; }
protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_radius","r"), &WickedShapeCapsule::set_radius);
        ClassDB::bind_method(D_METHOD("get_radius"), &WickedShapeCapsule::get_radius);
        ClassDB::bind_method(D_METHOD("set_height","h"), &WickedShapeCapsule::set_height);
        ClassDB::bind_method(D_METHOD("get_height"), &WickedShapeCapsule::get_height);
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"radius"), "set_radius","get_radius");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"height"), "set_height","get_height");
    }
};

// ---------------------------------------------------------------------------
// Cylinder (along Y axis)
// ---------------------------------------------------------------------------
class WickedShapeCylinder : public WickedShape {
    GDCLASS(WickedShapeCylinder, WickedShape);
    real_t radius = 0.5;
    real_t height = 1.0;
public:
    WickedShapeCylinder(real_t r=0.5, real_t h=1.0) : radius(MAX(r,0.0)), height(MAX(h,0.0)) {}
    virtual ShapeType get_shape_type() const override { return ShapeType::CYLINDER; }
    virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
        vec3 local_dir = p_transform.basis.xform_inv(p_dir);
        real_t half_h = height * 0.5f;
        real_t cap_y = (local_dir.y >= 0) ? half_h : -half_h;
        real_t rad_len = Math::sqrt(local_dir.x*local_dir.x + local_dir.z*local_dir.z);
        vec3 support_local(0, cap_y, 0);
        if (rad_len > CMP_EPSILON) {
            real_t s = radius / rad_len;
            support_local.x += local_dir.x * s;
            support_local.z += local_dir.z * s;
        }
        return p_transform.xform(support_local);
    }
    virtual mat3 compute_inertia(real_t p_mass) const override {
        real_t r2 = radius*radius, h2 = height*height;
        real_t I_xy = p_mass*(3.0f*r2 + h2)/12.0f;
        real_t I_z = p_mass * r2 * 0.5f;
        return mat3().scaled(vec3(I_xy, I_xy, I_z));
    }
    virtual aabb get_local_aabb() const override {
        real_t half_h = height*0.5f;
        return aabb(vec3(-radius, -half_h, -radius), vec3(radius*2, height, radius*2));
    }
    void set_radius(real_t r) { radius = MAX(r,0.0); }
    real_t get_radius() const { return radius; }
    void set_height(real_t h) { height = MAX(h,0.0); }
    real_t get_height() const { return height; }
protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_radius","r"), &WickedShapeCylinder::set_radius);
        ClassDB::bind_method(D_METHOD("get_radius"), &WickedShapeCylinder::get_radius);
        ClassDB::bind_method(D_METHOD("set_height","h"), &WickedShapeCylinder::set_height);
        ClassDB::bind_method(D_METHOD("get_height"), &WickedShapeCylinder::get_height);
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"radius"), "set_radius","get_radius");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"height"), "set_height","get_height");
    }
};

// ---------------------------------------------------------------------------
// Cone (along Y axis)
// ---------------------------------------------------------------------------
class WickedShapeCone : public WickedShape {
    GDCLASS(WickedShapeCone, WickedShape);
    real_t radius = 0.5;
    real_t height = 1.0;
public:
    WickedShapeCone(real_t r=0.5, real_t h=1.0) : radius(MAX(r,0.0)), height(MAX(h,0.0)) {}
    virtual ShapeType get_shape_type() const override { return ShapeType::CONE; }
    virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
        vec3 local_dir = p_transform.basis.xform_inv(p_dir).normalized();
        real_t half_h = height * 0.5f;
        if (local_dir.y > 0.999f) return p_transform.xform(vec3(0, half_h, 0));
        real_t rad_len = Math::sqrt(local_dir.x*local_dir.x + local_dir.z*local_dir.z);
        vec3 base_point(0, -half_h, 0);
        if (rad_len > CMP_EPSILON) {
            real_t s = radius / rad_len;
            base_point.x = local_dir.x * s;
            base_point.z = local_dir.z * s;
        }
        vec3 tip(0, half_h, 0);
        return p_transform.xform((tip.dot(local_dir) > base_point.dot(local_dir)) ? tip : base_point);
    }
    virtual mat3 compute_inertia(real_t p_mass) const override {
        real_t r2 = radius*radius, h2 = height*height;
        real_t I_xy = p_mass*(3.0f*r2 + h2)/20.0f;
        real_t I_z = p_mass * r2 * 0.3f;
        return mat3().scaled(vec3(I_xy, I_xy, I_z));
    }
    virtual aabb get_local_aabb() const override {
        real_t half_h = height*0.5f;
        return aabb(vec3(-radius, -half_h, -radius), vec3(radius*2, height, radius*2));
    }
    void set_radius(real_t r) { radius = MAX(r,0.0); }
    real_t get_radius() const { return radius; }
    void set_height(real_t h) { height = MAX(h,0.0); }
    real_t get_height() const { return height; }
protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_radius","r"), &WickedShapeCone::set_radius);
        ClassDB::bind_method(D_METHOD("get_radius"), &WickedShapeCone::get_radius);
        ClassDB::bind_method(D_METHOD("set_height","h"), &WickedShapeCone::set_height);
        ClassDB::bind_method(D_METHOD("get_height"), &WickedShapeCone::get_height);
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"radius"), "set_radius","get_radius");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"height"), "set_height","get_height");
    }
};

// ---------------------------------------------------------------------------
// Convex hull (point cloud)
// ---------------------------------------------------------------------------
class WickedShapeConvexHull : public WickedShape {
    GDCLASS(WickedShapeConvexHull, WickedShape);
    LocalVector<vec3> vertices;
    aabb local_aabb;
public:
    WickedShapeConvexHull() {}
    void add_vertex(const vec3 &p_v) {
        vertices.push_back(p_v);
        if (vertices.size()==1) local_aabb = aabb(p_v, vec3());
        else local_aabb.expand_to(p_v);
    }
    void clear_vertices() { vertices.clear(); local_aabb = aabb(); }
    int get_vertex_count() const { return vertices.size(); }
    const vec3 &get_vertex(int p_idx) const { return vertices[p_idx]; }
    virtual ShapeType get_shape_type() const override { return ShapeType::CONVEX_HULL; }
    virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
        vec3 local_dir = p_transform.basis.xform_inv(p_dir);
        real_t best_dot = -INFINITY;
        int best = 0;
        for (int i=0; i<vertices.size(); ++i) {
            real_t d = vertices[i].dot(local_dir);
            if (d > best_dot) { best_dot = d; best = i; }
        }
        return p_transform.xform(vertices[best]);
    }
    virtual mat3 compute_inertia(real_t p_mass) const override {
        if (vertices.is_empty()) return mat3().scaled(vec3(1,1,1));
        real_t inv_n = 1.0f / real_t(vertices.size());
        real_t pt_mass = p_mass * inv_n;
        mat3 I; I.scale(vec3(0,0,0));
        for (const vec3 &v : vertices) {
            real_t x=v.x, y=v.y, z=v.z;
            vec3 diag(y*y+z*z, x*x+z*z, x*x+y*y);
            mat3 contrib; contrib.set(diag.x,0,0, 0,diag.y,0, 0,0,diag.z);
            contrib[0][1] = -pt_mass*x*y; contrib[1][0] = -pt_mass*x*y;
            contrib[0][2] = -pt_mass*x*z; contrib[2][0] = -pt_mass*x*z;
            contrib[1][2] = -pt_mass*y*z; contrib[2][1] = -pt_mass*y*z;
            I += contrib;
        }
        return I;
    }
    virtual aabb get_local_aabb() const override { return local_aabb; }
protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("add_vertex","vertex"), &WickedShapeConvexHull::add_vertex);
        ClassDB::bind_method(D_METHOD("clear_vertices"), &WickedShapeConvexHull::clear_vertices);
        ClassDB::bind_method(D_METHOD("get_vertex_count"), &WickedShapeConvexHull::get_vertex_count);
        ClassDB::bind_method(D_METHOD("get_vertex","index"), &WickedShapeConvexHull::get_vertex);
    }
};

// ---------------------------------------------------------------------------
// Triangle mesh (static/kinematic)
// ---------------------------------------------------------------------------
class WickedShapeTriMesh : public WickedShape {
    GDCLASS(WickedShapeTriMesh, WickedShape);
    LocalVector<vec3> vertices;
    LocalVector<int>  indices;
    aabb local_aabb;
public:
    WickedShapeTriMesh() {}
    void build(const LocalVector<vec3> &p_verts, const LocalVector<int> &p_indices) {
        vertices = p_verts; indices = p_indices;
        if (!vertices.is_empty()) {
            local_aabb = aabb(vertices[0], vec3());
            for (int i=1; i<vertices.size(); ++i) local_aabb.expand_to(vertices[i]);
        }
    }
    int get_triangle_count() const { return indices.size() / 3; }
    const vec3 &get_vertex(int p_idx) const { return vertices[p_idx]; }
    const int &get_index(int p_idx) const { return indices[p_idx]; }
    virtual ShapeType get_shape_type() const override { return ShapeType::TRIANGLE_MESH; }
    virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
        // Approximate convex support (not correct for concave meshes; use only for static AABB).
        vec3 local_dir = p_transform.basis.xform_inv(p_dir);
        real_t best_dot = -INFINITY;
        vec3 best_vertex(0,0,0);
        for (const vec3 &v : vertices) {
            real_t d = v.dot(local_dir);
            if (d > best_dot) { best_dot = d; best_vertex = v; }
        }
        return p_transform.xform(best_vertex);
    }
    virtual mat3 compute_inertia(real_t p_mass) const override {
        vec3 size = local_aabb.size;
        real_t Ix = (1.0f/12.0f)*p_mass*(size.y*size.y + size.z*size.z);
        real_t Iy = (1.0f/12.0f)*p_mass*(size.x*size.x + size.z*size.z);
        real_t Iz = (1.0f/12.0f)*p_mass*(size.x*size.x + size.y*size.y);
        return mat3().scaled(vec3(Ix, Iy, Iz));
    }
    virtual aabb get_local_aabb() const override { return local_aabb; }
protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("build","vertices","indices"), &WickedShapeTriMesh::build);
        ClassDB::bind_method(D_METHOD("get_triangle_count"), &WickedShapeTriMesh::get_triangle_count);
    }
};

// ---------------------------------------------------------------------------
// Heightfield
// ---------------------------------------------------------------------------
class WickedShapeHeightfield : public WickedShape {
    GDCLASS(WickedShapeHeightfield, WickedShape);
    int width = 1, depth = 1;
    real_t cell_size = 1.0;
    LocalVector<real_t> heights;
    real_t min_height = 0.0, max_height = 0.0;
public:
    WickedShapeHeightfield() {}
    void set_grid(int p_w, int p_d, real_t p_cell = 1.0) {
        width = MAX(p_w,2); depth = MAX(p_d,2); cell_size = MAX(p_cell,1e-6);
        heights.resize(width*depth);
        for (int i=0; i<heights.size(); ++i) heights[i]=0.0;
        min_height = max_height = 0.0;
    }
    void set_height(int x, int z, real_t h) {
        ERR_FAIL_INDEX(x,width); ERR_FAIL_INDEX(z,depth);
        heights[z*width + x] = h;
        if (h < min_height) min_height = h;
        if (h > max_height) max_height = h;
    }
    real_t get_height(int x, int z) const {
        ERR_FAIL_INDEX_V(x,width,0.0); ERR_FAIL_INDEX_V(z,depth,0.0);
        return heights[z*width + x];
    }
    virtual ShapeType get_shape_type() const override { return ShapeType::HEIGHTFIELD; }
    virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
        vec3 local_dir = p_transform.basis.xform_inv(p_dir);
        vec3 offset = vec3(-width*cell_size*0.5f, 0.0f, -depth*cell_size*0.5f);
        real_t best_dot = -INFINITY;
        vec3 best_vertex(0,0,0);
        for (int z=0; z<depth; ++z) {
            for (int x=0; x<width; ++x) {
                real_t h00 = heights[z*width + x];
                real_t h10 = (x+1<width) ? heights[z*width + (x+1)] : h00;
                real_t h01 = (z+1<depth) ? heights[(z+1)*width + x] : h00;
                real_t h11 = (x+1<width && z+1<depth) ? heights[(z+1)*width + (x+1)] : h00;
                real_t lx = offset.x + x*cell_size;
                real_t lz = offset.z + z*cell_size;
                vec3 corners[4] = { vec3(lx,h00,lz), vec3(lx+cell_size,h10,lz),
                    vec3(lx,h01,lz+cell_size), vec3(lx+cell_size,h11,lz+cell_size) };
                for (int c=0; c<4; ++c) {
                    real_t d = corners[c].dot(local_dir);
                    if (d > best_dot) { best_dot = d; best_vertex = corners[c]; }
                }
            }
        }
        return p_transform.xform(best_vertex);
    }
    virtual mat3 compute_inertia(real_t p_mass) const override {
        vec3 size = get_local_aabb().size;
        return mat3().scaled(vec3(
            (1.0f/12.0f)*p_mass*(size.y*size.y + size.z*size.z),
            (1.0f/12.0f)*p_mass*(size.x*size.x + size.z*size.z),
            (1.0f/12.0f)*p_mass*(size.x*size.x + size.y*size.y)));
    }
    virtual aabb get_local_aabb() const override {
        vec3 half_ext(width*cell_size*0.5f, (max_height-min_height)*0.5f, depth*cell_size*0.5f);
        return aabb(vec3(-half_ext.x, min_height, -half_ext.z), half_ext*2.0f);
    }
protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_grid","width","depth","cell_size"), &WickedShapeHeightfield::set_grid, DEFVAL(1.0));
        ClassDB::bind_method(D_METHOD("set_height","x","z","height"), &WickedShapeHeightfield::set_height);
        ClassDB::bind_method(D_METHOD("get_height","x","z"), &WickedShapeHeightfield::get_height);
    }
};

// ---------------------------------------------------------------------------
// Compound shape
// ---------------------------------------------------------------------------
class WickedShapeCompound : public WickedShape {
    GDCLASS(WickedShapeCompound, WickedShape);
public:
    struct SubShape {
        Ref<WickedShape> shape;
        mat4 local_transform;
    };
private:
    LocalVector<SubShape> sub_shapes;
public:
    WickedShapeCompound() {}
    void add_sub_shape(const Ref<WickedShape> &p_shape, const mat4 &p_local = mat4()) {
        ERR_FAIL_COND(p_shape.is_null());
        sub_shapes.push_back({p_shape, p_local});
    }
    void remove_sub_shape(int p_idx) { ERR_FAIL_INDEX(p_idx, sub_shapes.size()); sub_shapes.remove_at(p_idx); }
    void clear_sub_shapes() { sub_shapes.clear(); }
    int get_sub_shape_count() const { return sub_shapes.size(); }
    virtual ShapeType get_shape_type() const override { return ShapeType::COMPOUND; }
    virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_compound_xform) const override {
        vec3 best_point; real_t best_dot = -INFINITY;
        for (const SubShape &s : sub_shapes) {
            mat4 world = p_compound_xform * s.local_transform;
            vec3 pt = s.shape->get_support(p_dir, world);
            real_t d = pt.dot(p_dir);
            if (d > best_dot) { best_dot = d; best_point = pt; }
        }
        return best_point;
    }
    virtual mat3 compute_inertia(real_t p_mass) const override {
        if (sub_shapes.is_empty()) return mat3().scaled(vec3(1,1,1));
        real_t total_vol = 0.0; vec3 com(0,0,0);
        for (const SubShape &s : sub_shapes) {
            vec3 sz = s.shape->get_local_aabb().size;
            real_t v = sz.x*sz.y*sz.z;
            total_vol += v;
            com += s.local_transform.origin * v;
        }
        if (total_vol < CMP_EPSILON) return mat3().scaled(vec3(1,1,1));
        com /= total_vol;
        mat3 I; I.scale(vec3(0,0,0));
        for (const SubShape &s : sub_shapes) {
            vec3 sz = s.shape->get_local_aabb().size;
            real_t v = sz.x*sz.y*sz.z;
            real_t sub_mass = p_mass * v / total_vol;
            mat3 local = s.shape->compute_inertia(sub_mass);
            mat3 rot = s.local_transform.basis;
            mat3 rotated = rot * local * rot.transposed();
            vec3 d = s.local_transform.origin - com;
            real_t d2 = d.length_squared();
            for (int r=0; r<3; ++r) for (int c=0; c<3; ++c) {
                real_t add = rotated[r][c] + sub_mass * ((r==c ? d2 : 0.0f) - d[r]*d[c]);
                I[r][c] += add;
            }
        }
        return I;
    }
    virtual aabb get_local_aabb() const override {
        if (sub_shapes.is_empty()) return aabb();
        aabb res; bool first = true;
        for (const SubShape &s : sub_shapes) {
            aabb local = s.shape->get_local_aabb();
            mat4 world = s.local_transform;
            vec3 corners[8];
            vec3 mn = local.position, mx = local.position + local.size;
            corners[0] = world.xform(mn);
            corners[1] = world.xform(vec3(mx.x, mn.y, mn.z));
            corners[2] = world.xform(vec3(mx.x, mx.y, mn.z));
            corners[3] = world.xform(vec3(mn.x, mx.y, mn.z));
            corners[4] = world.xform(vec3(mn.x, mn.y, mx.z));
            corners[5] = world.xform(vec3(mx.x, mn.y, mx.z));
            corners[6] = world.xform(mx);
            corners[7] = world.xform(vec3(mn.x, mx.y, mx.z));
            for (int i=0; i<8; ++i) {
                if (first) { res = aabb(corners[i], vec3()); first = false; }
                else res.expand_to(corners[i]);
            }
        }
        return res;
    }
protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("add_sub_shape","shape","local_transform"), &WickedShapeCompound::add_sub_shape, DEFVAL(mat4()));
        ClassDB::bind_method(D_METHOD("remove_sub_shape","index"), &WickedShapeCompound::remove_sub_shape);
        ClassDB::bind_method(D_METHOD("clear_sub_shapes"), &WickedShapeCompound::clear_sub_shapes);
        ClassDB::bind_method(D_METHOD("get_sub_shape_count"), &WickedShapeCompound::get_sub_shape_count);
    }
};

} // namespace wicked

#endif // WICKED_COLLISION_SHAPE_H