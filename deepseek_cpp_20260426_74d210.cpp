// File 124: modules/genesis/src/sensors/depth_camera_sensor.h
// Depth camera sensor: simulates a range-imaging sensor (e.g., kinect, lidar-like)
// by casting rays into the scene using the Gaia BVH and rigid/FEM entities.
// Outputs a depth image buffer and optionally a point cloud.

#ifndef GENESIS_SENSORS_DEPTH_CAMERA_SENSOR_H
#define GENESIS_SENSORS_DEPTH_CAMERA_SENSOR_H

#include "base_sensor.h"
#include "../entities/base_entity.h"
#include "../entities/rigid_entity.h"
#include "../entities/fem_entity.h"
#include "../../../gaia/src/bvh/bvh.h"          // Gaia BVH for ray queries
#include "../../../gaia/src/bvh/query.h"        // ray-triangle intersect
#include "core/math/vector3.h"
#include "core/math/camera_matrix.h"
#include "core/templates/local_vector.h"

namespace genesis {

class DepthCameraSensor : public BaseSensor {
	GDCLASS(DepthCameraSensor, BaseSensor);

public:
	DepthCameraSensor() :
		resolution(320, 240),
		fov_y(60.0),                          // degrees
		near_clip(0.05),
		far_clip(20.0),
		max_rays_per_frame(100000),
		noise_sigma(0.0)                      // std of Gaussian range noise [m]
	{
		sensor_type = SensorType::LIDAR;      // or a new custom depth camera type
	}

	// --- Camera parameters ---
	void set_resolution(int w, int h) { resolution.x = MAX(w, 1); resolution.y = MAX(h, 1); }
	Vector2i get_resolution() const { return resolution; }

	void set_fov_y(real_t p_deg) { fov_y = CLAMP(p_deg, 1.0, 179.0); }
	real_t get_fov_y() const { return fov_y; }

	void set_near_clip(real_t p_near) { near_clip = MAX(p_near, 1e-6); }
	real_t get_near_clip() const { return near_clip; }

	void set_far_clip(real_t p_far) { far_clip = MAX(p_far, near_clip); }
	real_t get_far_clip() const { return far_clip; }

	void set_noise_sigma(real_t p_sigma) { noise_sigma = MAX(p_sigma, 0.0); }
	real_t get_noise_sigma() const { return noise_sigma; }

	// --- Access to depth buffer ---
	PackedRealArray get_depth_buffer() const {
		PackedRealArray buf;
		int total = resolution.x * resolution.y;
		buf.resize(total);
		for (int i = 0; i < total; ++i) buf.set(i, depth_buffer[i]);
		return buf;
	}

	// --- Override step to capture ---
	virtual bool step(real_t p_dt, const Ref<BaseEntity> &p_entity) override {
		if (!BaseSensor::step(p_dt, p_entity)) return false;
		capture(p_entity);
		return true;
	}

	// --- Provide the list of entities that can be seen (typically all active) ---
	void set_visible_entities(const LocalVector<Ref<BaseEntity>> &p_entities) {
		visible_entities = p_entities;
	}

protected:
	virtual void capture(const Ref<BaseEntity> &p_entity) override {
		ERR_FAIL_COND(p_entity.is_null());
		Transform3D cam_xform = p_entity->get_transform();
		// Determine camera space basis: typically the entity's -Z is forward (Godot convention)
		Vector3 eye = cam_xform.origin;
		Basis view_basis = cam_xform.basis;

		// Build view frustum parameters
		real_t aspect = real_t(resolution.x) / real_t(resolution.y);
		real_t fov_rad = Math::deg_to_rad(fov_y);
		real_t h_half = Math::tan(fov_rad * 0.5) * near_clip;
		real_t w_half = h_half * aspect;

		// Pixel size in world units at near plane
		Vector3 forward = -view_basis.get_column(2); // -Z
		Vector3 right   =  view_basis.get_column(0);
		Vector3 up      =  view_basis.get_column(1);

		int total_pixels = resolution.x * resolution.y;
		depth_buffer.resize(total_pixels);

		// Build a Gaia BVH of all visible entities' AABBs for acceleration
		gaia::bvh::BVH bvh;
		int n = visible_entities.size();
		if (n > 0) {
			LocalVector<AABB> aabbs;
			aabbs.resize(n);
			for (int i = 0; i < n; ++i) {
				if (visible_entities[i].is_valid()) 
					aabbs[i] = visible_entities[i]->get_aabb();
				else
					aabbs[i] = AABB(); // degenerate, will be skipped
			}
			bvh.build_final(aabbs);
		}

		// Cast rays for each pixel
		for (int y = 0; y < resolution.y; ++y) {
			for (int x = 0; x < resolution.x; ++x) {
				real_t px = (real_t(x) + 0.5) / real_t(resolution.x) * 2.0 - 1.0; // [-1, 1]
				real_t py = (real_t(y) + 0.5) / real_t(resolution.y) * 2.0 - 1.0;

				Vector3 origin = eye;
				Vector3 direction = forward * near_clip + right * px * w_half + up * py * h_half;
				direction.normalize();

				real_t best_t = far_clip;

				// Query BVH to find candidate entities
				AABB ray_aabb(origin, Vector3());
				ray_aabb.expand_to(origin + direction * far_clip);
				bvh.query_intersect(ray_aabb, [&](int prim_idx) {
					if (prim_idx < 0 || prim_idx >= visible_entities.size()) return;
					Ref<BaseEntity> &ent = visible_entities[prim_idx];
					if (ent.is_null()) return;

					// Determine entity type and compute intersection
					Ref<RigidEntity> rigid = ent;
					Ref<FEMEntity> fem = ent;

					if (rigid.is_valid()) {
						// For rigid entities, we can use their collider for exact intersection
						// Currently we approximate by intersecting with the convex shape via GJK/ray.
						// Minimal: we just use the AABB as a coarse test; then refine with GJK if needed.
						AABB ent_aabb = rigid->get_aabb();
						real_t t_entry = 0.0, t_exit = 0.0;
						if (gaia::bvh::intersect_ray_aabb(origin, direction, ent_aabb, 0.0, best_t, t_entry, t_exit)) {
							// For more accurate depth, we would cast ray against the actual collider.
							// Here we use the AABB entry point as an approximation.
							best_t = t_entry;
						}
					} else if (fem.is_valid()) {
						// FEM: test ray against all triangles of the tet mesh surface
						const gaia::mesh::TetMesh &mesh = fem->get_mesh();
						int tet_count = mesh.element_count();
						for (int t = 0; t < tet_count; ++t) {
							gaia::mesh::TetMesh::Tetrahedron tet = mesh.get_tetrahedron(t);
							const Vector3 &p0 = mesh.get_vertex(tet.v0);
							const Vector3 &p1 = mesh.get_vertex(tet.v1);
							const Vector3 &p2 = mesh.get_vertex(tet.v2);
							const Vector3 &p3 = mesh.get_vertex(tet.v3);
							// Four faces of tetrahedron
							test_ray_against_triangle(origin, direction, p0, p1, p2, best_t);
							test_ray_against_triangle(origin, direction, p0, p1, p3, best_t);
							test_ray_against_triangle(origin, direction, p0, p2, p3, best_t);
							test_ray_against_triangle(origin, direction, p1, p2, p3, best_t);
						}
					}
				});

				// Add noise if desired
				if (noise_sigma > 0.0 && best_t < far_clip) {
					RandomNumberGenerator rng;
					rng.randomize();
					real_t noise = rng.randfn(0.0, noise_sigma);
					best_t = MAX(near_clip, best_t + noise);
				}

				int idx = y * resolution.x + x;
				depth_buffer[idx] = best_t; // far_clip if no hit
			}
		}
	}

	void test_ray_against_triangle(const Vector3 &origin, const Vector3 &dir,
								   const Vector3 &v0, const Vector3 &v1, const Vector3 &v2,
								   real_t &inout_best_t) {
		real_t t, u, v;
		if (gaia::bvh::intersect_ray_triangle(origin, dir, v0, v1, v2, t, u, v)) {
			if (t > near_clip && t < inout_best_t)
				inout_best_t = t;
		}
	}

	virtual Dictionary get_data() const override {
		Dictionary dict;
		dict["depth"] = get_depth_buffer();
		dict["resolution"] = resolution;
		dict["fov_y"] = fov_y;
		dict["near"] = near_clip;
		dict["far"] = far_clip;
		return dict;
	}

	virtual void reset() override {
		BaseSensor::reset();
		depth_buffer.clear();
	}

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_resolution", "w", "h"), &DepthCameraSensor::set_resolution);
		ClassDB::bind_method(D_METHOD("get_resolution"), &DepthCameraSensor::get_resolution);
		ClassDB::bind_method(D_METHOD("set_fov_y", "degrees"), &DepthCameraSensor::set_fov_y);
		ClassDB::bind_method(D_METHOD("get_fov_y"), &DepthCameraSensor::get_fov_y);
		ClassDB::bind_method(D_METHOD("set_near_clip", "near"), &DepthCameraSensor::set_near_clip);
		ClassDB::bind_method(D_METHOD("get_near_clip"), &DepthCameraSensor::get_near_clip);
		ClassDB::bind_method(D_METHOD("set_far_clip", "far"), &DepthCameraSensor::set_far_clip);
		ClassDB::bind_method(D_METHOD("get_far_clip"), &DepthCameraSensor::get_far_clip);
		ClassDB::bind_method(D_METHOD("set_noise_sigma", "sigma"), &DepthCameraSensor::set_noise_sigma);
		ClassDB::bind_method(D_METHOD("get_noise_sigma"), &DepthCameraSensor::get_noise_sigma);
		ClassDB::bind_method(D_METHOD("get_depth_buffer"), &DepthCameraSensor::get_depth_buffer);
		ClassDB::bind_method(D_METHOD("set_visible_entities", "entities"), &DepthCameraSensor::set_visible_entities);
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "resolution"), "set_resolution", "get_resolution");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fov_y", PROPERTY_HINT_RANGE, "1,179,0.1"), "set_fov_y", "get_fov_y");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "near_clip"), "set_near_clip", "get_near_clip");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "far_clip"), "set_far_clip", "get_far_clip");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "noise_sigma"), "set_noise_sigma", "get_noise_sigma");
	}

private:
	Vector2i resolution;
	real_t fov_y;
	real_t near_clip;
	real_t far_clip;
	int max_rays_per_frame;
	real_t noise_sigma;
	LocalVector<real_t> depth_buffer; // depth per pixel, far = far_clip
	LocalVector<Ref<BaseEntity>> visible_entities;
};

} // namespace genesis

#endif // GENESIS_SENSORS_DEPTH_CAMERA_SENSOR_H