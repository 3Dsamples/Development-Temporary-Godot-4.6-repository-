// File 126: modules/genesis/src/sensors/lidar_sensor.h
// LIDAR sensor: simulates a rotating or solid‑state lidar by casting
// multiple rays in a predefined pattern (horizontal and vertical resolution).
// Outputs a packed array of hit points (world coordinates), intensities,
// and a point cloud. Uses Gaia BVH for fast ray‑scene intersections.

#ifndef GENESIS_SENSORS_LIDAR_SENSOR_H
#define GENESIS_SENSORS_LIDAR_SENSOR_H

#include "base_sensor.h"
#include "../entities/base_entity.h"
#include "../entities/rigid_entity.h"
#include "../entities/fem_entity.h"
#include "../entities/mpm_entity.h"
#include "../../../gaia/src/bvh/bvh.h"
#include "../../../gaia/src/bvh/query.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "core/math/transform_3d.h"
#include "core/math/random_number_generator.h"

namespace genesis {

class LidarSensor : public BaseSensor {
	GDCLASS(LidarSensor, BaseSensor);

public:
	LidarSensor() :
		horizontal_resolution(360),     // number of rays per 360° horizontally
		vertical_resolution(16),        // number of vertical channels
		horizontal_fov(360.0),          // degrees
		vertical_fov_min(-15.0),        // degrees (lowest beam)
		vertical_fov_max(15.0),         // degrees (highest beam)
		max_range(100.0),
		min_range(0.1),
		noise_sigma(0.01),              // Gaussian noise on distance
		dropout_probability(0.0)        // fraction of rays that return nothing (0‑1)
	{
		sensor_type = SensorType::LIDAR;
	}

	// --- LIDAR geometry ---
	void set_horizontal_resolution(int p_n) { horizontal_resolution = MAX(p_n, 1); }
	int get_horizontal_resolution() const { return horizontal_resolution; }

	void set_vertical_resolution(int p_n) { vertical_resolution = MAX(p_n, 1); }
	int get_vertical_resolution() const { return vertical_resolution; }

	void set_horizontal_fov(real_t p_deg) { horizontal_fov = CLAMP(p_deg, 1.0, 360.0); }
	real_t get_horizontal_fov() const { return horizontal_fov; }

	void set_vertical_fov_min(real_t p_deg) { vertical_fov_min = CLAMP(p_deg, -90.0, 90.0); }
	real_t get_vertical_fov_min() const { return vertical_fov_min; }

	void set_vertical_fov_max(real_t p_deg) { vertical_fov_max = CLAMP(p_deg, -90.0, 90.0); }
	real_t get_vertical_fov_max() const { return vertical_fov_max; }

	void set_max_range(real_t p_range) { max_range = MAX(p_range, min_range); }
	real_t get_max_range() const { return max_range; }

	void set_min_range(real_t p_range) { min_range = MAX(p_range, 0.0); }
	real_t get_min_range() const { return min_range; }

	void set_noise_sigma(real_t p_sigma) { noise_sigma = MAX(p_sigma, 0.0); }
	real_t get_noise_sigma() const { return noise_sigma; }

	void set_dropout_probability(real_t p_prob) { dropout_probability = CLAMP(p_prob, 0.0, 1.0); }
	real_t get_dropout_probability() const { return dropout_probability; }

	// --- Provide visible entities to the sensor ---
	void set_visible_entities(const LocalVector<Ref<BaseEntity>> &p_entities) {
		visible_entities = p_entities;
	}

	// --- Access recorded point cloud ---
	PackedVector3Array get_point_cloud() const {
		PackedVector3Array pc;
		pc.resize(hit_points.size());
		for (int i = 0; i < hit_points.size(); ++i) pc.set(i, hit_points[i]);
		return pc;
	}

	PackedRealArray get_ranges() const {
		PackedRealArray r;
		r.resize(ranges.size());
		for (int i = 0; i < ranges.size(); ++i) r.set(i, ranges[i]);
		return r;
	}

	// --- Step capture ---
	virtual bool step(real_t p_dt, const Ref<BaseEntity> &p_entity) override {
		if (!BaseSensor::step(p_dt, p_entity)) return false;
		capture(p_entity);
		return true;
	}

protected:
	virtual void capture(const Ref<BaseEntity> &p_entity) override {
		ERR_FAIL_COND(p_entity.is_null());
		Transform3D xform = p_entity->get_transform();
		Vector3 origin = xform.origin;
		Basis R = xform.basis;

		int total_rays = horizontal_resolution * vertical_resolution;
		hit_points.resize(total_rays);
		ranges.resize(total_rays);
		// Default: set all points to (0,0,0) and ranges to max_range (no hit)
		for (int i = 0; i < total_rays; ++i) {
			hit_points[i] = Vector3();
			ranges[i] = max_range;
		}

		// Build BVH of visible entities
		gaia::bvh::BVH bvh;
		int n = visible_entities.size();
		LocalVector<AABB> aabbs;
		LocalVector<int> entity_map; // maps bvh primitive index -> entity index
		if (n > 0) {
			for (int i = 0; i < n; ++i) {
				if (visible_entities[i].is_valid() && visible_entities[i]->is_active()) {
					aabbs.push_back(visible_entities[i]->get_aabb());
					entity_map.push_back(i);
				}
			}
			if (!aabbs.is_empty())
				bvh.build_final(aabbs);
		}

		RandomNumberGenerator rng;
		rng.randomize();

		// For each ray
		real_t h_step = Math::deg_to_rad(horizontal_fov) / real_t(horizontal_resolution);
		real_t v_min_rad = Math::deg_to_rad(vertical_fov_min);
		real_t v_max_rad = Math::deg_to_rad(vertical_fov_max);
		real_t v_step = (v_max_rad - v_min_rad) / real_t(MAX(vertical_resolution - 1, 1));

		for (int v = 0; v < vertical_resolution; ++v) {
			real_t v_angle = v_min_rad + v * v_step;
			real_t cos_v = Math::cos(v_angle);
			real_t sin_v = Math::sin(v_angle);
			for (int h = 0; h < horizontal_resolution; ++h) {
				// Account for dropout before expensive ray casting
				if (dropout_probability > 0.0 && rng.randf() < dropout_probability) continue;

				real_t h_angle = h * h_step;
				real_t cos_h = Math::cos(h_angle);
				real_t sin_h = Math::sin(h_angle);

				// Direction in local LIDAR frame: X forward, Y left, Z up (common convention)
				Vector3 local_dir(cos_v * cos_h, cos_v * sin_h, sin_v);
				Vector3 world_dir = R.xform(local_dir).normalized();

				real_t best_t = max_range;

				// Intersect with BVH
				AABB ray_aabb(origin, Vector3());
				ray_aabb.expand_to(origin + world_dir * max_range);
				// Query BVH to find candidate primitives
				bvh.query_intersect(ray_aabb, [&](int prim_idx) {
					if (prim_idx < 0 || prim_idx >= entity_map.size()) return;
					int ent_idx = entity_map[prim_idx];
					Ref<BaseEntity> ent = visible_entities[ent_idx];
					if (ent.is_null()) return;

					Ref<RigidEntity> rigid = ent;
					Ref<FEMEntity> fem = ent;
					Ref<MPMEntity> mpm = ent;

					if (rigid.is_valid()) {
						// Use collider for exact intersection
						AABB box = rigid->get_aabb();
						real_t t_entry, t_exit;
						if (gaia::bvh::intersect_ray_aabb(origin, world_dir, box, 0.0, best_t, t_entry, t_exit)) {
							best_t = t_entry;
						}
					} else if (fem.is_valid()) {
						const gaia::mesh::TetMesh &mesh = fem->get_mesh();
						int tet_count = mesh.element_count();
						for (int t = 0; t < tet_count; ++t) {
							gaia::mesh::TetMesh::Tetrahedron tet = mesh.get_tetrahedron(t);
							const Vector3 &p0 = mesh.get_vertex(tet.v0);
							const Vector3 &p1 = mesh.get_vertex(tet.v1);
							const Vector3 &p2 = mesh.get_vertex(tet.v2);
							const Vector3 &p3 = mesh.get_vertex(tet.v3);
							// Four faces
							real_t tt, uu, vv;
							if (gaia::bvh::intersect_ray_triangle(origin, world_dir, p0, p1, p2, tt, uu, vv))
								if (tt > min_range && tt < best_t) best_t = tt;
							if (gaia::bvh::intersect_ray_triangle(origin, world_dir, p0, p1, p3, tt, uu, vv))
								if (tt > min_range && tt < best_t) best_t = tt;
							if (gaia::bvh::intersect_ray_triangle(origin, world_dir, p0, p2, p3, tt, uu, vv))
								if (tt > min_range && tt < best_t) best_t = tt;
							if (gaia::bvh::intersect_ray_triangle(origin, world_dir, p1, p2, p3, tt, uu, vv))
								if (tt > min_range && tt < best_t) best_t = tt;
						}
					} else if (mpm.is_valid()) {
						// MPM: test against particle spheres (approximate)
						const LocalVector<MPMEntity::Particle> &particles = mpm->get_particles();
						for (const auto &p : particles) {
							real_t r = mpm->get_cell_size() * 0.3; // approximate particle radius
							// Ray-sphere intersection
							Vector3 oc = origin - p.position;
							real_t b = 2.0 * oc.dot(world_dir);
							real_t c = oc.length_squared() - r * r;
							real_t disc = b * b - 4 * c;
							if (disc >= 0) {
								real_t t0 = (-b - Math::sqrt(disc)) * 0.5;
								if (t0 > min_range && t0 < best_t) best_t = t0;
							}
						}
					}
				});

				// Apply noise
				if (noise_sigma > 0.0 && best_t < max_range) {
					best_t += rng.randfn(0.0, noise_sigma);
					best_t = CLAMP(best_t, min_range, max_range);
				}

				int idx = v * horizontal_resolution + h;
				ranges[idx] = best_t;
				hit_points[idx] = origin + world_dir * best_t;
			}
		}
	}

	virtual Dictionary get_data() const override {
		Dictionary dict;
		dict["point_cloud"] = get_point_cloud();
		dict["ranges"] = get_ranges();
		dict["horizontal_res"] = horizontal_resolution;
		dict["vertical_res"] = vertical_resolution;
		return dict;
	}

	virtual void reset() override {
		BaseSensor::reset();
		hit_points.clear();
		ranges.clear();
	}

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_horizontal_resolution", "n"), &LidarSensor::set_horizontal_resolution);
		ClassDB::bind_method(D_METHOD("get_horizontal_resolution"), &LidarSensor::get_horizontal_resolution);
		ClassDB::bind_method(D_METHOD("set_vertical_resolution", "n"), &LidarSensor::set_vertical_resolution);
		ClassDB::bind_method(D_METHOD("get_vertical_resolution"), &LidarSensor::get_vertical_resolution);
		ClassDB::bind_method(D_METHOD("set_horizontal_fov", "deg"), &LidarSensor::set_horizontal_fov);
		ClassDB::bind_method(D_METHOD("get_horizontal_fov"), &LidarSensor::get_horizontal_fov);
		ClassDB::bind_method(D_METHOD("set_vertical_fov_min", "deg"), &LidarSensor::set_vertical_fov_min);
		ClassDB::bind_method(D_METHOD("get_vertical_fov_min"), &LidarSensor::get_vertical_fov_min);
		ClassDB::bind_method(D_METHOD("set_vertical_fov_max", "deg"), &LidarSensor::set_vertical_fov_max);
		ClassDB::bind_method(D_METHOD("get_vertical_fov_max"), &LidarSensor::get_vertical_fov_max);
		ClassDB::bind_method(D_METHOD("set_max_range", "range"), &LidarSensor::set_max_range);
		ClassDB::bind_method(D_METHOD("get_max_range"), &LidarSensor::get_max_range);
		ClassDB::bind_method(D_METHOD("set_min_range", "range"), &LidarSensor::set_min_range);
		ClassDB::bind_method(D_METHOD("get_min_range"), &LidarSensor::get_min_range);
		ClassDB::bind_method(D_METHOD("set_noise_sigma", "sigma"), &LidarSensor::set_noise_sigma);
		ClassDB::bind_method(D_METHOD("get_noise_sigma"), &LidarSensor::get_noise_sigma);
		ClassDB::bind_method(D_METHOD("set_dropout_probability", "prob"), &LidarSensor::set_dropout_probability);
		ClassDB::bind_method(D_METHOD("get_dropout_probability"), &LidarSensor::get_dropout_probability);
		ClassDB::bind_method(D_METHOD("set_visible_entities", "entities"), &LidarSensor::set_visible_entities);
		ClassDB::bind_method(D_METHOD("get_point_cloud"), &LidarSensor::get_point_cloud);
		ClassDB::bind_method(D_METHOD("get_ranges"), &LidarSensor::get_ranges);
		ADD_PROPERTY(PropertyInfo(Variant::INT, "horizontal_resolution"), "set_horizontal_resolution", "get_horizontal_resolution");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "vertical_resolution"), "set_vertical_resolution", "get_vertical_resolution");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "horizontal_fov"), "set_horizontal_fov", "get_horizontal_fov");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "vertical_fov_min"), "set_vertical_fov_min", "get_vertical_fov_min");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "vertical_fov_max"), "set_vertical_fov_max", "get_vertical_fov_max");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_range"), "set_max_range", "get_max_range");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_range"), "set_min_range", "get_min_range");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "noise_sigma"), "set_noise_sigma", "get_noise_sigma");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dropout_probability", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_dropout_probability", "get_dropout_probability");
	}

private:
	int horizontal_resolution;
	int vertical_resolution;
	real_t horizontal_fov;
	real_t vertical_fov_min, vertical_fov_max;
	real_t max_range;
	real_t min_range;
	real_t noise_sigma;
	real_t dropout_probability;

	LocalVector<Vector3> hit_points;
	LocalVector<real_t> ranges;
	LocalVector<Ref<BaseEntity>> visible_entities;
};

} // namespace genesis

#endif // GENESIS_SENSORS_LIDAR_SENSOR_H