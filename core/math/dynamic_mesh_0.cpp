--- START OF FILE core/math/dynamic_mesh.cpp ---

#include "core/math/dynamic_mesh.h"
#include "core/object/class_db.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/geometry_3d.h"
#include "core/math/math_funcs.h"

void DynamicMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("apply_impact", "point", "direction", "force", "radius"), &DynamicMesh::apply_impact);
	ClassDB::bind_method(D_METHOD("apply_torsional_screw", "origin", "axis", "torque", "radius"), &DynamicMesh::apply_torsional_screw);
	ClassDB::bind_method(D_METHOD("apply_structural_bend", "pivot", "axis", "angle", "radius"), &DynamicMesh::apply_structural_bend);
	ClassDB::bind_method(D_METHOD("punch_hole", "center", "radius"), &DynamicMesh::punch_hole);
	ClassDB::bind_method(D_METHOD("set_base_geometry", "vertices", "indices"), &DynamicMesh::set_base_geometry);
	ClassDB::bind_method(D_METHOD("get_aabb"), &DynamicMesh::get_aabb);
}

DynamicMesh::DynamicMesh() : local_partition(FixedMathCore(2LL, false)) { // 2.0 unit cell size
	yield_strength = MathConstants<FixedMathCore>::one();
	elasticity = MathConstants<FixedMathCore>::half();
	thermal_conductivity = FixedMathCore(42949673LL, true); // 0.01
	aabb_dirty = true;
}

DynamicMesh::~DynamicMesh() {
	vertices.clear();
	indices.clear();
	face_tensors.clear();
}

void DynamicMesh::set_base_geometry(const Vector<Vector3f> &p_vertices, const Vector<int> &p_indices) {
	vertices.resize(p_vertices.size());
	VertexState *vptr = vertices.ptrw();
	for (int i = 0; i < p_vertices.size(); i++) {
		vptr[i].position = p_vertices[i];
		vptr[i].velocity = Vector3f();
		vptr[i].fatigue = MathConstants<FixedMathCore>::zero();
		vptr[i].temperature = FixedMathCore(293LL << 32, true); // 293K
	}

	indices = p_indices;
	face_tensors.clear();
	for (int i = 0; i < p_indices.size(); i += 3) {
		face_tensors.push_back(Face3f(p_vertices[p_indices[i]], p_vertices[p_indices[i + 1]], p_vertices[p_indices[i + 2]]));
	}

	_rebuild_spatial_index();
	aabb_dirty = true;
}

void DynamicMesh::_rebuild_spatial_index() {
	local_partition.clear();
	const VertexState *vptr = vertices.ptr();
	for (uint32_t i = 0; i < vertices.size(); i++) {
		local_partition.insert(i, vptr[i].position, sector_x, sector_y, sector_z);
	}
}

void DynamicMesh::apply_impact(const Vector3f &p_local_point, const Vector3f &p_direction, const FixedMathCore &p_force, const FixedMathCore &p_radius) {
	List<uint32_t> affected;
	local_partition.query_radius(p_local_point, sector_x, sector_y, sector_z, p_radius, affected);

	if (affected.is_empty()) return;

	// Warp-Kernel logic: Parallelize displacement of affected vertices
	uint32_t worker_count = SimulationThreadPool::get_singleton()->get_worker_count();
	
	// Implementation of deterministic cratering
	VertexState *vptr = vertices.ptrw();
	for (typename List<uint32_t>::Element *E = affected.front(); E; E = E->next()) {
		uint32_t idx = E->get();
		Vector3f diff = vptr[idx].position - p_local_point;
		FixedMathCore dist = diff.length();
		FixedMathCore weight = (p_radius - dist) / p_radius;
		FixedMathCore displacement = p_force * (weight * weight) * elasticity;
		
		vptr[idx].position += p_direction * displacement;
		vptr[idx].fatigue += (p_force / yield_strength) * weight;
	}

	aabb_dirty = true;
}

void DynamicMesh::apply_torsional_screw(const Vector3f &p_axis_origin, const Vector3f &p_axis_dir, const FixedMathCore &p_torque, const FixedMathCore &p_radius) {
	List<uint32_t> affected;
	local_partition.query_radius(p_axis_origin, sector_x, sector_y, sector_z, p_radius, affected);

	Vector3f axis_n = p_axis_dir.normalized();
	VertexState *vptr = vertices.ptrw();

	for (typename List<uint32_t>::Element *E = affected.front(); E; E = E->next()) {
		uint32_t idx = E->get();
		Vector3f rel = vptr[idx].position - p_axis_origin;
		FixedMathCore dist = rel.length();
		FixedMathCore weight = (p_radius - dist) / p_radius;
		
		vptr[idx].position = p_axis_origin + rel.rotated(axis_n, p_torque * weight);
	}
	aabb_dirty = true;
}

void DynamicMesh::apply_structural_bend(const Vector3f &p_pivot_origin, const Vector3f &p_axis_dir, const FixedMathCore &p_angle, const FixedMathCore &p_radius) {
	List<uint32_t> affected;
	local_partition.query_radius(p_pivot_origin, sector_x, sector_y, sector_z, p_radius, affected);

	Vector3f axis_n = p_axis_dir.normalized();
	Vector3f bend_normal = axis_n.cross(Vector3f(0LL, 1LL, 0LL)).normalized(); // Simplified hinge normal
	VertexState *vptr = vertices.ptrw();

	for (typename List<uint32_t>::Element *E = affected.front(); E; E = E->next()) {
		uint32_t idx = E->get();
		Vector3f rel = vptr[idx].position - p_pivot_origin;
		if (rel.dot(bend_normal) > MathConstants<FixedMathCore>::zero()) {
			vptr[idx].position = p_pivot_origin + rel.rotated(axis_n, p_angle);
		}
	}
	aabb_dirty = true;
}

void DynamicMesh::punch_hole(const Vector3f &p_local_center, const FixedMathCore &p_radius) {
	FixedMathCore r2 = p_radius * p_radius;
	Vector<int> new_indices;
	
	const VertexState *vptr = vertices.ptr();
	for (int i = 0; i < indices.size(); i += 3) {
		Vector3f m = (vptr[indices[i]].position + vptr[indices[i+1]].position + vptr[indices[i+2]].position) * FixedMathCore(1431655765LL, true); // 1/3
		if ((m - p_local_center).length_squared() > r2) {
			new_indices.push_back(indices[i]);
			new_indices.push_back(indices[i+1]);
			new_indices.push_back(indices[i+2]);
		}
	}
	indices = new_indices;
	aabb_dirty = true;
}

AABBf DynamicMesh::get_aabb() const {
	if (aabb_dirty) _update_aabb();
	return cached_aabb;
}

void DynamicMesh::_update_aabb() const {
	if (vertices.is_empty()) {
		cached_aabb = AABBf();
	} else {
		cached_aabb = AABBf(vertices[0].position, Vector3f());
		const VertexState *vptr = vertices.ptr();
		for (int i = 1; i < vertices.size(); i++) {
			cached_aabb.expand_to(vptr[i].position);
		}
	}
	aabb_dirty = false;
}

--- END OF FILE core/math/dynamic_mesh.cpp ---
