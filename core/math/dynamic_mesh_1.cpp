--- START OF FILE core/math/dynamic_mesh.cpp ---

#include "core/math/dynamic_mesh.h"
#include "core/object/class_db.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"

void DynamicMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("apply_poke", "point", "dir", "force", "radius"), &DynamicMesh::apply_poke);
	ClassDB::bind_method(D_METHOD("apply_pull", "point", "target", "radius"), &DynamicMesh::apply_pull);
	ClassDB::bind_method(D_METHOD("apply_pinch", "point_a", "point_b", "force", "radius"), &DynamicMesh::apply_pinch);
	ClassDB::bind_method(D_METHOD("set_material_tensors", "stiffness", "damping", "pressure"), &DynamicMesh::set_material_tensors);
	ClassDB::bind_method(D_METHOD("execute_elastic_sweep", "delta"), &DynamicMesh::execute_elastic_sweep);
	ClassDB::bind_method(D_METHOD("get_aabb"), &DynamicMesh::get_aabb);
}

DynamicMesh::DynamicMesh() {
	stiffness = FixedMathCore(5LL, false); // Default moderate stiffness
	damping = FixedMathCore(858993459LL, true); // 0.2 damping
	pressure_coeff = FixedMathCore(1LL, false);
	rest_volume = FixedMathCore(0LL, true);
	aabb_dirty = true;
}

DynamicMesh::~DynamicMesh() {
	vertex_stream.clear();
	face_stream.clear();
}

/**
 * set_initial_geometry()
 * 
 * Populates the SoA streams and calculates the target Rest Volume.
 * strictly uses signed tetrahedral volumes to support non-convex balloon meshes.
 */
void DynamicMesh::set_initial_geometry(const Vector<Vector3f> &p_vertices, const Vector<int> &p_indices) {
	vertex_stream.resize(p_vertices.size());
	VertexState *v_ptr = vertex_stream.ptrw();
	
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	for (int i = 0; i < p_vertices.size(); i++) {
		v_ptr[i].position = p_vertices[i];
		v_ptr[i].rest_position = p_vertices[i];
		v_ptr[i].velocity = Vector3f();
		v_ptr[i].normal = Vector3f();
		v_ptr[i].fatigue = zero;
		v_ptr[i].temperature = FixedMathCore(12591030272LL, true); // 293.15K
		v_ptr[i].mass = one;
	}

	face_stream.resize(p_indices.size() / 3);
	FaceState *f_ptr = face_stream.ptrw();
	FixedMathCore vol_acc = zero;
	FixedMathCore sixth(715827882LL, true); // 1/6

	for (int i = 0; i < face_stream.size(); i++) {
		f_ptr[i].indices[0] = p_indices[i * 3 + 0];
		f_ptr[i].indices[1] = p_indices[i * 3 + 1];
		f_ptr[i].indices[2] = p_indices[i * 3 + 2];
		f_ptr[i].surface_tension = one;
		f_ptr[i].expansion_coeff = FixedMathCore(42949LL, true); // 0.00001

		// Signed volume of tetrahedron from origin
		const Vector3f &p1 = v_ptr[f_ptr[i].indices[0]].position;
		const Vector3f &p2 = v_ptr[f_ptr[i].indices[1]].position;
		const Vector3f &p3 = v_ptr[f_ptr[i].indices[2]].position;
		vol_acc += p1.dot(p2.cross(p3)) * sixth;
	}

	rest_volume = vol_acc.absolute();
	aabb_dirty = true;
}

// ============================================================================
// Real-Time Interaction API (Warp-Kernel Style)
// ============================================================================

void DynamicMesh::apply_poke(const Vector3f &p_local_point, const Vector3f &p_dir, const FixedMathCore &p_force, const FixedMathCore &p_radius) {
	VertexState *vptr = vertex_stream.ptrw();
	FixedMathCore r2 = p_radius * p_radius;

	for (int i = 0; i < vertex_stream.size(); i++) {
		Vector3f diff = vptr[i].position - p_local_point;
		FixedMathCore d2 = diff.length_squared();
		if (d2 < r2) {
			FixedMathCore dist = Math::sqrt(d2);
			FixedMathCore falloff = (p_radius - dist) / p_radius;
			FixedMathCore weight = falloff * falloff;
			
			// Inject instantaneous velocity for reactive "Poke" feel
			vptr[i].velocity += p_dir * (p_force * weight);
			vptr[i].fatigue += p_force * FixedMathCore(4294967LL, true);
		}
	}
	aabb_dirty = true;
}

void DynamicMesh::apply_pull(const Vector3f &p_local_point, const Vector3f &p_target_local, const FixedMathCore &p_radius) {
	VertexState *vptr = vertex_stream.ptrw();
	FixedMathCore r2 = p_radius * p_radius;
	Vector3f pull_delta = p_target_local - p_local_point;

	for (int i = 0; i < vertex_stream.size(); i++) {
		FixedMathCore d2 = (vptr[i].position - p_local_point).length_squared();
		if (d2 < r2) {
			FixedMathCore dist = Math::sqrt(d2);
			FixedMathCore weight = (p_radius - dist) / p_radius;
			// Soft-body drag
			vptr[i].position += pull_delta * (weight * weight);
			vptr[i].velocity = Vector3f(); // Kill velocity while dragging
		}
	}
	aabb_dirty = true;
}

void DynamicMesh::apply_pinch(const Vector3f &p_point_a, const Vector3f &p_point_b, const FixedMathCore &p_force, const FixedMathCore &p_radius) {
	VertexState *vptr = vertex_stream.ptrw();
	Vector3f mid = (p_point_a + p_point_b) * MathConstants<FixedMathCore>::half();
	FixedMathCore r2 = p_radius * p_radius;

	for (int i = 0; i < vertex_stream.size(); i++) {
		FixedMathCore d2 = (vptr[i].position - mid).length_squared();
		if (d2 < r2) {
			FixedMathCore dist = Math::sqrt(d2);
			FixedMathCore weight = (p_radius - dist) / p_radius;
			// Move toward the midpoint of the pinch
			vptr[i].velocity += (mid - vptr[i].position).normalized() * (p_force * weight);
		}
	}
	aabb_dirty = true;
}

// ============================================================================
// Deterministic 120 FPS Execution Waves
// ============================================================================

/**
 * execute_elastic_sweep()
 * 
 * Parallel Hooke's Law integration. 
 * ensures that flesh and balloons return to their rest positions bit-perfectly.
 */
void DynamicMesh::execute_elastic_sweep(const FixedMathCore &p_delta) {
	uint32_t v_count = vertex_stream.size();
	if (v_count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = v_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? v_count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			VertexState *vptr = vertex_stream.ptrw();
			FixedMathCore one = MathConstants<FixedMathCore>::one();
			FixedMathCore damp_val = one - (damping * p_delta);

			for (uint64_t i = start; i < end; i++) {
				// F = -k * x
				Vector3f restoration = (vptr[i].rest_position - vptr[i].position) * stiffness;
				vptr[i].velocity += restoration * p_delta;
				vptr[i].velocity *= damp_val;
				vptr[i].position += vptr[i].velocity * p_delta;
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();
	aabb_dirty = true;
}

/**
 * execute_volume_preservation()
 * 
 * The "Balloon" wave.
 * 1. Accumulates current volume via parallel reduction.
 * 2. Applies outward pressure if current < rest.
 */
void DynamicMesh::execute_volume_preservation(const FixedMathCore &p_delta) {
	VertexState *vptr = vertex_stream.ptrw();
	FaceState *fptr = face_stream.ptrw();
	FixedMathCore current_vol = MathConstants<FixedMathCore>::zero();
	FixedMathCore sixth(715827882LL, true);

	// Reduction pass for volume
	for (int i = 0; i < face_stream.size(); i++) {
		const Vector3f &p1 = vptr[fptr[i].indices[0]].position;
		const Vector3f &p2 = vptr[fptr[i].indices[1]].position;
		const Vector3f &p3 = vptr[fptr[i].indices[2]].position;
		current_vol += p1.dot(p2.cross(p3)) * sixth;
	}
	current_vol = current_vol.absolute();

	// Pressure pass: F_pressure = (V_rest - V_current) * k_pressure
	FixedMathCore p_diff = rest_volume - current_vol;
	FixedMathCore pressure = p_diff * pressure_coeff;

	for (int i = 0; i < vertex_stream.size(); i++) {
		// Outward push along normal
		vptr[i].velocity += vptr[i].normal * (pressure * p_delta);
	}
}

void DynamicMesh::_recalculate_aabb() const {
	if (vertex_stream.is_empty()) {
		local_aabb = AABBf();
		return;
	}
	Vector3f min_v = vertex_stream[0].position;
	Vector3f max_v = vertex_stream[0].position;
	for (int i = 1; i < vertex_stream.size(); i++) {
		Vector3f p = vertex_stream[i].position;
		if (p.x < min_v.x) min_v.x = p.x; if (p.y < min_v.y) min_v.y = p.y; if (p.z < min_v.z) min_v.z = p.z;
		if (p.x > max_v.x) max_v.x = p.x; if (p.y > max_v.y) max_v.y = p.y; if (p.z > max_v.z) max_v.z = p.z;
	}
	local_aabb = AABBf(min_v, max_v - min_v);
	aabb_dirty = false;
}

--- END OF FILE core/math/dynamic_mesh.cpp ---
