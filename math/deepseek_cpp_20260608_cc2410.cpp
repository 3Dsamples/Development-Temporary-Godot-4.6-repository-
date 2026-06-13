// File 329: modules/wicked/src/world/wicked_world.h
// WickedWorld – main simulation world that owns all bodies, joints, materials,
// vehicles, and manages the physics step. Uses a dynamic AABB tree (DBVT)
// broad‑phase or optionally Gaia's BVH, and a sequential‑impulse solver with
// warm‑starting, friction, restitution, and island‑based parallelism.

#ifndef WICKED_WORLD_WICKED_WORLD_H
#define WICKED_WORLD_WICKED_WORLD_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_set.h"
#include "../core/wicked_types.h"
#include "../core/wicked_constants.h"

// Gaia broad‑phase (optional integration)
#include "../../../gaia/src/collision_detector/broad_phase.h"

namespace wicked {

// Forward declarations
class WickedBody;
class WickedJoint;
class WickedMaterial;
class WickedSolver;
class WickedIsland;
class WickedRaycastVehicle;
class WickedShape;
class WickedContactPoint;

class WickedWorld : public RefCounted {
	GDCLASS(WickedWorld, RefCounted);

public:
	WickedWorld();
	virtual ~WickedWorld();

	// -----------------------------------------------------------------------
	// Time step
	// -----------------------------------------------------------------------
	void step(real_t p_dt);
	real_t get_time() const { return world_time; }

	// -----------------------------------------------------------------------
	// Gravity
	// -----------------------------------------------------------------------
	void set_gravity(const vec3 &p_gravity);
	vec3 get_gravity() const { return gravity; }

	// -----------------------------------------------------------------------
	// Solver settings
	// -----------------------------------------------------------------------
	void set_solver_iterations(int p_iter);
	int get_solver_iterations() const { return solver_iterations; }
	void set_solver_method(SolverMethod p_method);
	SolverMethod get_solver_method() const { return solver_method; }
	void set_erp(real_t p_erp) { erp = CLAMP(p_erp, 0.0, 1.0); }
	real_t get_erp() const { return erp; }
	void set_erp2(real_t p_erp2) { erp2 = CLAMP(p_erp2, 0.0, 1.0); }
	real_t get_erp2() const { return erp2; }
	void set_cfm(real_t p_cfm) { cfm = MAX(p_cfm, 0.0); }
	real_t get_cfm() const { return cfm; }

	// -----------------------------------------------------------------------
	// Sleep deactivation
	// -----------------------------------------------------------------------
	void set_sleep_linear_threshold(real_t p_thres);
	void set_sleep_angular_threshold(real_t p_thres);
	void set_sleep_frames(int p_frames);
	real_t get_sleep_linear_threshold() const { return sleep_linear_threshold; }
	real_t get_sleep_angular_threshold() const { return sleep_angular_threshold; }
	int get_sleep_frames() const { return sleep_frames; }

	// -----------------------------------------------------------------------
	// Body management
	// -----------------------------------------------------------------------
	body_id create_body(const Ref<WickedBody> &p_body);
	void destroy_body(body_id p_id);
	Ref<WickedBody> get_body(body_id p_id) const;
	int get_body_count() const;
	LocalVector<body_id> get_body_ids() const;

	// Add a body to the world with an already assigned ID (for loading).
	void add_body_with_id(body_id p_id, const Ref<WickedBody> &p_body);

	// -----------------------------------------------------------------------
	// Joint management
	// -----------------------------------------------------------------------
	joint_id create_joint(const Ref<WickedJoint> &p_joint);
	void destroy_joint(joint_id p_id);
	Ref<WickedJoint> get_joint(joint_id p_id) const;
	LocalVector<joint_id> get_joint_ids() const;

	// -----------------------------------------------------------------------
	// Material management
	// -----------------------------------------------------------------------
	material_id create_material(const Ref<WickedMaterial> &p_material);
	void destroy_material(material_id p_id);
	Ref<WickedMaterial> get_material(material_id p_id) const;
	LocalVector<material_id> get_material_ids() const;

	// -----------------------------------------------------------------------
	// Vehicle management
	// -----------------------------------------------------------------------
	vehicle_id create_vehicle(const Ref<WickedRaycastVehicle> &p_vehicle);
	void destroy_vehicle(vehicle_id p_id);
	Ref<WickedRaycastVehicle> get_vehicle(vehicle_id p_id) const;

	// -----------------------------------------------------------------------
	// Collision filtering (layer / mask style)
	// -----------------------------------------------------------------------
	void set_body_layer(body_id p_body, uint32_t p_layer);
	uint32_t get_body_layer(body_id p_body) const;
	void set_body_mask(body_id p_body, uint32_t p_mask);
	uint32_t get_body_mask(body_id p_body) const;

	// -----------------------------------------------------------------------
	// CCD (continuous collision detection)
	// -----------------------------------------------------------------------
	void set_ccd_enabled(body_id p_body, bool p_enabled);
	bool is_ccd_enabled(body_id p_body) const;

	// -----------------------------------------------------------------------
	// Debug / profiling
	// -----------------------------------------------------------------------
	void set_debug_draw_enabled(bool p_enabled) { debug_draw = p_enabled; }
	bool is_debug_draw_enabled() const { return debug_draw; }

	// Contact pair access for external post‑processing
	const LocalVector<WickedContactPoint> &get_last_contacts() const { return last_contacts; }

protected:
	static void _bind_methods();

private:
	// -----------------------------------------------------------------------
	// Step internals
	// -----------------------------------------------------------------------
	void apply_gravity_and_forces(real_t dt);
	void integrate_velocities(real_t dt);
	void detect_collisions();
	void generate_contacts(const LocalVector<std::pair<body_id, body_id>> &pairs);
	void build_islands();
	void solve_islands(real_t dt);
	void integrate_positions(real_t dt);
	void update_activation_state();
	void step_vehicles(real_t dt);

	// -----------------------------------------------------------------------
	// DBVT (Dynamic Bounding Volume Tree) for broad‑phase
	// -----------------------------------------------------------------------
	struct DbvtNode {
		aabb box;
		int body_index;           // index into active body list, -1 if internal
		int parent;               // parent node index
		int child1;               // left child
		int child2;               // right child
		uint32_t height;          // leaf = 0
	};

	void update_dbvt();
	void dbvt_query_overlaps(const aabb &p_box, LocalVector<int> &p_indices) const;
	void dbvt_remove_body(int p_dbvt_index);
	void dbvt_insert_body(int p_body_index);

	// -----------------------------------------------------------------------
	// Broad‑phase interface (can be DBVT or Gaia)
	// -----------------------------------------------------------------------
	gaia::collision::BroadPhase gaia_broad_phase;
	bool use_gaia_bv = true;     // if true, use Gaia's BVH; else internal DBVT

	// -----------------------------------------------------------------------
	// DBVT data (if not using Gaia)
	// -----------------------------------------------------------------------
	LocalVector<DbvtNode> dbvt_nodes;
	int dbvt_root;
	LocalVector<int> dbvt_free_list;

	// -----------------------------------------------------------------------
	// Containers
	// -----------------------------------------------------------------------
	HashMap<body_id, Ref<WickedBody>> bodies;
	HashMap<joint_id, Ref<WickedJoint>> joints;
	HashMap<material_id, Ref<WickedMaterial>> materials;
	HashMap<vehicle_id, Ref<WickedRaycastVehicle>> vehicles;

	// ID counters
	body_id next_body_id = 1;
	joint_id next_joint_id = 1;
	material_id next_material_id = 1;
	vehicle_id next_vehicle_id = 1;

	// -----------------------------------------------------------------------
	// World state
	// -----------------------------------------------------------------------
	vec3 gravity = vec3(0.0, DEFAULT_GRAVITY, 0.0);
	real_t world_time = 0.0;
	int solver_iterations = DEFAULT_SOLVER_ITERATIONS;
	SolverMethod solver_method = SolverMethod::SEQUENTIAL_IMPULSES;
	real_t erp = DEFAULT_ERP;
	real_t erp2 = DEFAULT_ERP2;
	real_t cfm = DEFAULT_TAU;
	real_t sleep_linear_threshold = DEFAULT_SLEEP_LINEAR;
	real_t sleep_angular_threshold = DEFAULT_SLEEP_ANGULAR;
	int sleep_frames = DEFAULT_SLEEP_FRAMES;
	bool debug_draw = false;

	// -----------------------------------------------------------------------
	// Solver and island
	// -----------------------------------------------------------------------
	Ref<WickedSolver> solver;
	Ref<WickedIsland> island_manager;

	// -----------------------------------------------------------------------
	// Per‑body caches for fast step access
	// -----------------------------------------------------------------------
	struct BodyCache {
		body_id id;
		WickedBody *body;
		ActivationState activation;
		int sleep_counter;
		uint32_t layer;
		uint32_t mask;
		bool ccd;
	};
	LocalVector<BodyCache> body_cache;
	HashMap<body_id, int> body_cache_map;   // body ID -> index in cache

	void rebuild_body_cache();

	// -----------------------------------------------------------------------
	// Overlapping pairs (broad‑phase output)
	// -----------------------------------------------------------------------
	LocalVector<std::pair<body_id, body_id>> contact_pairs;

	// -----------------------------------------------------------------------
	// Generated contacts (narrow‑phase output)
	// -----------------------------------------------------------------------
	LocalVector<WickedContactPoint> last_contacts;

	// -----------------------------------------------------------------------
	// Island solving helper
	// -----------------------------------------------------------------------
	void process_islands(real_t dt);
};

} // namespace wicked

#endif // WICKED_WORLD_WICKED_WORLD_H