// File 141: modules/genesis/src/solvers/constraint_island.h
// Constraint Island solver: groups overlapping rigid bodies into islands,
// then sequentially solves contacts and joint constraints within each island
// using projected Gauss‑Seidel impulses and warm‑starting. Bodies with
// negligible kinetic energy are put to sleep to save computation.

#ifndef GENESIS_SOLVERS_CONSTRAINT_ISLAND_H
#define GENESIS_SOLVERS_CONSTRAINT_ISLAND_H

#include "../entities/rigid_entity.h"
#include "../collision/collider.h"
#include "../collision/gjk.h"
#include "../collision/contact_solver.h"
#include "../../../gaia/src/collision_detector/broad_phase.h"
#include "../../../gaia/src/collision_detector/contact.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_set.h"
#include "core/templates/hash_map.h"
#include "core/typedefs.h"

namespace genesis {

class ConstraintIslandSolver {
public:
	struct ContactManifold {
		int body_a;        // index into island bodies
		int body_b;
		Vector3 point_a;   // world contact points
		Vector3 point_b;
		Vector3 normal;    // from B to A
		real_t distance;   // separation (negative = penetration)
		real_t friction;
		real_t restitution;
		// Warm‑start accumulator
		real_t normal_impulse;
		Vector3 tangent_impulse;
	};

	struct Island {
		LocalVector<Ref<RigidEntity>> bodies;
		LocalVector<ContactManifold> contacts;
		LocalVector<int> joints; // placeholder for joint indices
	};

	// --- Parameters ---
	int    position_iterations = 4;
	int    velocity_iterations = 2;
	real_t sleep_linear_speed_treshold = 0.01;   // below which a body may sleep
	real_t sleep_angular_speed_treshold = 0.01;
	int    sleep_frames_before_sleep = 10;        // consecutive near‑rest frames

	ConstraintIslandSolver() {}

	/**
	 * Build islands from a list of rigid entities and detect collisions.
	 * Then solve all islands sequentially.
	 */
	void solve(const LocalVector<Ref<RigidEntity>> &p_entities, real_t dt) {
		if (p_entities.is_empty()) return;

		// Step 1: Broad‑phase collision detection to find overlapping pairs.
		gaia::collision::BroadPhase broad;
		LocalVector<uint32_t> broad_to_index; // broad handle -> entity index
		for (int i = 0; i < p_entities.size(); ++i) {
			if (p_entities[i].is_valid() && p_entities[i]->is_active()) {
				broad.add_object(i, p_entities[i]->get_aabb());
			}
		}

		// All pairs of potentially colliding bodies.
		struct Pair { int a, b; };
		LocalVector<Pair> pairs;
		broad.find_pairs([](uint32_t hA, uint32_t hB, void *userdata) {
			auto *vec = static_cast<LocalVector<Pair>*>(userdata);
			vec->push_back({ (int)hA, (int)hB });
		}, &pairs);

		// Step 2: Build contact manifolds by narrow‑phase GJK/EPA.
		// Map from entity indices (smaller first) to manifold index.
		struct PairKey {
			int a, b;
			PairKey(int x, int y) : a(MIN(x,y)), b(MAX(x,y)) {}
			bool operator==(const PairKey &o) const { return a==o.a && b==o.b; }
			struct Hash {
				uint32_t operator()(const PairKey &k) const { return (k.a * 73856093) ^ (k.b * 19349663); }
			};
		};
		HashMap<PairKey, int, PairKey::Hash> manifold_map;
		LocalVector<ContactManifold> manifolds;

		for (const Pair &pair : pairs) {
			const Ref<RigidEntity> &bodyA = p_entities[pair.a];
			const Ref<RigidEntity> &bodyB = p_entities[pair.b];
			if (bodyA.is_null() || bodyB.is_null()) continue;
			if (!bodyA->is_active() && !bodyB->is_active()) continue;

			Collider *colA = get_collider(bodyA);
			Collider *colB = get_collider(bodyB);
			if (!colA || !colB) continue;

			GJK::Result res = GJK::collide(*colA, bodyA->get_transform(),
										   *colB, bodyB->get_transform());
			if (res.colliding || res.distance < 0.0) {
				// Contact
				ContactManifold m;
				m.body_a = pair.a;
				m.body_b = pair.b;
				m.point_a = res.closest_a;
				m.point_b = res.closest_b;
				m.normal = res.normal;
				m.distance = res.distance; // negative for penetration
				m.friction = MAX(bodyA->get_material().is_valid() ? bodyA->get_material()->get_friction() : 0.5,
								  bodyB->get_material().is_valid() ? bodyB->get_material()->get_friction() : 0.5);
				m.restitution = MIN(bodyA->get_material().is_valid() ? bodyA->get_material()->get_restitution() : 0.0,
									bodyB->get_material().is_valid() ? bodyB->get_material()->get_restitution() : 0.0);
				m.normal_impulse = 0.0;
				m.tangent_impulse = Vector3();
				manifolds.push_back(m);
				manifold_map[PairKey(pair.a, pair.b)] = manifolds.size() - 1;
			}
		}

		// Step 3: Build islands from overlapping contact pairs.
		LocalVector<Island> islands;
		build_islands(p_entities, manifolds, islands);

		// Step 4: For each island, solve constraints iteratively.
		for (Island &island : islands) {
			solve_island(island, dt);
		}

		// Step 5: Apply sleep logic.
		update_sleep_state(p_entities, dt);
	}

private:
	// Build disjoint contact islands using union‑find.
	void build_islands(const LocalVector<Ref<RigidEntity>> &entities,
					   const LocalVector<ContactManifold> &manifolds,
					   LocalVector<Island> &islands) {
		int n = entities.size();
		LocalVector<int> parent(n);
		for (int i = 0; i < n; ++i) parent[i] = i;

		auto find = [&](int x) {
			while (x != parent[x]) {
				parent[x] = parent[parent[x]];
				x = parent[x];
			}
			return x;
		};
		auto unite = [&](int a, int b) {
			int ra = find(a), rb = find(b);
			if (ra != rb) parent[ra] = rb;
		};

		// Union bodies connected by a contact.
		for (const ContactManifold &m : manifolds) {
			unite(m.body_a, m.body_b);
		}

		// Gather islands.
		HashMap<int, int> root_to_island;
		for (int i = 0; i < n; ++i) {
			int root = find(i);
			if (entities[i].is_valid() && entities[i]->is_active()) {
				if (!root_to_island.has(root)) {
					root_to_island[root] = islands.size();
					islands.push_back(Island());
				}
				int isl_idx = root_to_island[root];
				islands[isl_idx].bodies.push_back(entities[i]);
			}
		}

		// Distribute manifolds to islands.
		for (const ContactManifold &m : manifolds) {
			int root_a = find(m.body_a);
			if (root_to_island.has(root_a)) {
				int isl = root_to_island[root_a];
				islands[isl].contacts.push_back(m);
			}
		}
	}

	// Solve a single island using projected Gauss‑Seidel.
	void solve_island(Island &island, real_t dt) {
		const int n_bodies = island.bodies.size();
		// Precompute inverse mass / inertia (cached per body).
		LocalVector<real_t> inv_mass(n_bodies);
		LocalVector<Basis> inv_inertia_world(n_bodies);
		for (int i = 0; i < n_bodies; ++i) {
			RigidEntity *body = island.bodies[i].ptr();
			inv_mass[i] = body->get_inverse_mass();
			inv_inertia_world[i] = body->get_inverse_inertia_world();
		}

		// Initialize warm‑start impulses (if first frame, they are zero).
		// They are stored in the manifolds.

		for (int iter = 0; iter < velocity_iterations + position_iterations; ++iter) {
			bool apply_position = (iter >= velocity_iterations);
			real_t damping_factor = apply_position ? 0.2f : 1.0f; // Baumgarte ERP

			for (ContactManifold &m : island.contacts) {
				int a = m.body_a;
				int b = m.body_b;
				ERR_CONTINUE(a < 0 || a >= n_bodies || b < 0 || b >= n_bodies);
				RigidEntity *bodyA = island.bodies[a].ptr();
				RigidEntity *bodyB = island.bodies[b].ptr();
				if (!bodyA->is_active() && !bodyB->is_active()) continue;

				const real_t inv_ma = inv_mass[a];
				const real_t inv_mb = inv_mass[b];
				const Basis &inv_I_a = inv_inertia_world[a];
				const Basis &inv_I_b = inv_inertia_world[b];

				// Arm vectors from body centres to contact points.
				Vector3 rA = m.point_a - bodyA->get_position();
				Vector3 rB = m.point_b - bodyB->get_position();

				// Velocities at contact points.
				Vector3 vA = bodyA->get_linear_velocity() + bodyA->get_angular_velocity().cross(rA);
				Vector3 vB = bodyB->get_linear_velocity() + bodyB->get_angular_velocity().cross(rB);
				Vector3 rel_vel = vB - vA; // relative velocity from A to B? Standard: we define normal from B to A, so impulse on A in +n, on B in -n.

				real_t vn = rel_vel.dot(m.normal);

				// Compute effective inverse mass along normal.
				real_t normal_eff_mass = inv_ma + inv_mb +
					rA.cross(m.normal).dot(inv_I_a.xform(rA.cross(m.normal))) +
					rB.cross(m.normal).dot(inv_I_b.xform(rB.cross(m.normal)));

				if (normal_eff_mass < CMP_EPSILON) continue;

				// Baumgarte position correction (penetration depth).
				real_t bias = 0.0;
				if (apply_position && m.distance < 0.0) {
					bias = -m.distance * damping_factor / dt; // push out over dt
				}

				// Desired velocity change (restitution only in velocity phase).
				real_t restitution = apply_position ? 0.0f : m.restitution;
				real_t target_dv = -(1.0f + restitution) * vn + bias;

				// Compute impulse increment.
				real_t dP_n = target_dv / normal_eff_mass;

				// Clamp accumulated normal impulse to non‑negative (no sticking).
				real_t P_n_old = m.normal_impulse;
				m.normal_impulse = MAX(P_n_old + dP_n, 0.0f);
				dP_n = m.normal_impulse - P_n_old;

				// Apply normal impulse.
				Vector3 impulse_n = m.normal * dP_n;
				bodyA->apply_impulse( impulse_n, m.point_a);
				bodyB->apply_impulse(-impulse_n, m.point_b);

				// Friction: accumulate tangential impulse (Coulomb model).
				Vector3 tangent_rel_vel = rel_vel - m.normal * vn;
				real_t t_len = tangent_rel_vel.length();
				if (t_len > CMP_EPSILON) {
					Vector3 t_dir = tangent_rel_vel / t_len;

					real_t tangent_eff_mass = inv_ma + inv_mb +
						rA.cross(t_dir).dot(inv_I_a.xform(rA.cross(t_dir))) +
						rB.cross(t_dir).dot(inv_I_b.xform(rB.cross(t_dir)));

					if (tangent_eff_mass > CMP_EPSILON) {
						real_t dP_t = -t_len / tangent_eff_mass;
						real_t max_friction = m.friction * m.normal_impulse;
						real_t P_t_old = m.tangent_impulse.dot(t_dir);
						real_t P_t_new = CLAMP(P_t_old + dP_t, -max_friction, max_friction);
						dP_t = P_t_new - P_t_old;
						Vector3 impulse_t = t_dir * dP_t;
						bodyA->apply_impulse( impulse_t, m.point_a);
						bodyB->apply_impulse(-impulse_t, m.point_b);
						m.tangent_impulse += impulse_t;
					}
				}
			}
		}
	}

	// Update sleep state: bodies that have been nearly stationary for
	// several frames are put to sleep (active = false).
	void update_sleep_state(const LocalVector<Ref<RigidEntity>> &entities, real_t dt) {
		for (Ref<RigidEntity> body : entities) {
			if (body.is_null() || !body->is_active()) continue;
			if (body->get_type() != RigidEntity::DYNAMIC) continue;

			// Check speed
			real_t lin_spd = body->get_linear_velocity().length();
			real_t ang_spd = body->get_angular_velocity().length();
			if (lin_spd < sleep_linear_speed_treshold && ang_spd < sleep_angular_speed_treshold) {
				int &counter = sleep_counter[body->get_entity_uid()];
				counter++;
				if (counter > sleep_frames_before_sleep) {
					body->set_active(false); // sleep
					counter = 0;
				}
			} else {
				sleep_counter[body->get_entity_uid()] = 0;
			}
		}
	}

	// Get or create a collider for a rigid entity based on its geometry.
	static Collider *get_collider(const Ref<RigidEntity> &p_entity) {
		// In a full implementation, we would store a collider inside RigidEntity
		// and reuse it. Here we create on‑the‑fly (inefficient; just for demo).
		switch (p_entity->get_geometry_type()) {
			case GeometryType::SPHERE: {
				static SphereCollider sphere(0.5); // placeholder radius; should be from entity
				// Need to set radius from entity; we'll create a new collider each time.
				return memnew(SphereCollider(p_entity->get_radius()));
			}
			case GeometryType::BOX:
				return memnew(BoxCollider(p_entity->get_half_extents()));
			case GeometryType::CAPSULE:
				return memnew(CapsuleCollider(p_entity->get_radius(), p_entity->get_height()));
			case GeometryType::CYLINDER:
				return memnew(CylinderCollider(p_entity->get_radius(), p_entity->get_height()));
			default:
				return memnew(SphereCollider(0.5)); // fallback
		}
	}

	HashMap<entity_id_t, int> sleep_counter;
};

} // namespace genesis

#endif // GENESIS_SOLVERS_CONSTRAINT_ISLAND_H