// File 132: modules/genesis/src/boundaries/sdf_boundary.h
// Signed distance function (SDF) boundary conditions for all Genesis solvers.
// Primitive SDFs (sphere, box, capsule, cylinder, torus) can be combined via
// union, intersection, and difference operators. The resulting field is used
// to project particles / grid nodes back to the surface and apply friction.

#ifndef GENESIS_BOUNDARIES_SDF_BOUNDARY_H
#define GENESIS_BOUNDARIES_SDF_BOUNDARY_H

#include "../core/genesis_types.h"
#include "../core/genesis_constants.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/templates/local_vector.h"

namespace genesis::boundaries {

/**
 * Base class for a signed distance function.
 * Returns the signed distance (negative = inside) and the outward gradient.
 */
class SDFBase {
public:
	virtual ~SDFBase() {}
	virtual real_t evaluate(const Vector3 &p_world) const = 0;
	virtual Vector3 gradient(const Vector3 &p_world) const {
		// Central finite difference with step proportional to query point size
		real_t eps = MAX(1e-6, p_world.length() * 1e-4);
		real_t dx = evaluate(p_world + Vector3(eps, 0, 0)) - evaluate(p_world - Vector3(eps, 0, 0));
		real_t dy = evaluate(p_world + Vector3(0, eps, 0)) - evaluate(p_world - Vector3(0, eps, 0));
		real_t dz = evaluate(p_world + Vector3(0, 0, eps)) - evaluate(p_world - Vector3(0, 0, eps));
		return Vector3(dx, dy, dz) / (2.0 * eps);
	}
};

// --- Primitive SDFs ---

class SDFSphere : public SDFBase {
public:
	Vector3 center; real_t radius;
	SDFSphere(const Vector3 &c, real_t r) : center(c), radius(r) {}
	virtual real_t evaluate(const Vector3 &p) const override {
		return p.distance_to(center) - radius;
	}
	virtual Vector3 gradient(const Vector3 &p) const override {
		Vector3 dir = p - center;
		real_t len = dir.length();
		return (len > CMP_EPSILON) ? dir / len : Vector3(0, 1, 0);
	}
};

class SDFBox : public SDFBase {
public:
	Vector3 center, half_extents;
	SDFBox(const Vector3 &c, const Vector3 &he) : center(c), half_extents(he) {}
	virtual real_t evaluate(const Vector3 &p) const override {
		Vector3 q = (p - center).abs() - half_extents;
		return q.max(0.0).length() + MIN(MAX(q.x, MAX(q.y, q.z)), 0.0);
	}
};

class SDFCapsule : public SDFBase {
public:
	Vector3 a, b; real_t radius;
	SDFCapsule(const Vector3 &pa, const Vector3 &pb, real_t r) : a(pa), b(pb), radius(r) {}
	virtual real_t evaluate(const Vector3 &p) const override {
		Vector3 ab = b - a;
		Vector3 ap = p - a;
		real_t t = CLAMP(ap.dot(ab) / MAX(ab.length_squared(), CMP_EPSILON), 0.0, 1.0);
		return p.distance_to(a + ab * t) - radius;
	}
};

class SDFCylinder : public SDFBase {
public:
	Vector3 a, b; real_t radius;
	SDFCylinder(const Vector3 &pa, const Vector3 &pb, real_t r) : a(pa), b(pb), radius(r) {}
	virtual real_t evaluate(const Vector3 &p) const override {
		Vector3 ab = b - a;
		Vector3 ap = p - a;
		real_t t = CLAMP(ap.dot(ab) / MAX(ab.length_squared(), CMP_EPSILON), 0.0, 1.0);
		Vector3 axis_pt = a + ab * t;
		real_t radial_dist = p.distance_to(axis_pt) - radius;
		real_t top_bottom = MAX(ap.dot(ab) - ab.length_squared(), -ap.dot(ab));
		// combine radial and axial distances
		real_t axial = MAX(t * ab.length() - ab.length(), -t * ab.length());
		return MAX(radial_dist, axial);
	}
};

class SDFTorus : public SDFBase {
public:
	Vector3 center; Vector3 normal; real_t major_radius, minor_radius;
	SDFTorus(const Vector3 &c, const Vector3 &n, real_t R, real_t r) : center(c), normal(n.normalized()), major_radius(R), minor_radius(r) {}
	virtual real_t evaluate(const Vector3 &p) const override {
		Vector3 q = p - center;
		real_t axial = q.dot(normal);
		Vector3 radial = q - normal * axial;
		real_t radial_len = radial.length();
		return Math::sqrt((radial_len - major_radius) * (radial_len - major_radius) + axial * axial) - minor_radius;
	}
};

// --- CSG operators ---

class SDFUnion : public SDFBase {
public:
	const SDFBase *a, *b;
	SDFUnion(const SDFBase *pA, const SDFBase *pB) : a(pA), b(pB) {}
	virtual real_t evaluate(const Vector3 &p) const override { return MIN(a->evaluate(p), b->evaluate(p)); }
};

class SDFIntersection : public SDFBase {
public:
	const SDFBase *a, *b;
	SDFIntersection(const SDFBase *pA, const SDFBase *pB) : a(pA), b(pB) {}
	virtual real_t evaluate(const Vector3 &p) const override { return MAX(a->evaluate(p), b->evaluate(p)); }
};

class SDFDifference : public SDFBase {
public:
	const SDFBase *a, *b; // result = a \ b
	SDFDifference(const SDFBase *pA, const SDFBase *pB) : a(pA), b(pB) {}
	virtual real_t evaluate(const Vector3 &p) const override { return MAX(a->evaluate(p), -b->evaluate(p)); }
};

/**
 * SDF boundary that can be attached to a World or solver.
 * Applies penalty forces / position projections to particles and velocities.
 */
class SDFBoundary : public RefCounted {
	GDCLASS(SDFBoundary, RefCounted);

public:
	SDFBoundary() : sdf(nullptr), stiffness(1e4), friction(0.5), restitution(0.0), enabled(true) {}

	void set_sdf(SDFBase *p_sdf) { sdf = p_sdf; }
	SDFBase *get_sdf() const { return sdf; }

	void set_stiffness(real_t p_k) { stiffness = MAX(p_k, 0.0); }
	real_t get_stiffness() const { return stiffness; }

	void set_friction(real_t p_mu) { friction = CLAMP(p_mu, 0.0, 1.0); }
	real_t get_friction() const { return friction; }

	void set_restitution(real_t p_e) { restitution = CLAMP(p_e, 0.0, 1.0); }
	real_t get_restitution() const { return restitution; }

	void set_enabled(bool p_en) { enabled = p_en; }
	bool is_enabled() const { return enabled; }

	// --- Enforce the boundary on a single particle position and velocity ---
	void apply_to_particle(Vector3 &p_pos, Vector3 &p_vel, real_t p_dt = 0.0) const {
		if (!enabled || !sdf) return;
		real_t d = sdf->evaluate(p_pos);
		if (d >= 0.0) return; // outside or on surface, no penetration

		Vector3 n = sdf->gradient(p_pos);
		n.normalize();

		// Push particle out to the surface
		p_pos -= n * d;

		// Velocity correction: reflect normal component and apply friction
		real_t vn = p_vel.dot(n);
		if (vn < 0) {
			Vector3 vt = p_vel - n * vn;
			real_t vt_len = vt.length();
			real_t friction_limit = friction * Math::abs(vn);
			if (vt_len > friction_limit && vt_len > CMP_EPSILON) {
				vt *= friction_limit / vt_len;
			}
			p_vel = vt - vn * restitution * n; // normal restitution
		}
	}

	// --- Enforce on a list of particles (e.g., SPH, MPM) ---
	void apply_to_particles(LocalVector<Vector3> &p_positions,
							LocalVector<Vector3> &p_velocities,
							real_t p_dt = 0.0) const {
		for (int i = 0; i < p_positions.size(); ++i) {
			apply_to_particle(p_positions[i], p_velocities[i], p_dt);
		}
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_sdf", "sdf"), &SDFBoundary::set_sdf);
		ClassDB::bind_method(D_METHOD("get_sdf"), &SDFBoundary::get_sdf);
		ClassDB::bind_method(D_METHOD("set_stiffness", "k"), &SDFBoundary::set_stiffness);
		ClassDB::bind_method(D_METHOD("get_stiffness"), &SDFBoundary::get_stiffness);
		ClassDB::bind_method(D_METHOD("set_friction", "mu"), &SDFBoundary::set_friction);
		ClassDB::bind_method(D_METHOD("get_friction"), &SDFBoundary::get_friction);
		ClassDB::bind_method(D_METHOD("set_restitution", "e"), &SDFBoundary::set_restitution);
		ClassDB::bind_method(D_METHOD("get_restitution"), &SDFBoundary::get_restitution);
		ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &SDFBoundary::set_enabled);
		ClassDB::bind_method(D_METHOD("is_enabled"), &SDFBoundary::is_enabled);
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stiffness"), "set_stiffness", "get_stiffness");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "friction"), "set_friction", "get_friction");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "restitution"), "set_restitution", "get_restitution");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	}

private:
	SDFBase *sdf;       // non‑owning pointer
	real_t stiffness;
	real_t friction;
	real_t restitution;
	bool enabled;
};

} // namespace genesis::boundaries

#endif // GENESIS_BOUNDARIES_SDF_BOUNDARY_H