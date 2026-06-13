// File 362: modules/gaia/src/vbd_physics/vbd_mass_spring.h
// Mass‑spring / XPBD constraint energy for VBD block descent.
// Implements distance, bending, and volume constraints as differentiable
// energy potentials suitable for VBD's block‑coordinate minimisation.
// Rewritten from Gaia's VBD_MassSpring.h for Godot 4.6.

#ifndef GAIA_VBD_MASS_SPRING_H
#define GAIA_VBD_MASS_SPRING_H

#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/typedefs.h"
#include "../mesh/tet_mesh.h"
#include "../mesh/tri_mesh.h"

namespace gaia::vbd {

// ---------------------------------------------------------------------------
// Distance spring energy between two particles.
// E = 0.5 * k * (|x1-x0| - L0)²
// ---------------------------------------------------------------------------
class VBDDistanceSpring {
public:
    VBDDistanceSpring(real_t p_stiffness = 1000.0, real_t p_rest_length = 0.0)
        : stiffness(p_stiffness), rest_length(p_rest_length) {}

    void set_stiffness(real_t k) { stiffness = MAX(k, 0.0); }
    real_t get_stiffness() const { return stiffness; }
    void set_rest_length(real_t L0) { rest_length = MAX(L0, 0.0); }
    real_t get_rest_length() const { return rest_length; }

    // Compute energy for the spring connecting two vertices.
    real_t energy(const Vector3 &p0, const Vector3 &p1) const {
        real_t d = p0.distance_to(p1);
        real_t diff = d - rest_length;
        return 0.5 * stiffness * diff * diff;
    }

    // Compute gradient (force) on each endpoint.
    // Force on p0: f0 = k*(d - L0) * (p0-p1)/d
    // Force on p1: f1 = -f0
    void gradient(const Vector3 &p0, const Vector3 &p1,
                  Vector3 &f0, Vector3 &f1) const {
        Vector3 dir = p0 - p1;
        real_t d = dir.length();
        if (d < CMP_EPSILON) { f0 = Vector3(); f1 = Vector3(); return; }
        dir /= d;
        real_t scale = stiffness * (d - rest_length);
        f0 = dir * scale;
        f1 = -f0;
    }

    // Compute local Hessian blocks for block‑descent.
    // Returns the 3x3 Hessian for each endpoint and the off‑diagonal block.
    void hessian(const Vector3 &p0, const Vector3 &p1,
                 Basis &H00, Basis &H11, Basis &H01) const {
        Vector3 dir = p0 - p1;
        real_t d = dir.length();
        if (d < CMP_EPSILON) {
            H00 = Basis().scaled(Vector3(stiffness, stiffness, stiffness));
            H11 = H00;
            H01 = -H00;
            return;
        }
        dir /= d;
        real_t scale = stiffness * (d - rest_length) / d;
        // H00 = stiffness * dir * dir^T + scale * (I - dir*dir^T)
        // H11 = H00
        // H01 = -H00
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                real_t delta = (i == j) ? 1.0 : 0.0;
                real_t val = stiffness * dir[i] * dir[j] + scale * (delta - dir[i] * dir[j]);
                H00[i][j] = val;
                H11[i][j] = val;
                H01[i][j] = -val;
            }
        }
    }

private:
    real_t stiffness;
    real_t rest_length;
};

// ---------------------------------------------------------------------------
// Bending spring energy (dihedral angle constraint).
// E = 0.5 * k * (θ - θ0)²   where θ is the dihedral angle between two faces.
// ---------------------------------------------------------------------------
class VBDBendingSpring {
public:
    VBDBendingSpring(real_t p_stiffness = 100.0, real_t p_rest_angle = 0.0)
        : stiffness(p_stiffness), rest_angle(p_rest_angle) {}

    void set_stiffness(real_t k) { stiffness = MAX(k, 0.0); }
    real_t get_stiffness() const { return stiffness; }
    void set_rest_angle(real_t a) { rest_angle = a; }
    real_t get_rest_angle() const { return rest_angle; }

    // Compute energy given four vertices (p0,p1 = shared edge, p2 = opposite in face1, p3 = opposite in face2).
    real_t energy(const Vector3 &p0, const Vector3 &p1,
                  const Vector3 &p2, const Vector3 &p3) const {
        Vector3 n1 = (p1 - p0).cross(p2 - p0);
        Vector3 n2 = (p1 - p0).cross(p3 - p0);
        real_t len1 = n1.length(), len2 = n2.length();
        if (len1 < CMP_EPSILON || len2 < CMP_EPSILON) return 0.0;
        n1 /= len1; n2 /= len2;
        real_t cos_angle = CLAMP(n1.dot(n2), -1.0, 1.0);
        real_t angle = Math::acos(cos_angle);
        return 0.5 * stiffness * (angle - rest_angle) * (angle - rest_angle);
    }

    // Compute the gradient of the energy with respect to the four vertices.
    void gradient(const Vector3 &p0, const Vector3 &p1,
                  const Vector3 &p2, const Vector3 &p3,
                  Vector3 &g0, Vector3 &g1, Vector3 &g2, Vector3 &g3) const {
        Vector3 n1 = (p1 - p0).cross(p2 - p0);
        Vector3 n2 = (p1 - p0).cross(p3 - p0);
        real_t len1 = n1.length(), len2 = n2.length();
        if (len1 < CMP_EPSILON || len2 < CMP_EPSILON) {
            g0 = g1 = g2 = g3 = Vector3();
            return;
        }
        n1 /= len1; n2 /= len2;
        real_t cos_angle = CLAMP(n1.dot(n2), -1.0, 1.0);
        real_t angle = Math::acos(cos_angle);
        real_t diff = angle - rest_angle;
        real_t scale = stiffness * diff;

        // Derivative of angle with respect to each vertex.
        // Using the formula: dθ/dx_i = (cot1 * ... ) see PBD bending.
        // Simplified: we compute the gradient using the formula for the dihedral constraint.
        Vector3 e = p1 - p0;
        real_t len_e = e.length();
        if (len_e < CMP_EPSILON) { g0 = g1 = g2 = g3 = Vector3(); return; }
        e /= len_e;

        // cotangent of angles at p2 and p3 with respect to edge e.
        real_t cot2 = cotangent(p0, p1, p2);
        real_t cot3 = cotangent(p0, p1, p3);
        real_t w = cot2 + cot3;

        // Gradient (simplified from Müller et al.)
        g0 = -scale * (cot2 * n1 + cot3 * n2) / w;
        g1 = -g0;
        g2 =  scale * (cot2 * (p1 - p0).cross(n1) / len1) * 0.5;
        g3 =  scale * (cot3 * (p1 - p0).cross(n2) / len2) * 0.5;
    }

private:
    static real_t cotangent(const Vector3 &a, const Vector3 &b, const Vector3 &v) {
        Vector3 ea = a - v, eb = b - v;
        real_t dot = ea.dot(eb);
        Vector3 cross = ea.cross(eb);
        real_t len_cross = cross.length();
        return (len_cross > CMP_EPSILON) ? dot / len_cross : 0.0;
    }

    real_t stiffness;
    real_t rest_angle;
};

// ---------------------------------------------------------------------------
// Volume preservation spring (tetrahedron).
// E = 0.5 * k * (V - V0)²   where V is the signed volume.
// ---------------------------------------------------------------------------
class VBDVolumeSpring {
public:
    VBDVolumeSpring(real_t p_stiffness = 1e6, real_t p_rest_volume = 0.0)
        : stiffness(p_stiffness), rest_volume(p_rest_volume) {}

    void set_stiffness(real_t k) { stiffness = MAX(k, 0.0); }
    real_t get_stiffness() const { return stiffness; }
    void set_rest_volume(real_t V0) { rest_volume = V0; }
    real_t get_rest_volume() const { return rest_volume; }

    // Volume of tetrahedron: V = (1/6) * ( (p1-p0) x (p2-p0) ) · (p3-p0)
    static real_t compute_volume(const Vector3 &p0, const Vector3 &p1,
                                 const Vector3 &p2, const Vector3 &p3) {
        return (p1 - p0).cross(p2 - p0).dot(p3 - p0) / 6.0;
    }

    real_t energy(const Vector3 &p0, const Vector3 &p1,
                  const Vector3 &p2, const Vector3 &p3) const {
        real_t V = compute_volume(p0, p1, p2, p3);
        real_t diff = V - rest_volume;
        return 0.5 * stiffness * diff * diff;
    }

    // Gradient (forces on vertices) using dV/dp_i.
    void gradient(const Vector3 &p0, const Vector3 &p1,
                  const Vector3 &p2, const Vector3 &p3,
                  Vector3 &g0, Vector3 &g1, Vector3 &g2, Vector3 &g3) const {
        real_t V = compute_volume(p0, p1, p2, p3);
        real_t diff = V - rest_volume;
        real_t scale = stiffness * diff / 6.0;
        // dV/dp0 = (p2-p1) x (p3-p1) / 6
        // dV/dp1 = (p2-p0) x (p3-p0) / 6
        // dV/dp2 = (p3-p0) x (p1-p0) / 6
        // dV/dp3 = (p1-p0) x (p2-p0) / 6
        g0 =  scale * (p2 - p1).cross(p3 - p1);
        g1 =  scale * (p2 - p0).cross(p3 - p0);
        g2 =  scale * (p3 - p0).cross(p1 - p0);
        g3 =  scale * (p1 - p0).cross(p2 - p0);
    }

private:
    real_t stiffness;
    real_t rest_volume;
};

} // namespace gaia::vbd

#endif // GAIA_VBD_MASS_SPRING_H