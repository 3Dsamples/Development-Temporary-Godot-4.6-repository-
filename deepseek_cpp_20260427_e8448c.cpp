// File 322: modules/vienna/src/ccd/vienna_ccd.h
// ViennaCCD – continuous collision detection using conservative advancement.
// Detects the time of impact between two convex shapes moving linearly.
// Used to prevent tunneling for fast‑moving objects.

#ifndef VIENNA_CCD_VIENNA_CCD_H
#define VIENNA_CCD_VIENNA_CCD_H

#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"
#include "../collision/vienna_shape.h"
#include "../bodies/vienna_body.h"
#include "../world/vienna_world.h"
#include "../../../gaia/src/collision_detector/narrow_phase.h" // GJK from Gaia
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace vienna {

class ViennaCCD {
public:
    struct CCDResult {
        bool hit = false;
        real_t toi = 1.0;          // time of impact [0,1]
        vec3 point_a;              // contact point on shape A at TOI
        vec3 point_b;              // contact point on shape B at TOI
        vec3 normal;               // from B to A
    };

    // Conservative‑advancement based CCD between two convex shapes.
    static CCDResult compute(const ViennaShape &p_shapeA, const mat4 &p_xformA,
                             const ViennaShape &p_shapeB, const mat4 &p_xformB,
                             const vec3 &p_relVel, real_t p_dt,
                             real_t p_maxDist = 0.01) {
        CCDResult result;
        // Relative velocity
        vec3 relVel = p_relVel;
        real_t speed = relVel.length();
        if (speed < CMP_EPSILON) {
            // Static test
            gaia::collision::GJK::Result res = gaia::collision::GJK::collide(p_shapeA, p_xformA, p_shapeB, p_xformB);
            if (res.colliding || res.distance < 0.0) {
                result.hit = true;
                result.toi = 0.0;
                result.point_a = res.closest_a;
                result.point_b = res.closest_b;
                result.normal = res.normal;
            }
            return result;
        }

        vec3 dir = relVel / speed;
        // Time variable
        real_t t = 0.0;
        mat4 xA = p_xformA;
        mat4 xB = p_xformB;
        int maxIter = 50;
        while (maxIter-- > 0) {
            gaia::collision::GJK::Result res = gaia::collision::GJK::collide(p_shapeA, xA, p_shapeB, xB);
            if (res.colliding || res.distance < 0.0) {
                result.hit = true;
                result.toi = CLAMP(t, 0.0, 1.0);
                result.point_a = res.closest_a;
                result.point_b = res.closest_b;
                result.normal = res.normal;
                return result;
            }
            real_t dist = res.distance;
            if (dist < p_maxDist) {
                result.hit = true;
                result.toi = CLAMP(t, 0.0, 1.0);
                result.point_a = res.closest_a;
                result.point_b = res.closest_b;
                result.normal = (p_shapeB.get_support(res.normal, xB) -
                                  p_shapeA.get_support(-res.normal, xA)).normalized();
                return result;
            }
            // Advance time by the safe distance
            real_t step = (dist - p_maxDist) / speed;
            t += step;
            if (t > 1.0) break; // no impact within the time step
            // Move shapes
            xA.origin += relVel * step; // A moves with relVel relative to B? RelVel is difference. If we treat A as moving with relVel and B stationary.
            // Actually we need to compute separate motions; but CCD typically moves A by vA*dt, B by vB*dt. The relative motion is relVel.
            // Assume A moves with relVel, B is stationary.
        }
        result.toi = 1.0;
        return result;
    }

    // CCD between two bodies using their current velocities and transforms.
    // Returns the earliest time of impact between the two.
    static CCDResult compute_bodies(const ViennaBody &p_bodyA, const ViennaBody &p_bodyB, real_t p_dt) {
        const ViennaShape *sA = p_bodyA.get_collision_shape().ptr();
        const ViennaShape *sB = p_bodyB.get_collision_shape().ptr();
        if (!sA || !sB) return CCDResult();

        const mat4 &xA = p_bodyA.get_transform();
        const mat4 &xB = p_bodyB.get_transform();
        vec3 velA = p_bodyA.get_linear_velocity();
        vec3 velB = p_bodyB.get_linear_velocity();
        vec3 relVel = velA - velB;  // relative velocity of A with respect to B

        return compute(*sA, xA, *sB, xB, relVel, p_dt);
    }
};

} // namespace vienna

#endif // VIENNA_CCD_VIENNA_CCD_H