//File group name : OrthoTree Math
//File 0087 : core/math/math.h
//Master include file for the entire OrthoTree math library.
//Includes all core/math sub‑modules and provides a unified interface.
//Also initialises the global math environment (epsilon, SIMD, etc.).

#ifndef ORTHOTREE_CORE_MATH_MATH_H_INCLUDED
#define ORTHOTREE_CORE_MATH_MATH_H_INCLUDED

// Basic components
#include "basic/scalar.h"
#include "basic/vector.h"
#include "basic/matrix.h"
#include "basic/quaternion.h"
#include "basic/transform.h"

// Geometry primitives
#include "geometry/aabb.h"
#include "geometry/ray.h"
#include "geometry/plane.h"
#include "geometry/sphere.h"
#include "geometry/capsule.h"
#include "geometry/cylinder.h"
#include "geometry/triangle.h"
#include "geometry/line_segment.h"
#include "geometry/polygon.h"
#include "geometry/obb.h"
#include "geometry/frustum.h"
#include "geometry/bezier.h"
#include "geometry/bspline.h"
#include "geometry/voxel_grid.h"
#include "geometry/marching_cubes.h"
#include "geometry/dual_contouring.h"

// Distance metrics
#include "distance/metrics.h"

// Interval arithmetic
#include "interval/interval.h"
#include "interval/affine.h"

// Numerical methods
#include "numerical/root_finding.h"
#include "numerical/integration.h"
#include "numerical/matrix_decomposition.h"

// Robust predicates
#include "robust/orientation.h"

// Approximations
#include "approx/fast_math.h"
#include "approx_knn.h"
#include "approx_radius.h"

// Statistics
#include "stats/basic_stats.h"

// Random sampling
#include "random/sampling.h"

// Point cloud processing
#include "point_cloud_processing.h"
#include "poisson_reconstruction.h"
#include "chamfer_distance.h"

// Spatial indices
#include "kdtree.h"
#include "rtree.h"
#include "morton.h"
#include "moving_objects.h"
#include "fast_marching.h"

// ============================================================================
//  Global math environment singleton (access all settings)
// ============================================================================
namespace OrthoTree {
namespace Math {

class GlobalMathEnvironment {
public:
    static GlobalMathEnvironment& instance() {
        static GlobalMathEnvironment env;
        return env;
    }

    // Default epsilon for all comparisons (used by nearlyEqual functions)
    void setDefaultEpsilon(T eps) { MathConfig::instance().setDefaultEpsilon(eps); }
    T defaultEpsilon() const { return MathConfig::instance().defaultEpsilon(); }

    // Global SIMD enable/disable
    void setUseSIMD(bool use) { MathConfig::instance().setUseSIMD(use); }
    bool useSIMD() const { return MathConfig::instance().useSIMD(); }

    // Fast math mode (accurate/approximate)
    void setFastMathMode(Approx::FastMathMode mode) { Approx::FastMathEnvironment::instance().setMode(mode); }
    Approx::FastMathMode fastMathMode() const { return Approx::FastMathEnvironment::instance().mode(); }

    // Approximate k‑NN default epsilon
    void setApproxKNNEpsilon(T eps) { ApproxKNNEnvironment::instance().setDefaultEpsilon(eps); }
    T approxKNNEpsilon() const { return ApproxKNNEnvironment::instance().defaultEpsilon(); }

    // Approx radius search default epsilon
    void setApproxRadiusEpsilon(T eps) { ApproxRadiusEnvironment::instance().setDefaultEpsilon(eps); }
    T approxRadiusEpsilon() const { return ApproxRadiusEnvironment::instance().defaultEpsilon(); }

    // R‑tree default max entries
    void setRTreeMaxEntries(size_t m) { RTreeEnvironment::instance().setDefaultMaxEntries(m); }
    size_t rtreeMaxEntries() const { return RTreeEnvironment::instance().defaultMaxEntries(); }

    // Kd‑tree default max leaf size
    void setKdTreeMaxLeafSize(size_t sz) { KdTreeEnvironment::instance().setDefaultMaxLeafSize(sz); }
    size_t kdTreeMaxLeafSize() const { return KdTreeEnvironment::instance().defaultMaxLeafSize(); }

    // Point cloud processing default radius
    void setPointCloudSearchRadius(T rad) { PointCloudProcessingEnvironment::instance().setDefaultSearchRadius(rad); }
    T pointCloudSearchRadius() const { return PointCloudProcessingEnvironment::instance().defaultSearchRadius(); }

    // Morton code bits per coordinate
    void setMortonBitsPerCoord(uint32_t bits) { MortonEnvironment::instance().setDefaultBitsPerCoord(bits); }
    uint32_t mortonBitsPerCoord() const { return MortonEnvironment::instance().defaultBitsPerCoord(); }

    // Reset all to defaults
    void resetToDefaults() {
        setDefaultEpsilon(T(1e-8));
        setUseSIMD(true);
        setFastMathMode(Approx::FastMathMode::Approximate);
        setApproxKNNEpsilon(T(0.1));
        setApproxRadiusEpsilon(T(0.1));
        setRTreeMaxEntries(8);
        setKdTreeMaxLeafSize(16);
        setPointCloudSearchRadius(T(0.1));
        setMortonBitsPerCoord(21);
    }

private:
    GlobalMathEnvironment() { resetToDefaults(); }
};

} // namespace Math
} // namespace OrthoTree

// ----------------------------------------------------------------------------
//  Macro to quickly set global epsilon for math functions
// ----------------------------------------------------------------------------
#define ORTHOTREE_MATH_EPS (::OrthoTree::Math::GlobalMathEnvironment::instance().defaultEpsilon())

#endif // ORTHOTREE_CORE_MATH_MATH_H_INCLUDED