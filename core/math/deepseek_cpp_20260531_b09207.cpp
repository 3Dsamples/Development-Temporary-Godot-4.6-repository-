//File 0058 : core/math/inertia_tensors.h
//Inertia tensor computation for rigid‑body primitives (sphere, box, cylinder, cone, capsule, convex mesh) with mass, center of mass, and parallel‑axis theorem using SIMD vector_math.
#ifndef CORE_MATH_INERTIA_TENSORS_H
#define CORE_MATH_INERTIA_TENSORS_H

#include "vector_math.h"
#include "matrix_math.h"
#include "math_constants.h"
#include <vector>
#include <cmath>
#include <limits>

namespace SimulationMath {
namespace inertia {

// -----------------------------------------------------------------------------
// 1. Symmetric 3x3 inertia tensor stored as 6 unique components
// -----------------------------------------------------------------------------
struct InertiaTensor3x3 {
    float Ixx, Iyy, Izz;   // moments of inertia
    float Ixy, Ixz, Iyz;   // products of inertia

    InertiaTensor3x3() noexcept : Ixx(0), Iyy(0), Izz(0), Ixy(0), Ixz(0), Iyz(0) {}

    static InertiaTensor3x3 zero() noexcept { return {}; }

    static InertiaTensor3x3 from_diagonal(float Ixx, float Iyy, float Izz) noexcept {
        InertiaTensor3x3 t;
        t.Ixx = Ixx; t.Iyy = Iyy; t.Izz = Izz;
        return t;
    }

    DirectX::XMMATRIX to_matrix() const noexcept {
        DirectX::XMMATRIX m;
        m.r[0] = DirectX::XMVectorSet(Ixx, Ixy, Ixz, 0.0f);
        m.r[1] = DirectX::XMVectorSet(Ixy, Iyy, Iyz, 0.0f);
        m.r[2] = DirectX::XMVectorSet(Ixz, Iyz, Izz, 0.0f);
        m.r[3] = DirectX::XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);
        return m;
    }

    void add(const InertiaTensor3x3& other) noexcept {
        Ixx += other.Ixx; Iyy += other.Iyy; Izz += other.Izz;
        Ixy += other.Ixy; Ixz += other.Ixz; Iyz += other.Iyz;
    }

    // Translate inertia tensor by offset vector t (from old reference point to new)
    void translate(DirectX::FXMVECTOR t, float mass) noexcept {
        float tx = vector_math::get_x(t), ty = vector_math::get_y(t), tz = vector_math::get_z(t);
        float t2 = tx*tx + ty*ty + tz*tz;
        Ixx += mass * (t2 - tx*tx);
        Iyy += mass * (t2 - ty*ty);
        Izz += mass * (t2 - tz*tz);
        Ixy -= mass * tx * ty;
        Ixz -= mass * tx * tz;
        Iyz -= mass * ty * tz;
    }
};

// -----------------------------------------------------------------------------
// 2. Volume of primitives
// -----------------------------------------------------------------------------
inline float sphere_volume(float radius) noexcept {
    return (4.0f / 3.0f) * constants::PIf * radius * radius * radius;
}
inline float box_volume(float hx, float hy, float hz) noexcept {
    return 8.0f * hx * hy * hz;
}
inline float cylinder_volume(float radius, float height) noexcept {
    return constants::PIf * radius * radius * height;
}
inline float cone_volume(float radius, float height) noexcept {
    return (1.0f / 3.0f) * constants::PIf * radius * radius * height;
}

// -----------------------------------------------------------------------------
// 3. Solid sphere
// -----------------------------------------------------------------------------
inline void solid_sphere_inertia(float radius, float density,
                                 float& mass, DirectX::XMVECTOR& com,
                                 InertiaTensor3x3& inertia) noexcept {
    mass = density * sphere_volume(radius);
    com = vector_math::zero();
    float I = (2.0f / 5.0f) * mass * radius * radius;
    inertia = InertiaTensor3x3::from_diagonal(I, I, I);
}

inline void solid_sphere_inertia_mass(float radius, float total_mass,
                                      DirectX::XMVECTOR& com,
                                      InertiaTensor3x3& inertia) noexcept {
    com = vector_math::zero();
    float I = (2.0f / 5.0f) * total_mass * radius * radius;
    inertia = InertiaTensor3x3::from_diagonal(I, I, I);
}

// -----------------------------------------------------------------------------
// 4. Solid box (half‑extents)
// -----------------------------------------------------------------------------
inline void solid_box_inertia(float hx, float hy, float hz, float density,
                               float& mass, DirectX::XMVECTOR& com,
                               InertiaTensor3x3& inertia) noexcept {
    mass = density * box_volume(hx, hy, hz);
    com = vector_math::zero();
    float Ixx = (1.0f / 3.0f) * mass * (hy*hy + hz*hz);
    float Iyy = (1.0f / 3.0f) * mass * (hx*hx + hz*hz);
    float Izz = (1.0f / 3.0f) * mass * (hx*hx + hy*hy);
    inertia = InertiaTensor3x3::from_diagonal(Ixx, Iyy, Izz);
}

inline void solid_box_inertia_mass(float hx, float hy, float hz, float total_mass,
                                    DirectX::XMVECTOR& com,
                                    InertiaTensor3x3& inertia) noexcept {
    com = vector_math::zero();
    float Ixx = (1.0f / 3.0f) * total_mass * (hy*hy + hz*hz);
    float Iyy = (1.0f / 3.0f) * total_mass * (hx*hx + hz*hz);
    float Izz = (1.0f / 3.0f) * total_mass * (hx*hx + hy*hy);
    inertia = InertiaTensor3x3::from_diagonal(Ixx, Iyy, Izz);
}

// -----------------------------------------------------------------------------
// 5. Solid cylinder (aligned with Y axis)
// -----------------------------------------------------------------------------
inline void solid_cylinder_inertia(float radius, float height, float density,
                                   float& mass, DirectX::XMVECTOR& com,
                                   InertiaTensor3x3& inertia) noexcept {
    mass = density * cylinder_volume(radius, height);
    com = vector_math::zero();
    float Iyy = 0.5f * mass * radius * radius;
    float Ixxz = (1.0f / 12.0f) * mass * (3.0f * radius * radius + height * height);
    inertia = InertiaTensor3x3::from_diagonal(Ixxz, Iyy, Ixxz);
}

inline void solid_cylinder_inertia_mass(float radius, float height, float total_mass,
                                         DirectX::XMVECTOR& com,
                                         InertiaTensor3x3& inertia) noexcept {
    com = vector_math::zero();
    float Iyy = 0.5f * total_mass * radius * radius;
    float Ixxz = (1.0f / 12.0f) * total_mass * (3.0f * radius * radius + height * height);
    inertia = InertiaTensor3x3::from_diagonal(Ixxz, Iyy, Ixxz);
}

// -----------------------------------------------------------------------------
// 6. Solid cone (apex at origin, axis along +Y)
// -----------------------------------------------------------------------------
inline void solid_cone_inertia(float radius, float height, float density,
                                float& mass, DirectX::XMVECTOR& com,
                                InertiaTensor3x3& inertia) noexcept {
    mass = density * cone_volume(radius, height);
    float com_y = 0.75f * height;
    com = vector_math::load3(0.0f, com_y, 0.0f);
    float Ixx = (3.0f/80.0f) * mass * (4.0f * radius * radius + height * height);
    float Iyy = (3.0f/10.0f) * mass * radius * radius;
    float Izz = Ixx;
    inertia = InertiaTensor3x3::from_diagonal(Ixx, Iyy, Izz);
}

inline void solid_cone_inertia_mass(float radius, float height, float total_mass,
                                     DirectX::XMVECTOR& com,
                                     InertiaTensor3x3& inertia) noexcept {
    float com_y = 0.75f * height;
    com = vector_math::load3(0.0f, com_y, 0.0f);
    float Ixx = (3.0f/80.0f) * total_mass * (4.0f * radius * radius + height * height);
    float Iyy = (3.0f/10.0f) * total_mass * radius * radius;
    float Izz = Ixx;
    inertia = InertiaTensor3x3::from_diagonal(Ixx, Iyy, Izz);
}

// -----------------------------------------------------------------------------
// 7. Solid capsule (hemispheres + cylinder, height = total length)
// -----------------------------------------------------------------------------
inline void solid_capsule_inertia(float radius, float height, float density,
                                   float& mass, DirectX::XMVECTOR& com,
                                   InertiaTensor3x3& inertia) noexcept {
    float cyl_length = std::max(0.0f, height - 2.0f * radius);
    float vol_cyl = constants::PIf * radius * radius * cyl_length;
    float vol_sph = sphere_volume(radius);
    mass = density * (vol_cyl + vol_sph);
    com = vector_math::zero();

    float m_cyl = density * vol_cyl;
    float m_hemi = density * (vol_sph / 2.0f);
    float Icyl_yy = 0.5f * m_cyl * radius * radius;
    float Icyl_xx = (1.0f/12.0f) * m_cyl * (3.0f * radius * radius + cyl_length * cyl_length);
    InertiaTensor3x3 I_total = InertiaTensor3x3::from_diagonal(Icyl_xx, Icyl_yy, Icyl_xx);

    const float Ihemi_axial  = (2.0f/5.0f) * m_hemi * radius * radius;
    const float Ihemi_transv = (83.0f/320.0f) * m_hemi * radius * radius;
    float top_offs = cyl_length/2.0f + (3.0f/8.0f)*radius;
    InertiaTensor3x3 I_top = InertiaTensor3x3::from_diagonal(Ihemi_transv, Ihemi_axial, Ihemi_transv);
    I_top.translate(vector_math::load3(0, top_offs, 0), m_hemi);
    I_total.add(I_top);
    InertiaTensor3x3 I_bot = InertiaTensor3x3::from_diagonal(Ihemi_transv, Ihemi_axial, Ihemi_transv);
    I_bot.translate(vector_math::load3(0, -top_offs, 0), m_hemi);
    I_total.add(I_bot);
    inertia = I_total;
}

inline void solid_capsule_inertia_mass(float radius, float height, float total_mass,
                                        DirectX::XMVECTOR& com,
                                        InertiaTensor3x3& inertia) noexcept {
    float cyl_length = std::max(0.0f, height - 2.0f * radius);
    float vol_cyl = constants::PIf * radius * radius * cyl_length;
    float vol_sph = sphere_volume(radius);
    float total_vol = vol_cyl + vol_sph;
    if (total_vol == 0.0f) { com = vector_math::zero(); inertia = InertiaTensor3x3::zero(); return; }
    float m_cyl = total_mass * (vol_cyl / total_vol);
    float m_hemi = total_mass * (vol_sph / (2.0f * total_vol));
    com = vector_math::zero();
    float Icyl_yy = 0.5f * m_cyl * radius * radius;
    float Icyl_xx = (1.0f/12.0f) * m_cyl * (3.0f * radius * radius + cyl_length * cyl_length);
    InertiaTensor3x3 I_total = InertiaTensor3x3::from_diagonal(Icyl_xx, Icyl_yy, Icyl_xx);
    const float Ihemi_axial  = (2.0f/5.0f) * m_hemi * radius * radius;
    const float Ihemi_transv = (83.0f/320.0f) * m_hemi * radius * radius;
    float top_offs = cyl_length/2.0f + (3.0f/8.0f)*radius;
    InertiaTensor3x3 I_top = InertiaTensor3x3::from_diagonal(Ihemi_transv, Ihemi_axial, Ihemi_transv);
    I_top.translate(vector_math::load3(0, top_offs, 0), m_hemi);
    I_total.add(I_top);
    InertiaTensor3x3 I_bot = InertiaTensor3x3::from_diagonal(Ihemi_transv, Ihemi_axial, Ihemi_transv);
    I_bot.translate(vector_math::load3(0, -top_offs, 0), m_hemi);
    I_total.add(I_bot);
    inertia = I_total;
}

// -----------------------------------------------------------------------------
// 8. Convex polyhedron (closed triangle mesh) – Eberly's divergence theorem method
// -----------------------------------------------------------------------------
inline void convex_mesh_inertia(const std::vector<DirectX::XMVECTOR>& vertices,
                                const std::vector<uint32_t>& indices,
                                float density,
                                float& mass, DirectX::XMVECTOR& com,
                                InertiaTensor3x3& inertia) noexcept {
    if (vertices.size() < 3 || indices.size() % 3 != 0) {
        mass = 0; com = vector_math::zero(); inertia = InertiaTensor3x3::zero();
        return;
    }

    double integral[10] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    for (size_t i = 0; i < indices.size(); i += 3) {
        DirectX::XMVECTOR v0 = vertices[indices[i]];
        DirectX::XMVECTOR v1 = vertices[indices[i+1]];
        DirectX::XMVECTOR v2 = vertices[indices[i+2]];
        float x0 = vector_math::get_x(v0), y0 = vector_math::get_y(v0), z0 = vector_math::get_z(v0);
        float x1 = vector_math::get_x(v1), y1 = vector_math::get_y(v1), z1 = vector_math::get_z(v1);
        float x2 = vector_math::get_x(v2), y2 = vector_math::get_y(v2), z2 = vector_math::get_z(v2);
        double f1x = x0, f1y = y0, f1z = z0;
        double f2x = x1, f2y = y1, f2z = z1;
        double f3x = x2, f3y = y2, f3z = z2;
        double d = f1x*(f2y*f3z - f2z*f3y) + f1y*(f2z*f3x - f2x*f3z) + f1z*(f2x*f3y - f2y*f3x);

        integral[0] += d;
        integral[1] += d * (f1x + f2x + f3x);
        integral[2] += d * (f1y + f2y + f3y);
        integral[3] += d * (f1z + f2z + f3z);
        integral[4] += d * (f1x*f1x + f1x*f2x + f2x*f2x + f1x*f3x + f2x*f3x + f3x*f3x);
        integral[5] += d * (f1y*f1y + f1y*f2y + f2y*f2y + f1y*f3y + f2y*f3y + f3y*f3y);
        integral[6] += d * (f1z*f1z + f1z*f2z + f2z*f2z + f1z*f3z + f2z*f3z + f3z*f3z);
        integral[7] += d * (2.0*f1x*f1y + f1x*f2y + f2x*f1y + 2.0*f2x*f2y + f1x*f3y + f3x*f1y + f2x*f3y + f3x*f2y + 2.0*f3x*f3y);
        integral[8] += d * (2.0*f1x*f1z + f1x*f2z + f2x*f1z + 2.0*f2x*f2z + f1x*f3z + f3x*f1z + f2x*f3z + f3x*f2z + 2.0*f3x*f3z);
        integral[9] += d * (2.0*f1y*f1z + f1y*f2z + f2y*f1z + 2.0*f2y*f2z + f1y*f3z + f3y*f1z + f2y*f3z + f3y*f2z + 2.0*f3y*f3z);
    }

    const double mult[10] = {
        1.0/6.0,
        1.0/24.0, 1.0/24.0, 1.0/24.0,
        1.0/60.0, 1.0/60.0, 1.0/60.0,
        1.0/120.0, 1.0/120.0, 1.0/120.0
    };

    double volume = integral[0] * mult[0];
    if (volume < 1e-12) {
        mass = 0.0f;
        com = vector_math::zero();
        inertia = InertiaTensor3x3::zero();
        return;
    }
    double inv_vol = 1.0 / volume;

    double cx = integral[1] * mult[1] * inv_vol;
    double cy = integral[2] * mult[2] * inv_vol;
    double cz = integral[3] * mult[3] * inv_vol;
    com = vector_math::load3(static_cast<float>(cx), static_cast<float>(cy), static_cast<float>(cz));

    double Ixx_o = integral[4] * mult[4];
    double Iyy_o = integral[5] * mult[5];
    double Izz_o = integral[6] * mult[6];
    double Ixy_o = integral[7] * mult[7];
    double Ixz_o = integral[8] * mult[8];
    double Iyz_o = integral[9] * mult[9];

    mass = density * static_cast<float>(volume);
    float M = mass;
    Ixx_o -= M * (cy*cy + cz*cz);
    Iyy_o -= M * (cx*cx + cz*cz);
    Izz_o -= M * (cx*cx + cy*cy);
    Ixy_o += M * cx * cy;
    Ixz_o += M * cx * cz;
    Iyz_o += M * cy * cz;

    inertia.Ixx = static_cast<float>(Ixx_o);
    inertia.Iyy = static_cast<float>(Iyy_o);
    inertia.Izz = static_cast<float>(Izz_o);
    inertia.Ixy = static_cast<float>(Ixy_o);
    inertia.Ixz = static_cast<float>(Ixz_o);
    inertia.Iyz = static_cast<float>(Iyz_o);
}

} // namespace inertia
} // namespace SimulationMath

#endif // CORE_MATH_INERTIA_TENSORS_H