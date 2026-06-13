//File 0078 : core/math/colorimetry.h (REVISED)
//Comprehensive color science: CIE XYZ/LAB/LUV, chromatic adaptation (Bradford), CIEDE2000, Oklab/OKLCH conversions, perfect tinting, and chromatization with gamut clamping.
#ifndef CORE_MATH_COLORIMETRY_H
#define CORE_MATH_COLORIMETRY_H

#include "vector_math.h"
#include "math_constants.h"
#include <cmath>
#include <algorithm>

namespace SimulationMath {
namespace colorimetry {

// -----------------------------------------------------------------------------
// 1. Standard illuminants (CIE 1931 2°)
// -----------------------------------------------------------------------------
struct Illuminant { float X, Y, Z; };
inline constexpr Illuminant D65 = {0.95047f, 1.00000f, 1.08883f};
inline constexpr Illuminant D50 = {0.96422f, 1.00000f, 0.82521f};
inline constexpr Illuminant D55 = {0.95682f, 1.00000f, 0.92149f};
inline constexpr Illuminant D75 = {0.94972f, 1.00000f, 1.22638f};

// -----------------------------------------------------------------------------
// 2. Linear sRGB ↔ CIE XYZ (D65)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR linear_rgb_to_xyz(DirectX::FXMVECTOR rgb) noexcept {
    float r = vector_math::get_x(rgb), g = vector_math::get_y(rgb), b = vector_math::get_z(rgb);
    return DirectX::XMVectorSet(
        0.4124564f*r + 0.3575761f*g + 0.1804375f*b,
        0.2126729f*r + 0.7151522f*g + 0.0721750f*b,
        0.0193339f*r + 0.1191920f*g + 0.9503041f*b, 1.0f);
}
inline DirectX::XMVECTOR xyz_to_linear_rgb(DirectX::FXMVECTOR xyz) noexcept {
    float X = vector_math::get_x(xyz), Y = vector_math::get_y(xyz), Z = vector_math::get_z(xyz);
    return DirectX::XMVectorSet(
        3.2404542f*X - 1.5371385f*Y - 0.4985314f*Z,
       -0.9692660f*X + 1.8760108f*Y + 0.0415560f*Z,
        0.0556434f*X - 0.2040259f*Y + 1.0572252f*Z, 1.0f);
}

// -----------------------------------------------------------------------------
// 3. CIE XYZ ↔ CIE LAB (D65 white point)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR xyz_to_lab(DirectX::FXMVECTOR xyz, const Illuminant& white = D65) noexcept {
    float X = vector_math::get_x(xyz)/white.X, Y = vector_math::get_y(xyz)/white.Y, Z = vector_math::get_z(xyz)/white.Z;
    const float d = 6.0f/29.0f, d3 = d*d*d;
    auto f = [&](float t) { return (t > d3) ? std::cbrt(t) : t/(3.0f*d*d) + 4.0f/29.0f; };
    float fx = f(X), fy = f(Y), fz = f(Z);
    return DirectX::XMVectorSet(116.0f*fy - 16.0f, 500.0f*(fx - fy), 200.0f*(fy - fz), 1.0f);
}
inline DirectX::XMVECTOR lab_to_xyz(DirectX::FXMVECTOR lab, const Illuminant& white = D65) noexcept {
    float L = vector_math::get_x(lab), a = vector_math::get_y(lab), b = vector_math::get_z(lab);
    float fy = (L + 16.0f)/116.0f, fx = a/500.0f + fy, fz = fy - b/200.0f;
    const float d = 6.0f/29.0f, d3 = d*d*d;
    auto inv_f = [&](float t) { return (t > d) ? t*t*t : 3.0f*d*d*(t - 4.0f/29.0f); };
    return DirectX::XMVectorSet(inv_f(fx)*white.X, inv_f(fy)*white.Y, inv_f(fz)*white.Z, 1.0f);
}

// -----------------------------------------------------------------------------
// 4. CIE XYZ ↔ CIE LUV (D65)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR xyz_to_luv(DirectX::FXMVECTOR xyz, const Illuminant& white = D65) noexcept {
    float X = vector_math::get_x(xyz), Y = vector_math::get_y(xyz), Z = vector_math::get_z(xyz);
    float denom = X + 15.0f*Y + 3.0f*Z;
    if (denom < 1e-12f) return DirectX::XMVectorSet(0,0,0,1);
    float u_prime = (4.0f*X)/denom, v_prime = (9.0f*Y)/denom;
    float un_denom = white.X + 15.0f*white.Y + 3.0f*white.Z;
    float un_prime = (4.0f*white.X)/un_denom, vn_prime = (9.0f*white.Y)/un_denom;
    float Yn = white.Y;
    float L = (Y/Yn > 0.008856f) ? 116.0f*std::cbrt(Y/Yn)-16.0f : 903.3f*(Y/Yn);
    return DirectX::XMVectorSet(L, 13.0f*L*(u_prime - un_prime), 13.0f*L*(v_prime - vn_prime), 1.0f);
}
inline DirectX::XMVECTOR luv_to_xyz(DirectX::FXMVECTOR luv, const Illuminant& white = D65) noexcept {
    float L = vector_math::get_x(luv), u = vector_math::get_y(luv), v = vector_math::get_z(luv);
    float Yn = white.Y;
    float un_denom = white.X + 15.0f*white.Y + 3.0f*white.Z;
    float un_prime = (4.0f*white.X)/un_denom, vn_prime = (9.0f*white.Y)/un_denom;
    float Y = (L > 8.0f) ? Yn*std::pow((L+16.0f)/116.0f, 3.0f) : Yn*L/903.3f;
    float u_prime = (L > 1e-12f) ? u/(13.0f*L) + un_prime : un_prime;
    float v_prime = (L > 1e-12f) ? v/(13.0f*L) + vn_prime : vn_prime;
    float X = Y * (9.0f*u_prime)/(4.0f*v_prime);
    float Z = Y * (12.0f - 3.0f*u_prime - 20.0f*v_prime)/(4.0f*v_prime);
    return DirectX::XMVectorSet(X, Y, Z, 1.0f);
}

// -----------------------------------------------------------------------------
// 5. Chromatic adaptation (Bradford)
// -----------------------------------------------------------------------------
inline std::array<std::array<float,3>,3> bradford_matrix() noexcept { return {{{0.8951f,0.2664f,-0.1614f},{-0.7502f,1.7135f,0.0367f},{0.0389f,-0.0685f,1.0296f}}}; }
inline std::array<std::array<float,3>,3> bradford_matrix_inverse() noexcept { return {{{0.9869929f,-0.1470543f,0.1599627f},{0.4323053f,0.5183603f,0.0492912f},{-0.0085287f,0.0400428f,0.9684867f}}}; }
inline DirectX::XMVECTOR chromatic_adaptation(DirectX::FXMVECTOR xyz, const Illuminant& src, const Illuminant& dst) noexcept {
    auto M = bradford_matrix(), Mi = bradford_matrix_inverse();
    float rho_s = M[0][0]*src.X + M[0][1]*src.Y + M[0][2]*src.Z;
    float gam_s = M[1][0]*src.X + M[1][1]*src.Y + M[1][2]*src.Z;
    float bet_s = M[2][0]*src.X + M[2][1]*src.Y + M[2][2]*src.Z;
    float rho_d = M[0][0]*dst.X + M[0][1]*dst.Y + M[0][2]*dst.Z;
    float gam_d = M[1][0]*dst.X + M[1][1]*dst.Y + M[1][2]*dst.Z;
    float bet_d = M[2][0]*dst.X + M[2][1]*dst.Y + M[2][2]*dst.Z;
    float X = vector_math::get_x(xyz), Y = vector_math::get_y(xyz), Z = vector_math::get_z(xyz);
    float rho = M[0][0]*X + M[0][1]*Y + M[0][2]*Z;
    float gam = M[1][0]*X + M[1][1]*Y + M[1][2]*Z;
    float bet = M[2][0]*X + M[2][1]*Y + M[2][2]*Z;
    rho *= rho_d/rho_s; gam *= gam_d/gam_s; bet *= bet_d/bet_s;
    return DirectX::XMVectorSet(Mi[0][0]*rho + Mi[0][1]*gam + Mi[0][2]*bet,
                                Mi[1][0]*rho + Mi[1][1]*gam + Mi[1][2]*bet,
                                Mi[2][0]*rho + Mi[2][1]*gam + Mi[2][2]*bet, 1.0f);
}

// -----------------------------------------------------------------------------
// 6. CIE76 / CIE94 / CIEDE2000 color differences
// -----------------------------------------------------------------------------
inline float delta_e_76(DirectX::FXMVECTOR lab1, DirectX::FXMVECTOR lab2) noexcept {
    DirectX::XMVECTOR d = DirectX::XMVectorSubtract(lab1, lab2);
    return vector_math::length3_scalar(d);
}
inline float delta_e_94(DirectX::FXMVECTOR lab1, DirectX::FXMVECTOR lab2, float kL=1,float kC=1,float kH=1,bool textiles=false) noexcept {
    float L1=vector_math::get_x(lab1),a1=vector_math::get_y(lab1),b1=vector_math::get_z(lab1);
    float L2=vector_math::get_x(lab2),a2=vector_math::get_y(lab2),b2=vector_math::get_z(lab2);
    float dL=L1-L2, C1=std::sqrt(a1*a1+b1*b1), C2=std::sqrt(a2*a2+b2*b2), dC=C1-C2, da=a1-a2, db=b1-b2;
    float dH_sq=std::max(0.0f, da*da+db*db - dC*dC), dH=std::sqrt(dH_sq);
    float SL=1.0f, SC=1.0f+(textiles?0.048f:0.045f)*C1, SH=1.0f+(textiles?0.014f:0.015f)*C1;
    float t1=dL/(kL*SL), t2=dC/(kC*SC), t3=dH/(kH*SH);
    return std::sqrt(t1*t1 + t2*t2 + t3*t3);
}
inline float delta_e_2000(DirectX::FXMVECTOR lab1, DirectX::FXMVECTOR lab2, float kL=1,float kC=1,float kH=1) noexcept {
    float L1=vector_math::get_x(lab1), a1=vector_math::get_y(lab1), b1=vector_math::get_z(lab1);
    float L2=vector_math::get_x(lab2), a2=vector_math::get_y(lab2), b2=vector_math::get_z(lab2);
    float L_bar=(L1+L2)*0.5f, C1=std::sqrt(a1*a1+b1*b1), C2=std::sqrt(a2*a2+b2*b2), C_bar=(C1+C2)*0.5f;
    float G=0.5f*(1.0f-std::sqrt(std::pow(C_bar,7.0f)/(std::pow(C_bar,7.0f)+std::pow(25.0f,7.0f))));
    float a1p=a1*(1.0f+G), a2p=a2*(1.0f+G), C1p=std::sqrt(a1p*a1p+b1*b1), C2p=std::sqrt(a2p*a2p+b2*b2), C_bar_p=(C1p+C2p)*0.5f;
    float h1p=std::atan2(b1,a1p); if(h1p<0)h1p+=2.0f*constants::PIf;
    float h2p=std::atan2(b2,a2p); if(h2p<0)h2p+=2.0f*constants::PIf;
    float delta_h, H_bar_p;
    if(std::fabs(h1p-h2p)<=constants::PIf){ delta_h=h2p-h1p; H_bar_p=(h1p+h2p)*0.5f; }
    else if(h1p+h2p<2.0f*constants::PIf){ delta_h=h2p-h1p+2.0f*constants::PIf; H_bar_p=(h1p+h2p+2.0f*constants::PIf)*0.5f; }
    else{ delta_h=h2p-h1p-2.0f*constants::PIf; H_bar_p=(h1p+h2p-2.0f*constants::PIf)*0.5f; }
    float T=1.0f-0.17f*std::cos(H_bar_p-0.5235987756f)+0.24f*std::cos(2.0f*H_bar_p)+0.32f*std::cos(3.0f*H_bar_p+0.1047197551f)-0.20f*std::cos(4.0f*H_bar_p-1.0995574288f);
    float dLp=L2-L1, dCp=C2p-C1p, dHp=2.0f*std::sqrt(C1p*C2p)*std::sin(delta_h*0.5f);
    float SL=1.0f+0.015f*(L_bar-50.0f)*(L_bar-50.0f)/std::sqrt(20.0f+(L_bar-50.0f)*(L_bar-50.0f));
    float SC=1.0f+0.045f*C_bar_p, SH=1.0f+0.015f*C_bar_p*T;
    float th=30.0f*constants::PIf/180.0f*std::exp(-std::pow((H_bar_p*180.0f/constants::PIf-275.0f)/25.0f,2.0f));
    float RC=2.0f*std::sqrt(std::pow(C_bar_p,7.0f)/(std::pow(C_bar_p,7.0f)+std::pow(25.0f,7.0f)));
    float RT=-std::sin(2.0f*th)*RC;
    float tL=dLp/(kL*SL), tC=dCp/(kC*SC), tH=dHp/(kH*SH);
    return std::sqrt(tL*tL + tC*tC + tH*tH + RT*tC*tH);
}

// -----------------------------------------------------------------------------
// 7. Yxy ↔ XYZ
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR xyz_to_Yxy(DirectX::FXMVECTOR xyz) noexcept {
    float X=vector_math::get_x(xyz), Y=vector_math::get_y(xyz), Z=vector_math::get_z(xyz);
    float sum=X+Y+Z; if(sum<1e-12f) return DirectX::XMVectorSet(Y,0.3457f,0.3585f,1);
    return DirectX::XMVectorSet(Y, X/sum, Y/sum, 1.0f);
}
inline DirectX::XMVECTOR Yxy_to_xyz(DirectX::FXMVECTOR Yxy) noexcept {
    float Y=vector_math::get_x(Yxy), x=vector_math::get_y(Yxy), y=vector_math::get_z(Yxy);
    if(y<1e-12f) return DirectX::XMVectorZero();
    return DirectX::XMVectorSet(x*Y/y, Y, (1.0f-x-y)*Y/y, 1.0f);
}

// -----------------------------------------------------------------------------
// 8. Color temperature to xy
// -----------------------------------------------------------------------------
inline void temperature_to_xy(float T_kelvin, float& out_x, float& out_y) noexcept {
    float invT=1000.0f/T_kelvin;
    if(T_kelvin<=4000.0f) out_x=0.179910f+0.8776956f*invT-0.2343589f*invT*invT-0.2661239f*invT*invT*invT;
    else if(T_kelvin<=25000.0f) out_x=0.240390f+0.2226347f*invT+2.1070379f*invT*invT-3.0258469f*invT*invT*invT;
    else out_x=0.25222f;
    if(T_kelvin<=2222.0f) out_y=-0.2661239f*out_x*out_x-0.2343589f*out_x+0.8776956f;
    else if(T_kelvin<=4000.0f) out_y=-0.4654496f*out_x*out_x+1.5294400f*out_x-0.7113025f;
    else out_y=-3.0000000f*out_x*out_x+2.8700000f*out_x-0.2750000f;
}

} // namespace colorimetry

// ============================================================================
// Perceptual Color Engine (Oklab/OKLCH) – fully integrated
// ============================================================================
namespace perceptual_color {

    struct Vector4 { float r,g,b,a; Vector4():r(0),g(0),b(0),a(1){} Vector4(float _r,float _g,float _b,float _a):r(_r),g(_g),b(_b),a(_a){} };
    struct ColorOklab { float L,a,b; };
    struct ColorOKLCH { float L,C,H; };

    class PerceptualColorEngine {
    public:
        static ColorOklab  LinearRGBToOklab(Vector4 rgb);
        static Vector4     OklabToLinearRGB(ColorOklab lab, float alpha);
        static ColorOKLCH  OklabToOKLCH(ColorOklab lab);
        static ColorOklab  OKLCHToOklab(ColorOKLCH lch);
        static Vector4 ApplyPerfectTint(Vector4 baseTexture, Vector4 tintColor, float factor);
        static Vector4 ApplyChromatization(Vector4 color, float scale);
    private:
        static inline float Clamp01(float v) { return std::max(0.0f, std::min(1.0f, v)); }
    };

    // Implementation
    inline ColorOklab PerceptualColorEngine::LinearRGBToOklab(Vector4 rgb) {
        float l = 0.4122214708f*rgb.r + 0.5363325363f*rgb.g + 0.0514459929f*rgb.b;
        float m = 0.2119034982f*rgb.r + 0.6806995451f*rgb.g + 0.1073969566f*rgb.b;
        float s = 0.0883024619f*rgb.r + 0.2817188376f*rgb.g + 0.6299787005f*rgb.b;
        float lp = std::cbrtf(l), mp = std::cbrtf(m), sp = std::cbrtf(s);
        ColorOklab res;
        res.L = 0.2104542553f*lp + 0.7936177850f*mp - 0.0040720468f*sp;
        res.a = 1.9779984951f*lp - 2.4285922050f*mp + 0.4505937099f*sp;
        res.b = 0.0259040371f*lp + 0.7827717662f*mp - 0.8086758033f*sp;
        return res;
    }
    inline Vector4 PerceptualColorEngine::OklabToLinearRGB(ColorOklab lab, float alpha) {
        float lp = lab.L + 0.3963377774f*lab.a + 0.2158037573f*lab.b;
        float mp = lab.L - 0.1055613458f*lab.a - 0.0638541728f*lab.b;
        float sp = lab.L - 0.0894841775f*lab.a - 1.2914855480f*lab.b;
        float l = lp*lp*lp, m = mp*mp*mp, s = sp*sp*sp;
        Vector4 res;
        res.r = +4.0767416621f*l - 3.3077115913f*m + 0.2309699292f*s;
        res.g = -1.2684380046f*l + 2.6097574011f*m - 0.3413193965f*s;
        res.b = -0.0041960863f*l - 0.7034186147f*m + 1.7076147010f*s;
        res.a = alpha;
        res.r = Clamp01(res.r); res.g = Clamp01(res.g); res.b = Clamp01(res.b);
        return res;
    }
    inline ColorOKLCH PerceptualColorEngine::OklabToOKLCH(ColorOklab lab) {
        ColorOKLCH res; res.L = lab.L; res.C = std::sqrt(lab.a*lab.a + lab.b*lab.b); res.H = std::atan2(lab.b, lab.a); return res;
    }
    inline ColorOklab PerceptualColorEngine::OKLCHToOklab(ColorOKLCH lch) {
        ColorOklab res; res.L = lch.L; res.a = lch.C * std::cos(lch.H); res.b = lch.C * std::sin(lch.H); return res;
    }
    inline Vector4 PerceptualColorEngine::ApplyPerfectTint(Vector4 baseTexture, Vector4 tintColor, float factor) {
        ColorOKLCH baseLCH = OklabToOKLCH(LinearRGBToOklab(baseTexture));
        ColorOKLCH tintLCH = OklabToOKLCH(LinearRGBToOklab(tintColor));
        ColorOKLCH targetLCH;
        targetLCH.L = baseLCH.L;
        targetLCH.H = tintLCH.H;
        float tintSaturation = tintLCH.C / (tintLCH.L + 0.00001f);
        targetLCH.C = baseLCH.C * Clamp01(tintSaturation);
        Vector4 tintedRGB = OklabToLinearRGB(OKLCHToOklab(targetLCH), baseTexture.a);
        Vector4 output;
        output.r = baseTexture.r + factor*(tintedRGB.r - baseTexture.r);
        output.g = baseTexture.g + factor*(tintedRGB.g - baseTexture.g);
        output.b = baseTexture.b + factor*(tintedRGB.b - baseTexture.b);
        output.a = baseTexture.a;
        return output;
    }
    inline Vector4 PerceptualColorEngine::ApplyChromatization(Vector4 color, float scale) {
        ColorOklab lab = LinearRGBToOklab(color);
        ColorOKLCH lch = OklabToOKLCH(lab);
        lch.C *= scale;
        return OklabToLinearRGB(OKLCHToOklab(lch), color.a);
    }

} // namespace perceptual_color

} // namespace SimulationMath

#endif // CORE_MATH_COLORIMETRY_H