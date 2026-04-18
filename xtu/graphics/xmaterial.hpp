// math/xmaterial.hpp

#ifndef XTENSOR_XMATERIAL_HPP
#define XTENSOR_XMATERIAL_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xfunction.hpp"
#include "../core/xview.hpp"
#include "xnorm.hpp"
#include "xlinalg.hpp"
#include "xstats.hpp"

#include <cmath>
#include <complex>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <vector>
#include <array>
#include <functional>
#include <stdexcept>
#include <limits>
#include <cstdint>
#include <cstring>
#include <random>
#include <tuple>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace material
        {
            // --------------------------------------------------------------------
            // Color space conversion utilities
            // --------------------------------------------------------------------
            struct RGB
            {
                double r, g, b;
                RGB(double red = 0.0, double green = 0.0, double blue = 0.0)
                    : r(red), g(green), b(blue) {}
                RGB clamp() const { return RGB(std::clamp(r,0.0,1.0), std::clamp(g,0.0,1.0), std::clamp(b,0.0,1.0)); }
            };
            
            struct HSV
            {
                double h, s, v;
                HSV(double hue = 0.0, double sat = 0.0, double val = 0.0)
                    : h(hue), s(sat), v(val) {}
            };
            
            struct HSL
            {
                double h, s, l;
                HSL(double hue = 0.0, double sat = 0.0, double light = 0.0)
                    : h(hue), s(sat), l(light) {}
            };
            
            struct XYZ
            {
                double x, y, z;
                XYZ(double _x=0.0, double _y=0.0, double _z=0.0) : x(_x), y(_y), z(_z) {}
            };
            
            struct Lab
            {
                double L, a, b;
                Lab(double _L=0.0, double _a=0.0, double _b=0.0) : L(_L), a(_a), b(_b) {}
            };
            
            // RGB <-> HSV
            inline HSV rgb2hsv(const RGB& rgb)
            {
                double r = rgb.r, g = rgb.g, b = rgb.b;
                double maxc = std::max({r,g,b});
                double minc = std::min({r,g,b});
                double delta = maxc - minc;
                double h = 0.0, s = 0.0, v = maxc;
                if (delta > 1e-10)
                {
                    s = delta / maxc;
                    if (maxc == r)
                        h = (g - b) / delta + (g < b ? 6.0 : 0.0);
                    else if (maxc == g)
                        h = (b - r) / delta + 2.0;
                    else
                        h = (r - g) / delta + 4.0;
                    h /= 6.0;
                }
                return HSV(h, s, v);
            }
            
            inline RGB hsv2rgb(const HSV& hsv)
            {
                double h = std::fmod(hsv.h, 1.0);
                if (h < 0) h += 1.0;
                double s = hsv.s, v = hsv.v;
                if (s == 0.0) return RGB(v, v, v);
                int i = static_cast<int>(h * 6.0);
                double f = h * 6.0 - i;
                double p = v * (1.0 - s);
                double q = v * (1.0 - s * f);
                double t = v * (1.0 - s * (1.0 - f));
                switch (i % 6)
                {
                    case 0: return RGB(v, t, p);
                    case 1: return RGB(q, v, p);
                    case 2: return RGB(p, v, t);
                    case 3: return RGB(p, q, v);
                    case 4: return RGB(t, p, v);
                    default: return RGB(v, p, q);
                }
            }
            
            // RGB <-> HSL
            inline HSL rgb2hsl(const RGB& rgb)
            {
                double r = rgb.r, g = rgb.g, b = rgb.b;
                double maxc = std::max({r,g,b});
                double minc = std::min({r,g,b});
                double l = (maxc + minc) / 2.0;
                double h = 0.0, s = 0.0;
                if (maxc != minc)
                {
                    double d = maxc - minc;
                    s = l > 0.5 ? d / (2.0 - maxc - minc) : d / (maxc + minc);
                    if (maxc == r)
                        h = (g - b) / d + (g < b ? 6.0 : 0.0);
                    else if (maxc == g)
                        h = (b - r) / d + 2.0;
                    else
                        h = (r - g) / d + 4.0;
                    h /= 6.0;
                }
                return HSL(h, s, l);
            }
            
            inline RGB hsl2rgb(const HSL& hsl)
            {
                double h = hsl.h, s = hsl.s, l = hsl.l;
                if (s == 0.0) return RGB(l, l, l);
                auto hue2rgb = [](double p, double q, double t) -> double {
                    if (t < 0.0) t += 1.0;
                    if (t > 1.0) t -= 1.0;
                    if (t < 1.0/6.0) return p + (q - p) * 6.0 * t;
                    if (t < 1.0/2.0) return q;
                    if (t < 2.0/3.0) return p + (q - p) * (2.0/3.0 - t) * 6.0;
                    return p;
                };
                double q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
                double p = 2.0 * l - q;
                return RGB(hue2rgb(p, q, h + 1.0/3.0),
                           hue2rgb(p, q, h),
                           hue2rgb(p, q, h - 1.0/3.0));
            }
            
            // RGB <-> XYZ (sRGB D65)
            inline XYZ rgb2xyz(const RGB& rgb)
            {
                auto linearize = [](double c) -> double {
                    if (c <= 0.04045) return c / 12.92;
                    return std::pow((c + 0.055) / 1.055, 2.4);
                };
                double r = linearize(rgb.r);
                double g = linearize(rgb.g);
                double b = linearize(rgb.b);
                double x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
                double y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
                double z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
                return XYZ(x, y, z);
            }
            
            inline RGB xyz2rgb(const XYZ& xyz)
            {
                double x = xyz.x, y = xyz.y, z = xyz.z;
                double r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314;
                double g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560;
                double b = x *  0.0556434 + y * -0.2040259 + z *  1.0572252;
                auto delinearize = [](double c) -> double {
                    if (c <= 0.0031308) return 12.92 * c;
                    return 1.055 * std::pow(c, 1.0/2.4) - 0.055;
                };
                return RGB(delinearize(r), delinearize(g), delinearize(b));
            }
            
            // XYZ <-> Lab (CIE 1976)
            inline Lab xyz2lab(const XYZ& xyz, const XYZ& white = XYZ(0.95047, 1.0, 1.08883))
            {
                auto f = [](double t) -> double {
                    if (t > 0.008856) return std::cbrt(t);
                    return (7.787 * t) + (16.0 / 116.0);
                };
                double fx = f(xyz.x / white.x);
                double fy = f(xyz.y / white.y);
                double fz = f(xyz.z / white.z);
                double L = 116.0 * fy - 16.0;
                double a = 500.0 * (fx - fy);
                double b = 200.0 * (fy - fz);
                return Lab(L, a, b);
            }
            
            inline XYZ lab2xyz(const Lab& lab, const XYZ& white = XYZ(0.95047, 1.0, 1.08883))
            {
                double fy = (lab.L + 16.0) / 116.0;
                double fx = lab.a / 500.0 + fy;
                double fz = fy - lab.b / 200.0;
                auto finv = [](double t) -> double {
                    double t3 = t * t * t;
                    if (t3 > 0.008856) return t3;
                    return (t - 16.0/116.0) / 7.787;
                };
                double x = finv(fx) * white.x;
                double y = finv(fy) * white.y;
                double z = finv(fz) * white.z;
                return XYZ(x, y, z);
            }
            
            inline Lab rgb2lab(const RGB& rgb) { return xyz2lab(rgb2xyz(rgb)); }
            inline RGB lab2rgb(const Lab& lab) { return xyz2rgb(lab2xyz(lab)); }
            
            // --------------------------------------------------------------------
            // Material parameters (PBR metallic-roughness model)
            // --------------------------------------------------------------------
            struct Material
            {
                RGB albedo = RGB(1.0, 1.0, 1.0);
                double metallic = 0.0;
                double roughness = 0.5;
                double ao = 1.0;               // ambient occlusion
                RGB emission = RGB(0.0, 0.0, 0.0);
                double transparency = 0.0;
                double ior = 1.5;              // index of refraction
                double anisotropic = 0.0;
                double sheen = 0.0;
                double sheen_tint = 0.0;
                double clearcoat = 0.0;
                double clearcoat_roughness = 0.0;
                double specular = 0.5;
                double specular_tint = 0.0;
                double subsurface = 0.0;
            };
            
            // --------------------------------------------------------------------
            // Utility functions for materials
            // --------------------------------------------------------------------
            inline double perceptual_roughness_to_alpha(double roughness)
            {
                return roughness * roughness;
            }
            
            inline double alpha_to_perceptual_roughness(double alpha)
            {
                return std::sqrt(std::max(alpha, 0.0));
            }
            
            inline RGB compute_f0(const RGB& albedo, double metallic)
            {
                RGB dielectric_f0(0.04, 0.04, 0.04);
                return dielectric_f0 * (1.0 - metallic) + albedo * metallic;
            }
            
            // --------------------------------------------------------------------
            // Fresnel (Schlick approximation)
            // --------------------------------------------------------------------
            inline RGB fresnel_schlick(double cos_theta, const RGB& f0)
            {
                return f0 + (RGB(1.0,1.0,1.0) - f0) * std::pow(std::max(1.0 - cos_theta, 0.0), 5.0);
            }
            
            inline double fresnel_schlick(double cos_theta, double f0)
            {
                return f0 + (1.0 - f0) * std::pow(std::max(1.0 - cos_theta, 0.0), 5.0);
            }
            
            // Fresnel for conductors (complex IOR)
            inline RGB fresnel_conductor(double cos_theta, const RGB& f0, const RGB& k)
            {
                // Full Fresnel equations for metals (simplified)
                RGB one(1.0,1.0,1.0);
                RGB cos2 = cos_theta * cos_theta;
                RGB sin2 = 1.0 - cos2;
                RGB eta2 = f0;
                RGB k2 = k * k;
                RGB t0 = eta2 - k2 - sin2;
                RGB a2plusb2 = t0 * t0 + 4.0 * eta2 * k2;
                RGB a = (a2plusb2 * a2plusb2).apply([](double x){ return std::sqrt(std::sqrt(x)); }); // sqrt of sqrt
                RGB t1 = a + cos2;
                RGB a2 = (a + cos_theta * cos_theta) * 0.5;
                RGB t2 = 2.0 * cos_theta * a;
                RGB Rs = (a2 - t2) / (a2 + t2);
                RGB Rp = Rs * ((cos2 * a2plusb2) / (sin2 * a2plusb2 + cos2 * cos2)).apply([](double x){ return std::sqrt(x); });
                return (Rs + Rp) * 0.5;
            }
            
            // --------------------------------------------------------------------
            // Normal Distribution Functions (NDF)
            // --------------------------------------------------------------------
            // GGX / Trowbridge-Reitz
            inline double ndf_ggx(double n_dot_h, double alpha)
            {
                double a2 = alpha * alpha;
                double denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
                return a2 / (M_PI * denom * denom);
            }
            
            // Beckmann
            inline double ndf_beckmann(double n_dot_h, double alpha)
            {
                double a2 = alpha * alpha;
                double cos2 = n_dot_h * n_dot_h;
                double tan2 = (1.0 - cos2) / cos2;
                if (std::isinf(tan2)) return 0.0;
                double denom = M_PI * a2 * cos2 * cos2;
                return std::exp(-tan2 / a2) / denom;
            }
            
            // Blinn-Phong
            inline double ndf_blinn_phong(double n_dot_h, double shininess)
            {
                return (shininess + 2.0) / (2.0 * M_PI) * std::pow(n_dot_h, shininess);
            }
            
            // Anisotropic GGX
            inline double ndf_anisotropic_ggx(double n_dot_h, double h_dot_x, double h_dot_y, double ax, double ay)
            {
                double a = h_dot_x / ax;
                double b = h_dot_y / ay;
                double denom = a*a + b*b + n_dot_h*n_dot_h;
                return 1.0 / (M_PI * ax * ay * denom * denom);
            }
            
            // --------------------------------------------------------------------
            // Geometry (Shadowing-Masking) Functions
            // --------------------------------------------------------------------
            // Smith GGX
            inline double geometry_smith_ggx(double n_dot_v, double alpha)
            {
                double a2 = alpha * alpha;
                double denom = n_dot_v + std::sqrt(a2 + (1.0 - a2) * n_dot_v * n_dot_v);
                return 2.0 * n_dot_v / denom;
            }
            
            inline double geometry_smith(double n_dot_v, double n_dot_l, double alpha)
            {
                return geometry_smith_ggx(n_dot_l, alpha) * geometry_smith_ggx(n_dot_v, alpha);
            }
            
            // Schlick-Beckmann
            inline double geometry_schlick_beckmann(double n_dot_x, double alpha)
            {
                double k = alpha * std::sqrt(2.0 / M_PI);
                return n_dot_x / (n_dot_x * (1.0 - k) + k);
            }
            
            inline double geometry_schlick(double n_dot_v, double n_dot_l, double alpha)
            {
                return geometry_schlick_beckmann(n_dot_v, alpha) * geometry_schlick_beckmann(n_dot_l, alpha);
            }
            
            // Cook-Torrance geometry
            inline double geometry_cook_torrance(double n_dot_h, double n_dot_v, double n_dot_l, double v_dot_h)
            {
                return std::min(1.0, std::min(2.0 * n_dot_h * n_dot_v / v_dot_h, 2.0 * n_dot_h * n_dot_l / v_dot_h));
            }
            
            // --------------------------------------------------------------------
            // Diffuse BRDFs
            // --------------------------------------------------------------------
            // Lambertian
            inline RGB diffuse_lambert(const RGB& albedo)
            {
                return albedo / M_PI;
            }
            
            // Oren-Nayar
            inline RGB diffuse_oren_nayar(const RGB& albedo, double roughness,
                                          double n_dot_v, double n_dot_l, double v_dot_l)
            {
                double sigma2 = roughness * roughness;
                double A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
                double B = 0.45 * sigma2 / (sigma2 + 0.09);
                double cos_phi_diff = v_dot_l - n_dot_v * n_dot_l;
                double sin_alpha = (n_dot_v > n_dot_l) ? std::sqrt(1.0 - n_dot_l*n_dot_l) : std::sqrt(1.0 - n_dot_v*n_dot_v);
                double tan_alpha = sin_alpha / std::max(n_dot_v, n_dot_l);
                return albedo / M_PI * (A + B * std::max(0.0, cos_phi_diff) * sin_alpha * tan_alpha);
            }
            
            // Burley (Disney diffuse)
            inline RGB diffuse_burley(const RGB& albedo, double roughness,
                                      double n_dot_v, double n_dot_l, double v_dot_h)
            {
                double fd90 = 0.5 + 2.0 * v_dot_h * v_dot_h * roughness;
                double fl = 1.0 + (fd90 - 1.0) * std::pow(1.0 - n_dot_l, 5.0);
                double fv = 1.0 + (fd90 - 1.0) * std::pow(1.0 - n_dot_v, 5.0);
                return albedo / M_PI * fl * fv;
            }
            
            // --------------------------------------------------------------------
            // Specular BRDF (Microfacet Cook-Torrance)
            // --------------------------------------------------------------------
            inline RGB specular_brdf(const RGB& f0, double roughness, double metallic,
                                     double n_dot_v, double n_dot_l, double n_dot_h, double v_dot_h)
            {
                double alpha = perceptual_roughness_to_alpha(roughness);
                double D = ndf_ggx(n_dot_h, alpha);
                double G = geometry_smith(n_dot_v, n_dot_l, alpha);
                RGB F = fresnel_schlick(v_dot_h, f0);
                RGB numerator = F * D * G;
                double denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
                return numerator / denominator;
            }
            
            // --------------------------------------------------------------------
            // Disney Principled BSDF
            // --------------------------------------------------------------------
            inline RGB disney_brdf(const Material& mat, double n_dot_v, double n_dot_l, double n_dot_h,
                                   double v_dot_h, double l_dot_h)
            {
                RGB albedo = mat.albedo;
                double metallic = mat.metallic;
                double roughness = mat.roughness;
                double anisotropic = mat.anisotropic;
                double sheen = mat.sheen;
                double sheen_tint = mat.sheen_tint;
                double clearcoat = mat.clearcoat;
                double specular = mat.specular;
                double specular_tint = mat.specular_tint;
                
                // Base diffuse
                RGB diffuse = diffuse_burley(albedo, roughness, n_dot_v, n_dot_l, v_dot_h);
                
                // Sheen
                RGB sheen_color = albedo * sheen_tint + RGB(1.0-sheen_tint, 1.0-sheen_tint, 1.0-sheen_tint);
                RGB sheen_brdf = sheen_color * sheen * fresnel_schlick(v_dot_h, 0.0);
                
                // Specular
                RGB f0 = compute_f0(albedo, metallic);
                RGB spec = specular_brdf(f0, roughness, metallic, n_dot_v, n_dot_l, n_dot_h, v_dot_h);
                
                // Clearcoat
                double cc_alpha = perceptual_roughness_to_alpha(mat.clearcoat_roughness);
                double cc_D = ndf_ggx(n_dot_h, cc_alpha);
                double cc_G = geometry_smith(n_dot_v, n_dot_l, 0.25);
                double cc_F = fresnel_schlick(v_dot_h, 0.04);
                RGB cc = RGB(cc_F) * cc_D * cc_G / (4.0 * n_dot_v * n_dot_l);
                
                return (diffuse * (1.0 - metallic) + spec) * (1.0 - clearcoat) + cc * clearcoat + sheen_brdf;
            }
            
            // --------------------------------------------------------------------
            // Lighting utilities
            // --------------------------------------------------------------------
            inline double schlick_ggx_visibility(double n_dot_v, double n_dot_l, double alpha)
            {
                return geometry_smith(n_dot_v, n_dot_l, alpha);
            }
            
            inline double G1_GGX(double n_dot_x, double alpha)
            {
                return geometry_smith_ggx(n_dot_x, alpha);
            }
            
            // --------------------------------------------------------------------
            // Texture mapping helpers
            // --------------------------------------------------------------------
            template <class E>
            inline auto apply_gamma(const xexpression<E>& img, double gamma = 2.2)
            {
                auto result = eval(img);
                double inv_gamma = 1.0 / gamma;
                for (auto& v : result)
                    v = std::pow(std::max(v, 0.0), inv_gamma);
                return result;
            }
            
            template <class E>
            inline auto remove_gamma(const xexpression<E>& img, double gamma = 2.2)
            {
                auto result = eval(img);
                for (auto& v : result)
                    v = std::pow(std::max(v, 0.0), gamma);
                return result;
            }
            
            // Convert sRGB to linear
            template <class E>
            inline auto srgb_to_linear(const xexpression<E>& img)
            {
                auto result = eval(img);
                for (auto& v : result)
                {
                    if (v <= 0.04045)
                        v = v / 12.92;
                    else
                        v = std::pow((v + 0.055) / 1.055, 2.4);
                }
                return result;
            }
            
            // Convert linear to sRGB
            template <class E>
            inline auto linear_to_srgb(const xexpression<E>& img)
            {
                auto result = eval(img);
                for (auto& v : result)
                {
                    if (v <= 0.0031308)
                        v = 12.92 * v;
                    else
                        v = 1.055 * std::pow(v, 1.0/2.4) - 0.055;
                }
                return result;
            }
            
            // Tone mapping (Reinhard, ACES, Uncharted 2)
            inline RGB tonemap_reinhard(const RGB& color)
            {
                return RGB(color.r / (1.0 + color.r), color.g / (1.0 + color.g), color.b / (1.0 + color.b));
            }
            
            inline RGB tonemap_aces(const RGB& color)
            {
                double a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
                auto f = [=](double x) { return (x*(a*x+b)) / (x*(c*x+d)+e); };
                return RGB(f(color.r), f(color.g), f(color.b)).clamp();
            }
            
            inline RGB tonemap_uncharted2(const RGB& color)
            {
                double A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;
                auto f = [=](double x) { return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F)) - E/F; };
                double exposure_bias = 2.0;
                RGB curr = color * exposure_bias;
                RGB mapped(f(curr.r), f(curr.g), f(curr.b));
                RGB white_scale = RGB(1.0) / tonemap_uncharted2(RGB(11.2));
                return mapped * white_scale;
            }
            
            // --------------------------------------------------------------------
            // Image based lighting utilities
            // --------------------------------------------------------------------
            template <class E>
            inline auto equirectangular_to_cubemap(const xexpression<E>& equi)
            {
                // Convert equirectangular panorama to 6 cubemap faces
                const auto& img = equi.derived_cast();
                if (img.dimension() != 3 || img.shape()[2] != 3)
                    XTENSOR_THROW(std::invalid_argument, "equirectangular_to_cubemap: expected HxWx3 image");
                size_t h = img.shape()[0];
                size_t w = img.shape()[1];
                size_t face_size = std::min(h, w) / 4;
                
                // Placeholder - would output 6xface_sizexface_sizex3
                xarray_container<double> result({6, face_size, face_size, 3});
                // Not fully implemented due to complexity, but structure provided
                return result;
            }
            
            // Prefilter environment map for specular
            template <class E>
            inline auto prefilter_specular_cubemap(const xexpression<E>& cubemap, size_t mip_levels = 5)
            {
                // Implementation would use importance sampling
                // Return mipmapped cubemap
                XTENSOR_THROW(not_implemented_error, "prefilter_specular_cubemap not fully implemented");
                return xarray_container<double>();
            }
            
            // --------------------------------------------------------------------
            // Baking utilities
            // --------------------------------------------------------------------
            struct bake_params
            {
                size_t width = 1024;
                size_t height = 1024;
                size_t samples = 64;
                bool use_gpu = false;
            };
            
            inline auto bake_ambient_occlusion(const std::function<bool(double,double,double)>& inside_func,
                                               const bake_params& params = {})
            {
                // Ray-traced AO baking (simplified)
                // Returns a 2D array of AO values
                xarray_container<double> result({params.height, params.width});
                std::mt19937 gen(42);
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                // Placeholder implementation
                for (size_t i = 0; i < result.size(); ++i) result.flat(i) = 1.0;
                return result;
            }
            
        } // namespace material
        
        // Bring material namespace into xt
        using material::RGB;
        using material::HSV;
        using material::HSL;
        using material::XYZ;
        using material::Lab;
        using material::rgb2hsv;
        using material::hsv2rgb;
        using material::rgb2hsl;
        using material::hsl2rgb;
        using material::rgb2xyz;
        using material::xyz2rgb;
        using material::xyz2lab;
        using material::lab2xyz;
        using material::rgb2lab;
        using material::lab2rgb;
        using material::Material;
        using material::fresnel_schlick;
        using material::fresnel_conductor;
        using material::ndf_ggx;
        using material::ndf_beckmann;
        using material::ndf_blinn_phong;
        using material::geometry_smith;
        using material::geometry_cook_torrance;
        using material::diffuse_lambert;
        using material::diffuse_oren_nayar;
        using material::diffuse_burley;
        using material::specular_brdf;
        using material::disney_brdf;
        using material::tonemap_reinhard;
        using material::tonemap_aces;
        using material::tonemap_uncharted2;
        using material::apply_gamma;
        using material::remove_gamma;
        using material::srgb_to_linear;
        using material::linear_to_srgb;
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XMATERIAL_HPP

// math/xmaterial.hpp