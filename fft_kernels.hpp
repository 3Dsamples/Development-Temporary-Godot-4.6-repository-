// src/fft_kernels.hpp
#pragma once

#include "fft_simd.hpp"
#include <cstddef>
#include <complex>
#include <cmath>

namespace fastfft {
namespace kernels {

using simd::vfloat;
using simd::vdouble;
using simd::simd_traits;

// ============================================================================
// Radix-2 butterfly (decimation-in-time)
// ============================================================================
template<typename T>
inline void radix2_dit(std::complex<T>* data, size_t stride, size_t n) {
    using vtype = typename simd_traits<T>::vtype;
    constexpr size_t W = simd_traits<T>::width;
    
    for (size_t i = 0; i < n; i += 2*stride) {
        T* a_real = reinterpret_cast<T*>(&data[i]);
        T* b_real = reinterpret_cast<T*>(&data[i + stride]);
        
        for (size_t k = 0; k < stride; k += W) {
            vtype ar = vtype::load_aligned(&a_real[2*k]);
            vtype ai = vtype::load_aligned(&a_real[2*k + W]);
            vtype br = vtype::load_aligned(&b_real[2*k]);
            vtype bi = vtype::load_aligned(&b_real[2*k + W]);
            
            vtype sum_r = ar + br;
            vtype sum_i = ai + bi;
            vtype diff_r = ar - br;
            vtype diff_i = ai - bi;
            
            sum_r.store_aligned(&a_real[2*k]);
            sum_i.store_aligned(&a_real[2*k + W]);
            diff_r.store_aligned(&b_real[2*k]);
            diff_i.store_aligned(&b_real[2*k + W]);
        }
    }
}

// ============================================================================
// Radix-3 butterfly
// ============================================================================
template<typename T>
inline void radix3_dit(std::complex<T>* data, size_t stride, size_t n) {
    const T c1 = -0.5;
    const T s1 = T(0.86602540378443864676);  // sqrt(3)/2
    
    for (size_t i = 0; i < n; i += 3*stride) {
        std::complex<T>* a = &data[i];
        std::complex<T>* b = &data[i + stride];
        std::complex<T>* c = &data[i + 2*stride];
        
        for (size_t k = 0; k < stride; ++k) {
            T ar = a[k].real(); T ai = a[k].imag();
            T br = b[k].real(); T bi = b[k].imag();
            T cr = c[k].real(); T ci = c[k].imag();
            
            T t0 = br + cr;
            T t1 = ar - T(0.5) * t0;
            T t2 = s1 * (bi - ci);
            T t3 = s1 * (cr - br);
            
            a[k] = std::complex<T>(ar + t0, ai);
            b[k] = std::complex<T>(t1 + t2, ai - s1 * (br - cr));
            c[k] = std::complex<T>(t1 - t2, ai + t3);
        }
    }
}

// ============================================================================
// Radix-4 butterfly (SIMD optimized)
// ============================================================================
template<typename T>
inline void radix4_dit(std::complex<T>* data, size_t stride, size_t n) {
    using vtype = typename simd_traits<T>::vtype;
    constexpr size_t W = simd_traits<T>::width;
    
    for (size_t i = 0; i < n; i += 4*stride) {
        T* a_real = reinterpret_cast<T*>(&data[i]);
        T* b_real = reinterpret_cast<T*>(&data[i + stride]);
        T* c_real = reinterpret_cast<T*>(&data[i + 2*stride]);
        T* d_real = reinterpret_cast<T*>(&data[i + 3*stride]);
        
        for (size_t k = 0; k < stride; k += W) {
            vtype ar = vtype::load_aligned(&a_real[2*k]);
            vtype ai = vtype::load_aligned(&a_real[2*k + W]);
            vtype br = vtype::load_aligned(&b_real[2*k]);
            vtype bi = vtype::load_aligned(&b_real[2*k + W]);
            vtype cr = vtype::load_aligned(&c_real[2*k]);
            vtype ci = vtype::load_aligned(&c_real[2*k + W]);
            vtype dr = vtype::load_aligned(&d_real[2*k]);
            vtype di = vtype::load_aligned(&d_real[2*k + W]);
            
            vtype tr0 = ar + cr;
            vtype ti0 = ai + ci;
            vtype tr1 = ar - cr;
            vtype ti1 = ai - ci;
            vtype tr2 = br + dr;
            vtype ti2 = bi + di;
            vtype tr3 = br - dr;
            vtype ti3 = bi - di;
            
            // Result 0
            (tr0 + tr2).store_aligned(&a_real[2*k]);
            (ti0 + ti2).store_aligned(&a_real[2*k + W]);
            
            // Result 1 (multiply by -j: swap and negate imag)
            (tr1 + ti3).store_aligned(&b_real[2*k]);
            (ti1 - tr3).store_aligned(&b_real[2*k + W]);
            
            // Result 2 (multiply by -1)
            (tr0 - tr2).store_aligned(&c_real[2*k]);
            (ti0 - ti2).store_aligned(&c_real[2*k + W]);
            
            // Result 3 (multiply by j: swap and negate real)
            (tr1 - ti3).store_aligned(&d_real[2*k]);
            (ti1 + tr3).store_aligned(&d_real[2*k + W]);
        }
    }
}

// ============================================================================
// Radix-5 butterfly
// ============================================================================
template<typename T>
inline void radix5_dit(std::complex<T>* data, size_t stride, size_t n) {
    const T c1 = T(0.3090169943749474241);   // cos(2pi/5)
    const T c2 = T(-0.8090169943749474241);  // cos(4pi/5)
    const T s1 = T(0.95105651629515357212);  // sin(2pi/5)
    const T s2 = T(0.58778525229247312917);  // sin(4pi/5)
    
    for (size_t i = 0; i < n; i += 5*stride) {
        std::complex<T>* a = &data[i];
        std::complex<T>* b = &data[i + stride];
        std::complex<T>* c = &data[i + 2*stride];
        std::complex<T>* d = &data[i + 3*stride];
        std::complex<T>* e = &data[i + 4*stride];
        
        for (size_t k = 0; k < stride; ++k) {
            T ar = a[k].real(); T ai = a[k].imag();
            T br = b[k].real(); T bi = b[k].imag();
            T cr = c[k].real(); T ci = c[k].imag();
            T dr = d[k].real(); T di = d[k].imag();
            T er = e[k].real(); T ei = e[k].imag();
            
            T t0 = br + er;
            T t1 = cr + dr;
            T t2 = br - er;
            T t3 = cr - dr;
            T t4 = t0 + t1;
            T t5 = c1 * t0 + c2 * t1;
            T t6 = c2 * t0 + c1 * t1;
            T t7 = s1 * t2 + s2 * t3;
            T t8 = s2 * t2 - s1 * t3;
            
            a[k] = std::complex<T>(ar + t4, ai);
            b[k] = std::complex<T>(ar + t5, ai + t7);
            c[k] = std::complex<T>(ar + t6, ai + t8);
            d[k] = std::complex<T>(ar + t6, ai - t8);
            e[k] = std::complex<T>(ar + t5, ai - t7);
        }
    }
}

// ============================================================================
// Radix-7 butterfly
// ============================================================================
template<typename T>
inline void radix7_dit(std::complex<T>* data, size_t stride, size_t n) {
    const T c1 = T(0.62348980185873353053);  // cos(2pi/7)
    const T c2 = T(-0.22252093395631440429); // cos(4pi/7)
    const T c3 = T(-0.90096886790241912624); // cos(6pi/7)
    const T s1 = T(0.78183148246802980871);  // sin(2pi/7)
    const T s2 = T(0.97492791218182360702);  // sin(4pi/7)
    const T s3 = T(0.43388373911755812048);  // sin(6pi/7)
    
    for (size_t i = 0; i < n; i += 7*stride) {
        std::complex<T>* inout = &data[i];
        for (size_t k = 0; k < stride; ++k) {
            std::complex<T> v[7];
            for (size_t j = 0; j < 7; ++j) {
                v[j] = inout[j*stride + k];
            }
            
            // Direct DFT for radix-7
            for (size_t j = 0; j < 7; ++j) {
                std::complex<T> sum = 0;
                for (size_t m = 0; m < 7; ++m) {
                    T angle = -T(2.0 * 3.14159265358979323846) * T(j * m) / T(7);
                    T wr = std::cos(angle);
                    T wi = std::sin(angle);
                    sum += v[m] * std::complex<T>(wr, wi);
                }
                inout[j*stride + k] = sum;
            }
        }
    }
}

// ============================================================================
// Radix-8 butterfly (SIMD, fully unrolled)
// ============================================================================
template<typename T>
inline void radix8_dit(std::complex<T>* data, size_t stride, size_t n) {
    using vtype = typename simd_traits<T>::vtype;
    constexpr size_t W = simd_traits<T>::width;
    const T sqrt2_2 = T(0.70710678118654752440);
    
    for (size_t i = 0; i < n; i += 8*stride) {
        T* d0_real = reinterpret_cast<T*>(&data[i]);
        T* d1_real = reinterpret_cast<T*>(&data[i + stride]);
        T* d2_real = reinterpret_cast<T*>(&data[i + 2*stride]);
        T* d3_real = reinterpret_cast<T*>(&data[i + 3*stride]);
        T* d4_real = reinterpret_cast<T*>(&data[i + 4*stride]);
        T* d5_real = reinterpret_cast<T*>(&data[i + 5*stride]);
        T* d6_real = reinterpret_cast<T*>(&data[i + 6*stride]);
        T* d7_real = reinterpret_cast<T*>(&data[i + 7*stride]);
        
        for (size_t k = 0; k < stride; k += W) {
            vtype r0 = vtype::load_aligned(&d0_real[2*k]);
            vtype i0 = vtype::load_aligned(&d0_real[2*k + W]);
            vtype r1 = vtype::load_aligned(&d1_real[2*k]);
            vtype i1 = vtype::load_aligned(&d1_real[2*k + W]);
            vtype r2 = vtype::load_aligned(&d2_real[2*k]);
            vtype i2 = vtype::load_aligned(&d2_real[2*k + W]);
            vtype r3 = vtype::load_aligned(&d3_real[2*k]);
            vtype i3 = vtype::load_aligned(&d3_real[2*k + W]);
            vtype r4 = vtype::load_aligned(&d4_real[2*k]);
            vtype i4 = vtype::load_aligned(&d4_real[2*k + W]);
            vtype r5 = vtype::load_aligned(&d5_real[2*k]);
            vtype i5 = vtype::load_aligned(&d5_real[2*k + W]);
            vtype r6 = vtype::load_aligned(&d6_real[2*k]);
            vtype i6 = vtype::load_aligned(&d6_real[2*k + W]);
            vtype r7 = vtype::load_aligned(&d7_real[2*k]);
            vtype i7 = vtype::load_aligned(&d7_real[2*k + W]);
            
            // Stage 1: pairs (0,4), (2,6), (1,5), (3,7)
            vtype tr04 = r0 + r4; vtype ti04 = i0 + i4;
            vtype tr04_ = r0 - r4; vtype ti04_ = i0 - i4;
            vtype tr26 = r2 + r6; vtype ti26 = i2 + i6;
            vtype tr26_ = r2 - r6; vtype ti26_ = i2 - i6;
            vtype tr15 = r1 + r5; vtype ti15 = i1 + i5;
            vtype tr15_ = r1 - r5; vtype ti15_ = i1 - i5;
            vtype tr37 = r3 + r7; vtype ti37 = i3 + i7;
            vtype tr37_ = r3 - r7; vtype ti37_ = i3 - i7;
            
            // Stage 2
            vtype tr0426 = tr04 + tr26; vtype ti0426 = ti04 + ti26;
            vtype tr0426_ = tr04 - tr26; vtype ti0426_ = ti04 - ti26;
            vtype tr1537 = tr15 + tr37; vtype ti1537 = ti15 + ti37;
            vtype tr1537_ = tr15 - tr37; vtype ti1537_ = ti15 - ti37;
            
            // Twiddle for odd indices (multiply by sqrt(2)/2)
            vtype sqrt2 = vtype::set1(sqrt2_2);
            vtype tr15_tw = (tr15_ + ti15_) * sqrt2;
            vtype ti15_tw = (ti15_ - tr15_) * sqrt2;
            vtype tr37_tw = (tr37_ - ti37_) * sqrt2;
            vtype ti37_tw = (ti37_ + tr37_) * sqrt2;
            
            // Stage 3 and output
            (tr0426 + tr1537).store_aligned(&d0_real[2*k]);
            (ti0426 + ti1537).store_aligned(&d0_real[2*k + W]);
            
            (tr0426_ + tr15_tw).store_aligned(&d1_real[2*k]);
            (ti0426_ + ti15_tw).store_aligned(&d1_real[2*k + W]);
            
            (tr04_ + ti26_).store_aligned(&d2_real[2*k]);
            (ti04_ - tr26_).store_aligned(&d2_real[2*k + W]);
            
            (tr04_ - ti26_ + tr37_tw).store_aligned(&d3_real[2*k]);
            (ti04_ + tr26_ + ti37_tw).store_aligned(&d3_real[2*k + W]);
            
            (tr0426 - tr1537).store_aligned(&d4_real[2*k]);
            (ti0426 - ti1537).store_aligned(&d4_real[2*k + W]);
            
            (tr0426_ - tr15_tw).store_aligned(&d5_real[2*k]);
            (ti0426_ - ti15_tw).store_aligned(&d5_real[2*k + W]);
            
            (tr04_ - ti26_).store_aligned(&d6_real[2*k]);
            (ti04_ + tr26_).store_aligned(&d6_real[2*k + W]);
            
            (tr04_ + ti26_ - tr37_tw).store_aligned(&d7_real[2*k]);
            (ti04_ - tr26_ - ti37_tw).store_aligned(&d7_real[2*k + W]);
        }
    }
}

// ============================================================================
// Radix-11 butterfly
// ============================================================================
template<typename T>
inline void radix11_dit(std::complex<T>* data, size_t stride, size_t n) {
    for (size_t i = 0; i < n; i += 11*stride) {
        for (size_t k = 0; k < stride; ++k) {
            std::complex<T> v[11];
            for (size_t j = 0; j < 11; ++j) {
                v[j] = data[i + j*stride + k];
            }
            for (size_t j = 0; j < 11; ++j) {
                std::complex<T> sum = 0;
                for (size_t m = 0; m < 11; ++m) {
                    T angle = -T(2.0 * 3.14159265358979323846) * T(j * m) / T(11);
                    T wr = std::cos(angle);
                    T wi = std::sin(angle);
                    sum += v[m] * std::complex<T>(wr, wi);
                }
                data[i + j*stride + k] = sum;
            }
        }
    }
}

// ============================================================================
// Kernel dispatcher: call appropriate radix kernel
// ============================================================================
template<typename T>
inline void small_fft_dispatch(std::complex<T>* data, size_t stride, size_t radix, size_t n) {
    switch (radix) {
        case 2: radix2_dit<T>(data, stride, n); break;
        case 3: radix3_dit<T>(data, stride, n); break;
        case 4: radix4_dit<T>(data, stride, n); break;
        case 5: radix5_dit<T>(data, stride, n); break;
        case 7: radix7_dit<T>(data, stride, n); break;
        case 8: radix8_dit<T>(data, stride, n); break;
        case 11: radix11_dit<T>(data, stride, n); break;
        default:
            // Unsupported radix; fallback to generic O(n^2) DFT
            for (size_t i = 0; i < n; i += radix*stride) {
                std::complex<T>* block = &data[i];
                for (size_t k = 0; k < stride; ++k) {
                    std::complex<T> tmp[32]; // reasonable max radix
                    for (size_t j = 0; j < radix; ++j) {
                        tmp[j] = block[j*stride + k];
                    }
                    for (size_t j = 0; j < radix; ++j) {
                        std::complex<T> sum = 0;
                        for (size_t m = 0; m < radix; ++m) {
                            T angle = -T(2.0 * 3.14159265358979323846) * T(j * m) / T(radix);
                            T wr = std::cos(angle);
                            T wi = std::sin(angle);
                            sum += tmp[m] * std::complex<T>(wr, wi);
                        }
                        block[j*stride + k] = sum;
                    }
                }
            }
            break;
    }
}

} // namespace kernels
} // namespace fastfft