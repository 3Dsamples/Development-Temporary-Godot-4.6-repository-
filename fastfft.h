// include/fastfft/fastfft.h
#ifndef FASTFFT_H
#define FASTFFT_H

#include <stddef.h>

/* C99 complex types are optional; use float complex and double complex if available */
#ifdef __STDC_NO_COMPLEX__
  /* Complex not supported in this C implementation; use separate real/imag arrays */
  #define FASTFFT_NO_COMPLEX 1
#else
  #include <complex.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Opaque plan handle
 * -------------------------------------------------------------------------- */
typedef struct fastfft_plan fastfft_plan;

/* --------------------------------------------------------------------------
 * Transform direction
 * -------------------------------------------------------------------------- */
typedef enum {
    FASTFFT_FORWARD = -1,
    FASTFFT_BACKWARD = 1
} fastfft_direction;

/* --------------------------------------------------------------------------
 * Transform type
 * -------------------------------------------------------------------------- */
typedef enum {
    FASTFFT_C2C,    /* Complex to complex                      */
    FASTFFT_R2C,    /* Real to complex (half‑complex output)   */
    FASTFFT_C2R     /* Complex to real (half‑complex input)    */
} fastfft_transform_type;

/* --------------------------------------------------------------------------
 * Plan creation
 * -------------------------------------------------------------------------- */

/* Create a complex‑to‑complex plan.
 * length : transform size (any positive integer)
 * dir    : FASTFFT_FORWARD or FASTFFT_BACKWARD
 * Returns NULL on failure. */
fastfft_plan* fastfft_plan_create_c2c(size_t length, fastfft_direction dir);

/* Create a real‑to‑complex plan (forward transform).
 * length : number of real samples
 * Returns NULL on failure. */
fastfft_plan* fastfft_plan_create_r2c(size_t length);

/* Create a complex‑to‑real plan (inverse transform).
 * length : number of real samples (same as original R2C length)
 * Returns NULL on failure. */
fastfft_plan* fastfft_plan_create_c2r(size_t length);

/* Destroy a plan and free all associated memory. */
void fastfft_plan_destroy(fastfft_plan* plan);

/* --------------------------------------------------------------------------
 * Execution (single precision / float)
 * -------------------------------------------------------------------------- */

/* Complex‑to‑complex, out‑of‑place.
 * plan : plan created with fastfft_plan_create_c2c
 * in   : input array of size length
 * out  : output array of size length (may alias in) */
void fastfft_execute_c2c_f32(const fastfft_plan* plan,
                             const float _Complex* in,
                             float _Complex* out);

/* Complex‑to‑complex, in‑place.
 * plan : plan created with fastfft_plan_create_c2c
 * data : input/output array of size length */
void fastfft_execute_c2c_inplace_f32(const fastfft_plan* plan,
                                     float _Complex* data);

/* Real‑to‑complex.
 * plan : plan created with fastfft_plan_create_r2c
 * in   : real input array of size length
 * out  : complex output array of size length/2 + 1 */
void fastfft_execute_r2c_f32(const fastfft_plan* plan,
                             const float* in,
                             float _Complex* out);

/* Complex‑to‑real.
 * plan : plan created with fastfft_plan_create_c2r
 * in   : complex input array of size length/2 + 1
 * out  : real output array of size length */
void fastfft_execute_c2r_f32(const fastfft_plan* plan,
                             const float _Complex* in,
                             float* out);

/* --------------------------------------------------------------------------
 * Execution (double precision / double)
 * -------------------------------------------------------------------------- */

/* Complex‑to‑complex, out‑of‑place. */
void fastfft_execute_c2c_f64(const fastfft_plan* plan,
                             const double _Complex* in,
                             double _Complex* out);

/* Complex‑to‑complex, in‑place. */
void fastfft_execute_c2c_inplace_f64(const fastfft_plan* plan,
                                     double _Complex* data);

/* Real‑to‑complex. */
void fastfft_execute_r2c_f64(const fastfft_plan* plan,
                             const double* in,
                             double _Complex* out);

/* Complex‑to‑real. */
void fastfft_execute_c2r_f64(const fastfft_plan* plan,
                             const double _Complex* in,
                             double* out);

/* --------------------------------------------------------------------------
 * One‑shot convenience functions (no plan reuse)
 * -------------------------------------------------------------------------- */

/* Single‑precision complex FFT. */
void fastfft_fft_f32(const float _Complex* in,
                     float _Complex* out,
                     size_t length,
                     fastfft_direction dir);

/* Double‑precision complex FFT. */
void fastfft_fft_f64(const double _Complex* in,
                     double _Complex* out,
                     size_t length,
                     fastfft_direction dir);

/* Single‑precision real FFT. */
void fastfft_rfft_f32(const float* in,
                      float _Complex* out,
                      size_t length);

/* Double‑precision real FFT. */
void fastfft_rfft_f64(const double* in,
                      double _Complex* out,
                      size_t length);

/* Single‑precision inverse real FFT. */
void fastfft_irfft_f32(const float _Complex* in,
                       float* out,
                       size_t length);

/* Double‑precision inverse real FFT. */
void fastfft_irfft_f64(const double _Complex* in,
                       double* out,
                       size_t length);

/* --------------------------------------------------------------------------
 * Plan properties
 * -------------------------------------------------------------------------- */

/* Return the transform length. */
size_t fastfft_plan_get_size(const fastfft_plan* plan);

/* Return the transform type (C2C, R2C, or C2R). */
fastfft_transform_type fastfft_plan_get_type(const fastfft_plan* plan);

/* Return the transform direction (Forward or Backward). */
fastfft_direction fastfft_plan_get_direction(const fastfft_plan* plan);

/* --------------------------------------------------------------------------
 * Utilities
 * -------------------------------------------------------------------------- */

/* Return the required alignment (in bytes) for optimal SIMD performance.
 * Use aligned_alloc or _aligned_malloc to satisfy this requirement. */
size_t fastfft_alignment(void);

/* Return the smallest highly composite size >= n.
 * Such sizes have only small prime factors (2,3,5,7,11,13) and run fastest. */
size_t fastfft_next_good_size(size_t n);

/* Return the library version string. */
const char* fastfft_version(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTFFT_H */