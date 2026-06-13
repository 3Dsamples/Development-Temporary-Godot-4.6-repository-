// system name : onetbb-warp
// File 0018 : core/math/filters.h
// Description : Digital filters (FIR, IIR), PID controllers, Kalman, smoothing for simulation.

#ifndef __TBB_WARP_CORE_MATH_FILTERS_H
#define __TBB_WARP_CORE_MATH_FILTERS_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include <cmath>
#include <vector>
#include <deque>
#include <algorithm>
#include <numeric>
#include <type_traits>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Exponential Moving Average (low‑pass)
// ============================================================

template<typename T>
struct exp_avg {
    T value;
    float alpha;
    exp_avg(float a = 0.1f) : value(T(0)), alpha(a) {}
    T update(T input) noexcept { value = value * (1.0f - alpha) + input * alpha; return value; }
    T get() const noexcept { return value; }
    void reset(T v = T(0)) noexcept { value = v; }
};

// ============================================================
// One‑Euro filter (adaptive low‑pass with speed‑dependent cutoff)
// ============================================================

template<typename T>
struct one_euro_filter {
    T value;
    T derivative;
    float min_cutoff;
    float beta;
    float d_cutoff;
    float prev_time;

    one_euro_filter(float min_cutoff_ = 1.0f, float beta_ = 0.0f, float d_cutoff_ = 1.0f)
        : value(T(0)), derivative(T(0)), min_cutoff(min_cutoff_), beta(beta_),
          d_cutoff(d_cutoff_), prev_time(0.0f) {}

    T update(T input, float time) noexcept {
        float dt = time - prev_time;
        if (dt < 1e-6f) dt = 1e-6f;
        float dx = length(value - input);
        float rate = dx / dt;
        float cutoff = min_cutoff + beta * rate;
        float tau = 1.0f / (TAU_F * cutoff);
        float alpha = dt / (tau + dt);
        T new_deriv = (input - value) / dt;
        derivative = derivative + (new_deriv - derivative) * alpha;
        float d_rate = length(derivative);
        float d_tau = 1.0f / (TAU_F * d_cutoff);
        float d_alpha = dt / (d_tau + dt);
        value = value + (input - value) * d_alpha;
        prev_time = time;
        return value;
    }
    void reset(T v = T(0)) noexcept { value = v; derivative = T(0); }
};

// ============================================================
// Biquad filter (IIR) – direct form I
// ============================================================

struct biquad_params {
    float b0, b1, b2, a1, a2;
};

inline biquad_params lowpass_biquad(float cutoff, float Q, float sample_rate) {
    float w0 = TAU_F * cutoff / sample_rate;
    float cos_w0 = std::cos(w0);
    float sin_w0 = std::sin(w0);
    float alpha = sin_w0 / (2.0f * Q);
    float b0 = (1.0f - cos_w0) * 0.5f;
    float b1 = 1.0f - cos_w0;
    float b2 = (1.0f - cos_w0) * 0.5f;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * cos_w0;
    float a2 = 1.0f - alpha;
    return {b0/a0, b1/a0, b2/a0, a1/a0, a2/a0};
}

inline biquad_params highpass_biquad(float cutoff, float Q, float sample_rate) {
    float w0 = TAU_F * cutoff / sample_rate;
    float cos_w0 = std::cos(w0);
    float sin_w0 = std::sin(w0);
    float alpha = sin_w0 / (2.0f * Q);
    float b0 = (1.0f + cos_w0) * 0.5f;
    float b1 = -(1.0f + cos_w0);
    float b2 = (1.0f + cos_w0) * 0.5f;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * cos_w0;
    float a2 = 1.0f - alpha;
    return {b0/a0, b1/a0, b2/a0, a1/a0, a2/a0};
}

inline biquad_params bandpass_biquad(float cutoff, float Q, float sample_rate) {
    float w0 = TAU_F * cutoff / sample_rate;
    float cos_w0 = std::cos(w0);
    float sin_w0 = std::sin(w0);
    float alpha = sin_w0 / (2.0f * Q);
    float b0 = alpha;
    float b1 = 0.0f;
    float b2 = -alpha;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * cos_w0;
    float a2 = 1.0f - alpha;
    return {b0/a0, b1/a0, b2/a0, a1/a0, a2/a0};
}

inline biquad_params notch_biquad(float cutoff, float Q, float sample_rate) {
    float w0 = TAU_F * cutoff / sample_rate;
    float cos_w0 = std::cos(w0);
    float sin_w0 = std::sin(w0);
    float alpha = sin_w0 / (2.0f * Q);
    float b0 = 1.0f;
    float b1 = -2.0f * cos_w0;
    float b2 = 1.0f;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * cos_w0;
    float a2 = 1.0f - alpha;
    return {b0/a0, b1/a0, b2/a0, a1/a0, a2/a0};
}

template<typename T>
struct biquad {
    biquad_params p;
    T x1, x2, y1, y2;

    biquad(const biquad_params& params) : p(params), x1(T(0)), x2(T(0)), y1(T(0)), y2(T(0)) {}

    T update(T input) noexcept {
        T output = p.b0 * input + p.b1 * x1 + p.b2 * x2 - p.a1 * y1 - p.a2 * y2;
        x2 = x1; x1 = input;
        y2 = y1; y1 = output;
        return output;
    }

    void reset() noexcept { x1 = x2 = y1 = y2 = T(0); }
};

// ============================================================
// PID controller
// ============================================================

template<typename T>
struct pid_controller {
    T kp, ki, kd;
    T integral;
    T prev_error;
    T output_min, output_max;
    bool clamp;

    pid_controller(T p, T i, T d, T min_out = T(-1e6), T max_out = T(1e6), bool cl = false)
        : kp(p), ki(i), kd(d), integral(T(0)), prev_error(T(0)),
          output_min(min_out), output_max(max_out), clamp(cl) {}

    T update(T setpoint, T measurement, float dt) noexcept {
        T error = setpoint - measurement;
        integral += error * dt;
        if (clamp) integral = math::clamp(integral, output_min / (ki + T(1e-12)), output_max / (ki + T(1e-12)));
        T derivative = (dt > 1e-9f) ? (error - prev_error) / dt : T(0);
        T output = kp * error + ki * integral + kd * derivative;
        if (clamp) output = math::clamp(output, output_min, output_max);
        prev_error = error;
        return output;
    }

    void reset(T initial_integral = T(0)) noexcept { integral = initial_integral; prev_error = T(0); }
};

// ============================================================
// Kalman filter (1D – position + velocity)
// ============================================================

template<typename T>
struct kalman_1d {
    T x;        // state: position
    T v;        // state: velocity
    T p11, p12, p21, p22; // covariance matrix
    T q_pos, q_vel, r_meas;

    kalman_1d(T process_noise_pos = T(0.01), T process_noise_vel = T(0.1), T measurement_noise = T(1.0))
        : x(T(0)), v(T(0)), p11(T(1)), p12(T(0)), p21(T(0)), p22(T(1)),
          q_pos(process_noise_pos), q_vel(process_noise_vel), r_meas(measurement_noise) {}

    T update(T measurement, float dt) noexcept {
        // Predict
        x = x + v * dt;
        v = v;
        p11 = p11 + T(2) * dt * p12 + dt * dt * p22 + q_pos;
        p12 = p12 + dt * p22;
        p21 = p12;
        p22 = p22 + q_vel;
        // Update
        T k1 = p11 / (p11 + r_meas);
        T k2 = p21 / (p11 + r_meas);
        T y = measurement - x;
        x = x + k1 * y;
        v = v + k2 * y;
        p11 = (T(1) - k1) * p11;
        p12 = (T(1) - k1) * p12;
        p21 = p12;
        p22 = p22 - k2 * p21;
        return x;
    }
    T velocity() const noexcept { return v; }
    void reset(T pos = T(0), T vel = T(0)) noexcept { x = pos; v = vel; p11=T(1); p12=T(0); p21=T(0); p22=T(1); }
};

// ============================================================
// Kalman filter (3D – position, velocity, acceleration per axis)
// ============================================================

template<typename T>
struct kalman_3d {
    vector3<T> pos, vel, acc;
    T p[3][3];
    T q, r;

    kalman_3d(T process_noise = T(0.01), T measurement_noise = T(1.0))
        : pos(), vel(), acc(), q(process_noise), r(measurement_noise) {
        for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) p[i][j] = (i==j) ? T(1) : T(0);
    }

    vector3<T> update(const vector3<T>& measurement, float dt) noexcept {
        for (int i=0; i<3; ++i) {
            T x_arr[3] = {pos[i], vel[i], acc[i]};
            T P[3][3];
            for (int r0=0; r0<3; ++r0) for (int c0=0; c0<3; ++c0) P[r0][c0] = (r0==c0) ? p[i][i] : T(0);
            // Predict
            x_arr[0] += x_arr[1] * dt + x_arr[2] * T(0.5) * dt * dt;
            x_arr[1] += x_arr[2] * dt;
            T f[3][3] = {{1, dt, 0.5f*dt*dt},{0,1,dt},{0,0,1}};
            T ft[3][3] = {{1,0,0},{dt,1,0},{0.5f*dt*dt,dt,1}};
            T fpf[3][3];
            for (int r0=0; r0<3; ++r0) for (int c0=0; c0<3; ++c0) {
                fpf[r0][c0] = T(0);
                for (int k=0; k<3; ++k) fpf[r0][c0] += f[r0][k] * P[k][c0];
            }
            T Q[3][3] = {{q,0,0},{0,q,0},{0,0,q}};
            for (int r0=0; r0<3; ++r0) for (int c0=0; c0<3; ++c0) {
                P[r0][c0] = T(0);
                for (int k=0; k<3; ++k) P[r0][c0] += fpf[r0][k] * ft[k][c0];
                P[r0][c0] += Q[r0][c0];
            }
            T K[3];
            T S = P[0][0] + r;
            K[0] = P[0][0] / S;
            K[1] = P[1][0] / S;
            K[2] = P[2][0] / S;
            T y = measurement[i] - x_arr[0];
            x_arr[0] += K[0] * y;
            x_arr[1] += K[1] * y;
            x_arr[2] += K[2] * y;
            T I_KH[3][3] = {{1-K[0],0,0},{-K[1],1,0},{-K[2],0,1}};
            for (int r0=0; r0<3; ++r0) for (int c0=0; c0<3; ++c0) {
                T sum = T(0);
                for (int k=0; k<3; ++k) sum += I_KH[r0][k] * P[k][c0];
                p[r0][c0] = sum;
            }
            pos[i] = x_arr[0];
            vel[i] = x_arr[1];
            acc[i] = x_arr[2];
        }
        return pos;
    }
};

// ============================================================
// Moving average filter (FIR)
// ============================================================

template<typename T>
struct moving_average_filter {
    std::deque<T> window;
    std::size_t max_size;
    T sum;

    explicit moving_average_filter(std::size_t size) : max_size(size), sum(T(0)) {}

    T update(T value) noexcept {
        window.push_back(value);
        sum += value;
        if (window.size() > max_size) { sum -= window.front(); window.pop_front(); }
        return sum / static_cast<T>(window.size());
    }
    void reset() noexcept { window.clear(); sum = T(0); }
};

// ============================================================
// Median filter
// ============================================================

template<typename T>
struct median_filter {
    std::deque<T> buffer;
    std::size_t max_size;

    explicit median_filter(std::size_t size) : max_size(size) {}

    T update(T value) noexcept {
        buffer.push_back(value);
        if (buffer.size() > max_size) buffer.pop_front();
        std::vector<T> sorted(buffer.begin(), buffer.end());
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size()/2, sorted.end());
        return sorted[sorted.size()/2];
    }
    void reset() noexcept { buffer.clear(); }
};

// ============================================================
// Savitzky‑Golay filter (smoothing differentiation)
// ============================================================

template<typename T>
struct savitzky_golay {
    std::deque<T> buffer;
    std::size_t window;
    std::vector<float> coeff;
    int order;

    savitzky_golay(std::size_t window_size = 5, int poly_order = 2) : window(window_size), order(poly_order) {
        compute_coefficients();
    }

    void compute_coefficients() {
        std::size_t m = (window - 1) / 2;
        // Compute smoothing coefficients via least squares (simple: use pre‑computed for order 2, window 5)
        // For simplicity, provide common Savitzky‑Golay coefficients for window=5, order=2
        if (window == 5 && order == 2) coeff = {-3.0f/35.0f, 12.0f/35.0f, 17.0f/35.0f, 12.0f/35.0f, -3.0f/35.0f};
        else if (window == 7 && order == 2) coeff = {-2.0f/21.0f, 3.0f/21.0f, 6.0f/21.0f, 7.0f/21.0f, 6.0f/21.0f, 3.0f/21.0f, -2.0f/21.0f};
        else if (window == 9 && order == 2) coeff = {-21.0f/231.0f, 14.0f/231.0f, 39.0f/231.0f, 54.0f/231.0f, 59.0f/231.0f, 54.0f/231.0f, 39.0f/231.0f, 14.0f/231.0f, -21.0f/231.0f};
        else {
            // Fallback: uniform coefficients
            coeff.assign(window, 1.0f / window);
        }
    }

    T update(T value) noexcept {
        buffer.push_back(value);
        if (buffer.size() > window) buffer.pop_front();
        T result = T(0);
        if (buffer.size() == window) {
            for (std::size_t i = 0; i < window; ++i)
                result = result + buffer[i] * coeff[i];
        } else {
            result = value;
        }
        return result;
    }

    void reset() noexcept { buffer.clear(); }
};

// ============================================================
// Hysteresis (Schmitt trigger with deadband)
// ============================================================

template<typename T>
struct hysteresis {
    T low, high;
    bool state;

    hysteresis(T low_threshold, T high_threshold, bool initial = false) noexcept
        : low(low_threshold), high(high_threshold), state(initial) {}

    bool update(T value) noexcept {
        if (state && value < low) state = false;
        else if (!state && value > high) state = true;
        return state;
    }
};

// ============================================================
// Rate limiter (slew rate)
// ============================================================

template<typename T>
struct rate_limiter {
    T value;
    T max_rate_up;
    T max_rate_down;

    rate_limiter(T max_up, T max_down, T initial = T(0))
        : value(initial), max_rate_up(max_up), max_rate_down(max_down) {}

    T update(T target, float dt) noexcept {
        T diff = target - value;
        T max_change_up = max_rate_up * dt;
        T max_change_down = max_rate_down * dt;
        if (diff > max_change_up) diff = max_change_up;
        else if (diff < -max_change_down) diff = -max_change_down;
        value += diff;
        return value;
    }
};

// ============================================================
// Butterworth filter (N‑th order, low‑pass) implemented as cascaded biquads
// ============================================================

template<typename T>
struct butterworth_lp {
    std::vector<biquad<T>> stages;

    butterworth_lp(int order, float cutoff, float sample_rate) {
        for (int k = 1; k <= order/2; ++k) {
            float theta = PI_F * (2.0f * k + order - 1) / (2.0f * order);
            float s_real = -std::sin(theta);
            float s_imag = std::cos(theta);
            float w0 = TAU_F * cutoff / sample_rate;
            float warped = 2.0f * sample_rate * std::tan(w0 * 0.5f);
            float norm = warped * warped;
            float a0 = norm - 2.0f * s_real * warped + s_real*s_real + s_imag*s_imag;
            float b0 = norm / a0;
            float b1 = 2.0f * norm / a0;
            float b2 = norm / a0;
            float a1 = 2.0f * (norm - (s_real*s_real + s_imag*s_imag)) / a0;
            float a2 = (norm + 2.0f * s_real * warped + s_real*s_real + s_imag*s_imag) / a0;
            stages.emplace_back(biquad_params{b0, b1, b2, a1, a2});
        }
    }

    T update(T input) noexcept {
        for (auto& s : stages) input = s.update(input);
        return input;
    }
    void reset() noexcept { for (auto& s : stages) s.reset(); }
};

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_FILTERS_H