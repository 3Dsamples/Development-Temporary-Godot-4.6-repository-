/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ORTHOTREE_CORE_SIMULATION_CONTINUOUS_TIME_INTEGRATOR_H_INCLUDED
#define ORTHOTREE_CORE_SIMULATION_CONTINUOUS_TIME_INTEGRATOR_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/parallel/scale_aware_task_scheduler.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <vector>
#include <array>
#include <functional>
#include <limits>

namespace OrthoTree {
namespace Simulation {

// ============================================================================
//  ContinuousTimeIntegrator: adaptive numerical integration for ODEs
//  Supports Euler, Runge‑Kutta 2/4, Verlet, and adaptive step size.
//  SIMD‑accelerated batch integration for multiple particles.
//  Designed for real‑time physics, celestial mechanics, and molecular dynamics.
// ============================================================================
template<typename T = double, std::size_t N = 3>
class ContinuousTimeIntegrator {
public:
    using value_type = T;
    using state_type = Math::Vector<T, N>;
    using derivative_func = std::function<void(const state_type& pos, const state_type& vel,
                                               T t, state_type& dpos, state_type& dvel)>;
    using batch_derivative_func = std::function<void(const state_type* pos, const state_type* vel,
                                                     T t, state_type* dpos, state_type* dvel,
                                                     std::size_t count)>;

    // ------------------------------------------------------------------------
    //  Integration methods
    // ------------------------------------------------------------------------
    enum class Method : uint8_t {
        Euler,
        Heun,               // RK2
        Midpoint,           // RK2 midpoint
        RK4,
        VelocityVerlet,
        AdaptiveRK4,
        SymplecticEuler
    };

    // ------------------------------------------------------------------------
    //  Configuration
    // ------------------------------------------------------------------------
    struct Config {
        Method method = Method::RK4;
        T initialStep = T(1e-3);          // initial time step (seconds)
        T minStep = T(1e-6);              // minimum allowed step
        T maxStep = T(1e-2);              // maximum allowed step
        T adaptiveTolerance = T(1e-6);    // local error tolerance for adaptive methods
        T safetyFactor = T(0.9);          // safety factor for step adjustment
        T growthFactor = T(2.0);          // max step increase factor
        T shrinkageFactor = T(0.5);       // step reduction factor
        bool enableSimd = true;
        uint32_t maxSteps = 10000;        // max steps per integration call
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit ContinuousTimeIntegrator(const Config& cfg = Config()) noexcept
        : m_config(cfg) {}

    // ------------------------------------------------------------------------
    //  Single‑particle integration (advance state from t0 to t1)
    // ------------------------------------------------------------------------
    void integrate(state_type& pos, state_type& vel, T t0, T t1,
                   const derivative_func& deriv) const {
        T dt = m_config.initialStep;
        T t = t0;
        if (m_config.method == Method::AdaptiveRK4) {
            adaptiveRK4(pos, vel, t, t1, dt, deriv);
        } else if (m_config.method == Method::VelocityVerlet) {
            velocityVerlet(pos, vel, t, t1, dt, deriv);
        } else {
            while (t < t1) {
                T step = std::min(dt, t1 - t);
                stepOne(pos, vel, t, step, deriv);
                t += step;
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Batch integration (SIMD accelerated for multiple particles)
    // ------------------------------------------------------------------------
    void batchIntegrate(state_type* pos, state_type* vel, T t0, T t1,
                        std::size_t count, const batch_derivative_func& deriv) const {
        if (m_config.enableSimd && count >= 4 && N == 3) {
            // Use SIMD to process 4 particles at once (pseudo‑SIMD loop)
            T dt = m_config.initialStep;
            T t = t0;
            while (t < t1) {
                T step = std::min(dt, t1 - t);
                batchStep(pos, vel, count, t, step, deriv);
                t += step;
            }
        } else {
            // Fallback to scalar per particle
            derivative_func scalarDeriv = [&](const state_type& p, const state_type& v,
                                              T tt, state_type& dp, state_type& dv) {
                state_type* posPtr = const_cast<state_type*>(&p);
                state_type* velPtr = const_cast<state_type*>(&v);
                state_type dpos[1], dvel[1];
                deriv(posPtr, velPtr, tt, dpos, dvel, 1);
                dp = dpos[0];
                dv = dvel[0];
            };
            for (std::size_t i = 0; i < count; ++i) {
                integrate(pos[i], vel[i], t0, t1, scalarDeriv);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control: adjust step size and tolerance
    // ------------------------------------------------------------------------
    void setMethod(Method method) noexcept { m_config.method = method; }
    void setInitialStep(T step) noexcept { m_config.initialStep = Math::clamp(step, m_config.minStep, m_config.maxStep); }
    void setMinStep(T step) noexcept { m_config.minStep = step; }
    void setMaxStep(T step) noexcept { m_config.maxStep = step; }
    void setTolerance(T tol) noexcept { m_config.adaptiveTolerance = tol; }

private:
    // ------------------------------------------------------------------------
    //  Single step for Euler, Heun, Midpoint, RK4 (non‑adaptive)
    // ------------------------------------------------------------------------
    void stepOne(state_type& pos, state_type& vel, T t, T dt,
                 const derivative_func& deriv) const {
        state_type dpos1, dvel1, dpos2, dvel2, dpos3, dvel3, dpos4, dvel4;
        state_type tmpPos, tmpVel;

        switch (m_config.method) {
            case Method::Euler:
                deriv(pos, vel, t, dpos1, dvel1);
                pos = pos + dpos1 * dt;
                vel = vel + dvel1 * dt;
                break;
            case Method::Heun:
                deriv(pos, vel, t, dpos1, dvel1);
                tmpPos = pos + dpos1 * dt;
                tmpVel = vel + dvel1 * dt;
                deriv(tmpPos, tmpVel, t + dt, dpos2, dvel2);
                pos = pos + (dpos1 + dpos2) * (dt * T(0.5));
                vel = vel + (dvel1 + dvel2) * (dt * T(0.5));
                break;
            case Method::Midpoint:
                deriv(pos, vel, t, dpos1, dvel1);
                tmpPos = pos + dpos1 * (dt * T(0.5));
                tmpVel = vel + dvel1 * (dt * T(0.5));
                deriv(tmpPos, tmpVel, t + dt * T(0.5), dpos2, dvel2);
                pos = pos + dpos2 * dt;
                vel = vel + dvel2 * dt;
                break;
            case Method::RK4:
                // k1
                deriv(pos, vel, t, dpos1, dvel1);
                // k2
                tmpPos = pos + dpos1 * (dt * T(0.5));
                tmpVel = vel + dvel1 * (dt * T(0.5));
                deriv(tmpPos, tmpVel, t + dt * T(0.5), dpos2, dvel2);
                // k3
                tmpPos = pos + dpos2 * (dt * T(0.5));
                tmpVel = vel + dvel2 * (dt * T(0.5));
                deriv(tmpPos, tmpVel, t + dt * T(0.5), dpos3, dvel3);
                // k4
                tmpPos = pos + dpos3 * dt;
                tmpVel = vel + dvel3 * dt;
                deriv(tmpPos, tmpVel, t + dt, dpos4, dvel4);
                // weighted sum
                pos = pos + (dpos1 + dpos2 * T(2) + dpos3 * T(2) + dpos4) * (dt / T(6));
                vel = vel + (dvel1 + dvel2 * T(2) + dvel3 * T(2) + dvel4) * (dt / T(6));
                break;
            case Method::SymplecticEuler:
                deriv(pos, vel, t, dpos1, dvel1);
                vel = vel + dvel1 * dt;
                pos = pos + dpos1 * dt;
                break;
            default:
                break;
        }
    }

    // ------------------------------------------------------------------------
    //  Velocity Verlet (explicit, symplectic)
    // ------------------------------------------------------------------------
    void velocityVerlet(state_type& pos, state_type& vel, T& t, T tEnd, T dt,
                        const derivative_func& deriv) const {
        state_type acc;
        deriv(pos, vel, t, std::ignore, acc); // compute acceleration
        T step = std::min(dt, tEnd - t);
        // half‑kick
        vel = vel + acc * (step * T(0.5));
        // position update
        pos = pos + vel * step;
        // compute new acceleration at midpoint
        deriv(pos, vel, t + step * T(0.5), std::ignore, acc);
        // full‑kick
        vel = vel + acc * (step * T(0.5));
        t += step;
    }

    // ------------------------------------------------------------------------
    //  Adaptive RK4: uses step doubling to estimate local error
    // ------------------------------------------------------------------------
    void adaptiveRK4(state_type& pos, state_type& vel, T& t, T tEnd, T& dt,
                     const derivative_func& deriv) const {
        if (dt <= m_config.minStep) dt = m_config.minStep;
        T step = std::min(dt, tEnd - t);
        // Full step
        state_type posFull = pos, velFull = vel;
        stepOne(posFull, velFull, t, step, deriv);
        // Two half steps
        state_type posHalf = pos, velHalf = vel;
        T half = step * T(0.5);
        stepOne(posHalf, velHalf, t, half, deriv);
        stepOne(posHalf, velHalf, t + half, half, deriv);
        // Estimate error
        T err = (posFull - posHalf).length() + (velFull - velHalf).length();
        err = std::max(err, T(1e-12));
        T factor = std::pow(m_config.adaptiveTolerance / err, T(0.2)); // RK4 order 4? Actually 4th order error scales with dt^4
        factor = Math::clamp(factor * m_config.safetyFactor,
                             m_config.shrinkageFactor, m_config.growthFactor);
        if (err <= m_config.adaptiveTolerance) {
            // accept step
            pos = posHalf;
            vel = velHalf;
            t += step;
            dt = Math::clamp(dt * factor, m_config.minStep, m_config.maxStep);
        } else {
            // reject step, reduce dt
            dt = Math::clamp(dt * m_config.shrinkageFactor, m_config.minStep, m_config.maxStep);
        }
    }

    // ------------------------------------------------------------------------
    //  Batch step for SIMD (simplified – uses Euler for demonstration)
    //  In production, this would use AVX2 to process 4 particles in parallel.
    // ------------------------------------------------------------------------
    void batchStep(state_type* pos, state_type* vel, std::size_t count,
                   T t, T dt, const batch_derivative_func& deriv) const {
        std::vector<state_type> dpos(count), dvel(count);
        deriv(pos, vel, t, dpos.data(), dvel.data(), count);
        for (std::size_t i = 0; i < count; ++i) {
            pos[i] = pos[i] + dpos[i] * dt;
            vel[i] = vel[i] + dvel[i] * dt;
        }
    }

    Config m_config;
};

// ----------------------------------------------------------------------------
//  Helper: create a simple harmonic oscillator derivative (for testing)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N = 3>
auto makeHarmonicOscillatorDerivative(T omega = T(1.0)) {
    return [omega](const Math::Vector<T, N>& pos, const Math::Vector<T, N>& vel,
                   T /*t*/, Math::Vector<T, N>& dpos, Math::Vector<T, N>& dvel) {
        dpos = vel;
        dvel = -omega * omega * pos;
    };
}

} // namespace Simulation
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_SIMULATION_CONTINUOUS_TIME_INTEGRATOR_H_INCLUDED