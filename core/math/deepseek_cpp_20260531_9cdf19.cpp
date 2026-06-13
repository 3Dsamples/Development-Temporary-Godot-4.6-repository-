//File 0061 : core/math/kalman_filter.h
//Linear Kalman filter and Extended Kalman filter (EKF) for state estimation; uses Eigen for matrix operations, fully mathematical with predict/update steps.
#ifndef CORE_MATH_KALMAN_FILTER_H
#define CORE_MATH_KALMAN_FILTER_H

#include "linear_algebra.h"         // provides Eigen types and solve
#include "math_constants.h"
#include <Eigen/Dense>
#include <functional>
#include <cmath>

namespace SimulationMath {
namespace estimation {

using VectorXf = Eigen::VectorXf;
using MatrixXf = Eigen::MatrixXf;

// -----------------------------------------------------------------------------
// 1. Linear Kalman Filter for systems with linear dynamics and measurement
// -----------------------------------------------------------------------------
class KalmanFilter {
public:
    KalmanFilter(int state_dim, int measurement_dim)
        : x_(VectorXf::Zero(state_dim)),
          P_(MatrixXf::Identity(state_dim, state_dim)),
          F_(MatrixXf::Identity(state_dim, state_dim)),
          H_(MatrixXf::Zero(measurement_dim, state_dim)),
          Q_(MatrixXf::Zero(state_dim, state_dim)),
          R_(MatrixXf::Zero(measurement_dim, measurement_dim))
    {}

    // Set state transition matrix F, process noise covariance Q, measurement matrix H, measurement noise R.
    void set_dynamics(const MatrixXf& F, const MatrixXf& Q) { F_ = F; Q_ = Q; }
    void set_measurement(const MatrixXf& H, const MatrixXf& R) { H_ = H; R_ = R; }
    void set_state(const VectorXf& x, const MatrixXf& P) { x_ = x; P_ = P; }

    // Predict step: x = F * x, P = F * P * F^T + Q
    void predict() noexcept {
        x_ = F_ * x_;
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    // Predict with optional control input u and control matrix B
    void predict(const VectorXf& u, const MatrixXf& B) noexcept {
        x_ = F_ * x_ + B * u;
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    // Update step given measurement z: Kalman gain K = P * H^T * (H * P * H^T + R)^-1, x += K*(z - H*x), P = (I - K*H) * P
    void update(const VectorXf& z) noexcept {
        MatrixXf S = H_ * P_ * H_.transpose() + R_;              // innovation covariance
        MatrixXf K = P_ * H_.transpose() * S.inverse();          // Kalman gain
        x_ += K * (z - H_ * x_);                                 // update state
        P_ = (MatrixXf::Identity(P_.rows(), P_.cols()) - K * H_) * P_; // update covariance (Joseph form optional)
    }

    const VectorXf& state() const noexcept { return x_; }
    const MatrixXf& covariance() const noexcept { return P_; }

private:
    VectorXf x_;      // state
    MatrixXf P_;      // estimate error covariance
    MatrixXf F_;      // state transition
    MatrixXf H_;      // measurement matrix
    MatrixXf Q_;      // process noise covariance
    MatrixXf R_;      // measurement noise covariance
};

// -----------------------------------------------------------------------------
// 2. Extended Kalman Filter (EKF) for non‑linear systems
// -----------------------------------------------------------------------------
class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter(int state_dim, int measurement_dim)
        : x_(VectorXf::Zero(state_dim)),
          P_(MatrixXf::Identity(state_dim, state_dim)),
          state_dim_(state_dim), meas_dim_(measurement_dim)
    {}

    // Set initial state estimate and covariance
    void set_state(const VectorXf& x, const MatrixXf& P) { x_ = x; P_ = P; }

    // Predict step using non‑linear state transition function f and its Jacobian F (both functions of current state)
    // f: state -> next state, F: Jacobian df/dx evaluated at current state
    void predict(const std::function<VectorXf(const VectorXf&)>& f,
                 const std::function<MatrixXf(const VectorXf&)>& F_jacobian,
                 const MatrixXf& Q) noexcept {
        // Propagate state through non‑linear function
        x_ = f(x_);
        // Linearize dynamics matrix around predicted state
        MatrixXf F = F_jacobian(x_);
        P_ = F * P_ * F.transpose() + Q;
    }

    // Predict with control input u and non‑linear dynamics g(x,u) and Jacobians A (wrt x), B (wrt u)
    void predict_with_control(const std::function<VectorXf(const VectorXf&, const VectorXf&)>& g,
                              const std::function<MatrixXf(const VectorXf&, const VectorXf&)>& A_jac,
                              const std::function<MatrixXf(const VectorXf&, const VectorXf&)>& B_jac,
                              const VectorXf& u, const MatrixXf& Q) noexcept {
        // State propagation
        x_ = g(x_, u);
        MatrixXf A = A_jac(x_, u);
        MatrixXf B = B_jac(x_, u);
        P_ = A * P_ * A.transpose() + B * MatrixXf::Zero(B.cols(), B.cols()) * B.transpose() + Q; // assume control noise separate, here just Q
    }

    // Update step using measurement function h and its Jacobian H (h: state -> predicted measurement)
    void update(const VectorXf& z,
                const std::function<VectorXf(const VectorXf&)>& h,
                const std::function<MatrixXf(const VectorXf&)>& H_jacobian,
                const MatrixXf& R) noexcept {
        VectorXf z_pred = h(x_);
        MatrixXf H = H_jacobian(x_);
        // Innovation covariance
        MatrixXf S = H * P_ * H.transpose() + R;
        // Kalman gain
        MatrixXf K = P_ * H.transpose() * S.inverse();
        // Update state
        x_ += K * (z - z_pred);
        // Update covariance using Joseph form for better numerical stability
        MatrixXf I = MatrixXf::Identity(state_dim_, state_dim_);
        MatrixXf I_KH = I - K * H;
        P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
    }

    const VectorXf& state() const noexcept { return x_; }
    const MatrixXf& covariance() const noexcept { return P_; }

private:
    VectorXf x_;
    MatrixXf P_;
    int state_dim_;
    int meas_dim_;
};

} // namespace estimation
} // namespace SimulationMath

#endif // CORE_MATH_KALMAN_FILTER_H