//File 0064 : core/math/moving_average.h
//Efficient moving average algorithms (SMA, EMA, DEMA, WMA) with incremental updates and windowed variance for real‑time signal processing.
#ifndef CORE_MATH_MOVING_AVERAGE_H
#define CORE_MATH_MOVING_AVERAGE_H

#include <cstdint>
#include <vector>
#include <deque>
#include <cmath>
#include <numeric>

namespace SimulationMath {
namespace moving_avg {

// -----------------------------------------------------------------------------
// 1. Simple Moving Average (fixed window) using a ring buffer for O(1) per sample
// -----------------------------------------------------------------------------
class SimpleMovingAverage {
public:
    explicit SimpleMovingAverage(size_t window_size) noexcept
        : window_size_(window_size), buffer_(window_size, 0.0f), sum_(0.0f), index_(0), count_(0) {}

    // Add new value, returns current average
    float add(float value) noexcept {
        if (window_size_ == 0) return value;
        if (count_ < window_size_) {
            // Filling phase
            buffer_[index_] = value;
            sum_ += value;
            ++count_;
            index_ = (index_ + 1) % window_size_;
            return sum_ / count_;
        } else {
            // Sliding window
            sum_ -= buffer_[index_];
            buffer_[index_] = value;
            sum_ += value;
            index_ = (index_ + 1) % window_size_;
            return sum_ / window_size_;
        }
    }

    float current() const noexcept { return (count_ == 0) ? 0.0f : sum_ / count_; }
    void reset() noexcept { sum_ = 0.0f; index_ = 0; count_ = 0; }

private:
    size_t window_size_;
    std::vector<float> buffer_;
    float sum_;
    size_t index_;
    size_t count_;
};

// -----------------------------------------------------------------------------
// 2. Cumulative Moving Average (all samples)
// -----------------------------------------------------------------------------
class CumulativeMovingAverage {
public:
    CumulativeMovingAverage() noexcept : mean_(0.0f), n_(0) {}

    float add(float value) noexcept {
        ++n_;
        mean_ += (value - mean_) / n_;
        return mean_;
    }

    float current() const noexcept { return mean_; }
    size_t count() const noexcept { return n_; }

private:
    float mean_;
    size_t n_;
};

// -----------------------------------------------------------------------------
// 3. Exponential Moving Average (EMA)
//    alpha = 2/(N+1) for a roughly equivalent window of N samples
// -----------------------------------------------------------------------------
class ExponentialMovingAverage {
public:
    explicit ExponentialMovingAverage(float alpha) noexcept : alpha_(alpha), value_(0.0f), initialized_(false) {}

    float add(float x) noexcept {
        if (!initialized_) {
            value_ = x;
            initialized_ = true;
        } else {
            value_ = alpha_ * x + (1.0f - alpha_) * value_;
        }
        return value_;
    }

    float current() const noexcept { return value_; }
    void set_alpha(float alpha) noexcept { alpha_ = alpha; }

private:
    float alpha_;
    float value_;
    bool initialized_;
};

// -----------------------------------------------------------------------------
// 4. Double Exponential Moving Average (DEMA) – reduces lag
//    DEMA = 2 * EMA1 - EMA2, where EMA2 is EMA of EMA1
// -----------------------------------------------------------------------------
class DoubleExponentialMovingAverage {
public:
    DoubleExponentialMovingAverage(float alpha) noexcept
        : ema1_(alpha), ema2_(alpha) {}

    float add(float x) noexcept {
        float e1 = ema1_.add(x);
        float e2 = ema2_.add(e1);
        return 2.0f * e1 - e2;
    }

    float current() const noexcept {
        return 2.0f * ema1_.current() - ema2_.current();
    }

private:
    ExponentialMovingAverage ema1_;
    ExponentialMovingAverage ema2_;
};

// -----------------------------------------------------------------------------
// 5. Weighted Moving Average (linear weights) over a fixed window
// -----------------------------------------------------------------------------
class WeightedMovingAverage {
public:
    explicit WeightedMovingAverage(size_t window_size) noexcept
        : window_size_(window_size), buffer_(window_size, 0.0f), index_(0), count_(0) {
        // precompute weight divisor
        weight_sum_ = window_size_ * (window_size_ + 1) / 2.0f;
    }

    float add(float value) noexcept {
        if (window_size_ == 0) return value;
        buffer_[index_] = value;
        index_ = (index_ + 1) % window_size_;
        if (count_ < window_size_) ++count_;

        // compute weighted sum
        float weighted = 0.0f;
        size_t idx = index_;
        for (size_t i = 0; i < count_; ++i) {
            idx = (idx == 0) ? window_size_ - 1 : idx - 1;
            weighted += buffer_[idx] * static_cast<float>(count_ - i);
        }
        return weighted / weight_sum_;
    }

private:
    size_t window_size_;
    std::vector<float> buffer_;
    size_t index_;
    size_t count_;
    float weight_sum_;
};

// -----------------------------------------------------------------------------
// 6. Exponentially Weighted Moving Variance and Standard Deviation
//    Uses Welford‑like algorithm for exponential forgetting
// -----------------------------------------------------------------------------
class ExponentialMovingVariance {
public:
    ExponentialMovingVariance(float alpha) noexcept : alpha_(alpha), mean_(0.0f), var_(0.0f), initialized_(false) {}

    void add(float x) noexcept {
        if (!initialized_) {
            mean_ = x;
            var_ = 0.0f;
            initialized_ = true;
        } else {
            float diff = x - mean_;
            mean_ += alpha_ * diff;
            var_ = (1.0f - alpha_) * (var_ + alpha_ * diff * diff);
        }
    }

    float mean() const noexcept { return mean_; }
    float variance() const noexcept { return var_; }
    float stddev() const noexcept { return std::sqrt(var_); }

private:
    float alpha_;
    float mean_;
    float var_;
    bool initialized_;
};

} // namespace moving_avg
} // namespace SimulationMath

#endif // CORE_MATH_MOVING_AVERAGE_H