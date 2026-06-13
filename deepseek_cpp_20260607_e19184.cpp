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
#pragma once
#ifndef ORTHOTREE_CORE_MATH_NUMERICAL_METHODS_H_INCLUDED
#define ORTHOTREE_CORE_MATH_NUMERICAL_METHODS_H_INCLUDED

#include <cmath>
#include <functional>
#include <limits>
#include <type_traits>

namespace OrthoTree::Math {

template<typename T>
constexpr T epsilon() noexcept {
    return std::numeric_limits<T>::epsilon();
}

template<typename T>
constexpr T pi() noexcept {
    return T{3.14159265358979323846264338327950288419716939937510L};
}

template<typename T>
constexpr T twoPi() noexcept {
    return T{2} * pi<T>();
}

template<typename T>
constexpr T halfPi() noexcept {
    return pi<T>() / T{2};
}

template<typename T>
constexpr T degToRad(T degrees) noexcept {
    return degrees * pi<T>() / T{180};
}

template<typename T>
constexpr T radToDeg(T radians) noexcept {
    return radians * T{180} / pi<T>();
}

template<typename T>
constexpr bool isZero(T value, T eps = epsilon<T>()) noexcept {
    return std::abs(value) <= eps;
}

template<typename T>
constexpr bool isEqual(T a, T b, T eps = epsilon<T>()) noexcept {
    return std::abs(a - b) <= eps;
}

template<typename T>
constexpr T clamp(T value, T low, T high) noexcept {
    return (value < low) ? low : ((value > high) ? high : value);
}

template<typename T>
constexpr T smoothStep(T edge0, T edge1, T x) noexcept {
    T t = clamp((x - edge0) / (edge1 - edge0), T{0}, T{1});
    return t * t * (T{3} - T{2} * t);
}

template<typename T>
constexpr T lerp(T a, T b, T t) noexcept {
    return a + (b - a) * t;
}

template<typename T>
constexpr T inverseLerp(T a, T b, T value) noexcept {
    return (value - a) / (b - a);
}

template<typename T>
constexpr T remap(T inLow, T inHigh, T outLow, T outHigh, T value) noexcept {
    return lerp(outLow, outHigh, inverseLerp(inLow, inHigh, value));
}

template<typename T>
constexpr T gaussian(T x, T mean = T{0}, T sigma = T{1}) noexcept {
    T diff = x - mean;
    return std::exp(-diff * diff / (T{2} * sigma * sigma)) / (sigma * std::sqrt(T{2} * pi<T>()));
}

template<typename T>
constexpr T sigmoid(T x, T steepness = T{1}) noexcept {
    return T{1} / (T{1} + std::exp(-steepness * x));
}

template<typename T>
constexpr T logistic(T x, T midpoint = T{0}, T steepness = T{1}) noexcept {
    return T{1} / (T{1} + std::exp(-steepness * (x - midpoint)));
}

template<typename T>
constexpr T relu(T x) noexcept {
    return (x > T{0}) ? x : T{0};
}

template<typename T>
constexpr T leakyRelu(T x, T alpha = T{0.01}) noexcept {
    return (x > T{0}) ? x : alpha * x;
}

template<typename T>
constexpr T softplus(T x, T beta = T{1}) noexcept {
    return std::log(T{1} + std::exp(beta * x)) / beta;
}

template<typename T, typename Func>
T findRootBisection(Func f, T low, T high, T tolerance = epsilon<T>(), int maxIter = 100) noexcept {
    T fLow = f(low);
    T fHigh = f(high);
    if (fLow * fHigh >= T{0}) return low;
    for (int i = 0; i < maxIter; ++i) {
        T mid = (low + high) / T{2};
        T fMid = f(mid);
        if (std::abs(fMid) < tolerance) return mid;
        if (fLow * fMid < T{0}) {
            high = mid;
            fHigh = fMid;
        } else {
            low = mid;
            fLow = fMid;
        }
    }
    return (low + high) / T{2};
}

template<typename T, typename Func>
T findRootNewton(Func f, Func fPrime, T initial, T tolerance = epsilon<T>(), int maxIter = 100) noexcept {
    T x = initial;
    for (int i = 0; i < maxIter; ++i) {
        T fx = f(x);
        if (std::abs(fx) < tolerance) return x;
        T fpx = fPrime(x);
        if (std::abs(fpx) < epsilon<T>()) return x;
        T dx = fx / fpx;
        x -= dx;
    }
    return x;
}

template<typename T, typename Func>
T integrateSimpson(Func f, T a, T b, int n = 100) noexcept {
    if (n % 2 != 0) ++n;
    T h = (b - a) / n;
    T sum = f(a) + f(b);
    for (int i = 1; i < n; ++i) {
        sum += (i % 2 == 0) ? T{2} * f(a + i * h) : T{4} * f(a + i * h);
    }
    return sum * h / T{3};
}

template<typename T, typename Func>
T integrateTrapezoidal(Func f, T a, T b, int n = 100) noexcept {
    T h = (b - a) / n;
    T sum = (f(a) + f(b)) / T{2};
    for (int i = 1; i < n; ++i) {
        sum += f(a + i * h);
    }
    return sum * h;
}

template<typename T, typename Func>
T differentiateCentral(Func f, T x, T h = T{1e-6}) noexcept {
    return (f(x + h) - f(x - h)) / (T{2} * h);
}

template<typename T, typename Func>
T differentiateForward(Func f, T x, T h = T{1e-6}) noexcept {
    return (f(x + h) - f(x)) / h;
}

template<typename T, typename Func>
T differentiateBackward(Func f, T x, T h = T{1e-6}) noexcept {
    return (f(x) - f(x - h)) / h;
}

template<typename T>
T goldenSectionSearch(const std::function<T(T)>& f, T a, T b, T tolerance = epsilon<T>()) noexcept {
    const T phi = (T{1} + std::sqrt(T{5})) / T{2};
    T c = b - (b - a) / phi;
    T d = a + (b - a) / phi;
    T fc = f(c);
    T fd = f(d);
    while (std::abs(b - a) > tolerance) {
        if (fc < fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) / phi;
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) / phi;
            fd = f(d);
        }
    }
    return (a + b) / T{2};
}

template<typename T>
T goldenSectionSearchMax(const std::function<T(T)>& f, T a, T b, T tolerance = epsilon<T>()) noexcept {
    auto g = [&f](T x) { return -f(x); };
    return goldenSectionSearch(g, a, b, tolerance);
}

template<typename T, typename Func>
T binarySearch(Func condition, T low, T high, int maxIter = 60) noexcept {
    for (int i = 0; i < maxIter; ++i) {
        T mid = (low + high) / T{2};
        if (condition(mid)) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return (low + high) / T{2};
}

template<typename T>
bool solveQuadratic(T a, T b, T c, T& r1, T& r2) noexcept {
    T disc = b * b - T{4} * a * c;
    if (disc < T{0}) return false;
    T sqrtDisc = std::sqrt(disc);
    T q = (b > T{0}) ? T{-0.5} * (b + sqrtDisc) : T{-0.5} * (b - sqrtDisc);
    r1 = q / a;
    r2 = c / q;
    return true;
}

template<typename T>
int solveCubic(T a, T b, T c, T d, T roots[3]) noexcept {
    if (isZero(a)) return solveQuadratic(b, c, d, roots[0], roots[1]) ? 2 : 0;
    T p = (T{3} * a * c - b * b) / (T{3} * a * a);
    T q = (T{2} * b * b * b - T{9} * a * b * c + T{27} * a * a * d) / (T{27} * a * a * a);
    T disc = (q * q) / T{4} + (p * p * p) / T{27};
    if (disc > T{0}) {
        T sqrtDisc = std::sqrt(disc);
        T u = std::cbrt(-q / T{2} + sqrtDisc);
        T v = std::cbrt(-q / T{2} - sqrtDisc);
        roots[0] = u + v - b / (T{3} * a);
        return 1;
    }
    if (isZero(disc)) {
        roots[0] = T{3} * q / p - b / (T{3} * a);
        roots[1] = -T{3} * q / (T{2} * p) - b / (T{3} * a);
        return 2;
    }
    T r = std::sqrt(-p * p * p / T{27});
    T phi = std::acos(-q / (T{2} * r));
    T theta = phi / T{3};
    for (int i = 0; i < 3; ++i) {
        roots[i] = T{2} * std::cbrt(r) * std::cos(theta + T{2} * pi<T>() * i / T{3}) - b / (T{3} * a);
    }
    return 3;
}

template<typename T>
T hermiteInterp(T p0, T p1, T t0, T t1, T t) noexcept {
    T t2 = t * t;
    T t3 = t2 * t;
    return (T{2} * t3 - T{3} * t2 + T{1}) * p0 +
           (T{-2} * t3 + T{3} * t2) * p1 +
           (t3 - T{2} * t2 + t) * t0 +
           (t3 - t2) * t1;
}

template<typename T>
T catmullRomInterp(T p0, T p1, T p2, T p3, T t) noexcept {
    T t2 = t * t;
    T t3 = t2 * t;
    return T{0.5} * ((T{-1} * t3 + T{2} * t2 - t) * p0 +
                     (T{3} * t3 - T{5} * t2 + T{2}) * p1 +
                     (T{-3} * t3 + T{4} * t2 + t) * p2 +
                     (t3 - t2) * p3);
}

template<typename T>
T bsplineInterp(T p0, T p1, T p2, T p3, T t) noexcept {
    T t2 = t * t;
    T t3 = t2 * t;
    return (T{1} / T{6}) * ((T{-1} * t3 + T{3} * t2 - T{3} * t + T{1}) * p0 +
                            (T{3} * t3 - T{6} * t2 + T{4}) * p1 +
                            (T{-3} * t3 + T{3} * t2 + T{3} * t + T{1}) * p2 +
                            t3 * p3);
}

} // namespace OrthoTree::Math

#endif // ORTHOTREE_CORE_MATH_NUMERICAL_METHODS_H_INCLUDED