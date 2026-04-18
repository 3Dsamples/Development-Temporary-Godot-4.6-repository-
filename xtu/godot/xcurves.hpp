// include/xtu/godot/xcurves.hpp
// xtensor-unified - Curve and Gradient resources for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XCURVES_HPP
#define XTU_GODOT_XCURVES_HPP

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/interp/xinterp.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Part 1: Curve - 1D curve resource
// #############################################################################

class Curve : public Resource {
    XTU_GODOT_REGISTER_CLASS(Curve, Resource)

public:
    enum TangentMode {
        TANGENT_FREE,
        TANGENT_LINEAR,
        TANGENT_MODE_COUNT
    };

    enum InterpolationType {
        INTERP_LINEAR,
        INTERP_CUBIC,
        INTERP_HERMITE
    };

    struct Point {
        float position = 0.0f;
        float value = 0.0f;
        float left_tangent = 0.0f;
        float right_tangent = 0.0f;
        TangentMode left_mode = TANGENT_FREE;
        TangentMode right_mode = TANGENT_FREE;
    };

private:
    std::vector<Point> m_points;
    float m_min_value = 0.0f;
    float m_max_value = 1.0f;
    bool m_bake_resolution_enabled = false;
    int m_bake_resolution = 100;
    std::vector<float> m_baked_values;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("Curve"); }

    void add_point(const vec2f& pos, float left_tan = 0.0f, float right_tan = 0.0f,
                   TangentMode left_mode = TANGENT_FREE, TangentMode right_mode = TANGENT_FREE) {
        std::lock_guard<std::mutex> lock(m_mutex);
        Point p;
        p.position = std::clamp(pos.x(), 0.0f, 1.0f);
        p.value = pos.y();
        p.left_tangent = left_tan;
        p.right_tangent = right_tan;
        p.left_mode = left_mode;
        p.right_mode = right_mode;
        m_points.push_back(p);
        sort_points();
        mark_dirty();
    }

    void remove_point(int idx) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx >= 0 && idx < static_cast<int>(m_points.size())) {
            m_points.erase(m_points.begin() + idx);
            mark_dirty();
        }
    }

    int get_point_count() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return static_cast<int>(m_points.size());
    }

    Point get_point(int idx) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return idx >= 0 && idx < static_cast<int>(m_points.size()) ? m_points[idx] : Point();
    }

    void set_point_value(int idx, float value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx >= 0 && idx < static_cast<int>(m_points.size())) {
            m_points[idx].value = value;
            mark_dirty();
        }
    }

    void set_point_offset(int idx, float offset) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx >= 0 && idx < static_cast<int>(m_points.size())) {
            m_points[idx].position = std::clamp(offset, 0.0f, 1.0f);
            sort_points();
            mark_dirty();
        }
    }

    void clear_points() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_points.clear();
        mark_dirty();
    }

    float interpolate(float offset) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_points.empty()) return 0.0f;
        if (offset <= m_points.front().position) return m_points.front().value;
        if (offset >= m_points.back().position) return m_points.back().value;

        if (m_bake_resolution_enabled && !m_baked_values.empty()) {
            float idx = offset * static_cast<float>(m_baked_values.size() - 1);
            size_t idx0 = static_cast<size_t>(idx);
            size_t idx1 = std::min(idx0 + 1, m_baked_values.size() - 1);
            float t = idx - static_cast<float>(idx0);
            return m_baked_values[idx0] * (1.0f - t) + m_baked_values[idx1] * t;
        }

        size_t idx = 0;
        while (idx < m_points.size() - 1 && m_points[idx + 1].position < offset) ++idx;

        const Point& p0 = m_points[idx];
        const Point& p1 = m_points[idx + 1];
        float t = (offset - p0.position) / (p1.position - p0.position);

        float m0 = get_right_tangent(p0);
        float m1 = get_left_tangent(p1);

        return interpolate_hermite(p0.value, p1.value, m0, m1, t);
    }

    float interpolate_baked(float offset) const {
        return interpolate(offset);
    }

    void set_min_value(float min_val) { m_min_value = min_val; mark_dirty(); }
    float get_min_value() const { return m_min_value; }

    void set_max_value(float max_val) { m_max_value = max_val; mark_dirty(); }
    float get_max_value() const { return m_max_value; }

    void set_bake_resolution_enabled(bool enabled) { m_bake_resolution_enabled = enabled; mark_dirty(); }
    bool is_bake_resolution_enabled() const { return m_bake_resolution_enabled; }

    void set_bake_resolution(int resolution) { m_bake_resolution = std::max(1, resolution); mark_dirty(); }
    int get_bake_resolution() const { return m_bake_resolution; }

    void bake() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_bake_resolution_enabled) return;
        m_baked_values.resize(m_bake_resolution);
        for (int i = 0; i < m_bake_resolution; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(m_bake_resolution - 1);
            m_baked_values[i] = interpolate(t);
        }
    }

    std::vector<float> get_baked_array() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_baked_values;
    }

private:
    void sort_points() {
        std::sort(m_points.begin(), m_points.end(),
                  [](const Point& a, const Point& b) { return a.position < b.position; });
    }

    float get_right_tangent(const Point& p) const {
        if (p.right_mode == TANGENT_LINEAR) return 0.0f;
        return p.right_tangent;
    }

    float get_left_tangent(const Point& p) const {
        if (p.left_mode == TANGENT_LINEAR) return 0.0f;
        return p.left_tangent;
    }

    float interpolate_hermite(float p0, float p1, float m0, float m1, float t) const {
        float t2 = t * t;
        float t3 = t2 * t;
        float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
        float h10 = t3 - 2.0f * t2 + t;
        float h01 = -2.0f * t3 + 3.0f * t2;
        float h11 = t3 - t2;
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
    }

    void mark_dirty() {
        if (m_bake_resolution_enabled) {
            bake();
        }
        emit_signal("changed");
    }
};

// #############################################################################
// Part 2: Curve2D - 2D path curve
// #############################################################################

class Curve2D : public Resource {
    XTU_GODOT_REGISTER_CLASS(Curve2D, Resource)

private:
    std::vector<vec2f> m_points;
    std::vector<vec2f> m_in_tangents;
    std::vector<vec2f> m_out_tangents;
    bool m_closed = false;
    float m_bake_interval = 5.0f;
    std::vector<vec2f> m_baked_points;
    std::vector<float> m_baked_distances;
    float m_baked_total_length = 0.0f;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("Curve2D"); }

    void add_point(const vec2f& pos, const vec2f& in_tan = vec2f(0, 0), const vec2f& out_tan = vec2f(0, 0), int idx = -1) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx < 0 || idx >= static_cast<int>(m_points.size())) {
            m_points.push_back(pos);
            m_in_tangents.push_back(in_tan);
            m_out_tangents.push_back(out_tan);
        } else {
            m_points.insert(m_points.begin() + idx, pos);
            m_in_tangents.insert(m_in_tangents.begin() + idx, in_tan);
            m_out_tangents.insert(m_out_tangents.begin() + idx, out_tan);
        }
        mark_dirty();
    }

    void set_point_position(int idx, const vec2f& pos) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx >= 0 && idx < static_cast<int>(m_points.size())) {
            m_points[idx] = pos;
            mark_dirty();
        }
    }

    vec2f get_point_position(int idx) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return idx >= 0 && idx < static_cast<int>(m_points.size()) ? m_points[idx] : vec2f(0);
    }

    void set_point_in(int idx, const vec2f& in_tan) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx >= 0 && idx < static_cast<int>(m_in_tangents.size())) {
            m_in_tangents[idx] = in_tan;
            mark_dirty();
        }
    }

    vec2f get_point_in(int idx) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return idx >= 0 && idx < static_cast<int>(m_in_tangents.size()) ? m_in_tangents[idx] : vec2f(0);
    }

    void set_point_out(int idx, const vec2f& out_tan) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx >= 0 && idx < static_cast<int>(m_out_tangents.size())) {
            m_out_tangents[idx] = out_tan;
            mark_dirty();
        }
    }

    vec2f get_point_out(int idx) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return idx >= 0 && idx < static_cast<int>(m_out_tangents.size()) ? m_out_tangents[idx] : vec2f(0);
    }

    void remove_point(int idx) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx >= 0 && idx < static_cast<int>(m_points.size())) {
            m_points.erase(m_points.begin() + idx);
            m_in_tangents.erase(m_in_tangents.begin() + idx);
            m_out_tangents.erase(m_out_tangents.begin() + idx);
            mark_dirty();
        }
    }

    int get_point_count() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return static_cast<int>(m_points.size());
    }

    void clear_points() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_points.clear();
        m_in_tangents.clear();
        m_out_tangents.clear();
        mark_dirty();
    }

    void set_closed(bool closed) { m_closed = closed; mark_dirty(); }
    bool is_closed() const { return m_closed; }

    void set_bake_interval(float interval) { m_bake_interval = std::max(0.1f, interval); mark_dirty(); }
    float get_bake_interval() const { return m_bake_interval; }

    vec2f sample(float offset) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_baked_points.empty()) return vec2f(0);
        if (!m_baked_distances.empty() && m_baked_total_length > 0.0f) {
            return sample_baked(offset);
        }
        return sample_baked(offset * static_cast<float>(m_baked_points.size() - 1));
    }

    vec2f sample_baked(float offset) const {
        if (m_baked_points.empty()) return vec2f(0);
        if (offset <= 0.0f) return m_baked_points.front();
        if (offset >= m_baked_total_length) return m_baked_points.back();

        auto it = std::lower_bound(m_baked_distances.begin(), m_baked_distances.end(), offset);
        size_t idx = std::distance(m_baked_distances.begin(), it);
        if (idx == 0) return m_baked_points.front();
        if (idx >= m_baked_points.size()) return m_baked_points.back();

        float t0 = m_baked_distances[idx - 1];
        float t1 = m_baked_distances[idx];
        float t = (offset - t0) / (t1 - t0);
        return m_baked_points[idx - 1] * (1.0f - t) + m_baked_points[idx] * t;
    }

    float get_baked_length() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_baked_total_length;
    }

    std::vector<vec2f> get_baked_points() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_baked_points;
    }

    std::vector<vec2f> tessellate(int max_stages = 5, float tolerance = 4.0f) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_baked_points;
    }

    void bake() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_baked_points.clear();
        m_baked_distances.clear();
        m_baked_total_length = 0.0f;

        if (m_points.size() < 2) return;

        int segments = m_closed ? static_cast<int>(m_points.size()) : static_cast<int>(m_points.size() - 1);
        for (int i = 0; i < segments; ++i) {
            int idx0 = i;
            int idx1 = (i + 1) % m_points.size();

            const vec2f& p0 = m_points[idx0];
            const vec2f& p1 = m_points[idx1];
            const vec2f& t0 = m_out_tangents[idx0];
            const vec2f& t1 = m_in_tangents[idx1];

            float segment_length = estimate_cubic_length(p0, p1, t0, t1);
            int samples = std::max(2, static_cast<int>(segment_length / m_bake_interval));

            for (int j = 0; j < samples; ++j) {
                float t = static_cast<float>(j) / static_cast<float>(samples - 1);
                vec2f pt = interpolate_cubic(p0, p1, t0, t1, t);
                m_baked_points.push_back(pt);

                if (j > 0) {
                    float dist = (pt - m_baked_points[m_baked_points.size() - 2]).length();
                    m_baked_total_length += dist;
                }
                m_baked_distances.push_back(m_baked_total_length);
            }
        }
    }

private:
    float estimate_cubic_length(const vec2f& p0, const vec2f& p1,
                                 const vec2f& t0, const vec2f& t1) const {
        return (p0 - p1).length();
    }

    vec2f interpolate_cubic(const vec2f& p0, const vec2f& p1,
                             const vec2f& t0, const vec2f& t1, float t) const {
        float t2 = t * t;
        float t3 = t2 * t;
        float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
        float h10 = t3 - 2.0f * t2 + t;
        float h01 = -2.0f * t3 + 3.0f * t2;
        float h11 = t3 - t2;
        return p0 * h00 + t0 * h10 + p1 * h01 + t1 * h11;
    }

    void mark_dirty() {
        bake();
        emit_signal("changed");
    }
};

// #############################################################################
// Part 3: Curve3D - 3D path curve
// #############################################################################

class Curve3D : public Resource {
    XTU_GODOT_REGISTER_CLASS(Curve3D, Resource)

private:
    std::vector<vec3f> m_points;
    std::vector<vec3f> m_in_tangents;
    std::vector<vec3f> m_out_tangents;
    bool m_closed = false;
    float m_bake_interval = 0.2f;
    std::vector<vec3f> m_baked_points;
    std::vector<float> m_baked_distances;
    float m_baked_total_length = 0.0f;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("Curve3D"); }

    void add_point(const vec3f& pos, const vec3f& in_tan = vec3f(0), const vec3f& out_tan = vec3f(0)) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_points.push_back(pos);
        m_in_tangents.push_back(in_tan);
        m_out_tangents.push_back(out_tan);
        mark_dirty();
    }

    int get_point_count() const { return static_cast<int>(m_points.size()); }
    void clear_points() { m_points.clear(); m_in_tangents.clear(); m_out_tangents.clear(); mark_dirty(); }
    void set_closed(bool closed) { m_closed = closed; mark_dirty(); }
    bool is_closed() const { return m_closed; }

    vec3f sample(float offset) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_baked_points.empty()) return vec3f(0);
        if (!m_baked_distances.empty() && m_baked_total_length > 0.0f) {
            return sample_baked(offset);
        }
        return sample_baked(offset * static_cast<float>(m_baked_points.size() - 1));
    }

    vec3f sample_baked(float offset) const {
        if (m_baked_points.empty()) return vec3f(0);
        if (offset <= 0.0f) return m_baked_points.front();
        if (offset >= m_baked_total_length) return m_baked_points.back();

        auto it = std::lower_bound(m_baked_distances.begin(), m_baked_distances.end(), offset);
        size_t idx = std::distance(m_baked_distances.begin(), it);
        if (idx == 0) return m_baked_points.front();
        if (idx >= m_baked_points.size()) return m_baked_points.back();

        float t0 = m_baked_distances[idx - 1];
        float t1 = m_baked_distances[idx];
        float t = (offset - t0) / (t1 - t0);
        return m_baked_points[idx - 1] * (1.0f - t) + m_baked_points[idx] * t;
    }

    float get_baked_length() const { return m_baked_total_length; }
    std::vector<vec3f> get_baked_points() const { return m_baked_points; }
    std::vector<vec3f> tessellate(int max_stages = 5, float tolerance = 0.1f) const { return m_baked_points; }

    void bake() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_baked_points.clear();
        m_baked_distances.clear();
        m_baked_total_length = 0.0f;

        if (m_points.size() < 2) return;

        int segments = m_closed ? static_cast<int>(m_points.size()) : static_cast<int>(m_points.size() - 1);
        for (int i = 0; i < segments; ++i) {
            int idx0 = i;
            int idx1 = (i + 1) % m_points.size();

            const vec3f& p0 = m_points[idx0];
            const vec3f& p1 = m_points[idx1];
            const vec3f& t0 = m_out_tangents[idx0];
            const vec3f& t1 = m_in_tangents[idx1];

            float segment_length = (p0 - p1).length();
            int samples = std::max(2, static_cast<int>(segment_length / m_bake_interval));

            for (int j = 0; j < samples; ++j) {
                float t = static_cast<float>(j) / static_cast<float>(samples - 1);
                vec3f pt = interpolate_cubic(p0, p1, t0, t1, t);
                m_baked_points.push_back(pt);

                if (j > 0) {
                    float dist = (pt - m_baked_points[m_baked_points.size() - 2]).length();
                    m_baked_total_length += dist;
                }
                m_baked_distances.push_back(m_baked_total_length);
            }
        }
    }

private:
    vec3f interpolate_cubic(const vec3f& p0, const vec3f& p1,
                             const vec3f& t0, const vec3f& t1, float t) const {
        float t2 = t * t;
        float t3 = t2 * t;
        float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
        float h10 = t3 - 2.0f * t2 + t;
        float h01 = -2.0f * t3 + 3.0f * t2;
        float h11 = t3 - t2;
        return p0 * h00 + t0 * h10 + p1 * h01 + t1 * h11;
    }

    void mark_dirty() {
        bake();
        emit_signal("changed");
    }
};

// #############################################################################
// Part 4: Gradient - Color gradient resource
// #############################################################################

class Gradient : public Resource {
    XTU_GODOT_REGISTER_CLASS(Gradient, Resource)

public:
    enum InterpolationMode {
        GRADIENT_INTERPOLATE_LINEAR,
        GRADIENT_INTERPOLATE_CONSTANT,
        GRADIENT_INTERPOLATE_CUBIC
    };

    struct Point {
        float offset = 0.0f;
        Color color = Color(1, 1, 1, 1);
    };

private:
    std::vector<Point> m_points;
    InterpolationMode m_interpolation_mode = GRADIENT_INTERPOLATE_LINEAR;
    std::vector<Color> m_baked_colors;
    int m_bake_resolution = 256;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("Gradient"); }

    Gradient() {
        m_points.push_back({0.0f, Color(1, 1, 1, 1)});
        m_points.push_back({1.0f, Color(1, 1, 1, 1)});
        bake();
    }

    void add_point(float offset, const Color& color) {
        std::lock_guard<std::mutex> lock(m_mutex);
        Point p;
        p.offset = std::clamp(offset, 0.0f, 1.0f);
        p.color = color;
        m_points.push_back(p);
        sort_points();
        bake();
    }

    void remove_point(int idx) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_points.size() <= 2) return;
        if (idx >= 0 && idx < static_cast<int>(m_points.size())) {
            m_points.erase(m_points.begin() + idx);
            bake();
        }
    }

    void set_point(int idx, const Point& p) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx >= 0 && idx < static_cast<int>(m_points.size())) {
            m_points[idx] = p;
            m_points[idx].offset = std::clamp(p.offset, 0.0f, 1.0f);
            sort_points();
            bake();
        }
    }

    Point get_point(int idx) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return idx >= 0 && idx < static_cast<int>(m_points.size()) ? m_points[idx] : Point();
    }

    int get_point_count() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return static_cast<int>(m_points.size());
    }

    void set_color(int idx, const Color& color) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx >= 0 && idx < static_cast<int>(m_points.size())) {
            m_points[idx].color = color;
            bake();
        }
    }

    Color get_color(int idx) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return idx >= 0 && idx < static_cast<int>(m_points.size()) ? m_points[idx].color : Color();
    }

    void set_offset(int idx, float offset) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx >= 0 && idx < static_cast<int>(m_points.size())) {
            m_points[idx].offset = std::clamp(offset, 0.0f, 1.0f);
            sort_points();
            bake();
        }
    }

    float get_offset(int idx) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return idx >= 0 && idx < static_cast<int>(m_points.size()) ? m_points[idx].offset : 0.0f;
    }

    void set_interpolation_mode(InterpolationMode mode) {
        m_interpolation_mode = mode;
        bake();
    }

    InterpolationMode get_interpolation_mode() const { return m_interpolation_mode; }

    Color get_color_at_offset(float offset) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_baked_colors.empty()) return Color();
        offset = std::clamp(offset, 0.0f, 1.0f);
        float idx = offset * static_cast<float>(m_baked_colors.size() - 1);
        size_t idx0 = static_cast<size_t>(idx);
        size_t idx1 = std::min(idx0 + 1, m_baked_colors.size() - 1);
        float t = idx - static_cast<float>(idx0);
        return m_baked_colors[idx0].lerp(m_baked_colors[idx1], t);
    }

    std::vector<Color> get_baked_array() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_baked_colors;
    }

    void set_bake_resolution(int resolution) {
        m_bake_resolution = std::max(2, resolution);
        bake();
    }

    int get_bake_resolution() const { return m_bake_resolution; }

    void bake() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_baked_colors.resize(m_bake_resolution);
        for (int i = 0; i < m_bake_resolution; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(m_bake_resolution - 1);
            m_baked_colors[i] = interpolate(t);
        }
        emit_signal("changed");
    }

private:
    void sort_points() {
        std::sort(m_points.begin(), m_points.end(),
                  [](const Point& a, const Point& b) { return a.offset < b.offset; });
    }

    Color interpolate(float offset) const {
        if (m_points.empty()) return Color();
        if (offset <= m_points.front().offset) return m_points.front().color;
        if (offset >= m_points.back().offset) return m_points.back().color;

        size_t idx = 0;
        while (idx < m_points.size() - 1 && m_points[idx + 1].offset < offset) ++idx;

        const Point& p0 = m_points[idx];
        const Point& p1 = m_points[idx + 1];
        float t = (offset - p0.offset) / (p1.offset - p0.offset);

        if (m_interpolation_mode == GRADIENT_INTERPOLATE_CONSTANT) {
            return p0.color;
        }
        return p0.color.lerp(p1.color, t);
    }
};

// #############################################################################
// Part 5: GradientTexture2D - 2D texture from gradient
// #############################################################################

class GradientTexture2D : public Texture2D {
    XTU_GODOT_REGISTER_CLASS(GradientTexture2D, Texture2D)

private:
    Ref<Gradient> m_gradient;
    int m_width = 256;
    int m_height = 1;
    bool m_repeat = false;
    vec2f m_fill_from = vec2f(0, 0);
    vec2f m_fill_to = vec2f(1, 0);
    bool m_radial = false;
    vec2f m_radial_center = vec2f(0.5f, 0.5f);
    Ref<ImageTexture> m_texture;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("GradientTexture2D"); }

    GradientTexture2D() {
        m_gradient.instance();
    }

    void set_gradient(const Ref<Gradient>& gradient) {
        m_gradient = gradient;
        update_texture();
    }

    Ref<Gradient> get_gradient() const { return m_gradient; }

    void set_width(int width) {
        m_width = std::max(1, width);
        update_texture();
    }

    int get_width() const override { return m_width; }

    void set_height(int height) {
        m_height = std::max(1, height);
        update_texture();
    }

    int get_height() const override { return m_height; }

    void set_repeat(bool repeat) { m_repeat = repeat; }
    bool get_repeat() const { return m_repeat; }

    void set_fill_from(const vec2f& from) { m_fill_from = from; update_texture(); }
    vec2f get_fill_from() const { return m_fill_from; }

    void set_fill_to(const vec2f& to) { m_fill_to = to; update_texture(); }
    vec2f get_fill_to() const { return m_fill_to; }

    void set_radial(bool radial) { m_radial = radial; update_texture(); }
    bool is_radial() const { return m_radial; }

    void set_radial_center(const vec2f& center) { m_radial_center = center; update_texture(); }
    vec2f get_radial_center() const { return m_radial_center; }

    RID get_rid() const override {
        return m_texture.is_valid() ? m_texture->get_rid() : RID();
    }

    void update_texture() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_gradient.is_valid()) return;

        Ref<Image> img;
        img.instance();
        img->create(m_width, m_height, false, Image::FORMAT_RGBA8);

        for (int y = 0; y < m_height; ++y) {
            for (int x = 0; x < m_width; ++x) {
                Color c;
                if (m_radial) {
                    vec2f p(static_cast<float>(x) / m_width, static_cast<float>(y) / m_height);
                    float dist = (p - m_radial_center).length();
                    c = m_gradient->get_color_at_offset(std::clamp(dist, 0.0f, 1.0f));
                } else {
                    vec2f p(static_cast<float>(x) / m_width, static_cast<float>(y) / m_height);
                    vec2f dir = m_fill_to - m_fill_from;
                    float len = dir.length();
                    if (len > 0.0f) {
                        float t = dot(p - m_fill_from, dir) / (len * len);
                        if (!m_repeat) t = std::clamp(t, 0.0f, 1.0f);
                        else t = t - std::floor(t);
                        c = m_gradient->get_color_at_offset(t);
                    } else {
                        c = m_gradient->get_color_at_offset(0.0f);
                    }
                }
                img->set_pixel(x, y, c);
            }
        }

        if (!m_texture.is_valid()) {
            m_texture.instance();
        }
        m_texture->create_from_image(img);
        emit_signal("changed");
    }
};

} // namespace godot

// Bring into main namespace
using godot::Curve;
using godot::Curve2D;
using godot::Curve3D;
using godot::Gradient;
using godot::GradientTexture2D;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XCURVES_HPP