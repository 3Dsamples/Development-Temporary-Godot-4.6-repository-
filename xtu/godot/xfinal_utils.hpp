// include/xtu/godot/xfinal_utils.hpp
// xtensor-unified - Final utility modules for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XFINAL_UTILS_HPP
#define XTU_GODOT_XFINAL_UTILS_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xcore.hpp"
#include "xtu/godot/xeditor.hpp"
#include "xtu/io/xio_json.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace utils {

// #############################################################################
// Part 1: Console - Colored console output and progress bars
// #############################################################################

class Console : public Object {
    XTU_GODOT_REGISTER_CLASS(Console, Object)

public:
    enum Color {
        COLOR_RESET,
        COLOR_BLACK,
        COLOR_RED,
        COLOR_GREEN,
        COLOR_YELLOW,
        COLOR_BLUE,
        COLOR_MAGENTA,
        COLOR_CYAN,
        COLOR_WHITE,
        COLOR_BRIGHT_BLACK,
        COLOR_BRIGHT_RED,
        COLOR_BRIGHT_GREEN,
        COLOR_BRIGHT_YELLOW,
        COLOR_BRIGHT_BLUE,
        COLOR_BRIGHT_MAGENTA,
        COLOR_BRIGHT_CYAN,
        COLOR_BRIGHT_WHITE
    };

    enum Style {
        STYLE_NORMAL,
        STYLE_BOLD,
        STYLE_DIM,
        STYLE_UNDERLINE,
        STYLE_BLINK,
        STYLE_REVERSE,
        STYLE_HIDDEN
    };

private:
    static Console* s_singleton;
    bool m_use_colors = true;
    bool m_use_unicode = true;
    std::mutex m_mutex;

public:
    static Console* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("Console"); }

    Console() { s_singleton = this; }
    ~Console() { s_singleton = nullptr; }

    void set_use_colors(bool use) { m_use_colors = use; }
    bool get_use_colors() const { return m_use_colors; }

    void set_use_unicode(bool use) { m_use_unicode = use; }
    bool get_use_unicode() const { return m_use_unicode; }

    void print_colored(const String& text, Color fg = COLOR_WHITE, Color bg = COLOR_BLACK, Style style = STYLE_NORMAL) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_use_colors) {
            std::cout << color_code(fg, bg, style) << text.to_std_string() << color_code(COLOR_RESET);
        } else {
            std::cout << text.to_std_string();
        }
    }

    void print_line_colored(const String& text, Color fg = COLOR_WHITE, Color bg = COLOR_BLACK, Style style = STYLE_NORMAL) {
        print_colored(text, fg, bg, style);
        std::cout << std::endl;
    }

    void print_success(const String& text) { print_line_colored("✓ " + text, COLOR_GREEN); }
    void print_warning(const String& text) { print_line_colored("⚠ " + text, COLOR_YELLOW); }
    void print_error(const String& text) { print_line_colored("✗ " + text, COLOR_RED); }
    void print_info(const String& text) { print_line_colored("ℹ " + text, COLOR_CYAN); }
    void print_debug(const String& text) { print_line_colored("🔍 " + text, COLOR_BRIGHT_BLACK); }

    void clear_screen() {
#ifdef XTU_OS_WINDOWS
        system("cls");
#else
        system("clear");
#endif
    }

    vec2i get_terminal_size() const {
#ifdef XTU_OS_WINDOWS
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
        return vec2i(csbi.srWindow.Right - csbi.srWindow.Left + 1,
                     csbi.srWindow.Bottom - csbi.srWindow.Top + 1);
#else
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        return vec2i(w.ws_col, w.ws_row);
#endif
    }

private:
    std::string color_code(Color fg, Color bg = COLOR_BLACK, Style style = STYLE_NORMAL) const {
        std::string code = "\033[";
        // Style
        switch (style) {
            case STYLE_BOLD: code += "1;"; break;
            case STYLE_DIM: code += "2;"; break;
            case STYLE_UNDERLINE: code += "4;"; break;
            case STYLE_BLINK: code += "5;"; break;
            case STYLE_REVERSE: code += "7;"; break;
            case STYLE_HIDDEN: code += "8;"; break;
            default: code += "0;"; break;
        }
        // Foreground
        switch (fg) {
            case COLOR_BLACK: code += "30"; break;
            case COLOR_RED: code += "31"; break;
            case COLOR_GREEN: code += "32"; break;
            case COLOR_YELLOW: code += "33"; break;
            case COLOR_BLUE: code += "34"; break;
            case COLOR_MAGENTA: code += "35"; break;
            case COLOR_CYAN: code += "36"; break;
            case COLOR_WHITE: code += "37"; break;
            case COLOR_BRIGHT_BLACK: code += "90"; break;
            case COLOR_BRIGHT_RED: code += "91"; break;
            case COLOR_BRIGHT_GREEN: code += "92"; break;
            case COLOR_BRIGHT_YELLOW: code += "93"; break;
            case COLOR_BRIGHT_BLUE: code += "94"; break;
            case COLOR_BRIGHT_MAGENTA: code += "95"; break;
            case COLOR_BRIGHT_CYAN: code += "96"; break;
            case COLOR_BRIGHT_WHITE: code += "97"; break;
            default: code += "39"; break;
        }
        // Background
        if (bg != COLOR_BLACK) {
            code += ";";
            switch (bg) {
                case COLOR_BLACK: code += "40"; break;
                case COLOR_RED: code += "41"; break;
                case COLOR_GREEN: code += "42"; break;
                case COLOR_YELLOW: code += "43"; break;
                case COLOR_BLUE: code += "44"; break;
                case COLOR_MAGENTA: code += "45"; break;
                case COLOR_CYAN: code += "46"; break;
                case COLOR_WHITE: code += "47"; break;
                case COLOR_BRIGHT_BLACK: code += "100"; break;
                case COLOR_BRIGHT_RED: code += "101"; break;
                case COLOR_BRIGHT_GREEN: code += "102"; break;
                case COLOR_BRIGHT_YELLOW: code += "103"; break;
                case COLOR_BRIGHT_BLUE: code += "104"; break;
                case COLOR_BRIGHT_MAGENTA: code += "105"; break;
                case COLOR_BRIGHT_CYAN: code += "106"; break;
                case COLOR_BRIGHT_WHITE: code += "107"; break;
                default: code += "49"; break;
            }
        }
        code += "m";
        return code;
    }
};

// #############################################################################
// Part 2: ProgressBar - Console progress bar
// #############################################################################

class ConsoleProgressBar {
private:
    String m_title;
    size_t m_total = 0;
    size_t m_current = 0;
    size_t m_width = 50;
    bool m_show_percentage = true;
    bool m_show_count = true;
    std::chrono::steady_clock::time_point m_start_time;
    std::mutex m_mutex;

public:
    ConsoleProgressBar(const String& title = "", size_t total = 100)
        : m_title(title), m_total(total) {
        m_start_time = std::chrono::steady_clock::now();
    }

    void set_total(size_t total) { m_total = total; }
    void set_width(size_t width) { m_width = width; }

    void update(size_t current) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_current = std::min(current, m_total);
        render();
    }

    void increment(size_t delta = 1) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_current = std::min(m_current + delta, m_total);
        render();
    }

    void finish(const String& message = "") {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_current = m_total;
        render();
        std::cout << std::endl;
        if (!message.empty()) {
            Console::get_singleton()->print_success(message);
        }
    }

private:
    void render() {
        float ratio = m_total > 0 ? static_cast<float>(m_current) / m_total : 0.0f;
        size_t filled = static_cast<size_t>(ratio * m_width);

        std::cout << "\r";
        if (!m_title.empty()) {
            std::cout << m_title.to_std_string() << " ";
        }

        std::cout << "[";
        for (size_t i = 0; i < m_width; ++i) {
            if (i < filled) std::cout << "=";
            else if (i == filled) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "]";

        if (m_show_percentage) {
            std::cout << " " << static_cast<int>(ratio * 100) << "%";
        }
        if (m_show_count) {
            std::cout << " (" << m_current << "/" << m_total << ")";
        }

        // ETA
        if (m_current > 0 && m_current < m_total) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_start_time).count();
            if (elapsed > 0) {
                size_t eta = static_cast<size_t>(elapsed / ratio - elapsed);
                std::cout << " ETA: " << format_time(eta);
            }
        }

        std::cout << std::flush;
    }

    static std::string format_time(size_t seconds) {
        if (seconds < 60) return std::to_string(seconds) + "s";
        if (seconds < 3600) return std::to_string(seconds / 60) + "m " + std::to_string(seconds % 60) + "s";
        return std::to_string(seconds / 3600) + "h " + std::to_string((seconds % 3600) / 60) + "m";
    }
};

// #############################################################################
// Part 3: PackageManager - Asset Library package management
// #############################################################################

class PackageManager : public Object {
    XTU_GODOT_REGISTER_CLASS(PackageManager, Object)

public:
    struct Package {
        String name;
        String version;
        String description;
        String author;
        String license;
        std::vector<String> dependencies;
        std::vector<String> files;
        bool installed = false;
        bool update_available = false;
    };

private:
    static PackageManager* s_singleton;
    std::unordered_map<String, Package> m_installed_packages;
    String m_packages_dir;
    String m_cache_dir;
    mutable std::mutex m_mutex;

public:
    static PackageManager* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("PackageManager"); }

    PackageManager() {
        s_singleton = this;
        m_packages_dir = "res://addons";
        m_cache_dir = OS::get_singleton()->get_cache_dir() + "/packages";
        DirAccess::make_dir_recursive(m_cache_dir);
        scan_installed_packages();
    }

    ~PackageManager() { s_singleton = nullptr; }

    void scan_installed_packages() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_installed_packages.clear();

        Ref<DirAccess> dir = DirAccess::open(m_packages_dir);
        if (!dir.is_valid()) return;

        dir->list_dir_begin();
        String item;
        while (!(item = dir->get_next()).empty()) {
            if (item == "." || item == "..") continue;
            String full_path = m_packages_dir + "/" + item;
            if (dir->current_is_dir()) {
                String cfg_path = full_path + "/package.json";
                if (FileAccess::file_exists(cfg_path)) {
                    Package pkg = parse_package_json(cfg_path);
                    pkg.installed = true;
                    m_installed_packages[pkg.name] = pkg;
                }
            }
        }
        dir->list_dir_end();
    }

    bool install_package(const String& name, const String& version = "") {
        // Download from Asset Library
        String url = "https://godotengine.org/asset-library/api/asset/" + name;
        // ... download and extract
        return true;
    }

    bool uninstall_package(const String& name) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_installed_packages.find(name);
        if (it == m_installed_packages.end()) return false;

        const Package& pkg = it->second;
        String pkg_dir = m_packages_dir + "/" + name;

        // Remove directory
        DirAccess::remove(pkg_dir);
        m_installed_packages.erase(it);
        return true;
    }

    bool update_package(const String& name) {
        return install_package(name, "latest");
    }

    std::vector<Package> get_installed_packages() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<Package> result;
        for (const auto& kv : m_installed_packages) {
            result.push_back(kv.second);
        }
        return result;
    }

    Package get_package_info(const String& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_installed_packages.find(name);
        return it != m_installed_packages.end() ? it->second : Package();
    }

    bool is_package_installed(const String& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_installed_packages.find(name) != m_installed_packages.end();
    }

    std::vector<String> check_updates() {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<String> updates;
        // Check each package against remote
        return updates;
    }

private:
    Package parse_package_json(const String& path) {
        Package pkg;
        Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
        if (!file.is_valid()) return pkg;

        String content = file->get_as_text();
        io::json::JsonValue json = io::json::JsonValue::parse(content.to_std_string());

        pkg.name = json["name"].as_string().c_str();
        pkg.version = json["version"].as_string().c_str();
        pkg.description = json["description"].as_string().c_str();
        pkg.author = json["author"].as_string().c_str();
        pkg.license = json["license"].as_string().c_str();

        if (json["dependencies"].is_object()) {
            for (const auto& kv : json["dependencies"].as_object()) {
                pkg.dependencies.push_back(String(kv.first.c_str()) + "@" + String(kv.second.as_string().c_str()));
            }
        }

        return pkg;
    }
};

// #############################################################################
// Part 4: RemoteInspector - Runtime remote debugging
// #############################################################################

class RemoteInspector : public Object {
    XTU_GODOT_REGISTER_CLASS(RemoteInspector, Object)

private:
    static RemoteInspector* s_singleton;
    bool m_enabled = false;
    int m_port = 6009;
    Node* m_inspected_node = nullptr;
    std::mutex m_mutex;
    std::function<void(const String&)> m_send_callback;

public:
    static RemoteInspector* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("RemoteInspector"); }

    RemoteInspector() { s_singleton = this; }
    ~RemoteInspector() { s_singleton = nullptr; }

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    void set_port(int port) { m_port = port; }
    int get_port() const { return m_port; }

    void set_send_callback(std::function<void(const String&)> cb) { m_send_callback = cb; }

    void inspect_node(Node* node) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_inspected_node = node;
        if (m_enabled) {
            send_node_info();
        }
    }

    void refresh() {
        if (m_enabled) {
            send_node_info();
        }
    }

    String get_node_json(Node* node = nullptr) const {
        if (!node) node = m_inspected_node;
        if (!node) return "{}";

        io::json::JsonValue json;
        json["name"] = io::json::JsonValue(node->get_name().to_std_string());
        json["class"] = io::json::JsonValue(node->get_class().to_std_string());
        json["path"] = io::json::JsonValue(node->get_path().to_std_string());

        // Properties
        io::json::JsonValue props;
        std::vector<PropertyInfo> prop_list = node->get_property_list();
        for (const auto& prop : prop_list) {
            if (prop.usage & PropertyUsage::EDITOR) {
                Variant value = node->get(prop.name);
                props[prop.name.to_std_string()] = variant_to_json(value);
            }
        }
        json["properties"] = props;

        // Children
        io::json::JsonValue children;
        for (int i = 0; i < node->get_child_count(); ++i) {
            children.as_array().push_back(io::json::JsonValue(node->get_child(i)->get_name().to_std_string()));
        }
        json["children"] = children;

        return json.dump().c_str();
    }

    void handle_command(const String& command_json) {
        io::json::JsonValue cmd = io::json::JsonValue::parse(command_json.to_std_string());
        String action = cmd["action"].as_string().c_str();

        if (action == "get_node") {
            send_node_info();
        } else if (action == "set_property") {
            String prop = cmd["property"].as_string().c_str();
            Variant value = json_to_variant(cmd["value"]);
            if (m_inspected_node) {
                m_inspected_node->set(prop, value);
            }
        } else if (action == "call_method") {
            String method = cmd["method"].as_string().c_str();
            // Call method on node
        }
    }

private:
    void send_node_info() {
        if (m_send_callback) {
            m_send_callback(get_node_json());
        }
    }

    static io::json::JsonValue variant_to_json(const Variant& v) {
        if (v.is_bool()) return io::json::JsonValue(v.as<bool>());
        if (v.is_num()) return io::json::JsonValue(v.as<double>());
        if (v.is_string()) return io::json::JsonValue(v.as<String>().to_std_string());
        if (v.is_nil()) return io::json::JsonValue();
        return io::json::JsonValue(v.as<String>().to_std_string());
    }

    static Variant json_to_variant(const io::json::JsonValue& json) {
        if (json.is_bool()) return Variant(json.as_bool());
        if (json.is_number()) return Variant(json.as_number());
        if (json.is_string()) return Variant(String(json.as_string().c_str()));
        return Variant();
    }
};

// #############################################################################
// Part 5: RandomNumberGenerator - Advanced RNG
// #############################################################################

class RandomNumberGenerator : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(RandomNumberGenerator, RefCounted)

public:
    enum Distribution {
        DIST_UNIFORM,
        DIST_NORMAL,
        DIST_EXPONENTIAL,
        DIST_POISSON,
        DIST_BERNOULLI,
        DIST_BINOMIAL,
        DIST_GAMMA,
        DIST_BETA
    };

private:
    std::mt19937_64 m_rng;
    uint64_t m_seed = 0;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("RandomNumberGenerator"); }

    RandomNumberGenerator() {
        std::random_device rd;
        m_seed = (static_cast<uint64_t>(rd()) << 32) | rd();
        m_rng.seed(m_seed);
    }

    void set_seed(uint64_t seed) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_seed = seed;
        m_rng.seed(seed);
    }

    uint64_t get_seed() const { return m_seed; }

    int rand_int(int min_val = 0, int max_val = std::numeric_limits<int>::max()) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::uniform_int_distribution<int> dist(min_val, max_val);
        return dist(m_rng);
    }

    float rand_float(float min_val = 0.0f, float max_val = 1.0f) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::uniform_real_distribution<float> dist(min_val, max_val);
        return dist(m_rng);
    }

    double rand_normal(double mean = 0.0, double stddev = 1.0) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::normal_distribution<double> dist(mean, stddev);
        return dist(m_rng);
    }

    double rand_exponential(double lambda = 1.0) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::exponential_distribution<double> dist(lambda);
        return dist(m_rng);
    }

    int rand_poisson(double mean = 1.0) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::poisson_distribution<int> dist(mean);
        return dist(m_rng);
    }

    bool rand_bernoulli(double p = 0.5) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::bernoulli_distribution dist(p);
        return dist(m_rng);
    }

    void randomize() {
        std::random_device rd;
        set_seed((static_cast<uint64_t>(rd()) << 32) | rd());
    }

    std::vector<int> permutation(int n) {
        std::vector<int> result(n);
        for (int i = 0; i < n; ++i) result[i] = i;
        std::lock_guard<std::mutex> lock(m_mutex);
        std::shuffle(result.begin(), result.end(), m_rng);
        return result;
    }

    template<typename T>
    T choice(const std::vector<T>& items) {
        if (items.empty()) return T();
        return items[rand_int(0, static_cast<int>(items.size()) - 1)];
    }

    template<typename T>
    std::vector<T> sample(const std::vector<T>& items, size_t k, bool replace = false) {
        std::vector<T> result;
        if (items.empty() || k == 0) return result;
        if (replace) {
            result.reserve(k);
            for (size_t i = 0; i < k; ++i) {
                result.push_back(choice(items));
            }
        } else {
            k = std::min(k, items.size());
            auto indices = permutation(static_cast<int>(items.size()));
            result.reserve(k);
            for (size_t i = 0; i < k; ++i) {
                result.push_back(items[indices[i]]);
            }
        }
        return result;
    }
};

// #############################################################################
// Part 6: GeometryUtils - Geometry helper functions
// #############################################################################

class GeometryUtils : public Object {
    XTU_GODOT_REGISTER_CLASS(GeometryUtils, Object)

public:
    static StringName get_class_static() { return StringName("GeometryUtils"); }

    static bool point_in_triangle_2d(const vec2f& p, const vec2f& a, const vec2f& b, const vec2f& c) {
        vec2f v0 = c - a;
        vec2f v1 = b - a;
        vec2f v2 = p - a;
        float dot00 = dot(v0, v0);
        float dot01 = dot(v0, v1);
        float dot02 = dot(v0, v2);
        float dot11 = dot(v1, v1);
        float dot12 = dot(v1, v2);
        float inv_denom = 1.0f / (dot00 * dot11 - dot01 * dot01);
        float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
        return (u >= 0) && (v >= 0) && (u + v <= 1);
    }

    static bool point_in_polygon_2d(const vec2f& p, const std::vector<vec2f>& polygon) {
        if (polygon.size() < 3) return false;
        bool inside = false;
        size_t n = polygon.size();
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            const vec2f& pi = polygon[i];
            const vec2f& pj = polygon[j];
            if (((pi.y() > p.y()) != (pj.y() > p.y())) &&
                (p.x() < (pj.x() - pi.x()) * (p.y() - pi.y()) / (pj.y() - pi.y()) + pi.x())) {
                inside = !inside;
            }
        }
        return inside;
    }

    static bool segment_intersects_segment_2d(const vec2f& a1, const vec2f& a2,
                                               const vec2f& b1, const vec2f& b2,
                                               vec2f* out_intersection = nullptr) {
        vec2f r = a2 - a1;
        vec2f s = b2 - b1;
        float rxs = cross(r, s);
        vec2f qp = b1 - a1;
        float qpxr = cross(qp, r);

        if (std::abs(rxs) < 1e-6f) {
            if (std::abs(qpxr) < 1e-6f) {
                // Collinear
                float t0 = dot(qp, r) / dot(r, r);
                float t1 = t0 + dot(s, r) / dot(r, r);
                if ((t0 >= 0 && t0 <= 1) || (t1 >= 0 && t1 <= 1) || (t0 < 0 && t1 > 1)) {
                    if (out_intersection) *out_intersection = a1 + r * std::max(0.0f, t0);
                    return true;
                }
                return false;
            }
            return false;
        }

        float t = cross(qp, s) / rxs;
        float u = cross(qp, r) / rxs;

        if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
            if (out_intersection) *out_intersection = a1 + r * t;
            return true;
        }
        return false;
    }

    static std::vector<vec2f> convex_hull_2d(std::vector<vec2f> points) {
        if (points.size() <= 3) return points;

        std::sort(points.begin(), points.end(), [](const vec2f& a, const vec2f& b) {
            return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
        });

        std::vector<vec2f> hull;
        for (int phase = 0; phase < 2; ++phase) {
            size_t start = hull.size();
            for (const vec2f& p : points) {
                while (hull.size() >= start + 2) {
                    vec2f a = hull[hull.size() - 2];
                    vec2f b = hull[hull.size() - 1];
                    if (cross(b - a, p - b) <= 0) {
                        hull.pop_back();
                    } else {
                        break;
                    }
                }
                hull.push_back(p);
            }
            hull.pop_back();
            std::reverse(points.begin(), points.end());
        }
        return hull;
    }

    static float polygon_area_2d(const std::vector<vec2f>& polygon) {
        if (polygon.size() < 3) return 0.0f;
        float area = 0.0f;
        size_t n = polygon.size();
        for (size_t i = 0; i < n; ++i) {
            size_t j = (i + 1) % n;
            area += cross(polygon[i], polygon[j]);
        }
        return std::abs(area) * 0.5f;
    }

    static vec2f polygon_centroid_2d(const std::vector<vec2f>& polygon) {
        if (polygon.size() < 3) return vec2f(0, 0);
        vec2f centroid(0, 0);
        float area = 0.0f;
        size_t n = polygon.size();
        for (size_t i = 0; i < n; ++i) {
            size_t j = (i + 1) % n;
            float cross_val = cross(polygon[i], polygon[j]);
            area += cross_val;
            centroid += (polygon[i] + polygon[j]) * cross_val;
        }
        if (std::abs(area) < 1e-6f) return vec2f(0, 0);
        return centroid / (3.0f * area);
    }

    static std::vector<vec2i> bresenham_line(int x0, int y0, int x1, int y1) {
        std::vector<vec2i> points;
        int dx = std::abs(x1 - x0);
        int dy = -std::abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;

        while (true) {
            points.push_back(vec2i(x0, y0));
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 >= dy) {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx) {
                err += dx;
                y0 += sy;
            }
        }
        return points;
    }
};

// #############################################################################
// Part 7: CoreBind - Core API bindings aggregator
// #############################################################################

class CoreBind : public Object {
    XTU_GODOT_REGISTER_CLASS(CoreBind, Object)

public:
    static StringName get_class_static() { return StringName("CoreBind"); }

    static void register_all() {
        // This class serves as an aggregator to ensure all core bindings are registered
        // The actual bindings are done in their respective classes via XTU_GODOT_REGISTER_CLASS
    }

    static String get_version_string() {
        return "Xtensor-Godot 4.6.0";
    }

    static Dictionary get_system_info() {
        Dictionary info;
        info["os"] = OS::get_singleton()->get_name();
        info["os_version"] = OS::get_singleton()->get_version();
        info["processor_count"] = OS::get_singleton()->get_processor_count();
        info["video_adapter"] = RenderingServer::get_singleton()->get_video_adapter_name();
        info["video_driver"] = RenderingServer::get_singleton()->get_video_driver_name();
        info["memory_total"] = static_cast<int64_t>(OS::get_singleton()->get_memory_total());
        info["memory_free"] = static_cast<int64_t>(OS::get_singleton()->get_memory_free());
        return info;
    }
};

} // namespace utils

// Bring into main namespace
using utils::Console;
using utils::ConsoleProgressBar;
using utils::PackageManager;
using utils::RemoteInspector;
using utils::RandomNumberGenerator;
using utils::GeometryUtils;
using utils::CoreBind;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XFINAL_UTILS_HPP