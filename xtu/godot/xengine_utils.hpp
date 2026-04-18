// include/xtu/godot/xengine_utils.hpp
// xtensor-unified - Engine utilities and helpers for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XENGINE_UTILS_HPP
#define XTU_GODOT_XENGINE_UTILS_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xcore.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/io/xio_json.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace utils {

// #############################################################################
// Part 1: Engine - Global engine configuration singleton
// #############################################################################

class Engine : public Object {
    XTU_GODOT_REGISTER_CLASS(Engine, Object)

private:
    static Engine* s_singleton;
    std::unordered_map<StringName, Object*> m_singletons;
    std::map<String, String> m_config;
    int m_fps = 0;
    float m_time_scale = 1.0f;
    uint64_t m_frames_drawn = 0;
    uint64_t m_physics_frames = 0;
    float m_physics_interpolation_fraction = 0.0f;
    bool m_editor_hint = false;
    bool m_project_manager_hint = false;
    bool m_pause = false;
    int m_target_fps = 0;
    mutable std::mutex m_mutex;

public:
    static Engine* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("Engine"); }

    Engine() {
        s_singleton = this;
        initialize_config();
    }

    ~Engine() { s_singleton = nullptr; }

    void set_editor_hint(bool enabled) { m_editor_hint = enabled; }
    bool is_editor_hint() const { return m_editor_hint; }

    void set_project_manager_hint(bool enabled) { m_project_manager_hint = enabled; }
    bool is_project_manager_hint() const { return m_project_manager_hint; }

    void set_time_scale(float scale) { m_time_scale = std::max(0.0f, scale); }
    float get_time_scale() const { return m_time_scale; }

    void set_pause(bool pause) { m_pause = pause; }
    bool is_pause() const { return m_pause; }

    void set_target_fps(int fps) { m_target_fps = fps; }
    int get_target_fps() const { return m_target_fps; }

    int get_frames_drawn() const { return static_cast<int>(m_frames_drawn); }
    int get_physics_frames() const { return static_cast<int>(m_physics_frames); }
    float get_physics_interpolation_fraction() const { return m_physics_interpolation_fraction; }

    void increment_frames_drawn() { ++m_frames_drawn; }
    void increment_physics_frames() { ++m_physics_frames; }
    void set_physics_interpolation_fraction(float fraction) { m_physics_interpolation_fraction = fraction; }

    void register_singleton(const StringName& name, Object* obj) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_singletons[name] = obj;
    }

    void unregister_singleton(const StringName& name) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_singletons.erase(name);
    }

    Object* get_singleton(const StringName& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_singletons.find(name);
        return it != m_singletons.end() ? it->second : nullptr;
    }

    bool has_singleton(const StringName& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_singletons.find(name) != m_singletons.end();
    }

    std::vector<StringName> get_singleton_list() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<StringName> result;
        for (const auto& kv : m_singletons) {
            result.push_back(kv.first);
        }
        return result;
    }

    void set_config(const String& key, const String& value) { m_config[key] = value; }
    String get_config(const String& key, const String& default_val = "") const {
        auto it = m_config.find(key);
        return it != m_config.end() ? it->second : default_val;
    }

    String get_version() const { return "4.6.0"; }
    String get_version_info() const { return "Godot Engine v4.6.0.xtensor - https://godotengine.org"; }
    String get_architecture() const {
#ifdef XTU_OS_WINDOWS
        return sizeof(void*) == 8 ? "x86_64" : "x86_32";
#else
        return sizeof(void*) == 8 ? "x86_64" : "x86_32";
#endif
    }

    void set_print_error_messages(bool enabled) { m_print_error_enabled = enabled; }
    bool is_printing_error_messages() const { return m_print_error_enabled; }

    Dictionary get_version_info_dict() const {
        Dictionary dict;
        dict["major"] = 4;
        dict["minor"] = 6;
        dict["patch"] = 0;
        dict["status"] = "stable";
        dict["build"] = "xtensor";
        dict["string"] = "4.6.0";
        return dict;
    }

private:
    bool m_print_error_enabled = true;

    void initialize_config() {
        m_config["name"] = "Godot Engine";
        m_config["version"] = "4.6.0";
    }
};

// #############################################################################
// Part 2: CommandLine - Command line argument parser
// #############################################################################

class CommandLine {
private:
    std::vector<String> m_args;
    std::map<String, String> m_options;
    std::vector<String> m_positional;
    bool m_parsed = false;

public:
    CommandLine(int argc, char* argv[]) {
        for (int i = 0; i < argc; ++i) {
            m_args.push_back(String(argv[i]));
        }
    }

    void parse() {
        if (m_parsed) return;

        for (size_t i = 1; i < m_args.size(); ++i) {
            const String& arg = m_args[i];
            if (arg.begins_with("--")) {
                String name = arg.substr(2);
                if (i + 1 < m_args.size() && !m_args[i + 1].begins_with("-")) {
                    m_options[name] = m_args[i + 1];
                    ++i;
                } else {
                    m_options[name] = "true";
                }
            } else if (arg.begins_with("-")) {
                String name = arg.substr(1);
                if (i + 1 < m_args.size() && !m_args[i + 1].begins_with("-")) {
                    m_options[name] = m_args[i + 1];
                    ++i;
                } else {
                    m_options[name] = "true";
                }
            } else {
                m_positional.push_back(arg);
            }
        }
        m_parsed = true;
    }

    bool has_option(const String& name) const { return m_options.find(name) != m_options.end(); }
    String get_option(const String& name, const String& default_val = "") const {
        auto it = m_options.find(name);
        return it != m_options.end() ? it->second : default_val;
    }

    std::vector<String> get_positional_args() const { return m_positional; }

    bool is_editor_mode() const {
        return has_option("editor") || has_option("e");
    }

    bool is_project_manager_mode() const {
        return has_option("project-manager") || m_positional.empty();
    }

    String get_project_path() const {
        if (has_option("path")) return get_option("path");
        if (has_option("p")) return get_option("p");
        if (!m_positional.empty()) return m_positional[0];
        return "";
    }

    bool is_headless() const { return has_option("headless"); }
    bool is_debug() const { return has_option("debug"); }
    String get_rendering_driver() const { return get_option("rendering-driver", "vulkan"); }
    int get_debug_port() const { return get_option("remote-debug-port", "6007").to_int(); }
};

// #############################################################################
// Part 3: SplashScreen - Startup splash screen
// #############################################################################

class SplashScreen {
private:
    static SplashScreen* s_instance;
    Ref<Texture2D> m_logo;
    Ref<Texture2D> m_background;
    Color m_bg_color = Color(0.2f, 0.2f, 0.2f, 1.0f);
    String m_message = "Loading...";
    float m_progress = 0.0f;
    bool m_visible = false;
    bool m_show_progress = true;
    vec2i m_size = {640, 400};
    std::mutex m_mutex;

public:
    static SplashScreen* get_singleton() {
        if (!s_instance) s_instance = new SplashScreen();
        return s_instance;
    }

    void set_logo(const Ref<Texture2D>& logo) { m_logo = logo; }
    void set_background(const Ref<Texture2D>& bg) { m_background = bg; }
    void set_background_color(const Color& color) { m_bg_color = color; }
    void set_size(const vec2i& size) { m_size = size; }

    void show() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_visible = true;
        render();
    }

    void hide() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_visible = false;
    }

    void set_message(const String& msg) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_message = msg;
        if (m_visible) render();
    }

    void set_progress(float progress) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_progress = std::clamp(progress, 0.0f, 1.0f);
        if (m_visible) render();
    }

    void set_show_progress(bool show) { m_show_progress = show; }

private:
    SplashScreen() = default;

    void render() {
        // Platform-specific window creation and rendering
        // This would use DisplayServer to create a borderless window
    }
};

// #############################################################################
// Part 4: ResourcePreloader - Async resource loading
// #############################################################################

class ResourcePreloader : public Object {
    XTU_GODOT_REGISTER_CLASS(ResourcePreloader, Object)

private:
    struct LoadTask {
        String path;
        String type;
        std::function<void(Ref<Resource>)> callback;
    };

    std::queue<LoadTask> m_queue;
    std::unordered_map<String, Ref<Resource>> m_cache;
    std::thread m_worker;
    std::atomic<bool> m_running{false};
    std::mutex m_mutex;
    std::condition_variable m_cv;

public:
    static StringName get_class_static() { return StringName("ResourcePreloader"); }

    ResourcePreloader() { start_worker(); }
    ~ResourcePreloader() { stop_worker(); }

    void add_resource(const String& path, const String& type = "",
                      std::function<void(Ref<Resource>)> callback = nullptr) {
        std::lock_guard<std::mutex> lock(m_mutex);
        LoadTask task;
        task.path = path;
        task.type = type;
        task.callback = callback;
        m_queue.push(task);
        m_cv.notify_one();
    }

    Ref<Resource> get_cached(const String& path) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_cache.find(path);
        return it != m_cache.end() ? it->second : Ref<Resource>();
    }

    bool is_cached(const String& path) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_cache.find(path) != m_cache.end();
    }

    void clear_cache() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cache.clear();
    }

    size_t get_queue_size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.size();
    }

    void wait_for_all() {
        while (get_queue_size() > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

private:
    void start_worker() {
        m_running = true;
        m_worker = std::thread([this]() { worker_loop(); });
    }

    void stop_worker() {
        m_running = false;
        m_cv.notify_all();
        if (m_worker.joinable()) m_worker.join();
    }

    void worker_loop() {
        while (m_running) {
            LoadTask task;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(lock, [this]() { return !m_queue.empty() || !m_running; });
                if (!m_running) break;
                task = std::move(m_queue.front());
                m_queue.pop();
            }

            Ref<Resource> res = ResourceLoader::load(task.path, task.type);
            if (res.is_valid()) {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_cache[task.path] = res;
            }

            if (task.callback) {
                task.callback(res);
            }
        }
    }
};

// #############################################################################
// Part 5: FileSystemWatcher - Directory change monitoring
// #############################################################################

class FileSystemWatcher : public Object {
    XTU_GODOT_REGISTER_CLASS(FileSystemWatcher, Object)

public:
    enum EventType {
        EVENT_CREATED,
        EVENT_DELETED,
        EVENT_MODIFIED,
        EVENT_RENAMED
    };

    struct Event {
        EventType type;
        String path;
        String old_path; // For rename
        uint64_t timestamp = 0;
    };

    using EventCallback = std::function<void(const Event&)>;

private:
    static FileSystemWatcher* s_singleton;
    std::vector<String> m_watch_paths;
    std::unordered_map<String, std::filesystem::file_time_type> m_file_times;
    EventCallback m_callback;
    std::thread m_watch_thread;
    std::atomic<bool> m_running{false};
    std::mutex m_mutex;
    int m_scan_interval_ms = 500;

public:
    static FileSystemWatcher* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("FileSystemWatcher"); }

    FileSystemWatcher() { s_singleton = this; }
    ~FileSystemWatcher() { stop(); s_singleton = nullptr; }

    void add_watch_path(const String& path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (std::find(m_watch_paths.begin(), m_watch_paths.end(), path) == m_watch_paths.end()) {
            m_watch_paths.push_back(path);
        }
    }

    void remove_watch_path(const String& path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = std::find(m_watch_paths.begin(), m_watch_paths.end(), path);
        if (it != m_watch_paths.end()) {
            m_watch_paths.erase(it);
        }
    }

    void set_callback(EventCallback cb) { m_callback = cb; }

    void set_scan_interval(int ms) { m_scan_interval_ms = std::max(100, ms); }

    void start() {
        if (m_running) return;
        m_running = true;
        scan_all_files();
        m_watch_thread = std::thread([this]() { watch_loop(); });
    }

    void stop() {
        m_running = false;
        if (m_watch_thread.joinable()) m_watch_thread.join();
    }

private:
    void scan_all_files() {
        for (const auto& path : m_watch_paths) {
            scan_directory(path);
        }
    }

    void scan_directory(const String& path) {
        std::filesystem::path p(path.to_std_string());
        if (!std::filesystem::exists(p)) return;

        for (const auto& entry : std::filesystem::recursive_directory_iterator(p)) {
            if (entry.is_regular_file()) {
                m_file_times[String(entry.path().string().c_str())] = entry.last_write_time();
            }
        }
    }

    void watch_loop() {
        while (m_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(m_scan_interval_ms));
            check_changes();
        }
    }

    void check_changes() {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& path : m_watch_paths) {
            check_directory_changes(path);
        }
    }

    void check_directory_changes(const String& path) {
        std::filesystem::path p(path.to_std_string());
        if (!std::filesystem::exists(p)) return;

        std::unordered_map<String, std::filesystem::file_time_type> current_files;

        for (const auto& entry : std::filesystem::recursive_directory_iterator(p)) {
            if (entry.is_regular_file()) {
                String file_path(entry.path().string().c_str());
                auto ftime = entry.last_write_time();
                current_files[file_path] = ftime;

                auto it = m_file_times.find(file_path);
                if (it == m_file_times.end()) {
                    notify(Event{EVENT_CREATED, file_path, "", 0});
                } else if (it->second != ftime) {
                    notify(Event{EVENT_MODIFIED, file_path, "", 0});
                }
            }
        }

        // Check for deleted files
        for (const auto& kv : m_file_times) {
            if (current_files.find(kv.first) == current_files.end()) {
                notify(Event{EVENT_DELETED, kv.first, "", 0});
            }
        }

        m_file_times = std::move(current_files);
    }

    void notify(const Event& ev) {
        if (m_callback) {
            m_callback(ev);
        }
    }
};

// #############################################################################
// Part 6: ThumbnailCache - Resource thumbnail cache
// #############################################################################

class ThumbnailCache : public Object {
    XTU_GODOT_REGISTER_CLASS(ThumbnailCache, Object)

private:
    static ThumbnailCache* s_singleton;
    std::unordered_map<String, Ref<Texture2D>> m_cache;
    String m_cache_dir;
    size_t m_max_cache_size = 1000;
    mutable std::mutex m_mutex;

public:
    static ThumbnailCache* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("ThumbnailCache"); }

    ThumbnailCache() {
        s_singleton = this;
        m_cache_dir = OS::get_singleton()->get_cache_dir() + "/thumbnails";
        DirAccess::make_dir_recursive(m_cache_dir);
        load_cache();
    }

    ~ThumbnailCache() { s_singleton = nullptr; }

    Ref<Texture2D> get_thumbnail(const String& path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_cache.find(path);
        if (it != m_cache.end()) {
            return it->second;
        }
        return Ref<Texture2D>();
    }

    void set_thumbnail(const String& path, const Ref<Texture2D>& thumbnail) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cache[path] = thumbnail;
        save_thumbnail(path, thumbnail);
        prune_cache();
    }

    bool has_thumbnail(const String& path) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_cache.find(path) != m_cache.end();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cache.clear();
    }

    Ref<Texture2D> generate_thumbnail(const String& path, const vec2i& size = vec2i(64, 64)) {
        String ext = path.get_extension().to_lower();
        if (ext == "png" || ext == "jpg" || ext == "webp") {
            return generate_image_thumbnail(path, size);
        } else if (ext == "tscn" || ext == "scn") {
            return generate_scene_thumbnail(path, size);
        } else if (ext == "gd") {
            return generate_script_thumbnail(path, size);
        }
        return Ref<Texture2D>();
    }

private:
    void load_cache() {
        Ref<DirAccess> dir = DirAccess::open(m_cache_dir);
        if (!dir.is_valid()) return;

        dir->list_dir_begin();
        String item;
        while (!(item = dir->get_next()).empty()) {
            if (item.ends_with(".png")) {
                String thumb_path = m_cache_dir + "/" + item;
                Ref<Texture2D> tex = ResourceLoader::load(thumb_path);
                if (tex.is_valid()) {
                    String original_path = item.substr(0, item.length() - 4).replace("_", "/");
                    m_cache[original_path] = tex;
                }
            }
        }
        dir->list_dir_end();
    }

    void save_thumbnail(const String& path, const Ref<Texture2D>& thumbnail) {
        String safe_name = path.replace("/", "_");
        String thumb_path = m_cache_dir + "/" + safe_name + ".png";
        ResourceSaver::save(thumbnail, thumb_path);
    }

    void prune_cache() {
        if (m_cache.size() <= m_max_cache_size) return;

        // Remove oldest entries (simplified - would use LRU in production)
        size_t to_remove = m_cache.size() - m_max_cache_size;
        auto it = m_cache.begin();
        while (to_remove > 0 && it != m_cache.end()) {
            it = m_cache.erase(it);
            --to_remove;
        }
    }

    Ref<Texture2D> generate_image_thumbnail(const String& path, const vec2i& size) {
        Ref<Image> img;
        img.instance();
        if (img->load(path) != OK) return Ref<Texture2D>();

        img->resize(size.x(), size.y(), Image::INTERPOLATE_LANCZOS);

        Ref<ImageTexture> tex;
        tex.instance();
        tex->create_from_image(img);
        return tex;
    }

    Ref<Texture2D> generate_scene_thumbnail(const String&, const vec2i&) {
        return Ref<Texture2D>();
    }

    Ref<Texture2D> generate_script_thumbnail(const String&, const vec2i&) {
        return Ref<Texture2D>();
    }
};

// #############################################################################
// Part 7: CrashHandler - Crash reporting
// #############################################################################

class CrashHandler {
private:
    static CrashHandler* s_instance;
    std::function<void(const String&, const String&)> m_callback;
    String m_report_path;
    bool m_installed = false;

public:
    static CrashHandler* get_singleton() {
        if (!s_instance) s_instance = new CrashHandler();
        return s_instance;
    }

    void install(const String& report_path = "") {
        if (m_installed) return;
        m_report_path = report_path.empty() ? OS::get_singleton()->get_user_data_dir() + "/crashes" : report_path;
        DirAccess::make_dir_recursive(m_report_path);
        m_installed = true;
    }

    void set_callback(std::function<void(const String&, const String&)> cb) { m_callback = cb; }

    String generate_report(const String& context = "") {
        String report;
        report += "=== Godot Crash Report ===\n";
        report += "Version: " + Engine::get_singleton()->get_version() + "\n";
        report += "OS: " + OS::get_singleton()->get_name() + "\n";
        report += "Time: " + String::num(OS::get_singleton()->get_unix_time()) + "\n";
        report += "Context: " + context + "\n\n";
        report += "Stack trace not available.\n";
        return report;
    }

    void save_report(const String& report) {
        String filename = m_report_path + "/crash_" + String::num(OS::get_singleton()->get_unix_time()) + ".txt";
        Ref<FileAccess> file = FileAccess::open(filename, FileAccess::WRITE);
        if (file.is_valid()) {
            file->store_string(report);
        }
    }
};

CrashHandler* CrashHandler::s_instance = nullptr;
SplashScreen* SplashScreen::s_instance = nullptr;

} // namespace utils

// Bring into main namespace
using utils::Engine;
using utils::CommandLine;
using utils::SplashScreen;
using utils::ResourcePreloader;
using utils::FileSystemWatcher;
using utils::ThumbnailCache;
using utils::CrashHandler;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XENGINE_UTILS_HPP