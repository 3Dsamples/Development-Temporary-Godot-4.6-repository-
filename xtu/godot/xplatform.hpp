// include/xtu/godot/xplatform.hpp
// xtensor-unified - Platform abstraction layer for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XPLATFORM_HPP
#define XTU_GODOT_XPLATFORM_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xcore.hpp"
#include "xtu/godot/xinput.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/godot/xmain_timer.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace platform {

// #############################################################################
// Platform detection
// #############################################################################
enum class PlatformType : uint8_t {
    PLATFORM_UNKNOWN = 0,
    PLATFORM_WINDOWS = 1,
    PLATFORM_LINUX = 2,
    PLATFORM_MACOS = 3,
    PLATFORM_ANDROID = 4,
    PLATFORM_IOS = 5,
    PLATFORM_WEB = 6
};

inline PlatformType get_current_platform() {
#ifdef XTU_OS_WINDOWS
    return PlatformType::PLATFORM_WINDOWS;
#elif defined(XTU_OS_LINUX)
    return PlatformType::PLATFORM_LINUX;
#elif defined(XTU_OS_MACOS)
    return PlatformType::PLATFORM_MACOS;
#elif defined(XTU_OS_ANDROID)
    return PlatformType::PLATFORM_ANDROID;
#elif defined(XTU_OS_IOS)
    return PlatformType::PLATFORM_IOS;
#elif defined(XTU_OS_WEB)
    return PlatformType::PLATFORM_WEB;
#else
    return PlatformType::PLATFORM_UNKNOWN;
#endif
}

// #############################################################################
// Window mode and flags
// #############################################################################
enum class WindowMode : uint8_t {
    MODE_WINDOWED = 0,
    MODE_FULLSCREEN = 1,
    MODE_EXCLUSIVE_FULLSCREEN = 2,
    MODE_MAXIMIZED = 3,
    MODE_MINIMIZED = 4
};

enum class WindowFlags : uint32_t {
    FLAG_NONE = 0,
    FLAG_RESIZABLE = 1 << 0,
    FLAG_BORDERLESS = 1 << 1,
    FLAG_ALWAYS_ON_TOP = 1 << 2,
    FLAG_TRANSPARENT = 1 << 3,
    FLAG_NO_FOCUS = 1 << 4
};

inline WindowFlags operator|(WindowFlags a, WindowFlags b) {
    return static_cast<WindowFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

// #############################################################################
// Display server events
// #############################################################################
enum class DisplayServerEvent : uint8_t {
    EVENT_NONE = 0,
    EVENT_WINDOW_RESIZE = 1,
    EVENT_WINDOW_MOVE = 2,
    EVENT_WINDOW_FOCUS = 3,
    EVENT_WINDOW_CLOSE = 4,
    EVENT_MOUSE_ENTER = 5,
    EVENT_MOUSE_EXIT = 6,
    EVENT_SCREEN_CHANGED = 7,
    EVENT_DPI_CHANGED = 8
};

// #############################################################################
// DisplayServer - Window and display management singleton
// #############################################################################
class DisplayServer : public Object {
    XTU_GODOT_REGISTER_CLASS(DisplayServer, Object)

public:
    static constexpr int MAIN_WINDOW_ID = 0;
    static constexpr int INVALID_WINDOW_ID = -1;

    enum CursorShape {
        CURSOR_ARROW = 0,
        CURSOR_IBEAM = 1,
        CURSOR_POINTING_HAND = 2,
        CURSOR_CROSS = 3,
        CURSOR_WAIT = 4,
        CURSOR_BUSY = 5,
        CURSOR_DRAG = 6,
        CURSOR_CAN_DROP = 7,
        CURSOR_FORBIDDEN = 8,
        CURSOR_VSIZE = 9,
        CURSOR_HSIZE = 10,
        CURSOR_BDIAGSIZE = 11,
        CURSOR_FDIAGSIZE = 12,
        CURSOR_MOVE = 13,
        CURSOR_VSPLIT = 14,
        CURSOR_HSPLIT = 15,
        CURSOR_HELP = 16,
        CURSOR_MAX = 17
    };

    enum FileDialogMode {
        FILE_DIALOG_MODE_OPEN_FILE = 0,
        FILE_DIALOG_MODE_OPEN_FILES = 1,
        FILE_DIALOG_MODE_OPEN_DIR = 2,
        FILE_DIALOG_MODE_SAVE_FILE = 3
    };

private:
    static DisplayServer* s_singleton;
    vec2i m_window_size = {1280, 720};
    vec2i m_window_position = {100, 100};
    WindowMode m_window_mode = WindowMode::MODE_WINDOWED;
    WindowFlags m_window_flags = WindowFlags::FLAG_RESIZABLE;
    String m_window_title = "Godot Engine";
    bool m_window_visible = true;
    CursorShape m_cursor_shape = CURSOR_ARROW;
    std::map<int, vec2i> m_custom_cursor_hotspots;
    std::vector<vec2i> m_screen_resolutions;
    int m_current_screen = 0;
    float m_screen_scale = 1.0f;
    std::mutex m_mutex;
    std::function<void(DisplayServerEvent, const Variant&)> m_event_callback;

public:
    static DisplayServer* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("DisplayServer"); }

    DisplayServer() { s_singleton = this; }
    ~DisplayServer() { s_singleton = nullptr; }

    // #########################################################################
    // Window management
    // #########################################################################
    void set_window_size(const vec2i& size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_window_size = size;
        apply_window_size();
    }

    vec2i get_window_size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_window_size;
    }

    void set_window_position(const vec2i& pos) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_window_position = pos;
        apply_window_position();
    }

    vec2i get_window_position() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_window_position;
    }

    void set_window_mode(WindowMode mode) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_window_mode = mode;
        apply_window_mode();
    }

    WindowMode get_window_mode() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_window_mode;
    }

    void set_window_flags(WindowFlags flags) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_window_flags = flags;
        apply_window_flags();
    }

    WindowFlags get_window_flags() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_window_flags;
    }

    void set_window_title(const String& title) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_window_title = title;
        apply_window_title();
    }

    String get_window_title() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_window_title;
    }

    void set_window_visible(bool visible) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_window_visible = visible;
        apply_window_visible();
    }

    bool is_window_visible() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_window_visible;
    }

    void set_window_min_size(const vec2i& size) {
        // Platform-specific
    }

    void set_window_max_size(const vec2i& size) {
        // Platform-specific
    }

    void center_window() {
        vec2i screen_size = get_screen_size();
        set_window_position((screen_size - m_window_size) / 2);
    }

    void request_attention() {
        // Flash window in taskbar
    }

    // #########################################################################
    // Screen management
    // #########################################################################
    int get_screen_count() const {
        return static_cast<int>(m_screen_resolutions.size());
    }

    vec2i get_screen_size(int screen = -1) const {
        if (screen < 0) screen = m_current_screen;
        if (screen >= 0 && screen < static_cast<int>(m_screen_resolutions.size())) {
            return m_screen_resolutions[screen];
        }
        return vec2i(1920, 1080);
    }

    vec2i get_screen_position(int screen = -1) const {
        return vec2i(0, 0);
    }

    float get_screen_scale(int screen = -1) const {
        return m_screen_scale;
    }

    float get_screen_refresh_rate(int screen = -1) const {
        return 60.0f;
    }

    void set_current_screen(int screen) {
        m_current_screen = screen;
    }

    int get_current_screen() const {
        return m_current_screen;
    }

    // #########################################################################
    // Cursor management
    // ########################################################################
    void set_cursor_shape(CursorShape shape) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cursor_shape = shape;
        apply_cursor_shape();
    }

    CursorShape get_cursor_shape() const {
        return m_cursor_shape;
    }

    void set_custom_mouse_cursor(const Ref<Texture2D>& cursor, CursorShape shape = CURSOR_ARROW, const vec2i& hotspot = vec2i(0, 0)) {
        m_custom_cursor_hotspots[static_cast<int>(shape)] = hotspot;
        // Store texture reference
    }

    void set_mouse_mode(Input::MouseMode mode) {
        Input::get_singleton()->set_mouse_mode(mode);
    }

    Input::MouseMode get_mouse_mode() const {
        return Input::get_singleton()->get_mouse_mode();
    }

    void warp_mouse(const vec2i& position) {
        // Warp mouse to position
    }

    vec2i mouse_get_position() const {
        return vec2i(0, 0);
    }

    // #########################################################################
    // Clipboard
    // #########################################################################
    void clipboard_set(const String& text) {
#ifdef XTU_OS_WINDOWS
        // Windows clipboard
#elif defined(XTU_OS_LINUX)
        // X11/Wayland clipboard
#elif defined(XTU_OS_MACOS)
        // Cocoa clipboard
#endif
    }

    String clipboard_get() const {
#ifdef XTU_OS_WINDOWS
        return String();
#elif defined(XTU_OS_LINUX)
        return String();
#elif defined(XTU_OS_MACOS)
        return String();
#else
        return String();
#endif
    }

    void clipboard_set_primary(const String& text) {
        // X11 primary selection (Linux only)
    }

    String clipboard_get_primary() const {
        return String();
    }

    // #########################################################################
    // Native dialogs
    // #########################################################################
    void file_dialog_show(const String& title, const String& current_dir, const String& filename,
                          FileDialogMode mode, const std::vector<String>& filters,
                          std::function<void(bool, const std::vector<String>&)> callback) {
        // Show platform native file dialog
    }

    void message_dialog_show(const String& title, const String& text,
                             const std::vector<String>& buttons,
                             std::function<void(int)> callback) {
        // Show platform native message box
    }

    void input_dialog_show(const String& title, const String& description, const String& placeholder,
                           std::function<void(const String&)> callback) {
        // Show platform native input dialog
    }

    void color_picker_show(const Color& initial_color,
                           std::function<void(const Color&)> callback) {
        // Show platform native color picker
    }

    // #########################################################################
    // System information
    // #########################################################################
    String get_system_font_path(const String& font_name) const {
        return String();
    }

    std::vector<String> get_system_fonts() const {
        return {};
    }

    String get_locale() const {
        return "en_US";
    }

    // #########################################################################
    // Power management
    // #########################################################################
    void set_screen_keep_on(bool enable) {
        // Prevent screen sleep
    }

    bool get_screen_keep_on() const {
        return false;
    }

    int get_power_state() const {
        return 0; // Unknown
    }

    int get_seconds_left() const {
        return -1;
    }

    // #########################################################################
    // Virtual keyboard
    // #########################################################################
    void virtual_keyboard_show(const String& existing_text, int selection_start, int selection_end) {
        // Show virtual keyboard on mobile
    }

    void virtual_keyboard_hide() {
        // Hide virtual keyboard
    }

    int virtual_keyboard_get_height() const {
        return 0;
    }

    // #########################################################################
    // Event handling
    // #########################################################################
    void set_event_callback(std::function<void(DisplayServerEvent, const Variant&)> callback) {
        m_event_callback = callback;
    }

    void process_events() {
        // Pump platform events
    }

private:
    void apply_window_size() {
        // Platform-specific window resize
    }

    void apply_window_position() {
        // Platform-specific window move
    }

    void apply_window_mode() {
        // Platform-specific fullscreen/windowed
    }

    void apply_window_flags() {
        // Platform-specific window style
    }

    void apply_window_title() {
        // Platform-specific window title
    }

    void apply_window_visible() {
        // Platform-specific show/hide
    }

    void apply_cursor_shape() {
        // Platform-specific cursor
    }
};

// #############################################################################
// PlatformOS - Extended OS functionality
// #############################################################################
class PlatformOS : public OS {
    XTU_GODOT_REGISTER_CLASS(PlatformOS, OS)

private:
    static PlatformOS* s_singleton;
    DisplayServer* m_display_server = nullptr;
    bool m_low_processor_usage_mode = true;
    bool m_verbose_stdout = false;
    int m_exit_code = 0;

public:
    static PlatformOS* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("PlatformOS"); }

    PlatformOS() {
        s_singleton = this;
        m_display_server = DisplayServer::get_singleton();
    }

    ~PlatformOS() { s_singleton = nullptr; }

    // #########################################################################
    // Main loop
    // #########################################################################
    int run() {
        initialize();
        main_loop();
        finalize();
        return m_exit_code;
    }

    void initialize() {
        // Initialize platform subsystems
        m_display_server->set_window_visible(true);
    }

    void main_loop() {
        MainTimerSync* timer = MainTimerSync::get_singleton();
        RenderingServer* rendering = RenderingServer::get_singleton();

        while (!timer->is_exit_requested()) {
            m_display_server->process_events();
            timer->begin_frame();

            // Process physics steps
            while (timer->should_physics_step()) {
                // Physics update
            }

            // Process frame
            // Scene update
            // Render
            rendering->render_frame();

            timer->end_frame();
        }
    }

    void finalize() {
        // Cleanup platform subsystems
    }

    void set_exit_code(int code) { m_exit_code = code; }

    // #########################################################################
    // Process execution
    // #########################################################################
    int execute(const String& path, const std::vector<String>& arguments,
                bool blocking = true, std::vector<uint8_t>* output = nullptr,
                std::vector<uint8_t>* error = nullptr, bool open_console = false) {
#ifdef XTU_OS_WINDOWS
        // CreateProcess
#else
        // fork + exec
#endif
        return -1;
    }

    Error shell_open(const String& uri) {
#ifdef XTU_OS_WINDOWS
        ShellExecuteA(nullptr, "open", uri.utf8(), nullptr, nullptr, SW_SHOW);
#elif defined(XTU_OS_LINUX)
        system(("xdg-open " + uri.to_std_string()).c_str());
#elif defined(XTU_OS_MACOS)
        system(("open " + uri.to_std_string()).c_str());
#endif
        return OK;
    }

    // #########################################################################
    // System directories
    // #########################################################################
    String get_user_data_dir() const override {
#ifdef XTU_OS_WINDOWS
        return String(std::getenv("APPDATA")) + "/Godot";
#elif defined(XTU_OS_MACOS)
        return String(std::getenv("HOME")) + "/Library/Application Support/Godot";
#else
        return String(std::getenv("HOME")) + "/.local/share/godot";
#endif
    }

    String get_config_dir() const override {
#ifdef XTU_OS_WINDOWS
        return String(std::getenv("APPDATA")) + "/Godot";
#elif defined(XTU_OS_MACOS)
        return String(std::getenv("HOME")) + "/Library/Application Support/Godot";
#else
        return String(std::getenv("HOME")) + "/.config/godot";
#endif
    }

    String get_cache_dir() const override {
#ifdef XTU_OS_WINDOWS
        return String(std::getenv("LOCALAPPDATA")) + "/Godot/cache";
#elif defined(XTU_OS_MACOS)
        return String(std::getenv("HOME")) + "/Library/Caches/Godot";
#else
        return String(std::getenv("HOME")) + "/.cache/godot";
#endif
    }

    String get_documents_dir() const override {
#ifdef XTU_OS_WINDOWS
        return String(std::getenv("USERPROFILE")) + "/Documents";
#else
        return String(std::getenv("HOME")) + "/Documents";
#endif
    }

    String get_downloads_dir() const override {
#ifdef XTU_OS_WINDOWS
        return String(std::getenv("USERPROFILE")) + "/Downloads";
#else
        return String(std::getenv("HOME")) + "/Downloads";
#endif
    }

    String get_desktop_dir() const override {
#ifdef XTU_OS_WINDOWS
        return String(std::getenv("USERPROFILE")) + "/Desktop";
#else
        return String(std::getenv("HOME")) + "/Desktop";
#endif
    }

    // #########################################################################
    // Mobile platform features
    // #########################################################################
    void set_orientation(int orientation) {
        // Mobile orientation lock
    }

    int get_orientation() const {
        return 0;
    }

    void set_keep_screen_on(bool enable) {
        m_display_server->set_screen_keep_on(enable);
    }

    bool get_keep_screen_on() const {
        return m_display_server->get_screen_keep_on();
    }

    void request_permission(const String& permission) {
        // Request runtime permission (Android/iOS)
    }

    bool has_permission(const String& permission) const {
        return true;
    }

    // #########################################################################
    // Web platform features
    // #########################################################################
    void eval_javascript(const String& code) {
        // Emscripten JS evaluation
    }

    void download_buffer(const std::vector<uint8_t>& data, const String& filename) {
        // Trigger browser download
    }
};

// #############################################################################
// Platform-specific main entry point
// #############################################################################
#ifdef XTU_OS_WINDOWS
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    PlatformOS os;
    return os.run();
}
#else
int main(int argc, char* argv[]) {
    PlatformOS os;
    return os.run();
}
#endif

} // namespace platform

// Bring into main namespace
using platform::DisplayServer;
using platform::PlatformOS;
using platform::PlatformType;
using platform::WindowMode;
using platform::WindowFlags;
using platform::DisplayServerEvent;
using platform::get_current_platform;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XPLATFORM_HPP