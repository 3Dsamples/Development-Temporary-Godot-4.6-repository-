// include/xtu/godot/xexporter.hpp
// xtensor-unified - Export system for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XEXPORTER_HPP
#define XTU_GODOT_XEXPORTER_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xeditor_export.hpp"
#include "xtu/godot/xcrypto.hpp"
#include "xtu/io/xio_json.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class WindowsExporter;
class LinuxExporter;
class MacOSExporter;
class AndroidExporter;
class IOSExporter;
class WebExporter;
class ExportManager;
class ExportReport;

// #############################################################################
// Export report status
// #############################################################################
enum class ExportReportStatus : uint8_t {
    STATUS_PENDING = 0,
    STATUS_PROCESSING = 1,
    STATUS_SUCCESS = 2,
    STATUS_WARNING = 3,
    STATUS_ERROR = 4
};

// #############################################################################
// Export report entry
// #############################################################################
struct ExportReportEntry {
    ExportReportStatus status = ExportReportStatus::STATUS_PENDING;
    String platform;
    String message;
    String detail;
    uint64_t timestamp = 0;
    size_t file_count = 0;
    size_t total_size = 0;
};

// #############################################################################
// ExportReport - Tracks export progress and results
// #############################################################################
class ExportReport : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(ExportReport, RefCounted)

private:
    std::vector<ExportReportEntry> m_entries;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("ExportReport"); }

    void add_entry(ExportReportStatus status, const String& platform, const String& message, const String& detail = "") {
        std::lock_guard<std::mutex> lock(m_mutex);
        ExportReportEntry entry;
        entry.status = status;
        entry.platform = platform;
        entry.message = message;
        entry.detail = detail;
        entry.timestamp = OS::get_singleton()->get_ticks_msec();
        m_entries.push_back(entry);
    }

    std::vector<ExportReportEntry> get_entries() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_entries;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_entries.clear();
    }

    String generate_summary() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        String summary;
        for (const auto& entry : m_entries) {
            String status_str;
            switch (entry.status) {
                case ExportReportStatus::STATUS_SUCCESS: status_str = "[OK]"; break;
                case ExportReportStatus::STATUS_WARNING: status_str = "[WARN]"; break;
                case ExportReportStatus::STATUS_ERROR: status_str = "[ERROR]"; break;
                default: status_str = "[...]"; break;
            }
            summary += status_str + " " + entry.platform + ": " + entry.message + "\n";
        }
        return summary;
    }
};

// #############################################################################
// ExportConfig - Per-platform export configuration
// #############################################################################
struct ExportConfig {
    String name;
    String export_path;
    bool debug = false;
    bool embed_pck = false;
    String icon_path;
    String application_name;
    String bundle_identifier;
    String version;
    String version_code;
    std::map<String, Variant> custom_options;
    std::vector<String> exclude_patterns;
    std::vector<String> include_patterns;
};

// #############################################################################
// WindowsExporter - Windows platform exporter
// #############################################################################
class WindowsExporter : public EditorExportPlatform {
    XTU_GODOT_REGISTER_CLASS(WindowsExporter, EditorExportPlatform)

private:
    std::vector<String> m_binary_extensions = {"exe", "console.exe"};

public:
    static StringName get_class_static() { return StringName("WindowsExporter"); }

    WindowsExporter() {
        m_name = "Windows Desktop";
        m_os_name = "Windows";
    }

    std::vector<String> get_binary_extensions() const override { return m_binary_extensions; }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back({VariantType::STRING, "custom_template/debug"});
        options.push_back({VariantType::STRING, "custom_template/release"});
        options.push_back({VariantType::BOOL, "binary_format/embed_pck"});
        options.push_back({VariantType::BOOL, "binary_format/console_wrapper"});
        options.push_back({VariantType::STRING, "application/icon"});
        options.push_back({VariantType::STRING, "application/file_version"});
        options.push_back({VariantType::STRING, "application/product_version"});
        options.push_back({VariantType::STRING, "application/company_name"});
        options.push_back({VariantType::STRING, "application/product_name"});
        options.push_back({VariantType::STRING, "application/file_description"});
        options.push_back({VariantType::STRING, "application/copyright"});
        options.push_back({VariantType::STRING, "application/trademarks"});
        options.push_back({VariantType::BOOL, "codesign/enable"});
        options.push_back({VariantType::STRING, "codesign/identity"});
        options.push_back({VariantType::STRING, "codesign/password"});
        options.push_back({VariantType::STRING, "codesign/timestamp_url"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("windows");
        features.push_back("pc");
        features.push_back("x86_64");
        if (preset.is_valid() && preset->get_option("binary_format/console_wrapper", false).as<bool>()) {
            features.push_back("console");
        }
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        ExportConfig config = build_config(preset, debug, path);
        Ref<ExportReport> report;
        report.instance();
        
        // Build PCK file
        String pck_path = path.get_basename() + ".pck";
        Error err = export_pack(preset, debug, pck_path, flags);
        if (err != OK) {
            report->add_entry(ExportReportStatus::STATUS_ERROR, m_name, "Failed to build PCK file");
            return err;
        }
        
        // Get template executable
        String template_path = get_template_path(preset, debug);
        if (template_path.empty()) {
            report->add_entry(ExportReportStatus::STATUS_ERROR, m_name, "Template executable not found");
            return ERR_FILE_NOT_FOUND;
        }
        
        // Copy template to output
        Ref<FileAccess> src = FileAccess::open(template_path, FileAccess::READ);
        Ref<FileAccess> dst = FileAccess::open(path, FileAccess::WRITE);
        if (!src.is_valid() || !dst.is_valid()) {
            return ERR_FILE_CANT_WRITE;
        }
        
        std::vector<uint8_t> exe_data = src->get_buffer(src->get_length());
        dst->store_buffer(exe_data);
        
        // Embed PCK if requested
        if (preset->get_option("binary_format/embed_pck", true).as<bool>()) {
            embed_pck_in_exe(path, pck_path);
            DirAccess::remove(pck_path);
        }
        
        // Apply icon
        if (preset->has_option("application/icon")) {
            String icon_path = preset->get_option("application/icon").as<String>();
            if (!icon_path.empty()) {
                set_exe_icon(path, icon_path);
            }
        }
        
        // Code signing
        if (preset->get_option("codesign/enable", false).as<bool>()) {
            String identity = preset->get_option("codesign/identity").as<String>();
            String password = preset->get_option("codesign/password").as<String>();
            String timestamp = preset->get_option("codesign/timestamp_url").as<String>();
            sign_executable(path, identity, password, timestamp);
        }
        
        report->add_entry(ExportReportStatus::STATUS_SUCCESS, m_name, "Export completed", path);
        return OK;
    }

private:
    ExportConfig build_config(const Ref<EditorExportPreset>& preset, bool debug, const String& path) {
        ExportConfig cfg;
        cfg.name = preset->get_name();
        cfg.export_path = path;
        cfg.debug = debug;
        cfg.embed_pck = preset->get_option("binary_format/embed_pck", true).as<bool>();
        cfg.icon_path = preset->get_option("application/icon").as<String>();
        cfg.application_name = preset->get_option("application/product_name").as<String>();
        cfg.version = preset->get_option("application/file_version").as<String>();
        return cfg;
    }

    String get_template_path(const Ref<EditorExportPreset>& preset, bool debug) {
        String template_key = debug ? "custom_template/debug" : "custom_template/release";
        String custom = preset->get_option(template_key).as<String>();
        if (!custom.empty() && FileAccess::file_exists(custom)) {
            return custom;
        }
        // Fall back to built-in template
        String template_dir = OS::get_singleton()->get_executable_path().get_base_dir() + "/templates";
        String arch = "windows_x86_64";
        String exe_name = debug ? "godot.windows.template_debug.x86_64.exe" : "godot.windows.template_release.x86_64.exe";
        return template_dir + "/" + arch + "/" + exe_name;
    }

    void embed_pck_in_exe(const String& exe_path, const String& pck_path) {
        // Append PCK data to EXE with marker
        Ref<FileAccess> exe = FileAccess::open(exe_path, FileAccess::READ_WRITE);
        Ref<FileAccess> pck = FileAccess::open(pck_path, FileAccess::READ);
        if (!exe.is_valid() || !pck.is_valid()) return;
        
        exe->seek_end();
        std::vector<uint8_t> pck_data = pck->get_buffer(pck->get_length());
        
        // Write marker and size
        const char marker[] = "GDPCK";
        exe->store_buffer({marker, marker + 5});
        exe->store_64(pck_data.size());
        exe->store_buffer(pck_data);
    }

    void set_exe_icon(const String& exe_path, const String& icon_path) {
        // Windows: UpdateResource with icon
    }

    void sign_executable(const String& exe_path, const String& identity, const String& password, const String& timestamp) {
        // Windows: signtool.exe
    }
};

// #############################################################################
// LinuxExporter - Linux platform exporter
// #############################################################################
class LinuxExporter : public EditorExportPlatform {
    XTU_GODOT_REGISTER_CLASS(LinuxExporter, EditorExportPlatform)

private:
    std::vector<String> m_binary_extensions = {"x86_32", "x86_64", "arm32", "arm64"};

public:
    static StringName get_class_static() { return StringName("LinuxExporter"); }

    LinuxExporter() {
        m_name = "Linux/X11";
        m_os_name = "Linux";
    }

    std::vector<String> get_binary_extensions() const override { return m_binary_extensions; }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back({VariantType::STRING, "custom_template/debug"});
        options.push_back({VariantType::STRING, "custom_template/release"});
        options.push_back({VariantType::BOOL, "binary_format/embed_pck"});
        options.push_back({VariantType::STRING, "binary_format/architecture", PropertyHint::ENUM, "x86_64,x86_32,arm64,arm32"});
        options.push_back({VariantType::STRING, "application/icon"});
        options.push_back({VariantType::STRING, "application/name"});
        options.push_back({VariantType::STRING, "application/version"});
        options.push_back({VariantType::BOOL, "package/appimage"});
        options.push_back({VariantType::STRING, "package/categories"});
        options.push_back({VariantType::STRING, "package/comment"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("linux");
        features.push_back("pc");
        features.push_back("x11");
        String arch = preset.is_valid() ? preset->get_option("binary_format/architecture", "x86_64").as<String>() : "x86_64";
        features.push_back(arch);
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        String arch = preset->get_option("binary_format/architecture", "x86_64").as<String>();
        String template_path = get_template_path(preset, debug, arch);
        if (template_path.empty()) return ERR_FILE_NOT_FOUND;
        
        // Copy template
        Ref<FileAccess> src = FileAccess::open(template_path, FileAccess::READ);
        Ref<FileAccess> dst = FileAccess::open(path, FileAccess::WRITE);
        if (!src.is_valid() || !dst.is_valid()) return ERR_FILE_CANT_WRITE;
        
        dst->store_buffer(src->get_buffer(src->get_length()));
        
        // Set executable permission
#ifndef XTU_OS_WINDOWS
        chmod(path.utf8(), 0755);
#endif
        
        // Build PCK
        String pck_path = path.get_basename() + ".pck";
        export_pack(preset, debug, pck_path, flags);
        
        if (preset->get_option("binary_format/embed_pck", false).as<bool>()) {
            embed_pck_in_elf(path, pck_path);
        }
        
        // Create AppImage if requested
        if (preset->get_option("package/appimage", false).as<bool>()) {
            create_appimage(path, preset);
        }
        
        return OK;
    }

private:
    String get_template_path(const Ref<EditorExportPreset>& preset, bool debug, const String& arch) {
        String template_key = debug ? "custom_template/debug" : "custom_template/release";
        String custom = preset->get_option(template_key).as<String>();
        if (!custom.empty()) return custom;
        
        String template_dir = OS::get_singleton()->get_executable_path().get_base_dir() + "/templates";
        String bin_name = debug ? "godot.linuxbsd.template_debug." : "godot.linuxbsd.template_release.";
        return template_dir + "/linux_" + arch + "/" + bin_name + arch;
    }

    void embed_pck_in_elf(const String& exe_path, const String& pck_path) {
        // ELF section injection
    }

    void create_appimage(const String& exe_path, const Ref<EditorExportPreset>& preset) {
        // Build AppImage structure
    }
};

// #############################################################################
// MacOSExporter - macOS platform exporter
// #############################################################################
class MacOSExporter : public EditorExportPlatform {
    XTU_GODOT_REGISTER_CLASS(MacOSExporter, EditorExportPlatform)

private:
    std::vector<String> m_binary_extensions = {"zip", "dmg"};

public:
    static StringName get_class_static() { return StringName("MacOSExporter"); }

    MacOSExporter() {
        m_name = "macOS";
        m_os_name = "macOS";
    }

    std::vector<String> get_binary_extensions() const override { return m_binary_extensions; }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back({VariantType::STRING, "custom_template/debug"});
        options.push_back({VariantType::STRING, "custom_template/release"});
        options.push_back({VariantType::STRING, "application/icon"});
        options.push_back({VariantType::STRING, "application/bundle_identifier"});
        options.push_back({VariantType::STRING, "application/signature"});
        options.push_back({VariantType::STRING, "application/short_version"});
        options.push_back({VariantType::STRING, "application/version"});
        options.push_back({VariantType::STRING, "application/copyright"});
        options.push_back({VariantType::BOOL, "application/codesign/enable"});
        options.push_back({VariantType::STRING, "application/codesign/identity"});
        options.push_back({VariantType::BOOL, "application/notarization/enable"});
        options.push_back({VariantType::STRING, "application/notarization/apple_id"});
        options.push_back({VariantType::STRING, "application/notarization/team_id"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("macos");
        features.push_back("pc");
        features.push_back("x86_64");
        features.push_back("arm64");
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        String temp_dir = OS::get_singleton()->get_user_data_dir() + "/temp/macos_export";
        DirAccess::make_dir_recursive(temp_dir);
        
        // Create .app bundle structure
        String bundle_name = preset->get_option("application/name", "GodotApp").as<String>();
        String app_dir = temp_dir + "/" + bundle_name + ".app";
        String contents_dir = app_dir + "/Contents";
        String macos_dir = contents_dir + "/MacOS";
        String resources_dir = contents_dir + "/Resources";
        
        DirAccess::make_dir_recursive(macos_dir);
        DirAccess::make_dir_recursive(resources_dir);
        
        // Copy executable
        String template_path = get_template_path(preset, debug);
        String exe_path = macos_dir + "/" + bundle_name;
        Ref<FileAccess> src = FileAccess::open(template_path, FileAccess::READ);
        Ref<FileAccess> dst = FileAccess::open(exe_path, FileAccess::WRITE);
        if (src.is_valid() && dst.is_valid()) {
            dst->store_buffer(src->get_buffer(src->get_length()));
            chmod(exe_path.utf8(), 0755);
        }
        
        // Create Info.plist
        String plist = generate_info_plist(preset);
        Ref<FileAccess> plist_file = FileAccess::open(contents_dir + "/Info.plist", FileAccess::WRITE);
        if (plist_file.is_valid()) plist_file->store_string(plist);
        
        // Copy icon
        String icon_path = preset->get_option("application/icon").as<String>();
        if (!icon_path.empty()) {
            DirAccess::copy(icon_path, resources_dir + "/Icon.icns");
        }
        
        // Export PCK
        String pck_path = resources_dir + "/" + bundle_name + ".pck";
        export_pack(preset, debug, pck_path, flags);
        
        // Code signing
        if (preset->get_option("application/codesign/enable", false).as<bool>()) {
            String identity = preset->get_option("application/codesign/identity").as<String>();
            sign_app_bundle(app_dir, identity);
        }
        
        // Package
        String final_path = path;
        if (final_path.ends_with(".zip")) {
            zip_directory(app_dir, final_path);
        } else if (final_path.ends_with(".dmg")) {
            create_dmg(app_dir, final_path);
        }
        
        DirAccess::remove(temp_dir);
        return OK;
    }

private:
    String get_template_path(const Ref<EditorExportPreset>& preset, bool debug) {
        String template_key = debug ? "custom_template/debug" : "custom_template/release";
        return preset->get_option(template_key).as<String>();
    }

    String generate_info_plist(const Ref<EditorExportPreset>& preset) {
        String bundle_id = preset->get_option("application/bundle_identifier", "com.example.game").as<String>();
        String version = preset->get_option("application/version", "1.0").as<String>();
        String short_version = preset->get_option("application/short_version", version).as<String>();
        String copyright = preset->get_option("application/copyright", "").as<String>();
        String name = preset->get_option("application/name", "GodotApp").as<String>();
        
        return R"(<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>)" + name + R"(</string>
    <key>CFBundleIdentifier</key>
    <string>)" + bundle_id + R"(</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>)" + name + R"(</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>)" + short_version + R"(</string>
    <key>CFBundleVersion</key>
    <string>)" + version + R"(</string>
    <key>CFBundleSignature</key>
    <string>???</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
</dict>
</plist>)";
    }

    void sign_app_bundle(const String& app_path, const String& identity) {
        // codesign --deep --force --sign "identity" app_path
    }

    void zip_directory(const String& dir_path, const String& zip_path) {
        // zip -r zip_path dir_path
    }

    void create_dmg(const String& app_path, const String& dmg_path) {
        // hdiutil create
    }
};

// #############################################################################
// AndroidExporter - Android platform exporter
// #############################################################################
class AndroidExporter : public EditorExportPlatform {
    XTU_GODOT_REGISTER_CLASS(AndroidExporter, EditorExportPlatform)

private:
    std::vector<String> m_binary_extensions = {"apk", "aab"};

public:
    static StringName get_class_static() { return StringName("AndroidExporter"); }

    AndroidExporter() {
        m_name = "Android";
        m_os_name = "Android";
    }

    std::vector<String> get_binary_extensions() const override { return m_binary_extensions; }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back({VariantType::STRING, "custom_template/debug"});
        options.push_back({VariantType::STRING, "custom_template/release"});
        options.push_back({VariantType::STRING, "package/unique_name"});
        options.push_back({VariantType::STRING, "package/name"});
        options.push_back({VariantType::BOOL, "package/signed"});
        options.push_back({VariantType::STRING, "package/keystore/debug"});
        options.push_back({VariantType::STRING, "package/keystore/release"});
        options.push_back({VariantType::STRING, "package/keystore/release_user"});
        options.push_back({VariantType::STRING, "package/keystore/release_password"});
        options.push_back({VariantType::BOOL, "screen/immersive_mode"});
        options.push_back({VariantType::INT, "version/code"});
        options.push_back({VariantType::STRING, "version/name"});
        options.push_back({VariantType::BOOL, "gradle_build/use_gradle"});
        options.push_back({VariantType::INT, "gradle_build/min_sdk"});
        options.push_back({VariantType::INT, "gradle_build/target_sdk"});
        options.push_back({VariantType::STRING, "architectures/armeabi-v7a"});
        options.push_back({VariantType::STRING, "architectures/arm64-v8a"});
        options.push_back({VariantType::STRING, "architectures/x86"});
        options.push_back({VariantType::STRING, "architectures/x86_64"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("android");
        features.push_back("mobile");
        if (preset->get_option("architectures/arm64-v8a", true).as<bool>()) features.push_back("arm64");
        if (preset->get_option("architectures/armeabi-v7a", false).as<bool>()) features.push_back("armv7");
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        String build_dir = OS::get_singleton()->get_user_data_dir() + "/temp/android_export";
        DirAccess::make_dir_recursive(build_dir);
        
        // Prepare Gradle project
        prepare_gradle_project(build_dir, preset, debug);
        
        // Build PCK
        String pck_path = build_dir + "/assets/godot.pck";
        export_pack(preset, debug, pck_path, flags);
        
        // Run Gradle build
        String cmd = "cd " + build_dir + " && ./gradlew ";
        cmd += debug ? "assembleDebug" : "assembleRelease";
        
        int result = system(cmd.utf8());
        if (result != 0) {
            return ERR_COMPILATION_FAILED;
        }
        
        // Copy output
        String apk_path = build_dir + "/app/build/outputs/apk/" + (debug ? "debug" : "release") + "/app-" + (debug ? "debug" : "release") + ".apk";
        if (path.ends_with(".aab")) {
            String aab_path = build_dir + "/app/build/outputs/bundle/" + (debug ? "debug" : "release") + "/app-" + (debug ? "debug" : "release") + ".aab";
            DirAccess::copy(aab_path, path);
        } else {
            DirAccess::copy(apk_path, path);
        }
        
        DirAccess::remove(build_dir);
        return OK;
    }

private:
    void prepare_gradle_project(const String& dir, const Ref<EditorExportPreset>& preset, bool debug) {
        String package_name = preset->get_option("package/unique_name", "com.example.game").as<String>();
        String app_name = preset->get_option("package/name", "GodotApp").as<String>();
        int version_code = preset->get_option("version/code", 1).as<int>();
        String version_name = preset->get_option("version/name", "1.0").as<String>();
        int min_sdk = preset->get_option("gradle_build/min_sdk", 21).as<int>();
        int target_sdk = preset->get_option("gradle_build/target_sdk", 34).as<int>();
        
        // Create Gradle files
        String build_gradle = generate_build_gradle(package_name, version_code, version_name, min_sdk, target_sdk);
        Ref<FileAccess> f = FileAccess::open(dir + "/app/build.gradle", FileAccess::WRITE);
        if (f.is_valid()) f->store_string(build_gradle);
        
        // Create manifest
        String manifest = generate_manifest(package_name, app_name, preset);
        f = FileAccess::open(dir + "/app/src/main/AndroidManifest.xml", FileAccess::WRITE);
        if (f.is_valid()) f->store_string(manifest);
        
        // Copy template
        String template_path = get_template_path(preset, debug);
        // Extract template...
    }

    String get_template_path(const Ref<EditorExportPreset>& preset, bool debug) {
        String template_key = debug ? "custom_template/debug" : "custom_template/release";
        return preset->get_option(template_key).as<String>();
    }

    String generate_build_gradle(const String& package, int version_code, const String& version_name, int min_sdk, int target_sdk) {
        return R"(
plugins {
    id 'com.android.application'
}

android {
    namespace ')" + package + R"('
    compileSdk )" + String::num(target_sdk) + R"(
    
    defaultConfig {
        applicationId ')" + package + R"('
        minSdk )" + String::num(min_sdk) + R"(
        targetSdk )" + String::num(target_sdk) + R"(
        versionCode )" + String::num(version_code) + R"(
        versionName ')" + version_name + R"('
    }
    
    buildTypes {
        release {
            minifyEnabled false
        }
    }
    
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
}

dependencies {
    implementation 'androidx.appcompat:appcompat:1.6.1'
}
)";
    }

    String generate_manifest(const String& package, const String& app_name, const Ref<EditorExportPreset>& preset) {
        bool immersive = preset->get_option("screen/immersive_mode", false).as<bool>();
        return R"(<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package=")" + package + R"(">
    
    <uses-feature android:glEsVersion="0x00030000" android:required="true" />
    <uses-feature android:name="android.hardware.touchscreen" android:required="false" />
    
    <application
        android:label=")" + app_name + R"("
        android:icon="@mipmap/ic_launcher"
        android:allowBackup="true"
        android:hardwareAccelerated="true">
        
        <activity
            android:name=".GodotApp"
            android:configChanges="orientation|keyboardHidden|screenSize"
            android:launchMode="singleTask"
            android:exported="true")";
        if (immersive) result += R"(
            android:theme="@android:style/Theme.NoTitleBar.Fullscreen")";
        result += R"(>
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>)";
        return result;
    }
};

// #############################################################################
// IOSExporter - iOS platform exporter
// #############################################################################
class IOSExporter : public EditorExportPlatform {
    XTU_GODOT_REGISTER_CLASS(IOSExporter, EditorExportPlatform)

private:
    std::vector<String> m_binary_extensions = {"ipa", "app"};

public:
    static StringName get_class_static() { return StringName("IOSExporter"); }

    IOSExporter() {
        m_name = "iOS";
        m_os_name = "iOS";
    }

    std::vector<String> get_binary_extensions() const override { return m_binary_extensions; }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back({VariantType::STRING, "custom_template/debug"});
        options.push_back({VariantType::STRING, "custom_template/release"});
        options.push_back({VariantType::STRING, "application/app_store_team_id"});
        options.push_back({VariantType::STRING, "application/provisioning_profile_uuid_debug"});
        options.push_back({VariantType::STRING, "application/provisioning_profile_uuid_release"});
        options.push_back({VariantType::STRING, "application/bundle_identifier"});
        options.push_back({VariantType::STRING, "application/short_version"});
        options.push_back({VariantType::STRING, "application/version"});
        options.push_back({VariantType::STRING, "application/copyright"});
        options.push_back({VariantType::BOOL, "capabilities/access_wifi"});
        options.push_back({VariantType::BOOL, "capabilities/arkit"});
        options.push_back({VariantType::BOOL, "capabilities/camera"});
        options.push_back({VariantType::BOOL, "capabilities/microphone"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("ios");
        features.push_back("mobile");
        features.push_back("arm64");
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        // iOS export requires Xcode project generation
        String build_dir = OS::get_singleton()->get_user_data_dir() + "/temp/ios_export";
        DirAccess::make_dir_recursive(build_dir);
        
        String project_name = preset->get_option("application/name", "GodotApp").as<String>();
        String bundle_id = preset->get_option("application/bundle_identifier", "com.example.game").as<String>();
        
        // Generate Xcode project
        generate_xcode_project(build_dir, project_name, bundle_id, preset);
        
        // Copy template
        String template_path = get_template_path(preset, debug);
        // Extract template...
        
        // Build PCK
        String pck_path = build_dir + "/" + project_name + ".pck";
        export_pack(preset, debug, pck_path, flags);
        
        // Archive and export
        if (path.ends_with(".ipa")) {
            build_ipa(build_dir, project_name, path, preset);
        }
        
        DirAccess::remove(build_dir);
        return OK;
    }

private:
    String get_template_path(const Ref<EditorExportPreset>& preset, bool debug) {
        String template_key = debug ? "custom_template/debug" : "custom_template/release";
        return preset->get_option(template_key).as<String>();
    }

    void generate_xcode_project(const String& dir, const String& name, const String& bundle_id, const Ref<EditorExportPreset>& preset) {
        // Generate .xcodeproj
    }

    void build_ipa(const String& dir, const String& name, const String& ipa_path, const Ref<EditorExportPreset>& preset) {
        // xcodebuild archive and export
    }
};

// #############################################################################
// WebExporter - Web platform exporter
// #############################################################################
class WebExporter : public EditorExportPlatform {
    XTU_GODOT_REGISTER_CLASS(WebExporter, EditorExportPlatform)

private:
    std::vector<String> m_binary_extensions = {"html"};

public:
    static StringName get_class_static() { return StringName("WebExporter"); }

    WebExporter() {
        m_name = "Web";
        m_os_name = "Web";
    }

    std::vector<String> get_binary_extensions() const override { return m_binary_extensions; }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back({VariantType::STRING, "custom_template/debug"});
        options.push_back({VariantType::STRING, "custom_template/release"});
        options.push_back({VariantType::STRING, "html/head_include"});
        options.push_back({VariantType::STRING, "html/canvas_resize_policy", PropertyHint::ENUM, "None,Project,Adaptive"});
        options.push_back({VariantType::BOOL, "html/focus_canvas_on_start"});
        options.push_back({VariantType::BOOL, "html/experimental_virtual_keyboard"});
        options.push_back({VariantType::BOOL, "progressive_web_app/enabled"});
        options.push_back({VariantType::STRING, "progressive_web_app/offline_page"});
        options.push_back({VariantType::STRING, "progressive_web_app/display", PropertyHint::ENUM, "Fullscreen,Standalone,Minimal-UI,Browser"});
        options.push_back({VariantType::STRING, "progressive_web_app/orientation", PropertyHint::ENUM, "Any,Landscape,Portrait"});
        options.push_back({VariantType::STRING, "progressive_web_app/background_color"});
        options.push_back({VariantType::STRING, "progressive_web_app/theme_color"});
        options.push_back({VariantType::STRING, "progressive_web_app/icon_144x144"});
        options.push_back({VariantType::STRING, "progressive_web_app/icon_180x180"});
        options.push_back({VariantType::STRING, "progressive_web_app/icon_512x512"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("web");
        features.push_back("wasm");
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        String export_dir = path.get_base_dir();
        String base_name = path.get_basename().get_basename();
        
        // Copy template files
        String template_path = get_template_path(preset, debug);
        if (template_path.empty()) return ERR_FILE_NOT_FOUND;
        
        // Extract template .zip
        // For now, copy essential files
        
        // Build PCK
        String pck_path = export_dir + "/" + base_name + ".pck";
        export_pack(preset, debug, pck_path, flags);
        
        // Generate HTML
        String html = generate_html(preset, base_name);
        Ref<FileAccess> html_file = FileAccess::open(path, FileAccess::WRITE);
        if (html_file.is_valid()) html_file->store_string(html);
        
        // Generate PWA manifest if enabled
        if (preset->get_option("progressive_web_app/enabled", false).as<bool>()) {
            String manifest = generate_manifest(preset, base_name);
            Ref<FileAccess> manifest_file = FileAccess::open(export_dir + "/manifest.json", FileAccess::WRITE);
            if (manifest_file.is_valid()) manifest_file->store_string(manifest);
        }
        
        return OK;
    }

private:
    String get_template_path(const Ref<EditorExportPreset>& preset, bool debug) {
        String template_key = debug ? "custom_template/debug" : "custom_template/release";
        return preset->get_option(template_key).as<String>();
    }

    String generate_html(const Ref<EditorExportPreset>& preset, const String& base_name) {
        String head_include = preset->get_option("html/head_include").as<String>();
        String canvas_resize = preset->get_option("html/canvas_resize_policy", "Adaptive").as<String>();
        bool focus_on_start = preset->get_option("html/focus_canvas_on_start", true).as<bool>();
        
        String html = R"(<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Godot Web Export</title>
    )" + head_include + R"(
    <style>
        body { margin: 0; padding: 0; background: #000; }
        canvas { display: block; margin: auto; }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <script>
        const canvas = document.getElementById('canvas');
        const engine = new Engine();
        engine.init(')" + base_name + R"(.pck').then(() => {
            engine.start({ canvas: canvas });
    })" + (focus_on_start ? ".then(() => { canvas.focus(); })" : "") + R"(;
    </script>
</body>
</html>)";
        return html;
    }

    String generate_manifest(const Ref<EditorExportPreset>& preset, const String& base_name) {
        String name = preset->get_option("application/name", "GodotApp").as<String>();
        String short_name = name;
        String display = preset->get_option("progressive_web_app/display", "Standalone").as<String>();
        String orientation = preset->get_option("progressive_web_app/orientation", "Any").as<String>();
        String bg_color = preset->get_option("progressive_web_app/background_color", "#000000").as<String>();
        String theme_color = preset->get_option("progressive_web_app/theme_color", "#ffffff").as<String>();
        
        io::json::JsonValue json;
        json["name"] = io::json::JsonValue(name.to_std_string());
        json["short_name"] = io::json::JsonValue(short_name.to_std_string());
        json["display"] = io::json::JsonValue(display.to_lower().to_std_string());
        json["orientation"] = io::json::JsonValue(orientation.to_lower().to_std_string());
        json["background_color"] = io::json::JsonValue(bg_color.to_std_string());
        json["theme_color"] = io::json::JsonValue(theme_color.to_std_string());
        json["start_url"] = io::json::JsonValue("./" + base_name.to_std_string() + ".html");
        json["scope"] = io::json::JsonValue("./");
        
        return json.dump(2).c_str();
    }
};

// #############################################################################
// ExportManager - Central export registration
// #############################################################################
class ExportManager : public Object {
    XTU_GODOT_REGISTER_CLASS(ExportManager, Object)

private:
    static ExportManager* s_singleton;
    Ref<ExportReport> m_current_report;

public:
    static ExportManager* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("ExportManager"); }

    ExportManager() {
        s_singleton = this;
        register_default_exporters();
    }

    ~ExportManager() { s_singleton = nullptr; }

    void register_default_exporters() {
        EditorExport::get_singleton()->add_export_platform(Ref<EditorExportPlatform>(new WindowsExporter()));
        EditorExport::get_singleton()->add_export_platform(Ref<EditorExportPlatform>(new LinuxExporter()));
        EditorExport::get_singleton()->add_export_platform(Ref<EditorExportPlatform>(new MacOSExporter()));
        EditorExport::get_singleton()->add_export_platform(Ref<EditorExportPlatform>(new AndroidExporter()));
        EditorExport::get_singleton()->add_export_platform(Ref<EditorExportPlatform>(new IOSExporter()));
        EditorExport::get_singleton()->add_export_platform(Ref<EditorExportPlatform>(new WebExporter()));
    }

    Error export_all_presets(bool debug = false) {
        m_current_report.instance();
        m_current_report->clear();
        
        auto presets = EditorExport::get_singleton()->get_export_presets();
        std::atomic<int> completed{0};
        std::mutex error_mutex;
        Error final_error = OK;
        
        parallel::parallel_for(0, presets.size(), [&](size_t i) {
            const auto& preset = presets[i];
            String path = preset->get_export_path();
            if (path.empty()) {
                path = "build/" + preset->get_name() + (debug ? "_debug" : "_release");
            }
            
            Error err = EditorExport::get_singleton()->export_preset(preset, path, debug);
            
            if (err != OK) {
                std::lock_guard<std::mutex> lock(error_mutex);
                final_error = err;
            }
            
            ++completed;
            emit_signal("progress", static_cast<int>(completed), static_cast<int>(presets.size()));
        });
        
        emit_signal("export_completed", final_error == OK);
        return final_error;
    }

    Ref<ExportReport> get_current_report() const { return m_current_report; }
};

} // namespace editor

// Bring into main namespace
using editor::WindowsExporter;
using editor::LinuxExporter;
using editor::MacOSExporter;
using editor::AndroidExporter;
using editor::IOSExporter;
using editor::WebExporter;
using editor::ExportManager;
using editor::ExportReport;
using editor::ExportReportStatus;
using editor::ExportConfig;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XEXPORTER_HPP