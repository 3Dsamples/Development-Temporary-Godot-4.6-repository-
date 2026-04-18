// include/xtu/godot/xeditor_import.hpp
// xtensor-unified - Editor asset import system for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XEDITOR_IMPORT_HPP
#define XTU_GODOT_XEDITOR_IMPORT_HPP

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
#include "xtu/godot/xeditor.hpp"
#include "xtu/io/xio_json.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class EditorImportPlugin;
class ResourceImporter;
class ResourceImporterTexture;
class ResourceImporterScene;
class ResourceImporterAudio;
class ResourceImporterFont;
class ResourceImporterCSV;

// #############################################################################
// Import preset options
// #############################################################################
enum class ImportPreset : uint8_t {
    PRESET_2D = 0,
    PRESET_3D = 1,
    PRESET_UI = 2,
    PRESET_CUSTOM = 3
};

// #############################################################################
// EditorImportPlugin - Base class for custom importers
// #############################################################################
class EditorImportPlugin : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(EditorImportPlugin, RefCounted)

public:
    static StringName get_class_static() { return StringName("EditorImportPlugin"); }

    virtual String get_importer_name() const = 0;
    virtual String get_visible_name() const = 0;
    virtual std::vector<String> get_recognized_extensions() const = 0;
    virtual String get_save_extension() const = 0;
    virtual String get_resource_type() const = 0;
    virtual float get_priority() const { return 1.0f; }
    virtual int get_import_order() const { return 0; }
    virtual int get_preset_count() const { return 0; }
    virtual String get_preset_name(int idx) const { return String(); }
    virtual std::vector<PropertyInfo> get_import_options(const String& path, int preset) const = 0;
    virtual bool get_option_visibility(const String& path, const String& option, const std::map<String, Variant>& options) const { return true; }
    virtual Error import(const String& source_file, const String& save_path,
                         const std::map<String, Variant>& options,
                         std::vector<String>& r_platform_variants,
                         std::vector<String>& r_gen_files) = 0;
};

// #############################################################################
// ResourceImporter - Base importer class
// #############################################################################
class ResourceImporter : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(ResourceImporter, RefCounted)

protected:
    String m_importer_name;
    String m_visible_name;
    std::vector<String> m_extensions;
    float m_priority = 1.0f;
    int m_import_order = 0;

public:
    static StringName get_class_static() { return StringName("ResourceImporter"); }

    virtual String get_importer_name() const { return m_importer_name; }
    virtual String get_visible_name() const { return m_visible_name; }
    virtual std::vector<String> get_recognized_extensions() const { return m_extensions; }
    virtual String get_save_extension() const = 0;
    virtual String get_resource_type() const = 0;
    virtual float get_priority() const { return m_priority; }
    virtual int get_import_order() const { return m_import_order; }
    virtual int get_preset_count() const { return 0; }
    virtual String get_preset_name(int idx) const { return String(); }
    virtual std::vector<PropertyInfo> get_import_options(const String& path, int preset) const = 0;
    virtual bool get_option_visibility(const String& path, const String& option, const std::map<String, Variant>& options) const { return true; }
    virtual Error import(const String& source_file, const String& save_path,
                         const std::map<String, Variant>& options,
                         std::vector<String>& r_platform_variants,
                         std::vector<String>& r_gen_files) = 0;
};

// #############################################################################
// ResourceImporterTexture - Texture importer
// #############################################################################
class ResourceImporterTexture : public ResourceImporter {
    XTU_GODOT_REGISTER_CLASS(ResourceImporterTexture, ResourceImporter)

public:
    enum CompressMode : uint8_t {
        COMPRESS_LOSSLESS = 0,
        COMPRESS_LOSSY = 1,
        COMPRESS_VRAM_COMPRESSED = 2,
        COMPRESS_VRAM_UNCOMPRESSED = 3,
        COMPRESS_BASIS_UNIVERSAL = 4
    };

    static StringName get_class_static() { return StringName("ResourceImporterTexture"); }

    ResourceImporterTexture() {
        m_importer_name = "texture";
        m_visible_name = "Texture";
        m_extensions = {"png", "jpg", "jpeg", "bmp", "tga", "webp", "hdr", "exr"};
    }

    String get_save_extension() const override { return "tex"; }
    String get_resource_type() const override { return "CompressedTexture2D"; }

    std::vector<PropertyInfo> get_import_options(const String& path, int preset) const override {
        std::vector<PropertyInfo> options;
        options.push_back(PropertyInfo{VariantType::INT, "compress/mode"});
        options.push_back(PropertyInfo{VariantType::BOOL, "mipmaps/generate"});
        options.push_back(PropertyInfo{VariantType::INT, "mipmaps/limit"});
        options.push_back(PropertyInfo{VariantType::BOOL, "process/fix_alpha_border"});
        options.push_back(PropertyInfo{VariantType::BOOL, "process/premult_alpha"});
        options.push_back(PropertyInfo{VariantType::BOOL, "process/HDR_as_SRGB"});
        options.push_back(PropertyInfo{VariantType::FLOAT, "process/scale"});
        options.push_back(PropertyInfo{VariantType::INT, "detect_3d/compress_to"});
        return options;
    }

    Error import(const String& source_file, const String& save_path,
                 const std::map<String, Variant>& options,
                 std::vector<String>& r_platform_variants,
                 std::vector<String>& r_gen_files) override {
        // Implementation using Xtensor image processing
        return OK;
    }
};

// #############################################################################
// ResourceImporterScene - 3D scene importer (glTF, FBX, etc.)
// #############################################################################
class ResourceImporterScene : public ResourceImporter {
    XTU_GODOT_REGISTER_CLASS(ResourceImporterScene, ResourceImporter)

public:
    static StringName get_class_static() { return StringName("ResourceImporterScene"); }

    ResourceImporterScene() {
        m_importer_name = "scene";
        m_visible_name = "3D Scene";
        m_extensions = {"gltf", "glb", "fbx", "dae", "obj", "blend"};
    }

    String get_save_extension() const override { return "scn"; }
    String get_resource_type() const override { return "PackedScene"; }

    std::vector<PropertyInfo> get_import_options(const String& path, int preset) const override {
        std::vector<PropertyInfo> options;
        options.push_back(PropertyInfo{VariantType::FLOAT, "meshes/lightmap_texel_size"});
        options.push_back(PropertyInfo{VariantType::BOOL, "meshes/generate_lods"});
        options.push_back(PropertyInfo{VariantType::BOOL, "meshes/create_shadow_meshes"});
        options.push_back(PropertyInfo{VariantType::BOOL, "meshes/use_compression"});
        options.push_back(PropertyInfo{VariantType::INT, "skins/use_named_skins"});
        options.push_back(PropertyInfo{VariantType::FLOAT, "animation/fps"});
        options.push_back(PropertyInfo{VariantType::BOOL, "animation/trimming"});
        options.push_back(PropertyInfo{VariantType::BOOL, "animation/remove_immutable_tracks"});
        return options;
    }

    Error import(const String& source_file, const String& save_path,
                 const std::map<String, Variant>& options,
                 std::vector<String>& r_platform_variants,
                 std::vector<String>& r_gen_files) override {
        return OK;
    }
};

// #############################################################################
// ResourceImporterAudio - Audio importer
// #############################################################################
class ResourceImporterAudio : public ResourceImporter {
    XTU_GODOT_REGISTER_CLASS(ResourceImporterAudio, ResourceImporter)

public:
    static StringName get_class_static() { return StringName("ResourceImporterAudio"); }

    ResourceImporterAudio() {
        m_importer_name = "audio";
        m_visible_name = "Audio";
        m_extensions = {"wav", "mp3", "ogg", "flac", "aiff", "m4a"};
    }

    String get_save_extension() const override { return "sample"; }
    String get_resource_type() const override { return "AudioStream"; }

    std::vector<PropertyInfo> get_import_options(const String& path, int preset) const override {
        std::vector<PropertyInfo> options;
        options.push_back(PropertyInfo{VariantType::BOOL, "force/mono"});
        options.push_back(PropertyInfo{VariantType::FLOAT, "force/max_rate"});
        options.push_back(PropertyInfo{VariantType::INT, "compress/mode"});
        options.push_back(PropertyInfo{VariantType::FLOAT, "compress/bitrate"});
        options.push_back(PropertyInfo{VariantType::BOOL, "edit/trim"});
        options.push_back(PropertyInfo{VariantType::BOOL, "edit/loop"});
        return options;
    }

    Error import(const String& source_file, const String& save_path,
                 const std::map<String, Variant>& options,
                 std::vector<String>& r_platform_variants,
                 std::vector<String>& r_gen_files) override {
        return OK;
    }
};

// #############################################################################
// EditorFileSystemImport - Import manager
// #############################################################################
class EditorFileSystemImport : public Object {
    XTU_GODOT_REGISTER_CLASS(EditorFileSystemImport, Object)

private:
    static EditorFileSystemImport* s_singleton;
    std::vector<Ref<ResourceImporter>> m_importers;
    std::unordered_map<String, Ref<ResourceImporter>> m_importer_by_extension;
    std::mutex m_mutex;

public:
    static EditorFileSystemImport* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("EditorFileSystemImport"); }

    EditorFileSystemImport() {
        s_singleton = this;
        register_default_importers();
    }

    ~EditorFileSystemImport() { s_singleton = nullptr; }

    void add_importer(const Ref<ResourceImporter>& importer) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_importers.push_back(importer);
        for (const auto& ext : importer->get_recognized_extensions()) {
            m_importer_by_extension[ext] = importer;
        }
    }

    void remove_importer(const Ref<ResourceImporter>& importer) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = std::find(m_importers.begin(), m_importers.end(), importer);
        if (it != m_importers.end()) m_importers.erase(it);
        for (auto& kv : m_importer_by_extension) {
            if (kv.second == importer) kv.second = Ref<ResourceImporter>();
        }
    }

    Ref<ResourceImporter> get_importer_by_extension(const String& extension) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_importer_by_extension.find(extension);
        return it != m_importer_by_extension.end() ? it->second : Ref<ResourceImporter>();
    }

    Ref<ResourceImporter> get_importer_by_path(const String& path) const {
        String ext = path.get_extension();
        return get_importer_by_extension(ext);
    }

    void import_file(const String& path) {
        Ref<ResourceImporter> importer = get_importer_by_path(path);
        if (!importer.is_valid()) return;

        std::map<String, Variant> options = get_default_options(importer, path);
        std::vector<String> platform_variants;
        std::vector<String> gen_files;
        String save_path = path.get_basename() + "." + importer->get_save_extension();

        importer->import(path, save_path, options, platform_variants, gen_files);
    }

    void reimport_files(const std::vector<String>& paths) {
        parallel::parallel_for(0, paths.size(), [&](size_t i) {
            import_file(paths[i]);
        });
    }

private:
    void register_default_importers() {
        add_importer(Ref<ResourceImporter>(new ResourceImporterTexture()));
        add_importer(Ref<ResourceImporter>(new ResourceImporterScene()));
        add_importer(Ref<ResourceImporter>(new ResourceImporterAudio()));
    }

    std::map<String, Variant> get_default_options(const Ref<ResourceImporter>& importer, const String& path) const {
        std::map<String, Variant> options;
        auto props = importer->get_import_options(path, 0);
        for (const auto& prop : props) {
            // Set default values based on property hints
        }
        return options;
    }
};

} // namespace editor

using editor::EditorImportPlugin;
using editor::ResourceImporter;
using editor::ResourceImporterTexture;
using editor::ResourceImporterScene;
using editor::ResourceImporterAudio;
using editor::EditorFileSystemImport;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XEDITOR_IMPORT_HPP