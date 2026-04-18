// include/xtu/godot/xresource_importer.hpp
// xtensor-unified - Resource importers for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XRESOURCE_IMPORTER_HPP
#define XTU_GODOT_XRESOURCE_IMPORTER_HPP

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
#include "xtu/godot/xeditor_import.hpp"
#include "xtu/godot/xaudioserver.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/godot/xtext_server.hpp"
#include "xtu/godot/xtexture_compressor.hpp"
#include "xtu/graphics/xmesh.hpp"
#include "xtu/io/xio_json.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class ResourceImporterWAV;
class ResourceImporterOBJ;
class ResourceImporterImage;
class ResourceImporterFont;
class ResourceImporterCSV;
class ResourceImporterShader;

// #############################################################################
// Import priority levels
// #############################################################################
enum class ImportPriority : uint8_t {
    PRIORITY_LOWEST = 0,
    PRIORITY_LOW = 25,
    PRIORITY_NORMAL = 50,
    PRIORITY_HIGH = 75,
    PRIORITY_HIGHEST = 100
};

// #############################################################################
// Image import format
// #############################################################################
enum class ImageImportFormat : uint8_t {
    FORMAT_KEEP = 0,
    FORMAT_PNG = 1,
    FORMAT_WEBP = 2,
    FORMAT_BASIS = 3,
    FORMAT_ETC2 = 4,
    FORMAT_ASTC = 5
};

// #############################################################################
// Audio import format
// #############################################################################
enum class AudioImportFormat : uint8_t {
    FORMAT_KEEP = 0,
    FORMAT_OGG = 1,
    FORMAT_MP3 = 2,
    FORMAT_WAV = 3
};

// #############################################################################
// ResourceImporterWAV - WAV audio importer
// #############################################################################
class ResourceImporterWAV : public ResourceImporter {
    XTU_GODOT_REGISTER_CLASS(ResourceImporterWAV, ResourceImporter)

private:
    struct WAVHeader {
        char chunk_id[4];
        uint32_t chunk_size;
        char format[4];
        char subchunk1_id[4];
        uint32_t subchunk1_size;
        uint16_t audio_format;
        uint16_t num_channels;
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_sample;
        char subchunk2_id[4];
        uint32_t subchunk2_size;
    };

public:
    static StringName get_class_static() { return StringName("ResourceImporterWAV"); }

    ResourceImporterWAV() {
        m_importer_name = "wav";
        m_visible_name = "WAV Audio";
        m_extensions = {"wav"};
        m_priority = static_cast<float>(ImportPriority::PRIORITY_NORMAL);
    }

    String get_save_extension() const override { return "sample"; }
    String get_resource_type() const override { return "AudioStreamWAV"; }

    std::vector<PropertyInfo> get_import_options(const String& path, int preset) const override {
        std::vector<PropertyInfo> options;
        options.push_back({VariantType::BOOL, "force/mono"});
        options.push_back({VariantType::FLOAT, "force/max_rate", PropertyHint::NONE, "", 0, 192000});
        options.push_back({VariantType::INT, "compress/mode", PropertyHint::ENUM, "Disabled,OGG,MP3"});
        options.push_back({VariantType::INT, "compress/bitrate", PropertyHint::ENUM, "64,96,128,192,256,320"});
        options.push_back({VariantType::BOOL, "edit/loop"});
        options.push_back({VariantType::FLOAT, "edit/loop_offset"});
        return options;
    }

    Error import(const String& source_file, const String& save_path,
                 const std::map<String, Variant>& options,
                 std::vector<String>& r_platform_variants,
                 std::vector<String>& r_gen_files) override {
        Ref<FileAccess> file = FileAccess::open(source_file, FileAccess::READ);
        if (!file.is_valid()) return ERR_FILE_CANT_OPEN;

        WAVHeader header;
        file->get_buffer(reinterpret_cast<uint8_t*>(&header), sizeof(WAVHeader));

        if (std::memcmp(header.chunk_id, "RIFF", 4) != 0 ||
            std::memcmp(header.format, "WAVE", 4) != 0) {
            return ERR_FILE_UNRECOGNIZED;
        }

        bool force_mono = options.count("force/mono") ? options.at("force/mono").as<bool>() : false;
        bool loop = options.count("edit/loop") ? options.at("edit/loop").as<bool>() : false;
        float loop_offset = options.count("edit/loop_offset") ? options.at("edit/loop_offset").as<float>() : 0.0f;

        size_t data_size = header.subchunk2_size;
        std::vector<uint8_t> audio_data = file->get_buffer(data_size);

        // Convert to AudioStreamWAV format
        std::vector<uint8_t> sample_data;
        if (header.bits_per_sample == 16) {
            sample_data = convert_pcm16_to_float(audio_data, header.num_channels, force_mono);
        } else if (header.bits_per_sample == 8) {
            sample_data = convert_pcm8_to_float(audio_data, header.num_channels, force_mono);
        } else if (header.bits_per_sample == 32) {
            // 32-bit float already
            sample_data = audio_data;
        }

        // Save as resource
        io::json::JsonValue res_json;
        res_json["sample_rate"] = io::json::JsonValue(static_cast<double>(header.sample_rate));
        res_json["channels"] = io::json::JsonValue(static_cast<double>(force_mono ? 1 : header.num_channels));
        res_json["loop"] = io::json::JsonValue(loop);
        res_json["loop_offset"] = io::json::JsonValue(static_cast<double>(loop_offset));
        res_json["data"] = io::json::JsonValue(base64_encode(sample_data));

        String res_path = save_path + ".sample";
        Ref<FileAccess> out = FileAccess::open(res_path, FileAccess::WRITE);
        if (!out.is_valid()) return ERR_FILE_CANT_WRITE;
        out->store_string(res_json.dump());
        
        r_gen_files.push_back(res_path);
        return OK;
    }

private:
    std::vector<uint8_t> convert_pcm16_to_float(const std::vector<uint8_t>& data, int channels, bool force_mono) {
        size_t sample_count = data.size() / 2;
        const int16_t* samples = reinterpret_cast<const int16_t*>(data.data());
        
        int out_channels = force_mono ? 1 : channels;
        std::vector<float> float_samples(sample_count / channels * out_channels);
        
        if (force_mono) {
            for (size_t i = 0; i < sample_count / channels; ++i) {
                float sum = 0.0f;
                for (int c = 0; c < channels; ++c) {
                    sum += static_cast<float>(samples[i * channels + c]) / 32768.0f;
                }
                float_samples[i] = sum / channels;
            }
        } else {
            for (size_t i = 0; i < sample_count; ++i) {
                float_samples[i] = static_cast<float>(samples[i]) / 32768.0f;
            }
        }
        
        std::vector<uint8_t> result(float_samples.size() * 4);
        std::memcpy(result.data(), float_samples.data(), result.size());
        return result;
    }

    std::vector<uint8_t> convert_pcm8_to_float(const std::vector<uint8_t>& data, int channels, bool force_mono) {
        // Similar conversion for 8-bit PCM
        return {};
    }

    static String base64_encode(const std::vector<uint8_t>& data) {
        return Crypto::base64_encode(data);
    }
};

// #############################################################################
// ResourceImporterOBJ - OBJ/MTL mesh importer
// #############################################################################
class ResourceImporterOBJ : public ResourceImporter {
    XTU_GODOT_REGISTER_CLASS(ResourceImporterOBJ, ResourceImporter)

public:
    static StringName get_class_static() { return StringName("ResourceImporterOBJ"); }

    ResourceImporterOBJ() {
        m_importer_name = "obj";
        m_visible_name = "OBJ Mesh";
        m_extensions = {"obj"};
        m_priority = static_cast<float>(ImportPriority::PRIORITY_NORMAL);
    }

    String get_save_extension() const override { return "mesh"; }
    String get_resource_type() const override { return "ArrayMesh"; }

    std::vector<PropertyInfo> get_import_options(const String& path, int preset) const override {
        std::vector<PropertyInfo> options;
        options.push_back({VariantType::BOOL, "meshes/generate_lods"});
        options.push_back({VariantType::BOOL, "meshes/create_shadow_meshes"});
        options.push_back({VariantType::BOOL, "meshes/use_compression"});
        options.push_back({VariantType::FLOAT, "meshes/scale"});
        options.push_back({VariantType::VECTOR3, "meshes/offset"});
        options.push_back({VariantType::BOOL, "materials/import"});
        options.push_back({VariantType::STRING, "materials/location", PropertyHint::ENUM, "Separate,Node,File"});
        return options;
    }

    Error import(const String& source_file, const String& save_path,
                 const std::map<String, Variant>& options,
                 std::vector<String>& r_platform_variants,
                 std::vector<String>& r_gen_files) override {
        Ref<FileAccess> file = FileAccess::open(source_file, FileAccess::READ);
        if (!file.is_valid()) return ERR_FILE_CANT_OPEN;

        std::vector<vec3f> vertices;
        std::vector<vec3f> normals;
        std::vector<vec2f> uvs;
        std::vector<uint32_t> indices;
        std::vector<uint32_t> material_indices;
        std::unordered_map<String, uint32_t> material_map;
        
        float scale = options.count("meshes/scale") ? options.at("meshes/scale").as<float>() : 1.0f;
        vec3f offset = options.count("meshes/offset") ? options.at("meshes/offset").as<vec3f>() : vec3f(0);

        String line;
        uint32_t current_material = 0;
        std::unordered_map<String, uint32_t> vertex_cache;
        std::vector<vec3f> temp_positions;
        std::vector<vec3f> temp_normals;
        std::vector<vec2f> temp_uvs;

        while (!(line = file->get_line()).empty()) {
            if (line.begins_with("v ")) {
                auto parts = line.split(" ");
                if (parts.size() >= 4) {
                    vec3f v(parts[1].to_float() * scale + offset.x(),
                            parts[2].to_float() * scale + offset.y(),
                            parts[3].to_float() * scale + offset.z());
                    temp_positions.push_back(v);
                }
            } else if (line.begins_with("vn ")) {
                auto parts = line.split(" ");
                if (parts.size() >= 4) {
                    vec3f n(parts[1].to_float(), parts[2].to_float(), parts[3].to_float());
                    temp_normals.push_back(n);
                }
            } else if (line.begins_with("vt ")) {
                auto parts = line.split(" ");
                if (parts.size() >= 3) {
                    vec2f uv(parts[1].to_float(), 1.0f - parts[2].to_float());
                    temp_uvs.push_back(uv);
                }
            } else if (line.begins_with("f ")) {
                auto parts = line.split(" ");
                std::vector<uint32_t> face_indices;
                for (size_t i = 1; i < parts.size(); ++i) {
                    String key = parts[i];
                    auto it = vertex_cache.find(key);
                    if (it != vertex_cache.end()) {
                        face_indices.push_back(it->second);
                        continue;
                    }
                    
                    auto sub = parts[i].split("/");
                    int pos_idx = sub[0].to_int() - 1;
                    int uv_idx = sub.size() > 1 && !sub[1].empty() ? sub[1].to_int() - 1 : -1;
                    int norm_idx = sub.size() > 2 && !sub[2].empty() ? sub[2].to_int() - 1 : -1;
                    
                    vertices.push_back(temp_positions[pos_idx]);
                    uvs.push_back(uv_idx >= 0 ? temp_uvs[uv_idx] : vec2f(0, 0));
                    normals.push_back(norm_idx >= 0 ? temp_normals[norm_idx] : vec3f(0, 1, 0));
                    
                    uint32_t idx = static_cast<uint32_t>(vertices.size() - 1);
                    vertex_cache[key] = idx;
                    face_indices.push_back(idx);
                }
                
                for (size_t i = 1; i < face_indices.size() - 1; ++i) {
                    indices.push_back(face_indices[0]);
                    indices.push_back(face_indices[i]);
                    indices.push_back(face_indices[i + 1]);
                }
            } else if (line.begins_with("usemtl ")) {
                String mtl = line.substr(7).strip_edges();
                auto it = material_map.find(mtl);
                if (it != material_map.end()) {
                    current_material = it->second;
                } else {
                    current_material = static_cast<uint32_t>(material_map.size());
                    material_map[mtl] = current_material;
                }
            }
        }

        // Build mesh data
        Ref<ArrayMesh> mesh;
        mesh.instance();
        
        Array arrays;
        arrays.resize(Mesh::ARRAY_MAX);
        arrays[Mesh::ARRAY_VERTEX] = vertices;
        arrays[Mesh::ARRAY_NORMAL] = normals;
        arrays[Mesh::ARRAY_TEX_UV] = uvs;
        arrays[Mesh::ARRAY_INDEX] = indices;
        
        mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
        
        String res_path = save_path + ".mesh";
        ResourceSaver::save(mesh, res_path);
        r_gen_files.push_back(res_path);
        
        return OK;
    }
};

// #############################################################################
// ResourceImporterImage - Image texture importer
// #############################################################################
class ResourceImporterImage : public ResourceImporter {
    XTU_GODOT_REGISTER_CLASS(ResourceImporterImage, ResourceImporter)

public:
    static StringName get_class_static() { return StringName("ResourceImporterImage"); }

    ResourceImporterImage() {
        m_importer_name = "image";
        m_visible_name = "Image Texture";
        m_extensions = {"png", "jpg", "jpeg", "bmp", "tga", "webp", "hdr", "exr"};
        m_priority = static_cast<float>(ImportPriority::PRIORITY_HIGH);
    }

    String get_save_extension() const override { return "tex"; }
    String get_resource_type() const override { return "CompressedTexture2D"; }

    std::vector<PropertyInfo> get_import_options(const String& path, int preset) const override {
        std::vector<PropertyInfo> options;
        options.push_back({VariantType::INT, "compress/mode", PropertyHint::ENUM, 
                          "Lossless,Lossy,VRAM Compressed,VRAM Uncompressed,Basis Universal"});
        options.push_back({VariantType::BOOL, "mipmaps/generate"});
        options.push_back({VariantType::INT, "mipmaps/limit"});
        options.push_back({VariantType::BOOL, "process/fix_alpha_border"});
        options.push_back({VariantType::BOOL, "process/premult_alpha"});
        options.push_back({VariantType::BOOL, "process/HDR_as_SRGB"});
        options.push_back({VariantType::FLOAT, "process/scale"});
        options.push_back({VariantType::INT, "detect_3d/compress_to", PropertyHint::ENUM,
                          "Disabled,VRAM Compressed,Basis Universal"});
        return options;
    }

    Error import(const String& source_file, const String& save_path,
                 const std::map<String, Variant>& options,
                 std::vector<String>& r_platform_variants,
                 std::vector<String>& r_gen_files) override {
        // Load image data
        Ref<FileAccess> file = FileAccess::open(source_file, FileAccess::READ);
        if (!file.is_valid()) return ERR_FILE_CANT_OPEN;
        
        std::vector<uint8_t> image_data = file->get_buffer(file->get_length());
        
        // Detect format and decode
        Image image;
        Error err = image.load(source_file);
        if (err != OK) return err;
        
        bool mipmaps = options.count("mipmaps/generate") ? options.at("mipmaps/generate").as<bool>() : true;
        if (mipmaps) {
            image.generate_mipmaps();
        }
        
        // Apply compression
        int compress_mode = options.count("compress/mode") ? options.at("compress/mode").as<int>() : 0;
        if (compress_mode == 4) { // Basis Universal
            CompressionOptions comp_opts;
            comp_opts.format = TextureCompressionFormat::FORMAT_BASIS_UNIVERSAL;
            comp_opts.mipmaps = mipmaps;
            comp_opts.srgb = !options.count("process/HDR_as_SRGB") || options.at("process/HDR_as_SRGB").as<bool>();
            
            auto compressed = TextureCompressionManager::get_singleton()->compress_sync(
                image.get_data(), image.get_width(), image.get_height(),
                RenderingServer::TEXTURE_FORMAT_RGBA8, comp_opts);
            
            if (!compressed.empty()) {
                image_data = compressed;
            }
        }
        
        // Save texture
        Ref<CompressedTexture2D> texture;
        texture.instance();
        texture->load_from_image(image, CompressedTexture2D::COMPRESS_SOURCE_GENERIC);
        
        String res_path = save_path + ".tex";
        ResourceSaver::save(texture, res_path);
        r_gen_files.push_back(res_path);
        
        return OK;
    }
};

// #############################################################################
// ResourceImporterFont - Font importer (TTF/OTF)
// #############################################################################
class ResourceImporterFont : public ResourceImporter {
    XTU_GODOT_REGISTER_CLASS(ResourceImporterFont, ResourceImporter)

public:
    static StringName get_class_static() { return StringName("ResourceImporterFont"); }

    ResourceImporterFont() {
        m_importer_name = "font_data";
        m_visible_name = "Font Data";
        m_extensions = {"ttf", "otf", "woff", "woff2"};
        m_priority = static_cast<float>(ImportPriority::PRIORITY_NORMAL);
    }

    String get_save_extension() const override { return "fontdata"; }
    String get_resource_type() const override { return "FontFile"; }

    std::vector<PropertyInfo> get_import_options(const String& path, int preset) const override {
        std::vector<PropertyInfo> options;
        options.push_back({VariantType::BOOL, "antialiased"});
        options.push_back({VariantType::INT, "hinting", PropertyHint::ENUM, "None,Slight,Normal,Full"});
        options.push_back({VariantType::INT, "subpixel_positioning", PropertyHint::ENUM, "Disabled,Auto,Half,One Quarter"});
        options.push_back({VariantType::BOOL, "msdf"});
        options.push_back({VariantType::INT, "msdf_size"});
        options.push_back({VariantType::INT, "msdf_range"});
        options.push_back({VariantType::BOOL, "compress"});
        options.push_back({VariantType::BOOL, "preload/characters"});
        options.push_back({VariantType::STRING, "preload/config"});
        return options;
    }

    Error import(const String& source_file, const String& save_path,
                 const std::map<String, Variant>& options,
                 std::vector<String>& r_platform_variants,
                 std::vector<String>& r_gen_files) override {
        Ref<FileAccess> file = FileAccess::open(source_file, FileAccess::READ);
        if (!file.is_valid()) return ERR_FILE_CANT_OPEN;
        
        std::vector<uint8_t> font_data = file->get_buffer(file->get_length());
        
        Ref<FontFile> font;
        font.instance();
        font->set_data(font_data);
        
        // Apply settings
        if (options.count("antialiased")) {
            font->set_antialiased(options.at("antialiased").as<bool>());
        }
        if (options.count("hinting")) {
            font->set_hinting(static_cast<TextServer::Hinting>(options.at("hinting").as<int>()));
        }
        if (options.count("msdf") && options.at("msdf").as<bool>()) {
            int size = options.count("msdf_size") ? options.at("msdf_size").as<int>() : 48;
            int range = options.count("msdf_range") ? options.at("msdf_range").as<int>() : 8;
            font->set_msdf_size(size);
            font->set_msdf_range(range);
        }
        
        String res_path = save_path + ".fontdata";
        ResourceSaver::save(font, res_path);
        r_gen_files.push_back(res_path);
        
        return OK;
    }
};

// #############################################################################
// ResourceImporterCSV - CSV translation/data importer
// #############################################################################
class ResourceImporterCSV : public ResourceImporter {
    XTU_GODOT_REGISTER_CLASS(ResourceImporterCSV, ResourceImporter)

public:
    static StringName get_class_static() { return StringName("ResourceImporterCSV"); }

    ResourceImporterCSV() {
        m_importer_name = "csv";
        m_visible_name = "CSV Data";
        m_extensions = {"csv"};
        m_priority = static_cast<float>(ImportPriority::PRIORITY_LOW);
    }

    String get_save_extension() const override { return "translation"; }
    String get_resource_type() const override { return "Translation"; }

    std::vector<PropertyInfo> get_import_options(const String& path, int preset) const override {
        std::vector<PropertyInfo> options;
        options.push_back({VariantType::BOOL, "delimiter/comma"});
        options.push_back({VariantType::STRING, "delimiter/custom"});
        options.push_back({VariantType::BOOL, "headers/first_row"});
        options.push_back({VariantType::INT, "compress/mode", PropertyHint::ENUM, "None,PO"});
        return options;
    }

    Error import(const String& source_file, const String& save_path,
                 const std::map<String, Variant>& options,
                 std::vector<String>& r_platform_variants,
                 std::vector<String>& r_gen_files) override {
        Ref<FileAccess> file = FileAccess::open(source_file, FileAccess::READ);
        if (!file.is_valid()) return ERR_FILE_CANT_OPEN;
        
        String content = file->get_as_text();
        std::vector<String> lines = content.split("\n");
        
        if (lines.empty()) return ERR_PARSE_ERROR;
        
        char delimiter = ',';
        if (options.count("delimiter/custom")) {
            String custom = options.at("delimiter/custom").as<String>();
            if (!custom.empty()) delimiter = custom[0];
        }
        
        Ref<Translation> translation;
        translation.instance();
        
        // Parse CSV
        for (const String& line : lines) {
            auto parts = line.split(String::chr(delimiter));
            if (parts.size() >= 2) {
                String msgid = parts[0].strip_edges();
                String msgstr = parts[1].strip_edges();
                if (!msgid.empty()) {
                    translation->add_message(msgid, msgstr);
                }
            }
        }
        
        String res_path = save_path + ".translation";
        ResourceSaver::save(translation, res_path);
        r_gen_files.push_back(res_path);
        
        return OK;
    }
};

// #############################################################################
// ResourceImporterShader - Shader include importer
// #############################################################################
class ResourceImporterShader : public ResourceImporter {
    XTU_GODOT_REGISTER_CLASS(ResourceImporterShader, ResourceImporter)

public:
    static StringName get_class_static() { return StringName("ResourceImporterShader"); }

    ResourceImporterShader() {
        m_importer_name = "gdshaderinc";
        m_visible_name = "Shader Include";
        m_extensions = {"gdshaderinc", "glsl", "hlsl"};
        m_priority = static_cast<float>(ImportPriority::PRIORITY_LOW);
    }

    String get_save_extension() const override { return "gdshaderinc"; }
    String get_resource_type() const override { return "Resource"; }

    std::vector<PropertyInfo> get_import_options(const String& path, int preset) const override {
        return {};
    }

    Error import(const String& source_file, const String& save_path,
                 const std::map<String, Variant>& options,
                 std::vector<String>& r_platform_variants,
                 std::vector<String>& r_gen_files) override {
        // Simply copy the shader include file
        Ref<FileAccess> src = FileAccess::open(source_file, FileAccess::READ);
        if (!src.is_valid()) return ERR_FILE_CANT_OPEN;
        
        String content = src->get_as_text();
        
        String res_path = save_path + ".gdshaderinc";
        Ref<FileAccess> dst = FileAccess::open(res_path, FileAccess::WRITE);
        if (!dst.is_valid()) return ERR_FILE_CANT_WRITE;
        
        dst->store_string(content);
        r_gen_files.push_back(res_path);
        
        return OK;
    }
};

// #############################################################################
// ImportManager - Central import registration
// #############################################################################
class ImportManager : public Object {
    XTU_GODOT_REGISTER_CLASS(ImportManager, Object)

private:
    static ImportManager* s_singleton;

public:
    static ImportManager* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("ImportManager"); }

    ImportManager() {
        s_singleton = this;
        register_default_importers();
    }

    ~ImportManager() { s_singleton = nullptr; }

    void register_default_importers() {
        EditorFileSystemImport::get_singleton()->add_importer(Ref<ResourceImporter>(new ResourceImporterWAV()));
        EditorFileSystemImport::get_singleton()->add_importer(Ref<ResourceImporter>(new ResourceImporterOBJ()));
        EditorFileSystemImport::get_singleton()->add_importer(Ref<ResourceImporter>(new ResourceImporterImage()));
        EditorFileSystemImport::get_singleton()->add_importer(Ref<ResourceImporter>(new ResourceImporterFont()));
        EditorFileSystemImport::get_singleton()->add_importer(Ref<ResourceImporter>(new ResourceImporterCSV()));
        EditorFileSystemImport::get_singleton()->add_importer(Ref<ResourceImporter>(new ResourceImporterShader()));
    }
};

} // namespace editor

// Bring into main namespace
using editor::ResourceImporterWAV;
using editor::ResourceImporterOBJ;
using editor::ResourceImporterImage;
using editor::ResourceImporterFont;
using editor::ResourceImporterCSV;
using editor::ResourceImporterShader;
using editor::ImportManager;
using editor::ImportPriority;
using editor::ImageImportFormat;
using editor::AudioImportFormat;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XRESOURCE_IMPORTER_HPP