// include/xtu/godot/xmisc_modules.hpp
// xtensor-unified - Miscellaneous modules for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XMISC_MODULES_HPP
#define XTU_GODOT_XMISC_MODULES_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xcore.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/godot/xtext_server.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Part 1: OpenImageDenoise Wrapper (modules/denoise)
// #############################################################################
#ifdef XTU_USE_OIDN
#include <OpenImageDenoise/oidn.hpp>
#endif

class OIDNDenoiser : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(OIDNDenoiser, RefCounted)

public:
    enum Quality {
        QUALITY_FAST,
        QUALITY_BALANCED,
        QUALITY_HIGH
    };

private:
    Quality m_quality = Quality::QUALITY_BALANCED;
    bool m_hdr = true;
    bool m_clean_aux = true;
    mutable std::mutex m_mutex;
#ifdef XTU_USE_OIDN
    oidn::DeviceRef m_device;
#endif

public:
    static StringName get_class_static() { return StringName("OIDNDenoiser"); }

    OIDNDenoiser() {
#ifdef XTU_USE_OIDN
        m_device = oidn::newDevice();
        m_device.commit();
#endif
    }

    ~OIDNDenoiser() {
#ifdef XTU_USE_OIDN
        m_device.release();
#endif
    }

    void set_quality(Quality quality) { m_quality = quality; }
    Quality get_quality() const { return m_quality; }

    void set_hdr(bool hdr) { m_hdr = hdr; }
    bool is_hdr() const { return m_hdr; }

    std::vector<uint8_t> denoise(const std::vector<uint8_t>& color_data,
                                  const std::vector<uint8_t>& albedo_data,
                                  const std::vector<uint8_t>& normal_data,
                                  int width, int height) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<uint8_t> result(width * height * 4 * sizeof(float));

#ifdef XTU_USE_OIDN
        oidn::FilterRef filter = m_device.newFilter("RT");
        filter.setImage("color", color_data.data(), oidn::Format::Float3, width, height);
        filter.setImage("albedo", albedo_data.data(), oidn::Format::Float3, width, height);
        filter.setImage("normal", normal_data.data(), oidn::Format::Float3, width, height);
        filter.setImage("output", result.data(), oidn::Format::Float3, width, height);
        filter.set("hdr", m_hdr);
        filter.set("cleanAux", m_clean_aux);
        filter.commit();
        filter.execute();
#endif
        return result;
    }

    std::vector<uint8_t> denoise_lightmap(const std::vector<uint8_t>& lightmap_data,
                                           int width, int height) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<uint8_t> result(width * height * 4 * sizeof(float));

#ifdef XTU_USE_OIDN
        oidn::FilterRef filter = m_device.newFilter("RTLightmap");
        filter.setImage("color", lightmap_data.data(), oidn::Format::Float3, width, height);
        filter.setImage("output", result.data(), oidn::Format::Float3, width, height);
        filter.commit();
        filter.execute();
#endif
        return result;
    }
};

// #############################################################################
// Part 2: Fallback TextServer (modules/text_server_fb)
// #############################################################################
class TextServerFallback : public TextServer {
    XTU_GODOT_REGISTER_CLASS(TextServerFallback, TextServer)

private:
    static TextServerFallback* s_singleton;
    std::unordered_map<String, Ref<Font>> m_fonts;

public:
    static TextServerFallback* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("TextServerFallback"); }

    TextServerFallback() { s_singleton = this; }
    ~TextServerFallback() { s_singleton = nullptr; }

    Ref<Font> create_font() override {
        Ref<BitmapFont> font;
        font.instance();
        return font;
    }

    std::vector<TextLine> shape_text(const String& text, const Ref<Font>& font,
                                      int size, TextDirection direction) override {
        std::vector<TextLine> lines;
        TextLine line;
        line.direction = direction;
        line.ascent = font->get_ascent(size);
        line.descent = font->get_descent(size);

        std::string utf8 = text.to_std_string();
        const char* ptr = utf8.c_str();
        float x = 0.0f;

        while (*ptr) {
            char32_t c = 0;
            if ((*ptr & 0x80) == 0) {
                c = *ptr++;
            } else if ((*ptr & 0xE0) == 0xC0) {
                c = ((*ptr++ & 0x1F) << 6) | (*ptr++ & 0x3F);
            } else if ((*ptr & 0xF0) == 0xE0) {
                c = ((*ptr++ & 0x0F) << 12) | ((*ptr++ & 0x3F) << 6) | (*ptr++ & 0x3F);
            } else if ((*ptr & 0xF8) == 0xF0) {
                c = ((*ptr++ & 0x07) << 18) | ((*ptr++ & 0x3F) << 12) | ((*ptr++ & 0x3F) << 6) | (*ptr++ & 0x3F);
            } else {
                ++ptr;
                continue;
            }

            GlyphMetrics metrics = font->get_char_metrics(c, size);
            if (metrics.valid) {
                line.glyphs.push_back(metrics);
                x += metrics.advance.x();
            }
        }

        line.width = x;
        lines.push_back(line);
        return lines;
    }

    vec2f get_string_size(const String& text, const Ref<Font>& font, int size) override {
        auto lines = shape_text(text, font, size);
        if (lines.empty()) return vec2f(0, 0);
        return vec2f(lines[0].width, lines[0].ascent + lines[0].descent);
    }

    float get_ascent(const Ref<Font>& font, int size) override {
        return font.is_valid() ? font->get_ascent(size) : size * 0.8f;
    }

    float get_descent(const Ref<Font>& font, int size) override {
        return font.is_valid() ? font->get_descent(size) : size * 0.2f;
    }

    void draw_string(RID canvas_item, const vec2f& pos, const String& text,
                     const Ref<Font>& font, int size, const Color& color) override {
        if (font.is_valid()) {
            font->draw_string(canvas_item, pos, text, size, color);
        }
    }
};

// #############################################################################
// Part 3: ResourceFormatLoader/Saver Registry (core/io)
// #############################################################################
class ResourceFormatLoaderRegistry {
public:
    using LoaderFactory = std::function<Ref<ResourceFormatLoader>()>;
    using SaverFactory = std::function<Ref<ResourceFormatSaver>()>;

private:
    static ResourceFormatLoaderRegistry* s_singleton;
    std::map<String, std::vector<LoaderFactory>> m_loader_factories;
    std::map<String, std::vector<SaverFactory>> m_saver_factories;
    std::mutex m_mutex;

public:
    static ResourceFormatLoaderRegistry* get_singleton() {
        if (!s_singleton) s_singleton = new ResourceFormatLoaderRegistry();
        return s_singleton;
    }

    void register_loader(const String& extension, LoaderFactory factory) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_loader_factories[extension].push_back(factory);
    }

    void register_saver(const String& extension, SaverFactory factory) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_saver_factories[extension].push_back(factory);
    }

    std::vector<Ref<ResourceFormatLoader>> get_loaders(const String& extension) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<Ref<ResourceFormatLoader>> result;
        auto it = m_loader_factories.find(extension);
        if (it != m_loader_factories.end()) {
            for (const auto& factory : it->second) {
                result.push_back(factory());
            }
        }
        return result;
    }

    std::vector<Ref<ResourceFormatSaver>> get_savers(const String& extension) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<Ref<ResourceFormatSaver>> result;
        auto it = m_saver_factories.find(extension);
        if (it != m_saver_factories.end()) {
            for (const auto& factory : it->second) {
                result.push_back(factory());
            }
        }
        return result;
    }
};

// #############################################################################
// Part 4: Network File Access (core/io/file_access_network.h)
// #############################################################################
class FileAccessNetwork : public FileAccess {
    XTU_GODOT_REGISTER_CLASS(FileAccessNetwork, FileAccess)

private:
    String m_url;
    std::vector<uint8_t> m_data;
    size_t m_position = 0;
    bool m_open = false;
    std::mutex m_mutex;
    std::function<void(float)> m_progress_callback;

public:
    static StringName get_class_static() { return StringName("FileAccessNetwork"); }

    static Ref<FileAccessNetwork> open_http(const String& url) {
        Ref<FileAccessNetwork> fa;
        fa.instance();
        if (fa->open_internal(url)) {
            return fa;
        }
        return Ref<FileAccessNetwork>();
    }

    bool open_internal(const String& url) {
        m_url = url;
        // HTTP GET request
        std::string cmd = "curl -L -s '" + url.to_std_string() + "'";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) return false;

        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe)) {
            m_data.insert(m_data.end(), buffer, buffer + strlen(buffer));
        }
        pclose(pipe);
        m_open = !m_data.empty();
        return m_open;
    }

    void close() override {
        m_open = false;
        m_data.clear();
        m_position = 0;
    }

    bool is_open() const override { return m_open; }

    void seek(size_t pos) override {
        m_position = std::min(pos, m_data.size());
    }

    void seek_end(int64_t pos) override {
        m_position = m_data.size() + pos;
    }

    size_t get_position() const override { return m_position; }
    size_t get_length() const override { return m_data.size(); }
    bool eof_reached() const override { return m_position >= m_data.size(); }

    uint8_t get_8() override {
        if (m_position < m_data.size()) return m_data[m_position++];
        return 0;
    }

    std::vector<uint8_t> get_buffer(size_t len) override {
        len = std::min(len, m_data.size() - m_position);
        std::vector<uint8_t> result(m_data.begin() + m_position,
                                     m_data.begin() + m_position + len);
        m_position += len;
        return result;
    }

    void set_progress_callback(std::function<void(float)> cb) { m_progress_callback = cb; }
};

// #############################################################################
// Part 5: ZIP File/Dir Access (core/io)
// #############################################################################
#ifdef XTU_USE_MINIZIP
#include <minizip/unzip.h>
#endif

class FileAccessZIP : public FileAccess {
    XTU_GODOT_REGISTER_CLASS(FileAccessZIP, FileAccess)

private:
    String m_zip_path;
    String m_internal_path;
    std::vector<uint8_t> m_data;
    size_t m_position = 0;
    bool m_open = false;

public:
    static StringName get_class_static() { return StringName("FileAccessZIP"); }

    static Ref<FileAccessZIP> open_zip(const String& zip_path, const String& internal_path) {
        Ref<FileAccessZIP> fa;
        fa.instance();
        if (fa->open_internal(zip_path, internal_path)) {
            return fa;
        }
        return Ref<FileAccessZIP>();
    }

    bool open_internal(const String& zip_path, const String& internal_path) {
        m_zip_path = zip_path;
        m_internal_path = internal_path;

#ifdef XTU_USE_MINIZIP
        unzFile zf = unzOpen(zip_path.utf8());
        if (!zf) return false;

        if (unzLocateFile(zf, internal_path.utf8(), 0) != UNZ_OK) {
            unzClose(zf);
            return false;
        }

        unz_file_info file_info;
        unzGetCurrentFileInfo(zf, &file_info, nullptr, 0, nullptr, 0, nullptr, 0);

        if (unzOpenCurrentFile(zf) != UNZ_OK) {
            unzClose(zf);
            return false;
        }

        m_data.resize(file_info.uncompressed_size);
        int read = unzReadCurrentFile(zf, m_data.data(), static_cast<unsigned>(m_data.size()));
        unzCloseCurrentFile(zf);
        unzClose(zf);

        m_open = (read == static_cast<int>(m_data.size()));
        return m_open;
#else
        return false;
#endif
    }

    void close() override { m_open = false; m_data.clear(); }
    bool is_open() const override { return m_open; }
    void seek(size_t pos) override { m_position = std::min(pos, m_data.size()); }
    size_t get_position() const override { return m_position; }
    size_t get_length() const override { return m_data.size(); }
    bool eof_reached() const override { return m_position >= m_data.size(); }
    uint8_t get_8() override { return m_position < m_data.size() ? m_data[m_position++] : 0; }
    std::vector<uint8_t> get_buffer(size_t len) override {
        len = std::min(len, m_data.size() - m_position);
        std::vector<uint8_t> result(m_data.begin() + m_position, m_data.begin() + m_position + len);
        m_position += len;
        return result;
    }
};

class DirAccessZIP : public DirAccess {
    XTU_GODOT_REGISTER_CLASS(DirAccessZIP, DirAccess)

private:
    String m_zip_path;
    String m_current_dir;
    std::vector<String> m_entries;
    size_t m_list_index = 0;

public:
    static StringName get_class_static() { return StringName("DirAccessZIP"); }

    static Ref<DirAccessZIP> open_zip(const String& zip_path, const String& dir_path = "") {
        Ref<DirAccessZIP> da;
        da.instance();
        if (da->open_internal(zip_path, dir_path)) {
            return da;
        }
        return Ref<DirAccessZIP>();
    }

    bool open_internal(const String& zip_path, const String& dir_path) {
        m_zip_path = zip_path;
        m_current_dir = dir_path;

#ifdef XTU_USE_MINIZIP
        unzFile zf = unzOpen(zip_path.utf8());
        if (!zf) return false;

        if (unzGoToFirstFile(zf) != UNZ_OK) {
            unzClose(zf);
            return false;
        }

        do {
            char filename[512];
            unz_file_info file_info;
            unzGetCurrentFileInfo(zf, &file_info, filename, sizeof(filename), nullptr, 0, nullptr, 0);

            String full_path(filename);
            if (!dir_path.empty() && !full_path.begins_with(dir_path)) continue;

            String relative = full_path.substr(dir_path.length());
            if (relative.empty()) continue;

            size_t slash = relative.find('/');
            if (slash != String::npos) {
                String dir_name = relative.substr(0, slash);
                if (std::find(m_entries.begin(), m_entries.end(), dir_name) == m_entries.end()) {
                    m_entries.push_back(dir_name);
                }
            } else {
                m_entries.push_back(relative);
            }
        } while (unzGoToNextFile(zf) == UNZ_OK);

        unzClose(zf);
#endif
        return true;
    }

    void list_dir_begin() override { m_list_index = 0; }
    String get_next() override {
        if (m_list_index < m_entries.size()) {
            return m_entries[m_list_index++];
        }
        return String();
    }
    bool current_is_dir() const override {
        if (m_list_index > 0 && m_list_index <= m_entries.size()) {
            return !m_entries[m_list_index - 1].contains('.');
        }
        return false;
    }
    void list_dir_end() override { m_entries.clear(); }
    int get_directories_count() { return 0; }
    int get_files_count() { return 0; }
};

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XMISC_MODULES_HPP