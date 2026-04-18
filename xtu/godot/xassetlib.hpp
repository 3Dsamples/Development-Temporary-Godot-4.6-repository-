// include/xtu/godot/xassetlib.hpp
// xtensor-unified - Asset Library integration for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XASSETLIB_HPP
#define XTU_GODOT_XASSETLIB_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xeditor.hpp"
#include "xtu/godot/xcore.hpp"
#include "xtu/io/xio_json.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class AssetLibrary;
class AssetLibraryClient;
class AssetInstaller;
class AssetItem;

// #############################################################################
// Asset sort methods
// #############################################################################
enum class AssetSortMethod : uint8_t {
    SORT_RELEVANCE = 0,
    SORT_UPDATED = 1,
    SORT_CREATED = 2,
    SORT_NAME = 3,
    SORT_DOWNLOADS = 4,
    SORT_STARS = 5
};

// #############################################################################
// Asset category types
// #############################################################################
enum class AssetCategory : uint16_t {
    CATEGORY_ANY = 0,
    CATEGORY_2D_TOOLS = 1,
    CATEGORY_3D_TOOLS = 2,
    CATEGORY_ADDONS = 3,
    CATEGORY_DEMOS = 4,
    CATEGORY_MATERIALS = 5,
    CATEGORY_MODELS = 6,
    CATEGORY_PLUGINS = 7,
    CATEGORY_PROJECTS = 8,
    CATEGORY_SCRIPTS = 9,
    CATEGORY_SHADERS = 10,
    CATEGORY_TEXTURES = 11,
    CATEGORY_THEMES = 12,
    CATEGORY_TOOLS = 13
};

// #############################################################################
// Asset support levels
// #############################################################################
enum class AssetSupportLevel : uint8_t {
    SUPPORT_UNKNOWN = 0,
    SUPPORT_OFFICIAL = 1,
    SUPPORT_COMMUNITY = 2,
    SUPPORT_TESTING = 3
};

// #############################################################################
// Asset download state
// #############################################################################
enum class AssetDownloadState : uint8_t {
    STATE_IDLE = 0,
    STATE_QUEUED = 1,
    STATE_DOWNLOADING = 2,
    STATE_EXTRACTING = 3,
    STATE_COMPLETED = 4,
    STATE_ERROR = 5
};

// #############################################################################
// AssetItem - Single asset from the library
// #############################################################################
class AssetItem : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(AssetItem, RefCounted)

private:
    int m_id = 0;
    String m_title;
    String m_author;
    String m_description;
    String m_version;
    String m_license;
    String m_download_url;
    String m_icon_url;
    String m_browse_url;
    AssetCategory m_category = AssetCategory::CATEGORY_ANY;
    AssetSupportLevel m_support_level = AssetSupportLevel::SUPPORT_COMMUNITY;
    int m_stars = 0;
    int m_downloads = 0;
    uint64_t m_updated_at = 0;
    uint64_t m_created_at = 0;
    String m_godot_version;
    String m_engine_version;
    std::vector<String> m_tags;
    std::vector<String> m_previews;
    float m_cost = 0.0f;
    bool m_free = true;

public:
    static StringName get_class_static() { return StringName("AssetItem"); }

    void set_id(int id) { m_id = id; }
    int get_id() const { return m_id; }

    void set_title(const String& title) { m_title = title; }
    String get_title() const { return m_title; }

    void set_author(const String& author) { m_author = author; }
    String get_author() const { return m_author; }

    void set_description(const String& desc) { m_description = desc; }
    String get_description() const { return m_description; }

    void set_version(const String& ver) { m_version = ver; }
    String get_version() const { return m_version; }

    void set_license(const String& lic) { m_license = lic; }
    String get_license() const { return m_license; }

    void set_download_url(const String& url) { m_download_url = url; }
    String get_download_url() const { return m_download_url; }

    void set_icon_url(const String& url) { m_icon_url = url; }
    String get_icon_url() const { return m_icon_url; }

    void set_category(AssetCategory cat) { m_category = cat; }
    AssetCategory get_category() const { return m_category; }

    void set_support_level(AssetSupportLevel level) { m_support_level = level; }
    AssetSupportLevel get_support_level() const { return m_support_level; }

    void set_stars(int stars) { m_stars = stars; }
    int get_stars() const { return m_stars; }

    void set_downloads(int downloads) { m_downloads = downloads; }
    int get_downloads() const { return m_downloads; }

    void set_updated_at(uint64_t time) { m_updated_at = time; }
    uint64_t get_updated_at() const { return m_updated_at; }

    void set_godot_version(const String& ver) { m_godot_version = ver; }
    String get_godot_version() const { return m_godot_version; }

    void set_tags(const std::vector<String>& tags) { m_tags = tags; }
    const std::vector<String>& get_tags() const { return m_tags; }

    void set_free(bool free) { m_free = free; }
    bool is_free() const { return m_free; }

    static Ref<AssetItem> from_json(const io::json::JsonValue& json) {
        Ref<AssetItem> item;
        item.instance();
        item->set_id(json["id"].as_number());
        item->set_title(json["title"].as_string().c_str());
        item->set_author(json["author"].as_string().c_str());
        item->set_description(json["description"].as_string().c_str());
        item->set_version(json["version"].as_string().c_str());
        item->set_license(json["license"].as_string().c_str());
        item->set_download_url(json["download_url"].as_string().c_str());
        item->set_icon_url(json["icon_url"].as_string().c_str());
        item->set_category(static_cast<AssetCategory>(static_cast<int>(json["category"].as_number())));
        item->set_support_level(static_cast<AssetSupportLevel>(static_cast<int>(json["support_level"].as_number())));
        item->set_stars(static_cast<int>(json["stars"].as_number()));
        item->set_downloads(static_cast<int>(json["downloads"].as_number()));
        item->set_updated_at(static_cast<uint64_t>(json["updated_at"].as_number()));
        item->set_godot_version(json["godot_version"].as_string().c_str());
        return item;
    }
};

// #############################################################################
// AssetLibraryClient - HTTP client for Godot Asset Library API
// #############################################################################
class AssetLibraryClient : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(AssetLibraryClient, RefCounted)

public:
    struct SearchQuery {
        String filter;
        AssetCategory category = AssetCategory::CATEGORY_ANY;
        AssetSortMethod sort = AssetSortMethod::SORT_UPDATED;
        int page = 0;
        int page_size = 20;
        String godot_version;
        AssetSupportLevel support_level = AssetSupportLevel::SUPPORT_COMMUNITY;
    };

    struct SearchResult {
        std::vector<Ref<AssetItem>> items;
        int total_pages = 0;
        int total_items = 0;
        int current_page = 0;
    };

private:
    String m_api_url = "https://godotengine.org/asset-library/api";
    std::string m_user_agent = "Godot/4.6 Xtensor-AssetLib";
    int m_timeout_seconds = 30;
    std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("AssetLibraryClient"); }

    void set_api_url(const String& url) { m_api_url = url; }
    String get_api_url() const { return m_api_url; }

    void set_timeout(int seconds) { m_timeout_seconds = seconds; }
    int get_timeout() const { return m_timeout_seconds; }

    SearchResult search(const SearchQuery& query) {
        SearchResult result;
        String url = build_search_url(query);
        String response = http_get(url);
        if (!response.empty()) {
            parse_search_response(response, result);
        }
        return result;
    }

    Ref<AssetItem> get_asset(int asset_id) {
        String url = m_api_url + "/asset/" + String::num(asset_id);
        String response = http_get(url);
        if (!response.empty()) {
            io::json::JsonValue json = io::json::JsonValue::parse(response.to_std_string());
            return AssetItem::from_json(json);
        }
        return Ref<AssetItem>();
    }

    std::vector<Ref<AssetItem>> get_featured() {
        String url = m_api_url + "/asset/featured";
        String response = http_get(url);
        std::vector<Ref<AssetItem>> items;
        if (!response.empty()) {
            io::json::JsonValue json = io::json::JsonValue::parse(response.to_std_string());
            if (json.is_object() && json["items"].is_array()) {
                for (const auto& item_json : json["items"].as_array()) {
                    items.push_back(AssetItem::from_json(item_json));
                }
            }
        }
        return items;
    }

private:
    String build_search_url(const SearchQuery& query) const {
        String url = m_api_url + "/asset?";
        if (!query.filter.empty()) {
            url += "filter=" + url_encode(query.filter) + "&";
        }
        url += "category=" + String::num(static_cast<int>(query.category)) + "&";
        url += "sort=" + String::num(static_cast<int>(query.sort)) + "&";
        url += "page=" + String::num(query.page) + "&";
        url += "page_size=" + String::num(query.page_size) + "&";
        if (!query.godot_version.empty()) {
            url += "godot_version=" + query.godot_version + "&";
        }
        url += "support_level=" + String::num(static_cast<int>(query.support_level));
        return url;
    }

    String url_encode(const String& str) const {
        std::string s = str.to_std_string();
        std::ostringstream escaped;
        for (char c : s) {
            if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
                escaped << c;
            } else {
                escaped << '%' << std::hex << std::uppercase << static_cast<int>(static_cast<unsigned char>(c));
            }
        }
        return String(escaped.str().c_str());
    }

    String http_get(const String& url) const {
        // Simplified HTTP GET - in production would use proper HTTP client
        std::string cmd = "curl -s --max-time " + std::to_string(m_timeout_seconds) +
                          " -H 'User-Agent: " + m_user_agent + "' '" + url.to_std_string() + "'";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) return String();
        std::string result;
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe)) {
            result += buffer;
        }
        pclose(pipe);
        return String(result.c_str());
    }

    void parse_search_response(const String& response, SearchResult& result) const {
        io::json::JsonValue json = io::json::JsonValue::parse(response.to_std_string());
        if (json.is_object()) {
            result.total_pages = static_cast<int>(json["pages"].as_number());
            result.total_items = static_cast<int>(json["total"].as_number());
            result.current_page = static_cast<int>(json["page"].as_number());
            if (json["result"].is_array()) {
                for (const auto& item_json : json["result"].as_array()) {
                    result.items.push_back(AssetItem::from_json(item_json));
                }
            }
        }
    }
};

// #############################################################################
// AssetDownloader - Background asset downloader
// #############################################################################
class AssetDownloader : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(AssetDownloader, RefCounted)

public:
    struct DownloadTask {
        int asset_id = 0;
        String url;
        String local_path;
        AssetDownloadState state = AssetDownloadState::STATE_IDLE;
        float progress = 0.0f;
        String error_message;
        std::function<void()> completion_callback;
        std::function<void(float)> progress_callback;
    };

private:
    std::queue<DownloadTask> m_queue;
    DownloadTask m_current_task;
    std::thread m_worker;
    std::atomic<bool> m_worker_running{false};
    std::mutex m_mutex;
    String m_download_dir;

public:
    static StringName get_class_static() { return StringName("AssetDownloader"); }

    AssetDownloader() = default;
    ~AssetDownloader() { stop_worker(); }

    void set_download_dir(const String& dir) { m_download_dir = dir; }
    String get_download_dir() const { return m_download_dir; }

    void enqueue(const DownloadTask& task) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(task);
        start_worker();
    }

    void cancel_current() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_current_task.state == AssetDownloadState::STATE_DOWNLOADING) {
            m_current_task.state = AssetDownloadState::STATE_IDLE;
        }
    }

    void clear_queue() {
        std::lock_guard<std::mutex> lock(m_mutex);
        while (!m_queue.empty()) m_queue.pop();
    }

    DownloadTask get_current_task() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_current_task;
    }

    size_t queue_size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.size();
    }

private:
    void start_worker() {
        if (m_worker_running) return;
        m_worker_running = true;
        m_worker = std::thread([this]() { worker_loop(); });
    }

    void stop_worker() {
        m_worker_running = false;
        if (m_worker.joinable()) m_worker.join();
    }

    void worker_loop() {
        while (m_worker_running) {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_queue.empty()) {
                    break;
                }
                m_current_task = m_queue.front();
                m_queue.pop();
                m_current_task.state = AssetDownloadState::STATE_DOWNLOADING;
            }

            bool success = download_file(m_current_task.url, m_current_task.local_path, m_current_task);

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_current_task.state = success ? AssetDownloadState::STATE_COMPLETED : AssetDownloadState::STATE_ERROR;
                if (m_current_task.completion_callback) {
                    m_current_task.completion_callback();
                }
            }
        }
        m_worker_running = false;
    }

    bool download_file(const String& url, const String& path, DownloadTask& task) {
        String full_path = path;
        if (full_path.empty()) {
            full_path = m_download_dir + "/asset_" + String::num(task.asset_id) + ".zip";
        }

        std::string cmd = "curl -L -o '" + full_path.to_std_string() + "' '" + url.to_std_string() + "'";
        int ret = system(cmd.c_str());
        return ret == 0;
    }
};

// #############################################################################
// AssetInstaller - Extracts and installs assets
// #############################################################################
class AssetInstaller : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(AssetInstaller, RefCounted)

private:
    String m_asset_path;
    String m_target_path;
    bool m_ignore_asset_root = true;
    std::vector<String> m_ignore_directories;
    std::vector<String> m_ignore_files;
    std::function<void(float)> m_progress_callback;
    std::function<void(bool, const String&)> m_completion_callback;

public:
    static StringName get_class_static() { return StringName("AssetInstaller"); }

    void set_asset_path(const String& path) { m_asset_path = path; }
    String get_asset_path() const { return m_asset_path; }

    void set_target_path(const String& path) { m_target_path = path; }
    String get_target_path() const { return m_target_path; }

    void set_ignore_asset_root(bool ignore) { m_ignore_asset_root = ignore; }
    bool get_ignore_asset_root() const { return m_ignore_asset_root; }

    void add_ignore_directory(const String& dir) { m_ignore_directories.push_back(dir); }
    void add_ignore_file(const String& file) { m_ignore_files.push_back(file); }

    void set_progress_callback(std::function<void(float)> cb) { m_progress_callback = cb; }
    void set_completion_callback(std::function<void(bool, const String&)> cb) { m_completion_callback = cb; }

    void install() {
        String temp_dir = OS::get_singleton()->get_user_data_dir() + "/temp/asset_install";
        DirAccess::make_dir_recursive(temp_dir);

        // Unzip
        String cmd = "unzip -o '" + m_asset_path.to_std_string() + "' -d '" + temp_dir.to_std_string() + "'";
        int ret = system(cmd.c_str());

        if (ret != 0) {
            if (m_completion_callback) m_completion_callback(false, "Failed to extract archive");
            return;
        }

        // Find asset root (the folder containing project.godot or the first folder)
        String source_dir = temp_dir;
        if (m_ignore_asset_root) {
            Ref<DirAccess> dir = DirAccess::open(temp_dir);
            if (dir.is_valid()) {
                dir->list_dir_begin();
                String first_item;
                while (!(first_item = dir->get_next()).empty()) {
                    if (first_item != "." && first_item != ".." && dir->current_is_dir()) {
                        source_dir = temp_dir + "/" + first_item;
                        break;
                    }
                }
                dir->list_dir_end();
            }
        }

        // Copy files
        copy_directory(source_dir, m_target_path);

        // Cleanup
        DirAccess::remove(temp_dir);

        if (m_completion_callback) m_completion_callback(true, "");
    }

private:
    void copy_directory(const String& from, const String& to) {
        DirAccess::make_dir_recursive(to);
        Ref<DirAccess> dir = DirAccess::open(from);
        if (!dir.is_valid()) return;

        dir->list_dir_begin();
        String item;
        while (!(item = dir->get_next()).empty()) {
            if (item == "." || item == "..") continue;
            String src = from + "/" + item;
            String dst = to + "/" + item;

            bool ignore = false;
            for (const auto& pattern : m_ignore_directories) {
                if (item == pattern) { ignore = true; break; }
            }
            for (const auto& pattern : m_ignore_files) {
                if (item == pattern) { ignore = true; break; }
            }
            if (ignore) continue;

            if (dir->current_is_dir()) {
                copy_directory(src, dst);
            } else {
                DirAccess::copy(src, dst);
            }
        }
        dir->list_dir_end();
    }
};

// #############################################################################
// AssetLibrary - Main asset library manager
// #############################################################################
class AssetLibrary : public Object {
    XTU_GODOT_REGISTER_CLASS(AssetLibrary, Object)

private:
    static AssetLibrary* s_singleton;
    Ref<AssetLibraryClient> m_client;
    Ref<AssetDownloader> m_downloader;
    Ref<AssetInstaller> m_installer;
    String m_cache_dir;
    std::vector<Ref<AssetItem>> m_featured_cache;
    std::mutex m_mutex;

public:
    static AssetLibrary* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("AssetLibrary"); }

    AssetLibrary() {
        s_singleton = this;
        m_client.instance();
        m_downloader.instance();
        m_installer.instance();
        m_cache_dir = OS::get_singleton()->get_user_data_dir() + "/asset_cache";
        DirAccess::make_dir_recursive(m_cache_dir);
        m_downloader->set_download_dir(m_cache_dir);
    }

    ~AssetLibrary() { s_singleton = nullptr; }

    void search_assets(const AssetLibraryClient::SearchQuery& query,
                       std::function<void(const AssetLibraryClient::SearchResult&)> callback) {
        std::thread([this, query, callback]() {
            auto result = m_client->search(query);
            if (callback) callback(result);
        }).detach();
    }

    void get_asset(int asset_id, std::function<void(Ref<AssetItem>)> callback) {
        std::thread([this, asset_id, callback]() {
            auto item = m_client->get_asset(asset_id);
            if (callback) callback(item);
        }).detach();
    }

    void get_featured(std::function<void(std::vector<Ref<AssetItem>>)> callback) {
        std::thread([this, callback]() {
            if (!m_featured_cache.empty()) {
                if (callback) callback(m_featured_cache);
                return;
            }
            auto items = m_client->get_featured();
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_featured_cache = items;
            }
            if (callback) callback(items);
        }).detach();
    }

    void download_asset(int asset_id, const String& target_path,
                        std::function<void(bool, const String&)> callback) {
        Ref<AssetItem> item = m_client->get_asset(asset_id);
        if (!item.is_valid()) {
            if (callback) callback(false, "Asset not found");
            return;
        }

        AssetDownloader::DownloadTask task;
        task.asset_id = asset_id;
        task.url = item->get_download_url();
        task.local_path = m_cache_dir + "/" + String::num(asset_id) + ".zip";
        task.completion_callback = [this, task, target_path, callback]() {
            if (task.state == AssetDownloadState::STATE_COMPLETED) {
                install_asset(task.local_path, target_path, callback);
            } else {
                if (callback) callback(false, "Download failed");
            }
        };

        m_downloader->enqueue(task);
    }

    void install_asset(const String& asset_path, const String& target_path,
                       std::function<void(bool, const String&)> callback) {
        m_installer->set_asset_path(asset_path);
        m_installer->set_target_path(target_path);
        m_installer->set_completion_callback(callback);
        std::thread([this]() { m_installer->install(); }).detach();
    }

    void cancel_downloads() { m_downloader->cancel_current(); }
    size_t get_queue_size() const { return m_downloader->queue_size(); }

    void clear_cache() {
        DirAccess::remove(m_cache_dir);
        DirAccess::make_dir_recursive(m_cache_dir);
    }
};

} // namespace editor

// Bring into main namespace
using editor::AssetLibrary;
using editor::AssetLibraryClient;
using editor::AssetDownloader;
using editor::AssetInstaller;
using editor::AssetItem;
using editor::AssetCategory;
using editor::AssetSupportLevel;
using editor::AssetSortMethod;
using editor::AssetDownloadState;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XASSETLIB_HPP