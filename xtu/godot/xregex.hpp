// include/xtu/godot/xregex.hpp
// xtensor-unified - Regular Expression module for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XREGEX_HPP
#define XTU_GODOT_XREGEX_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class RegEx;
class RegExMatch;

// #############################################################################
// RegExMatch - Result of a regex search
// #############################################################################
class RegExMatch : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(RegExMatch, RefCounted)

private:
    std::vector<String> m_groups;
    std::vector<String> m_group_names;
    std::unordered_map<String, int> m_name_to_index;
    String m_subject;
    bool m_valid = false;

public:
    static StringName get_class_static() { return StringName("RegExMatch"); }

    void set_valid(bool valid) { m_valid = valid; }
    bool is_valid() const { return m_valid; }

    void set_subject(const String& subject) { m_subject = subject; }
    String get_subject() const { return m_subject; }

    void set_groups(const std::vector<String>& groups) { m_groups = groups; }

    String get_string(const Variant& name_or_idx) const {
        int idx = get_index(name_or_idx);
        return idx >= 0 && idx < static_cast<int>(m_groups.size()) ? m_groups[idx] : String();
    }

    int get_group_count() const {
        return static_cast<int>(m_groups.size());
    }

    std::vector<String> get_groups() const {
        return m_groups;
    }

    std::vector<String> get_names() const {
        return m_group_names;
    }

    void set_group_names(const std::vector<String>& names) {
        m_group_names = names;
        m_name_to_index.clear();
        for (size_t i = 0; i < names.size(); ++i) {
            if (!names[i].empty()) {
                m_name_to_index[names[i]] = static_cast<int>(i);
            }
        }
    }

    int get_start(int group = 0) const {
        if (group < 0 || group >= static_cast<int>(m_groups.size())) return -1;
        // Simplified: position tracking would require storing offsets
        return -1;
    }

    int get_end(int group = 0) const {
        if (group < 0 || group >= static_cast<int>(m_groups.size())) return -1;
        int start = get_start(group);
        return start >= 0 ? start + m_groups[group].length() : -1;
    }

private:
    int get_index(const Variant& v) const {
        if (v.is_num()) {
            return v.as<int>();
        } else if (v.is_string()) {
            String name = v.as<String>();
            auto it = m_name_to_index.find(name);
            return it != m_name_to_index.end() ? it->second : -1;
        }
        return -1;
    }
};

// #############################################################################
// RegEx - Regular expression engine
// #############################################################################
class RegEx : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(RegEx, RefCounted)

private:
    String m_pattern;
    std::regex m_regex;
    bool m_compiled = false;
    bool m_global = false;
    bool m_case_insensitive = false;
    bool m_multiline = false;
    bool m_dotall = false;
    bool m_unicode = true;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("RegEx"); }

    RegEx() = default;

    explicit RegEx(const String& pattern) : m_pattern(pattern) {
        compile(pattern);
    }

    void set_pattern(const String& pattern) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pattern = pattern;
        compile(pattern);
    }

    String get_pattern() const { return m_pattern; }

    void compile(const String& pattern) {
        m_compiled = false;
        if (pattern.empty()) return;

        try {
            std::regex::flag_type flags = std::regex::ECMAScript;
            if (!m_case_insensitive) flags |= std::regex::icase;
            if (m_multiline) flags |= std::regex::multiline;
            // Note: std::regex doesn't have direct dotall, but ECMAScript mode with '.' matching newline
            // can be achieved with [\s\S] pattern; we ignore dotall flag for simplicity.
            
            m_regex = std::regex(pattern.to_std_string(), flags);
            m_compiled = true;
        } catch (const std::regex_error&) {
            m_compiled = false;
        }
    }

    bool is_valid() const {
        return m_compiled;
    }

    void set_global(bool global) { m_global = global; }
    bool is_global() const { return m_global; }

    void set_case_insensitive(bool enable) {
        m_case_insensitive = enable;
        if (!m_pattern.empty()) compile(m_pattern);
    }
    bool is_case_insensitive() const { return m_case_insensitive; }

    void set_multiline(bool enable) {
        m_multiline = enable;
        if (!m_pattern.empty()) compile(m_pattern);
    }
    bool is_multiline() const { return m_multiline; }

    void set_dotall(bool enable) {
        m_dotall = enable;
        if (!m_pattern.empty()) compile(m_pattern);
    }
    bool is_dotall() const { return m_dotall; }

    void set_unicode(bool enable) {
        m_unicode = enable;
        if (!m_pattern.empty()) compile(m_pattern);
    }
    bool is_unicode() const { return m_unicode; }

    Ref<RegExMatch> search(const String& subject, int offset = 0, int end = -1) const {
        Ref<RegExMatch> match;
        match.instance();
        match->set_subject(subject);

        if (!m_compiled) return match;

        std::string subj = subject.to_std_string();
        if (offset < 0) offset = 0;
        if (offset >= static_cast<int>(subj.length())) return match;
        if (end < 0 || end > static_cast<int>(subj.length())) end = static_cast<int>(subj.length());
        if (offset >= end) return match;

        std::string search_str = subj.substr(offset, end - offset);
        std::smatch sm;
        if (std::regex_search(search_str, sm, m_regex)) {
            match->set_valid(true);
            std::vector<String> groups;
            for (size_t i = 0; i < sm.size(); ++i) {
                groups.push_back(String(sm[i].str().c_str()));
            }
            match->set_groups(groups);
        }

        return match;
    }

    std::vector<Ref<RegExMatch>> search_all(const String& subject, int offset = 0, int end = -1) const {
        std::vector<Ref<RegExMatch>> results;

        if (!m_compiled) return results;

        std::string subj = subject.to_std_string();
        if (offset < 0) offset = 0;
        if (offset >= static_cast<int>(subj.length())) return results;
        if (end < 0 || end > static_cast<int>(subj.length())) end = static_cast<int>(subj.length());

        std::string search_str = subj.substr(offset, end - offset);
        std::sregex_iterator it(search_str.begin(), search_str.end(), m_regex);
        std::sregex_iterator end_it;

        for (; it != end_it; ++it) {
            Ref<RegExMatch> match;
            match.instance();
            match->set_subject(subject);
            match->set_valid(true);
            std::vector<String> groups;
            for (size_t i = 0; i < it->size(); ++i) {
                groups.push_back(String((*it)[i].str().c_str()));
            }
            match->set_groups(groups);
            results.push_back(match);

            if (!m_global) break;
        }

        return results;
    }

    String sub(const String& subject, const String& replacement, bool all = false, int offset = 0, int end = -1) const {
        if (!m_compiled) return subject;

        std::string subj = subject.to_std_string();
        if (offset < 0) offset = 0;
        if (offset >= static_cast<int>(subj.length())) return subject;
        if (end < 0 || end > static_cast<int>(subj.length())) end = static_cast<int>(subj.length());
        if (offset >= end) return subject;

        std::string prefix = subj.substr(0, offset);
        std::string search_str = subj.substr(offset, end - offset);
        std::string suffix = subj.substr(end);

        std::string result;
        if (all) {
            result = std::regex_replace(search_str, m_regex, replacement.to_std_string(),
                                        std::regex_constants::format_default);
        } else {
            result = std::regex_replace(search_str, m_regex, replacement.to_std_string(),
                                        std::regex_constants::format_first_only);
        }

        return String((prefix + result + suffix).c_str());
    }

    void clear() {
        m_pattern.clear();
        m_compiled = false;
    }

    static String escape(const String& str) {
        static const char* special_chars = R"(.^$*+?()[{\|)";
        String result;
        for (char c : str.to_std_string()) {
            if (strchr(special_chars, c)) {
                result += '\\';
            }
            result += String::chr(c);
        }
        return result;
    }
};

} // namespace godot

// Bring into main namespace
using godot::RegEx;
using godot::RegExMatch;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XREGEX_HPP