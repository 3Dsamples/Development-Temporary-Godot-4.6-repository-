// include/xtu/godot/xlocalization.hpp
// xtensor-unified - Localization system for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XLOCALIZATION_HPP
#define XTU_GODOT_XLOCALIZATION_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <regex>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xcore.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class Translation;
class TranslationPO;
class TranslationServer;
class TranslationDomain;

// #############################################################################
// Translation locale structure
// #############################################################################
struct Locale {
    String language;
    String script;
    String country;
    String variant;

    Locale() = default;
    Locale(const String& locale_str) {
        parse(locale_str);
    }

    void parse(const String& str) {
        std::string s = str.to_std_string();
        size_t pos = 0;

        // language
        size_t end = s.find_first_of("-_");
        language = String(s.substr(0, end));
        if (end == std::string::npos) return;
        pos = end + 1;

        // script (if starts with capital)
        end = s.find_first_of("-_", pos);
        std::string part = s.substr(pos, end - pos);
        if (part.size() == 4 && std::isupper(part[0])) {
            script = String(part);
            if (end == std::string::npos) return;
            pos = end + 1;
            end = s.find_first_of("-_", pos);
            part = s.substr(pos, end - pos);
        }

        // country
        if (part.size() == 2 && std::isupper(part[0])) {
            country = String(part);
            if (end == std::string::npos) return;
            pos = end + 1;
            end = s.find_first_of("-_", pos);
            part = s.substr(pos, end - pos);
        }

        // variant
        if (!part.empty()) {
            variant = String(part);
        }
    }

    String to_string() const {
        String result = language;
        if (!script.empty()) result += "-" + script;
        if (!country.empty()) result += "-" + country;
        if (!variant.empty()) result += "-" + variant;
        return result;
    }

    bool matches(const Locale& other) const {
        if (language != other.language) return false;
        if (!script.empty() && !other.script.empty() && script != other.script) return false;
        if (!country.empty() && !other.country.empty() && country != other.country) return false;
        return true;
    }
};

// #############################################################################
// Plural rules for different languages
// #############################################################################
enum class PluralRule : uint8_t {
    PLURAL_ZERO = 0,
    PLURAL_ONE = 1,
    PLURAL_TWO = 2,
    PLURAL_FEW = 3,
    PLURAL_MANY = 4,
    PLURAL_OTHER = 5
};

class PluralRules {
public:
    static PluralRule get_plural_for_number(const String& locale, int n) {
        String lang = Locale(locale).language;
        // Simplified plural rules for common languages
        if (lang == "zh" || lang == "ja" || lang == "ko" || lang == "vi" || lang == "th") {
            return PluralRule::PLURAL_OTHER;
        }
        if (lang == "en" || lang == "de" || lang == "nl" || lang == "sv" || lang == "no" || lang == "da" || lang == "fi" || lang == "it" || lang == "pt" || lang == "es") {
            return (n == 1) ? PluralRule::PLURAL_ONE : PluralRule::PLURAL_OTHER;
        }
        if (lang == "fr") {
            return (n == 0 || n == 1) ? PluralRule::PLURAL_ONE : PluralRule::PLURAL_OTHER;
        }
        if (lang == "ru" || lang == "uk" || lang == "be" || lang == "sr" || lang == "hr") {
            int mod10 = n % 10;
            int mod100 = n % 100;
            if (mod10 == 1 && mod100 != 11) return PluralRule::PLURAL_ONE;
            if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) return PluralRule::PLURAL_FEW;
            return PluralRule::PLURAL_MANY;
        }
        if (lang == "ar") {
            if (n == 0) return PluralRule::PLURAL_ZERO;
            if (n == 1) return PluralRule::PLURAL_ONE;
            if (n == 2) return PluralRule::PLURAL_TWO;
            int mod100 = n % 100;
            if (mod100 >= 3 && mod100 <= 10) return PluralRule::PLURAL_FEW;
            if (mod100 >= 11 && mod100 <= 99) return PluralRule::PLURAL_MANY;
            return PluralRule::PLURAL_OTHER;
        }
        return PluralRule::PLURAL_OTHER;
    }
};

// #############################################################################
// Translation - Base class for all translations
// #############################################################################
class Translation : public Resource {
    XTU_GODOT_REGISTER_CLASS(Translation, Resource)

protected:
    Locale m_locale;
    std::unordered_map<String, String> m_messages;
    std::unordered_map<String, std::unordered_map<String, String>> m_context_messages;

public:
    static StringName get_class_static() { return StringName("Translation"); }

    void set_locale(const String& locale) { m_locale = Locale(locale); }
    String get_locale() const { return m_locale.to_string(); }

    virtual void add_message(const String& src, const String& dst, const String& context = "") {
        if (context.empty()) {
            m_messages[src] = dst;
        } else {
            m_context_messages[context][src] = dst;
        }
    }

    virtual void erase_message(const String& src, const String& context = "") {
        if (context.empty()) {
            m_messages.erase(src);
        } else {
            auto it = m_context_messages.find(context);
            if (it != m_context_messages.end()) {
                it->second.erase(src);
            }
        }
    }

    virtual String get_message(const String& src, const String& context = "") const {
        if (!context.empty()) {
            auto ctx_it = m_context_messages.find(context);
            if (ctx_it != m_context_messages.end()) {
                auto msg_it = ctx_it->second.find(src);
                if (msg_it != ctx_it->second.end()) {
                    return msg_it->second;
                }
            }
        }
        auto it = m_messages.find(src);
        return it != m_messages.end() ? it->second : String();
    }

    virtual bool has_message(const String& src, const String& context = "") const {
        if (!context.empty()) {
            auto ctx_it = m_context_messages.find(context);
            if (ctx_it != m_context_messages.end() && ctx_it->second.find(src) != ctx_it->second.end()) {
                return true;
            }
        }
        return m_messages.find(src) != m_messages.end();
    }

    virtual int get_message_count() const {
        int count = static_cast<int>(m_messages.size());
        for (const auto& kv : m_context_messages) {
            count += static_cast<int>(kv.second.size());
        }
        return count;
    }

    virtual std::vector<String> get_message_list() const {
        std::vector<String> result;
        for (const auto& kv : m_messages) {
            result.push_back(kv.first);
        }
        for (const auto& ctx : m_context_messages) {
            for (const auto& kv : ctx.second) {
                result.push_back(ctx.first + "::" + kv.first);
            }
        }
        return result;
    }
};

// #############################################################################
// TranslationPO - PO file loader (GNU gettext format)
// #############################################################################
class TranslationPO : public Translation {
    XTU_GODOT_REGISTER_CLASS(TranslationPO, Translation)

public:
    enum ParseState : uint8_t {
        STATE_NONE = 0,
        STATE_MSGID = 1,
        STATE_MSGSTR = 2,
        STATE_MSGCTXT = 3,
        STATE_MSGID_PLURAL = 4,
        STATE_MSGSTR_PLURAL = 5
    };

    static StringName get_class_static() { return StringName("TranslationPO"); }

    Error load_from_file(const String& path) {
        Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
        if (!file.is_valid()) return ERR_FILE_CANT_OPEN;

        String content = file->get_as_text();
        return parse_po_content(content);
    }

    Error parse_po_content(const String& content) {
        std::vector<String> lines = content.split("\n");
        ParseState state = STATE_NONE;
        String msgid, msgstr, msgctxt;
        std::vector<String> msgid_plural;
        std::vector<String> msgstr_plural;
        int plural_index = -1;

        for (const String& line : lines) {
            String trimmed = line.strip_edges();
            if (trimmed.empty() || trimmed[0] == '#') continue;

            if (trimmed.begins_with("msgctxt ")) {
                flush_current(state, msgid, msgstr, msgctxt, msgid_plural, msgstr_plural);
                state = STATE_MSGCTXT;
                msgctxt = parse_quoted_string(trimmed.substr(8));
            } else if (trimmed.begins_with("msgid ")) {
                flush_current(state, msgid, msgstr, msgctxt, msgid_plural, msgstr_plural);
                state = STATE_MSGID;
                msgid = parse_quoted_string(trimmed.substr(6));
            } else if (trimmed.begins_with("msgid_plural ")) {
                state = STATE_MSGID_PLURAL;
                msgid_plural.push_back(parse_quoted_string(trimmed.substr(13)));
            } else if (trimmed.begins_with("msgstr ")) {
                state = STATE_MSGSTR;
                msgstr = parse_quoted_string(trimmed.substr(7));
            } else if (trimmed.begins_with("msgstr[")) {
                state = STATE_MSGSTR_PLURAL;
                int idx = trimmed.substr(7).to_int();
                msgstr_plural.resize(std::max(msgstr_plural.size(), static_cast<size_t>(idx + 1)));
                size_t bracket = trimmed.find("]");
                if (bracket != String::npos) {
                    msgstr_plural[idx] = parse_quoted_string(trimmed.substr(bracket + 1));
                }
                plural_index = idx;
            } else if (trimmed[0] == '"') {
                // Continuation line
                String value = parse_quoted_string(trimmed);
                switch (state) {
                    case STATE_MSGID: msgid += value; break;
                    case STATE_MSGSTR: msgstr += value; break;
                    case STATE_MSGCTXT: msgctxt += value; break;
                    case STATE_MSGID_PLURAL: msgid_plural.back() += value; break;
                    case STATE_MSGSTR_PLURAL:
                        if (plural_index >= 0 && plural_index < static_cast<int>(msgstr_plural.size())) {
                            msgstr_plural[plural_index] += value;
                        }
                        break;
                    default: break;
                }
            }
        }
        flush_current(state, msgid, msgstr, msgctxt, msgid_plural, msgstr_plural);
        return OK;
    }

private:
    String parse_quoted_string(const String& str) {
        String result;
        bool in_escape = false;
        for (size_t i = 0; i < str.length(); ++i) {
            char c = str.utf8()[i];
            if (c == '"') {
                continue;
            }
            if (in_escape) {
                switch (c) {
                    case 'n': result += "\n"; break;
                    case 'r': result += "\r"; break;
                    case 't': result += "\t"; break;
                    case '\\': result += "\\"; break;
                    case '"': result += "\""; break;
                    default: result += String::chr(c); break;
                }
                in_escape = false;
            } else if (c == '\\') {
                in_escape = true;
            } else {
                result += String::chr(c);
            }
        }
        return result;
    }

    void flush_current(ParseState state, String& msgid, String& msgstr, String& msgctxt,
                       std::vector<String>& msgid_plural, std::vector<String>& msgstr_plural) {
        if (!msgid.empty()) {
            if (!msgid_plural.empty()) {
                // Store plural forms
                for (size_t i = 0; i < msgid_plural.size() && i < msgstr_plural.size(); ++i) {
                    String key = msgid + "|plural_" + String::num(static_cast<int64_t>(i));
                    add_message(key, msgstr_plural[i], msgctxt);
                }
            } else {
                add_message(msgid, msgstr, msgctxt);
            }
        }
        msgid.clear();
        msgstr.clear();
        msgctxt.clear();
        msgid_plural.clear();
        msgstr_plural.clear();
    }
};

// #############################################################################
// TranslationDomain - Context-specific translation
// #############################################################################
class TranslationDomain : public Resource {
    XTU_GODOT_REGISTER_CLASS(TranslationDomain, Resource)

private:
    String m_domain_name;
    std::unordered_map<String, Ref<Translation>> m_translations;

public:
    static StringName get_class_static() { return StringName("TranslationDomain"); }

    void set_domain_name(const String& name) { m_domain_name = name; }
    String get_domain_name() const { return m_domain_name; }

    void add_translation(const Ref<Translation>& translation) {
        m_translations[translation->get_locale()] = translation;
    }

    void remove_translation(const String& locale) {
        m_translations.erase(locale);
    }

    Ref<Translation> get_translation(const String& locale) const {
        auto it = m_translations.find(locale);
        return it != m_translations.end() ? it->second : Ref<Translation>();
    }

    String translate(const String& msgid, const String& context = "", const String& locale = "") const {
        String loc = locale;
        if (loc.empty()) {
            loc = TranslationServer::get_singleton()->get_locale();
        }
        Ref<Translation> trans = get_translation(loc);
        if (!trans.is_valid()) {
            // Try fallback locale
            Locale target(loc);
            for (const auto& kv : m_translations) {
                Locale candidate(kv.first);
                if (candidate.language == target.language) {
                    trans = kv.second;
                    break;
                }
            }
        }
        if (trans.is_valid()) {
            String result = trans->get_message(msgid, context);
            if (!result.empty()) return result;
        }
        return msgid;
    }
};

// #############################################################################
// TranslationServer - Global translation manager singleton
// #############################################################################
class TranslationServer : public Object {
    XTU_GODOT_REGISTER_CLASS(TranslationServer, Object)

private:
    static TranslationServer* s_singleton;
    String m_locale = "en";
    String m_fallback_locale = "en";
    std::unordered_map<String, Ref<TranslationDomain>> m_domains;
    Ref<TranslationDomain> m_default_domain;
    bool m_pseudo_localization = false;
    bool m_use_fuzzy = true;
    std::mutex m_mutex;

public:
    static TranslationServer* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("TranslationServer"); }

    TranslationServer() {
        s_singleton = this;
        m_default_domain.instance();
        m_default_domain->set_domain_name("default");
    }

    ~TranslationServer() { s_singleton = nullptr; }

    void set_locale(const String& locale) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_locale = locale;
    }

    String get_locale() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_locale;
    }

    void set_fallback_locale(const String& locale) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_fallback_locale = locale;
    }

    String get_fallback_locale() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_fallback_locale;
    }

    void add_translation(const Ref<Translation>& translation, const String& domain = "") {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (domain.empty()) {
            m_default_domain->add_translation(translation);
        } else {
            auto it = m_domains.find(domain);
            if (it == m_domains.end()) {
                Ref<TranslationDomain> dom;
                dom.instance();
                dom->set_domain_name(domain);
                m_domains[domain] = dom;
                it = m_domains.find(domain);
            }
            it->second->add_translation(translation);
        }
    }

    void remove_translation(const Ref<Translation>& translation) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_default_domain->remove_translation(translation->get_locale());
        for (auto& kv : m_domains) {
            kv.second->remove_translation(translation->get_locale());
        }
    }

    String translate(const String& msgid, const String& context = "", const String& domain = "") const {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Pseudo-localization for testing
        if (m_pseudo_localization) {
            return pseudo_localize(msgid);
        }

        String result;
        if (domain.empty()) {
            result = m_default_domain->translate(msgid, context, m_locale);
            if (result == msgid && m_locale != m_fallback_locale) {
                result = m_default_domain->translate(msgid, context, m_fallback_locale);
            }
        } else {
            auto it = m_domains.find(domain);
            if (it != m_domains.end()) {
                result = it->second->translate(msgid, context, m_locale);
                if (result == msgid && m_locale != m_fallback_locale) {
                    result = it->second->translate(msgid, context, m_fallback_locale);
                }
            } else {
                result = msgid;
            }
        }

        return result;
    }

    String translate_plural(const String& singular, const String& plural, int n,
                            const String& context = "", const String& domain = "") const {
        PluralRule rule = PluralRules::get_plural_for_number(m_locale, n);
        String key;
        if (rule == PluralRule::PLURAL_ONE) {
            return translate(singular, context, domain);
        }
        key = singular + "|plural_" + String::num(static_cast<int64_t>(static_cast<int>(rule)));
        String translated = translate(key, context, domain);
        if (translated == key) {
            return translate(plural, context, domain);
        }
        return translated;
    }

    void set_pseudo_localization(bool enabled) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pseudo_localization = enabled;
    }

    bool is_pseudo_localization_enabled() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_pseudo_localization;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_default_domain = Ref<TranslationDomain>();
        m_default_domain.instance();
        m_default_domain->set_domain_name("default");
        m_domains.clear();
    }

    void reload() {
        // Reload all translation files
    }

    std::vector<String> get_loaded_locales() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::unordered_set<String> locales;
        for (const auto& kv : m_domains) {
            // Collect locales
        }
        return std::vector<String>(locales.begin(), locales.end());
    }

private:
    String pseudo_localize(const String& text) const {
        // Add accents and extend text for testing
        std::string s = text.to_std_string();
        std::string result;
        for (char c : s) {
            if (c >= 'a' && c <= 'z') {
                result += static_cast<char>(c + 0x80);
            } else if (c >= 'A' && c <= 'Z') {
                result += static_cast<char>(c + 0x80);
            } else {
                result += c;
            }
        }
        return String("[" + result + "]");
    }
};

// #############################################################################
// Translation helper macros and functions
// #############################################################################
inline String tr(const String& msgid, const String& context = "", const String& domain = "") {
    return TranslationServer::get_singleton()->translate(msgid, context, domain);
}

inline String tr_n(const String& singular, const String& plural, int n,
                   const String& context = "", const String& domain = "") {
    return TranslationServer::get_singleton()->translate_plural(singular, plural, n, context, domain);
}

#define XTU_TR(msgid) ::xtu::godot::tr(msgid)
#define XTU_TRC(msgid, context) ::xtu::godot::tr(msgid, context)
#define XTU_TRD(msgid, domain) ::xtu::godot::tr(msgid, "", domain)
#define XTU_TRN(singular, plural, n) ::xtu::godot::tr_n(singular, plural, n)

} // namespace godot

// Bring into main namespace
using godot::Translation;
using godot::TranslationPO;
using godot::TranslationServer;
using godot::TranslationDomain;
using godot::Locale;
using godot::PluralRule;
using godot::PluralRules;
using godot::tr;
using godot::tr_n;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XLOCALIZATION_HPP