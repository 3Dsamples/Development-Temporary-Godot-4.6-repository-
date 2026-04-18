// include/xtu/godot/xxml.hpp
// xtensor-unified - XML Parser and DOM utilities for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XXML_HPP
#define XTU_GODOT_XXML_HPP

#include <cctype>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xcore.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace xml {

// #############################################################################
// Forward declarations
// #############################################################################
class XMLParser;
class XMLNode;
class XMLDocument;
class XMLWriter;

// #############################################################################
// XML node types
// #############################################################################
enum class XMLNodeType : uint8_t {
    NODE_NONE = 0,
    NODE_ELEMENT = 1,
    NODE_TEXT = 2,
    NODE_COMMENT = 3,
    NODE_CDATA = 4,
    NODE_PROCESSING_INSTRUCTION = 5,
    NODE_DOCUMENT = 6
};

// #############################################################################
// XML parser error types
// #############################################################################
enum class XMLError : uint8_t {
    ERR_NONE = 0,
    ERR_UNEXPECTED_EOF = 1,
    ERR_INVALID_CHARACTER = 2,
    ERR_UNCLOSED_TAG = 3,
    ERR_MISMATCHED_TAG = 4,
    ERR_DUPLICATE_ATTRIBUTE = 5,
    ERR_INVALID_NAME = 6,
    ERR_INVALID_ENTITY = 7,
    ERR_UNKNOWN = 8
};

// #############################################################################
// XML attribute
// #############################################################################
struct XMLAttribute {
    String name;
    String value;
    String namespace_uri;
    String prefix;
    
    XMLAttribute() = default;
    XMLAttribute(const String& n, const String& v) : name(n), value(v) {}
    XMLAttribute(const String& n, const String& v, const String& ns, const String& p)
        : name(n), value(v), namespace_uri(ns), prefix(p) {}
};

// #############################################################################
// XMLNode - DOM node
// #############################################################################
class XMLNode : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(XMLNode, RefCounted)

private:
    XMLNodeType m_type = XMLNodeType::NODE_NONE;
    String m_name;
    String m_value;
    std::vector<XMLAttribute> m_attributes;
    std::unordered_map<String, size_t> m_attribute_index;
    std::vector<Ref<XMLNode>> m_children;
    XMLNode* m_parent = nullptr;

public:
    static StringName get_class_static() { return StringName("XMLNode"); }

    void set_type(XMLNodeType type) { m_type = type; }
    XMLNodeType get_type() const { return m_type; }

    void set_name(const String& name) { m_name = name; }
    String get_name() const { return m_name; }

    void set_value(const String& value) { m_value = value; }
    String get_value() const { return m_value; }

    void set_attribute(const String& name, const String& value) {
        auto it = m_attribute_index.find(name);
        if (it != m_attribute_index.end()) {
            m_attributes[it->second].value = value;
        } else {
            m_attribute_index[name] = m_attributes.size();
            m_attributes.emplace_back(name, value);
        }
    }

    String get_attribute(const String& name, const String& default_val = "") const {
        auto it = m_attribute_index.find(name);
        return it != m_attribute_index.end() ? m_attributes[it->second].value : default_val;
    }

    bool has_attribute(const String& name) const {
        return m_attribute_index.find(name) != m_attribute_index.end();
    }

    const std::vector<XMLAttribute>& get_attributes() const { return m_attributes; }
    size_t get_attribute_count() const { return m_attributes.size(); }

    void add_child(const Ref<XMLNode>& child) {
        child->m_parent = this;
        m_children.push_back(child);
    }

    Ref<XMLNode> get_child(size_t idx) const {
        return idx < m_children.size() ? m_children[idx] : Ref<XMLNode>();
    }

    size_t get_child_count() const { return m_children.size(); }

    std::vector<Ref<XMLNode>> get_children_by_name(const String& name) const {
        std::vector<Ref<XMLNode>> result;
        for (const auto& child : m_children) {
            if (child->m_type == XMLNodeType::NODE_ELEMENT && child->m_name == name) {
                result.push_back(child);
            }
        }
        return result;
    }

    Ref<XMLNode> get_first_child_by_name(const String& name) const {
        for (const auto& child : m_children) {
            if (child->m_type == XMLNodeType::NODE_ELEMENT && child->m_name == name) {
                return child;
            }
        }
        return Ref<XMLNode>();
    }

    String get_text() const {
        String result;
        for (const auto& child : m_children) {
            if (child->m_type == XMLNodeType::NODE_TEXT || child->m_type == XMLNodeType::NODE_CDATA) {
                result += child->m_value;
            }
        }
        return result;
    }

    XMLNode* get_parent() const { return m_parent; }

    void remove_child(size_t idx) {
        if (idx < m_children.size()) {
            m_children.erase(m_children.begin() + idx);
        }
    }

    String to_string(int indent = 0, bool indent_attributes = false) const {
        String result;
        String indent_str = String(" ").repeat(indent * 2);

        switch (m_type) {
            case XMLNodeType::NODE_DOCUMENT:
                for (const auto& child : m_children) {
                    result += child->to_string(indent);
                }
                break;
            case XMLNodeType::NODE_ELEMENT:
                result += indent_str + "<" + m_name;
                for (const auto& attr : m_attributes) {
                    result += " " + attr.name + "=\"" + escape_xml(attr.value) + "\"";
                }
                if (m_children.empty()) {
                    result += " />\n";
                } else {
                    result += ">";
                    if (m_children.size() == 1 && 
                        (m_children[0]->m_type == XMLNodeType::NODE_TEXT ||
                         m_children[0]->m_type == XMLNodeType::NODE_CDATA)) {
                        result += m_children[0]->to_string(0);
                    } else {
                        result += "\n";
                        for (const auto& child : m_children) {
                            result += child->to_string(indent + 1);
                        }
                        result += indent_str;
                    }
                    result += "</" + m_name + ">\n";
                }
                break;
            case XMLNodeType::NODE_TEXT:
                result += indent_str + escape_xml(m_value) + "\n";
                break;
            case XMLNodeType::NODE_CDATA:
                result += indent_str + "<![CDATA[" + m_value + "]]>\n";
                break;
            case XMLNodeType::NODE_COMMENT:
                result += indent_str + "<!--" + m_value + "-->\n";
                break;
            case XMLNodeType::NODE_PROCESSING_INSTRUCTION:
                result += indent_str + "<?" + m_name + " " + m_value + "?>\n";
                break;
            default:
                break;
        }
        return result;
    }

private:
    static String escape_xml(const String& str) {
        String result;
        for (char c : str.to_std_string()) {
            switch (c) {
                case '<': result += "&lt;"; break;
                case '>': result += "&gt;"; break;
                case '&': result += "&amp;"; break;
                case '"': result += "&quot;"; break;
                case '\'': result += "&apos;"; break;
                default: result += String::chr(c); break;
            }
        }
        return result;
    }
};

// #############################################################################
// XMLParser - SAX style XML parser
// #############################################################################
class XMLParser : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(XMLParser, RefCounted)

public:
    using NodeHandler = std::function<void(const String&, const std::vector<XMLAttribute>&)>;
    using TextHandler = std::function<void(const String&)>;
    using EndHandler = std::function<void(const String&)>;

private:
    String m_data;
    size_t m_pos = 0;
    size_t m_line = 1;
    size_t m_column = 1;
    XMLError m_error = XMLError::ERR_NONE;
    String m_error_message;
    
    NodeHandler m_node_open_handler;
    NodeHandler m_node_close_handler;
    TextHandler m_text_handler;
    EndHandler m_end_handler;
    
    std::stack<String> m_tag_stack;

public:
    static StringName get_class_static() { return StringName("XMLParser"); }

    void set_data(const String& data) {
        m_data = data;
        m_pos = 0;
        m_line = 1;
        m_column = 1;
        m_error = XMLError::ERR_NONE;
        m_error_message.clear();
        while (!m_tag_stack.empty()) m_tag_stack.pop();
    }

    XMLError parse() {
        if (m_data.empty()) return XMLError::ERR_NONE;
        
        skip_whitespace();
        while (m_pos < m_data.length() && m_error == XMLError::ERR_NONE) {
            if (m_data[m_pos] == '<') {
                if (m_pos + 1 < m_data.length()) {
                    char next = m_data[m_pos + 1];
                    if (next == '/') {
                        parse_closing_tag();
                    } else if (next == '?') {
                        parse_processing_instruction();
                    } else if (next == '!') {
                        parse_comment_or_cdata();
                    } else {
                        parse_opening_tag();
                    }
                } else {
                    m_error = XMLError::ERR_UNEXPECTED_EOF;
                }
            } else {
                parse_text();
            }
            skip_whitespace();
        }
        
        if (m_error == XMLError::ERR_NONE && !m_tag_stack.empty()) {
            m_error = XMLError::ERR_UNCLOSED_TAG;
            m_error_message = "Unclosed tag: " + m_tag_stack.top();
        }
        
        return m_error;
    }

    Ref<XMLNode> parse_dom() {
        Ref<XMLNode> doc;
        doc.instance();
        doc->set_type(XMLNodeType::NODE_DOCUMENT);
        
        std::stack<Ref<XMLNode>> node_stack;
        node_stack.push(doc);
        
        m_node_open_handler = [&](const String& name, const std::vector<XMLAttribute>& attrs) {
            Ref<XMLNode> node;
            node.instance();
            node->set_type(XMLNodeType::NODE_ELEMENT);
            node->set_name(name);
            for (const auto& attr : attrs) {
                node->set_attribute(attr.name, attr.value);
            }
            node_stack.top()->add_child(node);
            node_stack.push(node);
        };
        
        m_node_close_handler = [&](const String&) {
            if (node_stack.size() > 1) {
                node_stack.pop();
            }
        };
        
        m_text_handler = [&](const String& text) {
            Ref<XMLNode> node;
            node.instance();
            node->set_type(XMLNodeType::NODE_TEXT);
            node->set_value(text);
            node_stack.top()->add_child(node);
        };
        
        if (parse() == XMLError::ERR_NONE) {
            return doc;
        }
        return Ref<XMLNode>();
    }

    void set_node_open_handler(NodeHandler handler) { m_node_open_handler = handler; }
    void set_node_close_handler(NodeHandler handler) { m_node_close_handler = handler; }
    void set_text_handler(TextHandler handler) { m_text_handler = handler; }

    XMLError get_error() const { return m_error; }
    String get_error_message() const { return m_error_message; }
    size_t get_error_line() const { return m_line; }
    size_t get_error_column() const { return m_column; }

private:
    void advance() {
        if (m_pos < m_data.length()) {
            if (m_data[m_pos] == '\n') {
                ++m_line;
                m_column = 1;
            } else {
                ++m_column;
            }
            ++m_pos;
        }
    }

    char current() const {
        return m_pos < m_data.length() ? m_data[m_pos] : '\0';
    }

    void skip_whitespace() {
        while (m_pos < m_data.length() && std::isspace(static_cast<unsigned char>(m_data[m_pos]))) {
            advance();
        }
    }

    String parse_name() {
        String result;
        while (m_pos < m_data.length() && 
               (std::isalnum(static_cast<unsigned char>(m_data[m_pos])) || 
                m_data[m_pos] == '_' || m_data[m_pos] == '-' || 
                m_data[m_pos] == ':' || m_data[m_pos] == '.')) {
            result += String::chr(m_data[m_pos]);
            advance();
        }
        return result;
    }

    String parse_until(char delim) {
        String result;
        while (m_pos < m_data.length() && m_data[m_pos] != delim) {
            result += String::chr(m_data[m_pos]);
            advance();
        }
        if (m_pos < m_data.length() && m_data[m_pos] == delim) {
            advance();
        }
        return result;
    }

    String parse_quoted_string() {
        char quote = m_data[m_pos];
        advance(); // Skip opening quote
        String result;
        while (m_pos < m_data.length() && m_data[m_pos] != quote) {
            if (m_data[m_pos] == '&') {
                result += parse_entity();
            } else {
                result += String::chr(m_data[m_pos]);
                advance();
            }
        }
        if (m_pos < m_data.length() && m_data[m_pos] == quote) {
            advance();
        }
        return result;
    }

    String parse_entity() {
        if (m_data[m_pos] != '&') return "";
        advance();
        String entity = parse_until(';');
        if (entity == "lt") return "<";
        if (entity == "gt") return ">";
        if (entity == "amp") return "&";
        if (entity == "quot") return "\"";
        if (entity == "apos") return "'";
        if (entity[0] == '#') {
            int code = 0;
            if (entity[1] == 'x') {
                code = std::stoi(entity.substr(2).to_std_string(), nullptr, 16);
            } else {
                code = std::stoi(entity.substr(1).to_std_string());
            }
            return String::chr(code);
        }
        return "&" + entity + ";";
    }

    void parse_opening_tag() {
        advance(); // Skip '<'
        String name = parse_name();
        std::vector<XMLAttribute> attributes;
        
        while (true) {
            skip_whitespace();
            char c = current();
            if (c == '>') {
                advance();
                break;
            } else if (c == '/') {
                advance();
                if (current() == '>') {
                    advance();
                    if (m_node_open_handler) {
                        m_node_open_handler(name, attributes);
                    }
                    if (m_node_close_handler) {
                        m_node_close_handler(name);
                    }
                    return;
                }
            } else {
                String attr_name = parse_name();
                skip_whitespace();
                if (current() == '=') {
                    advance();
                    skip_whitespace();
                    String attr_value = parse_quoted_string();
                    attributes.emplace_back(attr_name, attr_value);
                }
            }
        }
        
        m_tag_stack.push(name);
        if (m_node_open_handler) {
            m_node_open_handler(name, attributes);
        }
    }

    void parse_closing_tag() {
        advance(); // Skip '<'
        advance(); // Skip '/'
        String name = parse_name();
        skip_whitespace();
        if (current() == '>') {
            advance();
        }
        
        if (m_tag_stack.empty() || m_tag_stack.top() != name) {
            m_error = XMLError::ERR_MISMATCHED_TAG;
            m_error_message = "Expected </" + (m_tag_stack.empty() ? String("") : m_tag_stack.top()) + 
                              "> but found </" + name + ">";
            return;
        }
        
        m_tag_stack.pop();
        if (m_node_close_handler) {
            m_node_close_handler(name);
        }
    }

    void parse_text() {
        String text;
        while (m_pos < m_data.length() && m_data[m_pos] != '<') {
            if (m_data[m_pos] == '&') {
                text += parse_entity();
            } else {
                text += String::chr(m_data[m_pos]);
                advance();
            }
        }
        if (!text.empty() && m_text_handler) {
            m_text_handler(text);
        }
    }

    void parse_comment_or_cdata() {
        advance(); // Skip '<'
        advance(); // Skip '!'
        
        if (m_pos + 1 < m_data.length() && m_data[m_pos] == '-' && m_data[m_pos+1] == '-') {
            advance(); advance(); // Skip '--'
            String comment;
            while (m_pos + 2 < m_data.length() && 
                   !(m_data[m_pos] == '-' && m_data[m_pos+1] == '-' && m_data[m_pos+2] == '>')) {
                comment += String::chr(m_data[m_pos]);
                advance();
            }
            if (m_pos + 2 < m_data.length()) {
                advance(); advance(); advance(); // Skip '-->'
            }
        } else if (m_pos + 7 < m_data.length() && 
                   m_data.substr(m_pos, 7) == "[CDATA[") {
            for (int i = 0; i < 7; ++i) advance();
            String cdata;
            while (m_pos + 2 < m_data.length() && 
                   !(m_data[m_pos] == ']' && m_data[m_pos+1] == ']' && m_data[m_pos+2] == '>')) {
                cdata += String::chr(m_data[m_pos]);
                advance();
            }
            if (m_pos + 2 < m_data.length()) {
                advance(); advance(); advance(); // Skip ']]>'
            }
            if (m_text_handler) {
                m_text_handler(cdata);
            }
        }
    }

    void parse_processing_instruction() {
        advance(); advance(); // Skip '<?'
        String target = parse_name();
        skip_whitespace();
        String content;
        while (m_pos + 1 < m_data.length() && 
               !(m_data[m_pos] == '?' && m_data[m_pos+1] == '>')) {
            content += String::chr(m_data[m_pos]);
            advance();
        }
        if (m_pos + 1 < m_data.length()) {
            advance(); advance(); // Skip '?>'
        }
    }
};

// #############################################################################
// XMLWriter - Streaming XML output
// #############################################################################
class XMLWriter : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(XMLWriter, RefCounted)

private:
    std::ostringstream m_output;
    std::stack<String> m_tag_stack;
    bool m_tag_open = false;
    int m_indent = 0;
    bool m_pretty = true;
    String m_indent_str = "  ";

public:
    static StringName get_class_static() { return StringName("XMLWriter"); }

    void set_pretty(bool pretty) { m_pretty = pretty; }

    void start_document(const String& version = "1.0", const String& encoding = "UTF-8") {
        m_output << "<?xml version=\"" << version.to_std_string() 
                 << "\" encoding=\"" << encoding.to_std_string() << "\"?>";
        if (m_pretty) m_output << "\n";
    }

    void start_element(const String& name) {
        close_start_tag();
        if (m_pretty && !m_tag_stack.empty()) {
            m_output << std::string(m_indent * m_indent_str.length(), ' ');
        }
        m_output << "<" << name.to_std_string();
        m_tag_open = true;
        m_tag_stack.push(name);
        ++m_indent;
    }

    void end_element() {
        --m_indent;
        if (m_tag_open) {
            m_output << " />";
            m_tag_open = false;
        } else {
            if (m_pretty && !m_tag_stack.empty()) {
                m_output << std::string(m_indent * m_indent_str.length(), ' ');
            }
            String name = m_tag_stack.top();
            m_output << "</" << name.to_std_string() << ">";
        }
        m_tag_stack.pop();
        if (m_pretty) m_output << "\n";
    }

    void write_attribute(const String& name, const String& value) {
        if (m_tag_open) {
            m_output << " " << name.to_std_string() << "=\"" << escape_xml(value).to_std_string() << "\"";
        }
    }

    void write_text(const String& text) {
        close_start_tag();
        m_output << escape_xml(text).to_std_string();
    }

    void write_cdata(const String& data) {
        close_start_tag();
        m_output << "<![CDATA[" << data.to_std_string() << "]]>";
    }

    void write_comment(const String& comment) {
        close_start_tag();
        if (m_pretty) {
            m_output << std::string(m_indent * m_indent_str.length(), ' ');
        }
        m_output << "<!--" << comment.to_std_string() << "-->";
        if (m_pretty) m_output << "\n";
    }

    void write_raw(const String& xml) {
        close_start_tag();
        m_output << xml.to_std_string();
    }

    String to_string() const {
        return String(m_output.str().c_str());
    }

    void clear() {
        m_output.str("");
        m_output.clear();
        while (!m_tag_stack.empty()) m_tag_stack.pop();
        m_tag_open = false;
        m_indent = 0;
    }

private:
    void close_start_tag() {
        if (m_tag_open) {
            m_output << ">";
            m_tag_open = false;
            if (m_pretty) m_output << "\n";
        }
    }

    static String escape_xml(const String& str) {
        String result;
        for (char c : str.to_std_string()) {
            switch (c) {
                case '<': result += "&lt;"; break;
                case '>': result += "&gt;"; break;
                case '&': result += "&amp;"; break;
                case '"': result += "&quot;"; break;
                case '\'': result += "&apos;"; break;
                default: result += String::chr(c); break;
            }
        }
        return result;
    }
};

} // namespace xml

// Bring into main namespace
using xml::XMLParser;
using xml::XMLNode;
using xml::XMLWriter;
using xml::XMLNodeType;
using xml::XMLError;
using xml::XMLAttribute;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XXML_HPP