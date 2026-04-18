// io/xio_xml.hpp

#ifndef XTENSOR_XIO_XML_HPP
#define XTENSOR_XIO_XML_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <complex>
#include <functional>
#include <variant>
#include <optional>
#include <cctype>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace io
        {
            // --------------------------------------------------------------------
            // Simple XML Node representation (DOM-like)
            // --------------------------------------------------------------------
            class XmlNode
            {
            public:
                enum class Type
                {
                    Element,
                    Text,
                    Comment,
                    Declaration,
                    Document
                };

                using Attributes = std::map<std::string, std::string>;

                XmlNode() : m_type(Type::Element) {}
                explicit XmlNode(Type type) : m_type(type) {}
                explicit XmlNode(const std::string& name, Type type = Type::Element)
                    : m_name(name), m_type(type) {}

                // Type queries
                Type type() const { return m_type; }
                bool is_element() const { return m_type == Type::Element; }
                bool is_text() const { return m_type == Type::Text; }
                bool is_comment() const { return m_type == Type::Comment; }

                // Name (for elements)
                const std::string& name() const { return m_name; }
                void set_name(const std::string& name) { m_name = name; }

                // Text content (for text nodes)
                const std::string& text() const { return m_text; }
                void set_text(const std::string& text) { m_text = text; }

                // Attributes
                const Attributes& attributes() const { return m_attrs; }
                Attributes& attributes() { return m_attrs; }
                bool has_attribute(const std::string& name) const
                {
                    return m_attrs.find(name) != m_attrs.end();
                }
                std::string get_attribute(const std::string& name, const std::string& def = "") const
                {
                    auto it = m_attrs.find(name);
                    return it != m_attrs.end() ? it->second : def;
                }
                void set_attribute(const std::string& name, const std::string& value)
                {
                    m_attrs[name] = value;
                }

                // Children
                const std::vector<std::unique_ptr<XmlNode>>& children() const { return m_children; }
                std::vector<std::unique_ptr<XmlNode>>& children() { return m_children; }

                XmlNode* add_child(std::unique_ptr<XmlNode> child)
                {
                    m_children.push_back(std::move(child));
                    return m_children.back().get();
                }

                XmlNode* add_element(const std::string& name)
                {
                    return add_child(std::make_unique<XmlNode>(name, Type::Element));
                }

                XmlNode* add_text(const std::string& text)
                {
                    auto node = std::make_unique<XmlNode>(Type::Text);
                    node->set_text(text);
                    return add_child(std::move(node));
                }

                // Find first child element by name
                XmlNode* find_child(const std::string& name)
                {
                    for (auto& child : m_children)
                    {
                        if (child->is_element() && child->name() == name)
                            return child.get();
                    }
                    return nullptr;
                }

                const XmlNode* find_child(const std::string& name) const
                {
                    return const_cast<XmlNode*>(this)->find_child(name);
                }

                // Find all child elements by name
                std::vector<XmlNode*> find_children(const std::string& name)
                {
                    std::vector<XmlNode*> result;
                    for (auto& child : m_children)
                        if (child->is_element() && child->name() == name)
                            result.push_back(child.get());
                    return result;
                }

                // Serialize to string
                std::string to_string(bool pretty = false, int indent = 0) const
                {
                    std::ostringstream oss;
                    write(oss, pretty, indent);
                    return oss.str();
                }

                // Write to stream
                void write(std::ostream& os, bool pretty = false, int indent = 0) const
                {
                    std::string indent_str;
                    if (pretty)
                        indent_str = std::string(static_cast<size_t>(indent), ' ');
                    switch (m_type)
                    {
                        case Type::Document:
                            for (const auto& child : m_children)
                                child->write(os, pretty, indent);
                            break;
                        case Type::Declaration:
                            os << "<?xml";
                            for (const auto& attr : m_attrs)
                                os << " " << attr.first << "=\"" << escape(attr.second) << "\"";
                            os << "?>";
                            if (pretty) os << '\n';
                            break;
                        case Type::Element:
                            os << indent_str << "<" << m_name;
                            for (const auto& attr : m_attrs)
                                os << " " << attr.first << "=\"" << escape(attr.second) << "\"";
                            if (m_children.empty())
                            {
                                os << "/>";
                                if (pretty) os << '\n';
                            }
                            else
                            {
                                os << ">";
                                if (pretty) os << '\n';
                                bool has_text_only = true;
                                for (const auto& child : m_children)
                                    if (!child->is_text()) { has_text_only = false; break; }
                                if (has_text_only)
                                {
                                    for (const auto& child : m_children)
                                        os << child->text();
                                    os << "</" << m_name << ">";
                                    if (pretty) os << '\n';
                                }
                                else
                                {
                                    for (const auto& child : m_children)
                                        child->write(os, pretty, indent + 2);
                                    os << indent_str << "</" << m_name << ">";
                                    if (pretty) os << '\n';
                                }
                            }
                            break;
                        case Type::Text:
                            os << indent_str << escape(m_text);
                            if (pretty) os << '\n';
                            break;
                        case Type::Comment:
                            os << indent_str << "<!-- " << m_text << " -->";
                            if (pretty) os << '\n';
                            break;
                        default:
                            break;
                    }
                }

                static std::unique_ptr<XmlNode> create_document()
                {
                    return std::make_unique<XmlNode>(Type::Document);
                }

            private:
                std::string m_name;
                std::string m_text;
                Type m_type;
                Attributes m_attrs;
                std::vector<std::unique_ptr<XmlNode>> m_children;

                static std::string escape(const std::string& s)
                {
                    std::string result;
                    result.reserve(s.size());
                    for (char c : s)
                    {
                        switch (c)
                        {
                            case '&': result += "&amp;"; break;
                            case '<': result += "&lt;"; break;
                            case '>': result += "&gt;"; break;
                            case '"': result += "&quot;"; break;
                            case '\'': result += "&apos;"; break;
                            default: result += c; break;
                        }
                    }
                    return result;
                }

                static std::string unescape(const std::string& s)
                {
                    std::string result;
                    size_t i = 0;
                    while (i < s.size())
                    {
                        if (s[i] == '&')
                        {
                            if (s.compare(i, 5, "&amp;") == 0) { result += '&'; i += 5; }
                            else if (s.compare(i, 4, "&lt;") == 0) { result += '<'; i += 4; }
                            else if (s.compare(i, 4, "&gt;") == 0) { result += '>'; i += 4; }
                            else if (s.compare(i, 6, "&quot;") == 0) { result += '"'; i += 6; }
                            else if (s.compare(i, 6, "&apos;") == 0) { result += '\''; i += 6; }
                            else { result += s[i++]; }
                        }
                        else
                        {
                            result += s[i++];
                        }
                    }
                    return result;
                }
            };

            // --------------------------------------------------------------------
            // Simple XML Parser (non-validating)
            // --------------------------------------------------------------------
            class XmlParser
            {
            public:
                static std::unique_ptr<XmlNode> parse(const std::string& xml)
                {
                    XmlParser parser(xml);
                    return parser.parse_document();
                }

            private:
                const std::string& m_input;
                size_t m_pos = 0;
                size_t m_line = 1;
                size_t m_col = 1;

                XmlParser(const std::string& input) : m_input(input) {}

                void skip_whitespace()
                {
                    while (m_pos < m_input.size() && std::isspace(m_input[m_pos]))
                    {
                        if (m_input[m_pos] == '\n') { ++m_line; m_col = 1; }
                        else ++m_col;
                        ++m_pos;
                    }
                }

                char peek() const { return m_pos < m_input.size() ? m_input[m_pos] : '\0'; }
                char get()
                {
                    if (m_pos >= m_input.size()) return '\0';
                    char c = m_input[m_pos++];
                    if (c == '\n') { ++m_line; m_col = 1; }
                    else ++m_col;
                    return c;
                }

                bool match(const std::string& s)
                {
                    if (m_input.compare(m_pos, s.size(), s) == 0)
                    {
                        m_pos += s.size();
                        m_col += s.size();
                        return true;
                    }
                    return false;
                }

                std::unique_ptr<XmlNode> parse_document()
                {
                    auto doc = std::make_unique<XmlNode>(XmlNode::Type::Document);
                    skip_whitespace();
                    // Optional XML declaration
                    if (match("<?xml"))
                    {
                        auto decl = parse_declaration();
                        if (decl) doc->add_child(std::move(decl));
                    }
                    skip_whitespace();
                    // Parse root element
                    auto root = parse_element();
                    if (root) doc->add_child(std::move(root));
                    // Parse any trailing comments/whitespace
                    skip_whitespace();
                    while (match("<!--"))
                    {
                        auto comment = parse_comment();
                        if (comment) doc->add_child(std::move(comment));
                        skip_whitespace();
                    }
                    return doc;
                }

                std::unique_ptr<XmlNode> parse_declaration()
                {
                    auto node = std::make_unique<XmlNode>(XmlNode::Type::Declaration);
                    skip_whitespace();
                    while (peek() != '?' && peek() != '\0')
                    {
                        std::string name = parse_name();
                        skip_whitespace();
                        if (get() != '=') throw std::runtime_error("Expected '=' in XML declaration");
                        skip_whitespace();
                        char quote = get();
                        if (quote != '"' && quote != '\'') throw std::runtime_error("Expected quote");
                        std::string value;
                        while (peek() != quote && peek() != '\0')
                            value += get();
                        get(); // consume quote
                        node->set_attribute(name, value);
                        skip_whitespace();
                    }
                    if (!match("?>")) throw std::runtime_error("Unclosed XML declaration");
                    return node;
                }

                std::unique_ptr<XmlNode> parse_element()
                {
                    if (get() != '<') throw std::runtime_error("Expected '<'");
                    if (peek() == '/')
                    {
                        // Closing tag encountered where element expected
                        m_pos--; // push back '<'
                        return nullptr;
                    }
                    if (peek() == '!')
                    {
                        if (match("<!--"))
                            return parse_comment();
                        else if (match("<![CDATA["))
                            return parse_cdata();
                        else
                            throw std::runtime_error("Unknown markup");
                    }
                    if (peek() == '?')
                    {
                        return parse_declaration();
                    }

                    std::string name = parse_name();
                    auto node = std::make_unique<XmlNode>(name, XmlNode::Type::Element);
                    // Parse attributes
                    skip_whitespace();
                    while (peek() != '>' && peek() != '/' && peek() != '\0')
                    {
                        std::string attr_name = parse_name();
                        skip_whitespace();
                        if (get() != '=') throw std::runtime_error("Expected '='");
                        skip_whitespace();
                        char quote = get();
                        if (quote != '"' && quote != '\'') throw std::runtime_error("Expected quote");
                        std::string attr_value;
                        while (peek() != quote && peek() != '\0')
                            attr_value += get();
                        get(); // consume quote
                        node->set_attribute(attr_name, attr_value);
                        skip_whitespace();
                    }
                    if (peek() == '/')
                    {
                        get(); // '/'
                        if (get() != '>') throw std::runtime_error("Expected '>'");
                        return node;
                    }
                    get(); // '>'
                    // Parse content
                    while (true)
                    {
                        skip_whitespace();
                        if (match("</"))
                        {
                            std::string closing_name = parse_name();
                            if (closing_name != name)
                                throw std::runtime_error("Mismatched closing tag: expected </" + name + ">");
                            skip_whitespace();
                            if (get() != '>') throw std::runtime_error("Expected '>'");
                            break;
                        }
                        auto child = parse_node();
                        if (child)
                            node->add_child(std::move(child));
                        else
                        {
                            // Text content
                            std::string text = parse_text();
                            if (!text.empty())
                                node->add_text(text);
                        }
                    }
                    return node;
                }

                std::unique_ptr<XmlNode> parse_node()
                {
                    skip_whitespace();
                    if (peek() == '<')
                    {
                        if (m_input[m_pos+1] == '/')
                            return nullptr;
                        else if (m_input[m_pos+1] == '!')
                        {
                            if (match("<!--"))
                                return parse_comment();
                            else if (match("<![CDATA["))
                                return parse_cdata();
                        }
                        else
                            return parse_element();
                    }
                    return nullptr;
                }

                std::unique_ptr<XmlNode> parse_comment()
                {
                    auto node = std::make_unique<XmlNode>(XmlNode::Type::Comment);
                    std::string text;
                    while (!match("-->") && m_pos < m_input.size())
                        text += get();
                    node->set_text(text);
                    return node;
                }

                std::unique_ptr<XmlNode> parse_cdata()
                {
                    auto node = std::make_unique<XmlNode>(XmlNode::Type::Text);
                    std::string text;
                    while (!match("]]>") && m_pos < m_input.size())
                        text += get();
                    node->set_text(text);
                    return node;
                }

                std::string parse_text()
                {
                    std::string text;
                    while (peek() != '<' && peek() != '\0')
                        text += get();
                    // Trim
                    size_t start = 0;
                    while (start < text.size() && std::isspace(text[start])) ++start;
                    size_t end = text.size();
                    while (end > start && std::isspace(text[end-1])) --end;
                    if (start > 0 || end < text.size())
                        text = text.substr(start, end - start);
                    return text;
                }

                std::string parse_name()
                {
                    std::string name;
                    if (!std::isalpha(peek()) && peek() != '_' && peek() != ':')
                        throw std::runtime_error("Invalid name start character");
                    while (std::isalnum(peek()) || peek() == '_' || peek() == ':' || peek() == '-' || peek() == '.')
                        name += get();
                    return name;
                }
            };

            // --------------------------------------------------------------------
            // xarray to/from XML
            // --------------------------------------------------------------------
            namespace xml_detail
            {
                template<typename T>
                void array_to_xml_element(XmlNode* parent, const std::string& name,
                                          const xarray_container<T>& arr)
                {
                    XmlNode* elem = parent->add_element(name);
                    // Store shape as attribute
                    std::ostringstream shape_str;
                    for (size_t i = 0; i < arr.dimension(); ++i)
                    {
                        if (i > 0) shape_str << " ";
                        shape_str << arr.shape()[i];
                    }
                    elem->set_attribute("shape", shape_str.str());
                    
                    // Store dtype
                    if constexpr (std::is_same_v<T, float>)
                        elem->set_attribute("dtype", "float32");
                    else if constexpr (std::is_same_v<T, double>)
                        elem->set_attribute("dtype", "float64");
                    else if constexpr (std::is_same_v<T, int32_t>)
                        elem->set_attribute("dtype", "int32");
                    else if constexpr (std::is_same_v<T, int64_t>)
                        elem->set_attribute("dtype", "int64");
                    else if constexpr (std::is_same_v<T, uint8_t>)
                        elem->set_attribute("dtype", "uint8");
                    else
                        elem->set_attribute("dtype", "unknown");

                    // Store data as text (space separated)
                    std::ostringstream data_str;
                    data_str.precision(17);
                    for (size_t i = 0; i < arr.size(); ++i)
                    {
                        if (i > 0) data_str << " ";
                        data_str << arr.flat(i);
                    }
                    elem->add_text(data_str.str());
                }

                template<typename T>
                xarray_container<T> xml_to_array(const XmlNode* elem)
                {
                    if (!elem->is_element())
                        throw std::runtime_error("Expected element node");
                    
                    // Parse shape
                    std::string shape_str = elem->get_attribute("shape");
                    std::vector<size_t> shape;
                    std::istringstream shape_ss(shape_str);
                    size_t dim;
                    while (shape_ss >> dim)
                        shape.push_back(dim);
                    
                    xarray_container<T> result(shape);
                    
                    // Parse data text
                    const XmlNode* text_node = nullptr;
                    for (const auto& child : elem->children())
                    {
                        if (child->is_text())
                        {
                            text_node = child.get();
                            break;
                        }
                    }
                    if (!text_node)
                        throw std::runtime_error("No data text found in array element");
                    
                    std::string data_str = text_node->text();
                    std::istringstream data_ss(data_str);
                    for (size_t i = 0; i < result.size(); ++i)
                    {
                        T val;
                        data_ss >> val;
                        result.flat(i) = val;
                    }
                    return result;
                }

                // Recursively build XML from a map of named arrays
                void dict_to_xml(XmlNode* root, const std::map<std::string, xarray_container<double>>& dict)
                {
                    for (const auto& p : dict)
                        array_to_xml_element(root, p.first, p.second);
                }

                std::map<std::string, xarray_container<double>> xml_to_dict(const XmlNode* root)
                {
                    std::map<std::string, xarray_container<double>> result;
                    for (const auto& child : root->children())
                    {
                        if (child->is_element())
                            result[child->name()] = xml_to_array<double>(child.get());
                    }
                    return result;
                }
            } // namespace xml_detail

            // --------------------------------------------------------------------
            // Public XML I/O functions for xarray
            // --------------------------------------------------------------------
            template<typename T>
            inline std::string to_xml(const xarray_container<T>& arr,
                                      const std::string& root_name = "array",
                                      bool pretty = true)
            {
                auto doc = XmlNode::create_document();
                xml_detail::array_to_xml_element(doc.get(), root_name, arr);
                return doc->to_string(pretty);
            }

            template<typename T>
            inline xarray_container<T> from_xml(const std::string& xml_str,
                                                const std::string& root_name = "")
            {
                auto doc = XmlParser::parse(xml_str);
                const XmlNode* root = doc.get();
                // If document, get first element child
                if (root->type() == XmlNode::Type::Document && !root->children().empty())
                    root = root->children()[0].get();
                if (!root_name.empty())
                {
                    root = root->find_child(root_name);
                    if (!root)
                        throw std::runtime_error("Root element '" + root_name + "' not found");
                }
                return xml_detail::xml_to_array<T>(root);
            }

            // Save to file
            template<typename T>
            inline void save_xml(const std::string& filename, const xarray_container<T>& arr,
                                 const std::string& root_name = "array", bool pretty = true)
            {
                std::ofstream out(filename);
                if (!out)
                    XTENSOR_THROW(std::runtime_error, "Cannot open XML file for writing: " + filename);
                out << to_xml(arr, root_name, pretty);
            }

            // Load from file
            template<typename T>
            inline xarray_container<T> load_xml(const std::string& filename,
                                                const std::string& root_name = "")
            {
                std::ifstream in(filename);
                if (!in)
                    XTENSOR_THROW(std::runtime_error, "Cannot open XML file: " + filename);
                std::string content((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
                return from_xml<T>(content, root_name);
            }

            // --------------------------------------------------------------------
            // Dictionary of arrays to/from XML
            // --------------------------------------------------------------------
            inline std::string dict_to_xml(const std::map<std::string, xarray_container<double>>& dict,
                                           const std::string& root_name = "data",
                                           bool pretty = true)
            {
                auto doc = XmlNode::create_document();
                XmlNode* root = doc->add_element(root_name);
                xml_detail::dict_to_xml(root, dict);
                return doc->to_string(pretty);
            }

            inline std::map<std::string, xarray_container<double>> xml_to_dict(const std::string& xml_str,
                                                                                const std::string& root_name = "")
            {
                auto doc = XmlParser::parse(xml_str);
                const XmlNode* root = doc.get();
                if (root->type() == XmlNode::Type::Document && !root->children().empty())
                    root = root->children()[0].get();
                if (!root_name.empty())
                {
                    root = root->find_child(root_name);
                    if (!root)
                        throw std::runtime_error("Root element '" + root_name + "' not found");
                }
                return xml_detail::xml_to_dict(root);
            }

            inline void save_xml_dict(const std::string& filename,
                                      const std::map<std::string, xarray_container<double>>& dict,
                                      const std::string& root_name = "data", bool pretty = true)
            {
                std::ofstream out(filename);
                if (!out)
                    XTENSOR_THROW(std::runtime_error, "Cannot open XML file: " + filename);
                out << dict_to_xml(dict, root_name, pretty);
            }

            inline std::map<std::string, xarray_container<double>> load_xml_dict(const std::string& filename,
                                                                                  const std::string& root_name = "")
            {
                std::ifstream in(filename);
                if (!in)
                    XTENSOR_THROW(std::runtime_error, "Cannot open XML file: " + filename);
                std::string content((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
                return xml_to_dict(content, root_name);
            }

            // --------------------------------------------------------------------
            // XML Metadata support (attributes on root)
            // --------------------------------------------------------------------
            class XmlMetadata
            {
            public:
                void set(const std::string& key, const std::string& value) { m_attrs[key] = value; }
                std::string get(const std::string& key, const std::string& def = "") const
                {
                    auto it = m_attrs.find(key);
                    return it != m_attrs.end() ? it->second : def;
                }
                const std::map<std::string, std::string>& attributes() const { return m_attrs; }
                void clear() { m_attrs.clear(); }

            private:
                std::map<std::string, std::string> m_attrs;
            };

            template<typename T>
            inline std::string to_xml_with_metadata(const xarray_container<T>& arr,
                                                    const XmlMetadata& meta,
                                                    const std::string& root_name = "array",
                                                    bool pretty = true)
            {
                auto doc = XmlNode::create_document();
                XmlNode* root = doc->add_element(root_name);
                for (const auto& p : meta.attributes())
                    root->set_attribute(p.first, p.second);
                // Store array data as nested element
                xml_detail::array_to_xml_element(root, "data", arr);
                return doc->to_string(pretty);
            }

            template<typename T>
            inline std::pair<xarray_container<T>, XmlMetadata>
            from_xml_with_metadata(const std::string& xml_str, const std::string& root_name = "")
            {
                auto doc = XmlParser::parse(xml_str);
                const XmlNode* root = doc.get();
                if (root->type() == XmlNode::Type::Document && !root->children().empty())
                    root = root->children()[0].get();
                if (!root_name.empty())
                {
                    root = root->find_child(root_name);
                    if (!root) throw std::runtime_error("Root element not found");
                }
                XmlMetadata meta;
                for (const auto& p : root->attributes())
                    meta.set(p.first, p.second);
                const XmlNode* data_elem = root->find_child("data");
                if (!data_elem) throw std::runtime_error("Data element not found");
                return {xml_detail::xml_to_array<T>(data_elem), meta};
            }

        } // namespace io

        // Bring XML functions into xt namespace
        using io::XmlNode;
        using io::XmlParser;
        using io::XmlMetadata;
        using io::to_xml;
        using io::from_xml;
        using io::save_xml;
        using io::load_xml;
        using io::dict_to_xml;
        using io::xml_to_dict;
        using io::save_xml_dict;
        using io::load_xml_dict;
        using io::to_xml_with_metadata;
        using io::from_xml_with_metadata;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XIO_XML_HPP

// io/xio_xml.hpp