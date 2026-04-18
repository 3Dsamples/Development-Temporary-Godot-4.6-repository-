// include/xtu/godot/xgdscript_lsp.hpp
// xtensor-unified - GDScript Language Server Protocol for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XGDSCRIPT_LSP_HPP
#define XTU_GODOT_XGDSCRIPT_LSP_HPP

#include <atomic>
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
#include "xtu/godot/xgdscript.hpp"
#include "xtu/io/xio_json.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace lsp {

// #############################################################################
// Forward declarations
// #############################################################################
class GDScriptLanguageServer;
class GDScriptLanguageProtocol;
class GDScriptTextDocument;
class GDScriptWorkspace;

// #############################################################################
// LSP message types
// #############################################################################
enum class LSPMessageType : uint8_t {
    MSG_REQUEST = 0,
    MSG_RESPONSE = 1,
    MSG_NOTIFICATION = 2,
    MSG_ERROR = 3
};

// #############################################################################
// LSP diagnostic severity
// #############################################################################
enum class LSPDiagnosticSeverity : uint8_t {
    SEVERITY_ERROR = 1,
    SEVERITY_WARNING = 2,
    SEVERITY_INFO = 3,
    SEVERITY_HINT = 4
};

// #############################################################################
// LSP symbol kind
// #############################################################################
enum class LSPSymbolKind : uint8_t {
    KIND_FILE = 1,
    KIND_MODULE = 2,
    KIND_NAMESPACE = 3,
    KIND_PACKAGE = 4,
    KIND_CLASS = 5,
    KIND_METHOD = 6,
    KIND_PROPERTY = 7,
    KIND_FIELD = 8,
    KIND_CONSTRUCTOR = 9,
    KIND_ENUM = 10,
    KIND_INTERFACE = 11,
    KIND_FUNCTION = 12,
    KIND_VARIABLE = 13,
    KIND_CONSTANT = 14,
    KIND_STRING = 15,
    KIND_NUMBER = 16,
    KIND_BOOLEAN = 17,
    KIND_ARRAY = 18,
    KIND_OBJECT = 19,
    KIND_KEY = 20,
    KIND_NULL = 21,
    KIND_ENUM_MEMBER = 22,
    KIND_STRUCT = 23,
    KIND_EVENT = 24,
    KIND_OPERATOR = 25,
    KIND_TYPE_PARAMETER = 26
};

// #############################################################################
// LSP completion item kind
// #############################################################################
enum class LSPCompletionItemKind : uint8_t {
    COMPLETION_TEXT = 1,
    COMPLETION_METHOD = 2,
    COMPLETION_FUNCTION = 3,
    COMPLETION_CONSTRUCTOR = 4,
    COMPLETION_FIELD = 5,
    COMPLETION_VARIABLE = 6,
    COMPLETION_CLASS = 7,
    COMPLETION_INTERFACE = 8,
    COMPLETION_MODULE = 9,
    COMPLETION_PROPERTY = 10,
    COMPLETION_UNIT = 11,
    COMPLETION_VALUE = 12,
    COMPLETION_ENUM = 13,
    COMPLETION_KEYWORD = 14,
    COMPLETION_SNIPPET = 15,
    COMPLETION_COLOR = 16,
    COMPLETION_FILE = 17,
    COMPLETION_REFERENCE = 18,
    COMPLETION_FOLDER = 19,
    COMPLETION_ENUM_MEMBER = 20,
    COMPLETION_CONSTANT = 21,
    COMPLETION_STRUCT = 22,
    COMPLETION_EVENT = 23,
    COMPLETION_OPERATOR = 24,
    COMPLETION_TYPE_PARAMETER = 25
};

// #############################################################################
// LSP Position
// #############################################################################
struct LSPPosition {
    int line = 0;
    int character = 0;
    
    io::json::JsonValue to_json() const {
        io::json::JsonValue obj;
        obj["line"] = io::json::JsonValue(line);
        obj["character"] = io::json::JsonValue(character);
        return obj;
    }
    
    static LSPPosition from_json(const io::json::JsonValue& json) {
        LSPPosition pos;
        pos.line = static_cast<int>(json["line"].as_number());
        pos.character = static_cast<int>(json["character"].as_number());
        return pos;
    }
};

// #############################################################################
// LSP Range
// #############################################################################
struct LSPRange {
    LSPPosition start;
    LSPPosition end;
    
    io::json::JsonValue to_json() const {
        io::json::JsonValue obj;
        obj["start"] = start.to_json();
        obj["end"] = end.to_json();
        return obj;
    }
    
    static LSPRange from_json(const io::json::JsonValue& json) {
        LSPRange range;
        range.start = LSPPosition::from_json(json["start"]);
        range.end = LSPPosition::from_json(json["end"]);
        return range;
    }
};

// #############################################################################
// LSP Location
// #############################################################################
struct LSPLocation {
    String uri;
    LSPRange range;
    
    io::json::JsonValue to_json() const {
        io::json::JsonValue obj;
        obj["uri"] = io::json::JsonValue(uri.to_std_string());
        obj["range"] = range.to_json();
        return obj;
    }
};

// #############################################################################
// LSP Diagnostic
// #############################################################################
struct LSPDiagnostic {
    LSPRange range;
    LSPDiagnosticSeverity severity = LSPDiagnosticSeverity::SEVERITY_ERROR;
    String code;
    String source;
    String message;
    
    io::json::JsonValue to_json() const {
        io::json::JsonValue obj;
        obj["range"] = range.to_json();
        obj["severity"] = io::json::JsonValue(static_cast<int>(severity));
        obj["code"] = io::json::JsonValue(code.to_std_string());
        obj["source"] = io::json::JsonValue(source.to_std_string());
        obj["message"] = io::json::JsonValue(message.to_std_string());
        return obj;
    }
};

// #############################################################################
// LSP Completion Item
// #############################################################################
struct LSPCompletionItem {
    String label;
    LSPCompletionItemKind kind = LSPCompletionItemKind::COMPLETION_TEXT;
    String detail;
    String documentation;
    String insert_text;
    bool deprecated = false;
    int sort_text = 0;
    
    io::json::JsonValue to_json() const {
        io::json::JsonValue obj;
        obj["label"] = io::json::JsonValue(label.to_std_string());
        obj["kind"] = io::json::JsonValue(static_cast<int>(kind));
        if (!detail.empty()) obj["detail"] = io::json::JsonValue(detail.to_std_string());
        if (!documentation.empty()) obj["documentation"] = io::json::JsonValue(documentation.to_std_string());
        if (!insert_text.empty()) obj["insertText"] = io::json::JsonValue(insert_text.to_std_string());
        obj["deprecated"] = io::json::JsonValue(deprecated);
        obj["sortText"] = io::json::JsonValue(String::num(sort_text).to_std_string());
        return obj;
    }
};

// #############################################################################
// LSP Symbol Information
// #############################################################################
struct LSPSymbolInformation {
    String name;
    LSPSymbolKind kind = LSPSymbolKind::KIND_VARIABLE;
    LSPLocation location;
    String container_name;
    
    io::json::JsonValue to_json() const {
        io::json::JsonValue obj;
        obj["name"] = io::json::JsonValue(name.to_std_string());
        obj["kind"] = io::json::JsonValue(static_cast<int>(kind));
        obj["location"] = location.to_json();
        if (!container_name.empty()) {
            obj["containerName"] = io::json::JsonValue(container_name.to_std_string());
        }
        return obj;
    }
};

// #############################################################################
// LSP Message
// #############################################################################
struct LSPMessage {
    LSPMessageType type = LSPMessageType::MSG_REQUEST;
    String method;
    io::json::JsonValue params;
    int id = 0;
    
    static LSPMessage parse(const String& content) {
        LSPMessage msg;
        io::json::JsonValue json = io::json::JsonValue::parse(content.to_std_string());
        if (json.is_object()) {
            if (json["method"].is_string()) {
                msg.method = json["method"].as_string().c_str();
                msg.type = json["id"].is_null() ? LSPMessageType::MSG_NOTIFICATION : LSPMessageType::MSG_REQUEST;
            } else if (json["result"].is_defined() || json["error"].is_defined()) {
                msg.type = LSPMessageType::MSG_RESPONSE;
            }
            msg.params = json["params"];
            msg.id = static_cast<int>(json["id"].as_number());
        }
        return msg;
    }
    
    String serialize() const {
        io::json::JsonValue json;
        json["jsonrpc"] = io::json::JsonValue("2.0");
        if (type == LSPMessageType::MSG_REQUEST || type == LSPMessageType::MSG_NOTIFICATION) {
            json["method"] = io::json::JsonValue(method.to_std_string());
            json["params"] = params;
        }
        if (type == LSPMessageType::MSG_REQUEST) {
            json["id"] = io::json::JsonValue(static_cast<double>(id));
        }
        return json.dump().c_str();
    }
    
    static LSPMessage make_response(int id, const io::json::JsonValue& result) {
        LSPMessage msg;
        msg.type = LSPMessageType::MSG_RESPONSE;
        msg.id = id;
        msg.params = result;
        return msg;
    }
    
    static LSPMessage make_error(int id, int code, const String& message) {
        LSPMessage msg;
        msg.type = LSPMessageType::MSG_ERROR;
        msg.id = id;
        io::json::JsonValue error;
        error["code"] = io::json::JsonValue(static_cast<double>(code));
        error["message"] = io::json::JsonValue(message.to_std_string());
        msg.params = error;
        return msg;
    }
    
    static LSPMessage make_notification(const String& method, const io::json::JsonValue& params) {
        LSPMessage msg;
        msg.type = LSPMessageType::MSG_NOTIFICATION;
        msg.method = method;
        msg.params = params;
        return msg;
    }
};

// #############################################################################
// GDScriptTextDocument - Document management
// #############################################################################
class GDScriptTextDocument : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(GDScriptTextDocument, RefCounted)

private:
    String m_uri;
    String m_content;
    String m_language_id = "gdscript";
    int m_version = 0;
    std::vector<LSPDiagnostic> m_diagnostics;
    std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("GDScriptTextDocument"); }

    void set_uri(const String& uri) { m_uri = uri; }
    String get_uri() const { return m_uri; }

    void set_content(const String& content, int version) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_content = content;
        m_version = version;
    }

    String get_content() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_content;
    }

    int get_version() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_version;
    }

    void set_diagnostics(const std::vector<LSPDiagnostic>& diagnostics) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_diagnostics = diagnostics;
    }

    std::vector<LSPDiagnostic> get_diagnostics() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_diagnostics;
    }

    LSPPosition offset_to_position(int offset) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        LSPPosition pos;
        int line = 0;
        int character = 0;
        for (int i = 0; i < offset && i < static_cast<int>(m_content.length()); ++i) {
            if (m_content[i] == '\n') {
                ++line;
                character = 0;
            } else {
                ++character;
            }
        }
        pos.line = line;
        pos.character = character;
        return pos;
    }

    int position_to_offset(const LSPPosition& pos) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        int offset = 0;
        int line = 0;
        for (int i = 0; i < static_cast<int>(m_content.length()); ++i) {
            if (line == pos.line && (i - offset) == pos.character) {
                return i;
            }
            if (m_content[i] == '\n') {
                ++line;
                offset = i + 1;
            }
        }
        return static_cast<int>(m_content.length());
    }

    String get_text(const LSPRange& range) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        int start = position_to_offset(range.start);
        int end = position_to_offset(range.end);
        return m_content.substr(start, end - start);
    }
};

// #############################################################################
// GDScriptWorkspace - Project workspace management
// #############################################################################
class GDScriptWorkspace : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(GDScriptWorkspace, RefCounted)

private:
    String m_root_path;
    String m_root_uri;
    std::unordered_map<String, Ref<GDScriptTextDocument>> m_documents;
    std::unordered_map<String, std::vector<LSPSymbolInformation>> m_symbols;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("GDScriptWorkspace"); }

    void set_root_path(const String& path) {
        m_root_path = path;
        m_root_uri = "file://" + path;
    }

    String get_root_path() const { return m_root_path; }
    String get_root_uri() const { return m_root_uri; }

    void add_document(const Ref<GDScriptTextDocument>& doc) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_documents[doc->get_uri()] = doc;
    }

    void remove_document(const String& uri) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_documents.erase(uri);
        m_symbols.erase(uri);
    }

    Ref<GDScriptTextDocument> get_document(const String& uri) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_documents.find(uri);
        return it != m_documents.end() ? it->second : Ref<GDScriptTextDocument>();
    }

    std::vector<Ref<GDScriptTextDocument>> get_documents() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<Ref<GDScriptTextDocument>> result;
        for (const auto& kv : m_documents) {
            result.push_back(kv.second);
        }
        return result;
    }

    void update_symbols(const String& uri, const std::vector<LSPSymbolInformation>& symbols) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_symbols[uri] = symbols;
    }

    std::vector<LSPSymbolInformation> get_symbols(const String& uri) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_symbols.find(uri);
        return it != m_symbols.end() ? it->second : std::vector<LSPSymbolInformation>();
    }

    std::vector<LSPSymbolInformation> get_all_symbols() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<LSPSymbolInformation> result;
        for (const auto& kv : m_symbols) {
            result.insert(result.end(), kv.second.begin(), kv.second.end());
        }
        return result;
    }
};

// #############################################################################
// GDScriptLanguageProtocol - LSP message handling
// #############################################################################
class GDScriptLanguageProtocol : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(GDScriptLanguageProtocol, RefCounted)

private:
    std::function<void(const String&)> m_send_callback;
    std::queue<LSPMessage> m_outgoing_queue;
    std::string m_buffer;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("GDScriptLanguageProtocol"); }

    void set_send_callback(std::function<void(const String&)> cb) { m_send_callback = cb; }

    void receive_data(const String& data) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_buffer += data.to_std_string();
        parse_messages();
    }

    void send_message(const LSPMessage& msg) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_outgoing_queue.push(msg);
        flush_outgoing();
    }

    void flush_outgoing() {
        if (!m_send_callback) return;
        while (!m_outgoing_queue.empty()) {
            LSPMessage msg = std::move(m_outgoing_queue.front());
            m_outgoing_queue.pop();
            
            String content = msg.serialize();
            String header = "Content-Length: " + String::num(content.length()) + "\r\n\r\n";
            m_send_callback(header + content);
        }
    }

private:
    void parse_messages() {
        size_t pos = 0;
        while (pos < m_buffer.size()) {
            size_t header_end = m_buffer.find("\r\n\r\n", pos);
            if (header_end == std::string::npos) break;
            
            std::string header = m_buffer.substr(pos, header_end - pos);
            size_t content_length = 0;
            size_t cl_pos = header.find("Content-Length: ");
            if (cl_pos != std::string::npos) {
                content_length = std::stoull(header.substr(cl_pos + 16));
            }
            
            size_t content_start = header_end + 4;
            if (content_start + content_length > m_buffer.size()) break;
            
            std::string content = m_buffer.substr(content_start, content_length);
            LSPMessage msg = LSPMessage::parse(String(content.c_str()));
            if (msg.type != LSPMessageType::MSG_ERROR || msg.method.find("$/") == 0) {
                call_deferred("handle_message", msg);
            }
            
            pos = content_start + content_length;
        }
        if (pos > 0) {
            m_buffer = m_buffer.substr(pos);
        }
    }

    void handle_message(const LSPMessage& msg) {
        emit_signal("message_received", static_cast<int>(msg.type), msg.method, msg.id);
    }
};

// #############################################################################
// GDScriptLanguageServer - Main LSP server
// #############################################################################
class GDScriptLanguageServer : public Object {
    XTU_GODOT_REGISTER_CLASS(GDScriptLanguageServer, Object)

private:
    static GDScriptLanguageServer* s_singleton;
    Ref<GDScriptLanguageProtocol> m_protocol;
    Ref<GDScriptWorkspace> m_workspace;
    std::unordered_map<String, std::function<io::json::JsonValue(const io::json::JsonValue&)>> m_request_handlers;
    std::unordered_map<String, std::function<void(const io::json::JsonValue&)>> m_notification_handlers;
    bool m_initialized = false;
    std::mutex m_mutex;

public:
    static GDScriptLanguageServer* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("GDScriptLanguageServer"); }

    GDScriptLanguageServer() {
        s_singleton = this;
        m_protocol.instance();
        m_workspace.instance();
        m_protocol->connect("message_received", this, "on_message_received");
        register_handlers();
    }

    ~GDScriptLanguageServer() { s_singleton = nullptr; }

    void set_send_callback(std::function<void(const String&)> cb) {
        m_protocol->set_send_callback(cb);
    }

    void receive_data(const String& data) {
        m_protocol->receive_data(data);
    }

    void on_message_received(int type, const String& method, int id) {
        if (type == static_cast<int>(LSPMessageType::MSG_REQUEST)) {
            handle_request(method, id, io::json::JsonValue());
        } else if (type == static_cast<int>(LSPMessageType::MSG_NOTIFICATION)) {
            handle_notification(method, io::json::JsonValue());
        }
    }

    void initialize(const String& root_path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_workspace->set_root_path(root_path);
        m_initialized = true;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_initialized = false;
    }

    Ref<GDScriptWorkspace> get_workspace() const { return m_workspace; }

private:
    void register_handlers() {
        m_request_handlers["initialize"] = [this](const io::json::JsonValue& params) {
            return handle_initialize(params);
        };
        m_request_handlers["textDocument/completion"] = [this](const io::json::JsonValue& params) {
            return handle_completion(params);
        };
        m_request_handlers["textDocument/definition"] = [this](const io::json::JsonValue& params) {
            return handle_definition(params);
        };
        m_request_handlers["textDocument/hover"] = [this](const io::json::JsonValue& params) {
            return handle_hover(params);
        };
        m_request_handlers["textDocument/references"] = [this](const io::json::JsonValue& params) {
            return handle_references(params);
        };
        m_request_handlers["textDocument/documentSymbol"] = [this](const io::json::JsonValue& params) {
            return handle_document_symbol(params);
        };
        m_request_handlers["workspace/symbol"] = [this](const io::json::JsonValue& params) {
            return handle_workspace_symbol(params);
        };
        
        m_notification_handlers["initialized"] = [this](const io::json::JsonValue& params) {
            // Client is ready
        };
        m_notification_handlers["textDocument/didOpen"] = [this](const io::json::JsonValue& params) {
            handle_did_open(params);
        };
        m_notification_handlers["textDocument/didChange"] = [this](const io::json::JsonValue& params) {
            handle_did_change(params);
        };
        m_notification_handlers["textDocument/didClose"] = [this](const io::json::JsonValue& params) {
            handle_did_close(params);
        };
        m_notification_handlers["textDocument/didSave"] = [this](const io::json::JsonValue& params) {
            handle_did_save(params);
        };
    }

    void handle_request(const String& method, int id, const io::json::JsonValue& params) {
        auto it = m_request_handlers.find(method);
        if (it != m_request_handlers.end()) {
            io::json::JsonValue result = it->second(params);
            m_protocol->send_message(LSPMessage::make_response(id, result));
        } else {
            m_protocol->send_message(LSPMessage::make_error(id, -32601, "Method not found: " + method));
        }
    }

    void handle_notification(const String& method, const io::json::JsonValue& params) {
        auto it = m_notification_handlers.find(method);
        if (it != m_notification_handlers.end()) {
            it->second(params);
        }
    }

    io::json::JsonValue handle_initialize(const io::json::JsonValue& params) {
        io::json::JsonValue result;
        io::json::JsonValue capabilities;
        
        capabilities["textDocumentSync"] = io::json::JsonValue(1); // Full sync
        capabilities["completionProvider"] = io::json::JsonValue(std::map<std::string, io::json::JsonValue>{
            {"triggerCharacters", io::json::JsonValue(std::vector<io::json::JsonValue>{
                io::json::JsonValue("."), io::json::JsonValue(":")
            })}
        });
        capabilities["definitionProvider"] = io::json::JsonValue(true);
        capabilities["hoverProvider"] = io::json::JsonValue(true);
        capabilities["referencesProvider"] = io::json::JsonValue(true);
        capabilities["documentSymbolProvider"] = io::json::JsonValue(true);
        capabilities["workspaceSymbolProvider"] = io::json::JsonValue(true);
        
        result["capabilities"] = capabilities;
        
        io::json::JsonValue server_info;
        server_info["name"] = io::json::JsonValue("Xtensor GDScript Language Server");
        server_info["version"] = io::json::JsonValue("1.0.0");
        result["serverInfo"] = server_info;
        
        return result;
    }

    io::json::JsonValue handle_completion(const io::json::JsonValue& params) {
        std::vector<LSPCompletionItem> items;
        String uri = params["textDocument"]["uri"].as_string().c_str();
        LSPPosition pos = LSPPosition::from_json(params["position"]);
        
        Ref<GDScriptTextDocument> doc = m_workspace->get_document(uri);
        if (doc.is_valid()) {
            // Parse document and get completions at position
            items = get_completions_at_position(doc, pos);
        }
        
        io::json::JsonValue result;
        io::json::JsonValue items_array;
        for (const auto& item : items) {
            items_array.as_array().push_back(item.to_json());
        }
        result["items"] = items_array;
        result["isIncomplete"] = io::json::JsonValue(false);
        return result;
    }

    io::json::JsonValue handle_definition(const io::json::JsonValue& params) {
        String uri = params["textDocument"]["uri"].as_string().c_str();
        LSPPosition pos = LSPPosition::from_json(params["position"]);
        
        LSPLocation location = find_definition(uri, pos);
        
        if (location.uri.empty()) {
            return io::json::JsonValue();
        }
        return location.to_json();
    }

    io::json::JsonValue handle_hover(const io::json::JsonValue& params) {
        String uri = params["textDocument"]["uri"].as_string().c_str();
        LSPPosition pos = LSPPosition::from_json(params["position"]);
        
        String hover_text = get_hover_text(uri, pos);
        
        if (hover_text.empty()) {
            return io::json::JsonValue();
        }
        
        io::json::JsonValue result;
        io::json::JsonValue contents;
        contents["kind"] = io::json::JsonValue("markdown");
        contents["value"] = io::json::JsonValue(hover_text.to_std_string());
        result["contents"] = contents;
        return result;
    }

    io::json::JsonValue handle_references(const io::json::JsonValue& params) {
        String uri = params["textDocument"]["uri"].as_string().c_str();
        LSPPosition pos = LSPPosition::from_json(params["position"]);
        bool include_declaration = params["context"]["includeDeclaration"].as_bool();
        
        std::vector<LSPLocation> locations = find_references(uri, pos, include_declaration);
        
        io::json::JsonValue result;
        for (const auto& loc : locations) {
            result.as_array().push_back(loc.to_json());
        }
        return result;
    }

    io::json::JsonValue handle_document_symbol(const io::json::JsonValue& params) {
        String uri = params["textDocument"]["uri"].as_string().c_str();
        std::vector<LSPSymbolInformation> symbols = m_workspace->get_symbols(uri);
        
        io::json::JsonValue result;
        for (const auto& sym : symbols) {
            result.as_array().push_back(sym.to_json());
        }
        return result;
    }

    io::json::JsonValue handle_workspace_symbol(const io::json::JsonValue& params) {
        String query = params["query"].as_string().c_str();
        std::vector<LSPSymbolInformation> all_symbols = m_workspace->get_all_symbols();
        
        std::vector<LSPSymbolInformation> filtered;
        for (const auto& sym : all_symbols) {
            if (query.empty() || sym.name.to_lower().find(query.to_lower()) != String::npos) {
                filtered.push_back(sym);
            }
        }
        
        io::json::JsonValue result;
        for (const auto& sym : filtered) {
            result.as_array().push_back(sym.to_json());
        }
        return result;
    }

    void handle_did_open(const io::json::JsonValue& params) {
        String uri = params["textDocument"]["uri"].as_string().c_str();
        String content = params["textDocument"]["text"].as_string().c_str();
        int version = static_cast<int>(params["textDocument"]["version"].as_number());
        
        Ref<GDScriptTextDocument> doc;
        doc.instance();
        doc->set_uri(uri);
        doc->set_content(content, version);
        m_workspace->add_document(doc);
        
        publish_diagnostics(uri);
    }

    void handle_did_change(const io::json::JsonValue& params) {
        String uri = params["textDocument"]["uri"].as_string().c_str();
        int version = static_cast<int>(params["textDocument"]["version"].as_number());
        String content = params["contentChanges"][0]["text"].as_string().c_str();
        
        Ref<GDScriptTextDocument> doc = m_workspace->get_document(uri);
        if (doc.is_valid()) {
            doc->set_content(content, version);
            publish_diagnostics(uri);
        }
    }

    void handle_did_close(const io::json::JsonValue& params) {
        String uri = params["textDocument"]["uri"].as_string().c_str();
        m_workspace->remove_document(uri);
    }

    void handle_did_save(const io::json::JsonValue& params) {
        String uri = params["textDocument"]["uri"].as_string().c_str();
        publish_diagnostics(uri);
    }

    void publish_diagnostics(const String& uri) {
        Ref<GDScriptTextDocument> doc = m_workspace->get_document(uri);
        if (!doc.is_valid()) return;
        
        std::vector<LSPDiagnostic> diagnostics = analyze_document(doc);
        doc->set_diagnostics(diagnostics);
        
        io::json::JsonValue params;
        params["uri"] = io::json::JsonValue(uri.to_std_string());
        io::json::JsonValue diag_array;
        for (const auto& d : diagnostics) {
            diag_array.as_array().push_back(d.to_json());
        }
        params["diagnostics"] = diag_array;
        
        m_protocol->send_message(LSPMessage::make_notification("textDocument/publishDiagnostics", params));
    }

    std::vector<LSPDiagnostic> analyze_document(const Ref<GDScriptTextDocument>& doc) {
        std::vector<LSPDiagnostic> diagnostics;
        String content = doc->get_content();
        
        // Parse and analyze GDScript
        Ref<GDScript> script;
        script.instance();
        if (script->load_source_code(content)) {
            // If there are parse errors, convert to diagnostics
        }
        
        return diagnostics;
    }

    std::vector<LSPCompletionItem> get_completions_at_position(const Ref<GDScriptTextDocument>& doc, const LSPPosition& pos) {
        std::vector<LSPCompletionItem> items;
        
        // Built-in keywords
        std::vector<String> keywords = {"func", "var", "const", "if", "else", "elif", "for", "while", 
                                         "return", "class", "extends", "signal", "match", "break", "continue"};
        for (const auto& kw : keywords) {
            LSPCompletionItem item;
            item.label = kw;
            item.kind = LSPCompletionItemKind::COMPLETION_KEYWORD;
            item.insert_text = kw;
            items.push_back(item);
        }
        
        // Built-in types
        std::vector<String> types = {"int", "float", "bool", "String", "Vector2", "Vector3", "Color", 
                                      "Array", "Dictionary", "Node", "Resource"};
        for (const auto& t : types) {
            LSPCompletionItem item;
            item.label = t;
            item.kind = LSPCompletionItemKind::COMPLETION_CLASS;
            item.insert_text = t;
            items.push_back(item);
        }
        
        return items;
    }

    LSPLocation find_definition(const String& uri, const LSPPosition& pos) {
        LSPLocation location;
        // Find symbol definition using workspace symbols
        return location;
    }

    String get_hover_text(const String& uri, const LSPPosition& pos) {
        // Get documentation for symbol at position
        return String();
    }

    std::vector<LSPLocation> find_references(const String& uri, const LSPPosition& pos, bool include_declaration) {
        std::vector<LSPLocation> locations;
        // Find all references to symbol
        return locations;
    }
};

} // namespace lsp

// Bring into main namespace
using lsp::GDScriptLanguageServer;
using lsp::GDScriptLanguageProtocol;
using lsp::GDScriptTextDocument;
using lsp::GDScriptWorkspace;
using lsp::LSPMessage;
using lsp::LSPMessageType;
using lsp::LSPPosition;
using lsp::LSPRange;
using lsp::LSPLocation;
using lsp::LSPDiagnostic;
using lsp::LSPDiagnosticSeverity;
using lsp::LSPCompletionItem;
using lsp::LSPCompletionItemKind;
using lsp::LSPSymbolInformation;
using lsp::LSPSymbolKind;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XGDSCRIPT_LSP_HPP