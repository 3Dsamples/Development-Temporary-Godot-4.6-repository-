//File 0066 : io/xmime.hpp
//MIME type detection and I/O dispatch: auto-detect file format from extension or magic bytes and route to appropriate reader/writer.
#ifndef XTENSOR_XMIME_HPP
#define XTENSOR_XMIME_HPP

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xarray.hpp"
#include "../core/xeval.hpp"
#include "../core/xmath.hpp"
#include "../core/xbuilder.hpp"

namespace xt {
namespace io {

    /**
     * @enum file_format
     * @brief Enumeration of supported file formats for xtensor I/O.
     */
    enum class file_format {
        csv,
        tsv,
        json,
        npy,
        txt,
        binary,
        unknown
    };

    /**
     * Convert file_format enum to string.
     */
    inline const char* to_string(file_format fmt) noexcept {
        switch (fmt) {
            case file_format::csv:    return "csv";
            case file_format::tsv:    return "tsv";
            case file_format::json:   return "json";
            case file_format::npy:    return "npy";
            case file_format::txt:    return "txt";
            case file_format::binary: return "binary";
            default:                  return "unknown";
        }
    }

    /**
     * Convert string to file_format enum.
     */
    inline file_format format_from_string(const std::string& str) {
        std::string lower = str;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lower == "csv")    return file_format::csv;
        if (lower == "tsv")    return file_format::tsv;
        if (lower == "json")   return file_format::json;
        if (lower == "npy")    return file_format::npy;
        if (lower == "txt")    return file_format::txt;
        if (lower == "bin" || lower == "binary" || lower == "dat") return file_format::binary;
        return file_format::unknown;
    }

    namespace detail {

        /**
         * Detect file format from the file extension.
         */
        inline file_format detect_format_by_extension(const std::string& filename) {
            auto dot_pos = filename.rfind('.');
            if (dot_pos == std::string::npos) return file_format::unknown;
            std::string ext = filename.substr(dot_pos);
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".csv")  return file_format::csv;
            if (ext == ".tsv")  return file_format::tsv;
            if (ext == ".json") return file_format::json;
            if (ext == ".npy")  return file_format::npy;
            if (ext == ".txt")  return file_format::txt;
            if (ext == ".bin" || ext == ".dat") return file_format::binary;
            return file_format::unknown;
        }

        /**
         * Detect file format from magic bytes at the beginning of the file.
         */
        inline file_format detect_format_by_magic(const std::string& filename) {
            std::ifstream file(filename, std::ios::binary);
            if (!file) return file_format::unknown;
            char magic[8] = {};
            file.read(magic, 8);
            std::streamsize read_count = file.gcount();
            file.close();

            // NPY magic: "\x93NUMPY"
            if (read_count >= 6 && std::strncmp(magic, "\x93NUMPY", 6) == 0)
                return file_format::npy;

            // JSON starts with '[' or '{'
            if (read_count >= 1 && (magic[0] == '[' || magic[0] == '{'))
                return file_format::json;

            // CSV/TSV: look for comma or tab in first 100 bytes? We'll just try to parse
            // Return unknown and let caller fallback to extension.

            return file_format::unknown;
        }

        /**
         * Detect the delimiter for a delimiter-separated file by sampling the first line.
         */
        inline char detect_delimiter(const std::string& filename) {
            std::ifstream file(filename);
            if (!file) return ',';
            std::string line;
            if (!std::getline(file, line)) return ',';
            std::size_t commas = std::count(line.begin(), line.end(), ',');
            std::size_t tabs = std::count(line.begin(), line.end(), '\t');
            std::size_t semicolons = std::count(line.begin(), line.end(), ';');
            if (tabs > commas && tabs > semicolons) return '\t';
            if (semicolons > commas && semicolons > tabs) return ';';
            return ',';
        }

        /**
         * Check if a file has a header by comparing first row values against subsequent rows.
         * Simple heuristic: if first row contains non-numeric strings while second row is all numeric,
         * it's likely a header.
         */
        inline bool has_header_heuristic(const std::string& filename, char delimiter) {
            std::ifstream file(filename);
            if (!file) return false;
            std::string line1, line2;
            if (!std::getline(file, line1)) return false;
            if (!std::getline(file, line2)) return false;
            auto tokens1 = split_csv_line(line1, delimiter);
            auto tokens2 = split_csv_line(line2, delimiter);
            if (tokens1.size() != tokens2.size()) return false;
            std::size_t non_numeric_1 = 0, non_numeric_2 = 0;
            for (const auto& t : tokens1) {
                try { std::stod(t); } catch (...) { ++non_numeric_1; }
            }
            for (const auto& t : tokens2) {
                try { std::stod(t); } catch (...) { ++non_numeric_2; }
            }
            return non_numeric_1 > non_numeric_2;
        }

        // Forward declaration for split_csv_line (used by has_header_heuristic)
        inline std::vector<std::string> split_csv_line(const std::string& line, char delimiter) {
            std::vector<std::string> tokens;
            std::string current;
            for (std::size_t i = 0; i < line.size(); ++i) {
                if (line[i] == delimiter) {
                    tokens.push_back(current);
                    current.clear();
                } else {
                    current += line[i];
                }
            }
            tokens.push_back(current);
            return tokens;
        }
    }

    /**
     * @class xmime_manager
     * @brief Registry of I/O handlers for different file formats.
     *
     * Allows registering custom reader/writer functions for any format,
     * and provides automatic format detection and dispatch.
     */
    class xmime_manager {
    public:
        using reader_func = std::function<xarray_container<uvector<double>>(const std::string&)>;
        using writer_func = std::function<void(const std::string&, const xarray_container<uvector<double>>&)>;

        static xmime_manager& instance() {
            static xmime_manager mgr;
            return mgr;
        }

        /**
         * Register a reader for a specific format.
         */
        void register_reader(file_format fmt, reader_func reader) {
            m_readers[fmt] = std::move(reader);
        }

        /**
         * Register a writer for a specific format.
         */
        void register_writer(file_format fmt, writer_func writer) {
            m_writers[fmt] = std::move(writer);
        }

        /**
         * Check if a reader is registered for a format.
         */
        bool has_reader(file_format fmt) const {
            return m_readers.find(fmt) != m_readers.end();
        }

        /**
         * Check if a writer is registered for a format.
         */
        bool has_writer(file_format fmt) const {
            return m_writers.find(fmt) != m_writers.end();
        }

        /**
         * Read a file using the registered reader for its detected format.
         */
        auto read(const std::string& filename, file_format fmt = file_format::unknown) {
            if (fmt == file_format::unknown) {
                fmt = detect_format(filename);
            }
            if (!has_reader(fmt)) {
                throw std::runtime_error(std::string("No reader registered for format: ") + to_string(fmt));
            }
            return m_readers[fmt](filename);
        }

        /**
         * Write an array to a file using the registered writer for its format.
         */
        template <class E>
        void write(const std::string& filename, const E& expr, file_format fmt = file_format::unknown) {
            if (fmt == file_format::unknown) {
                fmt = detect_format(filename);
            }
            if (!has_writer(fmt)) {
                throw std::runtime_error(std::string("No writer registered for format: ") + to_string(fmt));
            }
            auto arr = xt::eval(expr);
            m_writers[fmt](filename, arr);
        }

        /**
         * Detect the format of a file using both magic bytes and extension.
         */
        file_format detect_format(const std::string& filename) const {
            file_format magic_fmt = detail::detect_format_by_magic(filename);
            if (magic_fmt != file_format::unknown) return magic_fmt;
            file_format ext_fmt = detail::detect_format_by_extension(filename);
            if (ext_fmt != file_format::unknown) return ext_fmt;
            return file_format::unknown;
        }

        /**
         * Detect the delimiter for a delimited text file.
         */
        char detect_delimiter(const std::string& filename) const {
            return detail::detect_delimiter(filename);
        }

        /**
         * Auto-detect whether a delimited file has a header.
         */
        bool has_header(const std::string& filename, char delimiter) const {
            return detail::has_header_heuristic(filename, delimiter);
        }

    private:
        std::map<file_format, reader_func> m_readers;
        std::map<file_format, writer_func> m_writers;

        xmime_manager() = default;
    };

    /**
     * Convenience function: auto-detect format and read a file.
     */
    inline auto read_file(const std::string& filename) {
        return xmime_manager::instance().read(filename);
    }

    /**
     * Convenience function: auto-detect format and write an array to a file.
     */
    template <class E>
    inline void write_file(const std::string& filename, const E& expr) {
        xmime_manager::instance().write(filename, expr);
    }

    /**
     * Register all default handlers for supported formats.
     */
    inline void register_default_handlers() {
        auto& mgr = xmime_manager::instance();

        // CSV reader/writer
        mgr.register_reader(file_format::csv, [](const std::string& filename) {
            return load_csv<double>(filename);
        });
        mgr.register_writer(file_format::csv, [](const std::string& filename, const auto& arr) {
            save_csv(filename, arr);
        });

        // TSV reader/writer
        mgr.register_reader(file_format::tsv, [](const std::string& filename) {
            return load_csv<double>(filename, '\t');
        });
        mgr.register_writer(file_format::tsv, [](const std::string& filename, const auto& arr) {
            save_csv(filename, arr, '\t');
        });

        // JSON reader/writer
        mgr.register_reader(file_format::json, [](const std::string& filename) {
            return load_json<double>(filename);
        });
        mgr.register_writer(file_format::json, [](const std::string& filename, const auto& arr) {
            save_json(filename, arr);
        });

        // NPY reader/writer
        mgr.register_reader(file_format::npy, [](const std::string& filename) {
            return load_npy<double>(filename);
        });
        mgr.register_writer(file_format::npy, [](const std::string& filename, const auto& arr) {
            save_npy(filename, arr);
        });

        // Plain text reader/writer
        mgr.register_reader(file_format::txt, [](const std::string& filename) {
            return load_txt<double>(filename);
        });
        mgr.register_writer(file_format::txt, [](const std::string& filename, const auto& arr) {
            std::ofstream file(filename);
            if (!file) throw std::runtime_error("Cannot open file: " + filename);
            auto shape = arr.shape();
            if (shape.size() == 1) {
                for (std::size_t i = 0; i < shape[0]; ++i) file << arr[i] << '\n';
            } else if (shape.size() == 2) {
                for (std::size_t i = 0; i < shape[0]; ++i) {
                    for (std::size_t j = 0; j < shape[1]; ++j) file << arr(i, j) << ' ';
                    file << '\n';
                }
            }
        });
    }

    /**
     * Auto-register default handlers at static initialization time.
     */
    namespace {
        struct auto_register_default_handlers {
            auto_register_default_handlers() {
                register_default_handlers();
            }
        };
        static auto_register_default_handlers auto_register_instance;
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_XMIME_HPP