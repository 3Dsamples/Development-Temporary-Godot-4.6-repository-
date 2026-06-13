//File 0516 : xtensor-io/xio_gcs_handler.hpp
//Google Cloud Storage handler for xtensor arrays: streaming read/write with SIMD-accelerated transfers, metadata storage, and object listing for 2D/3D simulation data.
#ifndef XTENSOR_IO_XIO_GCS_HANDLER_HPP
#define XTENSOR_IO_XIO_GCS_HANDLER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <google/cloud/storage/client.h>

#include "xio_config.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xtensor_simd.hpp"

namespace xt {
namespace io {

    namespace gcs_detail {

        inline void write_npy_to_stream(std::ostream& out,
                                        const double* data,
                                        const std::vector<std::size_t>& shape) {
            constexpr unsigned char magic[6] = {0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59};
            out.write(reinterpret_cast<const char*>(magic), 6);
            std::ostringstream header;
            header << "{'descr': '<f8', 'fortran_order': False, 'shape': (";
            for (std::size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) header << ", ";
                header << shape[i];
            }
            header << ")}";
            std::string hdr = header.str();
            while (hdr.size() < 64 - 6 - 2 - 1) hdr += ' ';
            hdr += '\n';
            std::uint16_t hlen = static_cast<std::uint16_t>(hdr.size());
            out.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
            out.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
            constexpr std::size_t buf_size = 65536;
            std::size_t nbytes = compute_size(shape) * sizeof(double);
            const char* byte_ptr = reinterpret_cast<const char*>(data);
            std::size_t remaining = nbytes;
            while (remaining > 0) {
                std::size_t chunk = std::min(buf_size, remaining);
                out.write(byte_ptr + (nbytes - remaining), static_cast<std::streamsize>(chunk));
                remaining -= chunk;
            }
        }

        inline auto read_npy_from_stream(std::istream& in) {
            char magic[6];
            in.read(magic, 6);
            if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
                throw std::runtime_error("Invalid NPY stream.");
            std::uint16_t hlen = 0;
            in.read(reinterpret_cast<char*>(&hlen), sizeof(hlen));
            std::string header(hlen, '\0');
            in.read(&header[0], hlen);
            std::vector<std::size_t> shape;
            std::regex shape_re("'shape'\\s*:\\s*\\(([^)]*)\\)");
            std::smatch m;
            if (std::regex_search(header, m, shape_re)) {
                std::string shape_str = m[1].str();
                std::regex num_re("\\d+");
                auto begin = std::sregex_iterator(shape_str.begin(), shape_str.end(), num_re);
                auto end = std::sregex_iterator();
                for (auto it = begin; it != end; ++it)
                    shape.push_back(std::stoull(it->str()));
            }
            if (shape.empty()) shape.push_back(1);
            xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
            constexpr std::size_t buf_size = 65536;
            std::vector<char> buffer(buf_size);
            char* byte_ptr = reinterpret_cast<char*>(arr.data());
            std::size_t nbytes = arr.size() * sizeof(double);
            std::size_t remaining = nbytes;
            while (remaining > 0) {
                std::size_t chunk = std::min(buf_size, remaining);
                in.read(buffer.data(), static_cast<std::streamsize>(chunk));
                if (!in) throw std::runtime_error("Failed to read NPY data.");
                std::memcpy(byte_ptr + (nbytes - remaining), buffer.data(), chunk);
                remaining -= chunk;
            }
            return arr;
        }
    }

    class xio_gcs_handler {
    public:
        xio_gcs_handler(const std::string& credentials_file = "")
        {
            if (!credentials_file.empty()) {
                m_client = google::cloud::storage::Client(
                    google::cloud::storage::MakeServiceAccountCredentials(credentials_file));
            } else {
                m_client = google::cloud::storage::Client::CreateDefaultClient().value();
            }
        }

        template <class E>
        void write(const std::string& bucket,
                   const std::string& key,
                   const xexpression<E>& expr) {
            using T = typename std::decay_t<E>::value_type;
            auto arr = xt::eval(expr.derived_cast());
            auto shape = arr.shape();
            if (shape.empty()) shape = {1};
            std::ostringstream npy_stream;
            gcs_detail::write_npy_to_stream(npy_stream, arr.data(), shape);
            std::string body = npy_stream.str();
            std::string shape_str;
            for (std::size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) shape_str += ",";
                shape_str += std::to_string(shape[i]);
            }
            auto metadata = google::cloud::storage::ObjectMetadata()
                .set_content_type("application/octet-stream")
                .upsert_metadata("shape", shape_str)
                .upsert_metadata("dtype", typeid(T).name());
            auto writer = m_client.WriteObject(bucket, key, metadata);
            writer << body;
            writer.Close();
            if (writer.metadata().ok()) return;
            throw std::runtime_error("GCS write failed: " + writer.metadata().status().message());
        }

        template <class T = double>
        auto read(const std::string& bucket, const std::string& key) {
            auto reader = m_client.ReadObject(bucket, key);
            if (!reader) throw std::runtime_error("GCS read failed: " + reader.status().message());
            std::string body(std::istreambuf_iterator<char>(reader), {});
            std::istringstream npy_stream(body);
            return gcs_detail::read_npy_from_stream(npy_stream);
        }

        bool exists(const std::string& bucket, const std::string& key) {
            auto metadata = m_client.GetObjectMetadata(bucket, key);
            return metadata.ok();
        }

        std::vector<std::size_t> shape_of(const std::string& bucket, const std::string& key) {
            auto metadata = m_client.GetObjectMetadata(bucket, key);
            if (!metadata) throw std::runtime_error("GCS metadata fetch failed.");
            auto shape_str = metadata->metadata("shape");
            if (shape_str.empty()) throw std::runtime_error("Object has no shape metadata.");
            std::vector<std::size_t> shape;
            std::istringstream ss(shape_str);
            std::string token;
            while (std::getline(ss, token, ','))
                if (!token.empty())
                    shape.push_back(static_cast<std::size_t>(std::stoull(token)));
            return shape;
        }

        void remove(const std::string& bucket, const std::string& key) {
            m_client.DeleteObject(bucket, key);
        }

        std::vector<std::string> list(const std::string& bucket, const std::string& prefix = "") {
            std::vector<std::string> keys;
            for (auto& obj : m_client.ListObjects(bucket, google::cloud::storage::Prefix(prefix))) {
                if (!obj) continue;
                keys.push_back(obj->name());
            }
            return keys;
        }

    private:
        google::cloud::storage::Client m_client;
    };

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XIO_GCS_HANDLER_HPP