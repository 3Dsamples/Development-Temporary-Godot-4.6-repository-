//File 0522 : xtensor-io/xtensor-io.hpp
//Top‑level include for xtensor‑io: aggregates all I/O modules (NPY, CSV, JSON, image, audio, binary, compression, cloud handlers) with full C++17 SIMD support.
#ifndef XTENSOR_IO_HPP
#define XTENSOR_IO_HPP

#include "xio_config.hpp"
#include "xio_binary.hpp"
#include "xio_blosc.hpp"
#include "xio_disk_handler.hpp"
#include "xio_file_wrapper.hpp"
#include "xio_stream_wrapper.hpp"
#include "xio_gzip.hpp"
#include "xio_zlib.hpp"
#include "xnpy.hpp"
#include "xnpz.hpp"
#include "xcsv.hpp"
#include "xjson.hpp"
#include "ximage.hpp"
#include "xaudio.hpp"
#include "xgdal.hpp"
#include "xhighfive.hpp"
#include "xfile_array.hpp"
#include "xchunk_store_manager.hpp"

#ifdef XTENSOR_IO_ENABLE_CLOUD
#include "xio_aws_handler.hpp"
#include "xio_gcs_handler.hpp"
#include "xio_vsilfile_wrapper.hpp"
#endif

namespace xt {
namespace io {

    // Re‑export commonly used functions
    using io::load_npy;
    using io::save_npy;
    using io::load_csv;
    using io::save_csv;
    using io::load_json;
    using io::save_json;
    using io::load_image;
    using io::save_image;
    using io::load_audio;
    using io::dump_audio;
    using io::load_binary;
    using io::save_binary;
    using io::blosc_compress_array;
    using io::blosc_decompress_array;
    using io::gzip_compress_array;
    using io::gzip_decompress_array;
    using io::zlib_compress_array;
    using io::zlib_decompress_array;

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_HPP