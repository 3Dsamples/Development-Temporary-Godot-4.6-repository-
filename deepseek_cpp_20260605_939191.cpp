//File 0216 : sparse/xsparse_io.hpp
//Sparse I/O: Matrix Market (.mtx) reader/writer, binary sparse serialization, and sparse-dense conversion for file I/O with SIMD-accelerated parsing.
#ifndef XTENSOR_XSPARSE_IO_HPP
#define XTENSOR_XSPARSE_IO_HPP

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xarray.hpp"
#include "../core/xeval.hpp"
#include "../core/xmath.hpp"
#include "../core/xstrides.hpp"
#include "../sparse/xsparse_config.hpp"
#include "../sparse/xcoo.hpp"
#include "../sparse/xcsr.hpp"
#include "../sparse/xcsc.hpp"

namespace xt {
namespace sparse {

    /**
     * @enum matrix_market_format
     * @brief Matrix Market storage schemes.
     */
    enum class matrix_market_format {
        coordinate,  // COO
        array        // dense
    };

    /**
     * @enum matrix_market_field
     * @brief Matrix Market field types.
     */
    enum class matrix_market_field {
        real,
        double_precision,
        complex,
        integer,
        pattern
    };

    /**
     * @enum matrix_market_symmetry
     * @brief Matrix Market symmetry types.
     */
    enum class matrix_market_symmetry {
        general,
        symmetric,
        skew_symmetric,
        hermitian
    };

    namespace detail {

        /**
         * Trim leading and trailing whitespace from a string.
         */
        inline std::string trim(const std::string& s)
        {
            auto start = s.find_first_not_of(" \t\n\r\f\v");
            if (start == std::string::npos) return "";
            auto end = s.find_last_not_of(" \t\n\r\f\v");
            return s.substr(start, end - start + 1);
        }

        /**
         * Convert string to lowercase.
         */
        inline std::string to_lower(std::string s)
        {
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        /**
         * Parse the Matrix Market banner line.
         * Format: %%MatrixMarket matrix <format> <field> <symmetry>
         */
        inline void parse_mm_banner(const std::string& line,
                                     matrix_market_format& fmt,
                                     matrix_market_field& field,
                                     matrix_market_symmetry& sym)
        {
            std::istringstream iss(line);
            std::string token;
            // Skip "%%MatrixMarket"
            iss >> token;
            if (token != "%%MatrixMarket")
                throw std::runtime_error("Not a valid Matrix Market file.");
            // Skip "matrix"
            iss >> token;
            if (to_lower(token) != "matrix")
                throw std::runtime_error("Only matrix type supported in Matrix Market.");
            // Format
            iss >> token;
            token = to_lower(token);
            if (token == "coordinate") fmt = matrix_market_format::coordinate;
            else if (token == "array") fmt = matrix_market_format::array;
            else throw std::runtime_error("Unknown Matrix Market format: " + token);
            // Field
            iss >> token;
            token = to_lower(token);
            if (token == "real") field = matrix_market_field::real;
            else if (token == "double") field = matrix_market_field::double_precision;
            else if (token == "complex") field = matrix_market_field::complex;
            else if (token == "integer") field = matrix_market_field::integer;
            else if (token == "pattern") field = matrix_market_field::pattern;
            else throw std::runtime_error("Unknown Matrix Market field: " + token);
            // Symmetry
            iss >> token;
            token = to_lower(token);
            if (token == "general") sym = matrix_market_symmetry::general;
            else if (token == "symmetric") sym = matrix_market_symmetry::symmetric;
            else if (token == "skew-symmetric") sym = matrix_market_symmetry::skew_symmetric;
            else if (token == "hermitian") sym = matrix_market_symmetry::hermitian;
            else throw std::runtime_error("Unknown Matrix Market symmetry: " + token);
        }

        /**
         * Fast number parser using std::from_chars.
         */
        template <class T>
        inline T parse_number(const char* start, const char* end)
        {
            T value{};
            auto [ptr, ec] = std::from_chars(start, end, value);
            if (ec != std::errc())
                throw std::runtime_error("Failed to parse number in Matrix Market file.");
            return value;
        }

        /**
         * Skip comment lines (starting with '%') in the input stream.
         * Returns the first non-comment line.
         */
        inline std::string skip_comments(std::ifstream& file)
        {
            std::string line;
            while (std::getline(file, line))
            {
                line = trim(line);
                if (line.empty()) continue;
                if (line[0] != '%') return line;
            }
            throw std::runtime_error("Unexpected end of Matrix Market file.");
        }
    }

    /**
     * Load a sparse matrix from a Matrix Market (.mtx) file.
     * Returns a CSR matrix for efficient computation.
     */
    template <class T = double>
    inline auto load_mtx(const std::string& filename)
    {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Cannot open Matrix Market file: " + filename);

        // Parse banner
        std::string line;
        std::getline(file, line);
        line = detail::trim(line);
        if (line.empty() || line[0] != '%')
            throw std::runtime_error("Invalid Matrix Market file: missing banner.");

        matrix_market_format fmt;
        matrix_market_field field;
        matrix_market_symmetry sym;
        detail::parse_mm_banner(line, fmt, field, sym);

        // Skip comment lines to get dimensions
        line = detail::skip_comments(file);
        std::istringstream dim_iss(line);
        std::size_t nrows, ncols, nnz = 0;
        dim_iss >> nrows >> ncols;
        if (fmt == matrix_market_format::coordinate)
            dim_iss >> nnz;

        // Create COO matrix and read entries
        xcoo_matrix<T> coo(nrows, ncols);
        if (nnz > 0) coo.reserve(nnz);

        bool pattern_field = (field == matrix_market_field::pattern);
        std::string token_line;
        while (std::getline(file, token_line))
        {
            token_line = detail::trim(token_line);
            if (token_line.empty()) continue;
            std::istringstream entry_iss(token_line);
            std::size_t row, col;
            entry_iss >> row >> col;
            // Convert 1-based to 0-based
            --row; --col;
            if (row >= nrows || col >= ncols)
                throw std::runtime_error("Matrix Market entry out of bounds.");

            T val = T(1);
            if (!pattern_field)
            {
                if constexpr (std::is_same_v<T, std::complex<float>> ||
                              std::is_same_v<T, std::complex<double>>)
                {
                    typename T::value_type re, im = 0;
                    entry_iss >> re >> im;
                    val = T(re, im);
                }
                else
                {
                    std::string num_str;
                    entry_iss >> num_str;
                    val = detail::parse_number<T>(num_str.data(), num_str.data() + num_str.size());
                }
            }
            coo.append(row, col, val);

            // Handle symmetry: also add transposed entry
            if (sym == matrix_market_symmetry::symmetric && row != col)
            {
                coo.append(col, row, val);
            }
            else if (sym == matrix_market_symmetry::skew_symmetric && row != col)
            {
                coo.append(col, row, -val);
            }
            else if (sym == matrix_market_symmetry::hermitian && row != col)
            {
                if constexpr (std::is_same_v<T, std::complex<float>> ||
                              std::is_same_v<T, std::complex<double>>)
                {
                    coo.append(col, row, std::conj(val));
                }
                else
                {
                    coo.append(col, row, val);
                }
            }
        }

        return xcsr_matrix<T>::from_coo(coo);
    }

    /**
     * Save a sparse matrix to a Matrix Market (.mtx) file.
     */
    template <class T>
    inline void save_mtx(const std::string& filename, const xcsr_matrix<T>& mat,
                         const std::string& comment = "")
    {
        std::ofstream file(filename);
        if (!file) throw std::runtime_error("Cannot open file for writing: " + filename);

        // Write banner
        file << "%%MatrixMarket matrix coordinate real general\n";
        if (!comment.empty())
            file << "% " << comment << "\n";

        // Write dimensions
        file << mat.rows() << " " << mat.cols() << " " << mat.nnz() << "\n";

        // Write entries (1-based indexing)
        for (std::size_t r = 0; r < mat.rows(); ++r)
        {
            for (std::size_t i = mat.row_ptr()[r]; i < mat.row_ptr()[r + 1]; ++i)
            {
                file << (r + 1) << " " << (mat.col_idx()[i] + 1) << " ";
                if constexpr (std::is_same_v<T, std::complex<float>> ||
                              std::is_same_v<T, std::complex<double>>)
                {
                    file << mat.values()[i].real() << " " << mat.values()[i].imag();
                }
                else
                {
                    file << mat.values()[i];
                }
                file << "\n";
            }
        }
    }

    /**
     * Save a sparse matrix to a binary file (custom format).
     * Format: [uint64 nrows][uint64 ncols][uint64 nnz]
     *         [uint64 row_indices[nnz]][uint64 col_indices[nnz]][T values[nnz]]
     */
    template <class T>
    inline void save_sparse_binary(const std::string& filename, const xcsr_matrix<T>& mat)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open binary file for writing: " + filename);

        std::uint64_t nrows = mat.rows();
        std::uint64_t ncols = mat.cols();
        std::uint64_t nnz = mat.nnz();
        file.write(reinterpret_cast<const char*>(&nrows), sizeof(nrows));
        file.write(reinterpret_cast<const char*>(&ncols), sizeof(ncols));
        file.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));

        // Build COO representation for writing
        for (std::size_t r = 0; r < mat.rows(); ++r)
        {
            for (std::size_t i = mat.row_ptr()[r]; i < mat.row_ptr()[r + 1]; ++i)
            {
                std::uint64_t row = r;
                std::uint64_t col = mat.col_idx()[i];
                file.write(reinterpret_cast<const char*>(&row), sizeof(row));
                file.write(reinterpret_cast<const char*>(&col), sizeof(col));
                file.write(reinterpret_cast<const char*>(&mat.values()[i]), sizeof(T));
            }
        }
    }

    /**
     * Load a sparse matrix from a binary file.
     */
    template <class T = double>
    inline auto load_sparse_binary(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open binary file: " + filename);

        std::uint64_t nrows, ncols, nnz;
        file.read(reinterpret_cast<char*>(&nrows), sizeof(nrows));
        file.read(reinterpret_cast<char*>(&ncols), sizeof(ncols));
        file.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));

        xcoo_matrix<T> coo(nrows, ncols);
        coo.reserve(nnz);
        for (std::uint64_t i = 0; i < nnz; ++i)
        {
            std::uint64_t row, col;
            T val;
            file.read(reinterpret_cast<char*>(&row), sizeof(row));
            file.read(reinterpret_cast<char*>(&col), sizeof(col));
            file.read(reinterpret_cast<char*>(&val), sizeof(val));
            coo.append(row, col, val);
        }
        return xcsr_matrix<T>::from_coo(coo);
    }

    /**
     * Print sparse matrix statistics to an output stream.
     */
    template <class T>
    inline std::ostream& operator<<(std::ostream& os, const xcsr_matrix<T>& mat)
    {
        os << "CSR sparse matrix: " << mat.rows() << " x " << mat.cols()
           << ", nnz = " << mat.nnz()
           << " (density = " << (100.0 * density(mat)) << "%)\n";
        os << "  memory: " << (memory_usage_bytes(mat) / 1024) << " KB\n";
        return os;
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_IO_HPP