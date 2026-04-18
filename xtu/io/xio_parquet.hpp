// io/xio_parquet.hpp

#ifndef XTENSOR_XIO_PARQUET_HPP
#define XTENSOR_XIO_PARQUET_HPP

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

// Parquet/Arrow support detection
#if __has_include(<arrow/api.h>) && __has_include(<parquet/api/reader.h>) && __has_include(<parquet/api/writer.h>)
    #define XTENSOR_HAS_PARQUET 1
    #include <arrow/api.h>
    #include <arrow/io/api.h>
    #include <arrow/ipc/api.h>
    #include <parquet/api/reader.h>
    #include <parquet/api/writer.h>
    #include <parquet/arrow/reader.h>
    #include <parquet/arrow/writer.h>
    #include <arrow/util/type_fwd.h>
#else
    #define XTENSOR_HAS_PARQUET 0
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace io
        {
            // --------------------------------------------------------------------
            // Parquet type mapping
            // --------------------------------------------------------------------
            enum class ParquetType
            {
                BOOLEAN,
                INT32,
                INT64,
                FLOAT,
                DOUBLE,
                BYTE_ARRAY,
                FIXED_LEN_BYTE_ARRAY
            };

            enum class ParquetCompression
            {
                UNCOMPRESSED,
                SNAPPY,
                GZIP,
                LZO,
                BROTLI,
                LZ4,
                ZSTD
            };

            // Schema field descriptor
            struct ParquetField
            {
                std::string name;
                ParquetType type;
                int type_length = 0;           // for FIXED_LEN_BYTE_ARRAY
                bool nullable = true;
                std::vector<ParquetField> children;  // for nested types
            };

            // --------------------------------------------------------------------
            // Parquet Reader/Writer implementation using Arrow
            // --------------------------------------------------------------------
#if XTENSOR_HAS_PARQUET
            namespace parquet_detail
            {
                // Convert xtensor data type to Arrow type
                template<typename T>
                std::shared_ptr<arrow::DataType> get_arrow_type()
                {
                    if constexpr (std::is_same_v<T, bool>)
                        return arrow::boolean();
                    else if constexpr (std::is_same_v<T, int8_t>)
                        return arrow::int8();
                    else if constexpr (std::is_same_v<T, int16_t>)
                        return arrow::int16();
                    else if constexpr (std::is_same_v<T, int32_t>)
                        return arrow::int32();
                    else if constexpr (std::is_same_v<T, int64_t>)
                        return arrow::int64();
                    else if constexpr (std::is_same_v<T, uint8_t>)
                        return arrow::uint8();
                    else if constexpr (std::is_same_v<T, uint16_t>)
                        return arrow::uint16();
                    else if constexpr (std::is_same_v<T, uint32_t>)
                        return arrow::uint32();
                    else if constexpr (std::is_same_v<T, uint64_t>)
                        return arrow::uint64();
                    else if constexpr (std::is_same_v<T, float>)
                        return arrow::float32();
                    else if constexpr (std::is_same_v<T, double>)
                        return arrow::float64();
                    else if constexpr (std::is_same_v<T, std::string>)
                        return arrow::utf8();
                    else
                        return arrow::float64();  // default
                }

                // Convert Arrow type to Parquet type
                inline ParquetType arrow_to_parquet_type(const std::shared_ptr<arrow::DataType>& type)
                {
                    switch (type->id())
                    {
                        case arrow::Type::BOOL:
                            return ParquetType::BOOLEAN;
                        case arrow::Type::INT8:
                        case arrow::Type::INT16:
                        case arrow::Type::INT32:
                        case arrow::Type::UINT8:
                        case arrow::Type::UINT16:
                        case arrow::Type::UINT32:
                            return ParquetType::INT32;
                        case arrow::Type::INT64:
                        case arrow::Type::UINT64:
                            return ParquetType::INT64;
                        case arrow::Type::FLOAT:
                            return ParquetType::FLOAT;
                        case arrow::Type::DOUBLE:
                            return ParquetType::DOUBLE;
                        case arrow::Type::STRING:
                        case arrow::Type::LARGE_STRING:
                        case arrow::Type::BINARY:
                        case arrow::Type::LARGE_BINARY:
                            return ParquetType::BYTE_ARRAY;
                        case arrow::Type::FIXED_SIZE_BINARY:
                            return ParquetType::FIXED_LEN_BYTE_ARRAY;
                        default:
                            return ParquetType::BYTE_ARRAY;
                    }
                }

                // Convert Parquet compression to Arrow compression
                inline arrow::Compression::type parquet_to_arrow_compression(ParquetCompression comp)
                {
                    switch (comp)
                    {
                        case ParquetCompression::UNCOMPRESSED:
                            return arrow::Compression::UNCOMPRESSED;
                        case ParquetCompression::SNAPPY:
                            return arrow::Compression::SNAPPY;
                        case ParquetCompression::GZIP:
                            return arrow::Compression::GZIP;
                        case ParquetCompression::LZO:
                            return arrow::Compression::LZO;
                        case ParquetCompression::BROTLI:
                            return arrow::Compression::BROTLI;
                        case ParquetCompression::LZ4:
                            return arrow::Compression::LZ4;
                        case ParquetCompression::ZSTD:
                            return arrow::Compression::ZSTD;
                        default:
                            return arrow::Compression::UNCOMPRESSED;
                    }
                }

                // Build Arrow schema from field descriptors
                inline std::shared_ptr<arrow::Schema> build_schema(const std::vector<ParquetField>& fields)
                {
                    std::vector<std::shared_ptr<arrow::Field>> arrow_fields;
                    for (const auto& f : fields)
                    {
                        std::shared_ptr<arrow::DataType> type;
                        if (!f.children.empty())
                        {
                            // Nested type - not fully implemented
                            type = arrow::struct_(build_schema(f.children));
                        }
                        else
                        {
                            switch (f.type)
                            {
                                case ParquetType::BOOLEAN:
                                    type = arrow::boolean();
                                    break;
                                case ParquetType::INT32:
                                    type = arrow::int32();
                                    break;
                                case ParquetType::INT64:
                                    type = arrow::int64();
                                    break;
                                case ParquetType::FLOAT:
                                    type = arrow::float32();
                                    break;
                                case ParquetType::DOUBLE:
                                    type = arrow::float64();
                                    break;
                                case ParquetType::BYTE_ARRAY:
                                    type = arrow::binary();
                                    break;
                                case ParquetType::FIXED_LEN_BYTE_ARRAY:
                                    type = arrow::fixed_size_binary(f.type_length);
                                    break;
                                default:
                                    type = arrow::binary();
                            }
                        }
                        arrow_fields.push_back(arrow::field(f.name, type, f.nullable));
                    }
                    return arrow::schema(arrow_fields);
                }

                // Create Arrow Array from xarray column data
                template<typename T>
                arrow::Result<std::shared_ptr<arrow::Array>> array_to_arrow(const xarray_container<T>& arr)
                {
                    if (arr.dimension() != 1)
                        return arrow::Status::Invalid("Column data must be 1D");

                    using BuilderType = typename arrow::CTypeTraits<T>::BuilderType;
                    BuilderType builder;
                    ARROW_RETURN_NOT_OK(builder.Reserve(arr.size()));

                    for (size_t i = 0; i < arr.size(); ++i)
                    {
                        if constexpr (std::is_same_v<T, bool>)
                            ARROW_RETURN_NOT_OK(builder.Append(arr(i)));
                        else if constexpr (std::is_same_v<T, std::string>)
                            ARROW_RETURN_NOT_OK(builder.Append(arr(i)));
                        else
                            ARROW_RETURN_NOT_OK(builder.Append(arr(i)));
                    }

                    std::shared_ptr<arrow::Array> result;
                    ARROW_RETURN_NOT_OK(builder.Finish(&result));
                    return result;
                }

                // Specialization for numeric types (use template above)
                // Arrow Array to xarray
                template<typename T>
                arrow::Result<xarray_container<T>> arrow_to_array(const std::shared_ptr<arrow::Array>& arr)
                {
                    xarray_container<T> result({static_cast<size_t>(arr->length())});
                    auto typed_arr = std::static_pointer_cast<typename arrow::CTypeTraits<T>::ArrayType>(arr);
                    for (int64_t i = 0; i < arr->length(); ++i)
                    {
                        if (typed_arr->IsNull(i))
                            result(static_cast<size_t>(i)) = T{};
                        else
                            result(static_cast<size_t>(i)) = typed_arr->Value(i);
                    }
                    return result;
                }

                // Arrow Table to map of xarrays
                inline arrow::Result<std::map<std::string, xarray_container<double>>> 
                table_to_map(const std::shared_ptr<arrow::Table>& table)
                {
                    std::map<std::string, xarray_container<double>> result;
                    for (int i = 0; i < table->num_columns(); ++i)
                    {
                        auto field = table->field(i);
                        auto column = table->column(i);
                        
                        // For simplicity, convert everything to double
                        xarray_container<double> arr({static_cast<size_t>(table->num_rows())});
                        
                        // Convert based on type
                        for (int64_t row = 0; row < table->num_rows(); ++row)
                        {
                            auto scalar_result = column->GetScalar(row);
                            if (!scalar_result.ok()) continue;
                            auto scalar = *scalar_result;
                            
                            if (scalar->type->id() == arrow::Type::DOUBLE)
                                arr(row) = std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value;
                            else if (scalar->type->id() == arrow::Type::FLOAT)
                                arr(row) = std::static_pointer_cast<arrow::FloatScalar>(scalar)->value;
                            else if (scalar->type->id() == arrow::Type::INT32)
                                arr(row) = std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value;
                            else if (scalar->type->id() == arrow::Type::INT64)
                                arr(row) = static_cast<double>(std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value);
                            else if (scalar->type->id() == arrow::Type::BOOL)
                                arr(row) = std::static_pointer_cast<arrow::BooleanScalar>(scalar)->value ? 1.0 : 0.0;
                        }
                        result[field->name()] = arr;
                    }
                    return result;
                }
            }
#endif // XTENSOR_HAS_PARQUET

            // --------------------------------------------------------------------
            // Parquet Writer
            // --------------------------------------------------------------------
            class ParquetWriter
            {
            public:
                ParquetWriter() = default;

                void open(const std::string& filename,
                          const std::vector<ParquetField>& schema = {},
                          ParquetCompression compression = ParquetCompression::SNAPPY)
                {
#if XTENSOR_HAS_PARQUET
                    m_filename = filename;
                    m_compression = compression;
                    
                    if (!schema.empty())
                    {
                        m_schema = parquet_detail::build_schema(schema);
                    }
                    
                    PARQUET_ASSIGN_OR_THROW(m_outfile, arrow::io::FileOutputStream::Open(filename));
#else
                    XTENSOR_THROW(std::runtime_error, "Parquet support not compiled (requires Arrow/Parquet)");
#endif
                }

                template<typename T>
                void write_column(const std::string& name, const xarray_container<T>& data)
                {
#if XTENSOR_HAS_PARQUET
                    if (data.dimension() != 1)
                        XTENSOR_THROW(std::invalid_argument, "ParquetWriter: column data must be 1D");
                    
                    m_columns[name] = data;
                    if (!m_schema)
                    {
                        // Build schema on the fly
                        std::vector<std::shared_ptr<arrow::Field>> fields;
                        for (const auto& p : m_columns)
                        {
                            auto type = parquet_detail::get_arrow_type<T>();
                            fields.push_back(arrow::field(p.first, type));
                        }
                        m_schema = arrow::schema(fields);
                    }
#else
                    XTENSOR_THROW(std::runtime_error, "Parquet support not compiled");
#endif
                }

                void write_table(const std::map<std::string, xarray_container<double>>& table)
                {
                    for (const auto& p : table)
                        write_column(p.first, p.second);
                }

                void close()
                {
#if XTENSOR_HAS_PARQUET
                    if (m_columns.empty()) return;
                    
                    size_t num_rows = m_columns.begin()->second.size();
                    for (const auto& p : m_columns)
                    {
                        if (p.second.size() != num_rows)
                            XTENSOR_THROW(std::runtime_error, "All columns must have same length");
                    }
                    
                    std::vector<std::shared_ptr<arrow::Array>> arrays;
                    std::vector<std::shared_ptr<arrow::Field>> fields;
                    
                    for (const auto& p : m_columns)
                    {
                        // Convert each column to Arrow Array
                        // For simplicity, we assume double type
                        arrow::DoubleBuilder builder;
                        PARQUET_THROW_NOT_OK(builder.Reserve(num_rows));
                        for (size_t i = 0; i < num_rows; ++i)
                            PARQUET_THROW_NOT_OK(builder.Append(p.second(i)));
                        std::shared_ptr<arrow::Array> arr;
                        PARQUET_THROW_NOT_OK(builder.Finish(&arr));
                        arrays.push_back(arr);
                        fields.push_back(arrow::field(p.first, arrow::float64()));
                    }
                    
                    auto schema = arrow::schema(fields);
                    auto table = arrow::Table::Make(schema, arrays, static_cast<int64_t>(num_rows));
                    
                    parquet::WriterProperties::Builder props_builder;
                    props_builder.compression(parquet_detail::parquet_to_arrow_compression(m_compression));
                    auto props = props_builder.build();
                    
                    PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(*table, 
                        arrow::default_memory_pool(), m_outfile, 1024*1024, props));
                    
                    m_outfile->Close();
                    m_columns.clear();
#else
                    XTENSOR_THROW(std::runtime_error, "Parquet support not compiled");
#endif
                }

                ~ParquetWriter()
                {
                    if (m_outfile)
                    {
                        try { close(); } catch (...) {}
                    }
                }

            private:
                std::string m_filename;
                ParquetCompression m_compression = ParquetCompression::SNAPPY;
#if XTENSOR_HAS_PARQUET
                std::shared_ptr<arrow::Schema> m_schema;
                std::shared_ptr<arrow::io::FileOutputStream> m_outfile;
                std::map<std::string, xarray_container<double>> m_columns;
#endif
            };

            // --------------------------------------------------------------------
            // Parquet Reader
            // --------------------------------------------------------------------
            class ParquetReader
            {
            public:
                ParquetReader() = default;

                void open(const std::string& filename)
                {
#if XTENSOR_HAS_PARQUET
                    m_filename = filename;
                    PARQUET_ASSIGN_OR_THROW(m_infile, arrow::io::ReadableFile::Open(filename));
                    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(m_infile, 
                        arrow::default_memory_pool(), &m_reader));
                    PARQUET_THROW_NOT_OK(m_reader->ReadTable(&m_table));
#else
                    XTENSOR_THROW(std::runtime_error, "Parquet support not compiled");
#endif
                }

                std::vector<std::string> list_columns() const
                {
#if XTENSOR_HAS_PARQUET
                    std::vector<std::string> cols;
                    if (m_table)
                    {
                        for (int i = 0; i < m_table->num_columns(); ++i)
                            cols.push_back(m_table->field(i)->name());
                    }
                    return cols;
#else
                    return {};
#endif
                }

                size_t num_rows() const
                {
#if XTENSOR_HAS_PARQUET
                    return m_table ? static_cast<size_t>(m_table->num_rows()) : 0;
#else
                    return 0;
#endif
                }

                xarray_container<double> read_column(const std::string& name) const
                {
#if XTENSOR_HAS_PARQUET
                    if (!m_table)
                        XTENSOR_THROW(std::runtime_error, "No Parquet file opened");
                    
                    auto col = m_table->GetColumnByName(name);
                    if (!col)
                        XTENSOR_THROW(std::runtime_error, "Column not found: " + name);
                    
                    size_t rows = static_cast<size_t>(m_table->num_rows());
                    xarray_container<double> result({rows});
                    
                    for (size_t i = 0; i < rows; ++i)
                    {
                        auto scalar_result = col->GetScalar(static_cast<int64_t>(i));
                        if (!scalar_result.ok())
                        {
                            result(i) = std::numeric_limits<double>::quiet_NaN();
                            continue;
                        }
                        auto scalar = *scalar_result;
                        
                        switch (scalar->type->id())
                        {
                            case arrow::Type::DOUBLE:
                                result(i) = std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value;
                                break;
                            case arrow::Type::FLOAT:
                                result(i) = std::static_pointer_cast<arrow::FloatScalar>(scalar)->value;
                                break;
                            case arrow::Type::INT32:
                                result(i) = std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value;
                                break;
                            case arrow::Type::INT64:
                                result(i) = static_cast<double>(std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value);
                                break;
                            case arrow::Type::BOOL:
                                result(i) = std::static_pointer_cast<arrow::BooleanScalar>(scalar)->value ? 1.0 : 0.0;
                                break;
                            default:
                                result(i) = 0.0;
                        }
                    }
                    return result;
#else
                    XTENSOR_THROW(std::runtime_error, "Parquet support not compiled");
                    return {};
#endif
                }

                std::map<std::string, xarray_container<double>> read_all() const
                {
                    std::map<std::string, xarray_container<double>> result;
#if XTENSOR_HAS_PARQUET
                    if (!m_table) return result;
                    for (const auto& col_name : list_columns())
                        result[col_name] = read_column(col_name);
#endif
                    return result;
                }

                void close()
                {
#if XTENSOR_HAS_PARQUET
                    m_table.reset();
                    m_reader.reset();
                    if (m_infile)
                    {
                        m_infile->Close();
                        m_infile.reset();
                    }
#endif
                }

                ~ParquetReader()
                {
                    close();
                }

            private:
                std::string m_filename;
#if XTENSOR_HAS_PARQUET
                std::shared_ptr<arrow::io::ReadableFile> m_infile;
                std::unique_ptr<parquet::arrow::FileReader> m_reader;
                std::shared_ptr<arrow::Table> m_table;
#endif
            };

            // --------------------------------------------------------------------
            // Convenience functions
            // --------------------------------------------------------------------
            inline void save_parquet(const std::string& filename,
                                     const std::map<std::string, xarray_container<double>>& table,
                                     ParquetCompression compression = ParquetCompression::SNAPPY)
            {
                ParquetWriter writer;
                writer.open(filename, {}, compression);
                writer.write_table(table);
                writer.close();
            }

            template<typename T>
            inline void save_parquet_column(const std::string& filename,
                                            const std::string& column_name,
                                            const xarray_container<T>& data,
                                            ParquetCompression compression = ParquetCompression::SNAPPY)
            {
                ParquetWriter writer;
                writer.open(filename, {}, compression);
                writer.write_column(column_name, data);
                writer.close();
            }

            inline std::map<std::string, xarray_container<double>> load_parquet(const std::string& filename)
            {
                ParquetReader reader;
                reader.open(filename);
                return reader.read_all();
            }

            inline xarray_container<double> load_parquet_column(const std::string& filename,
                                                                const std::string& column_name)
            {
                ParquetReader reader;
                reader.open(filename);
                return reader.read_column(column_name);
            }

            inline std::vector<std::string> list_parquet_columns(const std::string& filename)
            {
                ParquetReader reader;
                reader.open(filename);
                return reader.list_columns();
            }

            // --------------------------------------------------------------------
            // Write 2D array as table (rows = samples, columns = features)
            // --------------------------------------------------------------------
            template<typename T>
            inline void save_array_as_parquet(const std::string& filename,
                                              const xarray_container<T>& arr,
                                              const std::vector<std::string>& column_names = {},
                                              ParquetCompression compression = ParquetCompression::SNAPPY)
            {
                if (arr.dimension() != 2)
                    XTENSOR_THROW(std::invalid_argument, "Array must be 2D to save as Parquet table");
                
                size_t n_rows = arr.shape()[0];
                size_t n_cols = arr.shape()[1];
                
                std::vector<std::string> col_names = column_names;
                if (col_names.empty())
                {
                    for (size_t i = 0; i < n_cols; ++i)
                        col_names.push_back("col_" + std::to_string(i));
                }
                else if (col_names.size() != n_cols)
                {
                    XTENSOR_THROW(std::invalid_argument, "Column names count mismatch");
                }
                
                std::map<std::string, xarray_container<double>> table;
                for (size_t c = 0; c < n_cols; ++c)
                {
                    xarray_container<double> col({n_rows});
                    for (size_t r = 0; r < n_rows; ++r)
                        col(r) = static_cast<double>(arr(r, c));
                    table[col_names[c]] = col;
                }
                
                save_parquet(filename, table, compression);
            }

            template<typename T>
            inline xarray_container<T> load_parquet_as_array(const std::string& filename,
                                                             std::vector<std::string>* column_names = nullptr)
            {
                ParquetReader reader;
                reader.open(filename);
                auto columns = reader.list_columns();
                if (columns.empty())
                    return xarray_container<T>();
                
                size_t n_rows = reader.num_rows();
                size_t n_cols = columns.size();
                
                xarray_container<T> result({n_rows, n_cols});
                
                if (column_names)
                    *column_names = columns;
                
                for (size_t c = 0; c < n_cols; ++c)
                {
                    auto col_data = reader.read_column(columns[c]);
                    for (size_t r = 0; r < n_rows; ++r)
                        result(r, c) = static_cast<T>(col_data(r));
                }
                return result;
            }

        } // namespace io

        // Bring Parquet functions into xt namespace
        using io::ParquetWriter;
        using io::ParquetReader;
        using io::save_parquet;
        using io::load_parquet;
        using io::save_parquet_column;
        using io::load_parquet_column;
        using io::list_parquet_columns;
        using io::save_array_as_parquet;
        using io::load_parquet_as_array;
        using io::ParquetCompression;
        using io::ParquetField;
        using io::ParquetType;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XIO_PARQUET_HPP

// io/xio_parquet.hpp