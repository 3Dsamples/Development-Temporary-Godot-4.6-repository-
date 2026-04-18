// io/xio_avro.hpp

#ifndef XTENSOR_XIO_AVRO_HPP
#define XTENSOR_XIO_AVRO_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "xio_json.hpp"

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

// Avro support detection (using Avro C++ library)
#if __has_include(<avro/DataFile.hh>) && __has_include(<avro/Schema.hh>) && __has_include(<avro/Stream.hh>)
    #define XTENSOR_HAS_AVRO 1
    #include <avro/DataFile.hh>
    #include <avro/Schema.hh>
    #include <avro/Stream.hh>
    #include <avro/Encoder.hh>
    #include <avro/Decoder.hh>
    #include <avro/Specific.hh>
    #include <avro/Generic.hh>
#else
    #define XTENSOR_HAS_AVRO 0
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace io
        {
            // --------------------------------------------------------------------
            // Avro type mapping and schema building
            // --------------------------------------------------------------------
            enum class AvroType
            {
                NULL_TYPE,
                BOOLEAN,
                INT,
                LONG,
                FLOAT,
                DOUBLE,
                BYTES,
                STRING,
                ARRAY,
                MAP,
                ENUM,
                FIXED,
                RECORD,
                UNION
            };

            struct AvroField
            {
                std::string name;
                AvroType type;
                std::vector<AvroField> fields;      // for record
                AvroType items_type = AvroType::NULL_TYPE;  // for array
                AvroType values_type = AvroType::NULL_TYPE; // for map
                std::vector<std::string> symbols;     // for enum
                int fixed_size = 0;                   // for fixed
                std::optional<JsonValue> default_value;
                std::string doc;
                std::vector<AvroType> union_types;    // for union
            };

#if XTENSOR_HAS_AVRO
            namespace avro_detail
            {
                // Convert AvroField to Avro C++ Schema
                avro::Schema build_schema(const AvroField& field)
                {
                    avro::Schema schema;
                    switch (field.type)
                    {
                        case AvroType::NULL_TYPE:
                            schema = avro::NullSchema();
                            break;
                        case AvroType::BOOLEAN:
                            schema = avro::BoolSchema();
                            break;
                        case AvroType::INT:
                            schema = avro::IntSchema();
                            break;
                        case AvroType::LONG:
                            schema = avro::LongSchema();
                            break;
                        case AvroType::FLOAT:
                            schema = avro::FloatSchema();
                            break;
                        case AvroType::DOUBLE:
                            schema = avro::DoubleSchema();
                            break;
                        case AvroType::BYTES:
                            schema = avro::BytesSchema();
                            break;
                        case AvroType::STRING:
                            schema = avro::StringSchema();
                            break;
                        case AvroType::ARRAY:
                            {
                                AvroField item_field{"", field.items_type};
                                schema = avro::ArraySchema(build_schema(item_field));
                            }
                            break;
                        case AvroType::MAP:
                            {
                                AvroField value_field{"", field.values_type};
                                schema = avro::MapSchema(build_schema(value_field));
                            }
                            break;
                        case AvroType::ENUM:
                            schema = avro::EnumSchema(field.name, field.symbols);
                            break;
                        case AvroType::FIXED:
                            schema = avro::FixedSchema(field.name, field.fixed_size);
                            break;
                        case AvroType::RECORD:
                            {
                                std::vector<avro::RecordField> record_fields;
                                for (const auto& f : field.fields)
                                {
                                    record_fields.emplace_back(f.name, build_schema(f));
                                    if (!f.doc.empty())
                                        record_fields.back().setDoc(f.doc);
                                }
                                schema = avro::RecordSchema(field.name, record_fields);
                            }
                            break;
                        case AvroType::UNION:
                            {
                                std::vector<avro::Schema> schemas;
                                for (const auto& t : field.union_types)
                                {
                                    AvroField tmp{"", t};
                                    schemas.push_back(build_schema(tmp));
                                }
                                schema = avro::UnionSchema(schemas);
                            }
                            break;
                        default:
                            throw std::runtime_error("Unsupported Avro type");
                    }
                    if (!field.doc.empty())
                        schema.setDoc(field.doc);
                    return schema;
                }

                // Convert JSON schema to Avro schema
                avro::Schema json_to_avro_schema(const JsonValue& json_schema)
                {
                    std::string schema_str = json_schema.dump();
                    return avro::compileJsonSchemaFromString(schema_str);
                }

                // Avro datum to JsonValue (for debugging)
                JsonValue datum_to_json(const avro::GenericDatum& datum)
                {
                    std::ostringstream oss;
                    avro::EncoderPtr encoder = avro::jsonEncoder(datum.schema());
                    avro::encode(*encoder, datum);
                    encoder->flush();
                    // The JSON encoder writes to the stream; we'd need to parse it back.
                    // For simplicity, we'll use a manual conversion for basic types.
                    switch (datum.type())
                    {
                        case avro::AVRO_NULL:
                            return JsonValue(nullptr);
                        case avro::AVRO_BOOL:
                            return JsonValue(datum.value<bool>());
                        case avro::AVRO_INT:
                            return JsonValue(static_cast<int64_t>(datum.value<int32_t>()));
                        case avro::AVRO_LONG:
                            return JsonValue(datum.value<int64_t>());
                        case avro::AVRO_FLOAT:
                            return JsonValue(static_cast<double>(datum.value<float>()));
                        case avro::AVRO_DOUBLE:
                            return JsonValue(datum.value<double>());
                        case avro::AVRO_STRING:
                            return JsonValue(datum.value<std::string>());
                        case avro::AVRO_BYTES:
                            {
                                const auto& vec = datum.value<std::vector<uint8_t>>();
                                JsonValue::Array arr;
                                for (auto b : vec) arr.push_back(JsonValue(static_cast<int64_t>(b)));
                                return JsonValue(arr);
                            }
                        case avro::AVRO_ARRAY:
                            {
                                const auto& arr = datum.value<avro::GenericArray>();
                                JsonValue::Array jarr;
                                for (const auto& item : arr.value())
                                    jarr.push_back(datum_to_json(item));
                                return JsonValue(jarr);
                            }
                        case avro::AVRO_MAP:
                            {
                                const auto& map = datum.value<avro::GenericMap>();
                                JsonValue::Object jobj;
                                for (const auto& p : map.value())
                                    jobj[p.first] = datum_to_json(p.second);
                                return JsonValue(jobj);
                            }
                        case avro::AVRO_RECORD:
                            {
                                const auto& rec = datum.value<avro::GenericRecord>();
                                JsonValue::Object jobj;
                                for (size_t i = 0; i < rec.fieldCount(); ++i)
                                    jobj[rec.schema().nameAt(i)] = datum_to_json(rec.fieldAt(i));
                                return JsonValue(jobj);
                            }
                        case avro::AVRO_ENUM:
                            return JsonValue(datum.value<avro::GenericEnum>().symbol());
                        case avro::AVRO_FIXED:
                            {
                                const auto& fixed = datum.value<avro::GenericFixed>();
                                JsonValue::Array jarr;
                                for (size_t i = 0; i < fixed.value().size(); ++i)
                                    jarr.push_back(JsonValue(static_cast<int64_t>(fixed.value()[i])));
                                return JsonValue(jarr);
                            }
                        case avro::AVRO_UNION:
                            {
                                avro::GenericUnion union_val = datum.value<avro::GenericUnion>();
                                return datum_to_json(union_val.datum());
                            }
                        default:
                            return JsonValue(nullptr);
                    }
                }

                // JsonValue to Avro GenericDatum
                avro::GenericDatum json_to_datum(const JsonValue& jval, const avro::Schema& schema)
                {
                    avro::GenericDatum datum(schema);
                    switch (schema.type())
                    {
                        case avro::AVRO_NULL:
                            datum.value<avro::GenericNull>();
                            break;
                        case avro::AVRO_BOOL:
                            datum.value<bool>() = jval.as_bool();
                            break;
                        case avro::AVRO_INT:
                            datum.value<int32_t>() = static_cast<int32_t>(jval.as_int());
                            break;
                        case avro::AVRO_LONG:
                            datum.value<int64_t>() = jval.as_int();
                            break;
                        case avro::AVRO_FLOAT:
                            datum.value<float>() = static_cast<float>(jval.as_double());
                            break;
                        case avro::AVRO_DOUBLE:
                            datum.value<double>() = jval.as_double();
                            break;
                        case avro::AVRO_STRING:
                            datum.value<std::string>() = jval.as_string();
                            break;
                        case avro::AVRO_BYTES:
                            {
                                std::vector<uint8_t> bytes;
                                for (const auto& v : jval.as_array())
                                    bytes.push_back(static_cast<uint8_t>(v.as_int()));
                                datum.value<std::vector<uint8_t>>() = bytes;
                            }
                            break;
                        case avro::AVRO_ARRAY:
                            {
                                avro::GenericArray arr(schema);
                                for (const auto& v : jval.as_array())
                                    arr.value().push_back(json_to_datum(v, schema.root()->leafAt(0)));
                                datum.value<avro::GenericArray>() = arr;
                            }
                            break;
                        case avro::AVRO_MAP:
                            {
                                avro::GenericMap map(schema);
                                for (const auto& p : jval.as_object())
                                    map.value()[p.first] = json_to_datum(p.second, schema.root()->leafAt(0));
                                datum.value<avro::GenericMap>() = map;
                            }
                            break;
                        case avro::AVRO_RECORD:
                            {
                                avro::GenericRecord rec(schema);
                                for (size_t i = 0; i < schema.fields(); ++i)
                                {
                                    const std::string& fname = schema.nameAt(i);
                                    if (jval.contains(fname))
                                        rec.fieldAt(i) = json_to_datum(jval[fname], schema.fieldAt(i)->type());
                                }
                                datum.value<avro::GenericRecord>() = rec;
                            }
                            break;
                        case avro::AVRO_ENUM:
                            datum.value<avro::GenericEnum>() = avro::GenericEnum(schema, jval.as_string());
                            break;
                        case avro::AVRO_FIXED:
                            {
                                avro::GenericFixed fixed(schema);
                                auto& bytes = fixed.value();
                                const auto& arr = jval.as_array();
                                for (size_t i = 0; i < arr.size(); ++i)
                                    bytes[i] = static_cast<uint8_t>(arr[i].as_int());
                                datum.value<avro::GenericFixed>() = fixed;
                            }
                            break;
                        case avro::AVRO_UNION:
                            {
                                // For union, we need to determine which branch matches
                                // Simplified: use first branch if possible
                                avro::GenericUnion un(schema);
                                for (size_t i = 0; i < schema.types(); ++i)
                                {
                                    try
                                    {
                                        un.setBranch(i, json_to_datum(jval, schema.typeAt(i)));
                                        break;
                                    }
                                    catch (...) { continue; }
                                }
                                datum.value<avro::GenericUnion>() = un;
                            }
                            break;
                        default:
                            break;
                    }
                    return datum;
                }

                // Convert xarray to Avro array record
                template<typename T>
                avro::GenericDatum array_to_avro(const xarray_container<T>& arr)
                {
                    // Build a record schema: { "shape": array<int>, "data": array<T> }
                    avro::RecordSchema schema("Tensor");
                    schema.addField("shape", avro::ArraySchema(avro::IntSchema()));
                    if constexpr (std::is_same_v<T, double>)
                        schema.addField("data", avro::ArraySchema(avro::DoubleSchema()));
                    else if constexpr (std::is_same_v<T, float>)
                        schema.addField("data", avro::ArraySchema(avro::FloatSchema()));
                    else if constexpr (std::is_same_v<T, int64_t>)
                        schema.addField("data", avro::ArraySchema(avro::LongSchema()));
                    else
                        schema.addField("data", avro::ArraySchema(avro::DoubleSchema()));

                    avro::GenericRecord rec(schema);
                    // shape
                    avro::GenericArray shape_arr(schema.field("shape")->type());
                    for (auto s : arr.shape())
                        shape_arr.value().push_back(avro::GenericDatum(static_cast<int32_t>(s)));
                    rec.field("shape") = shape_arr;
                    // data
                    avro::GenericArray data_arr(schema.field("data")->type());
                    for (size_t i = 0; i < arr.size(); ++i)
                    {
                        if constexpr (std::is_same_v<T, double>)
                            data_arr.value().push_back(avro::GenericDatum(static_cast<double>(arr.flat(i))));
                        else if constexpr (std::is_same_v<T, float>)
                            data_arr.value().push_back(avro::GenericDatum(static_cast<float>(arr.flat(i))));
                        else if constexpr (std::is_same_v<T, int64_t>)
                            data_arr.value().push_back(avro::GenericDatum(static_cast<int64_t>(arr.flat(i))));
                        else
                            data_arr.value().push_back(avro::GenericDatum(static_cast<double>(arr.flat(i))));
                    }
                    rec.field("data") = data_arr;
                    return avro::GenericDatum(rec);
                }

                // Avro record to xarray
                template<typename T>
                xarray_container<T> avro_to_array(const avro::GenericDatum& datum)
                {
                    if (datum.type() != avro::AVRO_RECORD)
                        throw std::runtime_error("Expected record datum");
                    const avro::GenericRecord& rec = datum.value<avro::GenericRecord>();
                    
                    // Get shape
                    const avro::GenericArray& shape_arr = rec.field("shape").value<avro::GenericArray>();
                    std::vector<size_t> shape;
                    for (const auto& d : shape_arr.value())
                        shape.push_back(static_cast<size_t>(d.value<int32_t>()));
                    
                    // Get data
                    const avro::GenericArray& data_arr = rec.field("data").value<avro::GenericArray>();
                    xarray_container<T> result(shape);
                    for (size_t i = 0; i < data_arr.value().size(); ++i)
                    {
                        const auto& d = data_arr.value()[i];
                        if constexpr (std::is_same_v<T, double>)
                            result.flat(i) = static_cast<T>(d.value<double>());
                        else if constexpr (std::is_same_v<T, float>)
                            result.flat(i) = static_cast<T>(d.value<float>());
                        else if constexpr (std::is_same_v<T, int64_t>)
                            result.flat(i) = static_cast<T>(d.value<int64_t>());
                        else
                            result.flat(i) = static_cast<T>(d.value<double>());
                    }
                    return result;
                }
            }
#endif // XTENSOR_HAS_AVRO

            // --------------------------------------------------------------------
            // Avro Writer
            // --------------------------------------------------------------------
            class AvroWriter
            {
            public:
                AvroWriter() = default;

                void open(const std::string& filename, const AvroField& schema = {})
                {
#if XTENSOR_HAS_AVRO
                    m_filename = filename;
                    if (schema.type != AvroType::NULL_TYPE)
                    {
                        m_schema = avro_detail::build_schema(schema);
                    }
                    // Writer will be created on first write or explicitly
#else
                    XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
#endif
                }

                void open_with_json_schema(const std::string& filename, const std::string& json_schema)
                {
#if XTENSOR_HAS_AVRO
                    m_filename = filename;
                    m_schema = avro::compileJsonSchemaFromString(json_schema);
#else
                    XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
#endif
                }

                template<typename T>
                void write(const xarray_container<T>& data)
                {
#if XTENSOR_HAS_AVRO
                    if (!m_writer)
                    {
                        if (!m_schema.valid())
                        {
                            // Auto-generate schema for tensor
                            m_schema = avro_detail::array_to_avro(data).schema();
                        }
                        m_writer = std::make_unique<avro::DataFileWriter<avro::GenericDatum>>(
                            m_filename.c_str(), m_schema);
                    }
                    auto datum = avro_detail::array_to_avro(data);
                    m_writer->write(datum);
#else
                    XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
#endif
                }

                void write_generic(const JsonValue& record)
                {
#if XTENSOR_HAS_AVRO
                    if (!m_writer)
                        throw std::runtime_error("Writer not initialized with schema");
                    auto datum = avro_detail::json_to_datum(record, m_schema);
                    m_writer->write(datum);
#else
                    XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
#endif
                }

                void close()
                {
#if XTENSOR_HAS_AVRO
                    if (m_writer)
                    {
                        m_writer->close();
                        m_writer.reset();
                    }
#endif
                }

                ~AvroWriter()
                {
                    close();
                }

            private:
                std::string m_filename;
#if XTENSOR_HAS_AVRO
                avro::ValidSchema m_schema;
                std::unique_ptr<avro::DataFileWriter<avro::GenericDatum>> m_writer;
#endif
            };

            // --------------------------------------------------------------------
            // Avro Reader
            // --------------------------------------------------------------------
            class AvroReader
            {
            public:
                AvroReader() = default;

                void open(const std::string& filename)
                {
#if XTENSOR_HAS_AVRO
                    m_filename = filename;
                    m_reader = std::make_unique<avro::DataFileReader<avro::GenericDatum>>(filename.c_str());
#else
                    XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
#endif
                }

                bool has_next() const
                {
#if XTENSOR_HAS_AVRO
                    return m_reader && m_reader->hasMore();
#else
                    return false;
#endif
                }

                template<typename T>
                xarray_container<T> read_next()
                {
#if XTENSOR_HAS_AVRO
                    if (!m_reader || !m_reader->hasMore())
                        throw std::runtime_error("No more records");
                    avro::GenericDatum datum;
                    m_reader->read(datum);
                    return avro_detail::avro_to_array<T>(datum);
#else
                    XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
                    return {};
#endif
                }

                JsonValue read_next_generic()
                {
#if XTENSOR_HAS_AVRO
                    if (!m_reader || !m_reader->hasMore())
                        throw std::runtime_error("No more records");
                    avro::GenericDatum datum;
                    m_reader->read(datum);
                    return avro_detail::datum_to_json(datum);
#else
                    XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
                    return JsonValue();
#endif
                }

                template<typename T>
                std::vector<xarray_container<T>> read_all()
                {
                    std::vector<xarray_container<T>> result;
                    while (has_next())
                        result.push_back(read_next<T>());
                    return result;
                }

                avro::ValidSchema get_schema() const
                {
#if XTENSOR_HAS_AVRO
                    if (!m_reader) throw std::runtime_error("Reader not open");
                    return m_reader->readerSchema();
#else
                    throw std::runtime_error("Avro support not compiled");
#endif
                }

                void close()
                {
#if XTENSOR_HAS_AVRO
                    if (m_reader)
                    {
                        m_reader->close();
                        m_reader.reset();
                    }
#endif
                }

                ~AvroReader()
                {
                    close();
                }

            private:
                std::string m_filename;
#if XTENSOR_HAS_AVRO
                std::unique_ptr<avro::DataFileReader<avro::GenericDatum>> m_reader;
#endif
            };

            // --------------------------------------------------------------------
            // Convenience functions
            // --------------------------------------------------------------------
            template<typename T>
            inline void save_avro(const std::string& filename, const xarray_container<T>& data)
            {
                AvroWriter writer;
                writer.open(filename);
                writer.write(data);
                writer.close();
            }

            template<typename T>
            inline void save_avro_batch(const std::string& filename, const std::vector<xarray_container<T>>& batches)
            {
                AvroWriter writer;
                writer.open(filename);
                for (const auto& arr : batches)
                    writer.write(arr);
                writer.close();
            }

            template<typename T>
            inline std::vector<xarray_container<T>> load_avro(const std::string& filename)
            {
                AvroReader reader;
                reader.open(filename);
                return reader.read_all<T>();
            }

            inline JsonValue load_avro_schema(const std::string& filename)
            {
#if XTENSOR_HAS_AVRO
                AvroReader reader;
                reader.open(filename);
                auto schema = reader.get_schema();
                return JsonParser::parse(schema.toJson());
#else
                XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
                return JsonValue();
#endif
            }

            // Write multiple named arrays as Avro records (each record is a map)
            inline void save_avro_dict(const std::string& filename,
                                       const std::map<std::string, xarray_container<double>>& dict)
            {
                // Build schema: record with fields for each key (all arrays)
                AvroField record_schema{"", AvroType::RECORD};
                for (const auto& p : dict)
                {
                    AvroField field;
                    field.name = p.first;
                    field.type = AvroType::ARRAY;
                    field.items_type = AvroType::DOUBLE;
                    record_schema.fields.push_back(field);
                }
                AvroWriter writer;
                writer.open(filename, record_schema);
                // Create a JSON record
                JsonValue::Object jobj;
                for (const auto& p : dict)
                {
                    JsonValue::Array jarr;
                    for (size_t i = 0; i < p.second.size(); ++i)
                        jarr.push_back(JsonValue(p.second.flat(i)));
                    jobj[p.first] = JsonValue(jarr);
                }
                writer.write_generic(JsonValue(jobj));
                writer.close();
            }

            inline std::map<std::string, xarray_container<double>> load_avro_dict(const std::string& filename)
            {
                std::map<std::string, xarray_container<double>> result;
#if XTENSOR_HAS_AVRO
                AvroReader reader;
                reader.open(filename);
                if (!reader.has_next())
                    return result;
                JsonValue rec = reader.read_next_generic();
                if (rec.is_object())
                {
                    for (const auto& p : rec.as_object())
                    {
                        if (p.second.is_array())
                        {
                            const auto& arr = p.second.as_array();
                            xarray_container<double> vec({arr.size()});
                            for (size_t i = 0; i < arr.size(); ++i)
                                vec(i) = arr[i].as_double();
                            result[p.first] = vec;
                        }
                    }
                }
#endif
                return result;
            }

        } // namespace io

        // Bring Avro functions into xt namespace
        using io::AvroWriter;
        using io::AvroReader;
        using io::save_avro;
        using io::save_avro_batch;
        using io::load_avro;
        using io::load_avro_schema;
        using io::save_avro_dict;
        using io::load_avro_dict;
        using io::AvroField;
        using io::AvroType;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XIO_AVRO_HPP

// io/xio_avro.hpp