// io/xhdf5.hpp

#ifndef XTENSOR_XHDF5_HPP
#define XTENSOR_XHDF5_HPP

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

// HDF5 support detection
#if __has_include(<H5Cpp.h>)
    #define XTENSOR_HAS_HDF5 1
    #include <H5Cpp.h>
#else
    #define XTENSOR_HAS_HDF5 0
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace io
        {
            // --------------------------------------------------------------------
            // HDF5 type mapping
            // --------------------------------------------------------------------
            namespace hdf5_detail
            {
#if XTENSOR_HAS_HDF5
                // Get HDF5 datatype from C++ type
                template<typename T>
                struct hdf5_type
                {
                    static H5::DataType get() { return H5::PredType::NATIVE_DOUBLE; }
                };
                
                template<> struct hdf5_type<float> { static H5::DataType get() { return H5::PredType::NATIVE_FLOAT; } };
                template<> struct hdf5_type<double> { static H5::DataType get() { return H5::PredType::NATIVE_DOUBLE; } };
                template<> struct hdf5_type<int8_t> { static H5::DataType get() { return H5::PredType::NATIVE_INT8; } };
                template<> struct hdf5_type<int16_t> { static H5::DataType get() { return H5::PredType::NATIVE_INT16; } };
                template<> struct hdf5_type<int32_t> { static H5::DataType get() { return H5::PredType::NATIVE_INT32; } };
                template<> struct hdf5_type<int64_t> { static H5::DataType get() { return H5::PredType::NATIVE_INT64; } };
                template<> struct hdf5_type<uint8_t> { static H5::DataType get() { return H5::PredType::NATIVE_UINT8; } };
                template<> struct hdf5_type<uint16_t> { static H5::DataType get() { return H5::PredType::NATIVE_UINT16; } };
                template<> struct hdf5_type<uint32_t> { static H5::DataType get() { return H5::PredType::NATIVE_UINT32; } };
                template<> struct hdf5_type<uint64_t> { static H5::DataType get() { return H5::PredType::NATIVE_UINT64; } };
                template<> struct hdf5_type<bool> { static H5::DataType get() { return H5::PredType::NATIVE_HBOOL; } };
                template<> struct hdf5_type<std::complex<float>> { static H5::DataType get() { return H5::CompType(sizeof(std::complex<float>)); } };
                template<> struct hdf5_type<std::complex<double>> { static H5::DataType get() { return H5::CompType(sizeof(std::complex<double>)); } };
                template<> struct hdf5_type<std::string> { static H5::DataType get() { return H5::StrType(H5::PredType::C_S1, H5T_VARIABLE); } };

                // Helper to create dataspace from shape
                inline H5::DataSpace create_dataspace(const std::vector<hsize_t>& shape)
                {
                    if (shape.empty())
                        return H5::DataSpace(H5S_SCALAR);
                    return H5::DataSpace(static_cast<int>(shape.size()), shape.data());
                }

                // Helper to get shape from dataspace
                inline std::vector<size_t> get_shape(const H5::DataSpace& dspace)
                {
                    int ndims = dspace.getSimpleExtentNdims();
                    if (ndims == 0) return {};
                    std::vector<hsize_t> dims(static_cast<size_t>(ndims));
                    dspace.getSimpleExtentDims(dims.data());
                    std::vector<size_t> shape;
                    for (auto d : dims) shape.push_back(static_cast<size_t>(d));
                    return shape;
                }

                // Read attribute value
                template<typename T>
                inline T read_attribute(const H5::Attribute& attr)
                {
                    T value;
                    attr.read(hdf5_type<T>::get(), &value);
                    return value;
                }

                template<>
                inline std::string read_attribute<std::string>(const H5::Attribute& attr)
                {
                    H5::StrType stype = attr.getStrType();
                    std::string value;
                    attr.read(stype, value);
                    return value;
                }

                // Write attribute
                template<typename T>
                inline void write_attribute(H5::H5Object& obj, const std::string& name, const T& value)
                {
                    H5::DataSpace scalar(H5S_SCALAR);
                    H5::Attribute attr = obj.createAttribute(name, hdf5_type<T>::get(), scalar);
                    attr.write(hdf5_type<T>::get(), &value);
                }

                template<>
                inline void write_attribute<std::string>(H5::H5Object& obj, const std::string& name, const std::string& value)
                {
                    H5::StrType stype(H5::PredType::C_S1, H5T_VARIABLE);
                    H5::DataSpace scalar(H5S_SCALAR);
                    H5::Attribute attr = obj.createAttribute(name, stype, scalar);
                    attr.write(stype, value);
                }

                // Read dataset into xarray
                template<typename T>
                inline xarray_container<T> read_dataset(const H5::DataSet& dataset)
                {
                    H5::DataSpace dspace = dataset.getSpace();
                    auto shape = get_shape(dspace);
                    
                    xarray_container<T> result(shape);
                    if (result.size() > 0)
                        dataset.read(result.data(), hdf5_type<T>::get());
                    return result;
                }

                // Write xarray to dataset
                template<typename T>
                inline void write_dataset(H5::CommonFG& parent, const std::string& name,
                                          const xarray_container<T>& arr, bool compress = true)
                {
                    std::vector<hsize_t> dims;
                    for (auto s : arr.shape()) dims.push_back(static_cast<hsize_t>(s));
                    H5::DataSpace dspace = create_dataspace(dims);
                    
                    H5::DSetCreatPropList plist;
                    if (compress && arr.size() > 1024)
                    {
                        dims[0] = std::min(static_cast<hsize_t>(1024), dims[0]);
                        plist.setChunk(static_cast<int>(dims.size()), dims.data());
                        plist.setDeflate(6);
                    }
                    
                    H5::DataSet dataset = parent.createDataSet(name, hdf5_type<T>::get(), dspace, plist);
                    if (arr.size() > 0)
                        dataset.write(arr.data(), hdf5_type<T>::get());
                }

                // Recursively create groups
                inline H5::Group create_groups(H5::CommonFG& parent, const std::string& path)
                {
                    if (path.empty() || path == "/") return parent.openGroup("/");
                    H5::Group current = parent.openGroup("/");
                    std::istringstream iss(path);
                    std::string token;
                    while (std::getline(iss, token, '/'))
                    {
                        if (token.empty()) continue;
                        try
                        {
                            current = current.openGroup(token);
                        }
                        catch (H5::Exception&)
                        {
                            current = current.createGroup(token);
                        }
                    }
                    return current;
                }

                // Recursively open groups
                inline H5::Group open_groups(H5::CommonFG& parent, const std::string& path)
                {
                    if (path.empty() || path == "/") return parent.openGroup("/");
                    H5::Group current = parent.openGroup("/");
                    std::istringstream iss(path);
                    std::string token;
                    while (std::getline(iss, token, '/'))
                    {
                        if (token.empty()) continue;
                        current = current.openGroup(token);
                    }
                    return current;
                }

                // List objects in group
                inline std::vector<std::string> list_objects(H5::CommonFG& group, H5O_type_t obj_type = H5O_TYPE_ALL)
                {
                    std::vector<std::string> result;
                    hsize_t num_objs = group.getNumObjs();
                    for (hsize_t i = 0; i < num_objs; ++i)
                    {
                        std::string name = group.getObjnameByIdx(i);
                        if (obj_type == H5O_TYPE_ALL || group.childObjType(name) == obj_type)
                            result.push_back(name);
                    }
                    return result;
                }

                // Check if dataset exists
                inline bool exists(H5::CommonFG& parent, const std::string& name)
                {
                    try
                    {
                        parent.openDataSet(name);
                        return true;
                    }
                    catch (H5::Exception&)
                    {
                        return false;
                    }
                }

                // Check if group exists
                inline bool group_exists(H5::CommonFG& parent, const std::string& name)
                {
                    try
                    {
                        parent.openGroup(name);
                        return true;
                    }
                    catch (H5::Exception&)
                    {
                        return false;
                    }
                }
#endif // XTENSOR_HAS_HDF5
            } // namespace hdf5_detail

            // --------------------------------------------------------------------
            // HDF5 File class
            // --------------------------------------------------------------------
            class HDF5File
            {
            public:
                enum Mode
                {
                    ReadOnly,
                    ReadWrite,
                    Create,
                    Truncate
                };

                HDF5File() = default;
                
                explicit HDF5File(const std::string& filename, Mode mode = ReadOnly)
                {
                    open(filename, mode);
                }

                ~HDF5File()
                {
                    close();
                }

                void open(const std::string& filename, Mode mode = ReadOnly)
                {
#if XTENSOR_HAS_HDF5
                    close();
                    m_filename = filename;
                    m_mode = mode;
                    
                    try
                    {
                        H5::Exception::dontPrint();
                        
                        unsigned int flags = 0;
                        switch (mode)
                        {
                            case ReadOnly:
                                flags = H5F_ACC_RDONLY;
                                break;
                            case ReadWrite:
                                flags = H5F_ACC_RDWR;
                                break;
                            case Create:
                                flags = H5F_ACC_EXCL;
                                break;
                            case Truncate:
                                flags = H5F_ACC_TRUNC;
                                break;
                        }
                        
                        if (mode == Create || mode == Truncate)
                            m_file = std::make_unique<H5::H5File>(filename, flags);
                        else
                            m_file = std::make_unique<H5::H5File>(filename, flags);
                    }
                    catch (H5::FileIException& e)
                    {
                        if (mode == ReadWrite)
                        {
                            // Try to create if doesn't exist
                            m_file = std::make_unique<H5::H5File>(filename, H5F_ACC_TRUNC);
                        }
                        else
                        {
                            throw std::runtime_error("HDF5 file open error: " + std::string(e.getCDetailMsg()));
                        }
                    }
#else
                    XTENSOR_THROW(std::runtime_error, "HDF5 support not compiled");
#endif
                }

                void close()
                {
#if XTENSOR_HAS_HDF5
                    if (m_file)
                    {
                        m_file->close();
                        m_file.reset();
                    }
                    m_attributes.clear();
#endif
                }

                bool is_open() const
                {
#if XTENSOR_HAS_HDF5
                    return m_file != nullptr;
#else
                    return false;
#endif
                }

                // --------------------------------------------------------------------
                // Dataset operations
                // --------------------------------------------------------------------
                template<typename T>
                xarray_container<T> read_dataset(const std::string& path) const
                {
#if XTENSOR_HAS_HDF5
                    if (!m_file) XTENSOR_THROW(std::runtime_error, "HDF5 file not open");
                    try
                    {
                        H5::DataSet dataset = m_file->openDataSet(path);
                        return hdf5_detail::read_dataset<T>(dataset);
                    }
                    catch (H5::Exception& e)
                    {
                        XTENSOR_THROW(std::runtime_error, "HDF5 read error: " + std::string(e.getCDetailMsg()));
                    }
#else
                    XTENSOR_THROW(std::runtime_error, "HDF5 support not compiled");
                    return xarray_container<T>();
#endif
                }

                template<typename T>
                void write_dataset(const std::string& path, const xarray_container<T>& arr,
                                   bool compress = true)
                {
#if XTENSOR_HAS_HDF5
                    if (!m_file) XTENSOR_THROW(std::runtime_error, "HDF5 file not open");
                    if (m_mode == ReadOnly) XTENSOR_THROW(std::runtime_error, "File opened in read-only mode");
                    
                    // Split path into group path and dataset name
                    size_t pos = path.rfind('/');
                    std::string group_path = (pos == std::string::npos) ? "/" : path.substr(0, pos);
                    std::string dataset_name = (pos == std::string::npos) ? path : path.substr(pos + 1);
                    
                    try
                    {
                        H5::Group group = hdf5_detail::create_groups(*m_file, group_path);
                        hdf5_detail::write_dataset(group, dataset_name, arr, compress);
                    }
                    catch (H5::Exception& e)
                    {
                        XTENSOR_THROW(std::runtime_error, "HDF5 write error: " + std::string(e.getCDetailMsg()));
                    }
#else
                    XTENSOR_THROW(std::runtime_error, "HDF5 support not compiled");
#endif
                }

                // Check if dataset exists
                bool dataset_exists(const std::string& path) const
                {
#if XTENSOR_HAS_HDF5
                    if (!m_file) return false;
                    return hdf5_detail::exists(*m_file, path);
#else
                    return false;
#endif
                }

                // List datasets in a group
                std::vector<std::string> list_datasets(const std::string& group_path = "/") const
                {
#if XTENSOR_HAS_HDF5
                    if (!m_file) return {};
                    try
                    {
                        H5::Group group = hdf5_detail::open_groups(*m_file, group_path);
                        return hdf5_detail::list_objects(group, H5O_TYPE_DATASET);
                    }
                    catch (H5::Exception&)
                    {
                        return {};
                    }
#else
                    return {};
#endif
                }

                // List groups
                std::vector<std::string> list_groups(const std::string& group_path = "/") const
                {
#if XTENSOR_HAS_HDF5
                    if (!m_file) return {};
                    try
                    {
                        H5::Group group = hdf5_detail::open_groups(*m_file, group_path);
                        return hdf5_detail::list_objects(group, H5O_TYPE_GROUP);
                    }
                    catch (H5::Exception&)
                    {
                        return {};
                    }
#else
                    return {};
#endif
                }

                // Create group
                void create_group(const std::string& path)
                {
#if XTENSOR_HAS_HDF5
                    if (!m_file) XTENSOR_THROW(std::runtime_error, "HDF5 file not open");
                    if (m_mode == ReadOnly) XTENSOR_THROW(std::runtime_error, "File opened in read-only mode");
                    hdf5_detail::create_groups(*m_file, path);
#else
                    XTENSOR_THROW(std::runtime_error, "HDF5 support not compiled");
#endif
                }

                // --------------------------------------------------------------------
                // Attribute operations (on root group)
                // --------------------------------------------------------------------
                template<typename T>
                void set_attribute(const std::string& name, const T& value)
                {
#if XTENSOR_HAS_HDF5
                    if (!m_file) XTENSOR_THROW(std::runtime_error, "HDF5 file not open");
                    if (m_mode == ReadOnly) XTENSOR_THROW(std::runtime_error, "File opened in read-only mode");
                    hdf5_detail::write_attribute(*m_file, name, value);
#else
                    XTENSOR_THROW(std::runtime_error, "HDF5 support not compiled");
#endif
                }

                template<typename T>
                T get_attribute(const std::string& name, const T& default_val = T()) const
                {
#if XTENSOR_HAS_HDF5
                    if (!m_file) return default_val;
                    try
                    {
                        H5::Attribute attr = m_file->openAttribute(name);
                        return hdf5_detail::read_attribute<T>(attr);
                    }
                    catch (H5::Exception&)
                    {
                        return default_val;
                    }
#else
                    return default_val;
#endif
                }

                // --------------------------------------------------------------------
                // Read all datasets into a map
                // --------------------------------------------------------------------
                std::map<std::string, xarray_container<double>> read_all_datasets() const
                {
                    std::map<std::string, xarray_container<double>> result;
#if XTENSOR_HAS_HDF5
                    auto datasets = list_datasets("/");
                    for (const auto& ds : datasets)
                    {
                        try
                        {
                            result[ds] = read_dataset<double>("/" + ds);
                        }
                        catch (...)
                        {
                            // Skip if can't read as double
                        }
                    }
#endif
                    return result;
                }

            private:
#if XTENSOR_HAS_HDF5
                std::unique_ptr<H5::H5File> m_file;
                std::string m_filename;
                Mode m_mode = ReadOnly;
                std::map<std::string, std::string> m_attributes;
#endif
            };

            // --------------------------------------------------------------------
            // Convenience functions
            // --------------------------------------------------------------------
            template<typename T>
            inline xarray_container<T> load_hdf5(const std::string& filename, const std::string& dataset_path)
            {
                HDF5File file(filename, HDF5File::ReadOnly);
                return file.read_dataset<T>(dataset_path);
            }

            inline auto load_hdf5(const std::string& filename)
            {
                HDF5File file(filename, HDF5File::ReadOnly);
                return file.read_all_datasets();
            }

            template<typename T>
            inline void save_hdf5(const std::string& filename, const std::string& dataset_path,
                                  const xarray_container<T>& arr, bool compress = true)
            {
                HDF5File file(filename, HDF5File::Create);
                file.write_dataset(dataset_path, arr, compress);
            }

            inline void save_hdf5(const std::string& filename,
                                  const std::map<std::string, xarray_container<double>>& arrays,
                                  bool compress = true)
            {
                HDF5File file(filename, HDF5File::Create);
                for (const auto& p : arrays)
                    file.write_dataset(p.first, p.second, compress);
            }

            // --------------------------------------------------------------------
            // HDF5 attribute utilities
            // --------------------------------------------------------------------
            class HDF5Attributes
            {
            public:
                explicit HDF5Attributes(HDF5File& file) : m_file(file) {}
                
                template<typename T>
                void set(const std::string& key, const T& value) { m_file.set_attribute(key, value); }
                
                template<typename T>
                T get(const std::string& key, const T& default_val = T()) const
                {
                    return m_file.get_attribute(key, default_val);
                }
                
            private:
                HDF5File& m_file;
            };

        } // namespace io

        // Bring HDF5 functions into xt namespace
        using io::HDF5File;
        using io::load_hdf5;
        using io::save_hdf5;
        using io::HDF5Attributes;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XHDF5_HPP

// io/xhdf5.hpp