//File 0055 : core/xexception.hpp
//Exception handling with custom xtensor exception types, context-aware error messages, and conditional throw macros for bounds checking and shape validation.
#ifndef XTENSOR_XEXCEPTION_HPP
#define XTENSOR_XEXCEPTION_HPP

#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"

namespace xt
{
    /**
     * @class xexception
     * @brief Base exception class for all xtensor errors.
     *
     * Provides a what() message and optional context information (file, line,
     * function name) for debugging.
     */
    class xexception : public std::runtime_error
    {
    public:
        explicit xexception(const std::string& message)
            : std::runtime_error(message)
        {
        }

        xexception(const std::string& message, const char* file, int line, const char* func)
            : std::runtime_error(format_message(message, file, line, func))
        {
        }

    private:
        static std::string format_message(const std::string& msg, const char* file, int line, const char* func)
        {
            std::ostringstream oss;
            oss << msg << "\n  [at " << file << ":" << line << " in " << func << "]";
            return oss.str();
        }
    };

    /**
     * @class xout_of_range
     * @brief Exception thrown when an index is out of bounds.
     */
    class xout_of_range : public xexception
    {
    public:
        template <class... Args>
        explicit xout_of_range(const std::string& message, Args&&... args)
            : xexception(format_details(message, std::forward<Args>(args)...))
        {
        }

    private:
        template <class... Args>
        static std::string format_details(const std::string& msg, Args&&... args)
        {
            std::ostringstream oss;
            oss << msg;
            ((oss << " [" << args << "]"), ...);
            return oss.str();
        }
    };

    /**
     * @class xshape_error
     * @brief Exception thrown for shape-related errors (mismatch, broadcasting failure).
     */
    class xshape_error : public xexception
    {
    public:
        explicit xshape_error(const std::string& message)
            : xexception(message)
        {
        }

        template <class S1, class S2>
        static xshape_error incompatible_shapes(const S1& s1, const S2& s2)
        {
            std::ostringstream oss;
            oss << "Incompatible shapes: [";
            for (std::size_t i = 0; i < s1.size(); ++i)
            {
                oss << s1[i];
                if (i + 1 < s1.size()) oss << ", ";
            }
            oss << "] vs [";
            for (std::size_t i = 0; i < s2.size(); ++i)
            {
                oss << s2[i];
                if (i + 1 < s2.size()) oss << ", ";
            }
            oss << "]";
            return xshape_error(oss.str());
        }
    };

    /**
     * @class xbroadcast_error
     * @brief Exception thrown when broadcasting fails.
     */
    class xbroadcast_error : public xshape_error
    {
    public:
        explicit xbroadcast_error(const std::string& message)
            : xshape_error(message)
        {
        }
    };

    /**
     * @class xiterator_error
     * @brief Exception thrown for invalid iterator operations.
     */
    class xiterator_error : public xexception
    {
    public:
        explicit xiterator_error(const std::string& message)
            : xexception(message)
        {
        }
    };

    /**
     * @class xlayout_error
     * @brief Exception thrown for invalid layout operations.
     */
    class xlayout_error : public xexception
    {
    public:
        explicit xlayout_error(const std::string& message)
            : xexception(message)
        {
        }
    };

    /**
     * @class xsimd_error
     * @brief Exception thrown when SIMD operations fail.
     */
    class xsimd_error : public xexception
    {
    public:
        explicit xsimd_error(const std::string& message)
            : xexception(message)
        {
        }
    };

    /**
     * @class xio_error
     * @brief Exception thrown for I/O errors.
     */
    class xio_error : public xexception
    {
    public:
        explicit xio_error(const std::string& message)
            : xexception(message)
        {
        }
    };

    /*********************************************
     * Conditional throw macros (optional)
     *********************************************/
    #if !defined(XTENSOR_NO_BOUNDS_CHECK)
        #define XTENSOR_ASSERT(condition, message) \
            do { \
                if (!(condition)) \
                    throw ::xt::xexception(message, __FILE__, __LINE__, __func__); \
            } while (false)

        #define XTENSOR_ASSERT_SHAPE(condition, shape1, shape2) \
            do { \
                if (!(condition)) \
                    throw ::xt::xshape_error::incompatible_shapes(shape1, shape2); \
            } while (false)

        #define XTENSOR_BOUNDS_CHECK(index, shape) \
            do { \
                if (index >= shape) \
                    throw ::xt::xout_of_range("Index out of bounds", index, shape); \
            } while (false)
    #else
        #define XTENSOR_ASSERT(condition, message) ((void)0)
        #define XTENSOR_ASSERT_SHAPE(condition, shape1, shape2) ((void)0)
        #define XTENSOR_BOUNDS_CHECK(index, shape) ((void)0)
    #endif

    /**
     * Helper function to throw a detailed exception with index context.
     */
    template <class Index, class Shape>
    inline void check_index_bounds(const Index& idx, const Shape& shape)
    {
        for (std::size_t i = 0; i < idx.size(); ++i)
        {
            if (idx[i] >= shape[i])
            {
                std::ostringstream oss;
                oss << "Index [" << idx[i] << "] out of bounds for dimension "
                    << i << " with size " << shape[i];
                throw xout_of_range(oss.str());
            }
        }
    }

    /**
     * Helper function to validate that two shapes are broadcast-compatible.
     */
    template <class S1, class S2>
    inline void check_broadcast_shapes(const S1& s1, const S2& s2)
    {
        if (s1.size() != s2.size())
            throw xbroadcast_error("Broadcast shapes must have same rank.");
        for (std::size_t i = 0; i < s1.size(); ++i)
        {
            if (s1[i] != s2[i] && s1[i] != 1 && s2[i] != 1)
                throw xshape_error::incompatible_shapes(s1, s2);
        }
    }

    /**
     * Helper function to validate that an axis index is within range.
     */
    template <class Shape>
    inline void check_axis_bounds(std::ptrdiff_t axis, const Shape& shape)
    {
        std::size_t ndim = shape.size();
        if (axis < 0) axis += static_cast<std::ptrdiff_t>(ndim);
        if (axis < 0 || static_cast<std::size_t>(axis) >= ndim)
            throw xout_of_range("Axis out of bounds", axis, ndim);
    }

    /**
     * Helper function to validate that a dimension size is positive.
     */
    inline void check_dimension_positive(std::size_t dim, const char* context = "")
    {
        if (dim == 0)
        {
            std::ostringstream oss;
            oss << context << ": dimension must be positive.";
            throw xshape_error(oss.str());
        }
    }

} // namespace xt

#endif // XTENSOR_XEXCEPTION_HPP