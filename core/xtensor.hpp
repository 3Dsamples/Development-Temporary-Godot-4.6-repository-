// core/xtensor.hpp
#ifndef XTENSOR_HPP
#define XTENSOR_HPP

// ----------------------------------------------------------------------------
// xtensor.hpp – Main convenience header for xtensor + BigNumber + FFT
// ----------------------------------------------------------------------------
// Includes all essential components of xtensor, configured to use
// bignumber::BigNumber as the default value type, with FFT‑accelerated
// arithmetic fully integrated. This header is sufficient for most use cases
// in scientific simulation, real‑time animation, and multi‑scale physics.
// ----------------------------------------------------------------------------

// Configuration and forward declarations
#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"

// Core expression system
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xbroadcast.hpp"

// Container types
#include "xarray.hpp"
#include "xtensor.hpp"      // fixed-size tensor container (defined elsewhere)

// Views and adaptors
#include "xview.hpp"
#include "xstrided_view.hpp"
#include "xdynamic_view.hpp"
#include "xarray_adaptor.hpp"
#include "xtensor_adaptor.hpp"

// Reducers and accumulators
#include "xreducer.hpp"
#include "xaccumulator.hpp"

// Mathematical functions
#include "xmath.hpp"

// Sorting and searching
#include "xsorting.hpp"

// Random number generation
#include "xrandom.hpp"

// BLAS / LAPACK interface
#include "xblas.hpp"

// Linear algebra
#include "xlinalg.hpp"

// Norms
#include "xnorm.hpp"

// Missing data support
#include "xmissing.hpp"

// Statistics
#include "xstats.hpp"

// Metrics
#include "xmetrics.hpp"

// Graphics context (for visualisation)
#include "xgraphics.hpp"

// Transforms (FFT, DCT, etc.)
#include "xtransform.hpp"

// Material properties
#include "xmaterial.hpp"

// Intersection tests
#include "xintersection.hpp"

// Mesh and geometry
#include "xmesh.hpp"

// Renderer integration
#include "xrenderer.hpp"

// FFT module (explicit spectral operations)
#include "fft.hpp"

// Filters and windows
#include "lfilter.hpp"
#include "xwindows.hpp"

// Wavelet transforms
#include "xwavelet.hpp"

// Image processing
#include "ximage_processing.hpp"

// Texture compression
#include "xtexture_compressor.hpp"

// Interpolation
#include "xinterp.hpp"

// Optimization
#include "xoptimize.hpp"

// Integration
#include "xintegrate.hpp"

// Decomposition (matrix factorizations)
#include "xdecomposition.hpp"

// Clustering
#include "xcluster.hpp"

// Sparse matrix formats
#include "xcoo_scheme.hpp"
#include "xcsr_scheme.hpp"

// Image container
#include "ximage.hpp"

// I/O modules
#include "xnpz.hpp"
#include "xaudio.hpp"
#include "xhdf5.hpp"
#include "xio_json.hpp"
#include "xio_parquet.hpp"
#include "xio_avro.hpp"
#include "xio_xml.hpp"

// Variant support
#include "xvariant.hpp"

// ClassDB (reflection / serialization registry)
#include "xclassdb.hpp"

// Resource management
#include "xresource.hpp"

// ----------------------------------------------------------------------------
// BigNumber and FFT integration – finalisation
// ----------------------------------------------------------------------------
// The following ensures that all necessary specialisations for
// bignumber::BigNumber are available and that FFT‑accelerated multiplication
// is correctly dispatched throughout the library.

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    // ------------------------------------------------------------------------
    // Additional factory functions that leverage BigNumber
    // ------------------------------------------------------------------------
    template <class T = value_type, layout_type L = DEFAULT_LAYOUT>
    inline xarray_container<T, L> arange(T start, T stop, T step = T(1))
    {
        // TODO: implement arange with BigNumber support
        (void)start; (void)stop; (void)step;
        return xarray_container<T, L>();
    }

    template <class T = value_type, layout_type L = DEFAULT_LAYOUT>
    inline xarray_container<T, L> arange(T stop)
    {
        return arange(T(0), stop, T(1));
    }

    template <class T = value_type, layout_type L = DEFAULT_LAYOUT>
    inline xarray_container<T, L> linspace(T start, T stop, size_type num = 50, bool endpoint = true)
    {
        // TODO: implement linspace
        (void)start; (void)stop; (void)num; (void)endpoint;
        return xarray_container<T, L>();
    }

    template <class T = value_type, layout_type L = DEFAULT_LAYOUT>
    inline xarray_container<T, L> logspace(T start, T stop, size_type num = 50, T base = T(10), bool endpoint = true)
    {
        auto exponents = linspace(start, stop, num, endpoint);
        return xt::pow(base, exponents);
    }

    // ------------------------------------------------------------------------
    // Convenience function for FFT‑accelerated matrix multiplication
    // ------------------------------------------------------------------------
    template <class E1, class E2>
    inline auto matmul(const xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        // Matrix multiplication for BigNumber uses FFT‑based convolution
        // internally when dimensions are large.
        return linalg::dot(e1, e2);
    }

    // ------------------------------------------------------------------------
    // Ensure the FFT module is fully integrated
    // ------------------------------------------------------------------------
    namespace fft
    {
        // Re‑export common FFT functions for convenience
        using ::xt::fft::fft;
        using ::xt::fft::ifft;
        using ::xt::fft::rfft;
        using ::xt::fft::irfft;
        using ::xt::fft::fftn;
        using ::xt::fft::ifftn;
        using ::xt::fft::convolve;
        using ::xt::fft::correlate;
    }

} // namespace xt

// ----------------------------------------------------------------------------
// Bring important types and functions into the global xt namespace for ease
// ----------------------------------------------------------------------------
namespace xt
{
    // Core types
    using ::xt::xarray_container;
    using ::xt::xtensor_container;
    using ::xt::xarray_adaptor;
    using ::xt::xtensor_adaptor;
    using ::xt::xview;
    using ::xt::xstrided_view;

    // Convenience aliases for BigNumber
    using BigNumber = bignumber::BigNumber;
    using BigArray = xarray_container<BigNumber>;
    using BigTensor2 = xtensor_container<BigNumber, 2>;
    using BigTensor3 = xtensor_container<BigNumber, 3>;

    // Common functions
    using ::xt::zeros;
    using ::xt::ones;
    using ::xt::full;
    using ::xt::empty;
    using ::xt::arange;
    using ::xt::linspace;
    using ::xt::logspace;
    using ::xt::sum;
    using ::xt::prod;
    using ::xt::mean;
    using ::xt::stddev;
    using ::xt::variance;

    // FFT functions
    using ::xt::fft::fft;
    using ::xt::fft::ifft;
    using ::xt::fft::convolve;

} // namespace xt

#endif // XTENSOR_HPP // Core types
    using ::xt::xarray_container;
    using ::xt::xtensor_container;
    using ::xt::xarray_adaptor;
    using ::xt::xtensor_adaptor;
    using ::xt::xview;
    using ::xt::xstrided_view;

    // Convenience aliases for BigNumber
    using BigNumber = bignumber::BigNumber;
    using BigArray = xarray_container<BigNumber>;
    using BigTensor2 = xtensor_container<BigNumber, 2>;
    using BigTensor3 = xtensor_container<BigNumber, 3>;

    // Common functions
    using ::xt::zeros;
    using ::xt::ones;
    using ::xt::full;
    using ::xt::empty;
    using ::xt::arange;
    using ::xt::linspace;
    using ::xt::logspace;
    using ::xt::sum;
    using ::xt::prod;
    using ::xt::mean;
    using ::xt::stddev;
    using ::xt::variance;

    // FFT functions
    using ::xt::fft::fft;
    using ::xt::fft::ifft;
    using ::xt::fft::convolve;

} // namespace xt

#endif // XTENSOR_HPP