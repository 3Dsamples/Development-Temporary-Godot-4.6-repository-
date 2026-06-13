//File 0074 : core/xtensor.hpp
//Top-level include aggregating all xtensor modules: core, containers, views, I/O, math, and simulation, with optional SIMD acceleration.
#ifndef XTENSOR_XTENSOR_HPP
#define XTENSOR_XTENSOR_HPP

// Core configuration and traits
#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xexpression.hpp"
#include "xexpression_traits.hpp"
#include "xmath.hpp"
#include "xaccessible.hpp"
#include "xiterable.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xlayout.hpp"
#include "xshape.hpp"
#include "xstorage.hpp"
#include "xutils.hpp"
#include "xexception.hpp"
#include "xnoalias.hpp"
#include "xassign.hpp"
#include "xfunction.hpp"
#include "xscalar.hpp"
#include "xoperation.hpp"
#include "xgenerator.hpp"
#include "xbroadcast.hpp"
#include "xeval.hpp"
#include "xreducer.hpp"
#include "xaccumulator.hpp"
#include "xbuilder.hpp"
#include "xexpression_holder.hpp"
#include "xrepeat.hpp"
#include "xvectorize.hpp"
#include "xslice.hpp"
#include "xpad.hpp"
#include "xset_operation.hpp"
#include "xhistogram.hpp"
#include "xsort.hpp"
#include "xrandom.hpp"
#include "xnorm.hpp"
#include "xstatistics.hpp"
#include "xlinalg.hpp"
#include "xcomplex.hpp"
#include "xfft.hpp"
#include "xsignal.hpp"
#include "xinterpolate.hpp"
#include "xoptimize.hpp"
#include "xintegrate.hpp"
#include "xode.hpp"
#include "xpde.hpp"
#include "xgeometry.hpp"
#include "xsparse.hpp"
#include "xsimulation.hpp"
#include "xoptional.hpp"

// Containers
#include "xarray.hpp"
#include "xfixed.hpp"
#include "xcontainer.hpp"
#include "xadapt.hpp"
#include "xbuffer_adaptor.hpp"
#include "xchunked_array.hpp"

// Views
#include "xview.hpp"
#include "xstrided_view.hpp"
#include "xdynamic_view.hpp"
#include "xoffset_view.hpp"
#include "xmasked_view.hpp"
#include "xindex_view.hpp"
#include "xfunctor_view.hpp"
#include "xstrided_view_base.hpp"
#include "xview_utils.hpp"

// Iterators
#include "xaxis_iterator.hpp"
#include "xaxis_slice_iterator.hpp"

// I/O (optional – users may include individually to reduce compile time)
#ifdef XTENSOR_IO_ENABLED
#include "xcsv.hpp"
#include "xjson.hpp"
#include "xnpy.hpp"
#include "xmime.hpp"
#endif

// Convenience namespace for end-users
namespace xt {
    // Common aliases
    template <class T>
    using array = xarray<T>;

    template <class T, std::size_t N>
    using tensor = xtensor<T, N>;
}

#endif // XTENSOR_XTENSOR_HPP