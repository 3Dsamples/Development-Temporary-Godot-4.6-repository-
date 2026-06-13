//File 0116 : numdot/numdot.h
//Top-level include for the NumDot library, aggregating all modules: core, containers, math, I/O, and views.
#ifndef NUMDOT_NUMDOT_H
#define NUMDOT_NUMDOT_H

// Core configuration and forward declarations
#include "config.h"
#include "forward.h"
#include "types.h"
#include "shape.h"
#include "strides.h"
#include "slicing.h"

// Containers and expressions
#include "array.h"
#include "elementwise.h"
#include "broadcast.h"
#include "reductions.h"
#include "view.h"

// Mathematics
#include "math.h"
#include "linalg.h"
#include "random.h"

// Input/Output
#include "io.h"

// Convenience namespace for end users
namespace numdot {

    // Common type aliases
    template <class T>
    using Array = array<T, dynamic_rank>;

    template <class T, std::size_t N>
    using FixedArray = fixed_array<T, N>;

    // Bring in commonly used functions
    using ::numdot::sum;
    using ::numdot::prod;
    using ::numdot::mean;
    using ::numdot::max;
    using ::numdot::min;
    using ::numdot::abs;
    using ::numdot::sqrt;
    using ::numdot::exp;
    using ::numdot::log;
    using ::numdot::sin;
    using ::numdot::cos;
    using ::numdot::tan;
    using ::numdot::pow;
    using ::numdot::atan2;
    using ::numdot::deg2rad;
    using ::numdot::rad2deg;
    using ::numdot::clip;
    using ::numdot::sign;
    using ::numdot::isfinite;
    using ::numdot::isinf;
    using ::numdot::isnan;

    using ::numdot::linalg::dot;
    using ::numdot::linalg::matmul;
    using ::numdot::linalg::inv;
    using ::numdot::linalg::det;
    using ::numdot::linalg::solve;
    using ::numdot::linalg::trace;
    using ::numdot::linalg::eig_power;

    using ::numdot::random::rand;
    using ::numdot::random::randn;
    using ::numdot::random::uniform;
    using ::numdot::random::normal;
    using ::numdot::random::randint;
    using ::numdot::random::exponential;
    using ::numdot::random::gamma;
    using ::numdot::random::sobol;
    using ::numdot::random::choice;
    using ::numdot::random::shuffle;
    using ::numdot::random::permutation;
    using ::numdot::random::seed;

    using ::numdot::io::load;
    using ::numdot::io::save_npy;
    using ::numdot::io::load_npy;
    using ::numdot::io::save_csv;
    using ::numdot::io::load_csv;
    using ::numdot::io::save_json;
    using ::numdot::io::load_json;
    using ::numdot::io::save_txt;
    using ::numdot::io::load_txt;

} // namespace numdot

#endif // NUMDOT_NUMDOT_H