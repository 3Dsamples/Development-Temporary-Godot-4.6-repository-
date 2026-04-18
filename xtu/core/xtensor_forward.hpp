// core/xtensor_forward.hpp

#ifndef XTENSOR_FORWARD_HPP
#define XTENSOR_FORWARD_HPP

#include "xtensor_config.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>
#include <array>
#include <tuple>
#include <initializer_list>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        // --------------------------------------------------------------------
        // Forward declarations of core expression types
        // --------------------------------------------------------------------
        
        template <class D>
        class xexpression;
        
        template <class F, class... CT>
        class xfunction;
        
        template <class F, class R, class... CT>
        class xfunction_base;
        
        template <class CT, class X>
        class xbroadcast;
        
        template <class CT, class... S>
        class xview;
        
        template <class CT, class... S>
        class xstrided_view;
        
        template <class CT, class X, class O>
        class xreducer;
        
        template <class E>
        class xaccumulator;
        
        template <class E>
        class xoptional_assembly;
        
        template <class D>
        class xoptional_assembly_adaptor;
        
        template <class CT, class R>
        class xgenerator;
        
        template <class CT, class T>
        class xindex_view;
        
        template <class CT, class... S>
        class xmasked_view;
        
        template <class E>
        class xscalar;
        
        template <class T>
        class xcomplex;
        
        // --------------------------------------------------------------------
        // Forward declarations of container types
        // --------------------------------------------------------------------
        
        template <class T,
                  layout_type L = config::default_layout,
                  class A = default_allocator<T>,
                  class Tag = void>
        class xarray_container;
        
        template <class EC, layout_type L, class Tag>
        class xarray_adaptor;
        
        template <class T,
                  std::size_t N,
                  layout_type L = config::default_layout,
                  class Tag = void>
        class xtensor_container;
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        class xtensor_adaptor;
        
        template <class T, class Tag = void>
        using xarray = xarray_container<T, config::default_layout, default_allocator<T>, Tag>;
        
        template <class T, std::size_t N, class Tag = void>
        using xtensor = xtensor_container<T, N, config::default_layout, Tag>;
        
        template <class T>
        using xtensor_fixed = xtensor_container<T, 0, config::default_layout, void>;
        
        // --------------------------------------------------------------------
        // Forward declarations of shape and stride types
        // --------------------------------------------------------------------
        
        template <class T, std::size_t N = config::small_vector_size>
        class svector;
        
        template <class T, std::size_t N = config::small_vector_size>
        class small_vector;
        
        template <class T, class Tag = void>
        class xshape;
        
        template <class T, class Tag = void>
        class xstrides;
        
        template <class T, std::size_t N>
        class fixed_shape;
        
        // --------------------------------------------------------------------
        // Forward declarations of layout types
        // --------------------------------------------------------------------
        
        struct layout_type_t;
        
        template <layout_type L>
        struct layout_selector;
        
        template <class S>
        struct row_major_layout;
        
        template <class S>
        struct column_major_layout;
        
        // --------------------------------------------------------------------
        // Forward declarations of iterator types
        // --------------------------------------------------------------------
        
        template <class C, bool is_const>
        class xiterator;
        
        template <class C>
        class xstepper;
        
        template <class C>
        class xconst_stepper;
        
        template <class C>
        class xiterator_adaptor;
        
        template <class C, class enable = void>
        class xstorage_iterator;
        
        // --------------------------------------------------------------------
        // Forward declarations of indexing utilities
        // --------------------------------------------------------------------
        
        class xindex;
        
        template <class S>
        class xindex_builder;
        
        template <class S, class... Args>
        class xindex_range;
        
        class xall_tag;
        class xnewaxis_tag;
        class xellipsis_tag;
        
        template <class T>
        class xrange;
        
        template <class T>
        class xrange_adaptor;
        
        template <class E>
        class xplaceholder;
        
        template <class... E>
        class xindex_sequence;
        
        // --------------------------------------------------------------------
        // Forward declarations of assignment and operation tags
        // --------------------------------------------------------------------
        
        struct xtensor_expression_tag;
        struct xexpression_tag;
        struct xcontainer_tag;
        struct xview_tag;
        struct xfunction_tag;
        struct xreducer_tag;
        struct xbroadcast_tag;
        struct xscalar_tag;
        struct xgenerator_tag;
        
        // --------------------------------------------------------------------
        // Forward declarations of math function objects
        // --------------------------------------------------------------------
        
#define XTENSOR_FORWARD_UNARY_FUNCTION(NAME) \
        template <class T> struct NAME##_fun;
        
#define XTENSOR_FORWARD_BINARY_FUNCTION(NAME) \
        template <class T> struct NAME##_fun;
        
        XTENSOR_FORWARD_UNARY_FUNCTION(abs)
        XTENSOR_FORWARD_UNARY_FUNCTION(fabs)
        XTENSOR_FORWARD_UNARY_FUNCTION(sqrt)
        XTENSOR_FORWARD_UNARY_FUNCTION(cbrt)
        XTENSOR_FORWARD_UNARY_FUNCTION(exp)
        XTENSOR_FORWARD_UNARY_FUNCTION(exp2)
        XTENSOR_FORWARD_UNARY_FUNCTION(expm1)
        XTENSOR_FORWARD_UNARY_FUNCTION(log)
        XTENSOR_FORWARD_UNARY_FUNCTION(log2)
        XTENSOR_FORWARD_UNARY_FUNCTION(log10)
        XTENSOR_FORWARD_UNARY_FUNCTION(log1p)
        XTENSOR_FORWARD_UNARY_FUNCTION(sin)
        XTENSOR_FORWARD_UNARY_FUNCTION(cos)
        XTENSOR_FORWARD_UNARY_FUNCTION(tan)
        XTENSOR_FORWARD_UNARY_FUNCTION(asin)
        XTENSOR_FORWARD_UNARY_FUNCTION(acos)
        XTENSOR_FORWARD_UNARY_FUNCTION(atan)
        XTENSOR_FORWARD_UNARY_FUNCTION(sinh)
        XTENSOR_FORWARD_UNARY_FUNCTION(cosh)
        XTENSOR_FORWARD_UNARY_FUNCTION(tanh)
        XTENSOR_FORWARD_UNARY_FUNCTION(asinh)
        XTENSOR_FORWARD_UNARY_FUNCTION(acosh)
        XTENSOR_FORWARD_UNARY_FUNCTION(atanh)
        XTENSOR_FORWARD_UNARY_FUNCTION(erf)
        XTENSOR_FORWARD_UNARY_FUNCTION(erfc)
        XTENSOR_FORWARD_UNARY_FUNCTION(tgamma)
        XTENSOR_FORWARD_UNARY_FUNCTION(lgamma)
        XTENSOR_FORWARD_UNARY_FUNCTION(ceil)
        XTENSOR_FORWARD_UNARY_FUNCTION(floor)
        XTENSOR_FORWARD_UNARY_FUNCTION(trunc)
        XTENSOR_FORWARD_UNARY_FUNCTION(round)
        XTENSOR_FORWARD_UNARY_FUNCTION(nearbyint)
        XTENSOR_FORWARD_UNARY_FUNCTION(rint)
        XTENSOR_FORWARD_UNARY_FUNCTION(isnan)
        XTENSOR_FORWARD_UNARY_FUNCTION(isinf)
        XTENSOR_FORWARD_UNARY_FUNCTION(isfinite)
        XTENSOR_FORWARD_UNARY_FUNCTION(sign)
        XTENSOR_FORWARD_UNARY_FUNCTION(conj)
        XTENSOR_FORWARD_UNARY_FUNCTION(real)
        XTENSOR_FORWARD_UNARY_FUNCTION(imag)
        XTENSOR_FORWARD_UNARY_FUNCTION(arg)
        XTENSOR_FORWARD_UNARY_FUNCTION(norm)
        XTENSOR_FORWARD_UNARY_FUNCTION(proj)
        
        XTENSOR_FORWARD_BINARY_FUNCTION(add)
        XTENSOR_FORWARD_BINARY_FUNCTION(sub)
        XTENSOR_FORWARD_BINARY_FUNCTION(mul)
        XTENSOR_FORWARD_BINARY_FUNCTION(div)
        XTENSOR_FORWARD_BINARY_FUNCTION(modulus)
        XTENSOR_FORWARD_BINARY_FUNCTION(pow)
        XTENSOR_FORWARD_BINARY_FUNCTION(atan2)
        XTENSOR_FORWARD_BINARY_FUNCTION(hypot)
        XTENSOR_FORWARD_BINARY_FUNCTION(fmod)
        XTENSOR_FORWARD_BINARY_FUNCTION(remainder)
        XTENSOR_FORWARD_BINARY_FUNCTION(copysign)
        XTENSOR_FORWARD_BINARY_FUNCTION(nextafter)
        XTENSOR_FORWARD_BINARY_FUNCTION(fdim)
        XTENSOR_FORWARD_BINARY_FUNCTION(fmax)
        XTENSOR_FORWARD_BINARY_FUNCTION(fmin)
        XTENSOR_FORWARD_BINARY_FUNCTION(equal)
        XTENSOR_FORWARD_BINARY_FUNCTION(not_equal)
        XTENSOR_FORWARD_BINARY_FUNCTION(less)
        XTENSOR_FORWARD_BINARY_FUNCTION(less_equal)
        XTENSOR_FORWARD_BINARY_FUNCTION(greater)
        XTENSOR_FORWARD_BINARY_FUNCTION(greater_equal)
        XTENSOR_FORWARD_BINARY_FUNCTION(logical_and)
        XTENSOR_FORWARD_BINARY_FUNCTION(logical_or)
        XTENSOR_FORWARD_BINARY_FUNCTION(bitwise_and)
        XTENSOR_FORWARD_BINARY_FUNCTION(bitwise_or)
        XTENSOR_FORWARD_BINARY_FUNCTION(bitwise_xor)
        XTENSOR_FORWARD_BINARY_FUNCTION(left_shift)
        XTENSOR_FORWARD_BINARY_FUNCTION(right_shift)
        
#undef XTENSOR_FORWARD_UNARY_FUNCTION
#undef XTENSOR_FORWARD_BINARY_FUNCTION
        
        // --------------------------------------------------------------------
        // Forward declarations of reducer function objects
        // --------------------------------------------------------------------
        
        template <class T, class E = void>
        struct sum_fun;
        
        template <class T, class E = void>
        struct prod_fun;
        
        template <class T, class E = void>
        struct mean_fun;
        
        template <class T, class E = void>
        struct variance_fun;
        
        template <class T, class E = void>
        struct stddev_fun;
        
        template <class T, class E = void>
        struct amin_fun;
        
        template <class T, class E = void>
        struct amax_fun;
        
        template <class T, class E = void>
        struct all_fun;
        
        template <class T, class E = void>
        struct any_fun;
        
        template <class T, class E = void>
        struct norm_l0_fun;
        
        template <class T, class E = void>
        struct norm_l1_fun;
        
        template <class T, class E = void>
        struct norm_l2_fun;
        
        template <class T, class E = void>
        struct norm_linf_fun;
        
        template <class T, class E = void>
        struct norm_sq_fun;
        
        template <class T, class E = void>
        struct cumsum_fun;
        
        template <class T, class E = void>
        struct cumprod_fun;
        
        template <class T, class E = void>
        struct cummin_fun;
        
        template <class T, class E = void>
        struct cummax_fun;
        
        template <class T, class E = void>
        struct nancumsum_fun;
        
        template <class T, class E = void>
        struct nancumprod_fun;
        
        template <class T, class E = void>
        struct nansum_fun;
        
        template <class T, class E = void>
        struct nanprod_fun;
        
        template <class T, class E = void>
        struct nanmean_fun;
        
        template <class T, class E = void>
        struct nanvar_fun;
        
        template <class T, class E = void>
        struct nanstd_fun;
        
        template <class T, class E = void>
        struct nanmin_fun;
        
        template <class T, class E = void>
        struct nanmax_fun;
        
        template <class T, class E = void>
        struct argmin_fun;
        
        template <class T, class E = void>
        struct argmax_fun;
        
        template <class T, class E = void>
        struct median_fun;
        
        template <class T, class E = void>
        struct quantile_fun;
        
        template <class T, class E = void>
        struct ptp_fun;
        
        template <class T, class E = void>
        struct diff_fun;
        
        template <class T, class E = void>
        struct trapz_fun;
        
        // --------------------------------------------------------------------
        // Forward declarations of accumulator function objects
        // --------------------------------------------------------------------
        
        template <class T>
        struct sum_accumulator;
        
        template <class T>
        struct prod_accumulator;
        
        template <class T>
        struct min_accumulator;
        
        template <class T>
        struct max_accumulator;
        
        template <class T>
        struct mean_accumulator;
        
        template <class T>
        struct variance_accumulator;
        
        // --------------------------------------------------------------------
        // Forward declarations of evaluation strategies
        // --------------------------------------------------------------------
        
        struct immediate_assign_tag;
        struct lazy_assign_tag;
        
        template <class E, class Tag = lazy_assign_tag>
        class xexpression_assigner;
        
        // --------------------------------------------------------------------
        // Forward declarations of exception types
        // --------------------------------------------------------------------
        
        class broadcast_error;
        class dimension_mismatch;
        class index_error;
        class incompatible_shapes;
        class reduction_error;
        class accumulation_error;
        class iterator_error;
        class memory_allocation_error;
        class alignment_error;
        class not_implemented_error;
        class invalid_layout_error;
        class invalid_slice_error;
        class invalid_stride_error;
        
        // --------------------------------------------------------------------
        // Forward declarations of I/O related classes
        // --------------------------------------------------------------------
        
        template <class E>
        class xexpression_printer;
        
        template <class E>
        class xexpression_formatter;
        
        class xnpz_reader;
        class xnpz_writer;
        
        template <class T>
        class xcsv_reader;
        
        template <class T>
        class xcsv_writer;
        
        class xjson_reader;
        class xjson_writer;
        
        // --------------------------------------------------------------------
        // Forward declarations of sparse tensor types
        // --------------------------------------------------------------------
        
        template <class T>
        class xcoo_scheme;
        
        template <class T>
        class xcsr_scheme;
        
        template <class T>
        class xcsc_scheme;
        
        template <class T, class Scheme = xcoo_scheme<T>>
        class xsparse_tensor;
        
        template <class T>
        using xcoo_tensor = xsparse_tensor<T, xcoo_scheme<T>>;
        
        template <class T>
        using xcsr_tensor = xsparse_tensor<T, xcsr_scheme<T>>;
        
        // --------------------------------------------------------------------
        // Forward declarations of frame/labeled array types
        // --------------------------------------------------------------------
        
        template <class T>
        class xcoordinate;
        
        template <class T>
        class xdimension;
        
        template <class T>
        class xvariable;
        
        template <class T>
        class xdataframe;
        
        template <class T>
        class xseries;
        
        template <class T>
        class xgroupby;
        
        // --------------------------------------------------------------------
        // Forward declarations of signal processing types
        // --------------------------------------------------------------------
        
        template <class T>
        class xfft_plan;
        
        template <class T>
        class xifft_plan;
        
        template <class T>
        class xrfft_plan;
        
        template <class T>
        class xirfft_plan;
        
        template <class T>
        class xdct_plan;
        
        template <class T>
        class xdst_plan;
        
        template <class T>
        class xconvolution;
        
        template <class T>
        class xcorrelation;
        
        template <class T>
        class xfilter;
        
        template <class T>
        class xwindow_function;
        
        template <class T>
        class xwavelet;
        
        // --------------------------------------------------------------------
        // Forward declarations of random number generation types
        // --------------------------------------------------------------------
        
        class xrandom_engine;
        
        template <class Distribution, class Engine = xrandom_engine>
        class xrandom_generator;
        
        template <class T, class Engine = xrandom_engine>
        class xuniform_distribution;
        
        template <class T, class Engine = xrandom_engine>
        class xnormal_distribution;
        
        template <class T, class Engine = xrandom_engine>
        class xbernoulli_distribution;
        
        template <class T, class Engine = xrandom_engine>
        class xbinomial_distribution;
        
        template <class T, class Engine = xrandom_engine>
        class xpoisson_distribution;
        
        template <class T, class Engine = xrandom_engine>
        class xexponential_distribution;
        
        template <class T, class Engine = xrandom_engine>
        class xgamma_distribution;
        
        template <class T, class Engine = xrandom_engine>
        class xbeta_distribution;
        
        template <class T, class Engine = xrandom_engine>
        class xchisquared_distribution;
        
        template <class T, class Engine = xrandom_engine>
        class xstudent_t_distribution;
        
        template <class T, class Engine = xrandom_engine>
        class xfisher_f_distribution;
        
        template <class T, class Engine = xrandom_engine>
        class xlognormal_distribution;
        
        template <class T, class Engine = xrandom_engine>
        class xweibull_distribution;
        
        template <class T, class Engine = xrandom_engine>
        class xextreme_value_distribution;
        
        // --------------------------------------------------------------------
        // Forward declarations of linear algebra types
        // --------------------------------------------------------------------
        
        template <class T>
        class xmatrix;
        
        template <class T>
        class xvector;
        
        template <class T>
        class xsymmetric_matrix;
        
        template <class T>
        class xhermitian_matrix;
        
        template <class T>
        class xtriangular_matrix;
        
        template <class T>
        class xbanded_matrix;
        
        template <class T>
        class xdiagonal_matrix;
        
        template <class T>
        class xorthogonal_matrix;
        
        template <class T>
        class xpermutation_matrix;
        
        template <class T>
        class xdecomposition_qr;
        
        template <class T>
        class xdecomposition_lu;
        
        template <class T>
        class xdecomposition_cholesky;
        
        template <class T>
        class xdecomposition_svd;
        
        template <class T>
        class xdecomposition_eigen;
        
        template <class T>
        class xdecomposition_schur;
        
        template <class T>
        class xdecomposition_hessenberg;
        
        // --------------------------------------------------------------------
        // Forward declarations of optimization types
        // --------------------------------------------------------------------
        
        template <class T>
        class xoptimizer;
        
        template <class T>
        class xgradient_descent;
        
        template <class T>
        class xstochastic_gradient_descent;
        
        template <class T>
        class xadam_optimizer;
        
        template <class T>
        class xrmsprop_optimizer;
        
        template <class T>
        class xadagrad_optimizer;
        
        template <class T>
        class xadadelta_optimizer;
        
        template <class T>
        class xmomentum_optimizer;
        
        template <class T>
        class xnesterov_optimizer;
        
        template <class T>
        class xlbfgs_optimizer;
        
        template <class T>
        class xnelder_mead_optimizer;
        
        template <class T>
        class xpowell_optimizer;
        
        template <class T>
        class xcg_optimizer;
        
        template <class T>
        class xbfgs_optimizer;
        
        template <class T>
        class xnewton_optimizer;
        
        template <class T>
        class xlevenberg_marquardt_optimizer;
        
        // --------------------------------------------------------------------
        // Forward declarations of interpolation types
        // --------------------------------------------------------------------
        
        template <class T>
        class xinterpolator;
        
        template <class T>
        class xlinear_interpolator;
        
        template <class T>
        class xcubic_interpolator;
        
        template <class T>
        class xspline_interpolator;
        
        template <class T>
        class xbarycentric_interpolator;
        
        template <class T>
        class xnearest_interpolator;
        
        template <class T>
        class xpolynomial_interpolator;
        
        template <class T>
        class xakima_interpolator;
        
        template <class T>
        class xpchip_interpolator;
        
        // --------------------------------------------------------------------
        // Forward declarations of integration types
        // --------------------------------------------------------------------
        
        template <class T>
        class xintegrator;
        
        template <class T>
        class xtrapezoidal_integrator;
        
        template <class T>
        class xsimpson_integrator;
        
        template <class T>
        class xgauss_legendre_integrator;
        
        template <class T>
        class xromberg_integrator;
        
        template <class T>
        class xadaptive_integrator;
        
        template <class T>
        class xode_solver;
        
        template <class T>
        class xrunge_kutta4_solver;
        
        template <class T>
        class xdopri5_solver;
        
        template <class T>
        class xdop853_solver;
        
        template <class T>
        class xvode_solver;
        
        template <class T>
        class xlsoda_solver;
        
        // --------------------------------------------------------------------
        // Forward declarations of clustering types
        // --------------------------------------------------------------------
        
        template <class T>
        class xclusterer;
        
        template <class T>
        class xkmeans_clusterer;
        
        template <class T>
        class xhierarchical_clusterer;
        
        template <class T>
        class xdbscan_clusterer;
        
        template <class T>
        class xmean_shift_clusterer;
        
        template <class T>
        class xgaussian_mixture_clusterer;
        
        template <class T>
        class xaffinity_propagation_clusterer;
        
        template <class T>
        class xspectral_clusterer;
        
        template <class T>
        class xagglomerative_clusterer;
        
        template <class T>
        class xbirch_clusterer;
        
        // --------------------------------------------------------------------
        // Forward declarations of graphics and rendering types
        // --------------------------------------------------------------------
        
        template <class T>
        class xcanvas;
        
        template <class T>
        class xfigure;
        
        template <class T>
        class xaxes;
        
        template <class T>
        class xplot;
        
        template <class T>
        class xscatter_plot;
        
        template <class T>
        class xline_plot;
        
        template <class T>
        class xbar_plot;
        
        template <class T>
        class xhistogram_plot;
        
        template <class T>
        class xcontour_plot;
        
        template <class T>
        class xsurface_plot;
        
        template <class T>
        class xquiver_plot;
        
        template <class T>
        class xstream_plot;
        
        template <class T>
        class ximage_plot;
        
        template <class T>
        class xcolormap;
        
        // --------------------------------------------------------------------
        // Forward declarations of image processing types
        // --------------------------------------------------------------------
        
        template <class T>
        class ximage;
        
        template <class T>
        class ximage_filter;
        
        template <class T>
        class xconvolution_filter;
        
        template <class T>
        class xgaussian_filter;
        
        template <class T>
        class xmedian_filter;
        
        template <class T>
        class xsobel_filter;
        
        template <class T>
        class xlaplacian_filter;
        
        template <class T>
        class xcanny_edge_detector;
        
        template <class T>
        class xhough_transform;
        
        template <class T>
        class xmorphological_operator;
        
        template <class T>
        class xconnected_components;
        
        template <class T>
        class xwatershed_segmentation;
        
        template <class T>
        class xfeature_detector;
        
        template <class T>
        class xsift_detector;
        
        template <class T>
        class xsurf_detector;
        
        template <class T>
        class xorb_detector;
        
        // --------------------------------------------------------------------
        // Forward declarations of audio processing types
        // --------------------------------------------------------------------
        
        template <class T>
        class xaudio_signal;
        
        template <class T>
        class xaudio_stream;
        
        template <class T>
        class xaudio_player;
        
        template <class T>
        class xaudio_recorder;
        
        template <class T>
        class xaudio_effect;
        
        template <class T>
        class xreverb_effect;
        
        template <class T>
        class xdelay_effect;
        
        template <class T>
        class xchorus_effect;
        
        template <class T>
        class xflanger_effect;
        
        template <class T>
        class xphaser_effect;
        
        template <class T>
        class xequalizer_effect;
        
        template <class T>
        class xcompressor_effect;
        
        template <class T>
        class xdistortion_effect;
        
        template <class T>
        class xpitch_shifter;
        
        template <class T>
        class xtime_stretcher;
        
        template <class T>
        class xspectrogram;
        
        template <class T>
        class xmfcc_extractor;
        
        // --------------------------------------------------------------------
        // Forward declarations of Godot integration types
        // --------------------------------------------------------------------
        
        class xvariant;
        class xclassdb;
        class xresource;
        class xnode;
        class xscene;
        class xinput_event;
        class xgdscript;
        class xshader;
        class xmaterial;
        class xmesh;
        class xtexture;
        class xfont;
        class xanimation;
        class xphysics_body;
        class xarea;
        class xcamera;
        class xlight;
        class xenvironment;
        class xworld;
        class xviewport;
        class xcontrol;
        class xwindow;
        class xeditor_plugin;
        class xeditor_interface;
        class xeditor_inspector;
        class xeditor_file_system;
        class xeditor_selection;
        class xeditor_undo_redo;
        class xeditor_resource_preview;
        class xeditor_script_editor;
        class xeditor_debugger;
        class xeditor_profiler;
        class xeditor_asset_library;
        
        // --------------------------------------------------------------------
        // Type aliases for common configurations
        // --------------------------------------------------------------------
        
        template <class T>
        using xexpression_ptr = std::shared_ptr<xexpression<T>>;
        
        template <class T>
        using xexpression_ref = std::reference_wrapper<xexpression<T>>;
        
        template <class T>
        using xarray_ptr = std::shared_ptr<xarray<T>>;
        
        template <class T, std::size_t N>
        using xtensor_ptr = std::shared_ptr<xtensor<T, N>>;
        
        template <class T>
        using xoptional = std::optional<T>;
        
        template <class T>
        using xexpected = std::expected<T, std::error_code>;
        
        template <class T>
        using xresult = std::variant<T, std::error_code>;
        
        // --------------------------------------------------------------------
        // Tag types for compile-time dispatch
        // --------------------------------------------------------------------
        
        struct has_expression_tag {};
        struct has_container_tag {};
        struct has_view_tag {};
        struct has_function_tag {};
        struct has_broadcast_tag {};
        struct has_reducer_tag {};
        struct has_generator_tag {};
        struct has_scalar_tag {};
        struct has_iterator_tag {};
        struct has_layout_tag {};
        struct has_shape_tag {};
        struct has_strides_tag {};
        struct has_assign_tag {};
        struct has_eval_tag {};
        
        template <class T>
        struct expression_tag_of;
        
        template <class T>
        using expression_tag_of_t = typename expression_tag_of<T>::type;
        
        // --------------------------------------------------------------------
        // Enable if utilities for SFINAE
        // --------------------------------------------------------------------
        
        template <class T>
        using enable_xexpression = std::enable_if_t<
            std::is_base_of_v<xexpression<typename T::value_type>, T>
        >;
        
        template <class T>
        using enable_xcontainer = std::enable_if_t<
            std::is_base_of_v<xcontainer_tag, expression_tag_of_t<T>>
        >;
        
        template <class T>
        using enable_xview = std::enable_if_t<
            std::is_base_of_v<xview_tag, expression_tag_of_t<T>>
        >;
        
        template <class T>
        using enable_xfunction = std::enable_if_t<
            std::is_base_of_v<xfunction_tag, expression_tag_of_t<T>>
        >;
        
        template <class T>
        using enable_xscalar = std::enable_if_t<
            std::is_base_of_v<xscalar_tag, expression_tag_of_t<T>>
        >;
        
        template <class T>
        using enable_xreducer = std::enable_if_t<
            std::is_base_of_v<xreducer_tag, expression_tag_of_t<T>>
        >;
        
        template <class T>
        using enable_xbroadcast = std::enable_if_t<
            std::is_base_of_v<xbroadcast_tag, expression_tag_of_t<T>>
        >;
        
        template <class T>
        using enable_xgenerator = std::enable_if_t<
            std::is_base_of_v<xgenerator_tag, expression_tag_of_t<T>>
        >;
        
        template <class T>
        using enable_xtensor = std::enable_if_t<
            std::is_same_v<expression_tag_of_t<T>, xtensor_expression_tag>
        >;
        
        template <class E1, class E2>
        using enable_same_expression = std::enable_if_t<
            std::is_same_v<expression_tag_of_t<E1>, expression_tag_of_t<E2>>
        >;
        
        // --------------------------------------------------------------------
        // Constant expressions
        // --------------------------------------------------------------------
        
        XTENSOR_INLINE_VARIABLE xall_tag all = {};
        XTENSOR_INLINE_VARIABLE xnewaxis_tag newaxis = {};
        XTENSOR_INLINE_VARIABLE xellipsis_tag ellipsis = {};
        
        XTENSOR_INLINE_VARIABLE immediate_assign_tag immediate_assign = {};
        XTENSOR_INLINE_VARIABLE lazy_assign_tag lazy_assign = {};
        
        // --------------------------------------------------------------------
        // Helper metafunctions
        // --------------------------------------------------------------------
        
        template <class... E>
        struct common_value_type;
        
        template <class... E>
        using common_value_type_t = typename common_value_type<E...>::type;
        
        template <class... E>
        struct common_size_type;
        
        template <class... E>
        using common_size_type_t = typename common_size_type<E...>::type;
        
        template <class... E>
        struct common_difference_type;
        
        template <class... E>
        using common_difference_type_t = typename common_difference_type<E...>::type;
        
        template <class... E>
        struct common_shape_type;
        
        template <class... E>
        using common_shape_type_t = typename common_shape_type<E...>::type;
        
        template <class... E>
        struct common_strides_type;
        
        template <class... E>
        using common_strides_type_t = typename common_strides_type<E...>::type;
        
        // --------------------------------------------------------------------
        // Expression traits
        // --------------------------------------------------------------------
        
        template <class E>
        struct xexpression_traits;
        
        template <class E>
        using value_type_of = typename xexpression_traits<E>::value_type;
        
        template <class E>
        using reference_of = typename xexpression_traits<E>::reference;
        
        template <class E>
        using const_reference_of = typename xexpression_traits<E>::const_reference;
        
        template <class E>
        using pointer_of = typename xexpression_traits<E>::pointer;
        
        template <class E>
        using const_pointer_of = typename xexpression_traits<E>::const_pointer;
        
        template <class E>
        using size_type_of = typename xexpression_traits<E>::size_type;
        
        template <class E>
        using difference_type_of = typename xexpression_traits<E>::difference_type;
        
        template <class E>
        using shape_type_of = typename xexpression_traits<E>::shape_type;
        
        template <class E>
        using strides_type_of = typename xexpression_traits<E>::strides_type;
        
        template <class E>
        using layout_type_of = typename xexpression_traits<E>::layout_type;
        
        template <class E>
        using expression_tag_of = typename xexpression_traits<E>::expression_tag;
        
        template <class E>
        inline constexpr bool is_xexpression_v = xexpression_traits<E>::is_expression;
        
        template <class E>
        inline constexpr bool is_const_xexpression_v = xexpression_traits<E>::is_const;
        
        template <class E>
        inline constexpr bool is_mutable_xexpression_v = xexpression_traits<E>::is_mutable;
        
        template <class E>
        inline constexpr bool is_xscalar_v = std::is_base_of_v<xscalar_tag, expression_tag_of<E>>;
        
        template <class E>
        inline constexpr bool is_xarray_v = std::is_same_v<expression_tag_of<E>, xcontainer_tag>;
        
        template <class E>
        inline constexpr bool is_xtensor_v = std::is_same_v<expression_tag_of<E>, xtensor_expression_tag>;
        
        template <class E>
        inline constexpr std::size_t dimension_of = xexpression_traits<E>::dimension;
        
        template <class E>
        inline constexpr bool is_fixed_dimension_v = (dimension_of<E> != SIZE_MAX);
        
        template <class E>
        inline constexpr bool is_dynamic_dimension_v = (dimension_of<E> == SIZE_MAX);
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_FORWARD_HPP