// File 370: modules/gaia/src/parallelization/gpu_parallelization.h
// GPU‑accelerated parallelisation dispatcher for Gaia physics.
// When a rendering device is available (Vulkan/DX12), this module offloads
// large parallel loops (e.g., BVH refit, PBD constraint projection, MPM grid
// transfer) to compute shaders. Falls back to the CPU parallelisation
// (CPUParallelization) when no GPU is present.
// Uses Godot's RenderingDevice for cross‑platform GPU compute.
// All public methods are static; the internal shader pipeline is set up once
// during module initialisation.

#ifndef GAIA_GPU_PARALLELIZATION_H
#define GAIA_GPU_PARALLELIZATION_H

#include "cpu_parallelization.h"
#include "core/object/rendering_device.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace gaia::parallel {

class GPUParallelization {
public:
    // Initialise GPU resources. Must be called after the rendering device is available.
    static void initialize();
    // Free GPU resources (called at module shutdown).
    static void shutdown();
    // Returns true if GPU acceleration is active and ready.
    static bool is_available();

    // -----------------------------------------------------------------------
    // Generic parallel for on GPU (if available), else CPU fallback.
    // The work is described by a kernel name (pre‑compiled) and a set of
    // buffers (input / output).  For simplicity, the interface is:
    //   parallel_for("kernel_name", count, [args])
    // The user provides the data as typed arrays; the function dispatches
    // compute threads and waits for completion.
    // -----------------------------------------------------------------------
    template <typename Func>
    static void parallel_for(const char *p_kernel, int64_t p_count, Func &&p_cpu_func) {
        if (!is_available()) {
            CPUParallelization::parallel_for(p_count, p_cpu_func);
            return;
        }
        // GPU path: prepare buffers, dispatch, sync.
        // In a full implementation, we would build a compute pipeline per kernel,
        // bind buffers, and use RenderingDevice::compute_list_* to dispatch.
        // For demonstration, we fall back to CPU with a warning.
        WARN_PRINT_ONCE("GPU parallel_for not fully implemented – using CPU fallback.");
        CPUParallelization::parallel_for(p_count, p_cpu_func);
    }

    // -----------------------------------------------------------------------
    // GPU reduce: sum of an array of floats (with optional atomics on GPU).
    // -----------------------------------------------------------------------
    static real_t reduce_sum(const real_t *p_data, int64_t p_count);

    // -----------------------------------------------------------------------
    // GPU prefix sum (used for radix sort of Morton codes in LBVH).
    // -----------------------------------------------------------------------
    static void prefix_sum(const LocalVector<uint32_t> &p_in, LocalVector<uint32_t> &p_out);

    // -----------------------------------------------------------------------
    // GPU radix sort for unsigned 32‑bit keys (used for BVH building).
    // -----------------------------------------------------------------------
    static void radix_sort(const LocalVector<uint32_t> &p_keys, LocalVector<int32_t> &p_indices);

private:
    // Internal rendering device handle
    static RenderingDevice *rd;
    static bool initialized;
    // Pre‑compiled shader module (loaded from embedded SPIR‑V bytecode)
    static RID shader;
    // Pipeline cache for common kernels
    struct Kernel {
        String name;
        RID pipeline;
    };
    static LocalVector<Kernel> kernels;

    // Create a compute pipeline from an embedded SPIR‑V binary (loaded once).
    static RID create_compute_pipeline(const String &p_kernel_name);
};

// ---------------------------------------------------------------------------
// Inline implementations
// ---------------------------------------------------------------------------
inline RenderingDevice *GPUParallelization::rd = nullptr;
inline bool GPUParallelization::initialized = false;
inline RID GPUParallelization::shader;
inline LocalVector<Kernel> GPUParallelization::kernels;

inline void GPUParallelization::initialize() {
    if (initialized) return;
    rd = RenderingDevice::get_singleton();
    if (!rd) return;

    // Load embedded shader bytecode (would be compiled offline and embedded as array).
    // For now, we skip actual SPIR‑V loading and assume not available to avoid build errors.
    initialized = false; // not yet functional
}

inline void GPUParallelization::shutdown() {
    if (shader.is_valid()) {
        rd->free(shader);
        shader = RID();
    }
    for (Kernel &k : kernels) {
        if (k.pipeline.is_valid()) rd->free(k.pipeline);
    }
    kernels.clear();
    rd = nullptr;
    initialized = false;
}

inline bool GPUParallelization::is_available() {
    return initialized && rd && shader.is_valid();
}

inline real_t GPUParallelization::reduce_sum(const real_t *p_data, int64_t p_count) {
    // Placeholder: uses CPU fallback
    return ParallelReduction::sum(p_data, p_count);
}

inline void GPUParallelization::prefix_sum(const LocalVector<uint32_t> &p_in, LocalVector<uint32_t> &p_out) {
    // CPU fallback for prefix sum
    p_out.resize(p_in.size());
    if (p_in.is_empty()) return;
    p_out[0] = p_in[0];
    for (int i = 1; i < p_in.size(); ++i) {
        p_out[i] = p_out[i - 1] + p_in[i];
    }
}

inline void GPUParallelization::radix_sort(const LocalVector<uint32_t> &p_keys, LocalVector<int32_t> &p_indices) {
    // CPU fallback using the existing CPU radix sort from Gaia.
    ParallelSort::sort(p_keys, p_indices);
}

inline RID GPUParallelization::create_compute_pipeline(const String &p_kernel_name) {
    // In production, we would look up p_kernel_name in the shader reflection
    // and build a compute pipeline with appropriate bindings.
    // For now, return empty RID.
    return RID();
}

} // namespace gaia::parallel

#endif // GAIA_GPU_PARALLELIZATION_H