// File 29: modules/gaia/src/parallelization/cuda_utilities.h

#ifndef GAIA_PARALLEL_CUDA_UTILITIES_H
#define GAIA_PARALLEL_CUDA_UTILITIES_H

// ---------------------------------------------------------------------------
// CUDA utilities – optional GPU acceleration.
// This file provides a thin wrapper over CUDA runtime functions and a few
// convenience helpers. When the `cuda_enabled` SCons option is off (default),
// all calls become no‑ops and CUDA-dependent features degrade gracefully to
// CPU implementations elsewhere.
// ---------------------------------------------------------------------------

#include "core/error/error_macros.h"
#include "core/string/ustring.h"
#include "core/typedefs.h"

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace gaia::parallel::cuda {

// Initialise the CUDA subsystem. Call once at engine startup if CUDA is
// enabled. Returns true if a usable device was found.
inline bool initialize() {
#ifdef CUDA_ENABLED
	int device_count = 0;
	cudaError_t err = cudaGetDeviceCount(&device_count);
	if (err != cudaSuccess || device_count == 0) {
		ERR_PRINT("CUDA enabled but no devices found.");
		return false;
	}
	// Select device 0 for now.
	cudaSetDevice(0);
	return true;
#else
	return false;
#endif
}

// Free CUDA resources (should be called at module shutdown).
inline void shutdown() {
#ifdef CUDA_ENABLED
	cudaDeviceReset();
#endif
}

// Allocate device memory. Returns pointer or nullptr on failure.
inline void *malloc_device(size_t p_size) {
#ifdef CUDA_ENABLED
	void *ptr = nullptr;
	cudaError_t err = cudaMalloc(&ptr, p_size);
	if (err != cudaSuccess) {
		ERR_PRINT("cudaMalloc failed.");
		return nullptr;
	}
	return ptr;
#else
	ERR_FAIL_V_MSG(nullptr, "CUDA not enabled.");
#endif
}

// Free device memory.
inline void free_device(void *p_ptr) {
#ifdef CUDA_ENABLED
	if (p_ptr) cudaFree(p_ptr);
#else
	// no-op
#endif
}

// Copy host → device.
inline void memcpy_host_to_device(void *p_dst, const void *p_src, size_t p_size) {
#ifdef CUDA_ENABLED
	cudaMemcpy(p_dst, p_src, p_size, cudaMemcpyHostToDevice);
#else
	// no-op
#endif
}

// Copy device → host.
inline void memcpy_device_to_host(void *p_dst, const void *p_src, size_t p_size) {
#ifdef CUDA_ENABLED
	cudaMemcpy(p_dst, p_src, p_size, cudaMemcpyDeviceToHost);
#else
	// no-op
#endif
}

// Synchronise the default stream.
inline void sync() {
#ifdef CUDA_ENABLED
	cudaDeviceSynchronize();
#endif
}

// Simple RAII event for timing GPU work.
class CudaEvent {
public:
	CudaEvent() : event(nullptr) {
#ifdef CUDA_ENABLED
		cudaEventCreate(&event);
#endif
	}
	~CudaEvent() {
#ifdef CUDA_ENABLED
		if (event) cudaEventDestroy(event);
#endif
	}
	void record() {
#ifdef CUDA_ENABLED
		cudaEventRecord(event);
#endif
	}
	void wait() {
#ifdef CUDA_ENABLED
		cudaEventSynchronize(event);
#endif
	}
	float elapsed_ms(const CudaEvent &p_start) {
#ifdef CUDA_ENABLED
		float ms = 0.0f;
		cudaEventElapsedTime(&ms, p_start.event, event);
		return ms;
#else
		return 0.0f;
#endif
	}

private:
#ifdef CUDA_ENABLED
	cudaEvent_t event;
#endif
};

// Check for last error (debug helper).
inline String get_error_string() {
#ifdef CUDA_ENABLED
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		return String(cudaGetErrorString(err));
	}
	return String("success");
#else
	return String("CUDA not enabled");
#endif
}

} // namespace gaia::parallel::cuda

#endif // GAIA_PARALLEL_CUDA_UTILITIES_H