--- START OF FILE core/os/os.h ---

#ifndef OS_H
#define OS_H

#include "core/typedefs.h"
#include "core/string/ustring.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * OS Class
 * 
 * Abstract interface for Operating System services.
 * Re-engineered for bit-perfect timing and galactic-scale simulation stability.
 * Provides the hardware metadata required for Warp-Kernel thread affinity.
 */
class OS {
	static OS *singleton;

public:
	static _FORCE_INLINE_ OS *get_singleton() { return singleton; }

	// ------------------------------------------------------------------------
	// Deterministic Timing API
	// ------------------------------------------------------------------------

	/**
	 * get_ticks_usec()
	 * Returns microseconds since start as BigIntCore.
	 * Guarantees infinite session duration without timer wrap-around.
	 */
	virtual BigIntCore get_ticks_usec() const = 0;

	/**
	 * get_ticks_msec()
	 * Returns milliseconds as BigIntCore for macro-scale scheduling.
	 */
	virtual BigIntCore get_ticks_msec() const;

	/**
	 * get_deterministic_delta()
	 * Returns the current frame delta in FixedMathCore.
	 * Essential for bit-perfect TIER_DETERMINISTIC physics synchronization.
	 */
	virtual FixedMathCore get_deterministic_delta() const = 0;

	virtual void delay_usec(uint32_t p_usec) const = 0;

	// ------------------------------------------------------------------------
	// System Information (Warp/EnTT Optimization)
	// ------------------------------------------------------------------------

	virtual String get_name() const = 0;
	virtual String get_distribution_name() const { return get_name(); }
	virtual String get_version() const = 0;

	/**
	 * get_processor_count()
	 * Returns logical core count. Used by SimulationThreadPool to 
	 * balance Warp kernel workloads across hardware.
	 */
	virtual int get_processor_count() const = 0;

	// ------------------------------------------------------------------------
	// Process Management
	// ------------------------------------------------------------------------

	virtual Error execute(const String &p_path, const Vector<String> &p_arguments, int64_t *r_child_id = nullptr, bool p_blocking = false) = 0;
	virtual Error kill(int64_t p_pid) = 0;
	virtual int get_process_id() const = 0;

	// ------------------------------------------------------------------------
	// Environment & Paths
	// ------------------------------------------------------------------------

	virtual String get_environment(const String &p_var) const = 0;
	virtual void set_environment(const String &p_var, const String &p_value) const = 0;

	virtual String get_executable_path() const = 0;
	virtual String get_user_data_dir() const = 0;

	OS();
	virtual ~OS();
};

#endif // OS_H

--- END OF FILE core/os/os.h ---
