// File 375: modules/integration/unified_physics_event_bus.h
// High‑performance lock‑free event bus for the unified physics pipeline.
// Dispatches collision started/ended, trigger overlap, and joint break
// events from all engines (Newton, Genesis, Vienna, Wicked) to user scripts.
// Uses a triple‑buffer per event type, atomic indices, and callable storage
// for GDScript/C# callbacks.  All publish methods are inline and lock‑free;
// consumers read the latest stable buffer each frame without blocking.

#ifndef INTEGRATION_UNIFIED_PHYSICS_EVENT_BUS_H
#define INTEGRATION_UNIFIED_PHYSICS_EVENT_BUS_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/variant/variant.h"
#include "core/typedefs.h"
#include <atomic>

namespace unified {

// ---------------------------------------------------------------------------
// Event structures (compact, cache‑line aligned for multi‑producer)
// ---------------------------------------------------------------------------
struct alignas(64) CollisionEvent {
	uint64_t body_a;            // engine‑specific body IDs
	uint64_t body_b;
	uint8_t  engine;            // 0=Newton, 1=Genesis, 2=Vienna, 3=Wicked
	bool     started;           // true = collision started, false = ended
	uint32_t frame;             // frame index (for ordering)
};

struct alignas(64) TriggerEvent {
	uint64_t trigger_body;      // the trigger volume (area)
	uint64_t other_body;        // the body that entered/exited
	uint8_t  engine;
	bool     entered;
	uint32_t frame;
};

struct alignas(64) JointBreakEvent {
	uint64_t joint_id;          // engine‑specific joint ID
	uint64_t body_a;
	uint64_t body_b;
	uint8_t  engine;
	real_t   break_force;
	real_t   break_torque;
	uint32_t frame;
};

// ---------------------------------------------------------------------------
// Triple‑buffer: three buffers rotated by the producer; consumer reads the
// most recent stable buffer without blocking the producer.
// ---------------------------------------------------------------------------
template <typename T>
class TripleBuffer {
private:
	LocalVector<T> buffers[3];
	std::atomic<int> write_index { 0 };      // which buffer is being written to
	std::atomic<int> stable_index { 0 };     // which buffer is complete and ready for read

public:
	TripleBuffer() {}
	void reserve(int capacity) {
		for (int i = 0; i < 3; ++i) buffers[i].reserve(capacity);
	}
	void clear() {
		write_index.store(0, std::memory_order_relaxed);
		stable_index.store(0, std::memory_order_relaxed);
		for (int i = 0; i < 3; ++i) buffers[i].clear();
	}

	// Get a reference to the active write buffer (call once per frame).
	inline LocalVector<T> &get_write_buffer() {
		return buffers[write_index.load(std::memory_order_acquire)];
	}

	// Swap the completed write buffer to the stable read position.
	// Call after all producers have finished writing for the frame.
	inline void swap() {
		int prev_stable = stable_index.load(std::memory_order_relaxed);
		int new_write  = (prev_stable + 1) % 3;
		stable_index.store(write_index.load(std::memory_order_acquire), std::memory_order_release);
		write_index.store(new_write, std::memory_order_release);
		buffers[new_write].clear();
	}

	// Get the stable buffer for reading (call once per frame by consumer).
	inline const LocalVector<T> &get_read_buffer() const {
		return buffers[stable_index.load(std::memory_order_acquire)];
	}
};

// ---------------------------------------------------------------------------
// UnifiedPhysicsEventBus – a singletonish resource that collects and
// dispatches events from all physics engines.
// ---------------------------------------------------------------------------
class UnifiedPhysicsEventBus : public RefCounted {
	GDCLASS(UnifiedPhysicsEventBus, RefCounted);

	// Triple buffers for each event type
	TripleBuffer<CollisionEvent>   collision_buffer;
	TripleBuffer<TriggerEvent>     trigger_buffer;
	TripleBuffer<JointBreakEvent>  joint_break_buffer;

	// Callables registered by users (one per engine, or global)
	Callable global_collision_callback;
	Callable global_trigger_callback;
	Callable global_joint_break_callback;

	uint32_t frame_counter;

public:
	UnifiedPhysicsEventBus() : frame_counter(0) {
		collision_buffer.reserve(1024);
		trigger_buffer.reserve(256);
		joint_break_buffer.reserve(128);
	}

	// --- Registration ---
	void set_collision_callback(const Callable &p_cb) { global_collision_callback = p_cb; }
	void set_trigger_callback(const Callable &p_cb) { global_trigger_callback = p_cb; }
	void set_joint_break_callback(const Callable &p_cb) { global_joint_break_callback = p_cb; }

	// --- Producers (called from physics steps of each engine) ---
	inline void publish_collision(uint8_t engine, uint64_t a, uint64_t b, bool started) {
		CollisionEvent ev;
		ev.engine = engine;
		ev.body_a = a;
		ev.body_b = b;
		ev.started = started;
		ev.frame = frame_counter;
		collision_buffer.get_write_buffer().push_back(ev);
	}
	inline void publish_trigger(uint8_t engine, uint64_t trigger, uint64_t other, bool entered) {
		TriggerEvent ev;
		ev.engine = engine;
		ev.trigger_body = trigger;
		ev.other_body = other;
		ev.entered = entered;
		ev.frame = frame_counter;
		trigger_buffer.get_write_buffer().push_back(ev);
	}
	inline void publish_joint_break(uint8_t engine, uint64_t joint, uint64_t a, uint64_t b,
	                                real_t force, real_t torque) {
		JointBreakEvent ev;
		ev.engine = engine;
		ev.joint_id = joint;
		ev.body_a = a;
		ev.body_b = b;
		ev.break_force = force;
		ev.break_torque = torque;
		ev.frame = frame_counter;
		joint_break_buffer.get_write_buffer().push_back(ev);
	}

	// --- End of frame: swap buffers and dispatch callbacks ---
	void flush() {
		collision_buffer.swap();
		trigger_buffer.swap();
		joint_break_buffer.swap();

		// Dispatch callbacks with the stable read buffers.
		if (global_collision_callback.is_valid()) {
			const LocalVector<CollisionEvent> &events = collision_buffer.get_read_buffer();
			Array arr;
			for (const CollisionEvent &ev : events) {
				Dictionary d;
				d["engine"] = ev.engine;
				d["body_a"] = ev.body_a;
				d["body_b"] = ev.body_b;
				d["started"] = ev.started;
				arr.push_back(d);
			}
			global_collision_callback.callv({arr});
		}
		if (global_trigger_callback.is_valid()) {
			const LocalVector<TriggerEvent> &events = trigger_buffer.get_read_buffer();
			Array arr;
			for (const TriggerEvent &ev : events) {
				Dictionary d;
				d["engine"] = ev.engine;
				d["trigger"] = ev.trigger_body;
				d["other"] = ev.other_body;
				d["entered"] = ev.entered;
				arr.push_back(d);
			}
			global_trigger_callback.callv({arr});
		}
		if (global_joint_break_callback.is_valid()) {
			const LocalVector<JointBreakEvent> &events = joint_break_buffer.get_read_buffer();
			Array arr;
			for (const JointBreakEvent &ev : events) {
				Dictionary d;
				d["engine"] = ev.engine;
				d["joint"] = ev.joint_id;
				d["body_a"] = ev.body_a;
				d["body_b"] = ev.body_b;
				d["force"] = ev.break_force;
				d["torque"] = ev.break_torque;
				arr.push_back(d);
			}
			global_joint_break_callback.callv({arr});
		}

		frame_counter++;
	}

	// Clear all buffers and callbacks.
	void clear() {
		collision_buffer.clear();
		trigger_buffer.clear();
		joint_break_buffer.clear();
		global_collision_callback = Callable();
		global_trigger_callback = Callable();
		global_joint_break_callback = Callable();
		frame_counter = 0;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_collision_callback", "callable"), &UnifiedPhysicsEventBus::set_collision_callback);
		ClassDB::bind_method(D_METHOD("set_trigger_callback", "callable"), &UnifiedPhysicsEventBus::set_trigger_callback);
		ClassDB::bind_method(D_METHOD("set_joint_break_callback", "callable"), &UnifiedPhysicsEventBus::set_joint_break_callback);
		ClassDB::bind_method(D_METHOD("flush"), &UnifiedPhysicsEventBus::flush);
		ClassDB::bind_method(D_METHOD("clear"), &UnifiedPhysicsEventBus::clear);
	}
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_PHYSICS_EVENT_BUS_H