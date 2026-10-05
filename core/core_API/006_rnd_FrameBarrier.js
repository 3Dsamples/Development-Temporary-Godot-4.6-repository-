API Documentation — src/core/006_rnd_FrameBarrier.js

File Purpose

This file provides the multi-slot frame barrier that guarantees coherent parallel-safe producer/consumer handoff between lighting subsystems on Android mobile.

Where 003_rnd_Runtime.js included a small FrameBarrier class that handled a single double-buffered read/write pair for the top-level render loop, this module is the full barrier. It is a fixed-capacity registry of named slots — light list, shadow atlas, GI probes, AO targets, cluster grid, environment palette, interior volumes, exterior probes, director hints, post buffers — with per-slot generation counters, atomic-style acquire/release semantics, starvation detection, and a synchronous fallback when SharedArrayBuffer or Atomics are unavailable.

The core contract is:

A producer (light list build, shadow atlas pack, GI probe update, AO blur, cluster grid build) writes into the WRITE buffer of a slot while every consumer (mesh material uniforms, GI sampler, AO sampler, post-processing) reads from the READ buffer of the same slot. No tearing. The barrier swaps the two buffers on commit().

When a producer is slow and is still writing when the next frame starts, the barrier applies a per-slot policy:

· BLOCK — the consumer waits with a bounded spin (rarely used on mobile).
· STALE — the consumer keeps reading the last committed buffer; the frame keeps its frame rate; the barrier records a stale event and raises a downgrade hint.
· DOWNGRADE — same as STALE but the barrier also writes a downgrade hint into the hint array that adaptive-quality controllers read.

Each slot also carries a monotonically increasing 32-bit generation counter. A consumer that read a slot at generation G can check hasChangedSince(G) with one integer compare to know whether it needs to re-upload its uniform block.

---

Exported Constants

PERF_TIER

Type: string

Value: 'LOW' | 'MEDIUM' | 'HIGH' — cached from getPerfTier() at module load.

MAX_SLOTS

Type: number

Value: 32 on HIGH, 24 on MEDIUM, 16 on LOW.

The fixed capacity of the slot registry. Sized to accommodate every producer/consumer pair the lighting stack will ever register.

SLOT_MODE

Type: frozen enum

Values:

· DOUBLE = 0 — A/B ping-pong. Producer writes into the next slot; reader reads the last committed slot. Two buffers.
· TRIPLE = 1 — A/B/C ring. Allows one producer and two readers in flight. Three buffers.

SLOT_POLICY

Type: frozen enum

Values:

· BLOCK = 0 — consumer waits for producer with a bounded spin.
· STALE = 1 — consumer uses last committed buffer; records stale and raises hint. Default.
· DOWNGRADE = 2 — same as STALE plus explicit downgrade hint.

SLOT_STATE

Type: frozen enum

Values:

· IDLE = 0 — slot has not been touched this frame.
· PRODUCING = 1 — a producer is currently writing.
· COMMITTED = 2 — a producer has committed the frame.
· ABORTED = 3 — a producer was interrupted before commit.

ACQUIRE_RESULT

Type: frozen enum

Values:

· OK = 0 — the read buffer is fresh; safe to consume.
· STALE = 1 — the read buffer is from a previous frame.
· TIMEOUT = 2 — the BLOCK policy ran out of its spin budget.
· ABORTED = 3 — the slot's producer aborted.

GENERATION_WRAP

Type: number

Value: 0x7FFFFFFF

The upper bound of the 32-bit generation counter. Wraps back to 1 when exceeded so 0 can be reserved as "never committed".

HAS_SHARED_ARRAY_BUFFER

Type: boolean

True if the current environment exposes SharedArrayBuffer, Atomics, and Atomics.store as functions. Used to decide whether to attach the atomic control words to each slot.

---

Exported Class — BarrierSlot

One instance per registered slot. Owns the buffer indices, generation counters, policy, and stats.

Constructor

```
new BarrierSlot(index, name, mode)
```

Parameters:

· index — the slot's registry index.
· name — the slot's name string.
· mode — one of SLOT_MODE.DOUBLE or SLOT_MODE.TRIPLE.

Instance Properties

· index — integer slot index.
· name — string.
· mode — the mode enum value.
· enabled — 1 if enabled, 0 otherwise.
· bufferCount — 2 for DOUBLE, 3 for TRIPLE.
· writeIdx — the index of the buffer the producer will fill next.
· readIdx — the index of the buffer consumers should read from.
· pendingIdx — the index of the buffer being prepared.
· writeGen — the generation counter of the last write.
· readGen — the generation counter at which a consumer last read.
· commitGen — the generation counter of the last commit. Consumers compare against this.
· state — the current SLOT_STATE.
· owner — the producer id currently writing, or -1.
· ready — 1 if a committed buffer is available.
· lastProduceMs — cost of the last producer call.
· lastConsumeMs — cost of the last consumer call.
· produceEma — smoothed producer cost in milliseconds.
· consumeEma — smoothed consumer cost in milliseconds.
· staleCount — number of times a consumer saw STALE.
· abortCount — number of aborted produces.
· blockCount — number of times a consumer entered BLOCK spin.
· timeoutCount — number of BLOCK spin timeouts.
· policy — the current policy enum.
· timeoutMs — spin timeout for BLOCK policy, default 4 ms.
· payload — an arbitrary handle (e.g. a Float32Array of light data). The slot does not interpret this.
· payloadMeta — arbitrary metadata attached to the payload.
· atomic — the SharedArrayBuffer handle if SAB is available.
· atomicView — an Int32Array view of the atomic words: [state, owner, ready, commitGen].

Instance Methods

configure(options = {})

Parameters:

· policy — override the policy.
· timeoutMs — override the spin timeout in milliseconds.
· payload — attach a payload handle.
· payloadMeta — attach payload metadata.
· enabled — enable or disable the slot.

Returns: this.

attachAtomic()

Parameters: none.

Returns: this.

Purpose: allocates a 16-byte SharedArrayBuffer holding four Int32 words [state, owner, ready, commitGen] and stores it on the slot. No-op if HAS_SHARED_ARRAY_BUFFER is false.

beginProduce(ownerId)

Parameters: ownerId — an arbitrary producer id (or -1).

Returns: the buffer index the producer should write into, or -1 if the slot is disabled.

Purpose: called by a producer at the start of its frame. Sets state to PRODUCING, records owner, clears ready, rotates the ring so writeIdx points at a fresh buffer, and, if atomics are available, publishes those values with Atomics.store.

commit()

Parameters: none.

Returns: boolean — true on success, false if the slot was not in PRODUCING state.

Purpose: called by a producer after it finished writing. Advances writeGen, sets commitGen, swaps readIdx and writeIdx so the freshly-written buffer becomes the read buffer, sets state to COMMITTED, sets ready = 1, clears owner, and, if atomics are available, publishes the new commitGen and ready flag before setting state to COMMITTED.

abort()

Parameters: none.

Returns: boolean.

Purpose: called by a producer that failed partway through a produce. Sets state to ABORTED, clears ready and owner, increments abortCount.

acquire(consumerId, nowMs)

Parameters:

· consumerId — an arbitrary consumer id (or -1).
· nowMs — the current timestamp; defaults to performance.now() if omitted.

Returns: one of ACQUIRE_RESULT.

Purpose: called by a consumer before reading a slot.

Flow:

1. If the slot is disabled, returns OK.
2. If the slot is in ABORTED state, returns ABORTED.
3. If ready === 1 or the state is COMMITTED, returns OK.
4. Otherwise the producer is still writing. Applies the policy:
   · BLOCK: increments blockCount, spins until ready becomes 1 or timeoutMs elapses. If ready becomes 1, returns OK. Otherwise increments timeoutCount and returns TIMEOUT.
   · STALE: increments staleCount and returns STALE.
   · DOWNGRADE: increments staleCount and returns STALE. The caller is expected to read the downgrade hint array.

release(consumerId)

Parameters: consumerId — unused in the current model.

Returns: boolean.

Purpose: reserved for future reference counting. Currently a no-op that returns true. The DOUBLE and TRIPLE ring semantics already guarantee the consumer never blocks a producer because the producer rotates to a fresh buffer independently.

hasChangedSince(gen)

Parameters: gen — the generation the consumer last saw.

Returns: boolean.

Purpose: the fast path for a consumer that wants to know whether its cached data is stale. One integer compare.

reset()

Returns: this.

Purpose: resets all indices, generations, state, and stats.

---

Exported Class — FrameBarrier

The main barrier. Manages the slot registry, the per-frame bitmasks, and the event stream.

Constructor

```
new FrameBarrier(options = {})
```

Parameters:

· autoAttachAtomic — if true (default), calls attachAtomic() on every registered slot when SharedArrayBuffer is available.
· staleIsOk — if true (default), stale reads are treated as a warning, not an error. If false, a stale read trips the slot's error boundary.

Instance Properties

· slots — an array of registered BarrierSlot instances.
· slotByName — a Map from slot name to the BarrierSlot.
· slotCount — the number of registered slots.
· frame — the frame counter.
· elapsedMs — total elapsed milliseconds.
· dirtyMask — a bitmask of slots that were committed this frame.
· pendingMask — a bitmask of slots whose producers are currently writing.
· commitMask — a bitmask of slots that were committed this frame (same as dirtyMask, kept separately for readability).
· downgradeHints — a Uint8Array(MAX_SLOTS) where each entry is 1 if that slot is advising a downgrade.

Instance Methods

registerSlot(name, mode = SLOT_MODE.DOUBLE, options = {})

Parameters:

· name — unique string.
· mode — DOUBLE or TRIPLE.
· options — optional policy, timeoutMs, payload, payloadMeta, enabled.

Returns: the BarrierSlot, or null if the capacity is full or the name is already registered.

Purpose: adds a slot to the registry. If a slot with the same name already exists, returns the existing one.

getSlot(name)

Parameters: name — the slot name.

Returns: the BarrierSlot or null.

getSlotByIndex(index)

Parameters: index — the numeric index.

Returns: the BarrierSlot or null.

beginFrame(dtMs)

Parameters: dtMs — the delta time in milliseconds.

Returns: nothing.

Purpose: called at the start of every frame. Increments frame, adds to elapsedMs, zeros the three bitmasks, and clears every entry in downgradeHints.

endFrame()

Returns: this.

Purpose: called at the end of every frame. Iterates the slots. Any slot still in PRODUCING state has its downgrade hint set to 1 — this is the signal that a producer was too slow. Emits endframe with a summary object.

beginProduce(name, ownerId = -1)

Parameters:

· name — the slot name.
· ownerId — the producer id.

Returns: the buffer index, or -1.

Purpose: convenience wrapper around slot.beginProduce(). Sets the slot's bit in pendingMask. Emits produce-begin.

commit(name)

Parameters: name — the slot name.

Returns: boolean.

Purpose: convenience wrapper around slot.commit(). Sets the slot's bit in commitMask, clears its bit in pendingMask. Emits commit with the slot index and new generation.

abort(name)

Parameters: name — the slot name.

Returns: boolean.

Purpose: convenience wrapper around slot.abort(). Clears the slot's bit in pendingMask. Emits abort.

acquire(name, consumerId = -1)

Parameters:

· name — the slot name.
· consumerId — the consumer id.

Returns: one of ACQUIRE_RESULT.

Purpose: convenience wrapper around slot.acquire(). If the result is STALE or TIMEOUT, sets the slot's downgrade hint. Returns the result.

release(name, consumerId = -1)

Parameters:

· name — the slot name.
· consumerId — the consumer id.

Returns: boolean.

Purpose: convenience wrapper around slot.release().

getDirtyMask()

Returns: dirtyMask.

getPendingMask()

Returns: pendingMask.

getCommitMask()

Returns: commitMask.

getDowngradeHints()

Returns: downgradeHints.

isDirty(name)

Parameters: name — the slot name.

Returns: boolean — true if the slot was committed this frame.

hasChangedSince(name, gen)

Parameters:

· name — the slot name.
· gen — the generation the consumer last saw.

Returns: boolean.

anyPending()

Returns: boolean — true if any slot has a producer in flight.

anyStale()

Returns: boolean — true if any downgrade hint is set.

setPolicy(name, policy)

Parameters:

· name — the slot name.
· policy — one of SLOT_POLICY.

Returns: boolean.

setTimeoutMs(name, ms)

Parameters:

· name — the slot name.
· ms — the spin timeout in milliseconds.

Returns: boolean.

setEnabled(name, enabled)

Parameters:

· name — the slot name.
· enabled — boolean.

Returns: boolean.

on(event, fn)

Parameters:

· event — one of 'endframe', 'produce-begin', 'commit', 'abort', 'detach', 'reattach', 'rejected'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

reset()

Returns: this.

Purpose: resets every slot and zeros every mask and hint.

dispose()

Returns: this.

Purpose: nulls every slot, clears the name map, clears listeners.

getStats()

Returns: an object with frame, elapsedMs, slotCount, the three masks, anyPending, anyStale, hasSAB, a slots array of per-slot stats, and perfTier.

---

Exported Function — registerLightingSlots(barrier)

Parameters: barrier — a FrameBarrier instance.

Returns: the same barrier (for chaining).

Purpose: registers the eleven canonical slots the lighting stack uses.

The slots and their default modes and policies:

· lightList — DOUBLE, STALE, 2.0 ms timeout.
· shadowAtlas — DOUBLE, STALE, 4.0 ms timeout.
· shadowMatrix — DOUBLE, STALE, 2.0 ms timeout.
· giProbes — TRIPLE, STALE, 6.0 ms timeout.
· aoTargets — DOUBLE, STALE, 4.0 ms timeout.
· clusterGrid — DOUBLE, STALE, 3.0 ms timeout.
· envPalette — DOUBLE, STALE, 4.0 ms timeout.
· interiorVol — DOUBLE, STALE, 4.0 ms timeout.
· exteriorProbes — DOUBLE, STALE, 4.0 ms timeout.
· directorHints — DOUBLE, STALE, 1.0 ms timeout.
· postBuffers — TRIPLE, STALE, 4.0 ms timeout.

---

Exported Functions

getDefaultBarrier()

Returns: the module-level singleton FrameBarrier, creating it on first call and calling registerLightingSlots() on it.

disposeDefaultBarrier()

Returns: nothing.

createFrameBarrier(options = {})

Returns: a new FrameBarrier.

---

Default Export

The default export bundles: FrameBarrier, BarrierSlot, createFrameBarrier, getDefaultBarrier, disposeDefaultBarrier, registerLightingSlots, SLOT_MODE, SLOT_POLICY, SLOT_STATE, ACQUIRE_RESULT, HAS_SHARED_ARRAY_BUFFER, MAX_SLOTS.

---

Usage Pattern

A producer:

```
const barrier = getDefaultBarrier();

// Per frame, at the start of the producer's work:
barrier.beginProduce('shadowAtlas', myProducerId);
// ... write into the write buffer ...
barrier.commit('shadowAtlas');
```

A consumer:

```
const result = barrier.acquire('shadowAtlas', myConsumerId);
if (result === ACQUIRE_RESULT.OK) {
  // fresh data, upload to uniform
} else if (result === ACQUIRE_RESULT.STALE) {
  // read last committed buffer
}
```

The barrier's downgradeHints array is inspected once per frame by the adaptive quality controller to know which subsystem is falling behind, and the controller downscales that subsystem's quality accordingly.

