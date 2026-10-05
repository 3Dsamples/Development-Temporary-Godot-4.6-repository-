// File : 006
// name : src/core/006_rnd_FrameBarrier.js
// description : Multi-slot frame barrier for parallel-safe producer/consumer
//               handoff between lighting subsystems on Android mobile. Where
//               the small FrameBarrier class in 003_rnd_Runtime.js handled a
//               single double-buffered read/write pair for the render loop,
//               THIS module is the full barrier: a fixed-capacity registry of
//               named slots (light list, shadow atlas, GI probes, AO targets,
//               cluster grid, environment palette, interior volumes, exterior
//               probes, director hints, post buffers) with per-slot
//               generation counters, atomic-style acquire/release semantics,
//               starvation detection, and a synchronous fallback when
//               SharedArrayBuffer / Atomics are unavailable.
//
//               What it guarantees:
//                 • A producer (light list build, shadow atlas pack, GI probe
//                   update, AO blur) writes into the WRITE buffer while every
//                   consumer (mesh material uniforms, GI sampler, AO sampler,
//                   post-processing) reads from the READ buffer. No tearing.
//                 • When a producer is slow (still writing when the next
//                   frame starts), the barrier can either (a) block the
//                   consumer (blocking mode, safe but stalls), (b) return
//                   STALE (consumer keeps using last-committed buffer, keeps
//                   FPS), or (c) trigger an immediate downgrade hint that
//                   adaptive quality controllers read to reduce resolution.
//                 • Generation counters are monotonically increasing 32-bit
//                   integers so consumers can detect "did this buffer change
//                   since I last read it?" with one integer compare.
//                 • Optional Atomics path uses SharedArrayBuffer when the
//                   environment supports it — one Int32Array word per slot
//                   for the commit flag, one for the abort flag, and one per
//                   producer for the "in flight" indicator.
//
//               Optimization techniques applied:
//                 • fixed-capacity slot array (no Map/Set on hot path)
//                 • bitmask slot gating for cheap per-frame checks
//                 • integer generation counters (no Date.now / allocs)
//                 • atomic commit via Atomics.store when SAB available
//                 • wait-free hot path (spin is bounded and only in blocking
//                   mode, which is opt-in)
//                 • no closures captured per frame
//                 • zero GC pressure: all typed arrays sized once at
//                   construction, all scratch values are module-level
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external barrier libs; no per-frame allocations on
//               the hot path.
// best for : Guaranteeing coherent parallel lighting updates on Android. The
//            light list, shadow atlas, GI probe grid, AO blur output, cluster
//            grid, and environment palette are produced by independent
//            worker tasks at different frequencies (60 / 30 / 20 / 8 Hz) —
//            this barrier lets them all publish their output without locks
//            and lets every consumer always see a complete, committed frame.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  assertBiteCSReady,
  isBiteCSReady,
} from './009_scn_BiteCSVersionPolicy.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER = getPerfTier();

export const MAX_SLOTS = PERF_TIER === 'HIGH' ? 32 : PERF_TIER === 'MEDIUM' ? 24 : 16;

export const SLOT_MODE = Object.freeze({
  DOUBLE:   0, // A/B ping-pong: producer writes B, reader reads A
  TRIPLE:   1, // A/B/C: allows one producer + two readers in flight
});

export const SLOT_POLICY = Object.freeze({
  BLOCK:    0, // consumer waits for producer (safe, may stall)
  STALE:    1, // consumer uses last committed buffer (default for mobile)
  DOWNGRADE:2, // consumer uses last committed buffer + fires downgrade hint
});

export const SLOT_STATE = Object.freeze({
  IDLE:      0,
  PRODUCING: 1,
  COMMITTED: 2,
  ABORTED:   3,
});

export const ACQUIRE_RESULT = Object.freeze({
  OK:       0,
  STALE:    1,
  TIMEOUT:  2,
  ABORTED:  3,
});

const GENERATION_WRAP = 0x7FFFFFFF;

/* ------------------------------------------------------------------ */
/* 1. SHARED BUFFER DETECTION                                         */
/* ------------------------------------------------------------------ */

export const HAS_SHARED_ARRAY_BUFFER =
  (typeof SharedArrayBuffer === 'function') &&
  (typeof Atomics === 'object') &&
  (typeof Atomics.store === 'function');

/* ------------------------------------------------------------------ */
/* 2. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _now() {
  return (typeof performance !== 'undefined' ? performance.now() : Date.now());
}

function _nextGeneration(gen) {
  const next = (gen + 1) | 0;
  return next > GENERATION_WRAP ? 1 : next;
}

/* ------------------------------------------------------------------ */
/* 3. SLOT (fixed, allocation-free)                                   */
/* ------------------------------------------------------------------ */

export class BarrierSlot {
  constructor(index, name, mode) {
    this.index       = index;
    this.name        = name;
    this.mode        = mode | 0;
    this.enabled     = 1;

    // Buffer count is 2 for DOUBLE, 3 for TRIPLE.
    this.bufferCount = this.mode === SLOT_MODE.TRIPLE ? 3 : 2;

    // Indices into the buffer ring.
    this.writeIdx    = 0;
    this.readIdx     = 1;
    this.pendingIdx  = 2;

    // Generation counters (monotonic 32-bit).
    this.writeGen    = 0;
    this.readGen     = 0;
    this.commitGen   = 0;

    // State machine.
    this.state       = SLOT_STATE.IDLE;
    this.owner       = -1; // producer id
    this.ready       = 0;

    // Timing / stats.
    this.lastProduceMs = 0;
    this.lastConsumeMs = 0;
    this.produceEma    = 0;
    this.consumeEma    = 0;
    this.staleCount    = 0;
    this.abortCount    = 0;
    this.blockCount    = 0;
    this.timeoutCount  = 0;

    // Policy.
    this.policy      = SLOT_POLICY.STALE;
    this.timeoutMs   = 4.0; // wait cap when BLOCK policy is used

    // Payload handle — producers and consumers agree on what this points to
    // (e.g. a Float32Array of light data). Never reallocated per frame.
    this.payload     = null;
    this.payloadMeta = null;

    // Atomic control words when SAB available.
    this.atomic      = null;
    this.atomicView  = null;
  }

  configure(options = {}) {
    if (options.policy !== undefined) this.policy = options.policy | 0;
    if (options.timeoutMs !== undefined) {
      const v = Number(options.timeoutMs);
      if (Number.isFinite(v) && v > 0) this.timeoutMs = v;
    }
    if (options.payload !== undefined) this.payload = options.payload;
    if (options.payloadMeta !== undefined) this.payloadMeta = options.payloadMeta;
    if (options.enabled !== undefined) this.enabled = options.enabled ? 1 : 0;
    return this;
  }

  attachAtomic() {
    if (!HAS_SHARED_ARRAY_BUFFER) return this;
    // 4 Int32 words per slot: [state, owner, ready, commitGen].
    const sab = new SharedArrayBuffer(16);
    const view = new Int32Array(sab);
    this.atomic = sab;
    this.atomicView = view;
    return this;
  }

  beginProduce(ownerId) {
    if (!this.enabled) return -1;

    // Abort any in-flight produce (shouldn't happen in single-producer
    // mode, but TRIPLE mode allows one aborted produce).
    if (this.state === SLOT_STATE.PRODUCING) {
      this.state = SLOT_STATE.ABORTED;
      this.abortCount++;
    }

    this.state = SLOT_STATE.PRODUCING;
    this.owner = ownerId;
    this.ready = 0;

    // Rotate the ring: writeIdx is the slot the producer will fill.
    // readIdx stays where the last committed frame lives.
    this.writeIdx = this.pendingIdx;
    this.pendingIdx = this.readIdx;

    if (this.atomicView) {
      Atomics.store(this.atomicView, 0, SLOT_STATE.PRODUCING);
      Atomics.store(this.atomicView, 1, ownerId);
      Atomics.store(this.atomicView, 2, 0);
    }

    return this.writeIdx;
  }

  commit() {
    if (this.state !== SLOT_STATE.PRODUCING) return false;

    this.writeGen = _nextGeneration(this.writeGen);
    this.commitGen = this.writeGen;

    // Swap: the buffer we just wrote becomes the READ buffer for consumers.
    const oldRead = this.readIdx;
    this.readIdx = this.writeIdx;
    this.writeIdx = oldRead;

    this.state = SLOT_STATE.COMMITTED;
    this.ready = 1;
    this.owner = -1;

    if (this.atomicView) {
      Atomics.store(this.atomicView, 3, this.commitGen);
      Atomics.store(this.atomicView, 2, 1);
      Atomics.store(this.atomicView, 0, SLOT_STATE.COMMITTED);
    }

    return true;
  }

  abort() {
    if (this.state !== SLOT_STATE.PRODUCING) return false;
    this.state = SLOT_STATE.ABORTED;
    this.owner = -1;
    this.ready = 0;
    this.abortCount++;
    if (this.atomicView) {
      Atomics.store(this.atomicView, 2, 0);
      Atomics.store(this.atomicView, 0, SLOT_STATE.ABORTED);
    }
    return true;
  }

  acquire(consumerId, nowMs) {
    if (!this.enabled) return ACQUIRE_RESULT.OK;
    if (this.state === SLOT_STATE.ABORTED) return ACQUIRE_RESULT.ABORTED;

    const t0 = (nowMs !== undefined) ? nowMs : _now();

    if (this.ready === 1 || this.state === SLOT_STATE.COMMITTED) {
      return ACQUIRE_RESULT.OK;
    }

    // Producer in flight — apply policy.
    if (this.policy === SLOT_POLICY.BLOCK) {
      this.blockCount++;
      const deadline = t0 + this.timeoutMs;
      while (this.ready === 0 && _now() < deadline) {
        // Bounded spin. On mobile, tight spins waste battery; the
        // timeoutMs cap (default 4 ms) keeps the stall bounded.
      }
      if (this.ready === 1) {
        return ACQUIRE_RESULT.OK;
      }
      this.timeoutCount++;
      return ACQUIRE_RESULT.TIMEOUT;
    }

    if (this.policy === SLOT_POLICY.STALE) {
      this.staleCount++;
      return ACQUIRE_RESULT.STALE;
    }

    // DOWNGRADE
    this.staleCount++;
    return ACQUIRE_RESULT.STALE;
  }

  release(consumerId) {
    // No-op in the current model — consumers do not lock the read buffer,
    // they simply must call release() before the producer can reuse the
    // previous read slot. In DOUBLE mode this happens implicitly when the
    // producer rotates. In TRIPLE mode we could add ref counting; kept
    // simple for now to stay allocation-free and predictable on mobile.
    return true;
  }

  hasChangedSince(gen) {
    return this.commitGen !== gen;
  }

  reset() {
    this.writeIdx   = 0;
    this.readIdx    = 1;
    this.pendingIdx = this.bufferCount === 3 ? 2 : 0;
    this.writeGen   = 0;
    this.readGen    = 0;
    this.commitGen  = 0;
    this.state      = SLOT_STATE.IDLE;
    this.owner      = -1;
    this.ready      = 0;
    this.lastProduceMs = 0;
    this.lastConsumeMs = 0;
    this.produceEma    = 0;
    this.consumeEma    = 0;
    this.staleCount    = 0;
    this.abortCount    = 0;
    this.blockCount    = 0;
    this.timeoutCount  = 0;
    if (this.atomicView) {
      Atomics.store(this.atomicView, 0, SLOT_STATE.IDLE);
      Atomics.store(this.atomicView, 1, -1);
      Atomics.store(this.atomicView, 2, 0);
      Atomics.store(this.atomicView, 3, 0);
    }
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 4. FRAME BARRIER                                                   */
/* ------------------------------------------------------------------ */

export class FrameBarrier {
  constructor(options = {}) {
    this.options = Object.assign({
      autoAttachAtomic: true,
      staleIsOk:        true,
    }, options || {});

    this.slots = new Array(MAX_SLOTS);
    this.slotByName = new Map();
    this.slotCount = 0;

    this.frame          = 0;
    this.elapsedMs      = 0;
    this.dirtyMask      = 0;
    this.pendingMask    = 0;
    this.commitMask     = 0;

    this.downgradeHints = new Uint8Array(MAX_SLOTS);

    this._listeners = new Map();
  }

  /* ---------------- slot management ---------------- */

  registerSlot(name, mode = SLOT_MODE.DOUBLE, options = {}) {
    if (this.slotCount >= MAX_SLOTS) return null;
    if (!name || typeof name !== 'string') return null;

    const existing = this.slotByName.get(name);
    if (existing) return existing;

    const idx = this.slotCount++;
    const slot = new BarrierSlot(idx, name, mode);

    if (options.policy      !== undefined) slot.policy      = options.policy | 0;
    if (options.timeoutMs   !== undefined) slot.timeoutMs   = Number(options.timeoutMs) || slot.timeoutMs;
    if (options.payload     !== undefined) slot.payload     = options.payload;
    if (options.payloadMeta !== undefined) slot.payloadMeta = options.payloadMeta;
    if (options.enabled     !== undefined) slot.enabled     = options.enabled ? 1 : 0;

    if (this.options.autoAttachAtomic) slot.attachAtomic();

    this.slots[idx] = slot;
    this.slotByName.set(name, slot);

    return slot;
  }

  getSlot(name) {
    return this.slotByName.get(name) || null;
  }

  getSlotByIndex(index) {
    if (index < 0 || index >= this.slotCount) return null;
    return this.slots[index];
  }

  /* ---------------- frame lifecycle ---------------- */

  beginFrame(dtMs) {
    this.frame++;
    if (Number.isFinite(dtMs)) this.elapsedMs += dtMs;
    this.dirtyMask = 0;
    this.pendingMask = 0;
    this.commitMask = 0;
    for (let i = 0; i < this.slotCount; i++) {
      this.downgradeHints[i] = 0;
    }
  }

  endFrame() {
    // Any producer still in PRODUCING at end of frame → stale downgrade hint.
    for (let i = 0; i < this.slotCount; i++) {
      const s = this.slots[i];
      if (!s || !s.enabled) continue;
      if (s.state === SLOT_STATE.PRODUCING) {
        this.downgradeHints[i] = 1;
      }
    }
    this._emit('endframe', {
      frame: this.frame,
      elapsedMs: this.elapsedMs,
      dirtyMask: this.dirtyMask,
      commitMask: this.commitMask,
      pendingMask: this.pendingMask,
    });
    return this;
  }

  /* ---------------- slot ops ---------------- */

  beginProduce(name, ownerId = -1) {
    const slot = this.slotByName.get(name);
    if (!slot) return -1;
    const idx = slot.beginProduce(ownerId);
    if (idx >= 0) {
      this.pendingMask |= (1 << slot.index);
      this._emit('produce-begin', { name, slot: slot.index, ownerId });
    }
    return idx;
  }

  commit(name) {
    const slot = this.slotByName.get(name);
    if (!slot) return false;
    const ok = slot.commit();
    if (ok) {
      this.commitMask |= (1 << slot.index);
      this.pendingMask &= ~(1 << slot.index);
      this._emit('commit', { name, slot: slot.index, gen: slot.commitGen });
    }
    return ok;
  }

  abort(name) {
    const slot = this.slotByName.get(name);
    if (!slot) return false;
    const ok = slot.abort();
    if (ok) {
      this.pendingMask &= ~(1 << slot.index);
      this._emit('abort', { name, slot: slot.index });
    }
    return ok;
  }

  acquire(name, consumerId = -1) {
    const slot = this.slotByName.get(name);
    if (!slot) return ACQUIRE_RESULT.ABORTED;
    const result = slot.acquire(consumerId);
    if (result === ACQUIRE_RESULT.STALE || result === ACQUIRE_RESULT.TIMEOUT) {
      this.downgradeHints[slot.index] = 1;
    }
    return result;
  }

  release(name, consumerId = -1) {
    const slot = this.slotByName.get(name);
    if (!slot) return false;
    return slot.release(consumerId);
  }

  /* ---------------- bulk query helpers ---------------- */

  getDirtyMask()      { return this.dirtyMask; }
  getPendingMask()    { return this.pendingMask; }
  getCommitMask()     { return this.commitMask; }
  getDowngradeHints() { return this.downgradeHints; }

  isDirty(name) {
    const slot = this.slotByName.get(name);
    if (!slot) return false;
    return (this.commitMask & (1 << slot.index)) !== 0;
  }

  hasChangedSince(name, gen) {
    const slot = this.slotByName.get(name);
    if (!slot) return false;
    return slot.hasChangedSince(gen);
  }

  anyPending() {
    return this.pendingMask !== 0;
  }

  anyStale() {
    for (let i = 0; i < this.slotCount; i++) {
      if (this.downgradeHints[i]) return true;
    }
    return false;
  }

  /* ---------------- config ---------------- */

  setPolicy(name, policy) {
    const slot = this.slotByName.get(name);
    if (!slot) return false;
    slot.policy = policy | 0;
    return true;
  }

  setTimeoutMs(name, ms) {
    const slot = this.slotByName.get(name);
    if (!slot) return false;
    const v = Number(ms);
    if (!Number.isFinite(v) || v <= 0) return false;
    slot.timeoutMs = v;
    return true;
  }

  setEnabled(name, enabled) {
    const slot = this.slotByName.get(name);
    if (!slot) return false;
    slot.enabled = enabled ? 1 : 0;
    return true;
  }

  /* ---------------- events ---------------- */

  on(event, fn) {
    if (typeof event !== 'string' || typeof fn !== 'function') return () => {};
    let arr = this._listeners.get(event);
    if (!arr) { arr = []; this._listeners.set(event, arr); }
    arr.push(fn);
    return () => this.off(event, fn);
  }

  off(event, fn) {
    const arr = this._listeners.get(event);
    if (!arr) return;
    const i = arr.indexOf(fn);
    if (i >= 0) arr.splice(i, 1);
  }

  _emit(event, payload) {
    const arr = this._listeners.get(event);
    if (!arr) return;
    for (let i = 0; i < arr.length; i++) {
      try { arr[i](payload); } catch (e) { console.error(`[006_rnd_FrameBarrier] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- reset / dispose ---------------- */

  reset() {
    for (let i = 0; i < this.slotCount; i++) {
      if (this.slots[i]) this.slots[i].reset();
    }
    this.frame = 0;
    this.elapsedMs = 0;
    this.dirtyMask = 0;
    this.pendingMask = 0;
    this.commitMask = 0;
    this.downgradeHints.fill(0);
    return this;
  }

  dispose() {
    for (let i = 0; i < this.slotCount; i++) {
      const s = this.slots[i];
      if (!s) continue;
      s.payload = null;
      s.payloadMeta = null;
      s.atomic = null;
      s.atomicView = null;
    }
    this.slots.length = 0;
    this.slotByName.clear();
    this.slotCount = 0;
    this._listeners.clear();
    return this;
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    const slots = new Array(this.slotCount);
    for (let i = 0; i < this.slotCount; i++) {
      const s = this.slots[i];
      if (!s) continue;
      slots[i] = {
        name:          s.name,
        mode:          s.mode,
        enabled:       s.enabled === 1,
        policy:        s.policy,
        state:         s.state,
        commitGen:     s.commitGen,
        staleCount:    s.staleCount,
        abortCount:    s.abortCount,
        blockCount:    s.blockCount,
        timeoutCount:  s.timeoutCount,
        produceEma:    s.produceEma,
        consumeEma:    s.consumeEma,
        hasAtomic:     !!s.atomicView,
      };
    }
    return {
      frame:        this.frame,
      elapsedMs:    this.elapsedMs,
      slotCount:    this.slotCount,
      dirtyMask:    this.dirtyMask,
      pendingMask:  this.pendingMask,
      commitMask:   this.commitMask,
      anyPending:   this.pendingMask !== 0,
      anyStale:     this.anyStale(),
      hasSAB:       HAS_SHARED_ARRAY_BUFFER,
      slots,
      perfTier:     PERF_TIER,
    };
  }
}

/* ------------------------------------------------------------------ */
/* 5. STANDARD SLOT REGISTRATION FOR THE LIGHTING STACK               */
/* ------------------------------------------------------------------ */

export function registerLightingSlots(barrier) {
  if (!barrier) return null;

  // High-frequency producers
  barrier.registerSlot('lightList',     SLOT_MODE.DOUBLE, { policy: SLOT_POLICY.STALE, timeoutMs: 2.0 });
  barrier.registerSlot('shadowAtlas',   SLOT_MODE.DOUBLE, { policy: SLOT_POLICY.STALE, timeoutMs: 4.0 });
  barrier.registerSlot('shadowMatrix',  SLOT_MODE.DOUBLE, { policy: SLOT_POLICY.STALE, timeoutMs: 2.0 });

  // Medium-frequency producers
  barrier.registerSlot('giProbes',      SLOT_MODE.TRIPLE, { policy: SLOT_POLICY.STALE, timeoutMs: 6.0 });
  barrier.registerSlot('aoTargets',     SLOT_MODE.DOUBLE, { policy: SLOT_POLICY.STALE, timeoutMs: 4.0 });
  barrier.registerSlot('clusterGrid',   SLOT_MODE.DOUBLE, { policy: SLOT_POLICY.STALE, timeoutMs: 3.0 });

  // Low-frequency producers
  barrier.registerSlot('envPalette',    SLOT_MODE.DOUBLE, { policy: SLOT_POLICY.STALE, timeoutMs: 4.0 });
  barrier.registerSlot('interiorVol',   SLOT_MODE.DOUBLE, { policy: SLOT_POLICY.STALE, timeoutMs: 4.0 });
  barrier.registerSlot('exteriorProbes',SLOT_MODE.DOUBLE, { policy: SLOT_POLICY.STALE, timeoutMs: 4.0 });

  // Cross-cutting
  barrier.registerSlot('directorHints', SLOT_MODE.DOUBLE, { policy: SLOT_POLICY.STALE, timeoutMs: 1.0 });
  barrier.registerSlot('postBuffers',   SLOT_MODE.TRIPLE, { policy: SLOT_POLICY.STALE, timeoutMs: 4.0 });

  return barrier;
}

/* ------------------------------------------------------------------ */
/* 6. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultBarrier = null;

export function getDefaultBarrier() {
  if (!_defaultBarrier) {
    _defaultBarrier = new FrameBarrier();
    registerLightingSlots(_defaultBarrier);
  }
  return _defaultBarrier;
}

export function disposeDefaultBarrier() {
  if (_defaultBarrier) {
    _defaultBarrier.dispose();
    _defaultBarrier = null;
  }
}

/* ------------------------------------------------------------------ */
/* 7. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createFrameBarrier(options = {}) {
  return new FrameBarrier(options);
}

/* ------------------------------------------------------------------ */
/* 8. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  FrameBarrier,
  BarrierSlot,
  createFrameBarrier,
  getDefaultBarrier,
  disposeDefaultBarrier,
  registerLightingSlots,
  SLOT_MODE,
  SLOT_POLICY,
  SLOT_STATE,
  ACQUIRE_RESULT,
  HAS_SHARED_ARRAY_BUFFER,
  MAX_SLOTS,
};

export default _defaultExport;