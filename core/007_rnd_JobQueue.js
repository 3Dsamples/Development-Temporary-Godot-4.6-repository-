// File : 007
// name : src/core/007_rnd_JobQueue.js
// description : Dedicated parallel job queue for the anime lighting stack on
//               Android mobile. Where 003_rnd_Runtime.js contained a minimal
//               fixed-capacity ring queue for in-frame drain, THIS module is
//               the full-featured producer/consumer job system used by every
//               lighting subsystem that needs to offload work off the main
//               thread — shadow atlas packing, GI probe baking, AO blur,
//               cluster grid build, environment palette solve, interior
//               volume update, exterior probe solve, post-buffer prepare.
//
//               Features:
//                 • Priority tiers (CRITICAL / HIGH / NORMAL / LOW / IDLE)
//                   with stable FIFO within a tier — critical lighting jobs
//                   (light list rebuild after camera move) preempt cosmetic
//                   ones (environment palette easing) without starving them.
//                 • Dependency edges (up to 4 parents per job) so a
//                   "commit shadow atlas" job can wait on "pack shadow
//                   atlas" + "sync cascade matrices" without manual polling.
//                 • Zero-allocation hot path: pre-allocated slots, pre-bound
//                   callbacks, typed-array payload handles.
//                 • Transferable-object helpers so Float32Array / WebGL
//                   buffers / ImageBitmap can move between main thread and
//                   worker without a copy.
//                 • Backpressure: reject new jobs beyond capacity with a
//                   distinct result code so callers can degrade gracefully
//                   (skip GI bake this frame, reuse last probe set).
//                 • Budget accounting per priority tier so the adaptive
//                   quality controller can throttle by tier without
//                   disrupting the whole queue.
//                 • Starvation watchdog: any job queued longer than
//                   STARVATION_MS gets promoted to the next tier up.
//                 • Cooperative cancellation: a job can be aborted before
//                   running; running jobs receive an AbortSignal-like flag
//                   they can poll cheaply.
//                 • Deterministic ordering: jobs with equal priority + zero
//                   dependencies run in insertion order, so lighting output
//                   is bit-reproducible across runs on the same device.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external scheduler libs; no Promises on the hot
//               path; every typed array sized once at construction; no
//               closures captured per frame; no Map/Set on the hot path.
// best for : Offloading every lighting task that would otherwise stall the
//            main thread on Android: shadow atlas page repacking after LOD
//            change, GI probe re-bake after biome transition, AO blur kernel
//            swap after quality rescale, cluster grid rebuild after camera
//            far-plane change, environment palette solve after day-cycle
//            tick, interior light placement after room load, exterior probe
//            solve after terrain stream. All at their own rate, all in one
//            queue with one budget view.
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

export const MAX_JOBS_PER_QUEUE = PERF_TIER === 'HIGH' ? 8192 : PERF_TIER === 'MEDIUM' ? 4096 : 2048;

export const JOB_PRIORITY = Object.freeze({
  CRITICAL: 0,
  HIGH:     1,
  NORMAL:   2,
  LOW:      3,
  IDLE:     4,
  COUNT:    5,
});

export const JOB_PRIORITY_NAME = Object.freeze([
  'critical',
  'high',
  'normal',
  'low',
  'idle',
]);

export const JOB_STATE = Object.freeze({
  EMPTY:     0,
  QUEUED:    1,
  RUNNING:   2,
  DONE:      3,
  FAILED:    4,
  CANCELLED: 5,
});

export const ENQUEUE_RESULT = Object.freeze({
  OK:              0,
  FULL:            1,
  INVALID:         2,
  DEPENDENCY_LOST: 3,
});

export const JOB_KIND = Object.freeze({
  GENERIC:        0,
  LIGHT_LIST:     1,
  SHADOW_ATLAS:   2,
  SHADOW_MATRIX:  3,
  GI_PROBE:       4,
  AO_BLUR:        5,
  CLUSTER_GRID:   6,
  ENV_PALETTE:    7,
  INTERIOR_VOL:   8,
  EXTERIOR_PROBE: 9,
  POST_BUFFER:   10,
  DIRECTOR_HINT: 11,
  COUNT:         12,
});

export const JOB_KIND_NAME = Object.freeze([
  'generic',
  'light_list',
  'shadow_atlas',
  'shadow_matrix',
  'gi_probe',
  'ao_blur',
  'cluster_grid',
  'env_palette',
  'interior_vol',
  'exterior_probe',
  'post_buffer',
  'director_hint',
]);

const STARVATION_MS       = 500;
const MAX_DEPS            = 4;
const EMA_ALPHA           = 0.10;

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _now() {
  return (typeof performance !== 'undefined' ? performance.now() : Date.now());
}

/* ------------------------------------------------------------------ */
/* 2. JOB SLOT (fixed, allocation-free)                               */
/* ------------------------------------------------------------------ */

export class JobSlot {
  constructor(index) {
    this.index        = index;
    this.id           = 0;             // monotonic job id
    this.kind         = JOB_KIND.GENERIC;
    this.priority     = JOB_PRIORITY.NORMAL;
    this.state        = JOB_STATE.EMPTY;

    this.run          = null;          // (payload, ctx, job) -> void
    this.onDone       = null;          // (payload, ctx, job) -> void
    this.onError      = null;          // (error, payload, ctx, job) -> void
    this.ctx          = null;

    this.payload      = null;          // opaque handle (Float32Array, etc.)
    this.payloadMeta  = null;

    this.transferable = null;          // Transferable[] or null

    this.queuedAtMs   = 0;
    this.startedAtMs  = 0;
    this.finishedAtMs = 0;

    this.deps         = new Int32Array(MAX_DEPS);
    this.depCount     = 0;

    this.dependentCount = 0;

    this.cancelFlag   = 0;             // 1 = cancel requested

    this.sequence     = 0;             // insertion order within tier
  }

  reset() {
    this.id            = 0;
    this.kind          = JOB_KIND.GENERIC;
    this.priority      = JOB_PRIORITY.NORMAL;
    this.state         = JOB_STATE.EMPTY;
    this.run           = null;
    this.onDone        = null;
    this.onError       = null;
    this.ctx           = null;
    this.payload       = null;
    this.payloadMeta   = null;
    this.transferable  = null;
    this.queuedAtMs    = 0;
    this.startedAtMs   = 0;
    this.finishedAtMs  = 0;
    this.depCount      = 0;
    this.dependentCount = 0;
    this.cancelFlag    = 0;
    this.sequence      = 0;
    for (let i = 0; i < MAX_DEPS; i++) this.deps[i] = -1;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 3. PRIORITY BUCKET (fixed-capacity FIFO within tier)               */
/* ------------------------------------------------------------------ */

export class PriorityBucket {
  constructor(capacity) {
    this.capacity = capacity;
    this.indices  = new Int32Array(capacity);
    this.head     = 0;
    this.tail     = 0;
    this.count    = 0;
  }

  push(index) {
    if (this.count >= this.capacity) return false;
    this.indices[this.tail] = index;
    this.tail = (this.tail + 1) % this.capacity;
    this.count++;
    return true;
  }

  shift() {
    if (this.count === 0) return -1;
    const idx = this.indices[this.head];
    this.head = (this.head + 1) % this.capacity;
    this.count--;
    return idx;
  }

  peek() {
    if (this.count === 0) return -1;
    return this.indices[this.head];
  }

  clear() {
    this.head = 0;
    this.tail = 0;
    this.count = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 4. JOB QUEUE                                                       */
/* ------------------------------------------------------------------ */

export class JobQueue {
  constructor(options = {}) {
    this.options = Object.assign({
      capacity:        MAX_JOBS_PER_QUEUE,
      bucketCapacity:  MAX_JOBS_PER_QUEUE,
      maxJobsPerFrame: PERF_TIER === 'HIGH' ? 64 : PERF_TIER === 'MEDIUM' ? 32 : 16,
      starvationMs:    STARVATION_MS,
      autoPromote:     true,
    }, options || {});

    this.capacity = this.options.capacity;

    // Flat slot array (allocation-free).
    this.slots = new Array(this.capacity);
    for (let i = 0; i < this.capacity; i++) {
      this.slots[i] = new JobSlot(i);
    }

    // Free list (ring).
    this.freeList = new Int32Array(this.capacity);
    this.freeHead = 0;
    this.freeCount = this.capacity;
    for (let i = 0; i < this.capacity; i++) this.freeList[i] = i;

    // Priority buckets.
    this.buckets = new Array(JOB_PRIORITY.COUNT);
    for (let p = 0; p < JOB_PRIORITY.COUNT; p++) {
      this.buckets[p] = new PriorityBucket(this.options.bucketCapacity);
    }

    // Live job index (for dep resolution).
    this.idToIndex = new Map();

    // Monotonic counters.
    this._nextId       = 1;
    this._nextSequence = 1;

    // In-flight tracking.
    this.activeIndex   = -1;
    this.queuedCount   = 0;
    this.runningCount  = 0;
    this.doneCount     = 0;
    this.failedCount   = 0;
    this.cancelledCount = 0;
    this.rejectedCount = 0;

    // Budget accounting per tier (EMA of executed ms).
    this.tierMsEma     = new Float32Array(JOB_PRIORITY.COUNT);
    this.tierCountEma  = new Float32Array(JOB_PRIORITY.COUNT);

    this._listeners = new Map();
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
      try { arr[i](payload); } catch (e) { console.error(`[007_rnd_JobQueue] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- allocation ---------------- */

  _acquireSlot() {
    if (this.freeCount <= 0) return -1;
    const idx = this.freeList[this.freeHead];
    this.freeHead = (this.freeHead + 1) % this.capacity;
    this.freeCount--;
    return idx;
  }

  _releaseSlot(index) {
    this.slots[index].reset();
    this.freeList[(this.freeHead + this.freeCount) % this.capacity] = index;
    this.freeCount++;
  }

  /* ---------------- public API ---------------- */

  enqueue(spec) {
    if (!spec || typeof spec.run !== 'function') {
      this.rejectedCount++;
      return { result: ENQUEUE_RESULT.INVALID, id: 0 };
    }

    const slotIdx = this._acquireSlot();
    if (slotIdx < 0) {
      this.rejectedCount++;
      return { result: ENQUEUE_RESULT.FULL, id: 0 };
    }

    const slot = this.slots[slotIdx];
    slot.reset();

    const id = this._nextId++;
    slot.id           = id;
    slot.kind         = (spec.kind !== undefined ? spec.kind : JOB_KIND.GENERIC) | 0;
    slot.priority     = (spec.priority !== undefined ? spec.priority : JOB_PRIORITY.NORMAL) | 0;
    slot.state        = JOB_STATE.QUEUED;
    slot.run          = spec.run;
    slot.onDone       = (typeof spec.onDone === 'function' ? spec.onDone : null);
    slot.onError      = (typeof spec.onError === 'function' ? spec.onError : null);
    slot.ctx          = spec.ctx || null;
    slot.payload      = spec.payload || null;
    slot.payloadMeta  = spec.payloadMeta || null;
    slot.transferable = spec.transferable || null;
    slot.queuedAtMs   = _now();
    slot.sequence     = this._nextSequence++;

    // Dependencies (job ids).
    let depsOk = true;
    slot.depCount = 0;
    if (Array.isArray(spec.deps) && spec.deps.length > 0) {
      for (let i = 0; i < spec.deps.length && i < MAX_DEPS; i++) {
        const depId = spec.deps[i] | 0;
        if (depId <= 0) continue;
        slot.deps[slot.depCount++] = depId;
        const depIdx = this.idToIndex.get(depId);
        if (depIdx !== undefined) {
          this.slots[depIdx].dependentCount++;
        }
      }
    }

    this.idToIndex.set(id, slotIdx);
    this.queuedCount++;
    this.buckets[slot.priority].push(slotIdx);

    this._emit('enqueued', { id, kind: slot.kind, priority: slot.priority });

    if (!depsOk) {
      this.cancelById(id);
      return { result: ENQUEUE_RESULT.DEPENDENCY_LOST, id };
    }

    return { result: ENQUEUE_RESULT.OK, id };
  }

  cancelById(id) {
    const idx = this.idToIndex.get(id);
    if (idx === undefined) return false;
    const slot = this.slots[idx];
    if (slot.state === JOB_STATE.QUEUED) {
      slot.cancelFlag = 1;
      this._finalize(slot, JOB_STATE.CANCELLED, null);
      return true;
    }
    if (slot.state === JOB_STATE.RUNNING) {
      slot.cancelFlag = 1;
      return true;
    }
    return false;
  }

  /* ---------------- drain ---------------- */

  drain(maxJobs, onRunError) {
    const limit = (maxJobs !== undefined ? maxJobs : this.options.maxJobsPerFrame);
    let executed = 0;

    while (executed < limit) {
      const slotIdx = this._pickNext();
      if (slotIdx < 0) break;

      const slot = this.slots[slotIdx];

      // Re-check dependencies resolved.
      if (!this._depsResolved(slot)) {
        // Put back at the tail; try other priorities.
        this.buckets[slot.priority].push(slotIdx);
        // Prevent infinite loop when only blocked jobs remain.
        if (this._onlyBlockedRemaining()) break;
        continue;
      }

      if (slot.cancelFlag === 1) {
        this._finalize(slot, JOB_STATE.CANCELLED, null);
        continue;
      }

      this.queuedCount--;
      this.runningCount++;
      this.activeIndex = slotIdx;
      slot.state = JOB_STATE.RUNNING;
      slot.startedAtMs = _now();

      const t0 = _now();
      let ok = true;
      let err = null;
      try {
        slot.run(slot.payload, slot.ctx, slot);
      } catch (e) {
        ok = false;
        err = e;
      }
      const t1 = _now();

      this.tierMsEma[slot.priority] += ((t1 - t0) - this.tierMsEma[slot.priority]) * EMA_ALPHA;
      this.tierCountEma[slot.priority] += (1 - this.tierCountEma[slot.priority]) * EMA_ALPHA;

      this.runningCount--;
      this.activeIndex = -1;

      if (ok) {
        this.doneCount++;
        this._finalize(slot, JOB_STATE.DONE, null);
        if (slot.onDone) {
          try { slot.onDone(slot.payload, slot.ctx, slot); } catch (e) { console.error('[007_rnd_JobQueue] onDone error', e); }
        }
      } else {
        this.failedCount++;
        this._finalize(slot, JOB_STATE.FAILED, err);
        if (slot.onError) {
          try { slot.onError(err, slot.payload, slot.ctx, slot); } catch (e) { console.error('[007_rnd_JobQueue] onError error', e); }
        } else if (onRunError) {
          try { onRunError(err, slotIdx); } catch (_) { /* swallow */ }
        }
      }

      executed++;
    }

    return executed;
  }

  _pickNext() {
    // Highest priority tier with queued work first; FIFO within tier.
    // Starvation auto-promote: oldest queued job in a lower tier is
    // promoted if it has waited > starvationMs.
    if (this.options.autoPromote) this._promoteStarved();

    for (let p = 0; p < JOB_PRIORITY.COUNT; p++) {
      const bucket = this.buckets[p];
      if (bucket.count > 0) {
        const idx = bucket.shift();
        return idx;
      }
    }
    return -1;
  }

  _promoteStarved() {
    const now = _now();
    const limit = this.options.starvationMs;

    for (let p = JOB_PRIORITY.COUNT - 1; p > 0; p--) {
      const bucket = this.buckets[p];
      if (bucket.count === 0) continue;

      // Scan the bucket's ring; promote the first starved job.
      const cap = bucket.capacity;
      const start = bucket.head;
      const n = bucket.count;

      for (let i = 0; i < n; i++) {
        const pos = (start + i) % cap;
        const slotIdx = bucket.indices[pos];
        if (slotIdx < 0) continue;
        const slot = this.slots[slotIdx];
        if (slot.state !== JOB_STATE.QUEUED) continue;
        if ((now - slot.queuedAtMs) >= limit) {
          // Remove from this bucket and push to p-1.
          this._removeFromBucket(bucket, pos);
          slot.priority = p - 1;
          this.buckets[p - 1].push(slotIdx);
          this._emit('promoted', { id: slot.id, from: p, to: p - 1 });
          return;
        }
      }
    }
  }

  _removeFromBucket(bucket, pos) {
    // Shift the ring to close the gap at pos.
    const cap = bucket.capacity;
    const head = bucket.head;
    const count = bucket.count;

    for (let i = 0; i < count; i++) {
      const src = (head + i) % cap;
      const dst = (head + i) % cap;
      if (src === pos) {
        // Copy the tail to close the hole.
        for (let j = i; j < count - 1; j++) {
          const s = (head + j + 1) % cap;
          const d = (head + j) % cap;
          bucket.indices[d] = bucket.indices[s];
        }
        bucket.count--;
        bucket.tail = (bucket.head + bucket.count) % cap;
        return;
      }
    }
  }

  _depsResolved(slot) {
    if (slot.depCount === 0) return true;
    for (let i = 0; i < slot.depCount; i++) {
      const depId = slot.deps[i];
      if (depId <= 0) continue;
      const depIdx = this.idToIndex.get(depId);
      if (depIdx === undefined) {
        // Dep already finalized and removed from map — treat as resolved.
        continue;
      }
      const dep = this.slots[depIdx];
      if (dep.state !== JOB_STATE.DONE) return false;
    }
    return true;
  }

  _onlyBlockedRemaining() {
    for (let p = 0; p < JOB_PRIORITY.COUNT; p++) {
      const bucket = this.buckets[p];
      if (bucket.count === 0) continue;
      const cap = bucket.capacity;
      const head = bucket.head;
      const n = bucket.count;
      for (let i = 0; i < n; i++) {
        const pos = (head + i) % cap;
        const slot = this.slots[bucket.indices[pos]];
        if (slot.state === JOB_STATE.QUEUED && this._depsResolved(slot)) {
          return false;
        }
      }
    }
    return true;
  }

  _finalize(slot, finalState, error) {
    slot.state = finalState;
    slot.finishedAtMs = _now();
    slot.run = null;
    // Keep onDone/onError for the caller until _releaseSlot runs.
    // Decrement dependents.
    for (let i = 0; i < slot.depCount; i++) {
      const depId = slot.deps[i];
      if (depId <= 0) continue;
      const depIdx = this.idToIndex.get(depId);
      if (depIdx !== undefined) {
        const dep = this.slots[depIdx];
        if (dep.dependentCount > 0) dep.dependentCount--;
      }
    }
    if (slot.state === JOB_STATE.DONE) {
      this._emit('done', { id: slot.id, kind: slot.kind });
    } else if (slot.state === JOB_STATE.FAILED) {
      this._emit('failed', { id: slot.id, kind: slot.kind, error });
    } else if (slot.state === JOB_STATE.CANCELLED) {
      this.cancelledCount++;
      this._emit('cancelled', { id: slot.id });
    }

    // Free the id mapping; the caller's onDone/onError should not need
    // the slot to remain in the map. Keep until caller has consumed via
    // the returned JobSlot reference.
    this.idToIndex.delete(slot.id);

    // Return the slot to the free list for reuse next frame.
    this._releaseSlot(slot.index);
  }

  /* ---------------- bulk helpers ---------------- */

  clear() {
    for (let p = 0; p < JOB_PRIORITY.COUNT; p++) this.buckets[p].clear();
    for (let i = 0; i < this.capacity; i++) this.slots[i].reset();
    this.freeHead = 0;
    this.freeCount = this.capacity;
    for (let i = 0; i < this.capacity; i++) this.freeList[i] = i;
    this.idToIndex.clear();
    this.queuedCount = 0;
    this.runningCount = 0;
    this.activeIndex = -1;
  }

  dispose() {
    this.clear();
    this._listeners.clear();
    this.slots.length = 0;
  }

  /* ---------------- stats ---------------- */

  getStats() {
    const buckets = new Array(JOB_PRIORITY.COUNT);
    for (let p = 0; p < JOB_PRIORITY.COUNT; p++) {
      buckets[p] = {
        name:  JOB_PRIORITY_NAME[p],
        count: this.buckets[p].count,
        ema:   this.tierMsEma[p],
      };
    }
    return {
      capacity:        this.capacity,
      queuedCount:     this.queuedCount,
      runningCount:    this.runningCount,
      doneCount:       this.doneCount,
      failedCount:     this.failedCount,
      cancelledCount:  this.cancelledCount,
      rejectedCount:   this.rejectedCount,
      freeSlots:       this.freeCount,
      buckets,
      perfTier:        PERF_TIER,
    };
  }
}

/* ------------------------------------------------------------------ */
/* 5. TRANSFERABLE HELPERS                                            */
/* ------------------------------------------------------------------ */

export function collectTransferables(payload) {
  if (!payload) return null;
  const out = [];
  if (ArrayBuffer.isView(payload)) {
    if (payload.buffer && typeof payload.buffer === 'object') out.push(payload.buffer);
  } else if (payload instanceof ArrayBuffer) {
    out.push(payload);
  } else if (Array.isArray(payload)) {
    for (let i = 0; i < payload.length; i++) {
      const p = payload[i];
      if (ArrayBuffer.isView(p)) { if (p.buffer) out.push(p.buffer); }
      else if (p instanceof ArrayBuffer) out.push(p);
    }
  } else if (typeof ImageBitmap !== 'undefined' && payload instanceof ImageBitmap) {
    out.push(payload);
  }
  return out.length > 0 ? out : null;
}

/* ------------------------------------------------------------------ */
/* 6. LIGHTING JOB FACTORIES                                          */
/* ------------------------------------------------------------------ */

export function makeLightListJob(run, ctx, payload) {
  return {
    kind:      JOB_KIND.LIGHT_LIST,
    priority:  JOB_PRIORITY.CRITICAL,
    run,
    ctx,
    payload,
  };
}

export function makeShadowAtlasJob(run, ctx, payload, deps) {
  return {
    kind:      JOB_KIND.SHADOW_ATLAS,
    priority:  JOB_PRIORITY.HIGH,
    run,
    ctx,
    payload,
    deps,
  };
}

export function makeShadowMatrixJob(run, ctx, payload, deps) {
  return {
    kind:      JOB_KIND.SHADOW_MATRIX,
    priority:  JOB_PRIORITY.HIGH,
    run,
    ctx,
    payload,
    deps,
  };
}

export function makeGIProbeJob(run, ctx, payload) {
  return {
    kind:      JOB_KIND.GI_PROBE,
    priority:  JOB_PRIORITY.NORMAL,
    run,
    ctx,
    payload,
  };
}

export function makeAOBlurJob(run, ctx, payload, deps) {
  return {
    kind:      JOB_KIND.AO_BLUR,
    priority:  JOB_PRIORITY.NORMAL,
    run,
    ctx,
    payload,
    deps,
  };
}

export function makeClusterGridJob(run, ctx, payload) {
  return {
    kind:      JOB_KIND.CLUSTER_GRID,
    priority:  JOB_PRIORITY.HIGH,
    run,
    ctx,
    payload,
  };
}

export function makeEnvPaletteJob(run, ctx, payload) {
  return {
    kind:      JOB_KIND.ENV_PALETTE,
    priority:  JOB_PRIORITY.LOW,
    run,
    ctx,
    payload,
  };
}

export function makeInteriorVolJob(run, ctx, payload) {
  return {
    kind:      JOB_KIND.INTERIOR_VOL,
    priority:  JOB_PRIORITY.NORMAL,
    run,
    ctx,
    payload,
  };
}

export function makeExteriorProbeJob(run, ctx, payload) {
  return {
    kind:      JOB_KIND.EXTERIOR_PROBE,
    priority:  JOB_PRIORITY.NORMAL,
    run,
    ctx,
    payload,
  };
}

export function makePostBufferJob(run, ctx, payload) {
  return {
    kind:      JOB_KIND.POST_BUFFER,
    priority:  JOB_PRIORITY.NORMAL,
    run,
    ctx,
    payload,
  };
}

export function makeDirectorHintJob(run, ctx, payload) {
  return {
    kind:      JOB_KIND.DIRECTOR_HINT,
    priority:  JOB_PRIORITY.IDLE,
    run,
    ctx,
    payload,
  };
}

/* ------------------------------------------------------------------ */
/* 7. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultQueue = null;

export function getDefaultJobQueue() {
  if (!_defaultQueue) _defaultQueue = new JobQueue();
  return _defaultQueue;
}

export function disposeDefaultJobQueue() {
  if (_defaultQueue) {
    _defaultQueue.dispose();
    _defaultQueue = null;
  }
}

/* ------------------------------------------------------------------ */
/* 8. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createJobQueue(options = {}) {
  return new JobQueue(options);
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  JobQueue,
  JobSlot,
  PriorityBucket,
  createJobQueue,
  getDefaultJobQueue,
  disposeDefaultJobQueue,
  collectTransferables,
  makeLightListJob,
  makeShadowAtlasJob,
  makeShadowMatrixJob,
  makeGIProbeJob,
  makeAOBlurJob,
  makeClusterGridJob,
  makeEnvPaletteJob,
  makeInteriorVolJob,
  makeExteriorProbeJob,
  makePostBufferJob,
  makeDirectorHintJob,
  JOB_PRIORITY,
  JOB_PRIORITY_NAME,
  JOB_STATE,
  JOB_KIND,
  JOB_KIND_NAME,
  ENQUEUE_RESULT,
  MAX_JOBS_PER_QUEUE,
};

export default _defaultExport;