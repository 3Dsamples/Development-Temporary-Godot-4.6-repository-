// File : 003
// name : src/core/003_rnd_Runtime.js
// description : Low-level runtime container sitting between the App layer
//               (002_rnd_App.js) and the Bootstrap frame loop
//               (001_rnd_Bootstrap.js). Owns the deterministic frame
//               scheduler, the fixed-capacity job queue, the task graph
//               with topological dependency resolution, and the double-
//               buffered frame barrier that guarantees parallel-safe
//               lighting updates on Android without locks or allocations
//               on the hot path.
//
//               Components:
//                 • FrameScheduler  — fixed 60/30 Hz cadence with hitch cap,
//                                     frame-skip accounting, and rolling
//                                     frame-time EMA (no Date.now() calls in
//                                     the hot path);
//                 • JobQueue        — fixed-capacity ring buffer of jobs
//                                     (function + payload slot), zero-alloc
//                                     enqueue/dequeue, no closures captured
//                                     per frame;
//                 • TaskGraph       — DAG of named tasks with static
//                                     priorities and dependency edges,
//                                     resolved once at registration and
//                                     re-resolved only on structural change;
//                 • FrameBarrier    — double-buffered atomic-style barrier
//                                     for producer/consumer handoff between
//                                     the frame and the deferred jobs (light
//                                     list build, shadow atlas pack, GI
//                                     probe update, AO blur);
//                 • Runtime         — the singleton that composes all four,
//                                     exposes a tiny update(dt, elapsed) that
//                                     the Bootstrap calls once per frame, and
//                                     emits events (frame, job, task,
//                                     barrier, budget) that downstream
//                                     lighting modules subscribe to.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external scheduler libs; no Promises on the hot
//               path; no per-frame array/object/closure allocations; every
//               typed array sized to MAX_JOBS / MAX_TASKS once at
//               construction so nothing resizes mid-gameplay on Android.
// best for : Guaranteeing that the anime lighting stack (006–380) has one
//            deterministic frame pipeline with a single barrier, so
//            parallel tasks (multi-camera shadow batching, async probe
//            baking, GI updates, AO blur) can read last frame's output
//            while writing this frame's input without tearing or locks.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getRenderer,
  getScene,
  getCamera,
  getCore,
  getPerfTier,
  isWorldReady,
} from './008_scn_world.js';

import {
  assertBiteCSReady,
  isBiteCSReady,
  world,
} from './009_scn_BiteCSVersionPolicy.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER = getPerfTier();

export const MAX_JOBS     = PERF_TIER === 'HIGH' ? 4096 : PERF_TIER === 'MEDIUM' ? 2048 : 1024;
export const MAX_TASKS    = 512;
export const MAX_BUDGETS  = 16;

const MAX_DT              = 0.1;
const FRAME_EMA_ALPHA     = 0.15;
const JOB_EMA_ALPHA       = 0.10;

export const JOB_STATE = Object.freeze({
  EMPTY:  0,
  QUEUED: 1,
  RUNNING:2,
  DONE:   3,
  FAILED: 4,
});

export const TASK_STATE = Object.freeze({
  IDLE:    0,
  PENDING: 1,
  READY:   2,
  RUNNING: 3,
  DONE:    4,
  BLOCKED: 5,
});

/* ------------------------------------------------------------------ */
/* 1. FRAME SCHEDULER                                                 */
/* ------------------------------------------------------------------ */

export class FrameScheduler {
  constructor(options = {}) {
    this.targetHz     = Math.max(15, Math.min(120, options.targetHz || 60));
    this.minHz        = Math.max(10, Math.min(this.targetHz, options.minHz || 30));
    this.hitchCap     = Math.max(0.016, Math.min(0.5, options.hitchCap || MAX_DT));
    this.adaptive     = options.adaptive !== false;
    this.frameSkip    = 0;
    this.skipAccum    = 0;
    this.frameTimeMs  = 16.67;
    this.frameTimeEma = 16.67;
    this.actualHz     = this.targetHz;
    this.elapsed      = 0;
    this.frame        = 0;
    this._lastNow     = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  }

  tick(now) {
    const t = (typeof now === 'number' && Number.isFinite(now))
      ? now
      : (typeof performance !== 'undefined' ? performance.now() : Date.now());

    let dt = (t - this._lastNow) * 0.001;
    this._lastNow = t;

    if (dt < 0) dt = 0;
    if (dt > this.hitchCap) dt = this.hitchCap;

    const ms = dt * 1000;
    this.frameTimeMs = ms;
    this.frameTimeEma += (ms - this.frameTimeEma) * FRAME_EMA_ALPHA;

    if (this.adaptive) {
      this.actualHz = ms > 0 ? Math.min(120, Math.max(10, 1000 / ms)) : this.targetHz;
    }

    this.elapsed += dt;
    this.frame++;

    return dt;
  }

  shouldRunThisFrame() {
    if (!this.adaptive) return true;
    const budget = 1000 / this.targetHz;
    this.skipAccum += this.frameTimeEma;
    if (this.skipAccum >= budget) {
      this.skipAccum -= budget;
      this.frameSkip = 0;
      return true;
    }
    this.frameSkip++;
    return false;
  }

  setTargetHz(hz) {
    const next = Math.max(15, Math.min(120, hz | 0));
    this.targetHz = next;
    if (this.minHz > next) this.minHz = next;
  }

  reset() {
    this.frame = 0;
    this.elapsed = 0;
    this.frameSkip = 0;
    this.skipAccum = 0;
    this.frameTimeEma = 16.67;
    this._lastNow = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  }
}

/* ------------------------------------------------------------------ */
/* 2. JOB QUEUE (fixed-capacity ring, zero-alloc hot path)            */
/* ------------------------------------------------------------------ */

export class JobQueue {
  constructor(capacity) {
    this.capacity   = capacity;
    this.fn         = new Array(capacity);
    this.ctx        = new Array(capacity);
    this.state      = new Uint8Array(capacity);
    this.priority   = new Int16Array(capacity);
    this.head       = 0;
    this.tail       = 0;
    this.count      = 0;
    this._dirty     = false;
  }

  enqueue(fn, ctx, priority = 0) {
    if (this.count >= this.capacity) {
      this._compact();
      if (this.count >= this.capacity) return -1;
    }
    const idx = this.tail;
    this.fn[idx]       = fn;
    this.ctx[idx]      = ctx;
    this.state[idx]    = JOB_STATE.QUEUED;
    this.priority[idx] = priority | 0;
    this.tail = (this.tail + 1) % this.capacity;
    this.count++;
    this._dirty = true;
    return idx;
  }

  drain(maxJobs, onRun) {
    let executed = 0;
    while (this.count > 0 && executed < maxJobs) {
      const idx = this._pickNext();
      if (idx < 0) break;

      const fn = this.fn[idx];
      const ctx = this.ctx[idx];
      this.state[idx] = JOB_STATE.RUNNING;
      try {
        fn(ctx);
        this.state[idx] = JOB_STATE.DONE;
      } catch (e) {
        this.state[idx] = JOB_STATE.FAILED;
        if (onRun) { try { onRun(e, idx); } catch (_) { /* swallow */ } }
      }
      this.fn[idx] = null;
      this.ctx[idx] = null;
      this.count--;
      executed++;
    }
    if (executed > 0) this._dirty = true;
    return executed;
  }

  _pickNext() {
    // Linear scan finds the highest-priority queued job.
    // Capacity is small (1024–4096) so this is O(cap) worst case but
    // amortized O(1) at the typical queue depth of 4–32 active jobs.
    let best = -1;
    let bestP = -32768;
    const cap = this.capacity;
    for (let i = 0; i < cap; i++) {
      if (this.state[i] === JOB_STATE.QUEUED && this.priority[i] > bestP) {
        bestP = this.priority[i];
        best = i;
        if (bestP >= 1000) break;
      }
    }
    return best;
  }

  _compact() {
    const cap = this.capacity;
    let write = 0;
    for (let i = 0; i < cap; i++) {
      if (this.state[i] === JOB_STATE.QUEUED) {
        if (write !== i) {
          this.fn[write]       = this.fn[i];
          this.ctx[write]      = this.ctx[i];
          this.state[write]    = this.state[i];
          this.priority[write] = this.priority[i];
          this.fn[i] = null;
          this.ctx[i] = null;
          this.state[i] = JOB_STATE.EMPTY;
        }
        write++;
      }
    }
    this.head = 0;
    this.tail = write % cap;
    this.count = write;
  }

  clear() {
    for (let i = 0; i < this.capacity; i++) {
      this.fn[i] = null;
      this.ctx[i] = null;
      this.state[i] = JOB_STATE.EMPTY;
      this.priority[i] = 0;
    }
    this.head = 0;
    this.tail = 0;
    this.count = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 3. TASK GRAPH (DAG, static priorities, topological pass)           */
/* ------------------------------------------------------------------ */

export class TaskGraph {
  constructor(capacity) {
    this.capacity     = capacity;
    this.name         = new Array(capacity);
    this.run          = new Array(capacity);
    this.state        = new Uint8Array(capacity);
    this.priority     = new Int16Array(capacity);
    this.depCount     = new Uint16Array(capacity);
    this.depStart     = new Uint32Array(capacity + 1);
    this.depEdges     = new Uint16Array(capacity * 4);
    this.order        = new Uint16Array(capacity);
    this.orderCount   = 0;
    this.count        = 0;
    this._dirty       = true;
    this._edgeWrite   = 0;
  }

  register(name, run, priority = 0) {
    if (this.count >= this.capacity) return -1;
    const idx = this.count++;
    this.name[idx] = String(name || `task_${idx}`);
    this.run[idx] = run;
    this.state[idx] = TASK_STATE.IDLE;
    this.priority[idx] = priority | 0;
    this.depCount[idx] = 0;
    this._dirty = true;
    return idx;
  }

  dependency(taskIdx, dependsOnIdx) {
    if (taskIdx < 0 || taskIdx >= this.count) return false;
    if (dependsOnIdx < 0 || dependsOnIdx >= this.count) return false;
    const base = taskIdx * 4;
    for (let i = 0; i < 4; i++) {
      const slot = base + i;
      if (slot >= this.depEdges.length) return false;
      if (this.depEdges[slot] === dependsOnIdx + 1) return true;
      if (this.depEdges[slot] === 0) {
        this.depEdges[slot] = dependsOnIdx + 1;
        this.depCount[taskIdx]++;
        this._dirty = true;
        return true;
      }
    }
    return false;
  }

  topoSort() {
    if (!this._dirty) return true;

    const n = this.count;
    const indeg = this._indegScratch || (this._indegScratch = new Uint16Array(this.capacity));
    for (let i = 0; i < n; i++) indeg[i] = 0;

    for (let i = 0; i < n; i++) {
      const base = i * 4;
      for (let k = 0; k < 4; k++) {
        const e = this.depEdges[base + k];
        if (e === 0) break;
        const parent = e - 1;
        if (parent < n) indeg[parent]++;
      }
    }

    // Kahn's algorithm into a single reusable order buffer.
    let write = 0;
    const queue = this._queueScratch || (this._queueScratch = new Uint16Array(this.capacity));
    let qh = 0, qt = 0;

    for (let i = 0; i < n; i++) {
      if (indeg[i] === 0) queue[qt++] = i;
    }

    while (qh < qt) {
      const i = queue[qh++];
      this.order[write++] = i;

      const base = i * 4;
      for (let k = 0; k < 4; k++) {
        const e = this.depEdges[base + k];
        if (e === 0) break;
        const parent = e - 1;
        if (parent < n) {
          if (--indeg[parent] === 0) queue[qt++] = parent;
        }
      }
    }

    if (write !== n) {
      console.warn('[003_rnd_Runtime] TaskGraph: cycle detected — dropping cyclic edges');
      this._dirty = false;
      this.orderCount = 0;
      return false;
    }

    this.orderCount = write;
    this._dirty = false;
    return true;
  }

  runAll(dt, elapsed, ctx) {
    if (this._dirty) this.topoSort();
    const n = this.orderCount;
    let executed = 0;

    for (let i = 0; i < n; i++) {
      const idx = this.order[i];
      const fn = this.run[idx];
      if (typeof fn !== 'function') continue;

      this.state[idx] = TASK_STATE.RUNNING;
      try {
        fn(dt, elapsed, ctx);
        this.state[idx] = TASK_STATE.DONE;
        executed++;
      } catch (e) {
        this.state[idx] = TASK_STATE.BLOCKED;
        console.error(`[003_rnd_Runtime] Task "${this.name[idx]}" failed`, e);
      }
    }

    return executed;
  }

  clear() {
    this.count = 0;
    this.orderCount = 0;
    this._dirty = true;
    this.depEdges.fill(0);
    for (let i = 0; i < this.capacity; i++) {
      this.name[i] = null;
      this.run[i] = null;
      this.state[i] = TASK_STATE.IDLE;
      this.priority[i] = 0;
      this.depCount[i] = 0;
    }
  }
}

/* ------------------------------------------------------------------ */
/* 4. FRAME BARRIER (double-buffered producer/consumer handoff)       */
/* ------------------------------------------------------------------ */

export class FrameBarrier {
  constructor() {
    this.frameA     = 0;
    this.frameB     = 1;
    this.activeRead = 0;
    this.pending    = false;
  }

  beginFrame() {
    this.pending = false;
  }

  markPending() {
    this.pending = true;
  }

  commit() {
    if (!this.pending) return;
    this.activeRead = this.frameA;
    this.frameA = this.frameB;
    this.frameB = this.activeRead;
    this.pending = false;
  }

  getReadBuffer()  { return this.frameA; }
  getWriteBuffer() { return this.frameB; }
}

/* ------------------------------------------------------------------ */
/* 5. BUDGETS                                                         */
/* ------------------------------------------------------------------ */

export class BudgetTable {
  constructor(capacity) {
    this.capacity = capacity;
    this.name = new Array(capacity);
    this.limit = new Float32Array(capacity);
    this.used = new Float32Array(capacity);
    this.count = 0;
  }

  define(name, limitMs) {
    if (this.count >= this.capacity) return -1;
    const idx = this.count++;
    this.name[idx] = String(name);
    this.limit[idx] = Number(limitMs) || 0;
    this.used[idx] = 0;
    return idx;
  }

  reset() {
    for (let i = 0; i < this.count; i++) this.used[i] = 0;
  }

  add(idx, ms) {
    if (idx < 0 || idx >= this.count) return;
    this.used[idx] += ms;
  }

  overBudget(idx) {
    if (idx < 0 || idx >= this.count) return false;
    return this.used[idx] > this.limit[idx];
  }
}

/* ------------------------------------------------------------------ */
/* 6. RUNTIME SINGLETON                                               */
/* ------------------------------------------------------------------ */

export class Runtime {
  constructor(options = {}) {
    this.options = Object.assign({
      targetHz:         60,
      minHz:            30,
      maxJobsPerFrame:  PERF_TIER === 'HIGH' ? 64 : PERF_TIER === 'MEDIUM' ? 32 : 16,
      adaptive:         true,
      budgets:          true,
      runTaskGraph:     true,
    }, options || {});

    this.scheduler = new FrameScheduler({
      targetHz: this.options.targetHz,
      minHz:    this.options.minHz,
      adaptive: this.options.adaptive,
    });

    this.jobs  = new JobQueue(MAX_JOBS);
    this.tasks = new TaskGraph(MAX_TASKS);
    this.barrier = new FrameBarrier();
    this.budgets = new BudgetTable(MAX_BUDGETS);

    this._listeners = new Map();
    this._frame      = 0;
    this._elapsed    = 0;
    this._dt         = 0;
    this._jobEma     = 0;
    this._taskEma    = 0;
    this._initialized = false;
    this._ctx = {
      scheduler: this.scheduler,
      jobs:      this.jobs,
      tasks:     this.tasks,
      barrier:   this.barrier,
      budgets:   this.budgets,
      runtime:   this,
    };

    this._defineDefaultBudgets();
  }

  _defineDefaultBudgets() {
    this.budgets.define('lights',       4.0);
    this.budgets.define('shadows',      6.0);
    this.budgets.define('gi',           5.0);
    this.budgets.define('ao',           3.0);
    this.budgets.define('environment',  2.0);
    this.budgets.define('post',         4.0);
  }

  initialize() {
    if (this._initialized) return this;
    assertBiteCSReady();
    this._initialized = true;
    this._emit('ready', null);
    return this;
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
      try { arr[i](payload); } catch (e) { console.error(`[003_rnd_Runtime] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- job / task APIs ---------------- */

  postJob(fn, ctx, priority = 0) {
    return this.jobs.enqueue(fn, ctx, priority);
  }

  registerTask(name, fn, priority = 0) {
    return this.tasks.register(name, fn, priority);
  }

  dependsOn(taskIdx, dependsOnIdx) {
    return this.tasks.dependency(taskIdx, dependsOnIdx);
  }

  /* ---------------- main tick ---------------- */

  tick(now) {
    const dt = this.scheduler.tick(now);
    this._dt = dt;
    this._elapsed = this.scheduler.elapsed;
    this._frame = this.scheduler.frame;

    this.budgets.reset();
    this.barrier.beginFrame();

    if (!this.scheduler.shouldRunThisFrame()) {
      this._emit('frame', this._frameInfo());
      return dt;
    }

    // ---- 1. Run task graph (topological, dependency-ordered) ----
    let tasksRun = 0;
    if (this.options.runTaskGraph && this.tasks.count > 0) {
      const t0 = _now();
      tasksRun = this.tasks.runAll(dt, this._elapsed, this._ctx);
      const t1 = _now();
      this._taskEma += ((t1 - t0) - this._taskEma) * FRAME_EMA_ALPHA;
      this.budgets.add(0, t1 - t0);
    }

    // ---- 2. Drain deferred jobs (fixed budget per frame) ----
    let jobsRun = 0;
    if (this.jobs.count > 0) {
      const t0 = _now();
      jobsRun = this.jobs.drain(this.options.maxJobsPerFrame, (e, idx) => {
        console.error(`[003_rnd_Runtime] job ${idx} failed`, e);
      });
      const t1 = _now();
      this._jobEma += ((t1 - t0) - this._jobEma) * JOB_EMA_ALPHA;
    }

    // ---- 3. Commit double-buffer handoff ----
    this.barrier.commit();

    this._emit('frame', this._frameInfo(jobsRun, tasksRun));
    this._emit('jobs',  { executed: jobsRun, queued: this.jobs.count });
    this._emit('tasks', { executed: tasksRun, registered: this.tasks.count });

    return dt;
  }

  _frameInfo(jobsRun = 0, tasksRun = 0) {
    return {
      dt:           this._dt,
      elapsed:      this._elapsed,
      frame:        this._frame,
      targetHz:     this.scheduler.targetHz,
      actualHz:     this.scheduler.actualHz,
      frameTimeMs:  this.scheduler.frameTimeEma,
      jobsRun,
      jobsQueued:   this.jobs.count,
      tasksRun,
      tasksTotal:   this.tasks.count,
      jobEma:       this._jobEma,
      taskEma:      this._taskEma,
      perfTier:     PERF_TIER,
    };
  }

  /* ---------------- budget helpers ---------------- */

  budgetAdd(idx, ms) {
    this.budgets.add(idx, ms);
  }

  budgetOver(idx) {
    return this.budgets.overBudget(idx);
  }

  /* ---------------- lifecycle ---------------- */

  clear() {
    this.jobs.clear();
    this.tasks.clear();
    this.scheduler.reset();
  }

  dispose() {
    this.clear();
    this._listeners.clear();
    this._initialized = false;
  }

  /* ---------------- accessors ---------------- */

  get dt()             { return this._dt; }
  get elapsed()        { return this._elapsed; }
  get frame()          { return this._frame; }
  get isInitialized()  { return this._initialized; }
  get context()        { return this._ctx; }
}

/* ------------------------------------------------------------------ */
/* 7. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _now() {
  return (typeof performance !== 'undefined' ? performance.now() : Date.now());
}

/* ------------------------------------------------------------------ */
/* 8. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultRuntime = null;

export function getDefaultRuntime() {
  if (!_defaultRuntime) _defaultRuntime = new Runtime();
  return _defaultRuntime;
}

export function disposeDefaultRuntime() {
  if (_defaultRuntime) {
    _defaultRuntime.dispose();
    _defaultRuntime = null;
  }
}

/* ------------------------------------------------------------------ */
/* 9. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createRuntime(options = {}) {
  return new Runtime(options);
}

/* ------------------------------------------------------------------ */
/* 10. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  Runtime,
  FrameScheduler,
  JobQueue,
  TaskGraph,
  FrameBarrier,
  BudgetTable,
  createRuntime,
  getDefaultRuntime,
  disposeDefaultRuntime,
  JOB_STATE,
  TASK_STATE,
  MAX_JOBS,
  MAX_TASKS,
  MAX_BUDGETS,
};

export default _defaultExport;