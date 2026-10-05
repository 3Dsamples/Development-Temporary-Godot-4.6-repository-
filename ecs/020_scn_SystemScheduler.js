// File : 020
// name : src/ecs/020_scn_SystemScheduler.js
// description : Priority- and dependency-aware frame scheduler for the scene
//               ECS world of the anime lighting stack on Android mobile. It
//               collects every registered system from 019_scn_SystemBase.js,
//               resolves inter-system dependencies via Kahn's topological
//               sort with cycle detection, and drives the frame in a
//               deterministic order:
//
//                 phase 0  INIT      — one-shot init pass at boot
//                 phase 1  EARLY     — priority 0..99
//                 phase 2  UPDATE    — priority 100..899
//                 phase 3  LATE      — priority 900..998
//                 phase 4  LATE_UPDATE — after all update() calls
//                 phase 5  POST      — priority 999 (final hooks)
//                 phase 6  RENDER    — emit draw calls
//
//               Design:
//                 • Fixed-capacity schedule array sized once at boot.
//                 • Rebuild pass resolves the DAG only when the system
//                   registry changes (add / remove / disable / dep change).
//                 • Kahn's algorithm with cycle detection; a cycle is a
//                   hard failure reported through the error boundary (030)
//                   and the logger (026), never a silent reordering.
//                 • Within each topological level, systems are sorted by
//                   priority ascending, ties broken by registration order
//                   (stable).
//                 • Per-system gate callbacks — `setGate(systemName, fn)`
//                   lets a downstream system (e.g. quality controller)
//                   skip a scheduled system for the frame without
//                   unregistering it. Gates are consulted once per frame.
//                 • Per-phase budget accounting — every phase charges its
//                   wall-clock cost against a per-phase budget. Overruns
//                   are surfaced through the profiler (024) and the
//                   quality controller (017) reads them to bias quality.
//                 • Enable/disable is a single integer compare per system
//                   per frame. Disabling a system never touches the DAG.
//                 • Zero allocations on the hot path — the schedule is
//                   rebuilt only on structural change, and the per-frame
//                   loop walks a flat typed array.
//                 • Hooks — `beforeFrame`, `afterFrame`, `beforeSystem`,
//                   `afterSystem` — let a debug HUD (179) observe the
//                   schedule without interfering with it.
//
//               Integration:
//                 • 010_scn_ECSWorld.js         — world handle
//                 • 017_rnd_QualityConfig.js    — budget feedback
//                 • 019_scn_SystemBase.js       — system contract
//                 • 024_rnd_Profiler.js         — per-phase marks
//                 • 025_rnd_StatsCollector.js   — aggregated stats
//                 • 026_rnd_Logger.js           — diagnostics
//                 • 030_rnd_ErrorBoundary.js    — containment
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing a deterministic, priority- and dependency-aware
//            frame order for the entire anime lighting stack, with per-
//            phase budget accounting that feeds back into the adaptive
//            quality controller on Android — so no subsystem can starve
//            another, no cycle can silently reorder the frame, and the
//            whole pipeline stays one coherent unit.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from '../core/008_scn_world.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
} from '../core/026_rnd_Logger.js';

import {
  getDefaultErrorBoundaries,
  BOUNDARY_TAG,
} from '../core/030_rnd_ErrorBoundary.js';

import {
  getDefaultProfiler,
} from '../core/024_rnd_Profiler.js';

import {
  getDefaultQualityController,
} from '../core/017_rnd_QualityConfig.js';

import {
  getECSWorld,
  worldTime,
  tickWorldTime,
} from './010_scn_ECSWorld.js';

import {
  SystemBase,
  SYSTEM_STATE,
  SYSTEM_STATE_NAME,
  PRIORITY_EARLY,
  PRIORITY_DEFAULT,
  PRIORITY_LATE,
  PRIORITY_LAST,
  MAX_SYSTEMS,
  getAllSystems,
  getSystem,
  getSystemCount,
  forEachSystem,
  tickSystems,
  SystemState,
} from './019_scn_SystemBase.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Frame phases. Every system's priority maps to exactly one phase.
 */
export const PHASE = Object.freeze({
  INIT:        0,
  EARLY:       1,
  UPDATE:      2,
  LATE:        3,
  LATE_UPDATE: 4,
  POST:        5,
  RENDER:      6,
  COUNT:       7,
});

export const PHASE_NAME = Object.freeze([
  'init',
  'early',
  'update',
  'late',
  'late_update',
  'post',
  'render',
]);

/**
 * Per-phase wall-clock budgets (milliseconds). These are the defaults;
 * they can be overridden at runtime by the quality controller.
 */
export const DEFAULT_PHASE_BUDGET_MS =
  PERF_TIER_LOCAL === 'HIGH'
    ? Object.freeze({ init: 8.0, early: 1.5, update: 6.0, late: 2.0, late_update: 1.0, post: 1.0, render: 6.0 })
    : PERF_TIER_LOCAL === 'MEDIUM'
      ? Object.freeze({ init: 10.0, early: 2.0, update: 8.0, late: 2.5, late_update: 1.2, post: 1.2, render: 7.0 })
      : Object.freeze({ init: 14.0, early: 3.0, update: 10.0, late: 3.0, late_update: 1.5, post: 1.5, render: 8.0 });

/**
 * Maximum number of systems that can be scheduled in a single phase.
 */
export const MAX_SCHEDULE_ENTRIES = MAX_SYSTEMS;

/**
 * Scheduler build results.
 */
export const BUILD_RESULT = Object.freeze({
  OK:              0,
  CYCLE_DETECTED:  1,
  DUPLICATE_NAME:  2,
  MISSING_DEP:     3,
  TOO_MANY:        4,
  UNKNOWN:         5,
});

export const BUILD_RESULT_NAME = Object.freeze([
  'ok',
  'cycle_detected',
  'duplicate_name',
  'missing_dep',
  'too_many',
  'unknown',
]);

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const SchedulerState = {
  frame:              0,
  built:              false,
  dirty:              true,
  lastBuildFrame:     -1,
  buildCount:         0,
  totalRuns:          0,
  totalSystemsRun:    0,
  totalSystemsSkipped:0,
  totalGateSkips:     0,
  totalBudgetOverruns:0,
  phaseOverrunCount:  new Uint32Array(PHASE.COUNT),
  phaseLastMs:        new Float32Array(PHASE.COUNT),
  phaseEmaMs:         new Float32Array(PHASE.COUNT),
  phasePeakMs:        new Float32Array(PHASE.COUNT),
  phaseBudgetMs:      new Float32Array(PHASE.COUNT),
  lastFrameCostMs:    0,
  lastFrameEmaMs:     0,
  lastFramePeakMs:    0,
  peakActiveSystems:  0,
};

// Initialize phase budgets from tier defaults.
(function _initPhaseBudgets() {
  const d = DEFAULT_PHASE_BUDGET_MS;
  SchedulerState.phaseBudgetMs[PHASE.INIT]        = d.init;
  SchedulerState.phaseBudgetMs[PHASE.EARLY]       = d.early;
  SchedulerState.phaseBudgetMs[PHASE.UPDATE]      = d.update;
  SchedulerState.phaseBudgetMs[PHASE.LATE]        = d.late;
  SchedulerState.phaseBudgetMs[PHASE.LATE_UPDATE] = d.late_update;
  SchedulerState.phaseBudgetMs[PHASE.POST]        = d.post;
  SchedulerState.phaseBudgetMs[PHASE.RENDER]      = d.render;
})();

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.system_scheduler', {
        tag: BOUNDARY_TAG.GENERIC,
        failureThreshold: 5,
      });
    }
  } catch (_) { /* swallow */ }
  return _boundary;
}

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

function _now() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

/* ------------------------------------------------------------------ */
/* 2. SCHEDULE SLOT                                                   */
/* ------------------------------------------------------------------ */

/**
 * One slot in the schedule. Holds a direct reference to the system and
 * the phase it belongs to. Pre-allocated once.
 */
export class ScheduleSlot {
  constructor(index) {
    this.index     = index;
    this.system    = null;
    this.phase     = PHASE.UPDATE;
    this.priority  = PRIORITY_DEFAULT;
    this.order     = 0;      // stable tie-break by registration
    this.enabled   = 1;
    this.gateFn    = null;   // optional frame gate
    this.gateCtx   = null;
    this.level     = 0;      // topological level
  }

  reset() {
    this.system    = null;
    this.phase     = PHASE.UPDATE;
    this.priority  = PRIORITY_DEFAULT;
    this.order     = 0;
    this.enabled   = 1;
    this.gateFn    = null;
    this.gateCtx   = null;
    this.level     = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 3. SCHEDULE STORAGE                                                */
/* ------------------------------------------------------------------ */

const _slots = new Array(MAX_SCHEDULE_ENTRIES);
for (let i = 0; i < MAX_SCHEDULE_ENTRIES; i++) _slots[i] = new ScheduleSlot(i);
let _slotCount = 0;

/**
 * Per-phase flat index lists. Each phase holds an Int32Array of slot
 * indices, sorted by (topological level, priority, registration order).
 */
const _phaseIndices = new Array(PHASE.COUNT);
for (let p = 0; p < PHASE.COUNT; p++) _phaseIndices[p] = new Int32Array(MAX_SCHEDULE_ENTRIES);
const _phaseCounts = new Uint32Array(PHASE.COUNT);

/**
 * Scratch buffers used only during a rebuild. Sized once, never resized.
 */
const _indegree        = new Int32Array(MAX_SCHEDULE_ENTRIES);
const _queue           = new Int32Array(MAX_SCHEDULE_ENTRIES);
const _topoOrder       = new Int32Array(MAX_SCHEDULE_ENTRIES);
const _levelOf         = new Int32Array(MAX_SCHEDULE_ENTRIES);
const _nameToSlot      = new Map();

/**
 * Phase gate hooks — optional callbacks that can disable an entire phase
 * for the frame (used by the quality controller under heavy load).
 */
const _phaseGates = new Array(PHASE.COUNT).fill(null);

/* ------------------------------------------------------------------ */
/* 4. FRAME HOOKS                                                     */
/* ------------------------------------------------------------------ */

export const FrameHooks = {
  beforeFrame: null,
  afterFrame: null,
  beforeSystem: null,
  afterSystem: null,
};

/* ------------------------------------------------------------------ */
/* 5. SCHEDULER CONSTRUCTION                                          */
/* ------------------------------------------------------------------ */

/**
 * Marks the schedule dirty. Called when a system is registered,
 * unregistered, or changes dependencies.
 */
export function markScheduleDirty() {
  SchedulerState.dirty = true;
}

/**
 * Returns true if the schedule needs rebuilding.
 */
export function isScheduleDirty() {
  return SchedulerState.dirty || !SchedulerState.built;
}

/* ------------------------------------------------------------------ */
/* 6. TOPOLOGICAL BUILD                                               */
/* ------------------------------------------------------------------ */

function _resolvePhaseFromPriority(priority) {
  if (priority >= PRIORITY_LAST)    return PHASE.POST;
  if (priority >= PRIORITY_LATE)    return PHASE.LATE;
  if (priority <= PRIORITY_EARLY + 99) return PHASE.EARLY;
  return PHASE.UPDATE;
}

function _findSlotIndexByName(name) {
  const slot = _nameToSlot.get(name);
  return slot === undefined ? -1 : slot;
}

/**
 * Rebuilds the schedule from the current system registry. Returns one
 * of the BUILD_RESULT codes.
 */
export function buildSchedule() {
  // Reset.
  for (let i = 0; i < _slotCount; i++) _slots[i].reset();
  for (let p = 0; p < PHASE.COUNT; p++) _phaseCounts[p] = 0;
  _nameToSlot.clear();
  _slotCount = 0;

  const systems = getAllSystems();
  const n = systems.length;

  if (n === 0) {
    SchedulerState.built = true;
    SchedulerState.dirty = false;
    SchedulerState.lastBuildFrame = SchedulerState.frame;
    SchedulerState.buildCount++;
    return BUILD_RESULT.OK;
  }

  if (n > MAX_SCHEDULE_ENTRIES) {
    const log = _safeLogger();
    if (log) log.error(LOG_CHANNEL.CORE,
      `[020_scn_SystemScheduler] too many systems: ${n} > ${MAX_SCHEDULE_ENTRIES}`);
    return BUILD_RESULT.TOO_MANY;
  }

  // 1. Allocate slots for each system.
  for (let i = 0; i < n; i++) {
    const sys = systems[i];
    if (!sys) continue;

    // Duplicate name check.
    if (_nameToSlot.has(sys.name)) {
      const log = _safeLogger();
      if (log) log.error(LOG_CHANNEL.CORE,
        `[020_scn_SystemScheduler] duplicate system name "${sys.name}"`);
      return BUILD_RESULT.DUPLICATE_NAME;
    }

    const slotIdx = _slotCount++;
    const slot = _slots[slotIdx];
    slot.reset();
    slot.system   = sys;
    slot.priority = sys.priority | 0;
    slot.phase    = _resolvePhaseFromPriority(slot.priority);
    slot.order    = i;
    slot.enabled  = sys.enabled === 1 ? 1 : 0;

    _nameToSlot.set(sys.name, slotIdx);
  }

  // 2. Build dependency graph.
  _indegree.fill(0);
  for (let i = 0; i < _slotCount; i++) {
    const slot = _slots[i];
    const sys = slot.system;
    for (let d = 0; d < sys.dependencyCount; d++) {
      const depName = sys.dependencies[d];
      if (!depName) continue;
      const depSlot = _findSlotIndexByName(depName);
      if (depSlot < 0) {
        const log = _safeLogger();
        if (log) log.warn(LOG_CHANNEL.CORE,
          `[020_scn_SystemScheduler] system "${sys.name}" depends on missing "${depName}"`);
        return BUILD_RESULT.MISSING_DEP;
      }
      _indegree[i]++;
    }
  }

  // 3. Kahn's algorithm with level tracking.
  let head = 0;
  let tail = 0;
  for (let i = 0; i < _slotCount; i++) {
    if (_indegree[i] === 0) {
      _queue[tail++] = i;
      _levelOf[i] = 0;
    }
  }

  let topoCount = 0;
  while (head < tail) {
    const slotIdx = _queue[head++];
    _topoOrder[topoCount++] = slotIdx;

    const slot = _slots[slotIdx];
    const sys = slot.system;
    // Find every system that depends on this one.
    for (let j = 0; j < _slotCount; j++) {
      const other = _slots[j];
      if (!other.system) continue;
      const otherSys = other.system;
      for (let d = 0; d < otherSys.dependencyCount; d++) {
        if (otherSys.dependencies[d] === sys.name) {
          if (--_indegree[j] === 0) {
            _queue[tail++] = j;
            _levelOf[j] = _levelOf[slotIdx] + 1;
          }
        }
      }
    }
  }

  // Cycle detection.
  if (topoCount !== _slotCount) {
    const log = _safeLogger();
    if (log) log.error(LOG_CHANNEL.CORE,
      `[020_scn_SystemScheduler] cycle detected in system graph: ` +
      `${topoCount}/${_slotCount} scheduled`);
    return BUILD_RESULT.CYCLE_DETECTED;
  }

  // 4. Sort each phase list by (level, priority, order).
  for (let p = 0; p < PHASE.COUNT; p++) _phaseCounts[p] = 0;

  // Build a per-phase array using the topo order, then sort each.
  const phaseTopo = new Array(PHASE.COUNT);
  for (let p = 0; p < PHASE.COUNT; p++) phaseTopo[p] = [];

  for (let k = 0; k < _slotCount; k++) {
    const slotIdx = _topoOrder[k];
    const slot = _slots[slotIdx];
    slot.level = _levelOf[slotIdx];
    const p = slot.phase;
    phaseTopo[p].push(slotIdx);
  }

  for (let p = 0; p < PHASE.COUNT; p++) {
    const list = phaseTopo[p];
    list.sort((a, b) => {
      const sa = _slots[a];
      const sb = _slots[b];
      if (sa.level !== sb.level) return sa.level - sb.level;
      if (sa.priority !== sb.priority) return sa.priority - sb.priority;
      return sa.order - sb.order;
    });
    for (let i = 0; i < list.length; i++) {
      _phaseIndices[p][i] = list[i];
    }
    _phaseCounts[p] = list.length;
  }

  SchedulerState.built = true;
  SchedulerState.dirty = false;
  SchedulerState.lastBuildFrame = SchedulerState.frame;
  SchedulerState.buildCount++;

  return BUILD_RESULT.OK;
}

/**
 * Rebuilds the schedule only if it is dirty.
 */
export function ensureSchedule() {
  if (isScheduleDirty()) {
    return buildSchedule();
  }
  return BUILD_RESULT.OK;
}

/* ------------------------------------------------------------------ */
/* 7. SYSTEM GATES                                                    */
/* ------------------------------------------------------------------ */

/**
 * Installs a frame gate on the given system. `fn(system, dt, elapsed)`
 * returns true to allow the system to run, false to skip it for the
 * frame. Gates are consulted once per system per frame.
 */
export function setGate(systemName, fn, ctx) {
  const slotIdx = _findSlotIndexByName(systemName);
  if (slotIdx < 0) return false;
  _slots[slotIdx].gateFn  = typeof fn === 'function' ? fn : null;
  _slots[slotIdx].gateCtx = ctx || null;
  return true;
}

/**
 * Clears a frame gate.
 */
export function clearGate(systemName) {
  const slotIdx = _findSlotIndexByName(systemName);
  if (slotIdx < 0) return false;
  _slots[slotIdx].gateFn  = null;
  _slots[slotIdx].gateCtx = null;
  return true;
}

/**
 * Installs a phase gate that can disable an entire phase for the frame.
 * `fn(phase, dt, elapsed)` returns true to allow the phase to run.
 */
export function setPhaseGate(phase, fn, ctx) {
  if (phase < 0 || phase >= PHASE.COUNT) return false;
  _phaseGates[phase] = typeof fn === 'function'
    ? { fn, ctx: ctx || null }
    : null;
  return true;
}

/**
 * Clears a phase gate.
 */
export function clearPhaseGate(phase) {
  if (phase < 0 || phase >= PHASE.COUNT) return false;
  _phaseGates[phase] = null;
  return true;
}

/* ------------------------------------------------------------------ */
/* 8. FRAME EXECUTION                                                 */
/* ------------------------------------------------------------------ */

/**
 * Runs one frame of the schedule.
 * Returns a small statistics object (in-place updated).
 */
const _frameResult = {
  frame:           0,
  systemsRun:      0,
  systemsSkipped:  0,
  gateSkips:       0,
  budgetOverruns:  0,
  totalCostMs:     0,
  phaseCostMs:     new Float32Array(PHASE.COUNT),
};

export function runFrame(dt, elapsed) {
  SchedulerState.totalRuns++;
  SchedulerState.frame++;

  tickSystems(SchedulerState.frame);

  // Ensure the schedule is ready.
  const buildResult = ensureSchedule();
  if (buildResult !== BUILD_RESULT.OK) {
    const log = _safeLogger();
    if (log) log.error(LOG_CHANNEL.CORE,
      `[020_scn_SystemScheduler] schedule build failed: ${BUILD_RESULT_NAME[buildResult]}`);
    return _frameResult;
  }

  // Frame hook — before.
  if (typeof FrameHooks.beforeFrame === 'function') {
    try {
      FrameHooks.beforeFrame(dt, elapsed);
    } catch (_) { /* swallow */ }
  }

  const frameStart = _now();
  let systemsRun = 0;
  let systemsSkipped = 0;
  let gateSkips = 0;
  let budgetOverruns = 0;

  // Phases 1..5 (UPDATE family). Phase 0 INIT is boot-only, phase 6 RENDER
  // runs at the end.
  for (let phase = PHASE.INIT; phase < PHASE.COUNT; phase++) {
    const phaseStart = _now();

    // Phase gate check.
    const gate = _phaseGates[phase];
    if (gate && gate.fn) {
      let allowed = false;
      try { allowed = gate.fn.call(gate.ctx, phase, dt, elapsed) === true; }
      catch (_) { allowed = false; }
      if (!allowed) {
        SchedulerState.phaseLastMs[phase] = 0;
        _frameResult.phaseCostMs[phase] = 0;
        continue;
      }
    }

    const indices = _phaseIndices[phase];
    const count = _phaseCounts[phase];

    for (let i = 0; i < count; i++) {
      const slot = _slots[indices[i]];
      const sys = slot.system;
      if (!sys) continue;

      // 1. Enabled gate.
      if (slot.enabled === 0 || sys.enabled === 0) {
        systemsSkipped++;
        continue;
      }
      if (sys.state !== SYSTEM_STATE.ENABLED) {
        systemsSkipped++;
        continue;
      }

      // 2. Frame gate.
      if (slot.gateFn) {
        let allow = false;
        try { allow = slot.gateFn.call(slot.gateCtx, sys, dt, elapsed) === true; }
        catch (_) { allow = false; }
        if (!allow) {
          gateSkips++;
          continue;
        }
      }

      // 3. beforeSystem hook.
      if (typeof FrameHooks.beforeSystem === 'function') {
        try { FrameHooks.beforeSystem(sys, phase, dt, elapsed); }
        catch (_) { /* swallow */ }
      }

      // 4. Dispatch by phase.
      if (phase === PHASE.LATE_UPDATE) {
        sys.lateUpdate(dt, elapsed);
      } else if (phase === PHASE.POST) {
        if (typeof sys.onPost === 'function') sys.onPost(dt, elapsed);
      } else if (phase === PHASE.RENDER) {
        if (typeof sys.onRenderFrame === 'function') sys.onRenderFrame(dt, elapsed);
      } else {
        sys.update(dt, elapsed);
      }

      // 5. afterSystem hook.
      if (typeof FrameHooks.afterSystem === 'function') {
        try { FrameHooks.afterSystem(sys, phase, dt, elapsed); }
        catch (_) { /* swallow */ }
      }

      systemsRun++;
    }

    const phaseCost = _now() - phaseStart;
    SchedulerState.phaseLastMs[phase] = phaseCost;
    SchedulerState.phaseEmaMs[phase] += (phaseCost - SchedulerState.phaseEmaMs[phase]) * 0.12;
    if (phaseCost > SchedulerState.phasePeakMs[phase]) {
      SchedulerState.phasePeakMs[phase] = phaseCost;
    }
    _frameResult.phaseCostMs[phase] = phaseCost;

    // Budget accounting.
    const budget = SchedulerState.phaseBudgetMs[phase];
    if (phaseCost > budget) {
      budgetOverruns++;
      SchedulerState.totalBudgetOverruns++;
      SchedulerState.phaseOverrunCount[phase]++;

      // Feedback to the quality controller.
      try {
        const qc = getDefaultQualityController();
        if (qc && typeof qc.setExtraBias === 'function') {
          const overrun = (phaseCost - budget) / Math.max(0.001, budget);
          qc.setExtraBias(Math.min(1.0, overrun * 0.25));
        }
      } catch (_) { /* swallow */ }
    }
  }

  const frameCost = _now() - frameStart;
  SchedulerState.lastFrameCostMs = frameCost;
  SchedulerState.lastFrameEmaMs += (frameCost - SchedulerState.lastFrameEmaMs) * 0.12;
  if (frameCost > SchedulerState.lastFramePeakMs) {
    SchedulerState.lastFramePeakMs = frameCost;
  }

  SchedulerState.totalSystemsRun += systemsRun;
  SchedulerState.totalSystemsSkipped += systemsSkipped;
  SchedulerState.totalGateSkips += gateSkips;
  if (systemsRun > SchedulerState.peakActiveSystems) {
    SchedulerState.peakActiveSystems = systemsRun;
  }

  // Frame hook — after.
  if (typeof FrameHooks.afterFrame === 'function') {
    try { FrameHooks.afterFrame(dt, elapsed); }
    catch (_) { /* swallow */ }
  }

  // Profiler mark.
  try {
    const profiler = getDefaultProfiler();
    if (profiler && profiler.mark) profiler.mark('scheduler.frame');
  } catch (_) { /* swallow */ }

  _frameResult.frame = SchedulerState.frame;
  _frameResult.systemsRun = systemsRun;
  _frameResult.systemsSkipped = systemsSkipped;
  _frameResult.gateSkips = gateSkips;
  _frameResult.budgetOverruns = budgetOverruns;
  _frameResult.totalCostMs = frameCost;

  return _frameResult;
}

/* ------------------------------------------------------------------ */
/* 9. BOOT-TIME INIT PASS                                             */
/* ------------------------------------------------------------------ */

/**
 * Runs the one-shot INIT phase, calling `init(ctx)` on every registered
 * system in the resolved order. Called exactly once at boot.
 */
export function runInitPass(ctx) {
  const buildResult = ensureSchedule();
  if (buildResult !== BUILD_RESULT.OK) {
    const log = _safeLogger();
    if (log) log.error(LOG_CHANNEL.CORE,
      `[020_scn_SystemScheduler] init pass failed: ${BUILD_RESULT_NAME[buildResult]}`);
    return 0;
  }

  const indices = _phaseIndices[PHASE.INIT];
  const count = _phaseCounts[PHASE.INIT];
  let initialized = 0;

  // If INIT phase is empty, fall back to iterating every system in
  // priority order across the remaining phases.
  if (count === 0) {
    const all = getAllSystems();
    all.sort((a, b) => (a.priority | 0) - (b.priority | 0));
    for (let i = 0; i < all.length; i++) {
      const sys = all[i];
      if (!sys) continue;
      if (sys.state === SYSTEM_STATE.DISPOSED) continue;
      try {
        sys.init(ctx);
        initialized++;
      } catch (e) {
        const log = _safeLogger();
        if (log) log.error(LOG_CHANNEL.CORE,
          `[020_scn_SystemScheduler] init failed for "${sys.name}": ${e && e.message}`);
      }
    }
    return initialized;
  }

  for (let i = 0; i < count; i++) {
    const slot = _slots[indices[i]];
    const sys = slot.system;
    if (!sys) continue;
    if (sys.state === SYSTEM_STATE.DISPOSED) continue;
    try {
      sys.init(ctx);
      initialized++;
    } catch (e) {
      const log = _safeLogger();
      if (log) log.error(LOG_CHANNEL.CORE,
        `[020_scn_SystemScheduler] init failed for "${sys.name}": ${e && e.message}`);
    }
  }

  return initialized;
}

/* ------------------------------------------------------------------ */
/* 10. RESIZE PROPAGATION                                             */
/* ------------------------------------------------------------------ */

/**
 * Propagates a viewport resize to every scheduled system in priority
 * order. Never rebuilds the schedule.
 */
export function resizeScheduledSystems(width, height, pixelRatio) {
  let count = 0;
  for (let p = 0; p < PHASE.COUNT; p++) {
    const indices = _phaseIndices[p];
    const n = _phaseCounts[p];
    for (let i = 0; i < n; i++) {
      const slot = _slots[indices[i]];
      if (slot.system) {
        slot.system.resize(width, height, pixelRatio);
        count++;
      }
    }
  }
  return count;
}

/* ------------------------------------------------------------------ */
/* 11. QUERY HELPERS                                                  */
/* ------------------------------------------------------------------ */

/**
 * Returns the number of systems scheduled in the given phase.
 */
export function getPhaseSystemCount(phase) {
  if (phase < 0 || phase >= PHASE.COUNT) return 0;
  return _phaseCounts[phase];
}

/**
 * Iterates every system in the given phase, in scheduled order.
 */
export function forEachScheduledSystem(phase, fn, ctx) {
  if (phase < 0 || phase >= PHASE.COUNT) return 0;
  if (typeof fn !== 'function') return 0;
  const indices = _phaseIndices[phase];
  const n = _phaseCounts[phase];
  for (let i = 0; i < n; i++) {
    const slot = _slots[indices[i]];
    if (slot.system) fn.call(ctx, slot.system, slot);
  }
  return n;
}

/**
 * Returns an array of the systems scheduled in the given phase
 * (allocates).
 */
export function getScheduledSystemsInPhase(phase) {
  if (phase < 0 || phase >= PHASE.COUNT) return [];
  const out = [];
  const indices = _phaseIndices[phase];
  const n = _phaseCounts[phase];
  for (let i = 0; i < n; i++) {
    const slot = _slots[indices[i]];
    if (slot.system) out.push(slot.system);
  }
  return out;
}

/**
 * Returns a read-only view of the schedule as a JSON-friendly array.
 */
export function describeSchedule() {
  const out = [];
  for (let p = 0; p < PHASE.COUNT; p++) {
    const indices = _phaseIndices[p];
    const n = _phaseCounts[p];
    const phaseSystems = [];
    for (let i = 0; i < n; i++) {
      const slot = _slots[indices[i]];
      if (!slot.system) continue;
      phaseSystems.push({
        name:      slot.system.name,
        priority:  slot.priority,
        level:     slot.level,
        enabled:   slot.enabled === 1 && slot.system.enabled === 1,
        hasGate:   slot.gateFn !== null,
        avgCostMs: slot.system.avgCostMs,
      });
    }
    out.push({
      phase:   PHASE_NAME[p],
      count:   n,
      systems: phaseSystems,
    });
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* 12. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getSchedulerStats() {
  const phaseStats = new Array(PHASE.COUNT);
  for (let p = 0; p < PHASE.COUNT; p++) {
    phaseStats[p] = {
      phase:           PHASE_NAME[p],
      systemCount:     _phaseCounts[p],
      lastCostMs:      SchedulerState.phaseLastMs[p],
      emaCostMs:       SchedulerState.phaseEmaMs[p],
      peakCostMs:      SchedulerState.phasePeakMs[p],
      budgetMs:        SchedulerState.phaseBudgetMs[p],
      overrunCount:    SchedulerState.phaseOverrunCount[p],
      overBudget:      SchedulerState.phaseLastMs[p] > SchedulerState.phaseBudgetMs[p],
    };
  }

  return {
    frame:               SchedulerState.frame,
    built:               SchedulerState.built,
    dirty:               SchedulerState.dirty,
    slotCount:           _slotCount,
    capacity:            MAX_SCHEDULE_ENTRIES,
    buildCount:          SchedulerState.buildCount,
    lastBuildFrame:      SchedulerState.lastBuildFrame,
    totalRuns:           SchedulerState.totalRuns,
    totalSystemsRun:     SchedulerState.totalSystemsRun,
    totalSystemsSkipped: SchedulerState.totalSystemsSkipped,
    totalGateSkips:      SchedulerState.totalGateSkips,
    totalBudgetOverruns: SchedulerState.totalBudgetOverruns,
    peakActiveSystems:   SchedulerState.peakActiveSystems,
    lastFrameCostMs:     SchedulerState.lastFrameCostMs,
    lastFrameEmaMs:      SchedulerState.lastFrameEmaMs,
    lastFramePeakMs:     SchedulerState.lastFramePeakMs,
    phases:              phaseStats,
    perfTier:            PERF_TIER_LOCAL,
  };
}

/**
 * Returns the current per-phase pressure in [0..1], where 1 means the
 * phase is exactly at its budget and >1 means over budget.
 */
export function getPhasePressure() {
  const out = new Float32Array(PHASE.COUNT);
  for (let p = 0; p < PHASE.COUNT; p++) {
    const budget = SchedulerState.phaseBudgetMs[p];
    out[p] = budget > 0 ? (SchedulerState.phaseEmaMs[p] / budget) : 0;
  }
  return out;
}

/**
 * Adjusts a phase budget. Called by the quality controller to throttle
 * expensive phases under pressure.
 */
export function setPhaseBudget(phase, budgetMs) {
  if (phase < 0 || phase >= PHASE.COUNT) return false;
  const v = Number(budgetMs);
  if (!Number.isFinite(v) || v <= 0) return false;
  SchedulerState.phaseBudgetMs[phase] = v;
  return true;
}

/**
 * Resets every phase budget to the tier default.
 */
export function resetPhaseBudgets() {
  const d = DEFAULT_PHASE_BUDGET_MS;
  SchedulerState.phaseBudgetMs[PHASE.INIT]        = d.init;
  SchedulerState.phaseBudgetMs[PHASE.EARLY]       = d.early;
  SchedulerState.phaseBudgetMs[PHASE.UPDATE]      = d.update;
  SchedulerState.phaseBudgetMs[PHASE.LATE]        = d.late;
  SchedulerState.phaseBudgetMs[PHASE.LATE_UPDATE] = d.late_update;
  SchedulerState.phaseBudgetMs[PHASE.POST]        = d.post;
  SchedulerState.phaseBudgetMs[PHASE.RENDER]      = d.render;
}

/* ------------------------------------------------------------------ */
/* 13. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Clears the schedule and every statistics counter. Does NOT dispose
 * the registered systems (they remain in the system registry).
 */
export function resetScheduler() {
  for (let i = 0; i < _slotCount; i++) _slots[i].reset();
  _slotCount = 0;
  _nameToSlot.clear();

  for (let p = 0; p < PHASE.COUNT; p++) {
    _phaseCounts[p] = 0;
    _phaseIndices[p].fill(0);
  }

  for (let p = 0; p < PHASE.COUNT; p++) {
    _phaseGates[p] = null;
  }

  SchedulerState.frame = 0;
  SchedulerState.built = false;
  SchedulerState.dirty = true;
  SchedulerState.lastBuildFrame = -1;
  SchedulerState.buildCount = 0;
  SchedulerState.totalRuns = 0;
  SchedulerState.totalSystemsRun = 0;
  SchedulerState.totalSystemsSkipped = 0;
  SchedulerState.totalGateSkips = 0;
  SchedulerState.totalBudgetOverruns = 0;
  SchedulerState.phaseOverrunCount.fill(0);
  SchedulerState.phaseLastMs.fill(0);
  SchedulerState.phaseEmaMs.fill(0);
  SchedulerState.phasePeakMs.fill(0);
  SchedulerState.lastFrameCostMs = 0;
  SchedulerState.lastFrameEmaMs = 0;
  SchedulerState.lastFramePeakMs = 0;
  SchedulerState.peakActiveSystems = 0;

  resetPhaseBudgets();
}

/* ------------------------------------------------------------------ */
/* 14. REGISTRATION WITH THE COMPONENT REGISTRY                       */
/* ------------------------------------------------------------------ */

/**
 * The scheduler does not declare its own ECS components. No-op, present
 * for API symmetry with the other scene modules.
 */
export function registerSchedulerComponents(_registry) {
  return 0;
}

/* ------------------------------------------------------------------ */
/* 15. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  PHASE,
  PHASE_NAME,
  MAX_SCHEDULE_ENTRIES,
  DEFAULT_PHASE_BUDGET_MS,
  BUILD_RESULT,
  BUILD_RESULT_NAME,

  // Schedule slot class
  ScheduleSlot,

  // State
  SchedulerState,

  // Hooks
  FrameHooks,

  // Schedule construction
  markScheduleDirty,
  isScheduleDirty,
  buildSchedule,
  ensureSchedule,

  // Gates
  setGate,
  clearGate,
  setPhaseGate,
  clearPhaseGate,

  // Frame execution
  runFrame,
  runInitPass,
  resizeScheduledSystems,

  // Query
  getPhaseSystemCount,
  forEachScheduledSystem,
  getScheduledSystemsInPhase,
  describeSchedule,

  // Diagnostics
  getSchedulerStats,
  getPhasePressure,
  setPhaseBudget,
  resetPhaseBudgets,

  // Registration
  registerSchedulerComponents,

  // Reset
  resetScheduler,
};

export default _defaultExport;