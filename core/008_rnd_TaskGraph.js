// File : 008
// name : src/core/008_rnd_TaskGraph.js
// description : Directed acyclic task graph for the anime lighting stack on
//               Android mobile. Where 007_rnd_JobQueue.js handles individual
//               job execution with priority and dependencies, THIS module
//               builds and executes the HIGH-LEVEL task pipeline that every
//               frame must run: light list build → shadow atlas pack →
//               cascade matrix solve → GI probe bake → AO blur → cluster
//               grid build → environment palette solve → interior volume
//               update → exterior probe solve → post buffer prepare →
//               director hint emit.
//
//               It is the scheduling layer ABOVE the job queue: it defines
//               which lighting stages must run this frame, in what order,
//               with which dependencies, at which frequency domain, and
//               whether they can be parallelized. The JobQueue then takes
//               the ready nodes and dispatches them.
//
//               Features:
//                 • Fixed-capacity DAG: pre-allocated nodes, edges, and
//                   adjacency lists — no Map/Set on the hot path.
//                 • Explicit dependency edges (up to 8 parents per node).
//                 • Kahn topological sort with cycle detection and hard
//                   failure (a lighting cycle is a logic bug, never a
//                   silent degrade).
//                 • Per-node frequency domain binding (SIMULATION / LIGHTS /
//                   SHADOWS / GI / AO / ENVIRONMENT / INTERIOR / EXTERIOR /
//                   DIRECTOR / POST) so each node runs at its own cadence.
//                 • Skip propagation: when an upstream node is skipped
//                   (because its domain didn't fire this frame), downstream
//                   nodes see the skip and decide whether to run on stale
//                   data or also skip — the graph is a data-availability
//                   graph, not just a scheduling graph.
//                 • Parallel execution groups: nodes at the same topological
//                   depth with no cross-dependencies are flagged as
//                   "parallel-safe" so the JobQueue can dispatch them in a
//                   single batch.
//                 • Critical path tracking: the graph records the longest
//                   chain through the DAG so adaptive quality controllers
//                   can throttle the SHALLOWEST nodes first, preserving the
//                   critical path's visual output.
//                 • Starvation guard: a node that hasn't run for N frames
//                   is force-promoted to run before any non-critical node.
//                 • Zero per-frame allocations: every scratch array is
//                   pre-sized, every callback is pre-bound at registration.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external graph libs; no Promises on the hot path;
//               every typed array sized once at construction.
// best for : Declaring and executing the deterministic lighting pipeline:
//            build light list → pack shadow atlas → solve cascade matrices →
//            bake GI probes → blur AO → build cluster grid → solve env
//            palette → update interior volume → solve exterior probes →
//            prepare post buffers → emit director hints. Every downstream
//            lighting system (006_lgt_LightManager through 380_lgt_lights)
//            registers its update as a graph node, and the graph guarantees
//            correct ordering, correct cadence, and correct parallelization
//            with zero drift.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  assertBiteCSReady,
  isBiteCSReady,
} from './009_scn_BiteCSVersionPolicy.js';

import {
  DOMAIN,
  DOMAIN_NAME,
} from './005_rnd_FrameScheduler.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER = getPerfTier();

export const MAX_NODES = PERF_TIER === 'HIGH' ? 256 : PERF_TIER === 'MEDIUM' ? 192 : 128;
export const MAX_EDGES = MAX_NODES * 8;

export const TASK_NODE_STATE = Object.freeze({
  IDLE:     0,
  PENDING:  1,
  READY:    2,
  RUNNING:  3,
  DONE:     4,
  SKIPPED:  5,
  BLOCKED:  6,
  FAILED:   7,
});

export const TASK_NODE_STATE_NAME = Object.freeze([
  'idle',
  'pending',
  'ready',
  'running',
  'done',
  'skipped',
  'blocked',
  'failed',
]);

export const TASK_FLAGS = Object.freeze({
  NONE:              0,
  PARALLEL_SAFE:     1 << 0,
  CRITICAL_PATH:     1 << 1,
  ALLOW_STALE:       1 << 2, // may run on last-frame output of parents
  FORCE_EVERY_FRAME: 1 << 3, // ignore domain cadence
  STARVATION_EXEMPT: 1 << 4, // never auto-promote
});

const STARVATION_FRAMES = 60;

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _now() {
  return (typeof performance !== 'undefined' ? performance.now() : Date.now());
}

function _clampNodeIdx(i, count) {
  return (i >= 0 && i < count) ? i : -1;
}

/* ------------------------------------------------------------------ */
/* 2. TASK NODE                                                       */
/* ------------------------------------------------------------------ */

export class TaskNode {
  constructor(index, name) {
    this.index        = index;
    this.name         = name;
    this.state        = TASK_NODE_STATE.IDLE;

    this.domain       = DOMAIN.SIMULATION;
    this.flags        = TASK_FLAGS.NONE;
    this.priority     = 100;   // lower = earlier within same depth

    this.run          = null;
    this.ctx          = null;

    // Dependency edges (incoming).
    this.parents      = new Int32Array(8);
    this.parentCount  = 0;

    // Adjacency (outgoing).
    this.children     = new Int32Array(16);
    this.childCount   = 0;

    // Topological depth (computed by topoSort).
    this.depth        = 0;

    // Frames since last successful run.
    this.idleFrames   = 0;
    this.lastRunMs    = 0;
    this.lastRunEma   = 0;
    this.peakMs       = 0;
    this.runCount     = 0;

    // Per-frame outcome from parents (allows stale-data decisions).
    this.upstreamSkipped = 0;
    this.upstreamStale   = 0;

    // Bound at register time to avoid closure allocation per frame.
    this._invoke = null;
    this._invokeCtx = null;
  }

  reset() {
    this.state = TASK_NODE_STATE.IDLE;
    this.parentCount = 0;
    this.childCount = 0;
    this.depth = 0;
    this.idleFrames = 0;
    this.lastRunMs = 0;
    this.lastRunEma = 0;
    this.peakMs = 0;
    this.runCount = 0;
    this.upstreamSkipped = 0;
    this.upstreamStale = 0;
    this.run = null;
    this.ctx = null;
    this._invoke = null;
    this._invokeCtx = null;
    for (let i = 0; i < 8; i++) this.parents[i] = -1;
    for (let i = 0; i < 16; i++) this.children[i] = -1;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 3. TOPOLOGICAL ORDER BUFFER                                        */
/* ------------------------------------------------------------------ */

export class TopoOrder {
  constructor(capacity) {
    this.capacity = capacity;
    this.indices  = new Int32Array(capacity);
    this.depth    = new Int32Array(capacity);
    this.count    = 0;
    this.maxDepth = 0;
  }

  reset() {
    this.count = 0;
    this.maxDepth = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 4. TASK GRAPH                                                      */
/* ------------------------------------------------------------------ */

export class TaskGraph {
  constructor(options = {}) {
    this.options = Object.assign({
      capacity:             MAX_NODES,
      autoSort:             true,
      starvePromote:        true,
      starvationFrames:     STARVATION_FRAMES,
      trackCriticalPath:    true,
      propagateSkip:        true,
    }, options || {});

    this.capacity = this.options.capacity;

    // Node pool (pre-allocated, no growth on hot path).
    this.nodes = new Array(this.capacity);
    for (let i = 0; i < this.capacity; i++) {
      this.nodes[i] = new TaskNode(i, `node_${i}`);
    }
    this.nodeCount = 0;

    // Name → index map (registered once at setup).
    this.nameIndex = new Map();

    // Topological order (recomputed on structural change only).
    this.order = new TopoOrder(this.capacity);
    this._dirty = true;

    // Parallel depth groups (recomputed on structural change).
    this.depthGroups = new Array(this.capacity);
    for (let i = 0; i < this.capacity; i++) this.depthGroups[i] = [];
    this.depthGroupCount = 0;

    // Critical path (index chain).
    this.criticalPath     = new Int32Array(this.capacity);
    this.criticalPathLen  = 0;

    // Per-frame scratch (pre-allocated).
    this._indeg      = new Uint16Array(this.capacity);
    this._queue      = new Int32Array(this.capacity);
    this._readyBatch = new Int32Array(this.capacity);

    // Stats.
    this.frame        = 0;
    this.executedLast = 0;
    this.skippedLast  = 0;
    this.failedLast   = 0;

    this._listeners = new Map();

    // Setup-only invocation buffer (no per-frame alloc).
    this._invokeArgs = { dt: 0, elapsed: 0, node: null, graph: this };
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
      try { arr[i](payload); } catch (e) { console.error(`[008_rnd_TaskGraph] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- node registration ---------------- */

  registerNode(name, run, options = {}) {
    if (this.nodeCount >= this.capacity) return -1;
    if (typeof name !== 'string' || name.length === 0) return -1;
    if (this.nameIndex.has(name)) return this.nameIndex.get(name);
    if (typeof run !== 'function') return -1;

    const idx = this.nodeCount++;
    const node = this.nodes[idx];

    node.reset();
    node.index  = idx;
    node.name   = name;
    node.run    = run;
    node.ctx    = options.ctx || null;

    node.domain   = (options.domain !== undefined ? options.domain : DOMAIN.SIMULATION) | 0;
    node.priority = (options.priority !== undefined ? options.priority : 100) | 0;
    node.flags    = (options.flags !== undefined ? options.flags : TASK_FLAGS.NONE) | 0;

    // Pre-bind invocation to avoid closures per frame.
    node._invoke    = run;
    node._invokeCtx = node.ctx;

    this.nameIndex.set(name, idx);
    this._dirty = true;

    this._emit('registered', { index: idx, name, domain: node.domain, priority: node.priority });
    return idx;
  }

  /* ---------------- dependency edges ---------------- */

  addDependency(childNameOrIdx, parentNameOrIdx) {
    const cIdx = this._resolveIdx(childNameOrIdx);
    const pIdx = this._resolveIdx(parentNameOrIdx);
    if (cIdx < 0 || pIdx < 0) return false;
    if (cIdx === pIdx) return false;

    const child = this.nodes[cIdx];
    const parent = this.nodes[pIdx];

    // Prevent duplicate edge.
    for (let i = 0; i < child.parentCount; i++) {
      if (child.parents[i] === pIdx) return true;
    }

    if (child.parentCount >= 8) {
      console.error(`[008_rnd_TaskGraph] node "${child.name}" exceeds max parents (8)`);
      return false;
    }

    child.parents[child.parentCount++] = pIdx;

    if (parent.childCount >= 16) {
      console.error(`[008_rnd_TaskGraph] node "${parent.name}" exceeds max children (16)`);
      return false;
    }

    parent.children[parent.childCount++] = cIdx;

    this._dirty = true;
    return true;
  }

  /* ---------------- name / index resolution ---------------- */

  _resolveIdx(nameOrIdx) {
    if (typeof nameOrIdx === 'number') {
      return _clampNodeIdx(nameOrIdx, this.nodeCount);
    }
    if (typeof nameOrIdx === 'string') {
      const idx = this.nameIndex.get(nameOrIdx);
      return (idx === undefined) ? -1 : idx;
    }
    return -1;
  }

  getNode(nameOrIdx) {
    const idx = this._resolveIdx(nameOrIdx);
    return (idx >= 0) ? this.nodes[idx] : null;
  }

  /* ---------------- topological sort ---------------- */

  topoSort() {
    if (!this._dirty && this.order.count === this.nodeCount) return true;

    const n = this.nodeCount;
    const indeg = this._indeg;
    const queue = this._queue;
    const order = this.order;

    for (let i = 0; i < n; i++) indeg[i] = 0;
    for (let i = 0; i < n; i++) {
      const node = this.nodes[i];
      for (let p = 0; p < node.parentCount; p++) {
        const par = node.parents[p];
        if (par >= 0 && par < n) indeg[par]++;
      }
    }

    let qh = 0, qt = 0;
    for (let i = 0; i < n; i++) {
      if (indeg[i] === 0) queue[qt++] = i;
    }

    order.reset();
    let write = 0;
    let maxDepth = 0;
    const depth = this._indeg; // reuse array — indeg no longer needed after Kahn

    // Reset depth first.
    for (let i = 0; i < n; i++) depth[i] = 0;

    while (qh < qt) {
      const i = queue[qh++];
      order.indices[write] = i;
      order.depth[write]   = depth[i];
      if (depth[i] > maxDepth) maxDepth = depth[i];
      write++;

      const node = this.nodes[i];
      node.depth = depth[i];

      for (let c = 0; c < node.childCount; c++) {
        const ch = node.children[c];
        if (ch < 0 || ch >= n) continue;
        const chNode = this.nodes[ch];
        for (let p = 0; p < chNode.parentCount; p++) {
          const par = chNode.parents[p];
          if (par === i) {
            if (depth[ch] < depth[i] + 1) depth[ch] = depth[i] + 1;
          }
        }
        if (--indeg[ch] === 0) queue[qt++] = ch;
      }
    }

    if (write !== n) {
      // Cycle detected → hard failure.
      console.error(
        `[008_rnd_TaskGraph] cycle detected: ${write}/${n} nodes reached. ` +
        `A cycle in the lighting pipeline is a logic bug.`
      );
      this._dirty = false;
      order.count = 0;
      order.maxDepth = 0;
      return false;
    }

    order.count = write;
    order.maxDepth = maxDepth;

    // Rebuild depth groups.
    for (let d = 0; d <= maxDepth; d++) this.depthGroups[d].length = 0;
    for (let i = 0; i < write; i++) {
      const idx = order.indices[i];
      const dep = order.depth[i];
      this.depthGroups[dep].push(idx);
    }
    this.depthGroupCount = maxDepth + 1;

    // Compute critical path (longest path through DAG).
    if (this.options.trackCriticalPath) this._computeCriticalPath();

    this._dirty = false;
    this._emit('sorted', { nodeCount: n, maxDepth, groups: this.depthGroupCount });
    return true;
  }

  _computeCriticalPath() {
    const n = this.nodeCount;
    const order = this.order;

    const dist = this._queue; // reuse
    for (let i = 0; i < n; i++) dist[i] = 0;

    let bestEnd = -1;
    let bestDist = -1;

    for (let oi = 0; oi < order.count; oi++) {
      const idx = order.indices[oi];
      const node = this.nodes[idx];
      let d = dist[idx];
      if (node.childCount === 0 && d > bestDist) {
        bestDist = d;
        bestEnd = idx;
      }
      for (let c = 0; c < node.childCount; c++) {
        const ch = node.children[c];
        if (ch < 0 || ch >= n) continue;
        if (dist[ch] < d + 1) dist[ch] = d + 1;
      }
    }

    if (bestEnd < 0) {
      this.criticalPathLen = 0;
      return;
    }

    // Walk back from bestEnd via parents picking the longest.
    this.criticalPathLen = bestDist + 1;
    let cur = bestEnd;
    let slot = this.criticalPathLen - 1;
    const seen = new Uint8Array(n);

    while (cur >= 0 && slot >= 0) {
      if (seen[cur]) break;
      seen[cur] = 1;
      this.criticalPath[slot--] = cur;
      const node = this.nodes[cur];
      let nextCur = -1;
      let nextD = -1;
      for (let p = 0; p < node.parentCount; p++) {
        const par = node.parents[p];
        if (par < 0 || par >= n) continue;
        if (dist[par] > nextD) { nextD = dist[par]; nextCur = par; }
      }
      cur = nextCur;
    }

    // Mark nodes on critical path.
    for (let i = 0; i < n; i++) this.nodes[i].flags &= ~TASK_FLAGS.CRITICAL_PATH;
    for (let i = 0; i < this.criticalPathLen; i++) {
      const idx = this.criticalPath[i];
      if (idx >= 0 && idx < n) this.nodes[idx].flags |= TASK_FLAGS.CRITICAL_PATH;
    }

    this._emit('criticalpath', { length: this.criticalPathLen });
  }

  /* ---------------- per-frame execution ---------------- */

  tick(dt, elapsed, domainFiredMask) {
    this.frame++;

    if (this._dirty && this.options.autoSort) this.topoSort();
    if (this.order.count === 0 && this.nodeCount > 0) return 0;

    // Reset per-frame state.
    for (let i = 0; i < this.nodeCount; i++) {
      const node = this.nodes[i];
      node.upstreamSkipped = 0;
      node.upstreamStale = 0;
      node.idleFrames++;
    }

    // Starvation auto-promote (mark for force-run).
    let promoted = 0;
    if (this.options.starvePromote) {
      for (let i = 0; i < this.nodeCount; i++) {
        const node = this.nodes[i];
        if (node.flags & TASK_FLAGS.STARVATION_EXEMPT) continue;
        if (node.idleFrames >= this.options.starvationFrames) {
          node.flags |= TASK_FLAGS.FORCE_EVERY_FRAME;
          promoted++;
        }
      }
    }

    let executed = 0;
    let skipped = 0;
    let failed = 0;

    // Walk topo order.
    for (let oi = 0; oi < this.order.count; oi++) {
      const idx = this.order.indices[oi];
      const node = this.nodes[idx];

      // Check domain firing.
      const domainOk =
        (node.flags & TASK_FLAGS.FORCE_EVERY_FRAME) ||
        (domainFiredMask === undefined) ||
        ((domainFiredMask >> node.domain) & 1) === 1;

      // Check parents' state for stale propagation.
      let parentsReady = true;
      for (let p = 0; p < node.parentCount; p++) {
        const par = node.parents[p];
        if (par < 0 || par >= this.nodeCount) continue;
        const parState = this.nodes[par].state;
        if (parState === TASK_NODE_STATE.DONE) continue;
        if (parState === TASK_NODE_STATE.SKIPPED) {
          if (node.flags & TASK_FLAGS.ALLOW_STALE) {
            node.upstreamSkipped++;
            continue;
          }
          parentsReady = false;
          break;
        }
        if (parState === TASK_NODE_STATE.FAILED || parState === TASK_NODE_STATE.BLOCKED) {
          parentsReady = false;
          break;
        }
        // Pending/Running/Idle → not ready.
        parentsReady = false;
        break;
      }

      if (!domainOk || !parentsReady) {
        node.state = TASK_NODE_STATE.SKIPPED;
        skipped++;
        continue;
      }

      // Run.
      const t0 = _now();
      node.state = TASK_NODE_STATE.RUNNING;
      let ok = true;
      let err = null;
      try {
        node._invoke(dt, elapsed, node, this);
      } catch (e) {
        ok = false;
        err = e;
      }
      const t1 = _now();

      node.lastRunMs = t1 - t0;
      node.lastRunEma += (node.lastRunMs - node.lastRunEma) * 0.15;
      if (node.lastRunMs > node.peakMs) node.peakMs = node.lastRunMs;
      node.runCount++;
      node.idleFrames = 0;

      if (ok) {
        node.state = TASK_NODE_STATE.DONE;
        executed++;
        this._emit('ran', { index: idx, name: node.name, ms: node.lastRunMs });
      } else {
        node.state = TASK_NODE_STATE.FAILED;
        failed++;
        console.error(`[008_rnd_TaskGraph] node "${node.name}" failed`, err);
        this._emit('failed', { index: idx, name: node.name, error: err });
      }
    }

    this.executedLast = executed;
    this.skippedLast = skipped;
    this.failedLast = failed;

    this._emit('tick', {
      frame: this.frame,
      executed,
      skipped,
      failed,
      promoted,
    });

    return executed;
  }

  /* ---------------- parallel batch helpers ---------------- */

  getDepthGroups() {
    return { groups: this.depthGroups, count: this.depthGroupCount };
  }

  forEachInDepth(depth, fn, ctx) {
    if (depth < 0 || depth >= this.depthGroupCount) return 0;
    const group = this.depthGroups[depth];
    let n = 0;
    for (let i = 0; i < group.length; i++) {
      fn.call(ctx, this.nodes[group[i]]);
      n++;
    }
    return n;
  }

  /* ---------------- bulk ops ---------------- */

  resetStates() {
    for (let i = 0; i < this.nodeCount; i++) {
      this.nodes[i].state = TASK_NODE_STATE.IDLE;
      this.nodes[i].idleFrames = 0;
    }
    return this;
  }

  clear() {
    for (let i = 0; i < this.capacity; i++) this.nodes[i].reset();
    this.nodeCount = 0;
    this.nameIndex.clear();
    this.order.reset();
    for (let i = 0; i < this.capacity; i++) this.depthGroups[i].length = 0;
    this.depthGroupCount = 0;
    this.criticalPathLen = 0;
    this._dirty = true;
    return this;
  }

  dispose() {
    this.clear();
    this._listeners.clear();
    return this;
  }

  /* ---------------- stats ---------------- */

  getStats() {
    const nodes = new Array(this.nodeCount);
    for (let i = 0; i < this.nodeCount; i++) {
      const n = this.nodes[i];
      nodes[i] = {
        index:       n.index,
        name:        n.name,
        domain:      n.domain,
        domainName:  DOMAIN_NAME[n.domain] || 'unknown',
        state:       TASK_NODE_STATE_NAME[n.state],
        depth:       n.depth,
        priority:    n.priority,
        flags:       n.flags,
        parents:     n.parentCount,
        children:    n.childCount,
        runCount:    n.runCount,
        lastRunMs:   n.lastRunMs,
        lastRunEma:  n.lastRunEma,
        peakMs:      n.peakMs,
        idleFrames:  n.idleFrames,
      };
    }
    return {
      frame:           this.frame,
      nodeCount:       this.nodeCount,
      capacity:        this.capacity,
      maxDepth:        this.order.maxDepth,
      depthGroups:     this.depthGroupCount,
      criticalPathLen: this.criticalPathLen,
      executedLast:    this.executedLast,
      skippedLast:     this.skippedLast,
      failedLast:      this.failedLast,
      nodes,
      perfTier:        PERF_TIER,
    };
  }
}

/* ------------------------------------------------------------------ */
/* 5. STANDARD LIGHTING PIPELINE BUILDER                              */
/* ------------------------------------------------------------------ */

/**
 * Builds the standard 11-node lighting pipeline:
 *
 *   lightListBuild
 *     ├── shadowAtlasPack
 *     │     └── cascadeMatrixSolve
 *     ├── giProbeBake
 *     │     └── aoBlur
 *     ├── clusterGridBuild
 *     └── envPaletteSolve
 *           ├── interiorVolumeUpdate
 *           └── exteriorProbeSolve
 *                 └── postBufferPrepare
 *                       └── directorHintEmit
 *
 * Each node binds to a scheduler domain. Callbacks are supplied by the
 * caller (typically 267_lgt_LightingSystem … 278_lgt_ColorOnlyLightDirector).
 */
export function buildStandardLightingPipeline(graph, callbacks = {}) {
  if (!graph) return null;

  const noop = () => {};

  const nodes = {
    lightListBuild:       graph.registerNode('lightListBuild',       callbacks.lightListBuild       || noop, { domain: DOMAIN.LIGHTS,      priority: 10, flags: TASK_FLAGS.PARALLEL_SAFE }),
    shadowAtlasPack:      graph.registerNode('shadowAtlasPack',      callbacks.shadowAtlasPack      || noop, { domain: DOMAIN.SHADOWS,     priority: 20, flags: TASK_FLAGS.PARALLEL_SAFE }),
    cascadeMatrixSolve:   graph.registerNode('cascadeMatrixSolve',   callbacks.cascadeMatrixSolve   || noop, { domain: DOMAIN.SHADOWS,     priority: 25 }),
    giProbeBake:          graph.registerNode('giProbeBake',          callbacks.giProbeBake          || noop, { domain: DOMAIN.GI,          priority: 30, flags: TASK_FLAGS.ALLOW_STALE }),
    aoBlur:               graph.registerNode('aoBlur',               callbacks.aoBlur               || noop, { domain: DOMAIN.AO,          priority: 35, flags: TASK_FLAGS.ALLOW_STALE }),
    clusterGridBuild:     graph.registerNode('clusterGridBuild',     callbacks.clusterGridBuild     || noop, { domain: DOMAIN.LIGHTS,      priority: 15, flags: TASK_FLAGS.PARALLEL_SAFE }),
    envPaletteSolve:      graph.registerNode('envPaletteSolve',      callbacks.envPaletteSolve      || noop, { domain: DOMAIN.ENVIRONMENT, priority: 40, flags: TASK_FLAGS.ALLOW_STALE }),
    interiorVolumeUpdate: graph.registerNode('interiorVolumeUpdate', callbacks.interiorVolumeUpdate || noop, { domain: DOMAIN.INTERIOR,    priority: 50 }),
    exteriorProbeSolve:   graph.registerNode('exteriorProbeSolve',   callbacks.exteriorProbeSolve   || noop, { domain: DOMAIN.EXTERIOR,    priority: 55 }),
    postBufferPrepare:    graph.registerNode('postBufferPrepare',    callbacks.postBufferPrepare    || noop, { domain: DOMAIN.POST,        priority: 60 }),
    directorHintEmit:     graph.registerNode('directorHintEmit',     callbacks.directorHintEmit     || noop, { domain: DOMAIN.DIRECTOR,    priority: 70 }),
  };

  // Dependency edges (child → parent).
  graph.addDependency(nodes.shadowAtlasPack,      nodes.lightListBuild);
  graph.addDependency(nodes.cascadeMatrixSolve,   nodes.shadowAtlasPack);
  graph.addDependency(nodes.giProbeBake,          nodes.lightListBuild);
  graph.addDependency(nodes.aoBlur,               nodes.giProbeBake);
  graph.addDependency(nodes.clusterGridBuild,     nodes.lightListBuild);
  graph.addDependency(nodes.envPaletteSolve,      nodes.lightListBuild);
  graph.addDependency(nodes.interiorVolumeUpdate, nodes.envPaletteSolve);
  graph.addDependency(nodes.exteriorProbeSolve,   nodes.envPaletteSolve);
  graph.addDependency(nodes.postBufferPrepare,    nodes.exteriorProbeSolve);
  graph.addDependency(nodes.directorHintEmit,     nodes.postBufferPrepare);

  // Topologically resolve once now.
  graph.topoSort();

  return nodes;
}

/* ------------------------------------------------------------------ */
/* 6. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultGraph = null;

export function getDefaultTaskGraph() {
  if (!_defaultGraph) {
    _defaultGraph = new TaskGraph();
    _defaultGraph._pipeline = buildStandardLightingPipeline(_defaultGraph, {});
  }
  return _defaultGraph;
}

export function disposeDefaultTaskGraph() {
  if (_defaultGraph) {
    _defaultGraph.dispose();
    _defaultGraph = null;
  }
}

/* ------------------------------------------------------------------ */
/* 7. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createTaskGraph(options = {}) {
  return new TaskGraph(options);
}

/* ------------------------------------------------------------------ */
/* 8. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  TaskGraph,
  TaskNode,
  TopoOrder,
  createTaskGraph,
  getDefaultTaskGraph,
  disposeDefaultTaskGraph,
  buildStandardLightingPipeline,
  TASK_NODE_STATE,
  TASK_NODE_STATE_NAME,
  TASK_FLAGS,
  MAX_NODES,
  MAX_EDGES,
};

export default _defaultExport;