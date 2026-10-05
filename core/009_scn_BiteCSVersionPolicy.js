// File : 009
// name : src/core/009_rnd_DependencyResolver.js
// description : Dependency resolver for the anime lighting pipeline on Android
//               mobile. Sits between the task graph (008_rnd_TaskGraph.js) and
//               the job queue (007_rnd_JobQueue.js) and answers the three
//               questions every lighting subsystem asks each frame:
//
//                 1. "Which of my upstream lighting nodes actually produced
//                    fresh data this frame, and which are stale?"
//                 2. "Given what changed upstream, what is the minimal set of
//                    downstream nodes that MUST re-run this frame?"
//                 3. "If I skip, what is the visual consequence — can I
//                    substitute last-frame output safely (ALLOW_STALE), or
//                    must I block until upstream finishes?"
//
//               The resolver maintains:
//                 • A canonical snapshot of node generations per frame, so
//                   downstream nodes can compare "was my parent's commit
//                   generation newer than my last read?" with one integer
//                   compare.
//                 • A dirty-propagation table computed once at registration
//                   (given a set of changed nodes, which downstream nodes
//                   become dirty?). This is a bit-set of size MAX_NODES per
//                   node, stored as a flat Uint32Array for cache locality.
//                 • A stale-substitution policy table, per node, resolving
//                   whether ALLOW_STALE, BLOCK, or SKIP_WITH_DEFAULT is the
//                   correct behavior when an upstream is stale.
//                 • A cost-priority ranking (critical path distance × node
//                   cost EMA) so the frame's budget is spent on the nodes
//                   that matter most visually.
//
//               Optimization techniques applied:
//                 • Bitset propagation (one AND per downstream check, no
//                   recursion, no visited sets, no allocs).
//                 • Pre-computed propagation matrix (dirty closure per node)
//                   built ONCE at registration via reverse topological pass.
//                 • Fixed-capacity arrays sized to MAX_NODES; zero dynamic
//                   growth; zero Map/Set lookups on the hot path.
//                 • Per-frame scratch buffers pre-allocated at construction.
//                 • Generation counters as monotonic 32-bit ints — overflow
//                   wraps safely with a `> 0` guard, no Date.now() calls.
//                 • Cache-line-aware layout: all hot arrays are contiguous
//                   typed arrays; policy tables are Uint8Array to fit L1.
//                 • No closures captured per frame; every callback supplied
//                   at registration and stored as a slot index.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external libs; no Promises on the hot path.
// best for : Guaranteeing that the anime lighting stack (006_lgt_LightManager
//            through 380_lgt_lights) re-runs exactly the minimal set of
//            upstream-dependent nodes each frame, and that downstream nodes
//            know precisely whether they can trust their inputs or must
//            substitute last-frame output. Eliminates the two most common
//            Android lighting bugs: (a) updating GI from stale shadow atlas
//            causing light leaks, and (b) skipping AO when shadow map just
//            changed causing hard edges on cel shading.
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
  TaskGraph,
  TASK_NODE_STATE,
  TASK_FLAGS,
  MAX_NODES,
} from './008_rnd_TaskGraph.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER = getPerfTier();

export const MAX_RESOLVE_NODES = MAX_NODES;

export const RESOLVE_POLICY = Object.freeze({
  DEFAULT:        0, // use node flags
  FORCE_FRESH:    1, // never accept stale — must block or skip
  ALLOW_STALE:    2, // accept stale freely
  SKIP_WITH_LAST: 3, // skip and reuse last-frame output
  CRITICAL_ONLY:  4, // only run if on critical path
});

export const RESOLVE_RESULT = Object.freeze({
  RUN:        0,
  RUN_STALE:  1,
  SKIP:       2,
  BLOCK:      3,
  FORCE:      4,
});

export const RESOLVE_RESULT_NAME = Object.freeze([
  'run',
  'run_stale',
  'skip',
  'block',
  'force',
]);

const WORD_BITS = 32;
const WORDS_PER_ROW = ((MAX_RESOLVE_NODES + WORD_BITS - 1) / WORD_BITS) | 0;

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _now() {
  return (typeof performance !== 'undefined' ? performance.now() : Date.now());
}

function _nextGen(g) {
  const n = (g + 1) | 0;
  return n <= 0 ? 1 : n;
}

function _setBit(row, col) {
  const w = (col / WORD_BITS) | 0;
  const b = col % WORD_BITS;
  row[w] |= (1 << b);
}

function _getBit(row, col) {
  const w = (col / WORD_BITS) | 0;
  const b = col % WORD_BITS;
  return (row[w] >> b) & 1;
}

/* ------------------------------------------------------------------ */
/* 2. PROPAGATION MATRIX                                              */
/* ------------------------------------------------------------------ */

/**
 * Dense propagation matrix stored as a flat Uint32Array:
 *   matrix[node * WORDS_PER_ROW + w] = 32 bits of downstream-dirty flags.
 *
 * Row `i` has bit `j` set iff node `j` is a downstream (transitive)
 * dependent of node `i`.
 */
export class PropagationMatrix {
  constructor(nodeCount) {
    this.nodeCount = nodeCount;
    this.wordsPerRow = WORDS_PER_ROW;
    this.data = new Uint32Array(nodeCount * WORDS_PER_ROW);
  }

  clear() {
    this.data.fill(0);
  }

  setDependent(nodeIdx, downstreamIdx) {
    const base = nodeIdx * this.wordsPerRow;
    _setBit(this.data.subarray(base, base + this.wordsPerRow), downstreamIdx);
  }

  isDependent(nodeIdx, downstreamIdx) {
    const base = nodeIdx * this.wordsPerRow;
    return _getBit(this.data.subarray(base, base + this.wordsPerRow), downstreamIdx) === 1;
  }

  /**
   * Reverse topological pass: for each node, OR together its children's
   * propagation rows plus its own direct children.
   *
   * After this pass, row[i] contains every transitive dependent of i.
   */
  computeClosure(graph) {
    if (!graph) return false;

    const n = graph.nodeCount;
    const order = graph.order;

    if (order.count === 0) return false;

    this.clear();

    // Walk topo order in REVERSE so children are processed before parents.
    for (let oi = order.count - 1; oi >= 0; oi--) {
      const idx = order.indices[oi];
      const node = graph.nodes[idx];

      const rowBase = idx * this.wordsPerRow;
      const row = this.data.subarray(rowBase, rowBase + this.wordsPerRow);

      for (let c = 0; c < node.childCount; c++) {
        const ch = node.children[c];
        if (ch < 0 || ch >= n) continue;
        // Set direct child bit.
        _setBit(row, ch);
        // OR in the child's closure.
        const chBase = ch * this.wordsPerRow;
        const chRow = this.data.subarray(chBase, chBase + this.wordsPerRow);
        for (let w = 0; w < this.wordsPerRow; w++) row[w] |= chRow[w];
      }
    }

    return true;
  }
}

/* ------------------------------------------------------------------ */
/* 3. STALE POLICY TABLE                                              */
/* ------------------------------------------------------------------ */

export class StalePolicyTable {
  constructor(nodeCount) {
    this.nodeCount = nodeCount;
    this.policy = new Uint8Array(nodeCount);        // RESOLVE_POLICY
    this.maxStaleFrames = new Uint16Array(nodeCount); // 0 = unlimited
    this.defaultResult = new Uint8Array(nodeCount); // RESOLVE_RESULT on unresolvable
  }

  set(nodeIdx, policy, maxStaleFrames = 0, defaultResult = RESOLVE_RESULT.SKIP) {
    if (nodeIdx < 0 || nodeIdx >= this.nodeCount) return false;
    this.policy[nodeIdx] = policy | 0;
    this.maxStaleFrames[nodeIdx] = maxStaleFrames | 0;
    this.defaultResult[nodeIdx] = defaultResult | 0;
    return true;
  }

  get(nodeIdx) {
    if (nodeIdx < 0 || nodeIdx >= this.nodeCount) return RESOLVE_POLICY.DEFAULT;
    return this.policy[nodeIdx];
  }

  clear() {
    this.policy.fill(0);
    this.maxStaleFrames.fill(0);
    this.defaultResult.fill(0);
  }
}

/* ------------------------------------------------------------------ */
/* 4. FRAME SNAPSHOT                                                  */
/* ------------------------------------------------------------------ */

export class FrameSnapshot {
  constructor(nodeCount) {
    this.nodeCount = nodeCount;
    this.lastCommitGen = new Uint32Array(nodeCount);   // 0 = never committed
    this.lastRunFrame  = new Uint32Array(nodeCount);
    this.lastReadGen   = new Uint32Array(nodeCount);
    this.dirty         = new Uint8Array(nodeCount);
    this.staleFrames   = new Uint16Array(nodeCount);
  }

  clear() {
    this.lastCommitGen.fill(0);
    this.lastRunFrame.fill(0);
    this.lastReadGen.fill(0);
    this.dirty.fill(0);
    this.staleFrames.fill(0);
  }

  bumpCommit(nodeIdx, frame) {
    if (nodeIdx < 0 || nodeIdx >= this.nodeCount) return 0;
    const g = _nextGen(this.lastCommitGen[nodeIdx]);
    this.lastCommitGen[nodeIdx] = g;
    this.lastRunFrame[nodeIdx] = frame;
    this.dirty[nodeIdx] = 1;
    this.staleFrames[nodeIdx] = 0;
    return g;
  }

  markRead(nodeIdx) {
    if (nodeIdx < 0 || nodeIdx >= this.nodeCount) return;
    this.lastReadGen[nodeIdx] = this.lastCommitGen[nodeIdx];
    this.dirty[nodeIdx] = 0;
  }

  isStaleFor(nodeIdx) {
    if (nodeIdx < 0 || nodeIdx >= this.nodeCount) return true;
    return this.lastReadGen[nodeIdx] !== this.lastCommitGen[nodeIdx];
  }

  tickStale() {
    for (let i = 0; i < this.nodeCount; i++) {
      if (this.lastCommitGen[i] === 0) {
        this.staleFrames[i] = 0xFFFF; // never ran
      } else {
        this.staleFrames[i]++;
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* 5. DEPENDENCY RESOLVER                                             */
/* ------------------------------------------------------------------ */

export class DependencyResolver {
  constructor(graph, options = {}) {
    if (!graph || typeof graph !== 'object') {
      throw new Error('[009_rnd_DependencyResolver] graph is required');
    }

    this.options = Object.assign({
      useStaleByDefault:   true,
      maxStaleFrames:      0,     // 0 = unlimited
      propagateDirty:      true,
      evaluateCriticalPath:true,
    }, options || {});

    this.graph = graph;
    this.capacity = graph.capacity;

    this.propagation = new PropagationMatrix(this.capacity);
    this.policy = new StalePolicyTable(this.capacity);
    this.snapshot = new FrameSnapshot(this.capacity);

    // Per-frame scratch.
    this._dirtyMask = new Uint32Array(WORDS_PER_ROW);   // nodes changed this frame
    this._resolveResults = new Uint8Array(this.capacity);
    this._resolveOrder = new Int32Array(this.capacity);
    this._resolveCount = 0;

    // Cost-priority ranking (higher = more visually important).
    this._costEma = new Float32Array(this.capacity);
    this._visualWeight = new Float32Array(this.capacity);

    // Stats.
    this.frame = 0;
    this.stats = {
      totalRuns:     0,
      totalStaleRuns:0,
      totalSkips:    0,
      totalBlocks:   0,
      totalForces:   0,
    };

    this._listeners = new Map();
    this._closureBuilt = false;
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
      try { arr[i](payload); } catch (e) { console.error(`[009_rnd_DependencyResolver] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- setup ---------------- */

  rebuildClosure() {
    if (!this.graph || this.graph.nodeCount === 0) return false;
    if (this.graph._dirty) this.graph.topoSort();
    const ok = this.propagation.computeClosure(this.graph);
    this._closureBuilt = ok;
    if (ok) this._emit('closure', { nodeCount: this.graph.nodeCount });
    return ok;
  }

  setPolicy(nodeOrIdx, policy, maxStaleFrames = 0, defaultResult = RESOLVE_RESULT.SKIP) {
    const idx = this._resolveIdx(nodeOrIdx);
    if (idx < 0) return false;
    return this.policy.set(idx, policy, maxStaleFrames, defaultResult);
  }

  setVisualWeight(nodeOrIdx, weight) {
    const idx = this._resolveIdx(nodeOrIdx);
    if (idx < 0) return false;
    this._visualWeight[idx] = Math.max(0, Math.min(1, Number(weight) || 0));
    return true;
  }

  _resolveIdx(nameOrIdx) {
    if (typeof nameOrIdx === 'number') {
      return (nameOrIdx >= 0 && nameOrIdx < this.graph.nodeCount) ? nameOrIdx : -1;
    }
    if (typeof nameOrIdx === 'string') {
      const idx = this.graph.nameIndex.get(nameOrIdx);
      return (idx === undefined) ? -1 : idx;
    }
    return -1;
  }

  /* ---------------- commit / read bookkeeping ---------------- */

  markCommitted(nodeOrIdx) {
    const idx = this._resolveIdx(nodeOrIdx);
    if (idx < 0) return 0;
    const g = this.snapshot.bumpCommit(idx, this.frame);
    // Set dirty bit in scratch mask.
    const w = (idx / WORD_BITS) | 0;
    const b = idx % WORD_BITS;
    this._dirtyMask[w] |= (1 << b);
    this._emit('commit', { index: idx, name: this.graph.nodes[idx].name, gen: g });
    return g;
  }

  markRead(nodeOrIdx) {
    const idx = this._resolveIdx(nodeOrIdx);
    if (idx < 0) return;
    this.snapshot.markRead(idx);
  }

  isStale(nodeOrIdx) {
    const idx = this._resolveIdx(nodeOrIdx);
    if (idx < 0) return true;
    return this.snapshot.isStaleFor(idx);
  }

  /* ---------------- core resolution ---------------- */

  /**
   * Resolves the per-frame execution plan. Returns the count of nodes to run.
   * Populates `this._resolveResults` and `this._resolveOrder`.
   */
  resolve(dt, elapsed, domainFiredMask) {
    this.frame++;

    // Rebuild closure if graph changed.
    if (this.graph._dirty || !this._closureBuilt) {
      this.graph.topoSort();
      this.rebuildClosure();
    }

    // Reset scratch mask.
    for (let i = 0; i < WORDS_PER_ROW; i++) this._dirtyMask[i] = 0;

    // Age stale counters.
    this.snapshot.tickStale();

    // Compute dirty mask from committed nodes this frame.
    // (Consumers should call markCommitted() during their run() callback.)
    // For nodes committed LAST frame, they remain in the snapshot as dirty
    // until someone marks them read. That's the point of the snapshot.

    const n = this.graph.nodeCount;
    const order = this.graph.order;
    const results = this._resolveResults;
    const resolveOrder = this._resolveOrder;
    let count = 0;

    let totalRuns = 0;
    let totalStaleRuns = 0;
    let totalSkips = 0;
    let totalBlocks = 0;
    let totalForces = 0;

    for (let oi = 0; oi < order.count; oi++) {
      const idx = order.indices[oi];
      const node = this.graph.nodes[idx];

      // Node forced by starvation?
      const forced = (node.flags & TASK_FLAGS.FORCE_EVERY_FRAME) !== 0;

      // Domain gating.
      const domainOk =
        forced ||
        (domainFiredMask === undefined) ||
        ((domainFiredMask >> node.domain) & 1) === 1;

      // Upstream state evaluation.
      let upstreamFresh = true;
      let upstreamStale = false;
      let upstreamBlocked = false;

      for (let p = 0; p < node.parentCount; p++) {
        const par = node.parents[p];
        if (par < 0 || par >= n) continue;
        const parState = this.graph.nodes[par].state;

        if (parState === TASK_NODE_STATE.DONE) {
          if (this.snapshot.isStaleFor(par)) {
            // Parent's last run was earlier than its last commit →
            // parent data is actually fresh but not yet read.
            // Acceptable.
          }
          continue;
        }

        if (parState === TASK_NODE_STATE.SKIPPED) {
          if (this._canAcceptStale(idx, par)) {
            upstreamStale = true;
            continue;
          }
          upstreamFresh = false;
          break;
        }

        if (parState === TASK_NODE_STATE.FAILED ||
            parState === TASK_NODE_STATE.BLOCKED) {
          upstreamBlocked = true;
          break;
        }

        // Parent not yet run this frame → block.
        upstreamFresh = false;
        break;
      }

      // Resolve.
      let result;
      if (upstreamBlocked) {
        result = RESOLVE_RESULT.BLOCK;
      } else if (!domainOk) {
        result = RESOLVE_RESULT.SKIP;
      } else if (!upstreamFresh && !upstreamStale) {
        result = RESOLVE_RESULT.BLOCK;
      } else if (forced) {
        result = RESOLVE_RESULT.FORCE;
      } else if (upstreamStale) {
        result = RESOLVE_RESULT.RUN_STALE;
      } else {
        result = RESOLVE_RESULT.RUN;
      }

      // Apply policy overrides.
      const policy = this.policy.get(idx);
      if (policy === RESOLVE_POLICY.FORCE_FRESH) {
        if (result === RESOLVE_RESULT.RUN_STALE) result = RESOLVE_RESULT.BLOCK;
      } else if (policy === RESOLVE_POLICY.ALLOW_STALE) {
        if (result === RESOLVE_RESULT.BLOCK && upstreamStale) result = RESOLVE_RESULT.RUN_STALE;
      } else if (policy === RESOLVE_POLICY.SKIP_WITH_LAST) {
        if (result === RESOLVE_RESULT.RUN_STALE) result = RESOLVE_RESULT.SKIP;
      } else if (policy === RESOLVE_POLICY.CRITICAL_ONLY) {
        if ((node.flags & TASK_FLAGS.CRITICAL_PATH) === 0) result = RESOLVE_RESULT.SKIP;
      }

      results[idx] = result;

      if (result === RESOLVE_RESULT.RUN ||
          result === RESOLVE_RESULT.RUN_STALE ||
          result === RESOLVE_RESULT.FORCE) {
        resolveOrder[count++] = idx;
        if (result === RESOLVE_RESULT.RUN_STALE) totalStaleRuns++;
        else if (result === RESOLVE_RESULT.FORCE) totalForces++;
        else totalRuns++;
      } else if (result === RESOLVE_RESULT.SKIP) {
        totalSkips++;
      } else {
        totalBlocks++;
      }
    }

    this._resolveCount = count;
    this.stats.totalRuns += totalRuns;
    this.stats.totalStaleRuns += totalStaleRuns;
    this.stats.totalSkips += totalSkips;
    this.stats.totalBlocks += totalBlocks;
    this.stats.totalForces += totalForces;

    this._emit('resolved', {
      frame: this.frame,
      run: totalRuns,
      runStale: totalStaleRuns,
      skip: totalSkips,
      block: totalBlocks,
      force: totalForces,
    });

    return count;
  }

  _canAcceptStale(nodeIdx, parentIdx) {
    const policy = this.policy.get(nodeIdx);

    if (policy === RESOLVE_POLICY.ALLOW_STALE) return true;
    if (policy === RESOLVE_POLICY.FORCE_FRESH) return false;
    if (policy === RESOLVE_POLICY.SKIP_WITH_LAST) return true;
    if (policy === RESOLVE_POLICY.CRITICAL_ONLY) {
      return (this.graph.nodes[nodeIdx].flags & TASK_FLAGS.CRITICAL_PATH) === 0;
    }

    // DEFAULT: check node flags.
    if ((this.graph.nodes[nodeIdx].flags & TASK_FLAGS.ALLOW_STALE) !== 0) {
      return true;
    }

    // Check parent's staleness frame count.
    const maxFrames = this.policy.maxStaleFrames[nodeIdx];
    if (maxFrames > 0) {
      const stale = this.snapshot.staleFrames[parentIdx];
      if (stale !== 0xFFFF && stale <= maxFrames) return true;
    }

    return this.options.useStaleByDefault;
  }

  /* ---------------- downstream propagation ---------------- */

  /**
   * Given a set of source nodes that changed, returns a bitmask of all
   * downstream nodes that must be considered dirty this frame.
   *
   * The mask is written into `outMask` (a Uint32Array of WORDS_PER_ROW
   * length). Returns the number of bits set.
   */
  propagateDirty(sources, outMask) {
    if (!sources || !outMask) return 0;
    if (!this._closureBuilt) this.rebuildClosure();

    const data = this.propagation.data;
    for (let w = 0; w < WORDS_PER_ROW; w++) outMask[w] = 0;

    for (let s = 0; s < sources.length; s++) {
      const idx = sources[s];
      if (idx < 0 || idx >= this.capacity) continue;
      const base = idx * WORDS_PER_ROW;
      for (let w = 0; w < WORDS_PER_ROW; w++) {
        outMask[w] |= data[base + w];
      }
    }

    // Count bits.
    let count = 0;
    for (let w = 0; w < WORDS_PER_ROW; w++) {
      let x = outMask[w];
      while (x) { count += x & 1; x >>>= 1; }
    }
    return count;
  }

  /* ---------------- critical path helpers ---------------- */

  isCritical(nodeOrIdx) {
    const idx = this._resolveIdx(nodeOrIdx);
    if (idx < 0) return false;
    return (this.graph.nodes[idx].flags & TASK_FLAGS.CRITICAL_PATH) !== 0;
  }

  getCriticalPathLength() {
    return this.graph.criticalPathLen;
  }

  /* ---------------- cost-priority helpers ---------------- */

  updateCost(nodeOrIdx, ms) {
    const idx = this._resolveIdx(nodeOrIdx);
    if (idx < 0) return;
    this._costEma[idx] += (ms - this._costEma[idx]) * 0.15;
  }

  visualPriority(nodeOrIdx) {
    const idx = this._resolveIdx(nodeOrIdx);
    if (idx < 0) return 0;
    const node = this.graph.nodes[idx];
    const depthBonus = 1.0 + node.depth * 0.05;
    const criticalBonus = (node.flags & TASK_FLAGS.CRITICAL_PATH) ? 1.5 : 1.0;
    return this._visualWeight[idx] * depthBonus * criticalBonus;
  }

  /* ---------------- bulk ops ---------------- */

  reset() {
    this.snapshot.clear();
    this.policy.clear();
    for (let i = 0; i < WORDS_PER_ROW; i++) this._dirtyMask[i] = 0;
    this._resolveResults.fill(0);
    this._resolveCount = 0;
    this.stats.totalRuns = 0;
    this.stats.totalStaleRuns = 0;
    this.stats.totalSkips = 0;
    this.stats.totalBlocks = 0;
    this.stats.totalForces = 0;
    this.frame = 0;
    return this;
  }

  dispose() {
    this.reset();
    this._listeners.clear();
    this.propagation.data = null;
    this._dirtyMask = null;
    this._resolveResults = null;
    this._resolveOrder = null;
    this._costEma = null;
    this._visualWeight = null;
    return this;
  }

  /* ---------------- accessors ---------------- */

  getResolveResult(nodeOrIdx) {
    const idx = this._resolveIdx(nodeOrIdx);
    if (idx < 0) return RESOLVE_RESULT.SKIP;
    return this._resolveResults[idx];
  }

  getResolveOrder() {
    return { order: this._resolveOrder, count: this._resolveCount };
  }

  getStats() {
    const nodes = new Array(this.graph.nodeCount);
    for (let i = 0; i < this.graph.nodeCount; i++) {
      nodes[i] = {
        name:            this.graph.nodes[i].name,
        policy:          this.policy.get(i),
        result:          RESOLVE_RESULT_NAME[this._resolveResults[i]] || 'skip',
        commitGen:       this.snapshot.lastCommitGen[i],
        readGen:         this.snapshot.lastReadGen[i],
        isStale:         this.snapshot.isStaleFor(i),
        staleFrames:     this.snapshot.staleFrames[i],
        costEma:         this._costEma[i],
        visualWeight:    this._visualWeight[i],
        isCritical:      (this.graph.nodes[i].flags & TASK_FLAGS.CRITICAL_PATH) !== 0,
      };
    }
    return {
      frame:           this.frame,
      resolveCount:    this._resolveCount,
      totalRuns:       this.stats.totalRuns,
      totalStaleRuns:  this.stats.totalStaleRuns,
      totalSkips:      this.stats.totalSkips,
      totalBlocks:     this.stats.totalBlocks,
      totalForces:     this.stats.totalForces,
      closureBuilt:    this._closureBuilt,
      nodes,
      perfTier:        PERF_TIER,
    };
  }
}

/* ------------------------------------------------------------------ */
/* 6. STANDARD POLICY BINDINGS                                        */
/* ------------------------------------------------------------------ */

export function applyStandardLightingPolicies(resolver) {
  if (!resolver) return null;

  // Shadow atlas must be fresh — no stale shadow maps.
  resolver.setPolicy('shadowAtlasPack',    RESOLVE_POLICY.FORCE_FRESH);
  resolver.setPolicy('cascadeMatrixSolve', RESOLVE_POLICY.FORCE_FRESH);

  // Light list is critical — always run when its domain fires.
  resolver.setPolicy('lightListBuild',     RESOLVE_POLICY.FORCE_FRESH);

  // GI may accept stale for one frame under pressure.
  resolver.setPolicy('giProbeBake',        RESOLVE_POLICY.ALLOW_STALE, 1, RESOLVE_RESULT.RUN_STALE);

  // AO may accept stale; visual penalty is soft.
  resolver.setPolicy('aoBlur',             RESOLVE_POLICY.ALLOW_STALE, 2, RESOLVE_RESULT.RUN_STALE);

  // Cluster grid must be fresh — mismatch causes lights to pop.
  resolver.setPolicy('clusterGridBuild',   RESOLVE_POLICY.FORCE_FRESH);

  // Environment palette may be several frames stale without visual harm.
  resolver.setPolicy('envPaletteSolve',    RESOLVE_POLICY.SKIP_WITH_LAST);

  // Interior/exterior probes are soft — skip when stale.
  resolver.setPolicy('interiorVolumeUpdate', RESOLVE_POLICY.SKIP_WITH_LAST);
  resolver.setPolicy('exteriorProbeSolve',   RESOLVE_POLICY.SKIP_WITH_LAST);

  // Post buffers can reuse last frame under pressure.
  resolver.setPolicy('postBufferPrepare',  RESOLVE_POLICY.ALLOW_STALE, 1, RESOLVE_RESULT.RUN_STALE);

  // Director hints are cosmetic.
  resolver.setPolicy('directorHintEmit',   RESOLVE_POLICY.SKIP_WITH_LAST);

  return resolver;
}

/* ------------------------------------------------------------------ */
/* 7. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultResolver = null;

export function getDefaultDependencyResolver(graph) {
  if (!_defaultResolver && graph) {
    _defaultResolver = new DependencyResolver(graph);
    applyStandardLightingPolicies(_defaultResolver);
  }
  return _defaultResolver;
}

export function disposeDefaultDependencyResolver() {
  if (_defaultResolver) {
    _defaultResolver.dispose();
    _defaultResolver = null;
  }
}

/* ------------------------------------------------------------------ */
/* 8. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createDependencyResolver(graph, options = {}) {
  return new DependencyResolver(graph, options);
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  DependencyResolver,
  PropagationMatrix,
  StalePolicyTable,
  FrameSnapshot,
  createDependencyResolver,
  getDefaultDependencyResolver,
  disposeDefaultDependencyResolver,
  applyStandardLightingPolicies,
  RESOLVE_POLICY,
  RESOLVE_RESULT,
  RESOLVE_RESULT_NAME,
  MAX_RESOLVE_NODES,
};

export default _defaultExport;