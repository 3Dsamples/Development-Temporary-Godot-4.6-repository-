API Documentation — src/core/009_rnd_DependencyResolver.js

File Purpose

This file is the dependency resolver for the anime lighting pipeline. It sits between the task graph (008_rnd_TaskGraph.js) and the job queue (007_rnd_JobQueue.js) and answers three questions every lighting subsystem asks each frame:

1. Which of my upstream lighting nodes actually produced fresh data this frame, and which are stale?
2. Given what changed upstream, what is the minimal set of downstream nodes that must re-run this frame?
3. If I skip, what is the visual consequence — can I safely substitute last-frame output of my parents (the ALLOW_STALE policy), or must I block until upstream finishes?

The resolver maintains four things:

· A canonical per-frame snapshot of node generations. Downstream nodes compare "was my parent's commit generation newer than my last read?" with one integer compare.
· A dirty-propagation table computed once at registration. Given a set of changed nodes, the table returns which downstream nodes become dirty. This is a bit-set of size MAX_NODES per node, stored as a flat Uint32Array for cache locality.
· A stale-substitution policy table. Per node, resolving whether ALLOW_STALE, BLOCK, or SKIP_WITH_DEFAULT is the correct behavior when an upstream is stale.
· A cost-priority ranking. Critical path distance multiplied by node cost EMA so the frame's budget is spent on the nodes that matter most visually.

The resolver exists to eliminate two of the most common Android lighting bugs:

· Updating GI from a stale shadow atlas, causing light leaks.
· Skipping AO when the shadow map just changed, causing hard edges on cel shading.

Both are corrected by explicitly making each node declare whether it can tolerate stale input, and by tracking generations so the graph can refuse to run nodes that would produce visibly wrong output.

---

Exported Constants

PERF_TIER

Type: string

Value: 'LOW' | 'MEDIUM' | 'HIGH' — cached from getPerfTier() at module load.

MAX_RESOLVE_NODES

Type: number

Value: MAX_NODES from 008_rnd_TaskGraph.js.

The fixed capacity of the resolver's internal tables. Matches the task graph's node capacity so the two are always dimensionally consistent.

RESOLVE_POLICY

Type: frozen enum

Policies that override a node's default stale-substitution behavior.

· DEFAULT = 0 — use the node's flag bits to decide.
· FORCE_FRESH = 1 — never accept stale; must block or skip.
· ALLOW_STALE = 2 — accept stale freely.
· SKIP_WITH_LAST = 3 — skip and reuse last-frame output.
· CRITICAL_ONLY = 4 — only run if the node is on the critical path.

RESOLVE_RESULT

Type: frozen enum

The decision the resolver emits per node per frame.

· RUN = 0 — run with fresh parents.
· RUN_STALE = 1 — run with one or more stale parents, permitted by policy.
· SKIP = 2 — skip this frame; parents' data is not ready or the domain did not fire.
· BLOCK = 3 — upstream was not ready and stale is not permitted.
· FORCE = 4 — the node was force-promoted by starvation.

RESOLVE_RESULT_NAME

Type: frozen array

Values: ['run', 'run_stale', 'skip', 'block', 'force'].

WORD_BITS

Internal constant, value 32. The number of bits in a Uint32Array word.

WORDS_PER_ROW

Internal constant, computed from MAX_RESOLVE_NODES and WORD_BITS. The number of words needed to represent one bit row of the propagation matrix.

---

Exported Class — PropagationMatrix

A dense bit matrix where row i holds every transitive downstream dependent of node i. Used to compute, in one pass, which nodes must be invalidated when a set of nodes changes.

Constructor

```
new PropagationMatrix(nodeCount)
```

Parameters: nodeCount — the number of task graph nodes.

Instance Properties

· nodeCount — the number of rows.
· wordsPerRow — the number of Uint32Array words per row.
· data — a flat Uint32Array(nodeCount * wordsPerRow) holding all the bits.

Instance Methods

clear()

Returns: nothing.

Purpose: zeroes the entire data array.

setDependent(nodeIdx, downstreamIdx)

Parameters:

· nodeIdx — the source node's index.
· downstreamIdx — the transitive dependent's index.

Returns: nothing.

Purpose: sets a single bit in the matrix. Used during setup only.

isDependent(nodeIdx, downstreamIdx)

Parameters: same as setDependent.

Returns: boolean.

computeClosure(graph)

Parameters: graph — a TaskGraph instance.

Returns: boolean — true on success, false if the graph has no topological order yet.

Purpose: fills the entire matrix by walking the graph in reverse topological order. For each node, ORs together the closure rows of its direct children plus the direct children themselves. After this call, row i contains every transitive dependent of node i.

Because the walk is reverse-topological, each child's closure is fully computed before it is ORed into its parent's row. This makes the whole computation a single linear pass.

---

Exported Class — StalePolicyTable

A per-node table of stale-substitution policies plus per-node maximum staleness limits.

Constructor

```
new StalePolicyTable(nodeCount)
```

Parameters: nodeCount — the number of task graph nodes.

Instance Properties

· nodeCount — the number of entries.
· policy — a Uint8Array(nodeCount) of RESOLVE_POLICY values.
· maxStaleFrames — a Uint16Array(nodeCount) where 0 means unlimited.
· defaultResult — a Uint8Array(nodeCount) of fallback RESOLVE_RESULT values.

Instance Methods

set(nodeIdx, policy, maxStaleFrames = 0, defaultResult = RESOLVE_RESULT.SKIP)

Parameters:

· nodeIdx — the node's index.
· policy — one of RESOLVE_POLICY.
· maxStaleFrames — the maximum number of stale frames permitted before the node must re-run.
· defaultResult — the result to fall back to when the policy cannot resolve.

Returns: boolean.

get(nodeIdx)

Parameters: nodeIdx — the node's index.

Returns: the RESOLVE_POLICY value.

clear()

Returns: nothing. Zeroes the three arrays.

---

Exported Class — FrameSnapshot

Per-node bookkeeping for the current frame and the previous frame. Used by the resolver to know whether a parent committed fresh data and whether the consumer has already read it.

Constructor

```
new FrameSnapshot(nodeCount)
```

Parameters: nodeCount — the number of nodes.

Instance Properties

· nodeCount — the number of entries.
· lastCommitGen — a Uint32Array(nodeCount) of the most recent commit generation per node. 0 means never committed.
· lastRunFrame — a Uint32Array(nodeCount) of the frame number when the node last ran.
· lastReadGen — a Uint32Array(nodeCount) of the generation the consumer last read.
· dirty — a Uint8Array(nodeCount) of dirty flags.
· staleFrames — a Uint16Array(nodeCount) of consecutive stale frames.

Instance Methods

clear()

Returns: nothing. Zeroes all arrays.

bumpCommit(nodeIdx, frame)

Parameters:

· nodeIdx — the node's index.
· frame — the current frame number.

Returns: the new generation (a 32-bit integer).

Purpose: increments the node's commit generation, records the frame, sets the dirty flag, and clears the stale counter. Called by producers when they commit fresh output.

markRead(nodeIdx)

Parameters: nodeIdx — the node's index.

Returns: nothing.

Purpose: records that a consumer has read the current generation, clearing the dirty flag.

isStaleFor(nodeIdx)

Parameters: nodeIdx — the node's index.

Returns: boolean.

Purpose: true if the last read generation differs from the last commit generation, meaning the consumer's view is stale.

tickStale()

Returns: nothing.

Purpose: ages every node's stale counter. Nodes that have never committed are marked with 0xFFFF, indicating unbounded staleness.

---

Exported Class — DependencyResolver

The main resolver.

Constructor

```
new DependencyResolver(graph, options = {})
```

Parameters:

· graph — a TaskGraph instance. Required.
· options.useStaleByDefault — if true, nodes without an explicit policy are allowed to run stale. Default true.
· options.maxStaleFrames — a global fallback limit for maxStaleFrames. Default 0 (unlimited).
· options.propagateDirty — reserved for future expansion. Default true.
· options.evaluateCriticalPath — reserved for future expansion. Default true.

Constructor work:

1. Stores the graph and its capacity.
2. Allocates propagation — a PropagationMatrix(capacity).
3. Allocates policy — a StalePolicyTable(capacity).
4. Allocates snapshot — a FrameSnapshot(capacity).
5. Allocates the per-frame scratch buffers _dirtyMask, _resolveResults, and _resolveOrder.
6. Allocates the cost ranking arrays _costEma and _visualWeight (both Float32Array).
7. Initializes stats counters.
8. Allocates _listeners (Map).
9. Sets _closureBuilt = false.

Instance Properties

· graph — the task graph the resolver is attached to.
· capacity — the node capacity.
· propagation — the propagation matrix.
· policy — the stale policy table.
· snapshot — the frame snapshot.
· frame — the frame counter.
· stats — counters: totalRuns, totalStaleRuns, totalSkips, totalBlocks, totalForces.

Instance Methods

on(event, fn)

Parameters:

· event — one of 'closure', 'commit', 'resolved'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

rebuildClosure()

Returns: boolean.

Purpose: re-runs graph.topoSort() if needed, then calls propagation.computeClosure(graph). Sets _closureBuilt = true. Emits closure.

setPolicy(nodeOrIdx, policy, maxStaleFrames = 0, defaultResult = RESOLVE_RESULT.SKIP)

Parameters:

· nodeOrIdx — the node name string or index.
· policy — one of RESOLVE_POLICY.
· maxStaleFrames — the maximum stale frames permitted.
· defaultResult — the fallback result.

Returns: boolean.

setVisualWeight(nodeOrIdx, weight)

Parameters:

· nodeOrIdx — the node name string or index.
· weight — a [0, 1] value indicating how visually important this node's output is.

Returns: boolean.

Purpose: feeds the cost-priority ranking. Critical nodes with high visual weight are preferred for re-running when there is a budget shortfall.

_resolveIdx(nameOrIdx)

Internal. Resolves a name or index to a node index.

markCommitted(nodeOrIdx)

Parameters: nodeOrIdx — the node name string or index.

Returns: the new commit generation, or 0.

Purpose: called by producers after they finish producing fresh output. Bumps the node's commit generation, records the dirty bit in _dirtyMask, emits commit.

markRead(nodeOrIdx)

Parameters: nodeOrIdx — the node name string or index.

Returns: nothing.

Purpose: records that the consumer read the current generation.

isStale(nodeOrIdx)

Parameters: nodeOrIdx — the node name string or index.

Returns: boolean.

resolve(dt, elapsed, domainFiredMask)

Parameters:

· dt — delta seconds (unused, reserved for future expansion).
· elapsed — total elapsed seconds (unused, reserved for future expansion).
· domainFiredMask — a bitmask from the FrameScheduler.

Returns: the number of nodes to run this frame.

Purpose: the main entry point. Called once per frame by EngineLoop.

Flow:

1. Increments frame.
2. Rebuilds the closure if the graph changed or if the closure has not been built yet.
3. Zeroes the scratch mask.
4. Ages the snapshot stale counters.
5. Walks the graph's topological order. For each node:
   · Checks whether the node was force-promoted by starvation.
   · Checks whether the node's domain fired.
   · Iterates the node's parents, evaluating their state: DONE means ready; SKIPPED is acceptable if the policy allows stale; FAILED or BLOCKED means the node is blocked.
   · Resolves a preliminary result: BLOCK, SKIP, FORCE, RUN_STALE, or RUN.
   · Applies the stale policy override.
   · Writes the result into _resolveResults, appends the node index to _resolveOrder if it will run.
6. Accumulates totals into stats.
7. Emits resolved.
8. Returns the count.

_canAcceptStale(nodeIdx, parentIdx)

Internal. Evaluates the node's policy and flags to decide whether it can run when a parent was skipped.

propagateDirty(sources, outMask)

Parameters:

· sources — an array of node indices that changed.
· outMask — a Uint32Array(WORDS_PER_ROW) to write the resulting bitmask into.

Returns: the number of bits set in outMask.

Purpose: uses the propagation matrix to compute the transitive closure of dirty nodes in one pass. This is how a subsystem that knows "the biome changed" gets a single bitmask telling it exactly which downstream lighting nodes need to re-run.

isCritical(nodeOrIdx)

Parameters: nodeOrIdx — the node name string or index.

Returns: boolean.

getCriticalPathLength()

Returns: the number of nodes on the critical path.

updateCost(nodeOrIdx, ms)

Parameters:

· nodeOrIdx — the node name string or index.
· ms — the cost in milliseconds.

Returns: nothing.

Purpose: updates the node's cost EMA for the priority ranking.

visualPriority(nodeOrIdx)

Parameters: nodeOrIdx — the node name string or index.

Returns: a numeric score combining the node's visual weight, its topological depth bonus, and its critical path bonus.

reset()

Returns: this. Resets the snapshot, policy, and every counter.

dispose()

Returns: this. Resets and nulls every internal array.

getResolveResult(nodeOrIdx)

Parameters: nodeOrIdx — the node name string or index.

Returns: the last RESOLVE_RESULT for that node.

getResolveOrder()

Returns: { order: Int32Array, count: number }.

Purpose: exposes the ordered list of nodes that will run this frame, in topological order.

getStats()

Returns: an object with frame, resolveCount, the five totals from stats, closureBuilt, a per-node array with name, policy, result, commitGen, readGen, isStale, staleFrames, costEma, visualWeight, isCritical, and perfTier.

---

Exported Function — applyStandardLightingPolicies(resolver)

Parameters: resolver — a DependencyResolver instance.

Returns: the same resolver.

Purpose: installs the canonical stale policies for the eleven-node standard lighting pipeline. The policies are:

· shadowAtlasPack — FORCE_FRESH. Shadow maps must never be stale.
· cascadeMatrixSolve — FORCE_FRESH.
· lightListBuild — FORCE_FRESH.
· giProbeBake — ALLOW_STALE with maxStaleFrames = 1 and default result RUN_STALE.
· aoBlur — ALLOW_STALE with maxStaleFrames = 2.
· clusterGridBuild — FORCE_FRESH.
· envPaletteSolve — SKIP_WITH_LAST.
· interiorVolumeUpdate — SKIP_WITH_LAST.
· exteriorProbeSolve — SKIP_WITH_LAST.
· postBufferPrepare — ALLOW_STALE with maxStaleFrames = 1.
· directorHintEmit — SKIP_WITH_LAST.

The policy choices reflect the visual consequences of staleness:

· Shadow maps, cluster grid, and light list have hard visual thresholds: a stale frame causes popping.
· GI and AO blur can tolerate one or two frames of staleness without visible artifacts.
· Environment palette, interior volumes, exterior probes, and director hints are soft: a stale frame is invisible.

---

Exported Functions

getDefaultDependencyResolver(graph)

Parameters: graph — a TaskGraph. Required on first call.

Returns: the module-level singleton DependencyResolver, creating it on first call and calling applyStandardLightingPolicies() on it.

disposeDefaultDependencyResolver()

Returns: nothing.

createDependencyResolver(graph, options = {})

Returns: a new DependencyResolver.

---

Default Export

The default export bundles: DependencyResolver, PropagationMatrix, StalePolicyTable, FrameSnapshot, createDependencyResolver, getDefaultDependencyResolver, disposeDefaultDependencyResolver, applyStandardLightingPolicies, RESOLVE_POLICY, RESOLVE_RESULT, RESOLVE_RESULT_NAME, MAX_RESOLVE_NODES.

---

Usage Pattern

The EngineLoop calls resolver.resolve(dt, elapsed, scheduler.domainsFiredMask) each frame and then walks getResolveOrder() to run the ready nodes:

```
import { getDefaultDependencyResolver } from './src/core/009_rnd_DependencyResolver.js';

const resolver = getDefaultDependencyResolver(graph);

// Per frame:
const runCount = resolver.resolve(dt, elapsed, scheduler.domainsFiredMask);
const { order, count } = resolver.getResolveOrder();

for (let i = 0; i < count; i++) {
  const nodeIdx = order[i];
  const node = graph.nodes[nodeIdx];
  const result = resolver.getResolveResult(nodeIdx);
  node.run(dt, elapsed, node, graph);
  if (result === RESOLVE_RESULT.RUN_STALE) {
    // This node ran on stale parents; the shader should use the
    // last committed generations of any uniform blocks it consumes.
  }
  resolver.markCommitted(nodeIdx);
}
```

A subsystem that knows a specific node's output changed:

```
resolver.markCommitted('giProbeBake');

const dirtyMask = new Uint32Array(WORDS_PER_ROW);
const count = resolver.propagateDirty(['giProbeBake'], dirtyMask);
// dirtyMask now contains every transitive dependent of giProbeBake.
```

The resolver is what allows the entire lighting pipeline to be scheduled at different frequencies without visible tearing — each node declares its own tolerance for stale input, and the resolver enforces that tolerance every frame.

