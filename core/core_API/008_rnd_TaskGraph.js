API Documentation — src/core/008_rnd_TaskGraph.js

File Purpose

This file is the directed acyclic task graph for the anime lighting stack. Where 007_rnd_JobQueue.js handles individual job execution with priority and dependencies, this module builds and executes the HIGH-LEVEL task pipeline that every frame must run.

The graph declares the canonical lighting pipeline as a DAG (directed acyclic graph) of named nodes, each of which is a function that runs at a specific frequency domain. It is the scheduling layer ABOVE the job queue — it defines which lighting stages must run this frame, in what order, with which dependencies, at which frequency, and whether they can be parallelized. The JobQueue then takes the ready nodes and dispatches them.

The standard pipeline the graph knows about:

```
lightListBuild
  ├── shadowAtlasPack
  │     └── cascadeMatrixSolve
  ├── giProbeBake
  │     └── aoBlur
  ├── clusterGridBuild
  └── envPaletteSolve
        ├── interiorVolumeUpdate
        └── exteriorProbeSolve
              └── postBufferPrepare
                    └── directorHintEmit
```

Eleven nodes in total. The graph guarantees that when lightListBuild runs, every downstream node that depends on it sees the freshly-built list.

The graph does six things a plain array of callbacks cannot:

1. Explicit dependency edges — up to eight parents per node. A node cannot run until all its parents have finished, or their output has been explicitly marked as stale.
2. Kahn topological sort with cycle detection — cycles in a lighting pipeline are logic bugs, so the graph fails loudly instead of silently misbehaving.
3. Per-node frequency domain binding — each node binds to one of the ten scheduler domains (SIMULATION / LIGHTS / SHADOWS / GI / AO / ENVIRONMENT / INTERIOR / EXTERIOR / DIRECTOR / POST). A node only runs on frames where its domain fired.
4. Skip propagation — when an upstream node is skipped because its domain did not fire, downstream nodes see the skip and decide whether to run on stale data or also skip. The graph is a data-availability graph, not just a scheduling graph.
5. Parallel execution groups — nodes at the same topological depth with no cross-dependencies are flagged as parallel-safe so the JobQueue can dispatch them in a single batch.
6. Critical path tracking — the graph records the longest chain through the DAG so adaptive quality controllers can throttle the shallowest nodes first, preserving the critical path's visual output.

Plus a starvation guard: a node that has not run for N frames is force-promoted to run before any non-critical node.

---

Exported Constants

PERF_TIER

Type: string

Value: 'LOW' | 'MEDIUM' | 'HIGH' — cached from getPerfTier() at module load.

MAX_NODES

Type: number

Value: 256 on HIGH, 192 on MEDIUM, 128 on LOW.

The fixed capacity of the graph. Sized to accommodate every named task the engine will ever register, with headroom for future expansions.

MAX_EDGES

Type: number

Value: MAX_NODES * 8.

The fixed capacity of the dependency edge table. Each node can have up to eight parents.

TASK_NODE_STATE

Type: frozen enum

Values:

· IDLE = 0 — node has not run this frame.
· PENDING = 1 — node is waiting for its dependencies.
· READY = 2 — node is ready but not yet running.
· RUNNING = 3 — node is currently executing.
· DONE = 4 — node finished successfully this frame.
· SKIPPED = 5 — node was skipped because its domain did not fire or a parent was stale.
· BLOCKED = 6 — node could not run because a parent failed.
· FAILED = 7 — node ran but threw an error.

TASK_NODE_STATE_NAME

Type: frozen array

Values: ['idle', 'pending', 'ready', 'running', 'done', 'skipped', 'blocked', 'failed'].

TASK_FLAGS

Type: frozen object

Bit flags for per-node behavior.

· NONE = 0 — no flags.
· PARALLEL_SAFE = 1 — node can run in parallel with siblings at the same depth.
· CRITICAL_PATH = 2 — node is on the critical path (computed automatically).
· ALLOW_STALE = 4 — node may run on last-frame output of its parents.
· FORCE_EVERY_FRAME = 8 — node ignores its domain cadence and runs every frame.
· STARVATION_EXEMPT = 16 — node is never auto-promoted by the starvation guard.

STARVATION_FRAMES

Type: number

Value: 60

A node that has not run for this many consecutive frames has FORCE_EVERY_FRAME set on it during tick(), forcing it to run regardless of its domain cadence.

---

Exported Class — TaskNode

One instance per registered node.

Constructor

```
new TaskNode(index, name)
```

Parameters:

· index — the node's array index.
· name — the node's name string.

Instance Properties

· index — the node's array index.
· name — the node's name string.
· state — one of TASK_NODE_STATE.
· domain — the scheduler domain this node belongs to (from DOMAIN in 005_rnd_FrameScheduler.js).
· flags — the node's flag bitmask.
· priority — the node's priority within its topological depth. Lower runs earlier.
· run — the node's function (dt, elapsed, node, graph) => void.
· ctx — an arbitrary context object passed to run.
· parents — an Int32Array(8) of parent node indices. -1 means empty.
· parentCount — the number of used entries in parents.
· children — an Int32Array(16) of child node indices. -1 means empty.
· childCount — the number of used entries in children.
· depth — the node's topological depth (computed by topoSort).
· idleFrames — the number of consecutive frames this node has not run.
· lastRunMs — the cost of the last run in milliseconds.
· lastRunEma — the smoothed cost in milliseconds.
· peakMs — the highest cost ever recorded.
· runCount — the total number of times this node has run.
· upstreamSkipped — the number of parents that were SKIPPED this frame.
· upstreamStale — the number of parents that ran on stale data this frame.
· _invoke — the pre-bound invocation function (same as run).
· _invokeCtx — the pre-bound context (same as ctx). Pre-bound to avoid closure allocation per frame.

Instance Methods

reset()

Returns: this.

Purpose: zeroes every field, resets parents and children to all -1, and clears run and ctx.

---

Exported Class — TopoOrder

A small container for the topological order plus depth information.

Constructor

```
new TopoOrder(capacity)
```

Parameters: capacity — the maximum number of nodes.

Instance Properties

· capacity — the fixed capacity.
· indices — an Int32Array(capacity) of node indices in topological order.
· depth — an Int32Array(capacity) of the topological depth of each node.
· count — the number of valid entries.
· maxDepth — the maximum depth encountered.

Instance Methods

reset()

Returns: nothing. Zeroes the count and max depth.

---

Exported Class — TaskGraph

The main graph.

Constructor

```
new TaskGraph(options = {})
```

Parameters:

· capacity — the maximum number of nodes. Default MAX_NODES.
· autoSort — whether to re-sort topologically on the next tick after a structural change. Default true.
· starvePromote — whether to auto-promote starved nodes. Default true.
· starvationFrames — the number of consecutive frames without a run before the node is force-promoted. Default STARVATION_FRAMES.
· trackCriticalPath — whether to compute the critical path on topoSort(). Default true.
· propagateSkip — whether to propagate skip from parents to children. Default true.

Constructor work:

1. Allocates the nodes array of capacity fresh TaskNode instances.
2. Allocates nameIndex (a Map) for name-to-index resolution.
3. Allocates order — a fresh TopoOrder sized to the capacity.
4. Allocates depthGroups — an array of capacity arrays, one per depth level.
5. Allocates criticalPath (an Int32Array(capacity)) and criticalPathLen.
6. Allocates the per-frame scratch arrays _indeg, _queue, and _readyBatch (all Int32Array or Uint16Array of capacity).
7. Allocates _listeners (Map).
8. Allocates _invokeArgs — a single reusable object with fields { dt, elapsed, node, graph }.
9. Sets _dirty = true so the first tick triggers a sort.

Instance Properties

· nodes — the array of TaskNode instances.
· nodeCount — the number of registered nodes.
· nameIndex — the Map from name to index.
· order — the TopoOrder instance.
· depthGroups — the array of per-depth node arrays.
· depthGroupCount — the number of populated depth levels.
· criticalPath — the Int32Array of node indices on the critical path.
· criticalPathLen — the number of nodes on the critical path.
· frame — the frame counter.
· executedLast — the number of nodes executed in the last tick.
· skippedLast — the number of nodes skipped in the last tick.
· failedLast — the number of nodes that failed in the last tick.

Instance Methods

on(event, fn)

Parameters:

· event — one of 'registered', 'sorted', 'criticalpath', 'ran', 'failed', 'tick'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

registerNode(name, run, options = {})

Parameters:

· name — a unique string.
· run — the node's function (dt, elapsed, node, graph) => void.
· options.ctx — an optional context object.
· options.domain — the scheduler domain. Default DOMAIN.SIMULATION.
· options.priority — the priority within the node's topological depth. Default 100.
· options.flags — the flag bitmask. Default TASK_FLAGS.NONE.

Returns: the node index, or -1 on failure.

Purpose: registers a new node. If the name is already registered, returns the existing index. Sets _dirty = true. Pre-binds _invoke and _invokeCtx on the node so no closures are created per frame. Emits registered.

addDependency(childNameOrIdx, parentNameOrIdx)

Parameters:

· childNameOrIdx — the child node's name string or index.
· parentNameOrIdx — the parent node's name string or index.

Returns: boolean.

Purpose: adds a dependency edge from the child to the parent, meaning the child runs after the parent. Verifies the child has not exceeded eight parents and the parent has not exceeded sixteen children. Sets _dirty = true.

_resolveIdx(nameOrIdx)

Internal. Resolves a name or integer to a node index. Returns -1 if the name is unknown or the index is out of range.

getNode(nameOrIdx)

Parameters: a name string or index.

Returns: the TaskNode, or null.

topoSort()

Returns: boolean — true on success, false if a cycle was detected.

Purpose: runs Kahn's algorithm over the graph.

Flow:

1. If _dirty is false and the order count matches nodeCount, returns true immediately.
2. Resets _indeg for every node.
3. Computes in-degree counts by iterating every node's parents array.
4. Initializes the _queue with nodes whose in-degree is zero.
5. Runs the Kahn loop: pops a node, appends it to order.indices, records its depth, decrements the in-degrees of its children, and enqueues children whose in-degree reaches zero.
6. If the write count does not reach nodeCount, a cycle exists. Logs an error, sets _dirty = false, and returns false.
7. Otherwise, writes order.count and order.maxDepth, rebuilds the depthGroups arrays, computes the critical path if trackCriticalPath is on, sets _dirty = false, and emits sorted.

_computeCriticalPath()

Internal. Uses a variation of the longest-path algorithm to find the chain of nodes with the greatest number of edges from root to leaf. Marks each node on the chain with TASK_FLAGS.CRITICAL_PATH. Emits criticalpath.

tick(dt, elapsed, domainFiredMask)

Parameters:

· dt — delta seconds.
· elapsed — total elapsed seconds.
· domainFiredMask — a bitmask of domains that fired this frame (from the FrameScheduler's domainsFiredMask).

Returns: the number of nodes executed.

Purpose: the main per-frame entry point.

Flow:

1. Increments frame.
2. Re-runs topoSort() if _dirty is set and autoSort is on.
3. Resets per-frame state: upstreamSkipped and upstreamStale on every node, and increments idleFrames on every node.
4. If starvePromote is on, iterates every node and sets FORCE_EVERY_FRAME on any node whose idleFrames >= starvationFrames, unless the node is STARVATION_EXEMPT.
5. Walks the topological order. For each node:
   · Checks whether its domain fired. A node with FORCE_EVERY_FRAME skips this check.
   · Checks each parent's state. If a parent is DONE, continues. If a parent is SKIPPED and the node is ALLOW_STALE, increments upstreamSkipped and continues. Otherwise, the node is not ready.
   · If the domain did not fire or a parent is not ready, marks the node SKIPPED and continues.
   · Otherwise, times and runs _invoke(dt, elapsed, node, this), updates timing EMA, increments runCount, resets idleFrames, marks the node DONE or FAILED.
6. Stores the totals in executedLast, skippedLast, failedLast.
7. Emits tick with a summary.
8. Returns the executed count.

getDepthGroups()

Returns: { groups: depthGroups, count: depthGroupCount }.

Purpose: exposes the depth groups so a caller can iterate them and dispatch every node in a depth level as a single parallel batch.

forEachInDepth(depth, fn, ctx)

Parameters:

· depth — the depth level.
· fn — a callback (node) => void.
· ctx — the context passed to fn.

Returns: the number of nodes visited.

resetStates()

Returns: this.

Purpose: sets every node's state to IDLE and zeroes its idleFrames.

clear()

Returns: this.

Purpose: resets every node, clears the name map, resets the topo order, and clears every depth group.

dispose()

Returns: this.

Purpose: clears and empties the listener map.

getStats()

Returns: an object with frame, nodeCount, capacity, maxDepth, depthGroups, criticalPathLen, executedLast, skippedLast, failedLast, a nodes array with per-node stats, and perfTier.

---

Exported Function — buildStandardLightingPipeline(graph, callbacks = {})

Parameters:

· graph — a TaskGraph instance.
· callbacks — an object mapping pipeline node names to functions.

Recognized callback keys (all optional; missing ones default to no-op):

· lightListBuild
· shadowAtlasPack
· cascadeMatrixSolve
· giProbeBake
· aoBlur
· clusterGridBuild
· envPaletteSolve
· interiorVolumeUpdate
· exteriorProbeSolve
· postBufferPrepare
· directorHintEmit

Returns: an object mapping the pipeline node names to their TaskNode instances.

Purpose: builds the standard eleven-node lighting pipeline in a graph. Registers every node with the appropriate domain and priority, wires the dependency edges, and topologically sorts the graph once.

The wiring is:

· shadowAtlasPack depends on lightListBuild
· cascadeMatrixSolve depends on shadowAtlasPack
· giProbeBake depends on lightListBuild
· aoBlur depends on giProbeBake
· clusterGridBuild depends on lightListBuild
· envPaletteSolve depends on lightListBuild
· interiorVolumeUpdate depends on envPaletteSolve
· exteriorProbeSolve depends on envPaletteSolve
· postBufferPrepare depends on exteriorProbeSolve
· directorHintEmit depends on postBufferPrepare

The domain assignments:

· lightListBuild — LIGHTS
· clusterGridBuild — LIGHTS
· shadowAtlasPack — SHADOWS
· cascadeMatrixSolve — SHADOWS
· giProbeBake — GI
· aoBlur — AO
· envPaletteSolve — ENVIRONMENT
· interiorVolumeUpdate — INTERIOR
· exteriorProbeSolve — EXTERIOR
· postBufferPrepare — POST
· directorHintEmit — DIRECTOR

The flags: lightListBuild, shadowAtlasPack, and clusterGridBuild are marked PARALLEL_SAFE. giProbeBake, aoBlur, and envPaletteSolve are marked ALLOW_STALE.

---

Exported Functions

getDefaultTaskGraph()

Returns: the module-level singleton TaskGraph, creating it on first call and calling buildStandardLightingPipeline() on it.

disposeDefaultTaskGraph()

Returns: nothing.

createTaskGraph(options = {})

Returns: a new TaskGraph.

---

Default Export

The default export bundles: TaskGraph, TaskNode, TopoOrder, createTaskGraph, getDefaultTaskGraph, disposeDefaultTaskGraph, buildStandardLightingPipeline, TASK_NODE_STATE, TASK_NODE_STATE_NAME, TASK_FLAGS, MAX_NODES, MAX_EDGES.

---

Usage Pattern

A lighting subsystem registers its work as nodes:

```
import { getDefaultTaskGraph, TASK_FLAGS } from './src/core/008_rnd_TaskGraph.js';
import { DOMAIN } from './src/core/005_rnd_FrameScheduler.js';

const graph = getDefaultTaskGraph();

graph.registerNode('myCustomLightStage', (dt, elapsed, node, g) => {
  // per-frame work
}, {
  domain:   DOMAIN.LIGHTS,
  priority: 5,
  flags:    TASK_FLAGS.PARALLEL_SAFE,
});

graph.addDependency('myCustomLightStage', 'lightListBuild');
```

Each frame, the scheduler calls graph.tick(dt, elapsed, scheduler.domainsFiredMask) and the graph runs every node whose domain fired and whose parents are resolved.

Because the graph knows the topological depth of every node, it can also dispatch parallel batches: forEachInDepth(0, (node) => dispatch(node)) runs every root node at once, then forEachInDepth(1, ...) runs every depth-1 node, and so on. On a Web Worker pool, this is what makes the parallel lighting pipeline safe.

