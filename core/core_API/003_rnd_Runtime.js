API Documentation — src/core/003_rnd_Runtime.js

File Purpose

This file is the low-level runtime container that sits between the App lifecycle layer (002_rnd_App.js) and the Bootstrap frame loop (001_rnd_Bootstrap.js). Where the Bootstrap owns the single requestAnimationFrame driver, the Runtime owns the deterministic scheduling primitives that every lighting subsystem uses to register, prioritize, and execute work each frame.

It owns four subsystems:

1. FrameScheduler — a fixed-cadence frame clock with hitch capping, frame-skip accounting, and a rolling frame-time EMA. This is the clock source for anything that needs to know how long the last frame took.
2. JobQueue — a fixed-capacity ring buffer of deferred jobs. Each job has a function, a context, a priority, and a state. The queue is drained with a bounded per-frame budget so no single frame can be starved by a long queue.
3. TaskGraph — a DAG of named tasks with static priorities and dependency edges. The graph is topologically sorted once at registration time and re-sorted only when the structure changes. Task execution order is deterministic and cached.
4. FrameBarrier — a double-buffered producer/consumer handoff. Producers write into the write buffer; consumers read from the read buffer. Commit swaps them atomically at the end of a frame, so parallel work in progress never tears the frame's visual output.

The Runtime class composes all four and exposes one tick(now) call that the Bootstrap invokes once per frame.

---

Exported Constants

MAX_JOBS

Type: number

Value: 4096 on HIGH tier, 2048 on MEDIUM, 1024 on LOW.

The fixed capacity of the job queue ring buffer. Sized to accommodate the worst-case burst of deferred lighting jobs (shadow atlas repack + GI probe bake + AO blur + cluster grid build) at peak scene complexity. No dynamic growth.

MAX_TASKS

Type: number

Value: 512

The fixed capacity of the task graph node array. Sized to accommodate every named task the engine will ever register, with headroom for future expansions.

MAX_BUDGETS

Type: number

Value: 16

The fixed capacity of the budget table. Each budget entry is a named per-frame time allocation (e.g. "shadows", "gi", "ao") that downstream systems can charge and query.

MAX_DT

Type: number

Value: 0.1

Frame-time hitch cap in seconds. Any dt greater than this is clamped.

FRAME_EMA_ALPHA

Type: number

Value: 0.15

Smoothing factor for the frame-time EMA in FrameScheduler.

JOB_EMA_ALPHA

Type: number

Value: 0.10

Smoothing factor for the job-drain-time EMA in Runtime.

JOB_STATE

Type: frozen enum

Values:

· EMPTY = 0
· QUEUED = 1
· RUNNING = 2
· DONE = 3
· FAILED = 4

TASK_STATE

Type: frozen enum

Values:

· IDLE = 0
· PENDING = 1
· READY = 2
· RUNNING = 3
· DONE = 4
· BLOCKED = 5

---

Exported Class — FrameScheduler

Constructor

```
new FrameScheduler(options = {})
```

Parameters:

· targetHz — desired frame rate. Default 60. Clamped to [15, 120].
· minHz — minimum allowed frame rate. Default 30. Clamped to [10, targetHz].
· hitchCap — max delta time in seconds. Default MAX_DT (0.1). Clamped to [0.016, 0.5].
· adaptive — boolean; if true, tracks actual Hz. Default true.

Instance Properties

· targetHz — the current target frame rate.
· minHz — the current minimum frame rate.
· hitchCap — the clamped hitch cap.
· adaptive — whether the scheduler tracks actual Hz.
· frameSkip — count of consecutive skipped frames (only meaningful when adaptive).
· skipAccum — accumulated frame time that has not yet crossed the target budget.
· frameTimeMs — the raw last-frame time in milliseconds.
· frameTimeEma — the smoothed last-frame time in milliseconds.
· actualHz — the measured effective frame rate.
· elapsed — total elapsed seconds since the last reset().
· frame — monotonic frame counter.

Instance Methods

tick(now)

Parameters: now — the current timestamp from requestAnimationFrame or performance.now.

Returns: the clamped delta time in seconds.

Purpose: updates the internal clock. Computes dt = (now - this._lastNow) / 1000, clamps to [0, hitchCap], updates frameTimeMs, frameTimeEma, actualHz, elapsed, and frame.

shouldRunThisFrame()

Returns: boolean.

Purpose: when adaptive is true, applies the frame-skip logic. Accumulates frameTimeEma into skipAccum; when the accumulator crosses the target budget (1000 / targetHz), resets the skip counter and returns true. Otherwise increments the skip counter and returns false.

When adaptive is false, always returns true.

setTargetHz(hz)

Parameters: hz — the new target frame rate.

Returns: nothing.

Purpose: updates targetHz and, if necessary, lowers minHz to stay consistent.

reset()

Returns: nothing.

Purpose: resets every counter — frame, elapsed, frameSkip, skipAccum, frameTimeEma — and re-anchors _lastNow to the current time.

---

Exported Class — JobQueue

Constructor

```
new JobQueue(capacity)
```

Parameters: capacity — the ring buffer size, typically MAX_JOBS.

Internal state:

· fn — an Array of job functions, one per slot.
· ctx — an Array of job contexts, one per slot.
· state — a Uint8Array of JOB_STATE values.
· priority — an Int16Array of priority values (higher = earlier).
· head, tail, count — ring buffer indices.
· _dirty — a flag indicating that slot state changed since the last _compact() call.

Instance Methods

enqueue(fn, ctx, priority = 0)

Parameters:

· fn — the function to run. Called as fn(ctx).
· ctx — the single argument passed to fn. Can be any value.
· priority — integer; higher values are picked first when the queue is drained.

Returns: the slot index on success, -1 if the queue is full even after compaction.

Purpose: adds a job to the tail of the ring. If the ring is full, calls _compact() first to reclaim empty slots. If still full, returns -1.

drain(maxJobs, onRun)

Parameters:

· maxJobs — the maximum number of jobs to run this call.
· onRun — optional error callback (error, slotIndex) => void.

Returns: the number of jobs executed.

Purpose: pulls up to maxJobs jobs in priority order (highest first) and runs each. Marks each slot RUNNING, runs, marks DONE or FAILED. Clears the slot's fn and ctx references so they can be garbage collected. Calls onRun on failure.

_pickNext()

Internal. Returns the slot index of the highest-priority queued job. Linear scan over the capacity. Early-exits if it finds a job at priority >= 1000.

_compact()

Internal. Rewrites the ring to remove holes created by executed jobs. Preserves order. Called only when enqueue cannot find a free slot.

clear()

Returns: nothing.

Purpose: resets every slot to EMPTY state and zeros the ring indices.

---

Exported Class — TaskGraph

Constructor

```
new TaskGraph(capacity)
```

Parameters: capacity — the fixed node capacity, typically MAX_TASKS.

Internal state:

· name — Array of task names.
· run — Array of (dt, elapsed, ctx) => void functions.
· state — Uint8Array of TASK_STATE values.
· priority — Int16Array.
· depCount — Uint16Array; number of parents per node.
· depStart — Uint32Array; unused reserved pointer.
· depEdges — Uint16Array of up to 4 dependency edges per node, encoded as parentIdx + 1 so 0 means empty.
· order — Uint16Array of the topological execution order.
· orderCount — the number of nodes in the topo order.
· count — the total number of registered nodes.
· _dirty — flag indicating the graph needs re-sorting.

Instance Methods

register(name, run, priority = 0)

Parameters:

· name — a unique string.
· run — the task function.
· priority — integer; lower values run earlier within the same topological depth.

Returns: the node index, or -1 if the graph is full.

Purpose: adds a task node. Sets _dirty so the next topoSort() call runs.

dependency(taskIdx, dependsOnIdx)

Parameters:

· taskIdx — the child node index.
· dependsOnIdx — the parent node index.

Returns: boolean.

Purpose: adds a dependency edge from taskIdx to dependsOnIdx, meaning taskIdx must run after dependsOnIdx. Each node can have up to four parents. Stores the parent index + 1 in the child's depEdges array so 0 can mean "empty".

topoSort()

Returns: boolean — true on success, false if a cycle was detected.

Purpose: runs Kahn's algorithm over the graph, producing a stable topological order in this.order. Detects cycles and warns if any node remains unresolved. Recomputes the order only when _dirty is true.

runAll(dt, elapsed, ctx)

Parameters:

· dt — delta seconds.
· elapsed — total elapsed seconds.
· ctx — an arbitrary context passed as the third argument to every task.

Returns: the number of tasks executed.

Purpose: iterates the topological order and calls each node's run(dt, elapsed, ctx). Wraps each call in try/catch. A failing node is marked BLOCKED and does not stop the rest of the graph.

clear()

Returns: nothing.

Purpose: resets every node and clears the order.

---

Exported Class — FrameBarrier

Constructor

```
new FrameBarrier()
```

Internal state:

· frameA — index of the current read buffer (0 or 1).
· frameB — index of the current write buffer (the other value).
· activeRead — temporary swap slot.
· pending — flag indicating that a commit is waiting.

Instance Methods

beginFrame()

Returns: nothing.

Purpose: clears the pending flag. Called at the start of every frame before producers begin writing.

markPending()

Returns: nothing.

Purpose: sets pending = true. Called by producers that have written new data into the write buffer.

commit()

Returns: nothing.

Purpose: if pending is true, swaps frameA and frameB via activeRead, then clears pending. After commit, the buffer that was being written becomes the buffer being read, and vice versa.

getReadBuffer()

Returns: the current read buffer index.

getWriteBuffer()

Returns: the current write buffer index.

---

Exported Class — BudgetTable

Constructor

```
new BudgetTable(capacity)
```

Parameters: capacity — typically MAX_BUDGETS.

Internal state:

· name — Array of budget names.
· limit — Float32Array of per-budget limits in milliseconds.
· used — Float32Array of per-budget used milliseconds this frame.
· count — number of defined budgets.

Instance Methods

define(name, limitMs)

Parameters:

· name — a string.
· limitMs — the per-frame budget in milliseconds.

Returns: the budget index, or -1 if full.

Purpose: creates a named budget entry.

reset()

Returns: nothing.

Purpose: zeros the used array. Called at the start of every frame.

add(idx, ms)

Parameters:

· idx — the budget index.
· ms — the milliseconds to charge.

Returns: nothing.

Purpose: adds to the used counter for that budget.

overBudget(idx)

Parameters: idx — the budget index.

Returns: boolean — true if used[idx] > limit[idx].

---

Exported Class — Runtime

Constructor

```
new Runtime(options = {})
```

Parameters:

· targetHz — default frame rate. Default 60.
· minHz — minimum frame rate. Default 30.
· maxJobsPerFrame — the drain budget. Default 64 on HIGH, 32 on MEDIUM, 16 on LOW.
· adaptive — passed to the scheduler. Default true.
· budgets — whether to define default budgets. Default true.
· runTaskGraph — whether the runtime should run the task graph each tick. Default true.

Constructor work:

1. Instantiates FrameScheduler, JobQueue, TaskGraph, FrameBarrier, BudgetTable.
2. Allocates _listeners (Map).
3. Allocates _ctx — a single reusable context object { scheduler, jobs, tasks, barrier, budgets, runtime }.
4. Calls _defineDefaultBudgets() which defines budgets for lights (4 ms), shadows (6 ms), gi (5 ms), ao (3 ms), environment (2 ms), and post (4 ms).

Instance Properties

· scheduler — the FrameScheduler.
· jobs — the JobQueue.
· tasks — the TaskGraph.
· barrier — the FrameBarrier.
· budgets — the BudgetTable.
· dt — the last clamped delta seconds.
· elapsed — total elapsed seconds.
· frame — the frame counter.
· isInitialized — boolean.
· context — the reusable context object.

Instance Methods

initialize()

Returns: this.

Purpose: calls assertBiteCSReady() to verify the ECS layer is up. Sets _initialized = true. Emits ready. Idempotent.

on(event, fn)

Returns: unsubscribe function.

Events: ready, frame, jobs, tasks.

off(event, fn)

Returns: nothing.

postJob(fn, ctx, priority = 0)

Returns: the slot index or -1.

Purpose: convenience wrapper around this.jobs.enqueue.

registerTask(name, fn, priority = 0)

Returns: the node index.

Purpose: convenience wrapper around this.tasks.register.

dependsOn(taskIdx, dependsOnIdx)

Returns: boolean.

Purpose: convenience wrapper around this.tasks.dependency.

tick(now)

Parameters: now — the rAF timestamp.

Returns: the clamped dt.

Purpose: the per-frame runtime step.

1. Calls this.scheduler.tick(now) to update the clock.
2. Updates this._dt, this._elapsed, this._frame.
3. Calls this.budgets.reset() and this.barrier.beginFrame().
4. If scheduler.shouldRunThisFrame() returns false, emits frame with the current info and returns.
5. If runTaskGraph is true and there are registered tasks, times and runs this.tasks.runAll(dt, elapsed, this._ctx).
6. If there are queued jobs, times and drains up to maxJobsPerFrame.
7. Calls this.barrier.commit() to swap the double buffers.
8. Emits frame, jobs, tasks.

_frameInfo(jobsRun, tasksRun)

Internal. Returns a fresh object summarizing the frame — used only by event emission, so the allocation is amortized against the rare emission cadence.

budgetAdd(idx, ms)

Returns: nothing.

budgetOver(idx)

Returns: boolean.

clear()

Returns: nothing.

Purpose: clears the job queue, task graph, and scheduler clock.

dispose()

Returns: nothing.

Purpose: clears state and empties the listener map.

---

Exported Functions

getDefaultRuntime()

Returns: the module-level singleton Runtime, creating it on first call.

Purpose: the runtime that 001_rnd_Bootstrap.js and 004_rnd_EngineLoop.js share.

disposeDefaultRuntime()

Returns: nothing.

Purpose: disposes and clears the singleton.

createRuntime(options = {})

Parameters: same as the Runtime constructor.

Returns: a fresh Runtime.

_now()

Internal helper returning the current high-resolution timestamp.

---

Default Export

The default export bundles every named export: Runtime, FrameScheduler, JobQueue, TaskGraph, FrameBarrier, BudgetTable, createRuntime, getDefaultRuntime, disposeDefaultRuntime, JOB_STATE, TASK_STATE, MAX_JOBS, MAX_TASKS, MAX_BUDGETS.

---

Usage Pattern

Inside 004_rnd_EngineLoop.js, the runtime is driven once per frame:

```
import { getDefaultRuntime } from './003_rnd_Runtime.js';

const runtime = getDefaultRuntime();
runtime.initialize();

// Inside the rAF callback:
runtime.tick(now);
```

Any lighting subsystem that wants to defer work:

```
runtime.postJob((ctx) => {
  // heavy work
}, myContext, 500);
```

Any subsystem that wants a per-frame task with dependencies:

```
const a = runtime.registerTask('buildLightList', () => {...});
const b = runtime.registerTask('packShadowAtlas', () => {...});
runtime.dependsOn(b, a);
```

The runtime guarantees that packShadowAtlas runs after buildLightList every frame, with no allocations, no closures, and no Map lookups on the hot path.

