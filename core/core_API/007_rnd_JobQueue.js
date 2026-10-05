API Documentation — src/core/007_rnd_JobQueue.js

File Purpose

This file is the dedicated parallel job queue for the anime lighting stack on Android mobile. Where 003_rnd_Runtime.js contained a minimal fixed-capacity ring queue for in-frame drain, this module is the full-featured producer/consumer job system used by every lighting subsystem that needs to offload work off the main thread or defer work to a later frame.

The queue exists to serve these workloads:

· Shadow atlas page repacking after an LOD change
· GI probe re-baking after a biome transition
· AO blur kernel swap after a quality rescale
· Cluster grid rebuild after a camera far-plane change
· Environment palette solving after a day-cycle tick
· Interior light placement after a room load
· Exterior probe solving after a terrain stream
· Post-buffer preparation after a resolution change
· Director hint emission after a scene transition

The queue does four things that a plain array of callbacks cannot:

1. Priority tiers — CRITICAL / HIGH / NORMAL / LOW / IDLE. A "rebuild the light list because the camera moved" job preempts a cosmetic "ease the environment palette" job without starving it.
2. Dependency edges — a job can declare up to four parent jobs. A "commit shadow atlas" job can wait on "pack shadow atlas" plus "sync cascade matrices" without any manual polling.
3. Backpressure — beyond capacity, new jobs are rejected with a distinct result code so callers can degrade gracefully (skip GI bake this frame, reuse last probe set) instead of blocking.
4. Starvation watchdog — any job queued longer than STARVATION_MS gets promoted to the next tier up so a flood of high-priority jobs cannot permanently starve a low-priority one.

Plus:

5. Budget accounting per tier — the queue tracks an EMA of execution cost per priority tier so the adaptive quality controller can throttle by tier without disturbing the whole queue.
6. Cooperative cancellation — a job can be aborted before running, and running jobs receive a cancelFlag they can poll cheaply.
7. Deterministic ordering — jobs with equal priority and zero dependencies run in insertion order, so lighting output is bit-reproducible across runs on the same device.

---

Exported Constants

PERF_TIER

Type: string

Value: 'LOW' | 'MEDIUM' | 'HIGH' — cached from getPerfTier() at module load.

MAX_JOBS_PER_QUEUE

Type: number

Value: 8192 on HIGH, 4096 on MEDIUM, 2048 on LOW.

The default capacity of a JobQueue when constructed without an explicit capacity. Sized to accommodate the worst-case burst of lighting jobs at peak scene complexity.

JOB_PRIORITY

Type: frozen enum

Values:

· CRITICAL = 0
· HIGH = 1
· NORMAL = 2
· LOW = 3
· IDLE = 4
· COUNT = 5

Lower integer values mean higher priority. The _pickNext method iterates from tier 0 upward, so tier 0 runs first.

JOB_PRIORITY_NAME

Type: frozen array

Values: ['critical', 'high', 'normal', 'low', 'idle'].

JOB_STATE

Type: frozen enum

Values:

· EMPTY = 0
· QUEUED = 1
· RUNNING = 2
· DONE = 3
· FAILED = 4
· CANCELLED = 5

ENQUEUE_RESULT

Type: frozen enum

Values:

· OK = 0 — job was accepted into the queue.
· FULL = 1 — the queue is at capacity and could not accept the job.
· INVALID = 2 — the spec was malformed (missing run function or null spec).
· DEPENDENCY_LOST = 3 — a declared dependency could not be resolved and the job was cancelled.

JOB_KIND

Type: frozen enum

Typed job categories used for diagnostics and for the per-kind stats.

Values:

· GENERIC = 0
· LIGHT_LIST = 1
· SHADOW_ATLAS = 2
· SHADOW_MATRIX = 3
· GI_PROBE = 4
· AO_BLUR = 5
· CLUSTER_GRID = 6
· ENV_PALETTE = 7
· INTERIOR_VOL = 8
· EXTERIOR_PROBE = 9
· POST_BUFFER = 10
· DIRECTOR_HINT = 11
· COUNT = 12

JOB_KIND_NAME

Type: frozen array

Values: ['generic', 'light_list', 'shadow_atlas', 'shadow_matrix', 'gi_probe', 'ao_blur', 'cluster_grid', 'env_palette', 'interior_vol', 'exterior_probe', 'post_buffer', 'director_hint'].

STARVATION_MS

Type: number

Value: 500

If a job has been queued for longer than this many milliseconds, the queue promotes it to the next tier up on the next _pickNext call.

MAX_DEPS

Type: number

Value: 4

Maximum number of parent jobs a single job can depend on.

EMA_ALPHA

Type: number

Value: 0.10

Smoothing factor for the per-tier execution-time EMA.

---

Exported Class — JobSlot

One instance per capacity slot in the queue. Holds a single job's full state.

Constructor

```
new JobSlot(index)
```

Parameters: index — the slot's array index.

Instance Properties

· index — the slot's array index.
· id — a monotonic job id, unique for the lifetime of the queue.
· kind — one of JOB_KIND.
· priority — one of JOB_PRIORITY.
· state — one of JOB_STATE.
· run — the function to execute. Signature: (payload, ctx, job) => void.
· onDone — optional callback (payload, ctx, job) => void.
· onError — optional callback (error, payload, ctx, job) => void.
· ctx — arbitrary context passed to run.
· payload — arbitrary payload handle (Float32Array, WebGLRenderTarget, etc.).
· payloadMeta — arbitrary metadata.
· transferable — an array of Transferable objects, or null.
· queuedAtMs — timestamp when the job was enqueued.
· startedAtMs — timestamp when the job started running.
· finishedAtMs — timestamp when the job finished.
· deps — an Int32Array(MAX_DEPS) of parent job ids. -1 means empty.
· depCount — the number of used entries in deps.
· dependentCount — the number of children that depend on this job.
· cancelFlag — 1 if cancellation was requested.
· sequence — the insertion sequence number within the queue.

Instance Methods

reset()

Returns: this.

Purpose: zeroes every field and resets deps to all -1. Called by the queue's _releaseSlot().

---

Exported Class — PriorityBucket

A fixed-capacity FIFO queue used internally for a single priority tier.

Constructor

```
new PriorityBucket(capacity)
```

Parameters: capacity — the number of slots this bucket can hold.

Instance Properties

· capacity — the bucket's fixed capacity.
· indices — an Int32Array(capacity) of slot indices.
· head — the read pointer.
· tail — the write pointer.
· count — the number of items in the bucket.

Instance Methods

push(index)

Parameters: index — the slot index to add.

Returns: boolean — true on success, false if the bucket is full.

shift()

Returns: the slot index at the head, or -1 if the bucket is empty.

Purpose: pops the oldest item. Advances head and decrements count.

peek()

Returns: the slot index at the head without removing it, or -1.

clear()

Returns: nothing.

Purpose: resets the ring pointers and count.

---

Exported Class — JobQueue

The main queue.

Constructor

```
new JobQueue(options = {})
```

Parameters:

· capacity — the number of slots. Default MAX_JOBS_PER_QUEUE.
· bucketCapacity — the capacity of each priority bucket. Default MAX_JOBS_PER_QUEUE.
· maxJobsPerFrame — the number of jobs to drain per drain() call. Default 64 on HIGH, 32 on MEDIUM, 16 on LOW.
· starvationMs — the starvation promotion threshold. Default STARVATION_MS.
· autoPromote — whether to auto-promote starved jobs. Default true.

Constructor work:

1. Allocates the slots array of capacity fresh JobSlot instances.
2. Allocates and initializes the freeList ring buffer of slot indices.
3. Allocates the five PriorityBucket instances, one per priority tier.
4. Allocates idToIndex (a Map) for dependency resolution.
5. Initializes _nextId and _nextSequence to 1.
6. Initializes counters: queuedCount, runningCount, doneCount, failedCount, cancelledCount, rejectedCount.
7. Allocates tierMsEma and tierCountEma (both Float32Array of JOB_PRIORITY.COUNT).
8. Allocates _listeners (Map).

Instance Properties

· capacity — the queue's slot capacity.
· slots — an array of JobSlot instances.
· freeList — an Int32Array of free slot indices.
· freeHead — the ring read pointer.
· freeCount — the number of free slots.
· buckets — an array of five PriorityBucket instances.
· idToIndex — the Map from job id to slot index.
· activeIndex — the slot index currently running, or -1.
· queuedCount — the number of queued jobs.
· runningCount — the number of jobs currently running (always 0 or 1 in single-threaded mode).
· doneCount — the total number of successfully completed jobs.
· failedCount — the total number of failed jobs.
· cancelledCount — the total number of cancelled jobs.
· rejectedCount — the total number of jobs rejected due to capacity.
· tierMsEma — a Float32Array of per-tier execution-time EMAs.
· tierCountEma — a Float32Array of per-tier jobs-per-drain EMAs.

Instance Methods

on(event, fn)

Parameters:

· event — one of 'enqueued', 'promoted', 'done', 'failed', 'cancelled'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Parameters:

· event — event name.
· fn — the callback to remove.

Returns: nothing.

enqueue(spec)

Parameters: spec — an object with these fields:

· run — required function (payload, ctx, job) => void.
· onDone — optional callback.
· onError — optional callback.
· ctx — arbitrary context.
· payload — arbitrary payload.
· payloadMeta — arbitrary metadata.
· transferable — optional array of Transferables.
· kind — one of JOB_KIND.
· priority — one of JOB_PRIORITY.
· deps — an array of parent job ids.

Returns: { result: ENQUEUE_RESULT, id: number }.

Purpose: the primary enqueue entry point.

Flow:

1. Verifies spec and spec.run. Returns INVALID if not.
2. Acquires a slot from the free list. Returns FULL if none available.
3. Resets the slot and populates it from the spec.
4. Resolves dependencies: for each parent id, looks up the slot index in idToIndex and, if found, increments the parent's dependentCount.
5. Registers the job id in idToIndex.
6. Increments queuedCount and pushes the slot index into the correct priority bucket.
7. Emits enqueued.
8. Returns { result: OK, id }.

cancelById(id)

Parameters: id — the job id.

Returns: boolean.

Purpose: sets the job's cancelFlag. If the job is still queued, finalizes it immediately as CANCELLED. If it is running, only sets the flag so the job's own code can poll it.

drain(maxJobs, onRunError)

Parameters:

· maxJobs — the maximum number of jobs to run in this call.
· onRunError — an optional error callback (error, slotIndex) => void.

Returns: the number of jobs executed.

Purpose: the per-frame drain entry point.

Flow:

1. Loops until maxJobs jobs have been executed or the queue is empty.
2. Picks the next runnable slot via _pickNext().
3. If the slot's dependencies are not all resolved, pushes it back into its bucket and, if no other runnable jobs remain, breaks the loop.
4. If the job was cancelled, finalizes it as CANCELLED.
5. Otherwise, transitions the slot to RUNNING, records the start time, invokes run(payload, ctx, job) in try/catch, records the finish time, updates tierMsEma and tierCountEma, transitions the slot to DONE or FAILED, and calls the appropriate callback.

_pickNext()

Internal. Returns the slot index of the highest-priority queued job. Calls _promoteStarved() first if autoPromote is on. Iterates priority tiers from 0 upward.

_promoteStarved()

Internal. Iterates priority tiers from the lowest upward, scanning each bucket for the first job older than starvationMs. Promotes it to the next tier up, emits promoted, and returns.

_removeFromBucket(bucket, pos)

Internal. Removes the entry at position pos from a ring bucket by shifting the tail entries down to close the gap.

_depsResolved(slot)

Internal. Returns true if every parent job id in the slot's deps array maps to a slot that is either DONE or no longer in idToIndex.

_onlyBlockedRemaining()

Internal. Returns true if the only queued jobs remaining are blocked on unresolved dependencies. Used to break infinite loops in drain().

_finalize(slot, finalState, error)

Internal. Transitions the slot to a terminal state, records the finish time, nulls the run callback, decrements dependent counts on parents, removes the id from idToIndex, emits done or failed or cancelled, and returns the slot to the free list.

clear()

Returns: nothing.

Purpose: empties every bucket, resets every slot, and resets all counters.

dispose()

Returns: nothing.

Purpose: clears and nulls every internal array.

getStats()

Returns: an object with capacity, queuedCount, runningCount, doneCount, failedCount, cancelledCount, rejectedCount, freeSlots, a buckets array with per-tier counts and EMAs, and perfTier.

---

Exported Function — collectTransferables(payload)

Parameters: payload — any value.

Returns: an array of Transferables, or null.

Purpose: helper that inspects a payload and returns the ArrayBuffers that can be transferred (zero-copy) via postMessage to a worker. Handles TypedArray, ArrayBuffer, arrays of those, and ImageBitmap.

---

Exported Job Factory Functions

Each factory returns a fully-populated spec object ready to pass to queue.enqueue().

makeLightListJob(run, ctx, payload)

Returns a spec with kind: JOB_KIND.LIGHT_LIST and priority: JOB_PRIORITY.CRITICAL.

makeShadowAtlasJob(run, ctx, payload, deps)

Returns a spec with kind: JOB_KIND.SHADOW_ATLAS, priority: JOB_PRIORITY.HIGH, and deps.

makeShadowMatrixJob(run, ctx, payload, deps)

Returns a spec with kind: JOB_KIND.SHADOW_MATRIX, priority: JOB_PRIORITY.HIGH, and deps.

makeGIProbeJob(run, ctx, payload)

Returns a spec with kind: JOB_KIND.GI_PROBE and priority: JOB_PRIORITY.NORMAL.

makeAOBlurJob(run, ctx, payload, deps)

Returns a spec with kind: JOB_KIND.AO_BLUR, priority: JOB_PRIORITY.NORMAL, and deps.

makeClusterGridJob(run, ctx, payload)

Returns a spec with kind: JOB_KIND.CLUSTER_GRID and priority: JOB_PRIORITY.HIGH.

makeEnvPaletteJob(run, ctx, payload)

Returns a spec with kind: JOB_KIND.ENV_PALETTE and priority: JOB_PRIORITY.LOW.

makeInteriorVolJob(run, ctx, payload)

Returns a spec with kind: JOB_KIND.INTERIOR_VOL and priority: JOB_PRIORITY.NORMAL.

makeExteriorProbeJob(run, ctx, payload)

Returns a spec with kind: JOB_KIND.EXTERIOR_PROBE and priority: JOB_PRIORITY.NORMAL.

makePostBufferJob(run, ctx, payload)

Returns a spec with kind: JOB_KIND.POST_BUFFER and priority: JOB_PRIORITY.NORMAL.

makeDirectorHintJob(run, ctx, payload)

Returns a spec with kind: JOB_KIND.DIRECTOR_HINT and priority: JOB_PRIORITY.IDLE.

---

Exported Functions

getDefaultJobQueue()

Returns: the module-level singleton JobQueue, creating it on first call.

disposeDefaultJobQueue()

Returns: nothing.

createJobQueue(options = {})

Returns: a new JobQueue.

---

Default Export

The default export bundles: JobQueue, JobSlot, PriorityBucket, createJobQueue, getDefaultJobQueue, disposeDefaultJobQueue, collectTransferables, all the make*Job factories, JOB_PRIORITY, JOB_PRIORITY_NAME, JOB_STATE, JOB_KIND, JOB_KIND_NAME, ENQUEUE_RESULT, MAX_JOBS_PER_QUEUE.

---

Usage Pattern

A lighting subsystem enqueues work with dependencies:

```
import { getDefaultJobQueue, makeLightListJob, makeShadowAtlasJob } from './src/core/007_rnd_JobQueue.js';

const queue = getDefaultJobQueue();

const lightListResult = queue.enqueue(makeLightListJob((payload, ctx, job) => {
  // rebuild the light list
}, myCtx, lightListPayload));

const shadowResult = queue.enqueue(makeShadowAtlasJob((payload, ctx, job) => {
  // repack the shadow atlas
}, myCtx, shadowPayload, [lightListResult.id]));
```

Each frame, the EngineLoop or Runtime drains the queue:

```
const executed = queue.drain(32, (err, idx) => {
  console.error(`job ${idx} failed`, err);
});
```

The tierMsEma array tells the adaptive quality controller how much time each priority tier is consuming per drain. If CRITICAL tier goes over its own implicit budget, the controller knows the light list rebuild is the bottleneck, not the GI bake.
