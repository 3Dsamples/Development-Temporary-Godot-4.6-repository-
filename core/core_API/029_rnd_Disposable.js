API Documentation — src/core/029_rnd_Disposable.js

File Purpose

This file provides the deterministic disposal and teardown primitive for the anime lighting stack. Every lighting subsystem that owns GPU or CPU resources registers them with a Disposable so teardown is guaranteed — no orphan render targets, no leaked worker ports, no dangling event subscriptions, no retained closures.

The distinction between this module and 014_rnd_ResourceRegistry.js is lifecycle versus storage:

· The registry tracks live GPU resources and reference counts. It answers "what resources exist, who owns them, and can they be released?"
· This module tracks lifecycle order. It answers "in what sequence must things be torn down so nothing touches a freed handle during teardown?"

The order matters because teardown is not a flat operation. Listeners must disconnect before signals disappear. Pools must return their slabs before the pool itself is disposed. Render targets must release before the renderer loses its context. If any of those happen out of order, the teardown throws, and on Android a thrown teardown during context loss is unrecoverable.

This module solves the ordering problem with four primitives:

1. Disposable — a base class with a dispose() method, a bag of teardown callbacks, and a linked list of child Disposables that get disposed first (reverse-registration order, LIFO).
2. DisposableBag — a lightweight collection of teardown callbacks executed in reverse-registration order, grouped by a TEARDOWN_ORDER rank. Used for ad-hoc resources that do not warrant a full class.
3. DisposableScope — a scoped lifetime container. Used for chunk unload, room exit, LOD swap — anywhere a group of resources must be released together.
4. DisposableRegistry — an optional global tracker so a debug tool can list every live Disposable and identify leaks.

The module also provides an auto-dispose safety net via FinalizationRegistry on HIGH-tier devices. If a Disposable is garbage-collected without being disposed, the finalizer logs a warning identifying the leaked object. This is a development-only feature that catches the classic bug where a chunk unloads but its resources stay alive.

The teardown order is enforced by five ranks:

· FIRST = 0 — listeners, signal connections
· EARLY = 1 — pools (release back to free list)
· NORMAL = 2 — render targets, buffers
· LATE = 3 — materials, geometries
· LAST = 4 — GPU resources that others reference

Every callback registered with a Disposable is grouped by its rank, and the bag runs rank 0 first, then rank 1, and so on. Within each rank, callbacks run in reverse-registration order. This is the exact inverse of setup order, which is what teardown requires.

The design constraint is that dispose is a cold path. It runs once per resource, and it never needs to be fast. But it must never throw on the caller. Every teardown callback is wrapped in try/catch, and one callback's throw never prevents its siblings from running.

---

Exported Constants

PERF_TIER_LOCAL

Internal. The cached PERF_TIER string from getPerfTier().

MAX_BAG_ENTRIES

Type: number

Value: 64 on HIGH, 48 on MEDIUM, 32 on LOW.

The maximum number of teardown callbacks a single bag can hold. Sized to accommodate the resource count of a typical subsystem (shadow system holds about twelve callbacks; GI system holds about eight; AO holds four).

MAX_SCOPE_CHILDREN

Type: number

Value: 32 on HIGH, 24 on MEDIUM, 16 on LOW.

The maximum number of child Disposables a single Disposable can own.

DISPOSE_STATE

Type: frozen enum

Values:

· ALIVE = 0 — the Disposable has not been disposed.
· DISPOSING = 1 — dispose() is currently running.
· DISPOSED = 2 — dispose() completed.
· FAILED = 3 — reserved for future expansion.

The DISPOSING state is critical for re-entrancy safety. If a teardown callback tries to call dispose() on the parent again, the parent sees DISPOSING and returns false instead of recursively disposing.

TEARDOWN_ORDER

Type: frozen enum

The five ranks in which teardown callbacks execute.

· FIRST = 0 — listeners, signal connections. These must run first so that a downstream system that receives a callback during teardown does not find a half-disposed resource.
· EARLY = 1 — pools. Returned to their free lists before other resources reference them.
· NORMAL = 2 — render targets, buffers, generic resources.
· LATE = 3 — materials, geometries. These reference other resources, so they go after the resources they reference.
· LAST = 4 — GPU resources that other resources reference. The renderer context itself, the WebGL render targets that hold depth attachments.

The rank semantics match the dependency direction of the resources. Resources that others depend on are torn down LAST.

FINALIZER_ENABLED

Type: boolean

True only if FinalizationRegistry is available AND the device is on HIGH tier. The finalizer is a development-time safety net and is intentionally disabled on lower tiers because it has non-trivial memory overhead.

---

Module-Level State (Not Exported Directly)

_disposableIdCounter

Type: number

Monotonic counter for Disposable ids.

_finalizer

Type: FinalizationRegistry | null

The lazily-created finalization registry.

_finalizerTokens

Type: Map<number, { label, state }>

The map from Disposable id to its tracking info. Used by the finalizer's callback to log leaks.

_defaultRegistry

Type: DisposableRegistry | null

The module-level singleton for the optional live tracker.

---

Internal Helper Functions (Documented)

_nextDisposableId()

Returns: the next monotonic Disposable id.

_now()

Returns: the current high-resolution timestamp.

_safeLogger()

Returns: the default logger, or null if it cannot be resolved. Used by the bag and the Disposable class to surface warnings and errors without creating a hard dependency on 026_rnd_Logger.js.

_ensureFinalizer()

Returns: the lazily-created FinalizationRegistry, or null if unavailable.

Purpose: creates the registry on first use. The registry's callback receives the Disposable's id and looks up its info in _finalizerTokens. If the tracked state is not DISPOSED, it logs a leak warning.

_registerWithFinalizer(disposable)

Parameters: disposable — the Disposable to register.

Returns: nothing.

Purpose: registers the Disposable with the finalizer and records its info in _finalizerTokens.

_unregisterWithFinalizer(token)

Parameters: token — the Disposable's id.

Returns: nothing.

Purpose: removes the Disposable from the finalizer and the token map when dispose() is called.

_releaseResourceFn(handle)

Parameters: handle — a resource handle from 014_rnd_ResourceRegistry.js.

Returns: nothing.

Purpose: the teardown callback used by DisposableBag.addResourceHandle. Calls forceRelease(handle) on the registry.

---

Exported Class — DisposableBag

A fixed-capacity collection of teardown callbacks, grouped by TEARDOWN_ORDER rank.

Constructor

```
new DisposableBag(label)
```

Parameters: label — a diagnostic label used in error messages.

Instance Properties

· label — the bag's label.
· capacity — the fixed capacity (MAX_BAG_ENTRIES).
· fn — an array of teardown callbacks.
· ctx — an array of contexts, parallel.
· name — an array of diagnostic names, parallel.
· order — a Uint8Array of TEARDOWN_ORDER ranks.
· count — the number of registered entries.
· _state — one of DISPOSE_STATE.
· _disposedCount — how many callbacks ran successfully.
· _failedCount — how many callbacks threw.

Instance Methods

add(fn, ctx, name, order = TEARDOWN_ORDER.NORMAL)

Parameters:

· fn — the teardown callback.
· ctx — an optional context object (passed as this).
· name — an optional diagnostic name.
· order — one of TEARDOWN_ORDER.

Returns: boolean.

Purpose: registers a callback. Returns false if the bag is disposed, disposing, or full, or if fn is not a function.

addDisposable(obj, name, order = TEARDOWN_ORDER.LATE)

Parameters:

· obj — an object with a dispose() method.
· name — a diagnostic name.
· order — the teardown rank. Default LATE.

Returns: boolean.

Purpose: convenience wrapper that registers obj.dispose with obj as the context.

addConnection(connection, name)

Parameters:

· connection — a Connection from 028_rnd_Signal.js.
· name — an optional name.

Returns: boolean.

Purpose: registers the connection's disconnect method with rank FIRST.

addResourceHandle(handle, name)

Parameters:

· handle — a resource handle from 014_rnd_ResourceRegistry.js.
· name — an optional name.

Returns: boolean.

Purpose: registers a callback that force-releases the resource handle. Uses rank NORMAL.

run()

Returns: boolean.

Purpose: executes every registered callback.

Flow:

1. If already disposed, returns true.
2. If disposing (re-entrant), returns false.
3. Sets _state = DISPOSING.
4. For each TEARDOWN_ORDER rank from FIRST to LAST:
   · Iterates the registered entries in reverse order (LIFO).
   · Calls each entry whose rank matches the current rank.
   · Wraps each call in try/catch. Increments _disposedCount or _failedCount.
   · Clears the entry's slots eagerly so a second run cannot double-fire.
5. Sets count = 0 and _state = DISPOSED.

reset()

Returns: this. Clears every entry and resets the state to ALIVE.

getStats()

Returns: an object with label, capacity, count, state, disposedCount, failedCount.

---

Exported Class — Disposable

The base class for every subsystem that owns resources.

Constructor

```
new Disposable(label)
```

Parameters: label — a diagnostic label. Used in log messages and error traces.

Constructor work:

1. Increments the global counter and assigns disposableId.
2. Stores the label.
3. Initializes _state = ALIVE.
4. Allocates a DisposableBag with the label.
5. Allocates _children — an array of MAX_SCOPE_CHILDREN entries.
6. Initializes _childCount = 0.
7. Records _createdAtMs.
8. On HIGH tier, registers with the finalizer.

Instance Properties

· disposableId — the unique id.
· label — the diagnostic label.
· state — one of DISPOSE_STATE.
· isDisposed — true after dispose.
· isDisposing — true during dispose.
· createdAtMs — the creation timestamp.
· disposedAtMs — the disposal timestamp.
· ageMs — the age in milliseconds.

Instance Methods

register(fn, ctx, name, order)

Parameters: same as DisposableBag.add.

Returns: boolean.

Purpose: registers a raw teardown callback on the internal bag.

registerDisposable(obj, name, order = TEARDOWN_ORDER.LATE)

Parameters: same as DisposableBag.addDisposable.

Returns: boolean.

registerConnection(connection, name)

Parameters: same as DisposableBag.addConnection.

Returns: boolean.

registerResource(handle, name)

Parameters: same as DisposableBag.addResourceHandle.

Returns: boolean.

registerGPU(obj, name)

Parameters:

· obj — an object with a dispose() method.
· name — a diagnostic name.

Returns: boolean.

Purpose: convenience wrapper for GPU resources. Uses TEARDOWN_ORDER.LATE.

registerChild(child)

Parameters: child — a child Disposable.

Returns: boolean.

Purpose: registers a child Disposable. Children are disposed BEFORE the parent's own bag entries so the parent still has valid resources while cleaning up children.

dispose()

Returns: boolean.

Purpose: the main teardown routine.

Flow:

1. If already disposed, returns true.
2. If disposing (re-entrant), returns false.
3. Sets _state = DISPOSING.
4. Records the start time.
5. Iterates the children in reverse-registration order. Calls child.dispose() on each. Wraps in try/catch.
6. Clears the children array.
7. Runs the internal bag.
8. If the subclass defines onDispose, calls it in try/catch.
9. Records _disposedAtMs.
10. On HIGH tier, unregisters from the finalizer.
11. Sets _state = DISPOSED.
12. On HIGH tier, records a profiler marker.
13. Returns true.

getStats()

Returns: an object with id, label, state, bag (the bag's stats), children (the current child count), ageMs.

Subclass Hook — onDispose()

If a subclass defines onDispose() as an instance method, it is called AFTER the bag runs. This is where subclass-specific cleanup happens — releasing a reference to a global, nulling out arrays that are not in the bag.

---

Exported Class — DisposableScope

A subclass of Disposable for scoped lifetimes.

Constructor

```
new DisposableScope(label)
```

Instance Properties

· _ownedCount — the number of resources owned by the scope.

Instance Methods

own(obj, name)

Parameters:

· obj — an object with a dispose() method.
· name — an optional diagnostic name.

Returns: boolean.

Purpose: owns a GPU resource. Increments _ownedCount.

track(connection, name)

Parameters:

· connection — a signal connection.
· name — an optional diagnostic name.

Returns: boolean.

Purpose: tracks a signal connection. Increments _ownedCount.

trackCallback(fn, ctx, name, order)

Parameters: same as DisposableBag.add.

Returns: boolean.

Purpose: tracks a raw callback.

trackResource(handle, name)

Parameters:

· handle — a resource handle from 014.
· name — an optional name.

Returns: boolean.

Purpose: tracks a resource handle.

end()

Returns: boolean.

Purpose: alias for dispose(). Reads better in scoped contexts.

ownedCount (getter)

Returns: the number of owned resources.

---

Exported Class — DisposableRegistry

An optional global tracker.

Constructor

```
new DisposableRegistry()
```

Instance Properties

· map — the Map from id to Disposable.
· capacity — the maximum tracked count.

Instance Methods

track(disposable)

Parameters: disposable — a Disposable.

Returns: boolean.

untrack(disposable)

Parameters: disposable — a Disposable.

Returns: boolean.

listLive()

Returns: an array of stats objects for every non-disposed Disposable in the registry.

count()

Returns: the number of live Disposables.

clear()

Returns: nothing.

---

Exported Functions

bagOf(label, entries)

Parameters:

· label — the bag's label.
· entries — an array of { fn, ctx, name, order } descriptors.

Returns: a DisposableBag with the entries pre-loaded.

runTeardown(fns)

Parameters: fns — an array of teardown functions.

Returns: the number of functions that ran successfully.

Purpose: convenience for ad-hoc teardowns. Runs every function in reverse order, swallowing and logging errors.

getDisposableRegistry()

Returns: the module-level singleton DisposableRegistry, creating it on first call.

trackDisposable(disposable)

Parameters: disposable — a Disposable.

Returns: boolean.

Purpose: registers the Disposable with the global tracker.

untrackDisposable(disposable)

Parameters: disposable — a Disposable.

Returns: boolean.

listLiveDisposables()

Returns: an array of stats objects for every live Disposable.

Purpose: the leak audit for a debug HUD. If the count keeps growing over time, some subsystem is creating Disposables and forgetting to dispose them.

---

Default Export

The default export bundles: Disposable, DisposableBag, DisposableScope, DisposableRegistry, bagOf, runTeardown, getDisposableRegistry, trackDisposable, untrackDisposable, listLiveDisposables, DISPOSE_STATE, TEARDOWN_ORDER, MAX_BAG_ENTRIES, MAX_SCOPE_CHILDREN.

---

Usage Pattern

A subsystem that extends Disposable:

```
import { Disposable } from './src/core/029_rnd_Disposable.js';

class ShadowSystem extends Disposable {
  constructor(scene) {
    super('shadowSystem');
    this.scene = scene;

    this.atlas = createShadowAtlas();
    this.registerGPU(this.atlas, 'shadow_atlas');

    this.atlasChangedConn = LIGHTING_SIGNALS.shadowAtlasChanged.connect(() => {
      this.markDirty();
    });
    this.registerConnection(this.atlasChangedConn, 'atlas_changed');

    this.targetHandle = registerRenderTarget(this.atlas.renderTarget, {
      ownerId: this.ownerId,
    });
    this.registerResource(this.targetHandle, 'atlas_rt');
  }

  onDispose() {
    this.scene = null;
    this.atlas = null;
  }
}

const shadowSystem = new ShadowSystem(scene);

// Later, on chunk unload or engine shutdown:
shadowSystem.dispose();
```

A chunk-scoped teardown using DisposableScope:

```
import { DisposableScope } from './src/core/029_rnd_Disposable.js';

function loadChunk(chunkId) {
  const scope = new DisposableScope(`chunk_${chunkId}`);

  const geometry = buildChunkGeometry(chunkId);
  scope.own(geometry, 'geometry');

  const material = buildChunkMaterial(chunkId);
  scope.own(material, 'material');

  const conn = someSignal.connect(() => {
    // ...
  });
  scope.track(conn, 'signal');

  return scope;
}

// Later, on chunk unload:
scope.end();
```

A debug HUD that audits leaks:

```
import { listLiveDisposables } from './src/core/029_rnd_Disposable.js';

const live = listLiveDisposables();
if (live.length > 500) {
  console.warn(`Possible leak: ${live.length} live Disposables`);
  for (const info of live.slice(0, 10)) {
    console.warn(`  ${info.label} (age ${info.ageMs.toFixed(0)}ms)`);
  }
}
```

The Disposable primitive is what makes the engine's teardown deterministic. When a chunk unloads, the scope runs every callback in the correct order: signal connections first, then pool returns, then render targets, then materials, then the parent GPU resources. When the engine shuts down, the top-level Disposable tree runs the same teardown recursively, child by child, so nothing touches a freed handle. On Android, where a botched teardown during context loss is unrecoverable, this determinism is what allows the engine to recover from the browser releasing the WebGL context.
