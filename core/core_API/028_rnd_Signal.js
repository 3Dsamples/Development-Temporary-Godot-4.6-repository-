API Documentation — src/core/028_rnd_Signal.js

File Purpose

This file provides the lightweight signal/slot primitive for the anime lighting stack. Where 027_rnd_EventBus.js owns named-topic fan-out with a per-frame drain queue, this module owns the simpler, tighter primitive: a fixed-capacity, priority-ordered listener chain attached directly to a value-carrying or notification-only object.

The distinction between a signal and an event bus topic is subtle but important:

· An event bus topic is identified by a string name resolved at emit time. It is best for cross-subsystem fan-out where the publisher does not know or care who subscribes.
· A signal is identified by object identity. It is best for local, high-frequency notifications where the emitter and the subscriber already know each other.

Both patterns exist because the lighting stack needs both. When the biome changes and ten subsystems must react, that is a topic. When the shadow atlas is repacked and the GI system that already holds a reference to the shadow atlas wants to know, that is a signal.

Signals are the pattern used by every lighting subsystem for its own state notifications. Examples:

· shadowAtlasChanged — a pulse signal emitted when the atlas is repacked
· giProbeDirty — a pulse signal emitted when probes are invalidated
· qualityLevelChanged — a value signal emitted when the quality level transitions
· biomeWeightsChanged — a value signal emitted when biome weights update
· dayCycleTick — a value signal emitted every day-cycle tick
· cameraMoved — a pulse signal emitted on any camera transform
· lightListChanged — a pulse signal emitted when the light count or order changes
· clusterGridChanged — a pulse signal emitted when the cluster grid rebuilds
· interiorVolumeChanged — a pulse signal emitted when the interior probes update
· exteriorProbeChanged — a pulse signal emitted when the exterior probes update
· contextLost — a pulse signal emitted on WebGL context loss
· frameTick — a pulse signal emitted once per frame

The module provides two signal types:

1. Signal — a value-carrying signal. Stores the last emitted value so late subscribers can read it immediately. Every emit writes the value, dispatches to listeners in priority order, and optionally appends to a bounded history ring.
2. PulseSignal — a notification-only signal. Cheaper than Signal by one field. Used for pure "this happened" notifications where no payload is needed.

Both types share the same connection API: connect(fn, ctx, priority, once) returns a Connection token with a disconnect() method. Priorities order the listener chain from lowest to highest. Ties break by registration order.

The module also provides a SignalGroup container for related signals, and three combination helpers — combine, merge, and anyOf — that derive new signals from input signals.

Finally, the module exports LIGHTING_SIGNALS — a pre-built SignalGroup containing every canonical signal the engine uses. Downstream subsystems import named signals from this group instead of creating their own, which guarantees a single source of truth per notification type.

The design constraint is zero per-frame allocations. emit and pulse touch only pre-allocated arrays. Connection tokens are allocated once per connect call; disconnecting is O(1). Listener arrays are parallel typed arrays sized at construction; adding or removing a listener shifts the tail in place.

---

Exported Constants

PERF_TIER_LOCAL

Internal. The cached PERF_TIER string from getPerfTier().

MAX_LISTENERS

Type: number

Value: 32 on HIGH, 24 on MEDIUM, 16 on LOW.

The default capacity of every signal's listener chain. Sized to accommodate the realistic fan-out for a single signal — most signals have three to six subscribers.

MAX_HISTORY

Type: number

Value: 64 on HIGH, 32 on MEDIUM, 16 on LOW.

The default capacity of the optional value-history ring. Only enabled when a signal is constructed with history: true.

SIGNAL_PRIORITY

Type: frozen object

Named priority constants for common cases.

· HIGHEST — 0
· HIGH — 100
· NORMAL — 500
· LOW — 800
· LOWEST — 999

Lower values run earlier.

DISPATCH_RESULT

Type: frozen enum

Values:

· OK = 0 — at least one listener ran.
· CANCELLED = 1 — a listener called stopPropagation().
· EMPTY = 2 — no listeners were connected.
· DISPOSED = 3 — the signal has been disposed.

PROCEDURAL_FLAG and PROCEDURAL_SOURCE

Not present in this file. These are defined in 032_rnd_NoImageTexturePolicy.js.

---

Module-Level State (Not Exported Directly)

_defaultSignalGroup

Not applicable — the module exports a directly-constructed LIGHTING_SIGNALS group.

---

Internal Helper Functions (Documented)

_now()

Returns: the current high-resolution timestamp via performance.now(), or Date.now() as a fallback.

_findInsertPos(priorities, count, priority)

Parameters:

· priorities — an Int16Array of priorities.
· count — the number of valid entries.
· priority — the priority of the listener to insert.

Returns: the insertion index.

Purpose: binary search for the first position where the existing priority is greater than the new one. Keeps the listener chain sorted in ascending priority order.

---

Exported Class — Connection

A handle returned by connect or once. Owns the listener's identity and provides disconnect().

Constructor

```
new Connection(signal, fn)
```

Parameters:

· signal — the signal the listener is attached to.
· fn — the listener function.

Instance Properties

· signal — the signal reference.
· fn — the listener function.
· active — true if the connection has not been disconnected.

Instance Methods

disconnect()

Returns: boolean — true on success.

Purpose: removes the listener from the signal. Idempotent — calling twice returns false the second time.

isActive()

Returns: active === true.

---

Exported Class — Signal

A value-carrying signal.

Constructor

```
new Signal(name, options = {})
```

Parameters:

· name — the signal's diagnostic name string.
· options.capacity — the listener chain capacity. Default MAX_LISTENERS.
· options.initialValue — the value to store before the first emit. Default null.
· options.history — if true, keeps a bounded ring of the last N emitted values. Default false.
· options.historyCapacity — the ring size when history is enabled. Default MAX_HISTORY.

Constructor work:

1. Allocates the four parallel listener arrays: _fn, _ctx, _prio, _once.
2. Initializes _count = 0.
3. Stores the initial value and sets _hasEmitted = false.
4. If history is enabled, allocates _historyValues and _historyFrames.
5. Allocates _dispatchState — the shared cancellation context.
6. Initializes the stats counters.

Instance Properties

· name — the signal name.
· value — the last emitted value.
· hasEmitted — true if emit has been called at least once.
· listenerCount — the current listener count.
· capacity — the listener chain capacity.
· disposed — true after dispose().

Instance Methods

connect(fn, ctx, priority = SIGNAL_PRIORITY.NORMAL, once = false)

Parameters:

· fn — the listener function (value, signal) => void.
· ctx — the context object passed as this. Default null.
· priority — the priority integer. Lower runs earlier.
· once — whether the listener is removed after the first call.

Returns: a Connection token, or null if the signal is disposed, fn is not a function, or the listener chain is full.

Purpose: registers a listener. Uses _findInsertPos to find the correct slot, _shiftUp to make room, then writes the listener into the parallel arrays. Emits a warning via the logger if the chain is full.

once(fn, ctx, priority)

Convenience wrapper for connect with once = true.

disconnect(fn)

Parameters: fn — the listener function to remove.

Returns: boolean.

Purpose: removes a listener by identity.

disconnectAll()

Returns: this. Clears every listener.

_removeListener(fn)

Internal. Linear scan for a listener by identity, then _shiftDown.

_shiftUp(pos)

Internal. Shifts listeners up to open a slot.

_shiftDown(pos)

Internal. Shifts listeners down to close a slot.

emit(value, frame)

Parameters:

· value — the value to emit.
· frame — an optional frame number for history tracking.

Returns: one of DISPATCH_RESULT.

Purpose: the main emission routine.

Flow:

1. If disposed, returns DISPOSED.
2. Increments _emits, stores _lastEmitMs, and records the frame if provided.
3. Stores the value in _value and sets _hasEmitted = true.
4. If history is enabled, appends to the history ring.
5. If _count === 0, returns EMPTY.
6. Resets the dispatch state flags.
7. Iterates the listener chain in priority order. Calls each listener in try/catch. If a listener sets cancelled, stops the loop. If the listener is once, removes it after the call.
8. Returns OK or CANCELLED.

stopPropagation()

Returns: nothing.

Purpose: sets the cancellation flag on the current dispatch context.

emitIfChanged(value, frame)

Parameters: same as emit.

Returns: EMPTY if the value equals the last emitted value, otherwise the result of emit.

Purpose: convenience for signals where subscribers only care about distinct values.

emitIfChangedBy(eps, value, frame)

Parameters:

· eps — the numeric epsilon.
· value — the value.
· frame — the optional frame.

Returns: EMPTY if the value is a number within eps of the last value, or a reference-equal non-number, otherwise the result of emit.

Purpose: numeric-tolerance variant of emitIfChanged.

getHistoryCount()

Returns: the number of valid history entries.

getHistoryCapacity()

Returns: the history ring size.

copyHistory(max, outValues, outFrames)

Parameters:

· max — the maximum number of entries.
· outValues — an array to receive the values.
· outFrames — an optional typed array to receive the frames.

Returns: the number of entries copied.

_safeLogger()

Internal. Lazily resolves the logger.

getStats()

Returns: an object with name, listeners, capacity, peakListeners, emits, dispatches, cancelled, rejected, hasEmitted, value, historyCount, lastEmitMs, lastEmitFrame, disposed.

dispose()

Returns: nothing. Clears every listener and nulls the history.

---

Exported Class — PulseSignal

A notification-only signal. Same shape as Signal minus the value and history fields.

Constructor

```
new PulseSignal(name, options = {})
```

Parameters:

· name — the signal name.
· options.capacity — the listener chain capacity. Default MAX_LISTENERS.

Instance Properties

· name — the signal name.
· listenerCount — the current count.
· capacity — the chain capacity.
· disposed — true after dispose.

Instance Methods

connect(fn, ctx, priority, once)

Parameters: same as Signal.connect, but fn receives (signal) as its only argument.

Returns: a Connection token.

once(fn, ctx, priority)

Convenience wrapper.

disconnect(fn)

Returns: boolean.

disconnectAll()

Returns: this.

_removeListener(fn)

Internal.

_shiftUp(pos)

Internal.

_shiftDown(pos)

Internal.

pulse(frame)

Parameters: frame — an optional frame number.

Returns: one of DISPATCH_RESULT.

Purpose: emits a pulse to every listener. Same iteration logic as Signal.emit but without value storage.

stopPropagation()

Returns: nothing.

_safeLogger()

Internal.

getStats()

Returns: an object with name, listeners, capacity, peakListeners, emits, dispatches, cancelled, rejected, lastEmitMs, lastEmitFrame, disposed.

dispose()

Returns: nothing.

---

Exported Class — SignalGroup

A named container of related signals.

Constructor

```
new SignalGroup(name)
```

Parameters: name — the group's diagnostic name.

Instance Properties

· name — the group name.
· map — a Map from signal name to signal instance.
· count — the number of signals in the group.

Instance Methods

ensureSignal(name, options)

Parameters:

· name — the signal name.
· options — passed to the Signal constructor.

Returns: the Signal instance, creating it on first call.

ensurePulse(name, options)

Parameters:

· name — the signal name.
· options — passed to the PulseSignal constructor.

Returns: the PulseSignal instance.

get(name)

Parameters: name — the signal name.

Returns: the signal, or null.

remove(name)

Parameters: name — the signal name.

Returns: boolean. Disposes the signal and removes it from the group.

disconnectAll()

Returns: this. Calls disconnectAll() on every signal in the group.

dispose()

Returns: this. Disposes every signal and clears the group.

getStats()

Returns: an object with name, count, and an array of per-signal stats.

---

Exported Functions

combine(name, inputs, options = {})

Parameters:

· name — the derived signal name.
· inputs — an array of Signal or PulseSignal instances.
· options — passed to the PulseSignal constructor.

Returns: a PulseSignal that pulses whenever any input emits or pulses.

Purpose: creates a derived signal that fires on any input. Useful for aggregating several "something changed" signals into one.

The derived signal connects to each input with a stable bound method so no closures are created per input.

merge(name, inputs, options = {})

Parameters:

· name — the derived signal name.
· inputs — an array of signals.
· options — passed to the Signal constructor.

Returns: a Signal that mirrors the last value emitted by any input. PulseSignal inputs contribute undefined.

Purpose: creates a derived value signal from several value signals.

anyOf(name, inputs, options = {})

Parameters:

· name — the derived signal name.
· inputs — an array of signals.
· options — passed to the Signal constructor.

Returns: a Signal<boolean> that emits true whenever any input emits or pulses.

Purpose: convenience for "has any of these happened?" signals.

---

The LIGHTING_SIGNALS Group

A pre-built SignalGroup containing every canonical signal the engine uses. Downstream subsystems import named signals from this group instead of creating their own.

The exported named signals are:

Shadow:

· shadowAtlasChanged — pulse.
· shadowCascadeChanged — pulse.
· shadowFilterChanged — pulse.

GI:

· giProbeDirty — pulse.
· giProbeRebaked — pulse.
· giBudgetChanged — value, initial 0.

AO:

· aoResolutionChanged — pulse.
· aoSamplesChanged — value, initial 0.

Lights:

· lightListChanged — pulse.
· clusterGridChanged — pulse.

Environment:

· dayCycleTick — value, initial 0.
· biomeWeightsChanged — value, initial null.
· envPaletteChanged — pulse.
· weatherChanged — pulse.

Interior / exterior:

· interiorEntered — pulse.
· interiorExited — pulse.
· exteriorChanged — pulse.

Quality / tier:

· qualityLevelChanged — value, initial 'high'.
· tierChanged — value, initial 'medium'.
· thermalChanged — value, initial 'nominal'.
· batteryChanged — value, initial 1.0.

Camera:

· cameraMoved — pulse.
· cameraTeleported — pulse.

Lifecycle:

· contextLost — pulse.
· contextRestored — pulse.
· visibilityChanged — value, initial true.

Frame:

· frameTick — pulse.
· frameHitch — value, initial 0.

Director / debug:

· directorHint — value, initial null.
· screenshotRequested — pulse.

Every signal in this group is created once at module load. Downstream code imports the named constant and calls connect or pulse directly.

---

Exported Factory Functions

createSignal(name, options = {})

Returns: a new Signal.

createPulseSignal(name, options = {})

Returns: a new PulseSignal.

createSignalGroup(name)

Returns: a new SignalGroup.

---

Default Export

The default export bundles: Signal, PulseSignal, SignalGroup, Connection, createSignal, createPulseSignal, createSignalGroup, combine, merge, anyOf, LIGHTING_SIGNALS, SIGNAL_PRIORITY, DISPATCH_RESULT, MAX_LISTENERS, MAX_HISTORY.

---

Usage Pattern

A subsystem that owns a pulse signal:

```
import {
  LIGHTING_SIGNALS,
} from './src/core/028_rnd_Signal.js';

const { shadowAtlasChanged } = LIGHTING_SIGNALS;

// In the shadow system, after a repack:
shadowAtlasChanged.pulse(currentFrame);
```

A subsystem that reacts to the pulse:

```
import { LIGHTING_SIGNALS } from './src/core/028_rnd_Signal.js';

const { shadowAtlasChanged } = LIGHTING_SIGNALS;

const conn = shadowAtlasChanged.connect(() => {
  giSystem.markAllProbesDirty();
}, null, 200);
```

A subsystem that wants to stop listening later:

```
conn.disconnect();
```

A signal with a value that carries state:

```
import { LIGHTING_SIGNALS } from './src/core/028_rnd_Signal.js';

const { dayCycleTick } = LIGHTING_SIGNALS;

// After computing the new day-cycle value:
dayCycleTick.emit(t, currentFrame);

// A late subscriber can read the last value:
console.log(dayCycleTick.value);
```

A derived signal that fires when any of several inputs change:

```
import {
  combine,
  LIGHTING_SIGNALS,
} from './src/core/028_rnd_Signal.js';

const anyLightingChange = combine('anyLightingChange', [
  LIGHTING_SIGNALS.shadowAtlasChanged,
  LIGHTING_SIGNALS.giProbeDirty,
  LIGHTING_SIGNALS.lightListChanged,
]);

anyLightingChange.connect(() => {
  // Re-batch the render state.
});
```

A subsystem that wants to prevent lower-priority listeners from running:

```
shadowAtlasChanged.connect((signal) => {
  if (isEmergencyMode()) {
    signal.stopPropagation();
  }
}, null, 0);
```

Because every signal in LIGHTING_SIGNALS is a singleton, the entire engine shares one instance per notification type. When the shadow system emits shadowAtlasChanged, every subscriber — GI, AO, post, debug HUD, regression capture — sees exactly the same event at exactly the same moment, in the exact order determined by priority. There is no per-subscriber queue, no topic name resolution, no Map lookup. Just a chain of function calls over a pre-allocated typed array.

The signal pattern is what makes the lighting stack's local notifications cheap. A signal emit costs a single _findInsertPos-free dispatch: one array iteration, zero allocations. On a typical frame, the lighting stack emits several hundred signals across all subsystems. Without this module, each one would involve a string comparison, a Map lookup, and a heap-allocated event object. With it, each one is a couple of integer loads and a function call.
