API Documentation — src/core/027_rnd_EventBus.js

File Purpose

This file is the high-performance event bus for the anime lighting stack. It provides a topic-based publish/subscribe system used by every lighting subsystem to signal state changes without direct coupling. When the shadow atlas rebuilds, when GI probes invalidate, when the quality level transitions, when the biome changes, when the light list is marked dirty, when the timer expires, when the WebGL context is lost — every one of those events flows through this bus.

The bus exists because direct coupling between subsystems on Android creates three distinct problems:

1. Circular imports — the shadow system needs to know when the light list changes, but the light manager needs to know when the shadow atlas is dirty. Both directions of dependency produce circular imports, which JavaScript handles but which make the module graph impossible to reason about.
2. Fan-out cost — when the biome changes, ten subsystems need to react. Without a bus, the biome system must import all ten and call them synchronously. With a bus, the biome system calls emit('env.biome.changed', payload) and the ten subscribers react on their own schedule.
3. Dispatch order control — some subscribers must run before others (shadow atlas must repack before GI probes bake, and GI must bake before AO blurs). Without a bus with priority ordering, the subscriber order depends on which module loaded first, which is fragile.

The bus solves all three with a design that is allocation-free on the hot path. Topics are compiled to integer ids at registration. Emitting an event touches only pre-allocated typed arrays. There are no Map lookups, no string comparisons, no closures on the hot path.

The bus provides two dispatch modes:

· emit() — synchronous dispatch to listeners in priority order. Used when the publisher must know all listeners ran before returning.
· post() — enqueue to a ring buffer and drain once per frame. Used for cross-domain fan-out so a single domain update does not cascade into unrelated systems.

The bus also supports wildcard topics: * receives every event, and category:* (e.g. lights:*) receives every event whose name starts with the prefix. Wildcards are used by the debug HUD, telemetry, and regression capture.

Listener leak detection is built in: each topic tracks a listener count, and when a topic exceeds a warning threshold the bus emits a warning through the logger.

---

Exported Constants

PERF_TIER_LOCAL

Internal. The cached PERF_TIER string from getPerfTier().

MAX_TOPICS

Type: number

Value: 128

The maximum number of registered topics. Topics are compiled to integer ids at registration. Sized to accommodate every named event the engine emits, with headroom for subsystem-specific events.

MAX_LISTENERS_PER_TOPIC

Type: number

Value: 32

The maximum number of listeners on a single topic. When a topic exceeds this, on() returns null and the bus warns.

MAX_QUEUE

Type: number

Value: 512 on HIGH, 256 on MEDIUM, 128 on LOW.

The size of the deferred event queue. Sized to accommodate the worst-case burst of deferred events during a biome transition or a chunk stream burst.

MAX_WILDCARDS

Type: number

Value: 16

The maximum number of registered wildcard patterns.

NO_PRIORITY

Type: number

Value: 1000

The default priority for listeners that do not specify one. Lower values run earlier.

EVENT_MODE

Type: frozen enum

Values:

· SYNC = 0 — synchronous dispatch.
· POST = 1 — deferred dispatch via the queue.
· BOTH = 2 — dispatch synchronously and also enqueue.

EVENT_SCOPE

Type: frozen enum

Values:

· GLOBAL = 0 — the event survives the frame boundary; it remains in the queue until drained.
· FRAME = 1 — the event is dropped at the end of the current frame if it was not drained.

DISPATCH_RESULT

Type: frozen enum

Values:

· OK = 0 — at least one listener ran.
· CANCELLED = 1 — a listener called stopPropagation().
· NO_TOPIC = 2 — the topic name was invalid or the topic registry is full.
· NO_LISTENERS = 3 — the topic exists but has no listeners.
· QUEUE_FULL = 4 — reserved.

LIGHTING_TOPIC

Type: frozen object

The canonical topic name registry. Every subsystem uses these strings instead of hard-coded literals, so a typo becomes a compile-time error instead of a runtime silent no-op.

Topic groups:

Lights:

· LIGHT_ADDED — 'lights.added'
· LIGHT_REMOVED — 'lights.removed'
· LIGHT_INTENSITY — 'lights.intensity'
· LIGHT_COLOR — 'lights.color'
· LIGHT_LIST_DIRTY — 'lights.list.dirty'
· LIGHT_CLUSTER_DIRTY — 'lights.cluster.dirty'

Shadows:

· SHADOW_ATLAS_DIRTY — 'shadows.atlas.dirty'
· SHADOW_MAP_RESIZED — 'shadows.map.resized'
· SHADOW_CASCADE_CHANGED — 'shadows.cascade.changed'
· SHADOW_FILTER_CHANGED — 'shadows.filter.changed'

GI:

· GI_PROBE_INVALIDATED — 'gi.probe.invalidated'
· GI_PROBE_REBAKED — 'gi.probe.rebaked'
· GI_BIOME_CHANGED — 'gi.biome.changed'
· GI_BUDGET_CHANGED — 'gi.budget.changed'

AO:

· AO_RES_CHANGED — 'ao.resolution.changed'
· AO_SAMPLE_CHANGED — 'ao.samples.changed'
· AO_DIRTY — 'ao.dirty'

Environment:

· ENV_DAYCYCLE_CHANGED — 'env.daycycle.changed'
· ENV_BIOME_CHANGED — 'env.biome.changed'
· ENV_PALETTE_CHANGED — 'env.palette.changed'
· ENV_WEATHER_CHANGED — 'env.weather.changed'

Interior/exterior:

· INTERIOR_ENTERED — 'interior.entered'
· INTERIOR_EXITED — 'interior.exited'
· EXTERIOR_CHANGED — 'exterior.changed'

Quality/tier:

· QUALITY_CHANGED — 'quality.changed'
· QUALITY_KNOB_CHANGED — 'quality.knob.changed'
· TIER_CHANGED — 'tier.changed'
· THERMAL_CHANGED — 'thermal.changed'
· BATTERY_CHANGED — 'battery.changed'

Frame lifecycle:

· FRAME_BEGIN — 'frame.begin'
· FRAME_END — 'frame.end'
· FRAME_HITCH — 'frame.hitch'
· FRAME_SLOW — 'frame.slow'

Context/lifecycle:

· CONTEXT_LOST — 'context.lost'
· CONTEXT_RESTORED — 'context.restored'
· VISIBILITY_HIDDEN — 'visibility.hidden'
· VISIBILITY_VISIBLE — 'visibility.visible'

Asset/manifest:

· ASSET_LOADED — 'asset.loaded'
· ASSET_FAILED — 'asset.failed'
· MANIFEST_COMPLETE — 'manifest.complete'

Director/debug:

· DIRECTOR_HINT — 'director.hint'
· DEBUG_VIEW_CHANGED — 'debug.view.changed'
· SCREENSHOT_REQUESTED — 'screenshot.requested'

Wildcards:

· WILDCARD_ALL — '*'
· WILDCARD_LIGHTS — 'lights:*'
· WILDCARD_SHADOWS — 'shadows:*'
· WILDCARD_GI — 'gi:*'
· WILDCARD_AO — 'ao:*'
· WILDCARD_ENV — 'env:*'
· WILDCARD_QUALITY — 'quality:*'
· WILDCARD_FRAME — 'frame:*'

---

Module-Level State (Not Exported Directly)

_defaultBus

Type: EventBus | null

The module-level singleton.

---

Internal Class — TopicSlot

One instance per registered topic.

Constructor

```
new TopicSlot(index, name)
```

Instance Properties

· index — the topic's array index.
· name — the topic name string.
· active — reserved.
· listenerFn — an array of listener functions.
· listenerCtx — an array of context objects, parallel to listenerFn.
· listenerPrio — an Int16Array of priorities, parallel.
· listenerOnce — a Uint8Array of once flags, parallel.
· listenerCount — the number of active listeners.
· emits — the total number of emits on this topic.
· syncEmits — the total number of synchronous emits.
· postedEmits — the total number of deferred emits.
· lastEmitFrame — the frame of the last emit.

Instance Methods

reset()

Returns: nothing. Clears every listener and counter.

---

Internal Class — WildcardSlot

One instance per registered wildcard pattern.

Constructor

```
new WildcardSlot(index, pattern)
```

Instance Properties

· index — the wildcard's array index.
· pattern — the pattern string.
· prefix — for xxx:* patterns, the prefix including the colon; null for *.
· listenerFn, listenerCtx, listenerPrio, listenerOnce, listenerCount — parallel listener arrays.
· emits — the total number of emits the wildcard has seen.

Instance Methods

matches(topicName)

Parameters: topicName — a topic name string.

Returns: boolean.

Purpose: pattern match. '*' matches everything. 'lights:*' matches any topic name that starts with 'lights:'.

reset()

Returns: nothing.

---

Internal Class — QueuedEvent

One instance per queue slot. Reused across frames.

Constructor

```
new QueuedEvent()
```

Instance Properties

· topicIdx — the topic index, or -1.
· payload — the payload reference.
· scope — one of EVENT_SCOPE.
· frame — the frame at enqueue time.

Instance Methods

reset()

Returns: nothing. Zeroes every field.

---

Exported Class — EventBus

The main bus.

Constructor

```
new EventBus(options = {})
```

Parameters:

· maxListenerWarn — the listener count at which a warning is logged. Default 24.
· logChannel — the logger channel to use. Default LOG_CHANNEL.CORE.
· strictTopics — reserved. Default false.
· enableWildcards — whether wildcards are allowed. Default true.
· autoDrain — reserved. Default true.

Constructor work:

1. Allocates topics — an array of 128 entries.
2. Initializes topicCount and topicByName (Map).
3. Allocates wildcards — an array of 16 entries.
4. Initializes wildcardCount.
5. Allocates queue — an array of MAX_QUEUE QueuedEvent instances.
6. Initializes the queue ring pointers.
7. Initializes frame, frameScopedHead.
8. Allocates _currentEvent — a shared dispatch context object.
9. Initializes the stats object.
10. Allocates _listeners (Map).
11. Lazily resolves the logger on first use.

Instance Properties

· topics — the topic slot array.
· topicCount — the number of registered topics.
· topicByName — the Map from name to index.
· wildcards — the wildcard slot array.
· wildcardCount — the number of registered wildcards.
· queue — the deferred event ring.
· queueHead, queueTail, queueCount, queueDropped — the ring pointers and drop counter.
· frame, frameScopedHead — the frame counter and the frame-scope boundary marker.
· stats — the aggregate stats object.

The stats object has totalEmits, totalSyncEmits, totalPostedEmits, totalDispatches, totalQueueDrains, totalDropped, totalCancelled, peakQueueCount.

Instance Methods

_log()

Internal. Lazily resolves the logger.

beginFrame(frameNumber)

Parameters: frameNumber — the current frame number.

Returns: this.

Purpose: sets the frame counter and records the current queue head as the frame-scope boundary marker.

endFrame()

Returns: this.

Purpose: drops every event in the queue that has scope === EVENT_SCOPE.FRAME. GLOBAL-scoped events survive the frame boundary. The compaction walks the ring and shifts surviving events to the front.

ensureTopic(name)

Parameters: name — the topic name string.

Returns: the topic index, or -1 if the registry is full.

Purpose: registers a topic on first use. Idempotent — subsequent calls with the same name return the existing index.

getTopicIndex(name)

Parameters: name — the topic name string.

Returns: the index, or -1 if unknown.

getTopicByName(name)

Parameters: name — the topic name string.

Returns: the TopicSlot, or null.

on(topicName, fn, ctx, priority, once)

Parameters:

· topicName — the topic name string.
· fn — the listener callback (payload, event) => void.
· ctx — the context object passed as this to fn.
· priority — the priority integer. Lower runs earlier.
· once — whether the listener is removed after the first call.

Returns: a token object with an unsubscribe method, or null on failure.

Purpose: registers a listener. Uses binary search to find the correct insertion position in the priority-ordered array. Shifts the tail of the array up to make room. Emits a warning if the topic exceeds maxListenerWarn.

once(topicName, fn, ctx, priority)

Convenience wrapper for on with once = true.

off(topicIdx, fn)

Parameters:

· topicIdx — the topic index.
· fn — the listener function.

Returns: boolean.

Purpose: removes a listener by function identity. Shifts the tail of the array down to close the gap.

offByName(topicName, fn)

Convenience wrapper for off with a name.

offAll(topicName)

Parameters: topicName — the topic name.

Returns: boolean.

Purpose: removes every listener on the topic.

_findInsertPos(slot, priority)

Internal. Binary search for the insertion position.

_shiftListenersUp(slot, pos)

Internal. Shifts listeners up to open a slot.

_shiftListenersDown(slot, pos)

Internal. Shifts listeners down to close a slot.

onWildcard(pattern, fn, ctx, priority, once)

Parameters: same as on, but the first argument is a wildcard pattern.

Returns: a token object with an unsubscribe method, or null.

Purpose: registers a wildcard listener. If the pattern ends with :*, extracts the prefix.

_addWildcardListener(slot, fn, ctx, priority, once)

Internal. Adds a listener to an existing wildcard slot.

offWildcard(pattern, fn)

Parameters:

· pattern — the wildcard pattern.
· fn — the listener function.

Returns: boolean.

emit(topicName, payload)

Parameters:

· topicName — the topic name.
· payload — the payload reference.

Returns: one of DISPATCH_RESULT.

Purpose: synchronous dispatch.

Flow:

1. Ensures the topic exists.
2. Increments the topic's emit counters.
3. If the topic has no listeners and there are no wildcards, returns NO_LISTENERS.
4. Sets up the shared dispatch context _currentEvent.
5. Iterates the topic listeners in priority order. Calls each one. If a listener sets the cancelled flag on the event, stops the loop. If a listener is once, removes it after the call.
6. If not cancelled, iterates the wildcards. For each matching wildcard, iterates its listeners in the same way.
7. Returns OK or CANCELLED.

post(topicName, payload, scope)

Parameters:

· topicName — the topic name.
· payload — the payload reference.
· scope — one of EVENT_SCOPE. Default GLOBAL.

Returns: one of DISPATCH_RESULT.

Purpose: enqueues a deferred event.

Flow:

1. Ensures the topic exists.
2. If the queue is full, drops the oldest event and increments the drop counter.
3. Writes the event into the ring at queueTail.
4. Advances the tail and increments the count.
5. Increments the topic's posted emit counter.

drain(maxEvents)

Parameters: maxEvents — the maximum number of events to dispatch.

Returns: the number of events dispatched.

Purpose: dispatches queued events in FIFO order. For each event, moves the head before dispatch so listeners can safely enqueue new events. Calls _dispatchFromQueue.

_dispatchFromQueue(slot, payload)

Internal. Dispatches a queued event to the topic's listeners. Same structure as the sync dispatch, but without emitting the "no listeners" result.

stopPropagation()

Returns: nothing.

Purpose: sets the cancelled flag on the current dispatch context. Any listener can call this during an emit to halt the rest of the chain.

getQueueCount()

Returns: the number of queued events.

getQueueCapacity()

Returns: MAX_QUEUE.

getQueueDropped()

Returns: the total number of dropped events.

clearQueue()

Returns: this. Resets every queued event and clears the ring.

getTopicCount()

Returns: the number of registered topics.

getWildcardCount()

Returns: the number of registered wildcards.

getListenerCount(topicName)

Parameters: topicName — the topic name.

Returns: the listener count, or 0.

hasTopic(topicName)

Parameters: topicName — the topic name.

Returns: boolean.

getStats()

Returns: an object with the aggregate counters, the queue status, and per-topic and per-wildcard summary arrays.

reset()

Returns: this. Resets every topic and wildcard, clears the queue, and zeroes the counters.

dispose()

Returns: this. Resets, nulls every internal array, and clears the topic and wildcard maps.

---

Exported Hot-Path Wrapper Functions

These delegate to the module-level singleton.

· eventBeginFrame(frameNumber)
· eventEndFrame()
· eventEmit(topicName, payload)
· eventPost(topicName, payload, scope)
· eventOn(topicName, fn, ctx, priority, once)
· eventOnce(topicName, fn, ctx, priority)
· eventOff(topicName, fn)
· eventOnWildcard(pattern, fn, ctx, priority, once)
· eventDrain(maxEvents)

---

Exported Functions

getDefaultEventBus()

Returns: the module-level singleton EventBus, creating it on first call.

disposeDefaultEventBus()

Returns: nothing.

createEventBus(options = {})

Returns: a new EventBus.

---

Default Export

The default export bundles: EventBus, LIGHTING_TOPIC, createEventBus, getDefaultEventBus, disposeDefaultEventBus, the nine event* hot-path wrappers, EVENT_MODE, EVENT_SCOPE, DISPATCH_RESULT, MAX_TOPICS, MAX_LISTENERS_PER_TOPIC, MAX_QUEUE, MAX_WILDCARDS, NO_PRIORITY.

---

Usage Pattern

A subsystem that publishes an event:

```
import {
  getDefaultEventBus,
  LIGHTING_TOPIC,
} from './src/core/027_rnd_EventBus.js';

const bus = getDefaultEventBus();

// When the biome changes:
bus.emit(LIGHTING_TOPIC.ENV_BIOME_CHANGED, { from: 'desert', to: 'snow' });

// When the light list is dirty:
bus.post(LIGHTING_TOPIC.LIGHT_LIST_DIRTY, null, EVENT_SCOPE.GLOBAL);
```

A subsystem that reacts to an event:

```
bus.on(LIGHTING_TOPIC.SHADOW_ATLAS_DIRTY, (payload) => {
  shadowAtlas.markDirty(payload.regions);
}, null, 100 /* priority: run early */);
```

A debug HUD that captures every event:

```
bus.onWildcard('*', (payload, event) => {
  console.log(`Event: ${event.topicName}`);
});
```

A subsystem that wants to prevent lower-priority listeners from running:

```
bus.on(LIGHTING_TOPIC.QUALITY_CHANGED, (payload, event) => {
  if (payload.severity >= 3) {
    emergencyDowngrade();
    event.cancelled = true;
  }
}, null, 0 /* highest priority */);
```

The bus is what decouples the lighting stack. The shadow system does not import the GI system; it emits shadows.atlas.dirty and the GI system, which subscribes to that topic, decides whether and when to react. When a subsystem is added, it registers its listeners; when a subsystem is removed, its listeners are dropped. The engine's coupling is entirely through topic names in the LIGHTING_TOPIC registry, so refactoring one subsystem never requires touching another.
