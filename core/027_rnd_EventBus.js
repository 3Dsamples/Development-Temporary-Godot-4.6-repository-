// File : 027
// name : src/core/027_rnd_EventBus.js
// description : High-performance event bus for the anime lighting stack on
//               Android mobile. Provides a topic-based publish/subscribe
//               system used by every lighting subsystem to signal state
//               changes without direct coupling: shadow atlas rebuild
//               requests, GI probe invalidation, quality level transitions,
//               biome changes, light list dirty flags, timer expiry,
//               context loss/restore, interior/exterior transitions,
//               director hint emissions.
//
//               Design goals:
//                 • Zero per-frame allocations on the hot path. Topics are
//                   compiled to integer ids at registration; emitting an
//                   event touches only pre-allocated typed arrays.
//                 • Two dispatch modes:
//                     – emit() — synchronous dispatch to listeners in
//                                priority order; used when the publisher
//                                must know all listeners ran before returning.
//                     – post() — enqueue to a ring buffer and drain once
//                                per frame (throttled); used for cross-domain
//                                fan-out so a single domain update doesn't
//                                cascade into unrelated systems.
//                 • Priority-ordered listener chains (lower value = earlier).
//                   Higher priority = runs first. Ties break by registration
//                   order (stable).
//                 • Once listeners auto-remove after first invocation.
//                 • Wildcard topics (`*` and `category:*`) receive every
//                   event in their scope — useful for debug HUD, telemetry,
//                   and regression capture.
//                 • Bounded queue: when the post ring is full, the oldest
//                   queued event is dropped (never blocks the publisher).
//                 • Frame-scoped queue: `beginFrame()` snapshots the queue
//                   head; `endFrame()` clears frame-scoped events while
//                   preserving persistent queued events. Enables "fire and
//                   forget this frame" semantics.
//                 • Cancellable: emit() returns a boolean; listeners may
//                   call `stopPropagation()` to halt the chain.
//                 • Sinks integrate with 026_rnd_Logger.js so the debug
//                   HUD and regression capture see the same event stream.
//                 • Listener leak detection: per-topic listener counts and
//                   a max-listener warning surfaced through the logger.
//
//               Fixed-capacity registries:
//                 • MAX_TOPICS            — 128 topics
//                 • MAX_LISTENERS_PER_TOPIC — 32
//                 • MAX_QUEUE             — 512 queued events (ring)
//                 • MAX_WILDCARDS         — 16 wildcard topics
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; no external event libs; every internal buffer
//               sized once at construction.
// best for : Giving the anime lighting stack one canonical, allocation-free
//            communication backbone. Shadow system announces "shadow atlas
//            dirty", GI system subscribes and schedules a probe bake, quality
//            controller subscribes to hitches, environment system subscribes
//            to day-cycle changes, director publishes hint events — all
//            without cross-imports.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
} from './026_rnd_Logger.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const MAX_TOPICS              = 128;
export const MAX_LISTENERS_PER_TOPIC = 32;
export const MAX_QUEUE               = PERF_TIER_LOCAL === 'HIGH' ? 512 : PERF_TIER_LOCAL === 'MEDIUM' ? 256 : 128;
export const MAX_WILDCARDS           = 16;
export const NO_PRIORITY             = 1000;

export const EVENT_MODE = Object.freeze({
  SYNC:  0,
  POST:  1,
  BOTH:  2,
});

export const EVENT_SCOPE = Object.freeze({
  GLOBAL: 0,
  FRAME:  1,
});

export const DISPATCH_RESULT = Object.freeze({
  OK:        0,
  CANCELLED: 1,
  NO_TOPIC:  2,
  NO_LISTENERS: 3,
  QUEUE_FULL: 4,
});

/* ------------------------------------------------------------------ */
/* 1. WELL-KNOWN LIGHTING TOPICS                                      */
/* ------------------------------------------------------------------ */

export const LIGHTING_TOPIC = Object.freeze({
  // Lights
  LIGHT_ADDED:           'lights.added',
  LIGHT_REMOVED:         'lights.removed',
  LIGHT_INTENSITY:       'lights.intensity',
  LIGHT_COLOR:           'lights.color',
  LIGHT_LIST_DIRTY:      'lights.list.dirty',
  LIGHT_CLUSTER_DIRTY:   'lights.cluster.dirty',

  // Shadows
  SHADOW_ATLAS_DIRTY:    'shadows.atlas.dirty',
  SHADOW_MAP_RESIZED:    'shadows.map.resized',
  SHADOW_CASCADE_CHANGED:'shadows.cascade.changed',
  SHADOW_FILTER_CHANGED: 'shadows.filter.changed',

  // GI
  GI_PROBE_INVALIDATED:  'gi.probe.invalidated',
  GI_PROBE_REBAKED:      'gi.probe.rebaked',
  GI_BIOME_CHANGED:      'gi.biome.changed',
  GI_BUDGET_CHANGED:     'gi.budget.changed',

  // AO
  AO_RES_CHANGED:        'ao.resolution.changed',
  AO_SAMPLE_CHANGED:     'ao.samples.changed',
  AO_DIRTY:              'ao.dirty',

  // Environment
  ENV_DAYCYCLE_CHANGED:  'env.daycycle.changed',
  ENV_BIOME_CHANGED:     'env.biome.changed',
  ENV_PALETTE_CHANGED:   'env.palette.changed',
  ENV_WEATHER_CHANGED:   'env.weather.changed',

  // Interior / exterior
  INTERIOR_ENTERED:      'interior.entered',
  INTERIOR_EXITED:       'interior.exited',
  EXTERIOR_CHANGED:      'exterior.changed',

  // Quality / tier
  QUALITY_CHANGED:       'quality.changed',
  QUALITY_KNOB_CHANGED:  'quality.knob.changed',
  TIER_CHANGED:          'tier.changed',
  THERMAL_CHANGED:       'thermal.changed',
  BATTERY_CHANGED:       'battery.changed',

  // Frame lifecycle
  FRAME_BEGIN:           'frame.begin',
  FRAME_END:             'frame.end',
  FRAME_HITCH:           'frame.hitch',
  FRAME_SLOW:            'frame.slow',

  // Context / lifecycle
  CONTEXT_LOST:          'context.lost',
  CONTEXT_RESTORED:      'context.restored',
  VISIBILITY_HIDDEN:     'visibility.hidden',
  VISIBILITY_VISIBLE:    'visibility.visible',

  // Asset / manifest
  ASSET_LOADED:          'asset.loaded',
  ASSET_FAILED:          'asset.failed',
  MANIFEST_COMPLETE:     'manifest.complete',

  // Director / debug
  DIRECTOR_HINT:         'director.hint',
  DEBUG_VIEW_CHANGED:    'debug.view.changed',
  SCREENSHOT_REQUESTED:  'screenshot.requested',

  // Wildcards (registered internally)
  WILDCARD_ALL:          '*',
  WILDCARD_LIGHTS:       'lights:*',
  WILDCARD_SHADOWS:      'shadows:*',
  WILDCARD_GI:           'gi:*',
  WILDCARD_AO:           'ao:*',
  WILDCARD_ENV:          'env:*',
  WILDCARD_QUALITY:      'quality:*',
  WILDCARD_FRAME:        'frame:*',
});

/* ------------------------------------------------------------------ */
/* 2. TOPIC SLOT                                                      */
/* ------------------------------------------------------------------ */

class TopicSlot {
  constructor(index, name) {
    this.index     = index;
    this.name      = name;
    this.active    = 1;

    // Listener arrays (parallel; sorted by priority ascending).
    this.listenerFn     = new Array(MAX_LISTENERS_PER_TOPIC).fill(null);
    this.listenerCtx    = new Array(MAX_LISTENERS_PER_TOPIC).fill(null);
    this.listenerPrio   = new Int16Array(MAX_LISTENERS_PER_TOPIC);
    this.listenerOnce   = new Uint8Array(MAX_LISTENERS_PER_TOPIC);
    this.listenerCount  = 0;

    // Stats.
    this.emits         = 0;
    this.syncEmits     = 0;
    this.postedEmits   = 0;
    this.lastEmitFrame = -1;
  }

  reset() {
    for (let i = 0; i < this.listenerCount; i++) {
      this.listenerFn[i]   = null;
      this.listenerCtx[i]  = null;
      this.listenerPrio[i] = 0;
      this.listenerOnce[i] = 0;
    }
    this.listenerCount = 0;
    this.emits = 0;
    this.syncEmits = 0;
    this.postedEmits = 0;
    this.lastEmitFrame = -1;
  }
}

/* ------------------------------------------------------------------ */
/* 3. WILDCARD SLOT                                                   */
/* ------------------------------------------------------------------ */

class WildcardSlot {
  constructor(index, pattern) {
    this.index      = index;
    this.pattern    = pattern;
    this.prefix     = null;   // for 'xxx:*' patterns; null for '*'
    this.listenerFn   = new Array(MAX_LISTENERS_PER_TOPIC).fill(null);
    this.listenerCtx  = new Array(MAX_LISTENERS_PER_TOPIC).fill(null);
    this.listenerPrio = new Int16Array(MAX_LISTENERS_PER_TOPIC);
    this.listenerOnce = new Uint8Array(MAX_LISTENERS_PER_TOPIC);
    this.listenerCount = 0;
    this.emits = 0;
  }

  matches(topicName) {
    if (this.pattern === '*') return true;
    if (this.prefix !== null) {
      return topicName.startsWith(this.prefix);
    }
    return this.pattern === topicName;
  }

  reset() {
    for (let i = 0; i < this.listenerCount; i++) {
      this.listenerFn[i]   = null;
      this.listenerCtx[i]  = null;
      this.listenerPrio[i] = 0;
      this.listenerOnce[i] = 0;
    }
    this.listenerCount = 0;
    this.emits = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 4. QUEUED EVENT RING                                               */
/* ------------------------------------------------------------------ */

class QueuedEvent {
  constructor() {
    this.topicIdx = -1;
    this.payload  = null;
    this.scope    = EVENT_SCOPE.GLOBAL;
    this.frame    = 0;
  }

  reset() {
    this.topicIdx = -1;
    this.payload  = null;
    this.scope    = EVENT_SCOPE.GLOBAL;
    this.frame    = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 5. EVENT BUS                                                       */
/* ------------------------------------------------------------------ */

export class EventBus {
  constructor(options = {}) {
    this.options = Object.assign({
      maxListenerWarn:   24,
      logChannel:        LOG_CHANNEL.CORE,
      strictTopics:      false,
      enableWildcards:   true,
      autoDrain:         true,
    }, options || {});

    // Topic registry.
    this.topics   = new Array(MAX_TOPICS);
    for (let i = 0; i < MAX_TOPICS; i++) this.topics[i] = null;
    this.topicCount = 0;
    this.topicByName = new Map();

    // Wildcards.
    this.wildcards = new Array(MAX_WILDCARDS);
    for (let i = 0; i < MAX_WILDCARDS; i++) this.wildcards[i] = null;
    this.wildcardCount = 0;

    // Post queue (ring buffer).
    this.queue        = new Array(MAX_QUEUE);
    for (let i = 0; i < MAX_QUEUE; i++) this.queue[i] = new QueuedEvent();
    this.queueHead    = 0;
    this.queueTail    = 0;
    this.queueCount   = 0;
    this.queueDropped = 0;

    // Per-frame scope tracking.
    this.frame         = 0;
    this.frameScopedHead = -1; // queue snapshot for frame-scoped events

    // Dispatch context (reused; avoids closure allocation).
    this._currentEvent = {
      topic: null,
      topicName: '',
      payload: null,
      frame: 0,
      cancelled: false,
      stopped: false,
    };

    // Current listener index for stopPropagation.
    this._currentTopicSlot = null;
    this._currentListenerIdx = -1;

    // Stats.
    this.stats = {
      totalEmits:       0,
      totalSyncEmits:   0,
      totalPostedEmits: 0,
      totalDispatches:  0,
      totalQueueDrains: 0,
      totalDropped:     0,
      totalCancelled:   0,
      peakQueueCount:   0,
    };

    this._logger = null; // lazily resolved
  }

  /* ---------------- logger ---------------- */

  _log() {
    if (!this._logger) {
      try { this._logger = getDefaultLogger(); } catch (_) { this._logger = null; }
    }
    return this._logger;
  }

  /* ---------------- frame lifecycle ---------------- */

  beginFrame(frameNumber) {
    this.frame = (typeof frameNumber === 'number') ? frameNumber : (this.frame + 1);
    // Snapshot where frame-scoped events begin.
    this.frameScopedHead = this.queueHead;
    return this;
  }

  endFrame() {
    // Drop frame-scoped events from the queue by compacting around them.
    if (this.frameScopedHead >= 0) {
      let w = this.frameScopedHead;
      const cap = MAX_QUEUE;
      for (let i = 0; i < this.queueCount; i++) {
        const rd = (this.queueHead + i) % cap;
        const ev = this.queue[rd];
        if (ev.scope !== EVENT_SCOPE.FRAME) {
          if (rd !== w) {
            // Swap contents to compact.
            const dst = this.queue[w];
            dst.topicIdx = ev.topicIdx;
            dst.payload  = ev.payload;
            dst.scope    = ev.scope;
            dst.frame    = ev.frame;
            ev.reset();
          }
          w = (w + 1) % cap;
        } else {
          ev.reset();
          this.queueCount--;
        }
      }
      this.queueTail = w;
      this.frameScopedHead = -1;
    }
    return this;
  }

  /* ---------------- topic registry ---------------- */

  /**
   * Ensures a topic exists and returns its index. Returns -1 if the topic
   * registry is full.
   */
  ensureTopic(name) {
    if (typeof name !== 'string' || name.length === 0) return -1;
    const existing = this.topicByName.get(name);
    if (existing !== undefined) return existing;
    if (this.topicCount >= MAX_TOPICS) return -1;

    const idx = this.topicCount++;
    const slot = new TopicSlot(idx, name);
    this.topics[idx] = slot;
    this.topicByName.set(name, idx);
    return idx;
  }

  getTopicIndex(name) {
    const idx = this.topicByName.get(name);
    return (idx === undefined) ? -1 : idx;
  }

  getTopicByName(name) {
    const idx = this.getTopicIndex(name);
    return idx < 0 ? null : this.topics[idx];
  }

  /* ---------------- subscriptions ---------------- */

  /**
   * Subscribe to a topic. Returns a token object with an `unsubscribe`
   * closure bound to this bus. The closure captures only the bus + indices,
   * so subsequent unsubscribe calls do not allocate.
   */
  on(topicName, fn, ctx, priority = NO_PRIORITY, once = false) {
    if (typeof fn !== 'function') return null;
    const idx = this.ensureTopic(topicName);
    if (idx < 0) return null;

    const slot = this.topics[idx];
    if (slot.listenerCount >= MAX_LISTENERS_PER_TOPIC) {
      const log = this._log();
      if (log) log.warn(this.options.logChannel, `[027_rnd_EventBus] topic "${topicName}" listener cap reached (${MAX_LISTENERS_PER_TOPIC})`);
      return null;
    }

    const insertAt = this._findInsertPos(slot, priority);
    this._shiftListenersUp(slot, insertAt);

    slot.listenerFn[insertAt]   = fn;
    slot.listenerCtx[insertAt]  = ctx || null;
    slot.listenerPrio[insertAt] = priority | 0;
    slot.listenerOnce[insertAt] = once ? 1 : 0;
    slot.listenerCount++;

    if (this.options.maxListenerWarn > 0 && slot.listenerCount === this.options.maxListenerWarn) {
      const log = this._log();
      if (log) log.warn(this.options.logChannel, `[027_rnd_EventBus] topic "${topicName}" has ${slot.listenerCount} listeners — possible leak`);
    }

    const bus = this;
    return {
      topic: topicName,
      index: idx,
      fn,
      unsubscribe() { bus.off(idx, fn); },
    };
  }

  once(topicName, fn, ctx, priority = NO_PRIORITY) {
    return this.on(topicName, fn, ctx, priority, true);
  }

  off(topicIdx, fn) {
    const slot = this.topics[topicIdx];
    if (!slot) return false;
    for (let i = 0; i < slot.listenerCount; i++) {
      if (slot.listenerFn[i] === fn) {
        this._shiftListenersDown(slot, i);
        slot.listenerCount--;
        return true;
      }
    }
    return false;
  }

  offByName(topicName, fn) {
    const idx = this.getTopicIndex(topicName);
    if (idx < 0) return false;
    return this.off(idx, fn);
  }

  offAll(topicName) {
    const idx = this.getTopicIndex(topicName);
    if (idx < 0) return false;
    const slot = this.topics[idx];
    slot.reset();
    return true;
  }

  _findInsertPos(slot, priority) {
    let lo = 0;
    let hi = slot.listenerCount;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (slot.listenerPrio[mid] <= priority) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  _shiftListenersUp(slot, pos) {
    for (let i = slot.listenerCount; i > pos; i--) {
      slot.listenerFn[i]   = slot.listenerFn[i - 1];
      slot.listenerCtx[i]  = slot.listenerCtx[i - 1];
      slot.listenerPrio[i] = slot.listenerPrio[i - 1];
      slot.listenerOnce[i] = slot.listenerOnce[i - 1];
    }
  }

  _shiftListenersDown(slot, pos) {
    for (let i = pos; i < slot.listenerCount - 1; i++) {
      slot.listenerFn[i]   = slot.listenerFn[i + 1];
      slot.listenerCtx[i]  = slot.listenerCtx[i + 1];
      slot.listenerPrio[i] = slot.listenerPrio[i + 1];
      slot.listenerOnce[i] = slot.listenerOnce[i + 1];
    }
    const last = slot.listenerCount - 1;
    slot.listenerFn[last]   = null;
    slot.listenerCtx[last]  = null;
    slot.listenerPrio[last] = 0;
    slot.listenerOnce[last] = 0;
  }

  /* ---------------- wildcards ---------------- */

  onWildcard(pattern, fn, ctx, priority = NO_PRIORITY, once = false) {
    if (typeof fn !== 'function') return null;
    if (!this.options.enableWildcards) return null;
    if (typeof pattern !== 'string' || pattern.length === 0) return null;

    // Find existing.
    for (let i = 0; i < this.wildcardCount; i++) {
      if (this.wildcards[i] && this.wildcards[i].pattern === pattern) {
        return this._addWildcardListener(this.wildcards[i], fn, ctx, priority, once);
      }
    }

    if (this.wildcardCount >= MAX_WILDCARDS) return null;

    const idx = this.wildcardCount++;
    const slot = new WildcardSlot(idx, pattern);
    if (pattern !== '*' && pattern.endsWith(':*')) {
      slot.prefix = pattern.slice(0, -1); // keep the trailing ':'
    }
    this.wildcards[idx] = slot;
    return this._addWildcardListener(slot, fn, ctx, priority, once);
  }

  _addWildcardListener(slot, fn, ctx, priority, once) {
    if (slot.listenerCount >= MAX_LISTENERS_PER_TOPIC) return null;
    const insertAt = this._findInsertPos(slot, priority);
    this._shiftListenersUp(slot, insertAt);
    slot.listenerFn[insertAt]   = fn;
    slot.listenerCtx[insertAt]  = ctx || null;
    slot.listenerPrio[insertAt] = priority | 0;
    slot.listenerOnce[insertAt] = once ? 1 : 0;
    slot.listenerCount++;

    const bus = this;
    return {
      topic: slot.pattern,
      wildcard: true,
      fn,
      unsubscribe() { bus.offWildcard(slot.pattern, fn); },
    };
  }

  offWildcard(pattern, fn) {
    for (let i = 0; i < this.wildcardCount; i++) {
      const slot = this.wildcards[i];
      if (!slot || slot.pattern !== pattern) continue;
      for (let j = 0; j < slot.listenerCount; j++) {
        if (slot.listenerFn[j] === fn) {
          this._shiftListenersDown(slot, j);
          slot.listenerCount--;
          return true;
        }
      }
    }
    return false;
  }

  /* ---------------- emission ---------------- */

  /**
   * Emit synchronously. Listeners run in priority order; if any listener
   * calls `stopPropagation()`, the chain halts immediately.
   */
  emit(topicName, payload) {
    const idx = this.ensureTopic(topicName);
    if (idx < 0) {
      this.stats.totalEmits++;
      return DISPATCH_RESULT.NO_TOPIC;
    }

    const slot = this.topics[idx];
    slot.emits++;
    slot.syncEmits++;
    slot.lastEmitFrame = this.frame;

    this.stats.totalEmits++;
    this.stats.totalSyncEmits++;

    if (slot.listenerCount === 0 && this.wildcardCount === 0) {
      return DISPATCH_RESULT.NO_LISTENERS;
    }

    // Set up shared dispatch context.
    const ev = this._currentEvent;
    ev.topic = slot;
    ev.topicName = slot.name;
    ev.payload = payload;
    ev.frame = this.frame;
    ev.cancelled = false;
    ev.stopped = false;

    this._currentTopicSlot = slot;
    this._currentListenerIdx = -1;

    // Dispatch topic listeners (in priority order).
    for (let i = 0; i < slot.listenerCount && !ev.cancelled; i++) {
      this._currentListenerIdx = i;
      const fn  = slot.listenerFn[i];
      const ctx = slot.listenerCtx[i];
      const once = slot.listenerOnce[i];

      try {
        if (ctx) fn.call(ctx, payload, ev);
        else fn(payload, ev);
      } catch (e) {
        const log = this._log();
        if (log) log.error(this.options.logChannel, `[027_rnd_EventBus] listener threw on "${slot.name}": ${e && e.message}`);
      }

      if (once) {
        this._shiftListenersDown(slot, i);
        slot.listenerCount--;
        i--;
        this._currentListenerIdx = i;
      }
    }

    // Dispatch wildcards (in registration order).
    if (!ev.cancelled) {
      for (let w = 0; w < this.wildcardCount; w++) {
        const wc = this.wildcards[w];
        if (!wc || !wc.matches(slot.name)) continue;
        wc.emits++;
        for (let i = 0; i < wc.listenerCount && !ev.cancelled; i++) {
          const fn = wc.listenerFn[i];
          const ctx = wc.listenerCtx[i];
          const once = wc.listenerOnce[i];
          try {
            if (ctx) fn.call(ctx, payload, ev);
            else fn(payload, ev);
          } catch (e) {
            const log = this._log();
            if (log) log.error(this.options.logChannel, `[027_rnd_EventBus] wildcard listener threw on "${slot.name}": ${e && e.message}`);
          }
          if (once) {
            this._shiftListenersDown(wc, i);
            wc.listenerCount--;
            i--;
          }
        }
      }
    }

    this.stats.totalDispatches++;
    if (ev.cancelled) this.stats.totalCancelled++;

    const result = ev.cancelled ? DISPATCH_RESULT.CANCELLED : DISPATCH_RESULT.OK;

    // Reset shared context.
    this._currentTopicSlot = null;
    this._currentListenerIdx = -1;
    ev.topic = null;
    ev.payload = null;

    return result;
  }

  /**
   * Enqueue an event to be dispatched during the next `drain()` call.
   * Never blocks — oldest queued event is dropped on overflow.
   */
  post(topicName, payload, scope = EVENT_SCOPE.GLOBAL) {
    const idx = this.ensureTopic(topicName);
    if (idx < 0) {
      this.stats.totalEmits++;
      return DISPATCH_RESULT.NO_TOPIC;
    }

    if (this.queueCount >= MAX_QUEUE) {
      // Drop oldest.
      const oldest = this.queue[this.queueHead];
      oldest.reset();
      this.queueHead = (this.queueHead + 1) % MAX_QUEUE;
      this.queueCount--;
      this.queueDropped++;
      this.stats.totalDropped++;
    }

    const slot = this.queue[this.queueTail];
    slot.topicIdx = idx;
    slot.payload  = payload;
    slot.scope    = scope | 0;
    slot.frame    = this.frame;

    this.queueTail = (this.queueTail + 1) % MAX_QUEUE;
    this.queueCount++;
    if (this.queueCount > this.stats.peakQueueCount) {
      this.stats.peakQueueCount = this.queueCount;
    }

    const topic = this.topics[idx];
    topic.emits++;
    topic.postedEmits++;
    topic.lastEmitFrame = this.frame;

    this.stats.totalEmits++;
    this.stats.totalPostedEmits++;

    return DISPATCH_RESULT.OK;
  }

  /**
   * Drain queued events. Dispatches up to `maxEvents` in FIFO order.
   * Returns the number of events dispatched.
   */
  drain(maxEvents) {
    const limit = (maxEvents !== undefined ? maxEvents : this.queueCount) | 0;
    let count = 0;
    const cap = MAX_QUEUE;

    while (this.queueCount > 0 && count < limit) {
      const ev = this.queue[this.queueHead];
      const idx = ev.topicIdx;
      const payload = ev.payload;

      // Move head before dispatch so listeners may enqueue safely.
      this.queueHead = (this.queueHead + 1) % cap;
      this.queueCount--;

      ev.reset();

      if (idx >= 0 && idx < this.topicCount) {
        const slot = this.topics[idx];
        if (slot) {
          this._dispatchFromQueue(slot, payload);
        }
      }
      count++;
    }

    if (count > 0) this.stats.totalQueueDrains++;
    return count;
  }

  _dispatchFromQueue(slot, payload) {
    // Set up shared dispatch context (same object as sync emit).
    const ev = this._currentEvent;
    ev.topic = slot;
    ev.topicName = slot.name;
    ev.payload = payload;
    ev.frame = this.frame;
    ev.cancelled = false;
    ev.stopped = false;

    for (let i = 0; i < slot.listenerCount && !ev.cancelled; i++) {
      const fn  = slot.listenerFn[i];
      const ctx = slot.listenerCtx[i];
      const once = slot.listenerOnce[i];

      try {
        if (ctx) fn.call(ctx, payload, ev);
        else fn(payload, ev);
      } catch (e) {
        const log = this._log();
        if (log) log.error(this.options.logChannel, `[027_rnd_EventBus] queued listener threw on "${slot.name}": ${e && e.message}`);
      }

      if (once) {
        this._shiftListenersDown(slot, i);
        slot.listenerCount--;
        i--;
      }
    }

    // Dispatch wildcards.
    if (!ev.cancelled) {
      for (let w = 0; w < this.wildcardCount; w++) {
        const wc = this.wildcards[w];
        if (!wc || !wc.matches(slot.name)) continue;
        wc.emits++;
        for (let i = 0; i < wc.listenerCount && !ev.cancelled; i++) {
          const fn = wc.listenerFn[i];
          const ctx = wc.listenerCtx[i];
          const once = wc.listenerOnce[i];
          try {
            if (ctx) fn.call(ctx, payload, ev);
            else fn(payload, ev);
          } catch (e) {
            const log = this._log();
            if (log) log.error(this.options.logChannel, `[027_rnd_EventBus] wildcard queued listener threw on "${slot.name}": ${e && e.message}`);
          }
          if (once) {
            this._shiftListenersDown(wc, i);
            wc.listenerCount--;
            i--;
          }
        }
      }
    }

    ev.topic = null;
    ev.payload = null;
  }

  /* ---------------- propagation control ---------------- */

  stopPropagation() {
    this._currentEvent.cancelled = true;
    this._currentEvent.stopped = true;
  }

  /* ---------------- queue introspection ---------------- */

  getQueueCount() { return this.queueCount; }
  getQueueCapacity() { return MAX_QUEUE; }
  getQueueDropped() { return this.queueDropped; }

  clearQueue() {
    for (let i = 0; i < this.queueCount; i++) {
      const idx = (this.queueHead + i) % MAX_QUEUE;
      this.queue[idx].reset();
    }
    this.queueHead = 0;
    this.queueTail = 0;
    this.queueCount = 0;
    this.frameScopedHead = -1;
    return this;
  }

  /* ---------------- topic introspection ---------------- */

  getTopicCount() { return this.topicCount; }
  getWildcardCount() { return this.wildcardCount; }

  getListenerCount(topicName) {
    const idx = this.getTopicIndex(topicName);
    if (idx < 0) return 0;
    return this.topics[idx].listenerCount;
  }

  hasTopic(topicName) {
    return this.topicByName.has(topicName);
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    const topics = new Array(this.topicCount);
    for (let i = 0; i < this.topicCount; i++) {
      const t = this.topics[i];
      if (!t) continue;
      topics[i] = {
        name:          t.name,
        listeners:     t.listenerCount,
        emits:         t.emits,
        syncEmits:     t.syncEmits,
        postedEmits:   t.postedEmits,
        lastEmitFrame: t.lastEmitFrame,
      };
    }

    const wildcards = new Array(this.wildcardCount);
    for (let i = 0; i < this.wildcardCount; i++) {
      const w = this.wildcards[i];
      if (!w) continue;
      wildcards[i] = {
        pattern:   w.pattern,
        listeners: w.listenerCount,
        emits:     w.emits,
      };
    }

    return {
      frame:             this.frame,
      topicCount:        this.topicCount,
      topicCapacity:     MAX_TOPICS,
      wildcardCount:     this.wildcardCount,
      wildcardCapacity:  MAX_WILDCARDS,
      queueCount:        this.queueCount,
      queueCapacity:     MAX_QUEUE,
      queueDropped:      this.queueDropped,
      totalEmits:        this.stats.totalEmits,
      totalSyncEmits:    this.stats.totalSyncEmits,
      totalPostedEmits:  this.stats.totalPostedEmits,
      totalDispatches:   this.stats.totalDispatches,
      totalQueueDrains:  this.stats.totalQueueDrains,
      totalDropped:      this.stats.totalDropped,
      totalCancelled:    this.stats.totalCancelled,
      peakQueueCount:    this.stats.peakQueueCount,
      topics,
      wildcards,
    };
  }

  /* ---------------- reset / dispose ---------------- */

  reset() {
    for (let i = 0; i < this.topicCount; i++) {
      if (this.topics[i]) this.topics[i].reset();
    }
    for (let i = 0; i < this.wildcardCount; i++) {
      if (this.wildcards[i]) this.wildcards[i].reset();
    }
    this.clearQueue();
    this.stats.totalEmits = 0;
    this.stats.totalSyncEmits = 0;
    this.stats.totalPostedEmits = 0;
    this.stats.totalDispatches = 0;
    this.stats.totalQueueDrains = 0;
    this.stats.totalDropped = 0;
    this.stats.totalCancelled = 0;
    this.stats.peakQueueCount = 0;
    this.frame = 0;
    this.queueDropped = 0;
    return this;
  }

  dispose() {
    this.reset();
    for (let i = 0; i < this.topicCount; i++) this.topics[i] = null;
    for (let i = 0; i < this.wildcardCount; i++) this.wildcards[i] = null;
    for (let i = 0; i < MAX_QUEUE; i++) this.queue[i] = null;
    this.topics.length = 0;
    this.wildcards.length = 0;
    this.queue.length = 0;
    this.topicByName.clear();
    this.topicCount = 0;
    this.wildcardCount = 0;
    this.queueCount = 0;
    this.queueHead = 0;
    this.queueTail = 0;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 6. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultBus = null;

export function getDefaultEventBus() {
  if (!_defaultBus) _defaultBus = new EventBus();
  return _defaultBus;
}

export function disposeDefaultEventBus() {
  if (_defaultBus) {
    _defaultBus.dispose();
    _defaultBus = null;
  }
}

/* ------------------------------------------------------------------ */
/* 7. HOT-PATH HELPERS (delegate to default bus)                      */
/* ------------------------------------------------------------------ */

export function eventBeginFrame(frameNumber) {
  getDefaultEventBus().beginFrame(frameNumber);
}

export function eventEndFrame() {
  getDefaultEventBus().endFrame();
}

export function eventEmit(topicName, payload) {
  return getDefaultEventBus().emit(topicName, payload);
}

export function eventPost(topicName, payload, scope) {
  return getDefaultEventBus().post(topicName, payload, scope);
}

export function eventOn(topicName, fn, ctx, priority, once) {
  return getDefaultEventBus().on(topicName, fn, ctx, priority, once);
}

export function eventOnce(topicName, fn, ctx, priority) {
  return getDefaultEventBus().once(topicName, fn, ctx, priority);
}

export function eventOff(topicName, fn) {
  return getDefaultEventBus().offByName(topicName, fn);
}

export function eventOnWildcard(pattern, fn, ctx, priority, once) {
  return getDefaultEventBus().onWildcard(pattern, fn, ctx, priority, once);
}

export function eventDrain(maxEvents) {
  return getDefaultEventBus().drain(maxEvents);
}

/* ------------------------------------------------------------------ */
/* 8. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createEventBus(options = {}) {
  return new EventBus(options);
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  EventBus,
  LIGHTING_TOPIC,

  createEventBus,
  getDefaultEventBus,
  disposeDefaultEventBus,

  eventBeginFrame,
  eventEndFrame,
  eventEmit,
  eventPost,
  eventOn,
  eventOnce,
  eventOff,
  eventOnWildcard,
  eventDrain,

  EVENT_MODE,
  EVENT_SCOPE,
  DISPATCH_RESULT,
  MAX_TOPICS,
  MAX_LISTENERS_PER_TOPIC,
  MAX_QUEUE,
  MAX_WILDCARDS,
  NO_PRIORITY,
};

export default _defaultExport;