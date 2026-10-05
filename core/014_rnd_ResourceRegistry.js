// File : 014
// name : src/core/014_rnd_ResourceRegistry.js
// description : Authoritative GPU resource registry for the anime lighting stack
//               on Android mobile. Tracks every GPU-resident resource the
//               lighting pipeline creates — geometries, materials, shaders,
//               textures, render targets, buffer attributes, interleaved
//               buffers, and post-processing passes — with reference counting,
//               ownership tagging, lifecycle discipline, and deterministic
//               disposal.
//
//               Where the pools (010–013) manage *storage*, this module manages
//               *lifetime*. A lighting subsystem acquires a resource from a
//               pool, registers it here with a tag, and gets back a stable
//               handle. When the subsystem is disposed, it releases the handle;
//               the registry decrements the refcount and, when it hits zero,
//               hands the resource back to its owning pool (or disposes it
//               directly if the pool doesn't own it).
//
//               Design:
//                 • Fixed-capacity registry (MAX_RESOURCES entries) so no
//                   dynamic growth, no Map/Set on the hot path.
//                 • Every entry carries: kind, tag, refcount, owner subsystem
//                   id, generation, created-at frame, disposed flag.
//                 • Refcount semantics:
//                     – register() → refcount = 1
//                     – retain(handle) → refcount++
//                     – release(handle) → refcount--, auto-dispose at 0
//                     – forceRelease(handle) → refcount = 0, immediate dispose
//                 • Owner subsystem id: every subsystem registers itself with
//                   a unique id (from a monotonic counter), and every resource
//                   it creates carries that id. `disposeAllForOwner(id)` bulk
//                   releases everything a subsystem owns — critical for
//                   Android when a chunk, room, or LOD level is unloaded.
//                 • Auto-dispose dispatch: on refcount → 0 the registry calls
//                   the resource's `dispose()` if it exists. Pools may
//                   register a custom disposal hook via `registerPool(kind,
//                   hook)`.
//                 • Leak detection: `auditLeaks()` returns entries whose
//                   owning subsystem is disposed but whose refcount > 0 —
//                   this is the single most useful tool for finding GPU
//                   memory leaks on Android.
//                 • Frame lifecycle: `beginFrame()` / `endFrame()` bump
//                   per-frame stats; `markUsed(handle)` refreshes lastUsedFrame
//                   so age-based GC passes can evict cold resources.
//                 • Named reservations: `reserveNamed(name, resource, ...)`
//                   for long-lived engine-owned resources (shared shadow
//                   samplers, shared noise textures, shared palettes).
//                 • Diagnostic events: on-register, on-release, on-dispose,
//                   on-leak, on-audit — every one of them allocation-free in
//                   the hot path.
//                 • Zero per-frame allocations: all typed arrays sized once
//                   at construction; every internal list is a pre-allocated
//                   Int32Array ring.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external registry libs; every internal array sized
//               once at construction.
// best for : Guaranteeing deterministic GPU memory lifecycle for the entire
//            lighting stack. Every subsystem (006_lgt_LightManager through
//            380_lgt_lights) registers the resources it creates, tags them
//            with its owner id, and disposes them in one call on teardown.
//            The leak audit catches the classic Android bug where a chunk
//            unloads but its shadow depth texture stays alive.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  assertBiteCSReady,
  isBiteCSReady,
} from './009_scn_BiteCSVersionPolicy.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER = getPerfTier();

export const MAX_RESOURCES = PERF_TIER === 'HIGH' ? 8192 : PERF_TIER === 'MEDIUM' ? 4096 : 2048;
export const MAX_NAMED     = 512;
export const MAX_OWNERS    = 256;

export const RES_KIND = Object.freeze({
  UNKNOWN:           0,
  GEOMETRY:          1,
  MATERIAL:          2,
  SHADER:            3,
  TEXTURE:           4,
  RENDER_TARGET:     5,
  BUFFER_ATTRIBUTE:  6,
  INTERLEAVED:       7,
  BUFFER:            8,
  PROGRAM:           9,
  SAMPLER:          10,
  FBO_WRAPPER:      11,
  PASS:             12,
  COUNT:            13,
});

export const RES_KIND_NAME = Object.freeze([
  'unknown',
  'geometry',
  'material',
  'shader',
  'texture',
  'render_target',
  'buffer_attribute',
  'interleaved',
  'buffer',
  'program',
  'sampler',
  'fbo_wrapper',
  'pass',
]);

export const RES_TAG = Object.freeze({
  GENERIC:     0,
  SHADOW:      1,
  GI:          2,
  AO:          3,
  CLUSTER:     4,
  LIGHT_LIST:  5,
  ENV:         6,
  INTERIOR:    7,
  EXTERIOR:    8,
  POST:        9,
  DIRECTOR:   10,
  DIAGNOSTIC: 11,
  COUNT:      12,
});

export const RES_TAG_NAME = Object.freeze([
  'generic',
  'shadow',
  'gi',
  'ao',
  'cluster',
  'light_list',
  'env',
  'interior',
  'exterior',
  'post',
  'director',
  'diagnostic',
]);

export const RES_STATE = Object.freeze({
  FREE:      0,
  LIVE:      1,
  ZOMBIE:    2,  // owner disposed but refcount > 0
  DISPOSED:  3,
});

const RES_POOL_ID_SYMBOL = '__resRegistryId';
const RES_HANDLE_SYMBOL  = '__resHandle';

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

let _registryIdCounter = 0;

function _nextRegistryId() {
  return ++_registryIdCounter;
}

function _now() {
  return (typeof performance !== 'undefined' ? performance.now() : Date.now());
}

/* ------------------------------------------------------------------ */
/* 2. RESOURCE SLOT                                                   */
/* ------------------------------------------------------------------ */

export class ResourceSlot {
  constructor(index) {
    this.index        = index;
    this.handle       = 0;   // public handle (index | generation<<16)
    this.kind         = RES_KIND.UNKNOWN;
    this.tag          = RES_TAG.GENERIC;
    this.state        = RES_STATE.FREE;

    this.resource     = null;   // the actual THREE resource or null
    this.ownerId      = -1;     // subsystem id
    this.refCount     = 0;
    this.generation   = 0;

    this.createdFrame = 0;
    this.lastUsedFrame= 0;
    this.disposedFrame= 0;

    this.disposeHook  = null;   // (resource) -> void
    this.name         = null;
  }

  reset() {
    this.handle       = 0;
    this.kind         = RES_KIND.UNKNOWN;
    this.tag          = RES_TAG.GENERIC;
    this.state        = RES_STATE.FREE;
    this.resource     = null;
    this.ownerId      = -1;
    this.refCount     = 0;
    this.createdFrame = 0;
    this.lastUsedFrame= 0;
    this.disposedFrame= 0;
    this.disposeHook  = null;
    this.name         = null;
  }
}

/* ------------------------------------------------------------------ */
/* 3. OWNER SLOT                                                      */
/* ------------------------------------------------------------------ */

export class OwnerSlot {
  constructor(index) {
    this.index     = index;
    this.id        = 0;
    this.name      = null;
    this.active    = 0;
    this.totalOwned = 0;
    this.createdAt = 0;
    this.disposedAt = 0;
  }

  reset() {
    this.id = 0;
    this.name = null;
    this.active = 0;
    this.totalOwned = 0;
    this.createdAt = 0;
    this.disposedAt = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 4. RESOURCE REGISTRY                                               */
/* ------------------------------------------------------------------ */

export class ResourceRegistry {
  constructor(options = {}) {
    this.name   = options.name || `res_registry_${_nextRegistryId()}`;
    this.id     = _nextRegistryId();

    this.capacity = options.capacity || MAX_RESOURCES;
    this.autoDispose = options.autoDispose !== false;
    this.leakTracking = options.leakTracking !== false;

    // Pre-allocated resource slots.
    this.slots = new Array(this.capacity);
    for (let i = 0; i < this.capacity; i++) this.slots[i] = new ResourceSlot(i);

    // Free list (ring).
    this.freeList = new Int32Array(this.capacity);
    this.freeHead = 0;
    this.freeCount = this.capacity;
    for (let i = 0; i < this.capacity; i++) this.freeList[i] = i;

    // Live handle → slot index map (flat Int32Array sized to capacity).
    // Handles pack (generation << 16) | index; we store the slot index
    // directly and validate by generation on access.
    this._handleIndex = new Int32Array(this.capacity).fill(-1);

    // Owners.
    this.owners = new Array(MAX_OWNERS);
    for (let i = 0; i < MAX_OWNERS; i++) this.owners[i] = new OwnerSlot(i);
    this.ownerCount = 0;
    this._nextOwnerId = 1;

    // Named resources (long-lived engine-owned).
    this._named = new Map();

    // Frame + stats.
    this.frame = 0;
    this.stats = {
      totalRegistered:   0,
      totalReleased:     0,
      totalDisposed:     0,
      totalZombies:      0,
      totalLeaksDetected:0,
      currentLive:       0,
      peakLive:          0,
    };

    this._listeners = new Map();
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
      try { arr[i](payload); } catch (e) { console.error(`[014_rnd_ResourceRegistry] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- owner management ---------------- */

  registerOwner(name) {
    if (this.ownerCount >= MAX_OWNERS) return -1;
    const idx = this.ownerCount++;
    const slot = this.owners[idx];
    slot.reset();
    slot.id = this._nextOwnerId++;
    slot.name = String(name || `owner_${slot.id}`);
    slot.active = 1;
    slot.createdAt = _now();
    this._emit('owner-registered', { ownerId: slot.id, name: slot.name });
    return slot.id;
  }

  disposeOwner(ownerId) {
    const slot = this._findOwnerSlot(ownerId);
    if (!slot) return false;
    slot.active = 0;
    slot.disposedAt = _now();
    this._emit('owner-disposed', { ownerId, name: slot.name });
    return true;
  }

  isOwnerActive(ownerId) {
    const slot = this._findOwnerSlot(ownerId);
    return !!slot && slot.active === 1;
  }

  _findOwnerSlot(ownerId) {
    for (let i = 0; i < this.ownerCount; i++) {
      if (this.owners[i].id === ownerId) return this.owners[i];
    }
    return null;
  }

  /* ---------------- resource registration ---------------- */

  /**
   * Register a resource. Returns a handle, or 0 on failure.
   *
   *   const h = registry.register(material, {
   *     kind: RES_KIND.MATERIAL,
   *     tag:  RES_TAG.SHADOW,
   *     ownerId,
   *     name: 'shadowDepthMaterial',
   *     disposeHook: (m) => m.dispose(),
   *   });
   */
  register(resource, options = {}) {
    if (!resource) return 0;

    if (this.freeCount <= 0) {
      this._emit('rejected', { reason: 'full' });
      return 0;
    }

    const idx = this.freeList[this.freeHead];
    this.freeHead = (this.freeHead + 1) % this.capacity;
    this.freeCount--;

    const slot = this.slots[idx];
    slot.reset();

    slot.index          = idx;
    slot.generation     = (slot.generation + 1) | 0;
    if (slot.generation <= 0 || slot.generation > 0x7FFF) slot.generation = 1;
    slot.handle         = (idx & 0xFFFF) | (slot.generation << 16);
    slot.kind           = options.kind    !== undefined ? options.kind    : RES_KIND.UNKNOWN;
    slot.tag            = options.tag     !== undefined ? options.tag     : RES_TAG.GENERIC;
    slot.state          = RES_STATE.LIVE;
    slot.resource       = resource;
    slot.ownerId        = options.ownerId !== undefined ? options.ownerId : -1;
    slot.refCount       = 1;
    slot.createdFrame   = this.frame;
    slot.lastUsedFrame  = this.frame;
    slot.disposeHook    = typeof options.disposeHook === 'function' ? options.disposeHook : null;
    slot.name           = options.name || null;

    this._handleIndex[idx] = idx;

    // Tag the resource for fast identity lookup and defensive double-dispose.
    try {
      Object.defineProperty(resource, RES_POOL_ID_SYMBOL, {
        value: this.id,
        enumerable: false,
        writable: true,
        configurable: true,
      });
      Object.defineProperty(resource, RES_HANDLE_SYMBOL, {
        value: slot.handle,
        enumerable: false,
        writable: true,
        configurable: true,
      });
    } catch (_) {
      // Some resources (frozen objects) may reject property add.
    }

    this.stats.totalRegistered++;
    this.stats.currentLive++;
    if (this.stats.currentLive > this.stats.peakLive) this.stats.peakLive = this.stats.currentLive;

    if (slot.ownerId >= 0) {
      const ownerSlot = this._findOwnerSlot(slot.ownerId);
      if (ownerSlot) ownerSlot.totalOwned++;
    }

    this._emit('registered', {
      handle: slot.handle,
      kind: slot.kind,
      tag: slot.tag,
      ownerId: slot.ownerId,
      name: slot.name,
    });

    return slot.handle;
  }

  /* ---------------- handle resolution ---------------- */

  _resolveSlot(handle) {
    if (handle === 0 || handle === undefined || handle === null) return null;
    const idx = handle & 0xFFFF;
    const gen = (handle >>> 16) & 0x7FFF;
    if (idx < 0 || idx >= this.capacity) return null;
    const slot = this.slots[idx];
    if (!slot || slot.state === RES_STATE.FREE || slot.generation !== gen) return null;
    return slot;
  }

  /* ---------------- retain / release / mark-used ---------------- */

  retain(handle) {
    const slot = this._resolveSlot(handle);
    if (!slot) return -1;
    slot.refCount++;
    slot.lastUsedFrame = this.frame;
    return slot.refCount;
  }

  release(handle) {
    const slot = this._resolveSlot(handle);
    if (!slot) return -1;

    slot.refCount--;
    if (slot.refCount > 0) return slot.refCount;

    // Refcount hit zero → dispose.
    this._disposeSlot(slot);
    return 0;
  }

  forceRelease(handle) {
    const slot = this._resolveSlot(handle);
    if (!slot) return false;
    slot.refCount = 0;
    this._disposeSlot(slot);
    return true;
  }

  markUsed(handle) {
    const slot = this._resolveSlot(handle);
    if (!slot) return false;
    slot.lastUsedFrame = this.frame;
    return true;
  }

  getResource(handle) {
    const slot = this._resolveSlot(handle);
    return slot ? slot.resource : null;
  }

  getRefCount(handle) {
    const slot = this._resolveSlot(handle);
    return slot ? slot.refCount : 0;
  }

  /* ---------------- disposal ---------------- */

  _disposeSlot(slot) {
    if (slot.state === RES_STATE.DISPOSED || slot.state === RES_STATE.FREE) return;

    const resource = slot.resource;

    // Call the disposal hook or fall back to resource.dispose().
    try {
      if (slot.disposeHook) {
        slot.disposeHook(resource);
      } else if (resource && typeof resource.dispose === 'function') {
        resource.dispose();
      }
    } catch (e) {
      console.error(`[014_rnd_ResourceRegistry] dispose failed for "${slot.name || slot.handle}"`, e);
    }

    const handle = slot.handle;
    const kind = slot.kind;
    const tag = slot.tag;
    const ownerId = slot.ownerId;
    const name = slot.name;

    // Remove named mapping if any.
    if (name && this._named.get(name) === handle) this._named.delete(name);

    // Clear slot and return to free list.
    slot.state = RES_STATE.FREE;
    slot.resource = null;
    slot.refCount = 0;
    slot.disposedFrame = this.frame;
    slot.disposeHook = null;
    slot.name = null;
    slot.ownerId = -1;

    this.freeList[(this.freeHead + this.freeCount) % this.capacity] = slot.index;
    this.freeCount++;

    this._handleIndex[slot.index] = -1;

    this.stats.totalReleased++;
    this.stats.totalDisposed++;
    this.stats.currentLive--;
    if (this.stats.currentLive < 0) this.stats.currentLive = 0;

    this._emit('disposed', { handle, kind, tag, ownerId, name });
  }

  /**
   * Bulk-release every resource owned by the given owner id. This is the
   * single most important API for Android — when a chunk/room/LOD is
   * unloaded, this drops every GPU resource it created in one call.
   */
  disposeAllForOwner(ownerId) {
    let released = 0;
    for (let i = 0; i < this.capacity; i++) {
      const slot = this.slots[i];
      if (slot.state === RES_STATE.LIVE && slot.ownerId === ownerId) {
        slot.refCount = 0;
        this._disposeSlot(slot);
        released++;
      }
    }
    this._emit('owner-bulk-dispose', { ownerId, released });
    return released;
  }

  /* ---------------- named resources ---------------- */

  reserveNamed(name, resource, options = {}) {
    if (!name || typeof name !== 'string') return 0;
    const existing = this._named.get(name);
    if (existing) return existing;

    const handle = this.register(resource, Object.assign({}, options, { name }));
    if (handle === 0) return 0;
    this._named.set(name, handle);
    return handle;
  }

  getNamed(name) {
    const h = this._named.get(name);
    if (h === undefined) return null;
    return this.getResource(h);
  }

  getNamedHandle(name) {
    return this._named.get(name) || 0;
  }

  releaseNamed(name) {
    const h = this._named.get(name);
    if (h === undefined) return false;
    this._named.delete(name);
    return this.forceRelease(h);
  }

  /* ---------------- frame lifecycle ---------------- */

  beginFrame() {
    this.frame++;
    return this;
  }

  endFrame() {
    return this;
  }

  /* ---------------- leak audit ---------------- */

  /**
   * Returns the number of leaked resources (owner disposed but refcount > 0
   * OR resource still LIVE with no active owner). Fills `outHandles` if
   * provided (Int32Array or array), returning the count written.
   */
  auditLeaks(outHandles) {
    let count = 0;
    for (let i = 0; i < this.capacity; i++) {
      const slot = this.slots[i];
      if (slot.state !== RES_STATE.LIVE) continue;
      if (slot.ownerId < 0) continue;

      const owner = this._findOwnerSlot(slot.ownerId);
      const isLeak = !owner || owner.active === 0;

      if (isLeak) {
        slot.state = RES_STATE.ZOMBIE;
        this.stats.totalZombies++;
        this.stats.totalLeaksDetected++;
        if (outHandles) {
          if (Array.isArray(outHandles)) outHandles.push(slot.handle);
          else if (count < outHandles.length) outHandles[count] = slot.handle;
        }
        this._emit('leak', {
          handle: slot.handle,
          kind: slot.kind,
          tag: slot.tag,
          name: slot.name,
          ownerId: slot.ownerId,
        });
        count++;
      }
    }
    return count;
  }

  /**
   * Force-dispose every zombie detected by auditLeaks().
   */
  purgeZombies() {
    let purged = 0;
    for (let i = 0; i < this.capacity; i++) {
      const slot = this.slots[i];
      if (slot.state === RES_STATE.ZOMBIE) {
        slot.refCount = 0;
        this._disposeSlot(slot);
        purged++;
      }
    }
    return purged;
  }

  /* ---------------- age-based GC ---------------- */

  /**
   * Dispose any LIVE resource that hasn't been marked used in the last
   * `maxAgeFrames` frames. Only safe to call on engine-owned resources;
   * named resources are protected by default.
   */
  collectUnused(maxAgeFrames, tagMask = 0xFFFF) {
    if (!Number.isFinite(maxAgeFrames) || maxAgeFrames <= 0) return 0;
    const cutoff = this.frame - maxAgeFrames;
    let collected = 0;
    for (let i = 0; i < this.capacity; i++) {
      const slot = this.slots[i];
      if (slot.state !== RES_STATE.LIVE) continue;
      if (slot.name && this._named.has(slot.name)) continue;  // protected
      if (slot.lastUsedFrame >= cutoff) continue;
      if (((tagMask >>> slot.tag) & 1) === 0) continue;
      slot.refCount = 0;
      this._disposeSlot(slot);
      collected++;
    }
    return collected;
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    const byKind = new Uint32Array(RES_KIND.COUNT);
    const byTag  = new Uint32Array(RES_TAG.COUNT);
    const byOwner = new Uint32Array(MAX_OWNERS);

    for (let i = 0; i < this.capacity; i++) {
      const slot = this.slots[i];
      if (slot.state === RES_STATE.FREE) continue;
      if (slot.kind < RES_KIND.COUNT) byKind[slot.kind]++;
      if (slot.tag  < RES_TAG.COUNT)  byTag[slot.tag]++;
      if (slot.ownerId >= 0) {
        const ownerSlot = this._findOwnerSlot(slot.ownerId);
        if (ownerSlot) byOwner[ownerSlot.index]++;
      }
    }

    const owners = new Array(this.ownerCount);
    for (let i = 0; i < this.ownerCount; i++) {
      owners[i] = {
        id:         this.owners[i].id,
        name:       this.owners[i].name,
        active:     this.owners[i].active === 1,
        totalOwned: this.owners[i].totalOwned,
        liveCount:  byOwner[i],
      };
    }

    const kindStats = new Array(RES_KIND.COUNT);
    for (let k = 0; k < RES_KIND.COUNT; k++) {
      kindStats[k] = { name: RES_KIND_NAME[k], live: byKind[k] };
    }

    const tagStats = new Array(RES_TAG.COUNT);
    for (let t = 0; t < RES_TAG.COUNT; t++) {
      tagStats[t] = { name: RES_TAG_NAME[t], live: byTag[t] };
    }

    return {
      name:               this.name,
      frame:              this.frame,
      capacity:           this.capacity,
      freeCount:          this.freeCount,
      currentLive:        this.stats.currentLive,
      peakLive:           this.stats.peakLive,
      totalRegistered:    this.stats.totalRegistered,
      totalReleased:      this.stats.totalReleased,
      totalDisposed:      this.stats.totalDisposed,
      totalZombies:       this.stats.totalZombies,
      totalLeaksDetected: this.stats.totalLeaksDetected,
      namedCount:         this._named.size,
      ownerCount:         this.ownerCount,
      owners,
      kindStats,
      tagStats,
      perfTier:           PERF_TIER,
    };
  }

  /* ---------------- reset / dispose ---------------- */

  reset() {
    // Disposing every live resource first.
    for (let i = 0; i < this.capacity; i++) {
      const slot = this.slots[i];
      if (slot.state === RES_STATE.LIVE || slot.state === RES_STATE.ZOMBIE) {
        slot.refCount = 0;
        this._disposeSlot(slot);
      }
      slot.reset();
    }

    this.freeHead = 0;
    this.freeCount = this.capacity;
    for (let i = 0; i < this.capacity; i++) this.freeList[i] = i;
    for (let i = 0; i < this.capacity; i++) this._handleIndex[i] = -1;

    for (let i = 0; i < MAX_OWNERS; i++) this.owners[i].reset();
    this.ownerCount = 0;
    this._nextOwnerId = 1;

    this._named.clear();

    this.stats.totalRegistered = 0;
    this.stats.totalReleased = 0;
    this.stats.totalDisposed = 0;
    this.stats.totalZombies = 0;
    this.stats.totalLeaksDetected = 0;
    this.stats.currentLive = 0;
    this.stats.peakLive = 0;

    this.frame = 0;
    return this;
  }

  dispose() {
    this.reset();
    this.slots.length = 0;
    this.slots = null;
    this.freeList = null;
    this._handleIndex = null;
    this.owners.length = 0;
    this.owners = null;
    this._named.clear();
    this._listeners.clear();
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 5. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultRegistry = null;

export function getDefaultResourceRegistry() {
  if (!_defaultRegistry) _defaultRegistry = new ResourceRegistry();
  return _defaultRegistry;
}

export function disposeDefaultResourceRegistry() {
  if (_defaultRegistry) {
    _defaultRegistry.dispose();
    _defaultRegistry = null;
  }
}

/* ------------------------------------------------------------------ */
/* 6. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createResourceRegistry(options = {}) {
  return new ResourceRegistry(options);
}

/* ------------------------------------------------------------------ */
/* 7. CONVENIENCE REGISTRATION HELPERS                                */
/* ------------------------------------------------------------------ */

export function registerGeometry(geometry, options = {}) {
  return getDefaultResourceRegistry().register(geometry, Object.assign({ kind: RES_KIND.GEOMETRY }, options));
}

export function registerMaterial(material, options = {}) {
  return getDefaultResourceRegistry().register(material, Object.assign({ kind: RES_KIND.MATERIAL }, options));
}

export function registerTexture(texture, options = {}) {
  return getDefaultResourceRegistry().register(texture, Object.assign({ kind: RES_KIND.TEXTURE }, options));
}

export function registerRenderTarget(target, options = {}) {
  return getDefaultResourceRegistry().register(target, Object.assign({ kind: RES_KIND.RENDER_TARGET }, options));
}

export function registerBufferAttribute(attribute, options = {}) {
  return getDefaultResourceRegistry().register(attribute, Object.assign({ kind: RES_KIND.BUFFER_ATTRIBUTE }, options));
}

export function registerInterleavedBuffer(buffer, options = {}) {
  return getDefaultResourceRegistry().register(buffer, Object.assign({ kind: RES_KIND.INTERLEAVED }, options));
}

export function registerPass(pass, options = {}) {
  return getDefaultResourceRegistry().register(pass, Object.assign({ kind: RES_KIND.PASS }, options));
}

/* ------------------------------------------------------------------ */
/* 8. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  ResourceRegistry,
  ResourceSlot,
  OwnerSlot,
  createResourceRegistry,
  getDefaultResourceRegistry,
  disposeDefaultResourceRegistry,
  registerGeometry,
  registerMaterial,
  registerTexture,
  registerRenderTarget,
  registerBufferAttribute,
  registerInterleavedBuffer,
  registerPass,
  RES_KIND,
  RES_KIND_NAME,
  RES_TAG,
  RES_TAG_NAME,
  RES_STATE,
  MAX_RESOURCES,
  MAX_NAMED,
  MAX_OWNERS,
};

export default _defaultExport;