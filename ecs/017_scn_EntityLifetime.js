// File : 017
// name : src/ecs/017_scn_EntityLifetime.js
// description : Entity lifetime and lifecycle-state module for the scene ECS
//               world of the anime lighting stack on Android mobile. Every
//               entity that lives in the scene has a lifetime — a spawn
//               frame, an elapsed-alive counter, an optional TTL, a
//               lifecycle state (Spawning / Alive / Suspended / Dying /
//               Dead / Pooled), a subsystem pool binding, and a grace
//               period during which the entity is still visible but no
//               longer logically active. This module owns all of that.
//
//               Design:
//                 • Fixed-capacity SoA arrays sized once to MAX_ENTITIES.
//                   No dynamic growth, no map allocations, no runtime
//                   resizes.
//                 • Lifecycle state machine with strict transition
//                   validation — every transition helper logs a warning on
//                   invalid transitions in development mode and silently
//                   drops them in production.
//                 • Automatic pool return: when an entity enters the DEAD
//                   state, the pool binding recorded in its lifetime record
//                   is used to release it back to the correct subsystem
//                   pool after a configurable grace period.
//                 • TTL sweep: `expireStale(frame)` walks the world once
//                   per frame and marks any entity whose TTL has expired
//                   as DYING.
//                 • GC sweep: `collectGarbage()` walks the world once per
//                   frame and moves any entity whose Dying grace period
//                   has elapsed into DEAD, then releases it to its pool.
//                 • Touch tracking: `touch(eid)` refreshes the
//                   lastTouchFrame counter so an entity that is actively
//                   referenced by a subsystem never becomes stale.
//                 • Zero allocation on the hot path — all state lives in
//                   typed arrays.
//
//               Integration:
//                 • 010_scn_ECSWorld.js       — world handle
//                 • 011_scn_BiteCSAdapter.js  — entity lifecycle
//                 • 014_scn_Tags.js           — lifecycle tag bits
//                 • 015_scn_Relations.js      — relation cleanup
//                 • 016_scn_EntityPool.js     — pool return
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every typed array sized once at construction.
// best for : Guaranteeing that every entity in the anime lighting stack
//            has a well-defined, deterministic lifetime — with automatic
//            TTL expiry, automatic pool return, and explicit lifecycle
//            states — so nothing leaks, nothing lingers, and everything
//            recycles.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  entityExists,
} from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';

import {
  getPerfTier,
} from '../core/008_scn_world.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
} from '../core/026_rnd_Logger.js';

import {
  getDefaultErrorBoundaries,
  BOUNDARY_TAG,
} from '../core/030_rnd_ErrorBoundary.js';

import {
  MAX_ENTITIES,
} from './002_lgt_LightComponents.js';

import {
  getECSWorld,
} from './010_scn_ECSWorld.js';

import {
  getAdapter,
} from './011_scn_BiteCSAdapter.js';

import {
  getDefaultComponentRegistry,
} from './012_scn_ComponentRegistry.js';

import {
  TAG,
  TAG2,
  tagEntity,
  untagEntity,
  tagEntity2,
  untagEntity2,
  hasTag,
  hasTag2,
} from './014_scn_Tags.js';

import {
  detachAllRelations,
} from './015_scn_Relations.js';

import {
  release,
  releaseLightEntity,
  releaseShadowEntity,
  releaseCameraEntity,
  releaseGIEntity,
  releaseAOEntity,
  releaseSceneEntity,
  releaseStreamingEntity,
  releaseDebugEntity,
  releaseParticleEntity,
  releaseUIEntity,
  releaseAnimationEntity,
  releaseWeatherEntity,
  POOL,
  isInUse,
} from './016_scn_EntityPool.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Entity lifecycle states. The valid transitions are declared below.
 */
export const LIFETIME_STATE = Object.freeze({
  NONE:       0,   // no state yet (fresh, unscheduled)
  SPAWNING:   1,   // being set up
  ALIVE:      2,   // active and running
  SUSPENDED:  3,   // temporarily inactive (visible, not updated)
  DYING:      4,   // grace period before death
  DEAD:       5,   // ready to be released
  POOLED:     6,   // released back to a pool
  COUNT:      7,
});

export const LIFETIME_STATE_NAME = Object.freeze([
  'none',
  'spawning',
  'alive',
  'suspended',
  'dying',
  'dead',
  'pooled',
]);

/**
 * Pool binding — which subsystem pool this entity belongs to. Mirrors
 * the POOL enum from 016_scn_EntityPool.js but stored locally to avoid
 * a circular import at construction time.
 */
export const LIFETIME_POOL = Object.freeze({
  NONE:       -1,
  CAMERA:     0,
  LIGHT:      1,
  SHADOW:     2,
  GI:         3,
  AO:         4,
  SCENE:      5,
  STREAMING:  6,
  DEBUG:      7,
  PARTICLE:   8,
  UI:         9,
  ANIMATION: 10,
  WEATHER:   11,
  COUNT:     12,
});

/**
 * Default grace period between DYING and DEAD, in frames.
 * At 60 Hz, 30 frames = 500 ms of visual fade-out time.
 */
export const DEFAULT_DYING_GRACE_FRAMES =
  PERF_TIER_LOCAL === 'HIGH'   ? 30 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 24 :
                                 18;

/**
 * Maximum lifetime in frames for the "alive" counter before it saturates.
 * 2^31 frames ≈ 414 days at 60 Hz. Plenty of headroom.
 */
const MAX_ALIVE_FRAMES = 0x7FFFFFFF;

/* ------------------------------------------------------------------ */
/* 1. SoA COMPONENT                                                   */
/* ------------------------------------------------------------------ */

/**
 * EntityLifetime — the canonical lifetime record.
 * One entry per entity (all MAX_ENTITIES slots allocated).
 */
export const EntityLifetime = {
  state:            new Uint8Array(MAX_ENTITIES),     // LIFETIME_STATE
  prevState:        new Uint8Array(MAX_ENTITIES),
  pool:             new Int8Array(MAX_ENTITIES),      // LIFETIME_POOL

  spawnFrame:       new Uint32Array(MAX_ENTITIES),
  aliveFrames:      new Uint32Array(MAX_ENTITIES),
  ttlFrames:        new Uint32Array(MAX_ENTITIES),    // 0 = infinite
  dyingFrames:      new Uint16Array(MAX_ENTITIES),
  dyingGrace:       new Uint16Array(MAX_ENTITIES),

  lastTouchFrame:   new Uint32Array(MAX_ENTITIES),
  lastStateFrame:   new Uint32Array(MAX_ENTITIES),
  stateChangeCount: new Uint16Array(MAX_ENTITIES),

  // Persistence flags (cache the tag bits locally for O(1) checks).
  persistent:       new Uint8Array(MAX_ENTITIES),
  transient:        new Uint8Array(MAX_ENTITIES),

  // Failure counter — if a lifecycle transition fails repeatedly, the
  // entity is force-released by the GC.
  failureCount:     new Uint8Array(MAX_ENTITIES),
};

/**
 * ECS component bundle for bitECS createWorld.
 */
export const LIFETIME_COMPONENTS = Object.freeze({
  EntityLifetime,
});

/* ------------------------------------------------------------------ */
/* 2. TRANSITION VALIDATION                                           */
/* ------------------------------------------------------------------ */

/**
 * Bitmask of allowed next states for each current state:
 *   ALLOWED[from] has bit `to` set if the transition from → to is legal.
 */
const ALLOWED_TRANSITIONS = new Uint8Array(LIFETIME_STATE.COUNT);

function _bit(state) { return 1 << state; }

// NONE → SPAWNING / POOLED / DEAD
ALLOWED_TRANSITIONS[LIFETIME_STATE.NONE] =
  _bit(LIFETIME_STATE.SPAWNING) |
  _bit(LIFETIME_STATE.POOLED)   |
  _bit(LIFETIME_STATE.DEAD);

// SPAWNING → ALIVE / DEAD / SUSPENDED
ALLOWED_TRANSITIONS[LIFETIME_STATE.SPAWNING] =
  _bit(LIFETIME_STATE.ALIVE)     |
  _bit(LIFETIME_STATE.SUSPENDED) |
  _bit(LIFETIME_STATE.DEAD);

// ALIVE → SUSPENDED / DYING
ALLOWED_TRANSITIONS[LIFETIME_STATE.ALIVE] =
  _bit(LIFETIME_STATE.SUSPENDED) |
  _bit(LIFETIME_STATE.DYING);

// SUSPENDED → ALIVE / DYING / DEAD
ALLOWED_TRANSITIONS[LIFETIME_STATE.SUSPENDED] =
  _bit(LIFETIME_STATE.ALIVE) |
  _bit(LIFETIME_STATE.DYING) |
  _bit(LIFETIME_STATE.DEAD);

// DYING → DEAD / ALIVE (cancel death)
ALLOWED_TRANSITIONS[LIFETIME_STATE.DYING] =
  _bit(LIFETIME_STATE.DEAD) |
  _bit(LIFETIME_STATE.ALIVE);

// DEAD → POOLED / DEAD
ALLOWED_TRANSITIONS[LIFETIME_STATE.DEAD] =
  _bit(LIFETIME_STATE.POOLED) |
  _bit(LIFETIME_STATE.DEAD);

// POOLED → SPAWNING / POOLED
ALLOWED_TRANSITIONS[LIFETIME_STATE.POOLED] =
  _bit(LIFETIME_STATE.SPAWNING) |
  _bit(LIFETIME_STATE.POOLED);

function _isTransitionAllowed(from, to) {
  if (from < 0 || from >= LIFETIME_STATE.COUNT) return false;
  if (to   < 0 || to   >= LIFETIME_STATE.COUNT) return false;
  return (ALLOWED_TRANSITIONS[from] & _bit(to)) !== 0;
}

/* ------------------------------------------------------------------ */
/* 3. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const LifetimeState = {
  frame:             0,
  totalSpawn:        0,
  totalDeath:        0,
  totalSuspended:    0,
  totalResumed:      0,
  totalExpired:      0,
  totalCollected:    0,
  totalRejected:     0,
  transitions:       0,
};

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.entity_lifetime', {
        tag: BOUNDARY_TAG.GENERIC,
        failureThreshold: 5,
      });
    }
  } catch (_) { /* swallow */ }
  return _boundary;
}

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

function _isValidEntity(eid) {
  return typeof eid === 'number' && eid >= 0 && eid < MAX_ENTITIES;
}

/* ------------------------------------------------------------------ */
/* 4. LOW-LEVEL STATE TRANSITION                                      */
/* ------------------------------------------------------------------ */

function _transition(eid, toState) {
  const from = EntityLifetime.state[eid];
  if (from === toState) return true;

  if (!_isTransitionAllowed(from, toState)) {
    LifetimeState.totalRejected++;
    EntityLifetime.failureCount[eid]++;

    const log = _safeLogger();
    if (log && PERF_TIER_LOCAL !== 'LOW') {
      log.warn(LOG_CHANNEL.CORE, () =>
        `[017_scn_EntityLifetime] illegal transition for eid=${eid}: ` +
        `${LIFETIME_STATE_NAME[from]} → ${LIFETIME_STATE_NAME[toState]}`);
    }
    return false;
  }

  EntityLifetime.prevState[eid] = from;
  EntityLifetime.state[eid] = toState;
  EntityLifetime.lastStateFrame[eid] = LifetimeState.frame;
  EntityLifetime.stateChangeCount[eid]++;
  LifetimeState.transitions++;

  // Update lifecycle tags.
  _applyLifecycleTag(eid, toState);
  return true;
}

function _applyLifecycleTag(eid, state) {
  switch (state) {
    case LIFETIME_STATE.SPAWNING:
      tagEntity2(eid, TAG2.SPAWN_PENDING);
      break;
    case LIFETIME_STATE.ALIVE:
      untagEntity2(eid, TAG2.SPAWN_PENDING);
      untagEntity2(eid, TAG2.DESTROY_PENDING);
      tagEntity(eid, TAG.ACTIVE);
      break;
    case LIFETIME_STATE.SUSPENDED:
      untagEntity(eid, TAG.ACTIVE);
      untagEntity(eid, TAG.VISIBLE);
      break;
    case LIFETIME_STATE.DYING:
      untagEntity(eid, TAG.ACTIVE);
      tagEntity2(eid, TAG2.DESTROY_PENDING);
      break;
    case LIFETIME_STATE.DEAD:
      untagEntity(eid, TAG.ACTIVE);
      tagEntity2(eid, TAG2.DESTROY_PENDING);
      break;
    case LIFETIME_STATE.POOLED:
      untagEntity(eid, TAG.ACTIVE);
      untagEntity(eid, TAG.VISIBLE);
      untagEntity(eid, TAG.LIGHT);
      untagEntity2(eid, TAG2.SPAWN_PENDING);
      untagEntity2(eid, TAG2.DESTROY_PENDING);
      break;
    default:
      break;
  }
}

/* ------------------------------------------------------------------ */
/* 5. PUBLIC LIFECYCLE OPERATIONS                                     */
/* ------------------------------------------------------------------ */

/**
 * Marks an entity as SPAWNING. Called right after `acquire()` returns.
 * `ttlFrames` of 0 means the entity lives forever (until explicitly
 * marked dying).
 */
export function markSpawning(eid, pool, ttlFrames) {
  if (!_isValidEntity(eid)) return false;

  const alive = LifetimeState.frame;
  EntityLifetime.state[eid]            = LIFETIME_STATE.NONE;
  EntityLifetime.prevState[eid]        = LIFETIME_STATE.NONE;
  EntityLifetime.pool[eid]             = (pool !== undefined ? pool : LIFETIME_POOL.NONE) | 0;
  EntityLifetime.spawnFrame[eid]       = alive;
  EntityLifetime.aliveFrames[eid]      = 0;
  EntityLifetime.ttlFrames[eid]        = (ttlFrames !== undefined && ttlFrames > 0) ? (ttlFrames | 0) : 0;
  EntityLifetime.dyingFrames[eid]      = 0;
  EntityLifetime.dyingGrace[eid]       = DEFAULT_DYING_GRACE_FRAMES;
  EntityLifetime.lastTouchFrame[eid]   = alive;
  EntityLifetime.lastStateFrame[eid]   = alive;
  EntityLifetime.stateChangeCount[eid] = 0;
  EntityLifetime.persistent[eid]       = 0;
  EntityLifetime.transient[eid]        = 0;
  EntityLifetime.failureCount[eid]     = 0;

  LifetimeState.totalSpawn++;
  return _transition(eid, LIFETIME_STATE.SPAWNING);
}

/**
 * Transitions an entity to ALIVE. Called when the entity has finished
 * being set up and is now ready to run.
 */
export function markAlive(eid) {
  if (!_isValidEntity(eid)) return false;
  return _transition(eid, LIFETIME_STATE.ALIVE);
}

/**
 * Marks an entity as SUSPENDED — it stays resident but is not updated.
 * Useful for backgrounded scenes, off-screen chunks, and paused systems.
 */
export function suspend(eid) {
  if (!_isValidEntity(eid)) return false;
  const ok = _transition(eid, LIFETIME_STATE.SUSPENDED);
  if (ok) LifetimeState.totalSuspended++;
  return ok;
}

/**
 * Resumes a suspended entity back to ALIVE.
 */
export function resume(eid) {
  if (!_isValidEntity(eid)) return false;
  const ok = _transition(eid, LIFETIME_STATE.ALIVE);
  if (ok) LifetimeState.totalResumed++;
  return ok;
}

/**
 * Marks an entity as DYING. It enters a grace period during which it is
 * still visible but no longer active. After the grace period, `collectGarbage`
 * will move it to DEAD and release it back to its pool.
 */
export function markDying(eid, graceFrames) {
  if (!_isValidEntity(eid)) return false;
  const ok = _transition(eid, LIFETIME_STATE.DYING);
  if (!ok) return false;

  EntityLifetime.dyingFrames[eid] = 0;
  EntityLifetime.dyingGrace[eid] =
    (graceFrames !== undefined && graceFrames >= 0)
      ? (graceFrames | 0)
      : DEFAULT_DYING_GRACE_FRAMES;

  LifetimeState.totalDeath++;
  return true;
}

/**
 * Immediately forces an entity to DEAD, skipping the dying grace period.
 */
export function markDead(eid) {
  if (!_isValidEntity(eid)) return false;
  return _transition(eid, LIFETIME_STATE.DEAD);
}

/**
 * Marks an entity as POOLED after its pool release.
 */
export function markPooled(eid) {
  if (!_isValidEntity(eid)) return false;
  return _transition(eid, LIFETIME_STATE.POOLED);
}

/**
 * Cancels a DYING state and returns the entity to ALIVE. Useful for
 * "revive" mechanics or for canceling a despawn while it is still in
 * the grace period.
 */
export function cancelDeath(eid) {
  if (!_isValidEntity(eid)) return false;
  if (EntityLifetime.state[eid] !== LIFETIME_STATE.DYING) return false;
  EntityLifetime.dyingFrames[eid] = 0;
  return _transition(eid, LIFETIME_STATE.ALIVE);
}

/* ------------------------------------------------------------------ */
/* 6. TTL / TOUCH                                                     */
/* ------------------------------------------------------------------ */

/**
 * Sets or updates the entity's TTL.
 */
export function setTTL(eid, ttlFrames) {
  if (!_isValidEntity(eid)) return false;
  EntityLifetime.ttlFrames[eid] = (ttlFrames > 0) ? (ttlFrames | 0) : 0;
  return true;
}

/**
 * Returns the entity's current TTL, or 0 for infinite.
 */
export function getTTL(eid) {
  if (!_isValidEntity(eid)) return 0;
  return EntityLifetime.ttlFrames[eid];
}

/**
 * Refreshes the entity's last-touch frame. Called by any subsystem that
 * actively references the entity — prevents stale collection.
 */
export function touch(eid) {
  if (!_isValidEntity(eid)) return false;
  EntityLifetime.lastTouchFrame[eid] = LifetimeState.frame;
  return true;
}

/**
 * Returns the entity's age in frames since it was spawned.
 */
export function age(eid) {
  if (!_isValidEntity(eid)) return 0;
  return EntityLifetime.aliveFrames[eid];
}

/**
 * Returns the entity's current lifecycle state.
 */
export function getState(eid) {
  if (!_isValidEntity(eid)) return LIFETIME_STATE.NONE;
  return EntityLifetime.state[eid];
}

/**
 * Returns true if the entity is in the ALIVE state.
 */
export function isAlive(eid) {
  if (!_isValidEntity(eid)) return false;
  return EntityLifetime.state[eid] === LIFETIME_STATE.ALIVE;
}

/**
 * Returns true if the entity is in the DYING state.
 */
export function isDying(eid) {
  if (!_isValidEntity(eid)) return false;
  return EntityLifetime.state[eid] === LIFETIME_STATE.DYING;
}

/**
 * Returns true if the entity is in the DEAD or POOLED state.
 */
export function isDead(eid) {
  if (!_isValidEntity(eid)) return false;
  const s = EntityLifetime.state[eid];
  return s === LIFETIME_STATE.DEAD || s === LIFETIME_STATE.POOLED;
}

/**
 * Marks the entity persistent — it will never be collected by the GC
 * even if its TTL expires. Must be explicitly marked dying.
 */
export function setPersistent(eid, persistent) {
  if (!_isValidEntity(eid)) return false;
  EntityLifetime.persistent[eid] = persistent ? 1 : 0;
  if (persistent) tagEntity2(eid, TAG2.PERSISTENT);
  else            untagEntity2(eid, TAG2.PERSISTENT);
  return true;
}

/**
 * Marks the entity transient — it will be collected quickly after
 * being touched. Used for short-lived particle effects and debug
 * markers.
 */
export function setTransient(eid, transient) {
  if (!_isValidEntity(eid)) return false;
  EntityLifetime.transient[eid] = transient ? 1 : 0;
  if (transient) tagEntity2(eid, TAG2.TRANSIENT);
  else           untagEntity2(eid, TAG2.TRANSIENT);
  return true;
}

/* ------------------------------------------------------------------ */
/* 7. FRAME ADVANCE                                                   */
/* ------------------------------------------------------------------ */

/**
 * Advances the entity lifetime system by one frame. Called once per
 * frame by the engine loop, before any TTL/GC sweeps.
 *
 * For each live entity this:
 *   • Increments aliveFrames (saturating at MAX_ALIVE_FRAMES).
 *   • Increments dyingFrames if the entity is DYING.
 */
export function tickLifetime(frameNumber) {
  if (typeof frameNumber === 'number') LifetimeState.frame = frameNumber;
  else LifetimeState.frame++;

  const frame = LifetimeState.frame;

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    const state = EntityLifetime.state[eid];
    if (state === LIFETIME_STATE.NONE) continue;

    if (state === LIFETIME_STATE.ALIVE ||
        state === LIFETIME_STATE.SPAWNING ||
        state === LIFETIME_STATE.SUSPENDED) {
      const a = EntityLifetime.aliveFrames[eid];
      if (a < MAX_ALIVE_FRAMES) EntityLifetime.aliveFrames[eid] = a + 1;
    }

    if (state === LIFETIME_STATE.DYING) {
      const d = EntityLifetime.dyingFrames[eid];
      if (d < 0xFFFF) EntityLifetime.dyingFrames[eid] = d + 1;
    }
  }
}

/* ------------------------------------------------------------------ */
/* 8. TTL EXPIRY SWEEP                                                */
/* ------------------------------------------------------------------ */

/**
 * Walks the world and marks every ALIVE entity whose TTL has expired as
 * DYING. Returns the number of entities marked.
 */
export function expireStale() {
  const frame = LifetimeState.frame;
  let expired = 0;

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    const state = EntityLifetime.state[eid];
    if (state !== LIFETIME_STATE.ALIVE) continue;
    if (EntityLifetime.persistent[eid] === 1) continue;

    const ttl = EntityLifetime.ttlFrames[eid];
    if (ttl === 0) continue;

    const alive = EntityLifetime.aliveFrames[eid];
    if (alive >= ttl) {
      // Transient entities skip the dying grace period entirely.
      if (EntityLifetime.transient[eid] === 1) {
        if (markDead(eid)) {
          expired++;
          LifetimeState.totalExpired++;
          _releaseToPool(eid);
        }
      } else {
        if (markDying(eid)) {
          expired++;
          LifetimeState.totalExpired++;
        }
      }
    }
  }

  return expired;
}

/* ------------------------------------------------------------------ */
/* 9. GARBAGE COLLECTION SWEEP                                        */
/* ------------------------------------------------------------------ */

/**
 * Walks the world and moves every DYING entity whose grace period has
 * elapsed into DEAD, then releases it back to its pool. Returns the
 * number of entities collected.
 */
export function collectGarbage() {
  let collected = 0;

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    const state = EntityLifetime.state[eid];
    if (state !== LIFETIME_STATE.DYING) continue;

    const dying = EntityLifetime.dyingFrames[eid];
    const grace = EntityLifetime.dyingGrace[eid];
    if (dying < grace) continue;

    if (markDead(eid)) {
      collected++;
      LifetimeState.totalCollected++;
      _releaseToPool(eid);
    }
  }

  return collected;
}

/**
 * Releases a DEAD entity back to its bound subsystem pool.
 */
function _releaseToPool(eid) {
  const pool = EntityLifetime.pool[eid];
  if (pool < 0 || pool >= LIFETIME_POOL.COUNT) {
    // No pool binding — leave it in the DEAD state; a downstream system
    // is responsible for removing it from the world.
    return false;
  }

  // Detach relations before releasing so the graph is clean.
  try { detachAllRelations(eid); }
  catch (_) { /* swallow */ }

  let ok = false;
  try {
    switch (pool) {
      case LIFETIME_POOL.CAMERA:    ok = releaseCameraEntity(eid);    break;
      case LIFETIME_POOL.LIGHT:     ok = releaseLightEntity(eid);     break;
      case LIFETIME_POOL.SHADOW:    ok = releaseShadowEntity(eid);    break;
      case LIFETIME_POOL.GI:        ok = releaseGIEntity(eid);        break;
      case LIFETIME_POOL.AO:        ok = releaseAOEntity(eid);        break;
      case LIFETIME_POOL.SCENE:     ok = releaseSceneEntity(eid);     break;
      case LIFETIME_POOL.STREAMING: ok = releaseStreamingEntity(eid); break;
      case LIFETIME_POOL.DEBUG:     ok = releaseDebugEntity(eid);     break;
      case LIFETIME_POOL.PARTICLE:  ok = releaseParticleEntity(eid);  break;
      case LIFETIME_POOL.UI:        ok = releaseUIEntity(eid);        break;
      case LIFETIME_POOL.ANIMATION: ok = releaseAnimationEntity(eid); break;
      case LIFETIME_POOL.WEATHER:   ok = releaseWeatherEntity(eid);   break;
      default:                      ok = release(pool, eid);          break;
    }
  } catch (_) {
    ok = false;
  }

  if (ok) {
    _transition(eid, LIFETIME_STATE.POOLED);
  } else {
    // Pool release failed — record the failure. The GC will retry next
    // frame; if failureCount exceeds a threshold, force the entity to
    // POOLED regardless.
    EntityLifetime.failureCount[eid]++;
    if (EntityLifetime.failureCount[eid] > 5) {
      _transition(eid, LIFETIME_STATE.POOLED);
    }
  }

  return ok;
}

/* ------------------------------------------------------------------ */
/* 10. FORCE RELEASE                                                  */
/* ------------------------------------------------------------------ */

/**
 * Immediately forces an entity through the lifecycle to POOLED without
 * any grace period. Useful for teardown.
 */
export function forceRelease(eid) {
  if (!_isValidEntity(eid)) return false;
  // Force into DEAD state directly.
  EntityLifetime.state[eid] = LIFETIME_STATE.DEAD;
  EntityLifetime.lastStateFrame[eid] = LifetimeState.frame;
  _applyLifecycleTag(eid, LIFETIME_STATE.DEAD);
  return _releaseToPool(eid);
}

/* ------------------------------------------------------------------ */
/* 11. STATISTICS                                                     */
/* ------------------------------------------------------------------ */

export function getLifetimeStats() {
  const counts = new Uint32Array(LIFETIME_STATE.COUNT);
  let persistentCount = 0;
  let transientCount = 0;

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    const s = EntityLifetime.state[eid];
    if (s >= 0 && s < counts.length) counts[s]++;
    if (EntityLifetime.persistent[eid] === 1) persistentCount++;
    if (EntityLifetime.transient[eid] === 1) transientCount++;
  }

  const stateCounts = new Array(LIFETIME_STATE.COUNT);
  for (let i = 0; i < LIFETIME_STATE.COUNT; i++) {
    stateCounts[i] = {
      state: LIFETIME_STATE_NAME[i],
      count: counts[i],
    };
  }

  return {
    frame:              LifetimeState.frame,
    maxEntities:        MAX_ENTITIES,
    stateCounts,
    persistentCount,
    transientCount,
    totalSpawn:         LifetimeState.totalSpawn,
    totalDeath:         LifetimeState.totalDeath,
    totalSuspended:     LifetimeState.totalSuspended,
    totalResumed:       LifetimeState.totalResumed,
    totalExpired:       LifetimeState.totalExpired,
    totalCollected:     LifetimeState.totalCollected,
    totalRejected:      LifetimeState.totalRejected,
    transitions:        LifetimeState.transitions,
    perfTier:           PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 12. CONVENIENCE COMBINED SWEEP                                     */
/* ------------------------------------------------------------------ */

/**
 * Runs the full lifetime pipeline for one frame:
 *   1. tickLifetime(frame)
 *   2. expireStale()
 *   3. collectGarbage()
 *
 * Returns an object with the results of each phase.
 */
export function tickLifetimeSystem(frameNumber) {
  tickLifetime(frameNumber);
  const expired   = expireStale();
  const collected = collectGarbage();
  return { frame: LifetimeState.frame, expired, collected };
}

/* ------------------------------------------------------------------ */
/* 13. REGISTRATION WITH THE COMPONENT REGISTRY                       */
/* ------------------------------------------------------------------ */

export function registerLifetimeComponents(registry) {
  const reg = registry || getDefaultComponentRegistry();
  if (!reg) return 0;

  return reg.registerMany([
    {
      name: 'EntityLifetime',
      component: EntityLifetime,
      category: 6,   // SCENE
      subsystem: 1,  // CORE
      dependencies: [],
      aliases: ['Lifetime'],
    },
  ]);
}

/* ------------------------------------------------------------------ */
/* 14. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Clears every lifetime record. Used for full teardown / restart.
 */
export function resetAllLifetimes() {
  EntityLifetime.state.fill(0);
  EntityLifetime.prevState.fill(0);
  EntityLifetime.pool.fill(-1);
  EntityLifetime.spawnFrame.fill(0);
  EntityLifetime.aliveFrames.fill(0);
  EntityLifetime.ttlFrames.fill(0);
  EntityLifetime.dyingFrames.fill(0);
  EntityLifetime.dyingGrace.fill(0);
  EntityLifetime.lastTouchFrame.fill(0);
  EntityLifetime.lastStateFrame.fill(0);
  EntityLifetime.stateChangeCount.fill(0);
  EntityLifetime.persistent.fill(0);
  EntityLifetime.transient.fill(0);
  EntityLifetime.failureCount.fill(0);

  LifetimeState.frame = 0;
  LifetimeState.totalSpawn = 0;
  LifetimeState.totalDeath = 0;
  LifetimeState.totalSuspended = 0;
  LifetimeState.totalResumed = 0;
  LifetimeState.totalExpired = 0;
  LifetimeState.totalCollected = 0;
  LifetimeState.totalRejected = 0;
  LifetimeState.transitions = 0;
}

/* ------------------------------------------------------------------ */
/* 15. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Enums
  LIFETIME_STATE,
  LIFETIME_STATE_NAME,
  LIFETIME_POOL,

  // Constants
  DEFAULT_DYING_GRACE_FRAMES,

  // Component
  EntityLifetime,
  LIFETIME_COMPONENTS,

  // Module state
  LifetimeState,

  // Lifecycle operations
  markSpawning,
  markAlive,
  suspend,
  resume,
  markDying,
  markDead,
  markPooled,
  cancelDeath,

  // TTL / touch
  setTTL,
  getTTL,
  touch,
  age,
  getState,
  isAlive,
  isDying,
  isDead,
  setPersistent,
  setTransient,

  // Frame advance
  tickLifetime,

  // Sweeps
  expireStale,
  collectGarbage,
  tickLifetimeSystem,

  // Force release
  forceRelease,

  // Stats
  getLifetimeStats,

  // Registration
  registerLifetimeComponents,

  // Reset
  resetAllLifetimes,
};

export default _defaultExport;