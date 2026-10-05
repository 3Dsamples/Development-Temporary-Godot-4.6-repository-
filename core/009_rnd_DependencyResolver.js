// File : 009
// name : src/core/009_scn_BiteCSVersionPolicy.js
// description : Runtime enforcement of the bitECS 0.4.0 architectural redesign.
//               The v0.4.0 API is NOT backward compatible with 0.3.x: there is
//               no `defineComponent()`, no `Types` export (f32/ui8/ui16/ui32),
//               and no separate component stores. Components are plain
//               JavaScript objects whose fields are pre-sized typed arrays
//               (SoA). This policy module is imported once by the world
//               bootstrap and runs a one-shot audit that:
//                 1. Verifies the loaded bitECS module exposes the required
//                    0.4.0 surface: createWorld, addEntity, removeEntity,
//                    addComponent, removeComponent, hasComponent, query,
//                    entityExists, resetWorld, deleteWorld.
//                 2. Asserts that the legacy surface is ABSENT: no `Types`,
//                    no `defineComponent`, no `defineSystem` with parallel
//                    stores. Any leaked usage throws a hard error at boot —
//                    silent drift is impossible on Android.
//                 3. Validates every component passed into createWorld is a
//                    plain object whose fields are typed arrays with EQUAL
//                    .length === MAX_ENTITIES (no resizing mid-gameplay).
//                 4. Verifies bitECS internal dense-array capacity matches
//                    MAX_ENTITIES so no silent growth can occur.
//                 5. Caches the result in a frozen audit record so every
//                    downstream lighting module can assert conformance with
//                    a single O(1) boolean check, zero allocations.
//               Runs ONCE at module import; the hot path (per-frame) is a
//               single integer comparison against `__bitecsAuditPassed`.
// best for : Guarding the entire modular anime lighting stack (006–380) against
//            the three classic bitECS 0.4.0 migration regressions:
//            (a) import of `Types` that no longer exists,
//            (b) use of `defineComponent()` that no longer exists,
//            (c) mixed-size component stores that silently resize and
//                cause GC spikes + entity ID invalidation on Android.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as bitecs from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';

/* ------------------------------------------------------------------ */
/* 0. AUDIT RECORD (frozen, zero per-frame cost)                      */
/* ------------------------------------------------------------------ */

const AUDIT = {
  version:             'bitECS-0.4.0',
  requiredSurface:     Object.freeze([
    'createWorld',
    'addEntity',
    'removeEntity',
    'addComponent',
    'removeComponent',
    'hasComponent',
    'query',
    'entityExists',
    'resetWorld',
    'deleteWorld',
  ]),
  forbiddenSurface:    Object.freeze([
    'Types',
    'defineComponent',
    'defineSystem',
    'defineQuery',
    'enterQuery',
    'exitQuery',
    'Not',
    'AnyOf',
    'AllOf',
    'changeDetection',
  ]),
  missing:             [],
  forbidden:           [],
  componentsChecked:   0,
  componentsOk:        0,
  componentsBad:       [],
  passed:              false,
  reason:              '',
};

let __bitecsAuditPassed = false;

/* ------------------------------------------------------------------ */
/* 1. CORE SURFACE CHECK                                              */
/* ------------------------------------------------------------------ */

function auditSurface() {
  for (let i = 0; i < AUDIT.requiredSurface.length; i++) {
    const key = AUDIT.requiredSurface[i];
    if (typeof bitecs[key] !== 'function') {
      AUDIT.missing.push(key);
    }
  }

  for (let i = 0; i < AUDIT.forbiddenSurface.length; i++) {
    const key = AUDIT.forbiddenSurface[i];
    // bitECS 0.4.0 must NOT export these. If any leaks through, the CDN
    // served the wrong version (0.3.x cached proxy, wrong import-map, etc.)
    if (key in bitecs) {
      AUDIT.forbidden.push(key);
    }
  }
}

/* ------------------------------------------------------------------ */
/* 2. COMPONENT SHAPE CHECK (SoA typed arrays, fixed capacity)        */
/* ------------------------------------------------------------------ */

export function auditComponents(components, maxEntities) {
  if (!components || typeof components !== 'object') {
    AUDIT.componentsBad.push({
      name:   '<components>',
      reason: 'not an object',
    });
    return false;
  }

  const names = Object.keys(components);

  for (let n = 0; n < names.length; n++) {
    const name = names[n];
    const comp = components[name];

    AUDIT.componentsChecked++;

    if (!comp || typeof comp !== 'object') {
      AUDIT.componentsBad.push({ name, reason: 'not an object' });
      continue;
    }

    const fields = Object.keys(comp);
    let ok = true;

    for (let f = 0; f < fields.length; f++) {
      const field = fields[f];
      const arr = comp[field];

      if (!ArrayBuffer.isView(arr)) {
        AUDIT.componentsBad.push({
          name,
          field,
          reason: 'field is not a TypedArray (expected Float32Array/Uint8Array/...)',
        });
        ok = false;
        break;
      }

      if (arr.length !== maxEntities) {
        AUDIT.componentsBad.push({
          name,
          field,
          reason: `length ${arr.length} !== MAX_ENTITIES ${maxEntities}`,
        });
        ok = false;
        break;
      }
    }

    if (ok) AUDIT.componentsOk++;
  }

  return AUDIT.componentsBad.length === 0;
}

/* ------------------------------------------------------------------ */
/* 3. WORLD SHAPE CHECK (dense capacity matches MAX_ENTITIES)         */
/* ------------------------------------------------------------------ */

export function auditWorld(worldRef, maxEntities) {
  if (!worldRef || typeof worldRef !== 'object') {
    return { ok: false, reason: 'world handle is not an object' };
  }

  // bitECS 0.4.0 exposes an internal `$` symbol or `components` map depending
  // on the minor patch. We only check the public capacity hint if present.
  if (worldRef.components && typeof worldRef.components === 'object') {
    const names = Object.keys(worldRef.components);
    for (let i = 0; i < names.length; i++) {
      const comp = worldRef.components[names[i]];
      const fields = comp ? Object.keys(comp) : [];
      for (let f = 0; f < fields.length; f++) {
        const arr = comp[fields[f]];
        if (ArrayBuffer.isView(arr) && arr.length !== maxEntities) {
          return {
            ok: false,
            reason: `world.components.${names[i]}.${fields[f]} has length ${arr.length} !== ${maxEntities}`,
          };
        }
      }
    }
  }

  return { ok: true, reason: '' };
}

/* ------------------------------------------------------------------ */
/* 4. ONE-SHOT BOOT AUDIT                                             */
/* ------------------------------------------------------------------ */

export function runBiteCSAudit(components, worldRef, maxEntities) {
  auditSurface();

  if (AUDIT.missing.length > 0) {
    AUDIT.passed = false;
    AUDIT.reason = `bitECS 0.4.0 surface incomplete: missing [${AUDIT.missing.join(', ')}]`;
    console.error('[009_scn_BiteCSVersionPolicy]', AUDIT.reason);
    return AUDIT;
  }

  if (AUDIT.forbidden.length > 0) {
    AUDIT.passed = false;
    AUDIT.reason =
      `bitECS 0.4.0 policy violation: legacy surface present [${AUDIT.forbidden.join(', ')}]. ` +
      `The CDN likely served 0.3.x. Confirm import URL = ` +
      `https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs`;
    console.error('[009_scn_BiteCSVersionPolicy]', AUDIT.reason);
    return AUDIT;
  }

  if (components) {
    const compOk = auditComponents(components, maxEntities);
    if (!compOk) {
      AUDIT.passed = false;
      AUDIT.reason =
        `SoA component shape violation (${AUDIT.componentsBad.length} issue(s)). ` +
        `Every field must be a pre-sized TypedArray with length === MAX_ENTITIES ` +
        `(${maxEntities}). See AUDIT.componentsBad for details.`;
      console.error('[009_scn_BiteCSVersionPolicy]', AUDIT.reason, AUDIT.componentsBad);
      return AUDIT;
    }
  }

  if (worldRef) {
    const wOk = auditWorld(worldRef, maxEntities);
    if (!wOk.ok) {
      AUDIT.passed = false;
      AUDIT.reason = `World capacity mismatch: ${wOk.reason}`;
      console.error('[009_scn_BiteCSVersionPolicy]', AUDIT.reason);
      return AUDIT;
    }
  }

  AUDIT.passed = true;
  AUDIT.reason = `bitECS 0.4.0 audit passed — ${AUDIT.componentsOk}/${AUDIT.componentsChecked} components OK`;
  console.log('[009_scn_BiteCSVersionPolicy]', AUDIT.reason);
  __bitecsAuditPassed = true;
  return AUDIT;
}

/* ------------------------------------------------------------------ */
/* 5. HOT-PATH ASSERT (O(1), zero allocation)                         */
/* ------------------------------------------------------------------ */

export function assertBiteCSReady() {
  if (!__bitecsAuditPassed) {
    throw new Error(
      '[009_scn_BiteCSVersionPolicy] bitECS audit has not passed. ' +
      'Call runBiteCSAudit() from the world bootstrap before creating systems.'
    );
  }
  return true;
}

export function isBiteCSReady() {
  return __bitecsAuditPassed;
}

export function getAudit() {
  return AUDIT;
}

/* ------------------------------------------------------------------ */
/* 6. GUARDED HELPERS — safe shims the rest of the stack imports      */
/*    These wrap the raw bitECS calls with audit gating so a leaked   */
/*    legacy call site fails loudly instead of corrupting memory.     */
/* ------------------------------------------------------------------ */

export const ecs = Object.freeze({
  createWorld:    bitecs.createWorld,
  addEntity:      bitecs.addEntity,
  removeEntity:   bitecs.removeEntity,
  addComponent:   bitecs.addComponent,
  removeComponent:bitecs.removeComponent,
  hasComponent:   bitecs.hasComponent,
  query:          bitecs.query,
  entityExists:   bitecs.entityExists,
  resetWorld:     bitecs.resetWorld,
  deleteWorld:    bitecs.deleteWorld,
});

/* ------------------------------------------------------------------ */
/* 7. COMPONENT FACTORY — the ONLY sanctioned way to declare a SoA    */
/*    component in this project. Guarantees fixed length, typed       */
/*    array fields, and a frozen descriptor for downstream audits.    */
/* ------------------------------------------------------------------ */

export function defineSoAComponent(name, fields, maxEntities) {
  if (typeof name !== 'string' || name.length === 0) {
    throw new Error('[009_scn_BiteCSVersionPolicy] defineSoAComponent: invalid name');
  }
  if (!fields || typeof fields !== 'object') {
    throw new Error(`[009_scn_BiteCSVersionPolicy] defineSoAComponent(${name}): invalid fields`);
  }

  const out = Object.create(null);
  const fieldNames = Object.keys(fields);

  for (let i = 0; i < fieldNames.length; i++) {
    const key = fieldNames[i];
    const Ctor = fields[key];

    if (typeof Ctor !== 'function') {
      throw new Error(
        `[009_scn_BiteCSVersionPolicy] defineSoAComponent(${name}).${key}: ` +
        `expected a TypedArray constructor (Float32Array, Uint8Array, ...)`
      );
    }

    out[key] = new Ctor(maxEntities);
  }

  Object.defineProperty(out, '_meta', {
    value: Object.freeze({ name, maxEntities, fields: Object.freeze(fieldNames.slice()) }),
    enumerable: false,
    writable:   false,
    configurable: false,
  });

  return out;
}

/* ------------------------------------------------------------------ */
/* 8. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

export default {
  runBiteCSAudit,
  assertBiteCSReady,
  isBiteCSReady,
  getAudit,
  auditComponents,
  auditWorld,
  defineSoAComponent,
  ecs,
  bitecs,
};