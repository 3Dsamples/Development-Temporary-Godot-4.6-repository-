API Documentation — src/core/009_scn_BiteCSVersionPolicy.js

File Purpose

This file enforces the bitECS 0.4.0 architectural redesign at runtime. bitECS 0.4.0 broke backward compatibility with 0.3.x: it removed defineComponent, removed the Types enum (f32, ui8, ui16, ui32, etc.), and moved to plain JavaScript objects whose fields are pre-sized typed arrays. Any accidental drift toward the old API — importing Types, calling defineComponent, mixing component shapes — will silently fail on Android, and the failures are subtle: silent resizing that causes GC spikes, entity IDs that become invalid, and SoA fields that read as undefined.

This module runs a one-shot audit at boot that:

· Verifies the loaded bitECS module exposes the required 0.4.0 surface (createWorld, addEntity, removeEntity, addComponent, removeComponent, hasComponent, query, entityExists, resetWorld, deleteWorld).
· Asserts the legacy surface is ABSENT (Types, defineComponent, defineSystem, defineQuery, enterQuery, exitQuery, Not, AnyOf, AllOf, changeDetection). If any of these leak through, the wrong version was served from the CDN.
· Validates every SoA component passed into createWorld is a plain object whose fields are typed arrays of exactly MAX_ENTITIES length.
· Verifies the world's internal dense-array capacity matches MAX_ENTITIES.
· Caches the audit result in a frozen record so every downstream module can assert conformance with one O(1) boolean check.
· Exposes the sanctioned defineSoAComponent factory so future components cannot accidentally break the invariant.
· Exposes a guarded ecs façade so call sites import the 0.4.0 surface from one place.

The audit runs ONCE at module import. After it passes, the hot path is a single integer comparison against a module-level boolean.

---

Exported Constants

AUDIT

Type: object (mutable, but only written during the boot audit)

Holds the audit record. Fields:

· version — the string 'bitECS-0.4.0', used in log messages.
· requiredSurface — frozen array of function names that MUST exist on the bitECS namespace.
· forbiddenSurface — frozen array of names that MUST NOT exist.
· missing — array of required names that were not found.
· forbidden — array of forbidden names that were found.
· componentsChecked — integer count of component fields inspected.
· componentsOk — integer count of component fields that passed.
· componentsBad — array of { name, field, reason } records for failures.
· passed — boolean, true only if every check succeeded.
· reason — human-readable string describing the outcome.

__bitecsAuditPassed

Type: boolean (module-private, not exported directly)

Set to true only after a successful runBiteCSAudit. Read by isBiteCSReady and assertBiteCSReady.

ecs

Type: frozen object

A façade re-exporting the ten sanctioned bitECS functions. Downstream modules import ecs instead of importing the raw bitECS namespace directly, so a single audit gate covers every call site.

Fields: createWorld, addEntity, removeEntity, addComponent, removeComponent, hasComponent, query, entityExists, resetWorld, deleteWorld.

bitecs (default export only)

The raw bitECS module namespace, exported for the rare case where a module needs to introspect the namespace (e.g., a debug tool listing all exports). Should not be used in production code paths.

---

Exported Functions

runBiteCSAudit(components, worldRef, maxEntities)

Parameters:

· components — the object passed to createWorld({ components }). May be null if the caller only wants to check the namespace surface.
· worldRef — the world handle returned by createWorld. May be null.
· maxEntities — the expected fixed capacity, typically 100000.

Returns: the AUDIT object.

Purpose: runs the four-stage audit.

Stage one: auditSurface() — verifies all required names exist and no forbidden names exist.

Stage two: auditComponents() — if components was supplied, iterates every field of every component and verifies each is an ArrayBuffer.isView with .length === maxEntities.

Stage three: auditWorld() — if worldRef was supplied and exposes a .components map, verifies every typed array inside has length maxEntities.

Stage four: writes the result into AUDIT, sets __bitecsAuditPassed to true on full success, and logs a summary via console.log or console.error.

If any stage fails, AUDIT.passed remains false, AUDIT.reason explains the failure, and the function returns the AUDIT object without throwing. The caller decides whether to throw. This is intentional because a boot audit failure should be recoverable — the caller may want to log and continue in a degraded mode.

auditComponents(components, maxEntities)

Parameters:

· components — the SoA component registry object.
· maxEntities — the expected length of every typed array.

Returns: boolean — true if every field of every component is a typed array of length maxEntities, false otherwise.

Purpose: pure helper that populates AUDIT.componentsBad with detailed failure records. Called by runBiteCSAudit. Exposed publicly for tools that want to re-audit after adding a component.

Behaviour:

For each top-level key of components, verifies the value is a non-null object. For each field of that value, verifies ArrayBuffer.isView(array) is true and array.length === maxEntities. On the first failing field, records { name, field, reason } and moves on to the next component.

auditWorld(worldRef, maxEntities)

Parameters:

· worldRef — a bitECS world handle.
· maxEntities — the expected length.

Returns: { ok: boolean, reason: string }.

Purpose: verifies the world's internally-held component arrays (if exposed via worldRef.components) match the expected capacity. If worldRef.components does not exist — some bitECS 0.4.0 patch versions hide it — the function returns { ok: true, reason: '' } because the check cannot run and failing open is safer than failing closed.

assertBiteCSReady()

Parameters: none.

Returns: boolean true on success.

Throws: an Error if __bitecsAuditPassed is false.

Purpose: the hot-path guard. Every downstream module that depends on bitECS 0.4.0 semantics should call this once at its own module init. It is O(1), zero-alloc, and reads a single module-level boolean.

isBiteCSReady()

Parameters: none.

Returns: boolean — the current value of __bitecsAuditPassed. Use this for non-throwing checks (e.g., inside a conditional branch in a debug tool).

getAudit()

Parameters: none.

Returns: the AUDIT object.

Purpose: exposes the audit record for debug HUDs, regression tools, and CI verification. Safe to call at any time — the record is written once and then read-only.

defineSoAComponent(name, fields, maxEntities)

Parameters:

· name — the component name, used in error messages and stored on the frozen _meta property.
· fields — an object mapping field names to TypedArray constructors (e.g., { x: Float32Array, y: Float32Array }).
· maxEntities — the fixed capacity.

Returns: a plain object whose keys are the supplied field names, each value is a freshly-allocated typed array of length maxEntities. A non-enumerable _meta property records { name, maxEntities, fields: [...] } as a frozen descriptor.

Throws: if name is not a non-empty string, if fields is not an object, or if any field value is not a constructor function.

Purpose: the ONLY sanctioned way to declare a new SoA component in this project. Guarantees the fixed-length, typed-array invariant that the rest of the engine relies on.

---

Internal Functions (Not Exported but Documented)

auditSurface()

No parameters, no return value. Populates AUDIT.missing and AUDIT.forbidden. Called by runBiteCSAudit.

_installDefaultLoaders()

Not present in this file — documented in 015_rnd_AssetManifest.js.

---

Usage Pattern

Downstream modules should do:

```
import { ecs, assertBiteCSReady, defineSoAComponent } from '../core/009_scn_BiteCSVersionPolicy.js';

assertBiteCSReady(); // O(1), zero-alloc, throws if audit failed

export const MyComponent = defineSoAComponent('MyComponent', {
  x: Float32Array,
  y: Float32Array,
  flag: Uint8Array,
}, 100000);
```

Any accidental import of Types or call to defineComponent elsewhere in the codebase will fail at the CDN load step — because the module namespace returned by the CDN will not contain those symbols — and this audit catches it at boot with a precise error message naming the offending symbol.

---

Default Export

The default export bundles every named export: runBiteCSAudit, assertBiteCSReady, isBiteCSReady, getAudit, auditComponents, auditWorld, defineSoAComponent, ecs, bitecs.
