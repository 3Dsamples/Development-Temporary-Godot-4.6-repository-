API Documentation — src/core/001_lgt_ThreeLightsOnlyPolicy.js

File Purpose

This file is the authoritative enforcement module for the anime lighting stack, guaranteeing that only the six sanctioned Three.js r185 light types are ever instantiated:

1. THREE.AmbientLight
2. THREE.HemisphereLight
3. THREE.DirectionalLight
4. THREE.PointLight
5. THREE.SpotLight
6. THREE.RectAreaLight

Any other light — legacy lights, custom shader-emitted lights, fake Object3D emissives, third-party light classes — is rejected at registration with a hard, typed error and a signal, so no lighting bug can silently corrupt the frame.

The policy exists because Three.js r185 has a precise set of light types with well-defined behavior on Android GPUs. A custom light class may work on desktop and fail on Mali. A shader-emitted light may pass a visual check and then break the shadow pipeline. A fake emissive Object3D may look correct in isolation but produce inconsistent output when the real light list is rebuilt. The policy eliminates that entire class of bug by making the sanctioned set explicit and enforced.

The module also provides two fully independent extension systems so the "add a new kind of light" workflow is possible WITHOUT breaking the policy:

Light behaviors. A behavior is a plain object with attach(light, ctx), update(dt, elapsed, light, ctx), and detach(light, ctx). Behaviors turn any sanctioned light into a richer light — flicker, pulse, day cycle, IES profile, cookie, shadow softness ramp, temperature drift, rim boost, biome blend, interior/exterior crossfade, magic emissive animation. Multiple behaviors stack on one light. No new light types are introduced.

Light composites. A composite is a named set of sanctioned lights plus behaviors registered as one logical "light kind." fire_light is a PointLight plus a flicker behavior plus a warm emissive proxy. moon_cycle is a DirectionalLight plus AmbientLight plus HemisphereLight plus a day-cycle behavior. neon is a RectAreaLight plus a pulse behavior. Callers instantiate composites by id and get back a CompositeHandle that exposes the same API as a single light.

The module provides:

· A fixed-capacity SoA registry tracking every sanctioned light.
· Per-light type classification and validation.
· A behavior registry with per-kind compatibility checks.
· A composite registry with per-composite member instantiation.
· Scene auditing that rejects non-sanctioned lights.
· Integration with the error boundary, validation, event bus, tier resolver, and logger.

Every subsystem that creates a light must register it through this module. The six sanctioned factories at the bottom of the file (createSanctionedAmbientLight, createSanctionedHemisphereLight, createSanctionedDirectionalLight, createSanctionedPointLight, createSanctionedSpotLight, createSanctionedRectAreaLight) are the ONLY sanctioned ways to create a light.

---

Exported Constants

PERF_TIER_LOCAL

Internal. The cached PERF_TIER string from getPerfTier().

MAX_LIGHTS

Type: number

Value: 512 on HIGH, 256 on MEDIUM, 128 on LOW.

The fixed capacity of the light registry. Sized to accommodate every light the lighting stack will ever register simultaneously — a full exterior scene with the sun, moon, hemisphere fill, ambient fill, plus a dozen point lights for magic and interior, plus rect-area lights for windows and neon.

MAX_BEHAVIORS_PER_LIGHT

Type: number

Value: 8

The maximum number of behaviors that can stack on a single light. Eight is generous — most lights use one or two, and the composite fire light uses two.

MAX_REGISTERED_BEHAVIORS

Type: number

Value: 64

The maximum number of registered behavior descriptors. Sized to accommodate every behavior the engine defines plus user-registered extensions.

MAX_REGISTERED_COMPOSITES

Type: number

Value: 32

The maximum number of registered composite descriptors.

MAX_COMPOSITE_MEMBERS

Type: number

Value: 8

The maximum number of member lights a single composite can declare.

SANCTIONED_LIGHT_TYPE

Type: frozen enum

The six sanctioned light types, keyed by an internal id.

· AMBIENT = 0
· HEMISPHERE = 1
· DIRECTIONAL = 2
· POINT = 3
· SPOT = 4
· RECT_AREA = 5
· COUNT = 6

SANCTIONED_LIGHT_TYPE_NAME

Type: frozen array

Values: ['ambient', 'hemisphere', 'directional', 'point', 'spot', 'rect_area'].

SANCTIONED_LIGHT_CLASS

Type: frozen array

The six Three.js class references, in the same order as the enum: [AmbientLight, HemisphereLight, DirectionalLight, PointLight, SpotLight, RectAreaLight].

FORBIDDEN_LIGHT_NAMES (internal)

Type: frozen array

The forbidden light class names. Includes LightProbe, LightProbeGenerator, HemisphereLightProbe, AmbientLightProbe, SpotLightShadow, and RectAreaLightUniformsLib. If any of these appears in the constructor name of a candidate light, the audit rejects.

COMPOSITE_LIGHT_KIND

Type: frozen enum

The category of a composite. Used for debug grouping and event tagging.

· CUSTOM = 0
· FIRE = 1
· MOON_CYCLE = 2
· SUN_CYCLE = 3
· NEON = 4
· MAGIC = 5
· INTERIOR_LAMP = 6
· WINDOW_SHAFT = 7
· CAUSTIC = 8
· AURORA = 9
· COUNT = 10

COMPOSITE_LIGHT_KIND_NAME

Type: frozen array

Values: ['custom', 'fire', 'moon_cycle', 'sun_cycle', 'neon', 'magic', 'interior_lamp', 'window_shaft', 'caustic', 'aurora'].

---

Module-Level State (Not Exported Directly)

_lightIdCounter

Type: number

Monotonic counter for registered light ids.

_defaultPolicy

Type: ThreeLightsOnlyPolicy | null

The module-level singleton.

---

Internal Helper Functions (Documented)

_nextLightId()

Returns: the next monotonic light id.

_now()

Returns: the current high-resolution timestamp.

_classifyLight(light)

Parameters: light — any value.

Returns: one of SANCTIONED_LIGHT_TYPE, or -1 if the object is not one of the six sanctioned types.

Purpose: reads the light's isAmbientLight, isHemisphereLight, isDirectionalLight, isPointLight, isSpotLight, or isRectAreaLight flags. If none of those flags is set, returns -1.

_isThreeLight(light)

Parameters: light — any value.

Returns: true if light.isLight === true.

Purpose: distinguishes a light from any other Object3D. Non-lights are silently ignored by the policy because the policy is only concerned with lights.

---

Exported Class — LightSlot

One instance per registered light.

Constructor

```
new LightSlot(index)
```

Parameters: index — the slot's array index.

Instance Properties

· index — the slot index.
· id — the monotonic light id.
· light — the actual THREE.Light.
· type — one of SANCTIONED_LIGHT_TYPE.
· compositeId — the id of the composite that owns this light, or -1.
· ownerId — the subsystem id that registered this light, or -1.
· behaviorIds — an Int32Array(MAX_BEHAVIORS_PER_LIGHT) of behavior registry indices.
· behaviorCtx — an array of contexts, parallel.
· behaviorCount — the number of behaviors.
· sceneRef — the scene the light was added to.
· registeredAt — the timestamp when registered.
· castShadow — 1 if the light casts shadows.
· receivesShadow — 1 if the light receives shadows.
· tags — free-form bitmask for downstream systems.
· updateMs — the cost of the last behavior update in milliseconds.
· lastUpdateFrame — the frame of the last behavior update.

Instance Methods

reset()

Returns: nothing. Zeroes every field.

---

Exported Class — BehaviorDescriptor

A frozen descriptor for a behavior.

Constructor

```
new BehaviorDescriptor(spec)
```

Parameters: spec — an object with:

· id — a unique string.
· name — an optional display name.
· attach(light, ctx) — optional, called once when the behavior is bound.
· update(dt, elapsed, light, ctx) — the per-frame function.
· detach(light, ctx) — optional, called on unbind.
· validate(light) — optional, returns boolean.
· kinds — an optional array of SANCTIONED_LIGHT_TYPE values that the behavior supports.

Instance Properties

· id, name, attach, update, detach, validate, kinds.

Instance Methods

supports(type)

Parameters: type — one of SANCTIONED_LIGHT_TYPE.

Returns: boolean. True if the behavior supports the light type.

---

Exported Class — CompositeDescriptor

A frozen descriptor for a composite.

Constructor

```
new CompositeDescriptor(spec)
```

Parameters: spec — an object with:

· id — a unique string.
· name — an optional display name.
· kind — one of COMPOSITE_LIGHT_KIND.
· members — an array of member descriptors.
· behaviors — an optional array of { id, ctx } behavior bindings applied to the anchor.
· create — an optional custom creator function.

Each member descriptor has:

· type — one of SANCTIONED_LIGHT_TYPE.
· name — an optional display name.
· color — an optional [r, g, b].
· intensity — an optional number.
· distance — for point/spot.
· angle — for spot.
· penumbra — for spot.
· decay — for point/spot.
· width, height — for rect area.
· position — an optional [x, y, z] offset from the composite's base position.
· target — an optional [x, y, z] offset.
· castShadow — boolean.
· behaviorIds — an optional array of behavior ids to bind to this member.

Instance Properties

· id, name, kind, members, behaviors, create.

---

Exported Class — CompositeHandle

The handle returned by createComposite. Exposes a single-light-like API.

Constructor

```
new CompositeHandle(descriptor, members)
```

Instance Properties

· descriptor — the CompositeDescriptor.
· members — the array of THREE.Light instances.
· registered — true if the composite is still live.
· anchor — the primary member, or null.

Instance Properties (Getters)

· light — the anchor.
· position — the anchor's position, or null.

Instance Methods

setIntensity(v)

Parameters: v — the intensity.

Returns: this. Sets every member's intensity.

setColor(rgb)

Parameters: rgb — a [r, g, b] array.

Returns: this. Sets every member's color. For a hemisphere light, also sets the ground color at 50 % of the value.

setVisible(visible)

Parameters: visible — boolean.

Returns: this. Sets every member's visible flag.

dispose(policy)

Parameters: policy — an optional policy instance.

Returns: boolean. Unregisters every member via the policy and clears the members array.

---

Exported Class — ThreeLightsOnlyPolicy

The main policy enforcer.

Constructor

```
new ThreeLightsOnlyPolicy(options = {})
```

Parameters:

· enforce — if true, the policy rejects non-sanctioned lights. Default true unless the tier is LOW.
· auditOnRegister — if true, audits every light at registration. Default true.
· tripBoundary — if true, violations trip a boundary. Default true.
· boundaryName — the boundary name. Default 'policy.three_lights_only'.
· logChannel — the logger channel. Default LOG_CHANNEL.LIGHTS.
· attachDefaultBehaviors — reserved. Default true.
· autoValidate — if true, validates light parameters at registration. Default true.

Constructor work:

1. Allocates slots — an array of MAX_LIGHTS entries.
2. Initializes count, byLight (Map), byId (Map).
3. Allocates behaviors, composites, behaviorByName, compositeByName.
4. Initializes the rejection log ring.
5. Initializes frame.
6. If tripBoundary, creates a boundary on the error manager.
7. Lazily resolves the event bus and logger.
8. Calls _installDefaultBehaviors() to register the seven canonical behaviors.
9. Calls _installDefaultComposites() to register the eight canonical composites.

Instance Properties

· options — the merged options.
· capacity — MAX_LIGHTS.
· slots — the light slot array.
· count — the number of registered lights.
· byLight — the Map from THREE.Light to slot index.
· byId — the Map from light id to slot index.
· behaviors — the behavior registry array.
· behaviorCount — the number of registered behaviors.
· behaviorByName — the Map from behavior id to descriptor.
· composites — the composite registry array.
· compositeCount — the number of registered composites.
· compositeByName — the Map from composite id to descriptor.
· rejections — the total rejection count.
· lastRejection — the last rejection record.
· frame — the current frame.

Instance Methods

_log()

Internal. Lazily resolves the logger.

beginFrame(frameNumber)

Parameters: frameNumber — the current frame number.

Returns: nothing.

_allocateSlot()

Internal. Returns a free slot index, or -1 if the registry is full.

registerLight(light, options = {})

Parameters:

· light — the THREE.Light to register.
· options.ownerId — the owning subsystem id.
· options.scene — the scene the light was added to.
· options.tags — a bitmask.
· options.behaviorIds — an optional array of behavior ids to bind immediately.
· options.behaviorCtx — an optional context for the behaviors.

Returns: the light's internal id, or -1 if rejected.

Purpose: the primary light registration entry point.

Flow:

1. Calls _auditLight(light). If the audit fails, returns -1.
2. If the light is already registered, returns the existing id.
3. Allocates a slot.
4. Populates the slot with the light, its type, owner, scene, timestamps, and shadow flags.
5. Adds the light to byLight and the slot to byId.
6. If autoValidate, calls _validateLightParameters.
7. If behaviorIds was provided, attaches each.
8. Logs a debug message if enforce is on.
9. Emits LIGHT_ADDED on the event bus.
10. Returns the id.

unregisterLight(light)

Parameters: light — the light to unregister.

Returns: boolean.

Purpose: detaches every behavior, removes the light from both maps, compacts the slot array, and emits LIGHT_REMOVED.

_auditLight(light)

Internal. Returns true if the light is sanctioned.

Flow:

1. If light is null, records a rejection with reason null.
2. If _isThreeLight(light) is false, returns true silently (non-lights are ignored).
3. Checks the constructor name against FORBIDDEN_LIGHT_NAMES. Any match is a rejection.
4. Calls _classifyLight(light). If the result is -1, the light is not one of the six sanctioned types, so the audit records an unknown_type rejection.

_validateLightParameters(slot)

Internal. Uses the default validator from 031_rnd_Validation.js to check the light's intensity, color, and type-specific parameters.

_reject(reason, light, message)

Internal. Records a rejection, logs an error, emits a lights.policy.rejected event, and trips the boundary if enforce is on.

auditScene(scene)

Parameters: scene — the scene to traverse.

Returns: the number of non-sanctioned lights discovered.

Purpose: iterates every object in the scene graph and rejects any light that is not one of the six sanctioned types. Used at boot and by the debug HUD.

registerBehavior(spec)

Parameters: spec — the behavior descriptor spec.

Returns: boolean. Registers a new behavior.

unregisterBehavior(id)

Returns: boolean.

getBehavior(id)

Returns: the BehaviorDescriptor, or null.

listBehaviors()

Returns: an array of behavior ids.

attachBehavior(light, behaviorId, ctx)

Parameters:

· light — the light to attach to.
· behaviorId — the behavior id.
· ctx — the context.

Returns: boolean.

Purpose: attaches a behavior to a light.

Flow:

1. Looks up the light's slot.
2. Looks up the behavior descriptor.
3. Checks the behavior's kinds compatibility against the light's type.
4. Runs the behavior's validate if present.
5. If the light's behavior count is full, returns false.
6. Calls the behavior's attach if present.
7. Records the behavior in the slot.

detachBehavior(light, behaviorId)

Parameters:

· light — the light.
· behaviorId — the behavior id.

Returns: boolean.

Purpose: detaches a behavior, calling its detach if present, and compacts the slot's behavior arrays.

_behaviorIndex(descriptor)

Internal. Returns the registry index of a behavior descriptor.

updateBehaviors(dt, elapsed)

Parameters:

· dt — the delta time in seconds.
· elapsed — total elapsed seconds.

Returns: nothing.

Purpose: the per-frame behavior tick. Iterates every registered light and runs each attached behavior's update.

registerComposite(spec)

Parameters: spec — the composite descriptor spec.

Returns: boolean.

Purpose: registers a composite. Verifies every member's type is a sanctioned one.

unregisterComposite(id)

Returns: boolean.

getComposite(id)

Returns: the CompositeDescriptor, or null.

listComposites()

Returns: an array of composite ids.

createComposite(id, scene, options = {})

Parameters:

· id — the composite id.
· scene — the scene to add the member lights to.
· options.position — a [x, y, z] base position.
· options.target — a [x, y, z] base target.
· options.ownerId — an owner id.

Returns: a CompositeHandle, or null.

Purpose: instantiates a composite.

Flow:

1. Looks up the descriptor. Returns null if unknown.
2. If the descriptor has a custom create function, calls it and uses its result if it returns a handle.
3. Otherwise, iterates the members. For each, calls _instantiateMember to create the underlying THREE.Light. Applies position and target offsets from the base position. Adds the light to the scene. Registers it with the policy. Pushes it into the members array.
4. If the composite declares top-level behaviors, attaches them to the anchor member.
5. Returns a new CompositeHandle.

_instantiateMember(m)

Internal. Instantiates a single member light from its descriptor. Uses a switch over the member's type to construct the correct THREE.Light subclass.

_installDefaultBehaviors()

Internal. Registers the seven canonical behaviors:

· flicker — intensity modulation with layered sine waves. Supports point, spot, and rect area.
· pulse — smooth sinusoidal modulation. Supports all types.
· day_cycle — advances a day-cycle value and updates the light's position and intensity. Supports directional, hemisphere, and ambient.
· ies_profile — broadcasts a ballast factor via userData. Supports point, spot, and rect area.
· temperature_drift — slowly shifts the color temperature. Supports all types.
· rim_boost — modulates rim intensity. Supports directional.
· biome_blend — crossfades a blend value via damping. Supports all types.
· interior_crossfade — broadcasts an indoor weight. Supports all types.

_installDefaultComposites()

Internal. Registers the eight canonical composites:

· fire_light — a PointLight plus a flicker behavior, plus a low-intensity AmbientLight for fill.
· moon_cycle — a DirectionalLight plus AmbientLight plus HemisphereLight, with a day_cycle behavior.
· sun_cycle — a DirectionalLight plus HemisphereLight plus AmbientLight, with a day_cycle behavior.
· neon — a RectAreaLight with a pulse behavior.
· magic_glow — a PointLight with a flicker behavior.
· interior_lamp — a PointLight with a temperature_drift behavior.
· window_shaft — a DirectionalLight.
· caustic — a PointLight with a pulse behavior.
· aurora — a HemisphereLight.

getCount(), getCapacity(), getRejections(), getLastRejection()

Trivial accessors.

getSlotByLight(light)

Returns: the LightSlot, or null.

getSlotById(id)

Returns: the LightSlot, or null.

forEachLight(fn, ctx)

Parameters:

· fn — a callback (light, slot) => void.
· ctx — the context.

Returns: nothing.

getStats()

Returns: an object with frame, registeredLights, capacity, totalBehaviors, rejectionTotal, lastRejection, registeredBehaviors, registeredComposites, behaviorIds, compositeIds, typeCounts (a per-type count array), enforce, perfTier.

reset()

Returns: this. Clears every slot and every registry.

dispose()

Returns: this. Resets and nulls every internal array.

---

Exported Hot-Path Wrapper Functions

These delegate to the module-level singleton.

· lightsBeginFrame(frameNumber) — sets the frame.
· registerLight(light, options) — registers a light.
· unregisterLight(light) — unregisters a light.
· attachLightBehavior(light, behaviorId, ctx) — attaches a behavior.
· detachLightBehavior(light, behaviorId) — detaches a behavior.
· updateLightBehaviors(dt, elapsed) — ticks every behavior.
· registerLightBehavior(spec) — registers a behavior.
· registerLightComposite(spec) — registers a composite.
· createCompositeLight(id, scene, options) — instantiates a composite.
· auditSceneLights(scene) — audits a scene.

---

Exported Sanctioned Light Factories

These are the ONLY sanctioned ways to create a light.

createSanctionedAmbientLight(spec = {})

Creates a THREE.AmbientLight, applies the given color and intensity, registers it, and returns it.

createSanctionedHemisphereLight(spec = {})

Creates a THREE.HemisphereLight with the given sky color, ground color, intensity, and position. Registers and returns.

createSanctionedDirectionalLight(spec = {})

Creates a THREE.DirectionalLight with the given color, intensity, position, target, and shadow flag. Registers and returns.

createSanctionedPointLight(spec = {})

Creates a THREE.PointLight with the given color, intensity, distance, decay, position, and shadow flag. Registers and returns.

createSanctionedSpotLight(spec = {})

Creates a THREE.SpotLight with the given color, intensity, distance, angle, penumbra, decay, position, target, and shadow flag. Registers and returns.

createSanctionedRectAreaLight(spec = {})

Creates a THREE.RectAreaLight with the given color, intensity, width, height, and position. Registers and returns.

All six factories accept the same spec shape:

· color — [r, g, b] in linear [0, 1].
· groundColor — for hemisphere only.
· intensity — scalar.
· distance — for point/spot.
· decay — for point/spot.
· angle — for spot.
· penumbra — for spot.
· width, height — for rect area.
· position — [x, y, z].
· target — [x, y, z].
· castShadow — boolean.
· ownerId — optional subsystem id.

---

Exported Functions

getDefaultThreeLightsOnlyPolicy()

Returns: the module-level singleton ThreeLightsOnlyPolicy, creating it on first call.

disposeDefaultThreeLightsOnlyPolicy()

Returns: nothing.

createThreeLightsOnlyPolicy(options = {})

Returns: a new ThreeLightsOnlyPolicy.

---

Default Export

The default export bundles: ThreeLightsOnlyPolicy, LightSlot, BehaviorDescriptor, CompositeDescriptor, CompositeHandle, createThreeLightsOnlyPolicy, getDefaultThreeLightsOnlyPolicy, disposeDefaultThreeLightsOnlyPolicy, the ten lgt* wrapper functions, the six sanctioned light factories, SANCTIONED_LIGHT_TYPE, SANCTIONED_LIGHT_TYPE_NAME, SANCTIONED_LIGHT_CLASS, COMPOSITE_LIGHT_KIND, COMPOSITE_LIGHT_KIND_NAME, FORBIDDEN_LIGHT_NAMES, MAX_LIGHTS, MAX_BEHAVIORS_PER_LIGHT, MAX_REGISTERED_BEHAVIORS, MAX_REGISTERED_COMPOSITES, MAX_COMPOSITE_MEMBERS.

---

Usage Pattern

A subsystem that creates a sanctioned light:

```
import { createSanctionedPointLight } from './src/core/001_lgt_ThreeLightsOnlyPolicy.js';

const magicLight = createSanctionedPointLight({
  color: [1.0, 0.85, 0.45],
  intensity: 3.5,
  distance: 8,
  decay: 2.0,
  position: [0, 1.2, 0],
  castShadow: false,
});

scene.add(magicLight);
```

A subsystem that attaches a behavior:

```
import {
  attachLightBehavior,
  updateLightBehaviors,
} from './src/core/001_lgt_ThreeLightsOnlyPolicy.js';

attachLightBehavior(magicLight, 'flicker', {
  baseIntensity: 3.5,
  amplitude: 0.15,
  hz: 8.0,
});

// Per frame:
updateLightBehaviors(dt, elapsed);
```

A subsystem that instantiates a composite:

```
import { createCompositeLight } from './src/core/001_lgt_ThreeLightsOnlyPolicy.js';

const campfire = createCompositeLight('fire_light', scene, {
  position: [10, 0, 5],
  ownerId: campSystem.ownerId,
});

// Later, adjust:
campfire.setIntensity(4.0);

// On chunk unload:
campfire.dispose();
```

A subsystem that registers a custom behavior:

```
import { registerLightBehavior } from './src/core/001_lgt_ThreeLightsOnlyPolicy.js';

registerLightBehavior({
  id: 'biome_reactive_color',
  name: 'Biome Reactive Color',
  kinds: [
    SANCTIONED_LIGHT_TYPE.DIRECTIONAL,
    SANCTIONED_LIGHT_TYPE.HEMISPHERE,
  ],
  attach(light, ctx) {
    ctx.baseColor = light.color.clone();
  },
  update(dt, elapsed, light, ctx) {
    const w = getBiomeWeight(0);
    light.color.setRGB(
      ctx.baseColor.r * (1 + w * 0.1),
      ctx.baseColor.g,
      ctx.baseColor.b * (1 - w * 0.1)
    );
  },
});
```

A subsystem that registers a custom composite:

```
import { registerLightComposite } from './src/core/001_lgt_ThreeLightsOnlyPolicy.js';
import { SANCTIONED_LIGHT_TYPE } from './src/core/001_lgt_ThreeLightsOnlyPolicy.js';

registerLightComposite({
  id: 'volcano_glow',
  name: 'Volcano Glow',
  kind: COMPOSITE_LIGHT_KIND.CUSTOM,
  members: [
    {
      type: SANCTIONED_LIGHT_TYPE.POINT,
      name: 'volcano_core',
      color: [1.0, 0.3, 0.05],
      intensity: 8.0,
      distance: 40,
      decay: 1.5,
      castShadow: false,
      behaviorIds: ['flicker'],
    },
    {
      type: SANCTIONED_LIGHT_TYPE.HEMISPHERE,
      name: 'volcano_haze',
      color: [0.6, 0.15, 0.05],
      intensity: 0.4,
    },
  ],
});
```

The policy is what guarantees the lighting stack's determinism. Every light is one of six known types, every behavior is a registered descriptor, every composite is a declared group of sanctioned lights. On Android, where a custom light class may work on one driver and fail on another, this constraint is what allows the engine to run the same shader on every device and produce the same anime look.

The extension systems — behaviors and composites — provide the freedom without breaking the constraint. A new "kind of light" is not a new class; it is a composite of sanctioned lights plus behaviors. The sanctioned set stays fixed, but the expressive range is unbounded.
