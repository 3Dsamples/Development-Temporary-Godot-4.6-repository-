API Documentation — src/ecs/002_lgt_LightComponents.js

File Purpose

This file provides the bitECS 0.4.0 SoA (Structure of Arrays) component definitions and entity factories for the entire light system. It is the ECS-side mirror of the policy module 001_lgt_ThreeLightsOnlyPolicy.js: where the policy module owns the runtime registry of live lights and their behaviors, this module owns the persistent data layout that the ECS pipeline reads and writes each frame.

The distinction matters. The policy module answers questions about a specific light instance ("is this light sanctioned?", "what behaviors are attached?", "should the atlas repack?"). This module answers questions about the population of lights as data ("how many point lights are in the shadow budget?", "what is the total emissive energy of all interior lights?", "which lights are visible in the current frustum?"). Every lighting system downstream — shadow, GI, AO, cluster, environment, interior, exterior, director — iterates these components as flat typed arrays, never as object references.

The file follows the bitECS 0.4.0 architectural redesign:

· No defineComponent calls.
· No Types enum.
· No separate component stores.
· Components are plain JavaScript objects whose fields are typed arrays sized to MAX_ENTITIES.
· They are passed into createWorld({ components }) once at boot.
· addComponent(world, eid, componentObject) attaches a component to an entity.
· Every typed array is allocated once and never resized.

The components declared here are:

1. LightRef — the core per-light state: type, intensity, color, range, spot parameters, shadow flag.
2. LightTransform — position, orientation, and scale for the light anchor.
3. LightShadow — per-light shadow configuration: bias, normal bias, cascade count, atlas tile.
4. LightBehavior — the per-light behavior binding table.
5. LightComposite — the per-light composite membership.
6. LightKind — a coarse classification tag used by schedulers and debug tools.
7. LightBudget — a per-light cost hint used by the budget manager.
8. LightPriority — a per-light render priority.
9. LightState — a per-light state machine: enabled, dirty, indoor, and other flags.

And the entity factories that spawn pre-configured lights:

· spawnSun
· spawnMoon
· spawnHemisphereLight
· spawnAmbientLight
· spawnDirectionalLight
· spawnPointLight
· spawnSpotLight
· spawnRectAreaLight
· spawnFireLight
· spawnNeonLight
· spawnMagicGlow
· spawnInteriorLamp
· spawnCausticLight
· spawnAuroraLight
· spawnWindowShaft

Every factory creates an ECS entity, attaches the six core components, initializes every field to the correct default, and returns the entity id. The factory never instantiates a THREE.Light — that is the job of the bridge module (a future 011_lgt_LightComponentSync.js) which reads the ECS state each frame and pushes it into the actual Three.js light objects.

---

Exported Constants

MAX_ENTITIES

Type: number

Value: 100000

The fixed capacity for every SoA component array in the file. Matches the value used by 008_scn_world.js and 009_scn_BiteCSVersionPolicy.js, so every component array in the entire engine has the same length. Nothing in this file resizes an array after allocation.

LIGHT_TYPE

Type: frozen enum

The type tag written into LightRef.type. Every value is a Uint8 in the range [0, 255].

Values:

· SUN = 0 — the primary directional light, driven by the day cycle.
· MOON = 1 — the secondary directional light, active at night.
· HEMI = 2 — the hemisphere fill light.
· AMBIENT = 3 — the ambient fill light.
· POINT = 4 — a point light.
· SPOT = 5 — a spot light.
· RECT = 6 — a rect area light.
· EMISSIVE = 7 — a non-Three.js light used as an emissive proxy.

This enum is the same set that 001_lgt_ThreeLightsOnlyPolicy.js classifies into, with the addition of SUN and MOON as distinct directional-light subkinds.

LIGHT_KIND

Type: frozen enum

A higher-level classification tag used for scheduling and debug grouping. Written into LightKind.value.

Values:

· UNKNOWN = 0
· SUN_CYCLE = 1 — a sun or moon directional light.
· SKY_FILL = 2 — a hemisphere or ambient fill light.
· POINT_STATIC = 3 — a static point light.
· POINT_DYNAMIC = 4 — a dynamic point light (flicker, pulse, etc.).
· SPOT_STATIC = 5
· SPOT_DYNAMIC = 6
· RECT_AREA = 7
· MAGIC_GLOW = 8 — a magic emissive proxy.
· INTERIOR_LAMP = 9
· WINDOW_SHAFT = 10
· CAUSTIC = 11
· AURORA = 12
· FIRE = 13
· NEON = 14
· COUNT = 15

LIGHT_STATE_FLAG

Type: frozen object of bit flags

Per-light boolean state packed into LightState.flags.

· ENABLED = 1 << 0 — the light is active.
· DIRTY = 1 << 1 — the light's parameters changed this frame and downstream systems must react.
· INDOOR = 1 << 2 — the light is indoors.
· OUTDOOR = 1 << 3 — the light is outdoors.
· VISIBLE = 1 << 4 — the light is currently visible.
· SHADOW_CASTING = 1 << 5 — the light casts shadows.
· FADING_IN = 1 << 6 — the light is fading in.
· FADING_OUT = 1 << 7 — the light is fading out.
· HAS_BEHAVIOR = 1 << 8 — the light has at least one attached behavior.
· HAS_COMPOSITE = 1 << 9 — the light belongs to a composite.
· IS_ANCHOR = 1 << 10 — the light is the anchor of its composite.
· RESERVED_BIT_11 = 1 << 11
· RESERVED_BIT_12 = 1 << 12
· RESERVED_BIT_13 = 1 << 13
· RESERVED_BIT_14 = 1 << 14
· RESERVED_BIT_15 = 1 << 15

The flags are stored in a Uint16Array, so sixteen flags fit exactly.

SHADOW_STATE_FLAG

Type: frozen object of bit flags

Per-light shadow state packed into LightShadow.flags.

· ENABLED = 1 << 0
· ATLAS_ALLOCATED = 1 << 1
· CASCADE_DIRTY = 1 << 2
· LAST_FRAME_DIRTY = 1 << 3
· PANCAKE_FIX = 1 << 4
· PCF_ENABLED = 1 << 5
· PCSS_ENABLED = 1 << 6
· RESERVED_BIT_7 = 1 << 7

BEHAVIOR_FLAG

Type: frozen object of bit flags

Per-behavior-slot flags packed into LightBehavior.flags[i].

· ACTIVE = 1 << 0
· ONCE = 1 << 1
· PAUSED = 1 << 2
· RESERVED_BIT_3 = 1 << 3
· RESERVED_BIT_4 = 1 << 4
· RESERVED_BIT_5 = 1 << 5
· RESERVED_BIT_6 = 1 << 6
· RESERVED_BIT_7 = 1 << 7

MAX_BEHAVIORS_PER_LIGHT_ECS

Type: number

Value: 8

The maximum number of behavior slots per light in the ECS. Matches MAX_BEHAVIORS_PER_LIGHT in 001_lgt_ThreeLightsOnlyPolicy.js.

MAX_COMPOSITE_MEMBERS_ECS

Type: number

Value: 8

The maximum number of composite-member slots per light in the ECS.

MAX_ATLAS_TILES_PER_LIGHT

Type: number

Value: 4

The maximum number of shadow atlas tiles a single light can occupy.

---

Exported Components

Every component is a plain JavaScript object whose fields are typed arrays of length MAX_ENTITIES. They are designed to be passed into createWorld({ components }) exactly as bitECS 0.4.0 expects.

LightRef

The core per-light descriptor.

Fields:

· type — Uint8Array. One of LIGHT_TYPE.
· subType — Uint8Array. Reserved for future expansion.
· intensity — Float32Array. Scalar intensity in the light's native units.
· colorR — Float32Array. Red component in linear [0, 1].
· colorG — Float32Array. Green component.
· colorB — Float32Array. Blue component.
· range — Float32Array. Effect radius for point and spot lights. Zero means infinite.
· spotAngle — Float32Array. Cone angle in radians for spot lights.
· penumbra — Float32Array. Soft-edge factor for spot lights.
· decay — Float32Array. Physical decay exponent for point and spot lights.
· rectWidth — Float32Array. Width of a rect area light.
· rectHeight — Float32Array. Height of a rect area light.
· energy — Float32Array. Total radiant energy, derived from intensity, range, and color. Updated by the budget manager.

LightTransform

The per-light spatial anchor.

Fields:

· x, y, z — Float32Array. World-space position.
· targetX, targetY, targetZ — Float32Array. Target position for directional and spot lights.
· qx, qy, qz, qw — Float32Array. Orientation quaternion.
· sx, sy, sz — Float32Array. Scale, usually 1.
· dirX, dirY, dirZ — Float32Array. Cached unit direction vector, updated by the transform sync system.

LightShadow

The per-light shadow configuration.

Fields:

· enabled — Uint8Array. 1 if the light casts shadows.
· mapSize — Uint16Array. Shadow map size in pixels. Power-of-two.
· cascadeCount — Uint8Array. Number of cascades for directional lights.
· bias — Float32Array. Shadow bias.
· normalBias — Float32Array. Shadow normal bias.
· softness — Float32Array. Softness factor for the PCF kernel.
· near — Float32Array. Near plane for the shadow camera.
· far — Float32Array. Far plane for the shadow camera.
· left, right, top, bottom — Float32Array. Orthographic bounds for the shadow camera.
· atlasTileX — Uint16Array(MAX_ENTITIES * MAX_ATLAS_TILES_PER_LIGHT). Per-light atlas tile X coordinates.
· atlasTileY — Uint16Array(MAX_ENTITIES * MAX_ATLAS_TILES_PER_LIGHT). Per-light atlas tile Y coordinates.
· atlasTileW — Uint16Array(MAX_ENTITIES * MAX_ATLAS_TILES_PER_LIGHT). Per-light atlas tile width.
· atlasTileH — Uint16Array(MAX_ENTITIES * MAX_ATLAS_TILES_PER_LIGHT). Per-light atlas tile height.
· atlasTileCount — Uint8Array. The number of tiles used.
· flags — Uint8Array. A bitmask of SHADOW_STATE_FLAG.
· lastUpdateFrame — Uint32Array. The frame of the last shadow map update.

The atlas tile arrays are the one exception to the "one field per entity" rule. Because a light can span up to four atlas tiles, the tile arrays are sized MAX_ENTITIES * MAX_ATLAS_TILES_PER_LIGHT and indexed as [eid * MAX_ATLAS_TILES_PER_LIGHT + tileIdx].

LightBehavior

The per-light behavior binding table.

Fields:

· count — Uint8Array. The number of active behaviors.
· id — Uint8Array(MAX_ENTITIES * MAX_BEHAVIORS_PER_LIGHT_ECS). Per-slot behavior registry id. 0 means empty.
· flags — Uint8Array(MAX_ENTITIES * MAX_BEHAVIORS_PER_LIGHT_ECS). Per-slot flags from BEHAVIOR_FLAG.
· phase — Float32Array(MAX_ENTITIES * MAX_BEHAVIORS_PER_LIGHT_ECS). Per-slot phase offset for time-varying behaviors.
· amplitude — Float32Array(MAX_ENTITIES * MAX_BEHAVIORS_PER_LIGHT_ECS). Per-slot amplitude.
· frequency — Float32Array(MAX_ENTITIES * MAX_BEHAVIORS_PER_LIGHT_ECS). Per-slot frequency in Hz.
· baseIntensity — Float32Array(MAX_ENTITIES * MAX_BEHAVIORS_PER_LIGHT_ECS). Per-slot base intensity to modulate.

The per-slot arrays are indexed as [eid * MAX_BEHAVIORS_PER_LIGHT_ECS + slot].

LightComposite

The per-light composite membership.

Fields:

· compositeId — Uint8Array. The composite registry id. 0 means none.
· memberIndex — Uint8Array. The member's index within its composite.
· anchorEid — Int32Array. The entity id of the composite's anchor. -1 if this entity is not in a composite.
· memberCount — Uint8Array. The number of members in the composite.
· members — Int32Array(MAX_ENTITIES * MAX_COMPOSITE_MEMBERS_ECS). Per-composite member entity ids. Indexed as [anchorEid * MAX_COMPOSITE_MEMBERS_ECS + slot].

LightKind

A coarse classification tag used for scheduling.

Fields:

· value — Uint8Array. One of LIGHT_KIND.
· schedulerDomain — Uint8Array. The index of the FrameScheduler domain that should update this light. Defaults to DOMAIN.LIGHTS (1).
· updateFrequency — Float32Array. How often the light should be updated in Hz.

LightBudget

A per-light cost hint used by the budget manager.

Fields:

· baseCost — Float32Array. The baseline rendering cost in arbitrary units.
· shadowCost — Float32Array. Additional cost when the light casts shadows.
· currentCost — Float32Array. The current frame's estimated cost.
· priority — Uint8Array. A numeric priority in [0, 255]. Higher is more important.
· budgetClass — Uint8Array. One of 0 (always on), 1 (high priority), 2 (normal), 3 (low).

LightPriority

The per-light render priority. Kept separate from LightBudget because the priority may change independently of the cost.

Fields:

· value — Uint8Array. The current priority in [0, 255].
· override — Uint8Array. 1 if the priority has been manually overridden.

LightState

The per-light state machine and status flags.

Fields:

· flags — Uint16Array. A bitmask of LIGHT_STATE_FLAG.
· lastUpdateFrame — Uint32Array. The frame of the last update.
· createdAtMs — Float64Array. Creation timestamp.
· ageMs — Float32Array. Runtime age in milliseconds.
· fadeT — Float32Array. Fade state in [0, 1].
· fadeTarget — Float32Array. The target fade value.
· visibility — Float32Array. Combined visibility in [0, 1], computed from frustum culling and fade.
· distanceToCamera — Float32Array. The cached distance to the active camera.
· screenRatio — Float32Array. The cached screen-space size ratio.

---

Exported Functions

_initCommonFields(world, eid)

Internal. Initializes every field of every component to a default value for the given entity. Called by every spawn factory.

Sets:

· LightRef defaults: type POINT, intensity 1, color white, range 0, spotAngle π/4, penumbra 0.1, decay 2, rect 1×1, energy 1.
· LightTransform defaults: origin, identity quaternion, scale 1.
· LightShadow defaults: disabled, mapSize 1024, cascadeCount 1, bias -0.0008, normalBias 0.020, softness 0.05, near 0.5, far 200, ortho bounds ±50.
· LightBehavior defaults: count 0, all slots cleared.
· LightComposite defaults: compositeId 0, memberIndex 0, anchorEid -1, memberCount 0.
· LightKind defaults: POINT_STATIC, domain LIGHTS, updateFrequency 60.
· LightBudget defaults: baseCost 1, shadowCost 0, currentCost 0, priority 128, budgetClass 2.
· LightPriority defaults: 128, override 0.
· LightState defaults: flags ENABLED | VISIBLE, created at now, age 0, fade 1, fadeTarget 1, visibility 1, distance 0, screenRatio 0.

spawnSun(world, options = {})

Parameters:

· world — the bitECS world handle.
· options.color — an optional [r, g, b] linear color. Default [1.0, 0.96, 0.85].
· options.intensity — default 1.25.
· options.position — default [50, 80, 30].
· options.target — default [0, 0, 0].
· options.castShadow — default true.
· options.shadowMapSize — default 2048.
· options.cascadeCount — default 4.

Returns: the entity id.

Purpose: spawns a directional light configured as the primary sun. Sets LightRef.type to SUN, LightKind.value to SUN_CYCLE, and marks the shadow state with ENABLED and CASCADE_DIRTY. Sets the LightShadow map size and cascade count.

spawnMoon(world, options = {})

Parameters: same shape as spawnSun, but with defaults tuned for night:

· color — [0.42, 0.48, 0.70].
· intensity — 0.35.
· position — [-40, 60, -30].
· castShadow — true.
· shadowMapSize — 1024.
· cascadeCount — 2.

Returns: the entity id.

Purpose: spawns the moon directional light. LightRef.type is MOON, LightKind.value is SUN_CYCLE.

spawnHemisphereLight(world, options = {})

Parameters:

· skyColor — default [0.45, 0.62, 0.85].
· groundColor — default [0.18, 0.22, 0.26].
· intensity — default 0.35.
· position — default [0, 50, 0].

Returns: the entity id.

Purpose: spawns a hemisphere fill light. LightRef.type is HEMI, LightKind.value is SKY_FILL.

spawnAmbientLight(world, options = {})

Parameters:

· color — default [0.20, 0.25, 0.30].
· intensity — default 0.15.

Returns: the entity id.

Purpose: spawns an ambient fill light. LightRef.type is AMBIENT, LightKind.value is SKY_FILL.

spawnDirectionalLight(world, options = {})

Parameters: full directional spec:

· color, intensity, position, target, castShadow, shadowMapSize, cascadeCount, bias, normalBias.

Returns: the entity id.

Purpose: generic directional light. Used when the caller needs a directional light that is not the sun or the moon. LightKind.value defaults to SUN_CYCLE unless options.kind overrides it.

spawnPointLight(world, options = {})

Parameters:

· color — default white.
· intensity — default 1.
· distance — default 0 (infinite).
· decay — default 2.
· position — default [0, 1, 0].
· castShadow — default false.
· shadowMapSize — default 512.
· kind — optional LIGHT_KIND. Default POINT_STATIC.

Returns: the entity id.

Purpose: generic point light. LightRef.type is POINT.

spawnSpotLight(world, options = {})

Parameters:

· color — default white.
· intensity — default 1.
· distance — default 0.
· decay — default 2.
· angle — default Math.PI / 4.
· penumbra — default 0.1.
· position, target, castShadow, shadowMapSize.

Returns: the entity id.

Purpose: generic spot light. LightRef.type is SPOT.

spawnRectAreaLight(world, options = {})

Parameters:

· color — default white.
· intensity — default 1.
· width, height — default 1 each.
· position, target.

Returns: the entity id.

Purpose: generic rect area light. LightRef.type is RECT, LightKind.value is RECT_AREA.

spawnFireLight(world, options = {})

Parameters:

· color — default [1.0, 0.55, 0.15].
· intensity — default 2.5.
· distance — default 12.
· position — default [0, 0.8, 0].
· castShadow — default true.
· flickerAmplitude — default 0.15.
· flickerHz — default 8.

Returns: the entity id.

Purpose: the canonical campfire light. Spawns a point light, attaches the flicker behavior into slot 0 of LightBehavior, and sets LightKind.value to FIRE. The behavior phase is randomized on spawn so multiple campfires flicker out of sync.

spawnNeonLight(world, options = {})

Parameters:

· color — default [0.9, 0.4, 0.7].
· intensity — default 3.0.
· width — default 1.2.
· height — default 0.2.
· pulseAmplitude — default 0.25.
· pulseHz — default 1.5.

Returns: the entity id.

Purpose: the canonical neon sign. Spawns a rect area light, attaches the pulse behavior, and sets LightKind.value to NEON.

spawnMagicGlow(world, options = {})

Parameters:

· color — default [1.0, 0.85, 0.45].
· intensity — default 3.5.
· distance — default 8.
· position — default [0, 1.2, 0].
· flickerAmplitude — default 0.20.
· flickerHz — default 6.

Returns: the entity id.

Purpose: the canonical magic emissive. Spawns a point light, attaches the flicker behavior, and sets LightKind.value to MAGIC_GLOW.

spawnInteriorLamp(world, options = {})

Parameters:

· color — default [1.0, 0.85, 0.65].
· intensity — default 1.8.
· distance — default 6.
· castShadow — default true.
· driftAmount — default 0.05.
· driftHz — default 0.2.

Returns: the entity id.

Purpose: the canonical interior lamp. Spawns a point light, attaches the temperature_drift behavior, and sets LightKind.value to INTERIOR_LAMP. Marks the INDOOR flag in LightState.

spawnCausticLight(world, options = {})

Parameters:

· color — default [0.6, 0.95, 1.0].
· intensity — default 0.8.
· distance — default 4.
· pulseAmplitude — default 0.30.
· pulseHz — default 2.0.

Returns: the entity id.

Purpose: the canonical water caustic proxy. Spawns a point light with a pulse behavior, sets LightKind.value to CAUSTIC.

spawnAuroraLight(world, options = {})

Parameters:

· color — default [0.35, 0.95, 0.75].
· intensity — default 0.45.

Returns: the entity id.

Purpose: the canonical aurora hemisphere proxy. LightKind.value is AURORA. LightRef.type is HEMI.

spawnWindowShaft(world, options = {})

Parameters:

· color — default [1.0, 0.94, 0.82].
· intensity — default 1.2.
· position, target, castShadow — default true.

Returns: the entity id.

Purpose: the canonical interior window shaft. Spawns a directional light and sets LightKind.value to WINDOW_SHAFT. Marks the INDOOR flag.

spawnCompositeMembers(world, anchorEid, memberEids)

Parameters:

· world — the bitECS world handle.
· anchorEid — the entity id of the composite's anchor.
· memberEids — an array of member entity ids.

Returns: nothing.

Purpose: links a set of member entities to their anchor. Sets LightComposite.memberCount[anchorEid], writes each member's entity id into LightComposite.members[anchorEid * MAX_COMPOSITE_MEMBERS_ECS + i], and for each member sets LightComposite.anchorEid, compositeId, and memberIndex. Marks the HAS_COMPOSITE flag on all involved entities and IS_ANCHOR on the anchor.

setLightPosition(world, eid, x, y, z)

Parameters:

· world — the bitECS world handle.
· eid — the light's entity id.
· x, y, z — the new position.

Returns: nothing.

Purpose: convenience setter for LightTransform.x, .y, .z. Marks the DIRTY flag on LightState.flags.

setLightColor(world, eid, r, g, b)

Parameters:

· world — the bitECS world handle.
· eid — the light's entity id.
· r, g, b — the new color components in linear [0, 1].

Returns: nothing.

Purpose: convenience setter for LightRef.colorR, .colorG, .colorB. Marks the DIRTY flag.

setLightIntensity(world, eid, intensity)

Parameters:

· world — the bitECS world handle.
· eid — the light's entity id.
· intensity — the new intensity.

Returns: nothing.

Purpose: convenience setter for LightRef.intensity. Marks the DIRTY flag.

markLightDirty(world, eid)

Parameters:

· world — the bitECS world handle.
· eid — the light's entity id.

Returns: nothing.

Purpose: sets the DIRTY flag on LightState.flags without changing any parameter.

clearLightDirty(world, eid)

Parameters: same as markLightDirty.

Returns: nothing.

Purpose: clears the DIRTY flag.

isLightEnabled(world, eid)

Parameters: same as markLightDirty.

Returns: boolean.

Purpose: reads the ENABLED flag.

setLightEnabled(world, eid, enabled)

Parameters:

· world — the bitECS world handle.
· eid — the light's entity id.
· enabled — boolean.

Returns: nothing.

Purpose: sets or clears the ENABLED flag.

getLightsByType(world, type, outEids)

Parameters:

· world — the bitECS world handle.
· type — one of LIGHT_TYPE.
· outEids — an array to receive matching entity ids.

Returns: the number of matches.

Purpose: iterates every live light entity and collects those whose LightRef.type matches. Used by the shadow system to find shadow-casting directional lights, by the cluster system to find point lights, and by the debug HUD to count lights per type.

sumLightEnergy(world, type)

Parameters:

· world — the bitECS world handle.
· type — one of LIGHT_TYPE, or -1 for all lights.

Returns: the summed LightRef.energy across matching lights.

Purpose: a single-pass aggregate used by the budget manager.

computeLightCost(world, eid)

Parameters:

· world — the bitECS world handle.
· eid — the light's entity id.

Returns: the computed cost.

Purpose: updates and returns LightBudget.currentCost from the light's base cost, shadow cost, and enabled flags. Called by the budget manager each frame.

disableAllLights(world)

Parameters: world — the bitECS world handle.

Returns: the number of lights disabled.

Purpose: iterates every live light and clears the ENABLED flag. Used by the interior-to-exterior transition system.

enableAllLights(world)

Parameters: world — the bitECS world handle.

Returns: the number of lights re-enabled.

Purpose: iterates every live light and sets the ENABLED flag.

---

Exported Helper Functions

getBehaviorSlot(world, eid, slot)

Parameters:

· world — the bitECS world handle.
· eid — the light's entity id.
· slot — the behavior slot index in [0, MAX_BEHAVIORS_PER_LIGHT_ECS).

Returns: an object with id, flags, phase, amplitude, frequency, and baseIntensity, read from the per-slot arrays at the correct offset.

Purpose: a convenience reader that encapsulates the [eid * MAX_BEHAVIORS_PER_LIGHT_ECS + slot] indexing.

setBehaviorSlot(world, eid, slot, spec)

Parameters:

· world — the bitECS world handle.
· eid — the light's entity id.
· slot — the behavior slot index.
· spec — an object with optional id, flags, phase, amplitude, frequency, baseIntensity.

Returns: boolean. True on success.

Purpose: a convenience writer that encapsulates the indexing and marks the light's HAS_BEHAVIOR flag.

clearBehaviorSlot(world, eid, slot)

Parameters: same as getBehaviorSlot.

Returns: boolean.

Purpose: clears the behavior slot's id and flags.

getCompositeMembers(world, anchorEid)

Parameters:

· world — the bitECS world handle.
· anchorEid — the composite anchor's entity id.

Returns: an array of member entity ids.

Purpose: reads LightComposite.members for the given anchor and returns a sliced array of the valid members.

---

Exported Aggregate Functions

getLightStats(world)

Parameters: world — the bitECS world handle.

Returns: an object with:

· total — the total number of live lights.
· byType — an array of counts indexed by LIGHT_TYPE.
· byKind — an array of counts indexed by LIGHT_KIND.
· enabled — the number of enabled lights.
· shadowCasting — the number of lights casting shadows.
· indoor — the number of indoor lights.
· withBehavior — the number of lights with at least one behavior.
· withComposite — the number of lights in a composite.
· totalEnergy — the sum of LightRef.energy across all lights.

Purpose: the debug HUD's primary view of the light population.

getLightSnapshotForStats(world, eid)

Parameters:

· world — the bitECS world handle.
· eid — the light's entity id.

Returns: a plain object with the light's type, kind, intensity, color, position, shadow state, and enabled flag.

Purpose: the stats collector's per-light snapshot. Registered as a named source in 025_rnd_StatsCollector.js.

---

Exported Default Object

The default export bundles every component, every factory, every setter, and every helper:

· The nine components: LightRef, LightTransform, LightShadow, LightBehavior, LightComposite, LightKind, LightBudget, LightPriority, LightState.
· The enums: LIGHT_TYPE, LIGHT_KIND, LIGHT_STATE_FLAG, SHADOW_STATE_FLAG, BEHAVIOR_FLAG.
· The constants: MAX_ENTITIES, MAX_BEHAVIORS_PER_LIGHT_ECS, MAX_COMPOSITE_MEMBERS_ECS, MAX_ATLAS_TILES_PER_LIGHT.
· The fifteen spawn factories.
· The setters and readers.
· The aggregate functions.

---

Usage Pattern

A subsystem that spawns a light and reads its state:

```
import {
  spawnFireLight,
  setLightPosition,
  markLightDirty,
  getLightStats,
} from './src/ecs/002_lgt_LightComponents.js';

const campfireEid = spawnFireLight(world, {
  color: [1.0, 0.55, 0.15],
  intensity: 2.5,
  distance: 12,
  position: [10, 0.8, 5],
  castShadow: true,
  flickerAmplitude: 0.15,
  flickerHz: 8.0,
});

// The light is live in the ECS but no THREE.Light exists yet.
// The sync system (011_lgt_LightComponentSync.js) will create
// the actual THREE.PointLight on the next frame.
```

A subsystem that iterates lights by type:

```
import {
  getLightsByType,
  LIGHT_TYPE,
} from './src/ecs/002_lgt_LightComponents.js';

const shadowCasters = [];
const n = getLightsByType(world, LIGHT_TYPE.SUN, shadowCasters);
for (let i = 0; i < n; i++) {
  const eid = shadowCasters[i];
  // read LightShadow.* arrays for this eid
}
```

A subsystem that reads the aggregate stats for the debug HUD:

```
import { getLightStats } from './src/ecs/002_lgt_LightComponents.js';

const stats = getLightStats(world);
console.log(`Lights: ${stats.total} (${stats.enabled} enabled)`);
console.log(`Point lights: ${stats.byType[LIGHT_TYPE.POINT]}`);
console.log(`Shadow casters: ${stats.shadowCasting}`);
```

The component definitions are the canonical data layout for the light system. Every lighting subsystem that needs to read or write light state does so through these typed arrays. Because the layout is fixed and shared, there is exactly one source of truth for each light's position, color, intensity, shadow configuration, behavior bindings, and state flags. The factories provide the sanctioned creation path, the setters provide the sanctioned mutation path, and the aggregate functions provide the sanctioned summary path.

