API Documentation — src/ecs/003_lgt_ShadowComponents.js

File Purpose

This file provides the bitECS 0.4.0 SoA component definitions and entity factories for the entire shadow system. It is the ECS-side data layout for every shadow-casting light in the anime lighting stack, covering directional cascades, atlas tiles, bias parameters, filter selection, temporal history, and per-light shadow state.

The problem it solves is that shadow state is large enough that storing it as JavaScript objects is prohibitively expensive on Android. A single directional light with four cascades needs four shadow camera matrices, four viewport rectangles, per-cascade near and far planes, per-cascade texel snap offsets, and per-cascade hysteresis counters. Multiplied by the number of lights that cast shadows, this becomes hundreds of float values that must be read every frame by the shadow atlas packer, the cascade stabilizer, the PCF sampler, and the shadow cache invalidator. Storing that data in objects with property access would cost a Map lookup plus a property dereference per value. Storing it in flat typed arrays indexed by entity id costs a single integer offset.

The file follows the bitECS 0.4.0 architectural redesign:

· No defineComponent calls, no Types enum, no separate component stores.
· Components are plain JavaScript objects whose fields are typed arrays sized to MAX_ENTITIES.
· Every typed array is allocated once at module load and never resized.

The components declared here are:

1. ShadowCaster — the core per-light shadow configuration: enabled flag, filter type, atlas tile reference, cascade count.
2. ShadowCamera — the per-light shadow camera parameters: near, far, ortho bounds, world matrix.
3. ShadowCascade — per-cascade state for directional lights: per-cascade camera, split distance, texel snap, hysteresis.
4. ShadowAtlas — per-light atlas allocation state: tile coordinates, tile size, page index, dirty flag.
5. ShadowBias — per-light bias parameters: depth bias, normal bias, slope bias, pancake fix flag, constant bias.
6. ShadowFilter — per-light filter configuration: PCF kernel size, PCSS sample count, softness, rotation angle.
7. ShadowHistory — per-light temporal history: previous frame's matrix, previous frame's bias, accumulators for the stochastic filter.
8. ShadowBudget — per-light shadow budget: cost estimate, priority, whether the shadow is currently allocated.

And the entity factories that spawn pre-configured shadow casters:

· spawnDirectionalShadowCaster
· spawnSunShadowCaster
· spawnMoonShadowCaster
· spawnSpotShadowCaster
· spawnPointShadowCaster
· spawnContactShadowProxy

Every factory creates an ECS entity, attaches the eight components, initializes them to sensible defaults, and returns the entity id. The factory never creates a THREE.Light or a THREE.WebGLRenderTarget — those are created by the sync system that reads the ECS state each frame.

---

Exported Constants

MAX_ENTITIES

Type: number

Value: 100000

The fixed capacity for every SoA component array. Matches the value used across the entire engine so every component array is dimensionally consistent.

SHADOW_TYPE

Type: frozen enum

The kind of shadow a light projects. Distinct from the shadow filter.

Values:

· NONE = 0 — the light does not cast shadows.
· DIRECTIONAL_CASCADE = 1 — a directional light with cascaded shadow maps.
· DIRECTIONAL_SINGLE = 2 — a directional light with a single shadow map.
· SPOT_SINGLE = 3 — a spot light with a single perspective shadow map.
· POINT_CUBE = 4 — a point light with a cube shadow map.
· CONTACT_PROXY = 5 — a screen-space contact shadow proxy (not a real shadow map).
· COUNT = 6

SHADOW_TYPE_NAME

Type: frozen array

Values: ['none', 'directional_cascade', 'directional_single', 'spot_single', 'point_cube', 'contact_proxy'].

SHADOW_FILTER

Type: frozen enum

The filter used to sample the shadow map.

Values:

· BASIC = 0 — single sample, hard edge.
· PCF = 1 — percentage closer filtering with a 3×3 or 5×5 kernel.
· PCF_SOFT = 2 — PCF with a larger kernel and a rotated disk.
· PCSS = 3 — percentage closer soft shadows with a blocker search.
· VSM = 4 — variance shadow maps (desktop only).
· ESM = 5 — exponential shadow maps (desktop only, disabled on mobile).
· COUNT = 6

SHADOW_FILTER_NAME

Type: frozen array

Values: ['basic', 'pcf', 'pcf_soft', 'pcss', 'vsm', 'esm'].

CASCADE_COUNT_MAX

Type: number

Value: 4

The maximum number of cascades per directional light. The engine supports one, two, three, or four cascades.

ATLAS_TILES_MAX

Type: number

Value: 4

The maximum number of atlas tiles a single light can occupy. A directional cascade with four cascades uses four tiles. A single spot light uses one tile. A point cube uses six tiles, but the point cube is reserved for a future expansion and is not currently allocated from the atlas.

SHADOW_STATE_FLAG

Type: frozen object of bit flags

Per-light shadow state flags packed into ShadowCaster.flags.

· ENABLED = 1 << 0
· ATLAS_ALLOCATED = 1 << 1
· ATLAS_DIRTY = 1 << 2
· CAMERA_DIRTY = 1 << 3
· HISTORY_VALID = 1 << 4
· PANCAKE_FIX = 1 << 5
· STABILIZED = 1 << 6
· STATIC_CACHED = 1 << 7
· TEMPORAL_ACCUM = 1 << 8
· RESERVED_BIT_9 = 1 << 9
· RESERVED_BIT_10 = 1 << 10
· RESERVED_BIT_11 = 1 << 11
· RESERVED_BIT_12 = 1 << 12
· RESERVED_BIT_13 = 1 << 13
· RESERVED_BIT_14 = 1 << 14
· RESERVED_BIT_15 = 1 << 15

The flags fit in a Uint16Array, so exactly sixteen states are available.

CASCADE_STATE_FLAG

Type: frozen object of bit flags

Per-cascade state flags packed into ShadowCascade.stateFlags.

· ACTIVE = 1 << 0
· DIRTY = 1 << 1
· STABILIZED = 1 << 2
· SNAPPED = 1 << 3
· HYSTERESIS_HOLD = 1 << 4
· RESERVED_BIT_5 = 1 << 5
· RESERVED_BIT_6 = 1 << 6
· RESERVED_BIT_7 = 1 << 7

DEFAULT_ATLAS_SIZE

Type: number

Value: 2048 on HIGH, 1024 on MEDIUM, 512 on LOW.

The default shadow atlas texture size in pixels. Written into ShadowAtlas.pageSizeX and pageSizeY at spawn.

DEFAULT_ATLAS_PAGES

Type: number

Value: 2 on HIGH, 1 on MEDIUM, 1 on LOW.

The default number of atlas pages. A page is a 2D texture. Multiple pages let the engine allocate shadows beyond a single texture's capacity.

---

Module-Level State (Not Exported Directly)

The module has no module-level mutable state. Every component array is allocated once and exported directly. Every factory reads and writes only those arrays.

---

Exported Components

Every component is a plain JavaScript object whose fields are typed arrays of length MAX_ENTITIES (or MAX_ENTITIES * slots for the per-cascade and per-tile arrays). They are designed to be passed into createWorld({ components }) exactly as bitECS 0.4.0 expects.

ShadowCaster

The core per-light shadow configuration.

Fields:

· type — Uint8Array. One of SHADOW_TYPE.
· filter — Uint8Array. One of SHADOW_FILTER.
· enabled — Uint8Array. 1 if the light casts shadows.
· lightEid — Int32Array. The entity id of the light that owns this caster. -1 if not bound.
· atlasPageIndex — Uint8Array. The atlas page the caster's tiles are allocated from.
· atlasTileCount — Uint8Array. The number of atlas tiles the caster occupies.
· cascadeCount — Uint8Array. The number of cascades. 1 for directional single, 1 for spot, 4 for a four-cascade directional.
· flags — Uint16Array. A bitmask of SHADOW_STATE_FLAG.
· lastUpdateFrame — Uint32Array. The frame of the last shadow map update.
· createdAtMs — Float64Array. Creation timestamp.

ShadowCamera

The per-light shadow camera parameters.

Fields:

· near — Float32Array. Near clip plane.
· far — Float32Array. Far clip plane.
· left, right, top, bottom — Float32Array. Orthographic bounds for directional lights.
· fov — Float32Array. Perspective field of view for spot lights.
· aspect — Float32Array. Perspective aspect ratio for spot lights.
· worldMatrix — Float32Array(MAX_ENTITIES * 16). The shadow camera's world matrix. Indexed as [eid * 16 + i].
· projectionMatrix — Float32Array(MAX_ENTITIES * 16). The shadow camera's projection matrix.
· viewMatrix — Float32Array(MAX_ENTITIES * 16). The shadow camera's view matrix.
· shadowMatrix — Float32Array(MAX_ENTITIES * 16). The combined world-to-shadow-texture matrix used by the shader.
· texelSizeX — Float32Array. The world-space size of one shadow texel along X. Used by the cascade stabilizer.
· texelSizeY — Float32Array. The world-space size of one shadow texel along Y.
· texelWorldX — Float32Array. The snapped world X position used by the texel snapper.
· texelWorldY — Float32Array. The snapped world Y position.
· texelWorldZ — Float32Array. The snapped world Z position.

ShadowCascade

Per-cascade state for directional lights. Because a single directional light can have up to four cascades, all per-cascade arrays are sized MAX_ENTITIES * CASCADE_COUNT_MAX and indexed as [eid * CASCADE_COUNT_MAX + cascadeIndex].

Fields:

· splitNear — Float32Array(MAX_ENTITIES * CASCADE_COUNT_MAX). The near split distance in world units.
· splitFar — Float32Array(MAX_ENTITIES * CASCADE_COUNT_MAX). The far split distance.
· splitRatio — Float32Array(MAX_ENTITIES * CASCADE_COUNT_MAX). The split ratio parameter used by the splitter.
· worldSize — Float32Array(MAX_ENTITIES * CASCADE_COUNT_MAX). The orthographic world-space size of the cascade.
· texelSize — Float32Array(MAX_ENTITIES * CASCADE_COUNT_MAX). The world-space size of one texel in this cascade.
· cameraMatrix — Float32Array(MAX_ENTITIES * CASCADE_COUNT_MAX * 16). The cascade's world matrix.
· atlasTileIndex — Uint8Array(MAX_ENTITIES * CASCADE_COUNT_MAX). The index of the atlas tile this cascade is allocated to, or 255 if not allocated.
· stateFlags — Uint8Array(MAX_ENTITIES * CASCADE_COUNT_MAX). A bitmask of CASCADE_STATE_FLAG.
· hysteresis — Uint8Array(MAX_ENTITIES * CASCADE_COUNT_MAX). A hysteresis counter used by the cascade stabilizer.
· lastResizeFrame — Uint32Array(MAX_ENTITIES * CASCADE_COUNT_MAX). The frame of the last world-size change.

The three-dimensional cameraMatrix array is indexed as [eid * CASCADE_COUNT_MAX * 16 + cascadeIndex * 16 + i].

ShadowAtlas

Per-light atlas allocation state. Tile arrays are sized MAX_ENTITIES * ATLAS_TILES_MAX and indexed as [eid * ATLAS_TILES_MAX + tileIndex].

Fields:

· pageSizeX — Uint16Array. The atlas page width in pixels. Every caster that shares a page uses the same value.
· pageSizeY — Uint16Array. The atlas page height.
· tileX — Uint16Array(MAX_ENTITIES * ATLAS_TILES_MAX). Tile X coordinate in pixels within the page.
· tileY — Uint16Array(MAX_ENTITIES * ATLAS_TILES_MAX). Tile Y coordinate.
· tileW — Uint16Array(MAX_ENTITIES * ATLAS_TILES_MAX). Tile width in pixels.
· tileH — Uint16Array(MAX_ENTITIES * ATLAS_TILES_MAX). Tile height in pixels.
· tileUVX — Float32Array(MAX_ENTITIES * ATLAS_TILES_MAX). The normalized U coordinate of the tile's top-left corner.
· tileUVY — Float32Array(MAX_ENTITIES * ATLAS_TILES_MAX). The normalized V coordinate.
· tileUVW — Float32Array(MAX_ENTITIES * ATLAS_TILES_MAX). The normalized width.
· tileUVH — Float32Array(MAX_ENTITIES * ATLAS_TILES_MAX). The normalized height.
· lastAllocFrame — Uint32Array(MAX_ENTITIES * ATLAS_TILES_MAX). The frame when the tile was allocated.
· lastUseFrame — Uint32Array(MAX_ENTITIES * ATLAS_TILES_MAX). The frame when the tile was last rendered.

ShadowBias

The per-light bias parameters. These match the fields the PCF and PCSS shaders read.

Fields:

· depthBias — Float32Array. The depth bias applied to the shadow comparison.
· normalBias — Float32Array. The normal-direction bias applied at sample time.
· slopeBias — Float32Array. The slope-scaled bias.
· constantBias — Float32Array. A constant additive bias.
· pancakeFix — Float32Array. The pancake fix intensity. Used by Mali and Adreno GPU families.
· minBias — Float32Array. The minimum bias. Never go below this even with slope scaling.
· biasScale — Float32Array. A global multiplier on the bias.

ShadowFilter

The per-light filter configuration.

Fields:

· kernelSize — Uint8Array. The PCF kernel size. Typically 3 or 5.
· pcssSampleCount — Uint8Array. The PCSS blocker search and sample count.
· softness — Float32Array. The filter softness in [0, 1].
· rotationAngle — Float32Array. The disk rotation angle applied to the PCF kernel.
· penumbraScale — Float32Array. The scale applied to the penumbra in PCSS.
· minPenumbra — Float32Array. The minimum penumbra radius.
· maxPenumbra — Float32Array. The maximum penumbra radius.

ShadowHistory

The per-light temporal history.

Fields:

· prevMatrix — Float32Array(MAX_ENTITIES * 16). The previous frame's shadow matrix.
· prevBias — Float32Array. The previous frame's bias.
· prevSoftness — Float32Array. The previous frame's softness.
· valid — Uint8Array. 1 if the history is valid, 0 otherwise.
· accumFrames — Uint8Array. The number of consecutive frames the history has been valid for.
· stabilityScore — Float32Array. A rolling stability score. When the score drops below a threshold the history is reset.
· lastResetFrame — Uint32Array. The frame of the last reset.

ShadowBudget

The per-light shadow budget.

Fields:

· costEstimate — Float32Array. The estimated per-frame cost of rendering this shadow.
· priority — Uint8Array. The priority in [0, 255]. Higher is more important.
· budgetClass — Uint8Array. One of 0 (always on), 1 (high), 2 (normal), 3 (low).
· reserved — Uint8Array. 1 if the shadow is currently allocated.
· lastCostFrame — Uint32Array. The frame of the last cost estimate.

---

Exported Functions

_initCommonFields(world, eid)

Internal. Initializes every field of every shadow component for the given entity to sensible defaults.

Sets:

· ShadowCaster defaults: type NONE, filter PCF, enabled 0, lightEid -1, atlasPageIndex 0, atlasTileCount 0, cascadeCount 1, flags 0.
· ShadowCamera defaults: near 0.5, far 200, ortho bounds ±50, fov π/4, aspect 1, all matrices identity, texel size 0.
· ShadowCascade defaults: splitNear 0.5, splitFar 50, splitRatio 0.5, worldSize 50, texel size 0, all camera matrices identity, atlas tile index 255, state flags 0, hysteresis 0.
· ShadowAtlas defaults: page size from DEFAULT_ATLAS_SIZE, all tile coordinates 0, all UVs 0.
· ShadowBias defaults: depth -0.0008, normal 0.020, slope 0, constant 0, pancake 0, min -0.002, scale 1.
· ShadowFilter defaults: kernel 3, PCSS samples 8, softness 0.05, rotation 0, penumbra scale 1, min 0, max 0.05.
· ShadowHistory defaults: identity matrix, bias -0.0008, softness 0.05, valid 0, accum 0, stability 0, last reset 0.
· ShadowBudget defaults: cost 1, priority 128, class 2, reserved 0, last cost frame 0.

spawnDirectionalShadowCaster(world, options = {})

Parameters:

· world — the bitECS world handle.
· options.lightEid — the light entity that owns this caster.
· options.cascadeCount — the number of cascades. Default 1.
· options.filter — one of SHADOW_FILTER. Default PCF.
· options.mapSize — the shadow map size. Default DEFAULT_ATLAS_SIZE.
· options.bias — the shadow bias. Default -0.0008.
· options.normalBias — the normal bias. Default 0.020.
· options.pancakeFix — whether to enable the pancake fix. Default false.
· options.budgetClass — the budget class. Default 1.
· options.priority — the priority. Default 200.

Returns: the entity id.

Purpose: the canonical directional shadow caster. Sets ShadowCaster.type to DIRECTIONAL_CASCADE for multiple cascades or DIRECTIONAL_SINGLE for one. Populates the camera near/far, orthographic bounds, cascade split parameters, and atlas page size. Initializes the atlas tile count to match the cascade count. Marks the caster with ENABLED and CAMERA_DIRTY.

spawnSunShadowCaster(world, options = {})

Parameters: same as spawnDirectionalShadowCaster, with sun-specific defaults:

· cascadeCount — 4.
· mapSize — 2048.
· bias — -0.0008.
· normalBias — 0.020.
· pancakeFix — true on Android, false on desktop.
· priority — 255 (the sun's shadow is the most important).

Returns: the entity id.

spawnMoonShadowCaster(world, options = {})

Parameters: same shape, with moon-specific defaults:

· cascadeCount — 2.
· mapSize — 1024.
· priority — 200.

Returns: the entity id.

spawnSpotShadowCaster(world, options = {})

Parameters:

· world — the bitECS world handle.
· options.lightEid — the light entity that owns this caster.
· options.filter — one of SHADOW_FILTER. Default PCF_SOFT.
· options.mapSize — the shadow map size. Default 512.
· options.fov — the spot cone angle. Default π/4.
· options.near, options.far — the shadow camera planes.
· options.bias, options.normalBias.
· options.pancakeFix.
· options.priority — default 150.

Returns: the entity id.

Purpose: canonical spot shadow caster. Sets ShadowCaster.type to SPOT_SINGLE. Populates the perspective camera parameters. Allocates one atlas tile.

spawnPointShadowCaster(world, options = {})

Parameters: similar to spawnSpotShadowCaster, with point-light defaults:

· mapSize — 256 per cube face.
· priority — default 100.

Returns: the entity id.

Purpose: reserved for future point cube shadow support. Sets ShadowCaster.type to POINT_CUBE. The current engine version does not allocate cube shadow pages, so the caster is initialized but not rendered until the cube path is enabled.

spawnContactShadowProxy(world, options = {})

Parameters:

· world — the bitECS world handle.
· options.lightEid — the light that owns the proxy.
· options.radius — the screen-space radius of the contact shadow. Default 0.02.
· options.thickness — the depth thickness. Default 0.05.
· options.priority — default 50.

Returns: the entity id.

Purpose: the canonical screen-space contact shadow proxy. Sets ShadowCaster.type to CONTACT_PROXY. No shadow map is allocated. Used for small, high-frequency shadows near geometry, where the contact shadow algorithm is cheaper and sharper than a real shadow map.

setCasterEnabled(world, eid, enabled)

Parameters:

· world — the bitECS world handle.
· eid — the shadow caster's entity id.
· enabled — boolean.

Returns: nothing.

Purpose: sets or clears the ENABLED flag on ShadowCaster.flags and updates the enabled field.

setCasterFilter(world, eid, filter)

Parameters:

· world — the bitECS world handle.
· eid — the shadow caster's entity id.
· filter — one of SHADOW_FILTER.

Returns: nothing.

Purpose: updates the filter and marks the caster dirty.

setCasterBias(world, eid, depthBias, normalBias, slopeBias = 0, constantBias = 0)

Parameters: the four bias values.

Returns: nothing.

Purpose: updates the bias parameters and marks ShadowCaster.flags with CAMERA_DIRTY.

setCasterCascadeCount(world, eid, count)

Parameters:

· world — the bitECS world handle.
· eid — the shadow caster's entity id.
· count — the new cascade count in [1, CASCADE_COUNT_MAX].

Returns: nothing.

Purpose: updates the cascade count and re-initializes the per-cascade state for the new count. Marks the atlas as dirty.

setCascadeSplit(world, eid, cascadeIndex, splitNear, splitFar)

Parameters:

· world — the bitECS world handle.
· eid — the shadow caster's entity id.
· cascadeIndex — the cascade index in [0, CASCADE_COUNT_MAX).
· splitNear, splitFar — the split distances.

Returns: nothing.

Purpose: updates the split for a specific cascade. Marks the cascade dirty and the caster's camera dirty.

allocateAtlasTile(world, eid, tileIndex, pageIndex, x, y, w, h)

Parameters:

· world — the bitECS world handle.
· eid — the shadow caster's entity id.
· tileIndex — the tile index in [0, ATLAS_TILES_MAX).
· pageIndex — the atlas page index.
· x, y, w, h — the pixel coordinates within the page.

Returns: nothing.

Purpose: writes the tile's pixel coordinates and the corresponding normalized UV coordinates into the atlas component, and marks the caster with ATLAS_ALLOCATED and ATLAS_DIRTY. The UVs are computed as x / pageSizeX, y / pageSizeY, w / pageSizeX, h / pageSizeY.

freeAtlasTile(world, eid, tileIndex)

Parameters:

· world — the bitECS world handle.
· eid — the shadow caster's entity id.
· tileIndex — the tile index.

Returns: nothing.

Purpose: clears the tile coordinates and UVs, and decrements atlasTileCount. When the count reaches zero, clears the ATLAS_ALLOCATED flag.

markCasterDirty(world, eid)

Parameters: the caster's entity id.

Returns: nothing.

Purpose: sets the CAMERA_DIRTY flag on ShadowCaster.flags.

isCasterEnabled(world, eid)

Parameters: the caster's entity id.

Returns: boolean.

Purpose: reads the ENABLED flag and the enabled field.

updateCasterMatrix(world, eid, shadowMatrixElements)

Parameters:

· world — the bitECS world handle.
· eid — the shadow caster's entity id.
· shadowMatrixElements — a 16-element array containing the world-to-shadow-texture matrix.

Returns: nothing.

Purpose: writes the matrix into ShadowCamera.shadowMatrix and copies the current matrix into ShadowHistory.prevMatrix if the caster's history is not already valid.

updateCascadeMatrix(world, eid, cascadeIndex, matrixElements)

Parameters:

· world — the bitECS world handle.
· eid — the shadow caster's entity id.
· cascadeIndex — the cascade index.
· matrixElements — a 16-element array.

Returns: nothing.

Purpose: writes the cascade's world matrix into ShadowCascade.cameraMatrix at the correct offset.

getShadowCasters(world, outEids)

Parameters:

· world — the bitECS world handle.
· outEids — an array to receive matching entity ids.

Returns: the number of casters collected.

Purpose: iterates every live shadow caster entity and collects those with the ENABLED flag. Used by the shadow atlas packer to build the render list.

getShadowCastersByType(world, type, outEids)

Parameters:

· world — the bitECS world handle.
· type — one of SHADOW_TYPE.
· outEids — an array to receive matching entity ids.

Returns: the number of matches.

Purpose: type-filtered variant of getShadowCasters.

sumShadowCost(world)

Parameters: world — the bitECS world handle.

Returns: the sum of ShadowBudget.costEstimate across all enabled casters.

Purpose: the shadow budget manager's aggregate estimate.

computeCasterCost(world, eid)

Parameters:

· world — the bitECS world handle.
· eid — the shadow caster's entity id.

Returns: the computed cost.

Purpose: updates ShadowBudget.costEstimate from the caster's map size, cascade count, filter, and priority. Formula: mapSize * mapSize * cascadeCount * filterMultiplier, scaled by the atlas page count. Filter multipliers: BASIC 1, PCF 1.5, PCF_SOFT 2, PCSS 3, VSM 1.2, ESM 1.5.

getShadowStats(world)

Parameters: world — the bitECS world handle.

Returns: an object with:

· total — the total number of shadow casters.
· enabled — the number of enabled casters.
· byType — an array of counts indexed by SHADOW_TYPE.
· byFilter — an array of counts indexed by SHADOW_FILTER.
· totalCascades — the sum of cascade counts across all casters.
· totalTiles — the sum of atlas tile counts across all casters.
· totalCost — the sum of ShadowBudget.costEstimate across all casters.
· atlasPages — the maximum page count across all casters.

Purpose: the debug HUD's primary view of the shadow population. Registered as a named source in 025_rnd_StatsCollector.js.

getShadowSnapshotForStats(world, eid)

Parameters:

· world — the bitECS world handle.
· eid — the shadow caster's entity id.

Returns: a plain object with the caster's type, filter, bias, cascade count, atlas tile count, and enabled flag.

Purpose: the per-caster stats snapshot.

clearAllShadowHistory(world)

Parameters: world — the bitECS world handle.

Returns: the number of casters whose history was cleared.

Purpose: sets ShadowHistory.valid to 0 for every caster. Called on context restore, tier change, or quality downgrade to force the temporal filter to restart.

---

Exported Default Object

The default export bundles every component, every factory, every setter, every getter, and every aggregate function:

· The eight components: ShadowCaster, ShadowCamera, ShadowCascade, ShadowAtlas, ShadowBias, ShadowFilter, ShadowHistory, ShadowBudget.
· The enums: SHADOW_TYPE, SHADOW_FILTER, SHADOW_STATE_FLAG, CASCADE_STATE_FLAG.
· The constants: MAX_ENTITIES, CASCADE_COUNT_MAX, ATLAS_TILES_MAX, DEFAULT_ATLAS_SIZE, DEFAULT_ATLAS_PAGES.
· The six spawn factories.
· The setters and readers.
· The aggregate functions.

---

Usage Pattern

A subsystem that spawns a sun shadow caster and reads its state:

```
import {
  spawnSunShadowCaster,
  getShadowCasters,
  computeCasterCost,
  getShadowStats,
} from './src/ecs/003_lgt_ShadowComponents.js';

const sunCasterEid = spawnSunShadowCaster(world, {
  lightEid: sunLightEid,
  cascadeCount: 4,
  mapSize: 2048,
  bias: -0.0008,
  normalBias: 0.020,
  pancakeFix: true,
  priority: 255,
});

// Each frame, the shadow system iterates the enabled casters:
const casters = [];
const n = getShadowCasters(world, casters);
for (let i = 0; i < n; i++) {
  const eid = casters[i];
  computeCasterCost(world, eid);
  // render the shadow, write the matrix, allocate the atlas tile
}
```

A subsystem that reads aggregate shadow stats:

```
import { getShadowStats } from './src/ecs/003_lgt_ShadowComponents.js';

const stats = getShadowStats(world);
console.log(`Shadow casters: ${stats.enabled}/${stats.total}`);
console.log(`Total cascades: ${stats.totalCascades}`);
console.log(`Atlas tiles used: ${stats.totalTiles}`);
```

A subsystem that updates a cascade split at runtime:

```
import {
  setCascadeSplit,
  markCasterDirty,
} from './src/ecs/003_lgt_ShadowComponents.js';

setCascadeSplit(world, sunCasterEid, 0, 0.5, 20.0);
setCascadeSplit(world, sunCasterEid, 1, 20.0, 60.0);
setCascadeSplit(world, sunCasterEid, 2, 60.0, 120.0);
setCascadeSplit(world, sunCasterEid, 3, 120.0, 200.0);
markCasterDirty(world, sunCasterEid);
```

The shadow component definitions are the canonical data layout for the shadow system. Every subsystem that needs to read or write shadow state — the atlas packer, the cascade splitter, the texel snapper, the bias controller, the PCF and PCSS samplers, the temporal accumulator, the budget manager — reads from these typed arrays. Because the layout is fixed and shared, there is exactly one source of truth for each shadow caster's camera, bias, filter, cascades, atlas allocation, and temporal history.

The per-cascade and per-tile arrays are the main structural feature of this file. Because a directional light can have up to four cascades and a light can occupy up to four atlas tiles, the components use a two-dimensional indexing scheme: [eid * MAX_SLOTS + slotIndex]. This keeps every value in a single flat typed array, so a cascade's parameters are contiguous in memory and cache-friendly when the shadow system iterates them.
