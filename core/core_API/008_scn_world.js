API Documentation — src/core/008_scn_world.js

File Purpose

This file is the single entry point that bridges the anime lighting stack with the underlying procedural world core (world.js). It owns:

· The single bitECS 0.4.0 world instance used by the entire engine
· All Shared-Of-Arrays (SoA) component definitions for lighting
· The single THREE.WebGLRenderer, THREE.Scene, and THREE.PerspectiveCamera
· The bridge to ProceduralWorldCore (which lives in core/world.js)
· The per-frame synchronization between ECS state and Three.js light instances
· The accessor surface that every downstream lighting module (006–380) reads from

The design constraint is that this file is the only place in the codebase that creates the world, renderer, scene, or camera. Every other module — 002_rnd_App.js, 003_rnd_Runtime.js, 001_lgt_ThreeLightsOnlyPolicy.js, and every lighting system — imports its handles from here, so there is exactly one source of truth per resource.

---

Exported Constants

MAX_ENTITIES

Type: number

Value: 100000

The fixed capacity for every SoA component array. All typed arrays in this module are allocated once at module load with exactly this length. Nothing in the engine ever resizes these arrays — that is a hard invariant because resizing typed arrays mid-gameplay on Android would cause GC spikes and invalidate every cached entity id held by downstream systems.

Transform

Type: { x, y, z, qx, qy, qz, qw, sx, sy, sz } — plain object of typed arrays

SoA component representing an entity's world transform. Each field is a Float32Array(MAX_ENTITIES):

· x, y, z — world-space position
· qx, qy, qz, qw — orientation quaternion (w = scalar component, rest = vector)
· sx, sy, sz — per-axis scale

Used by cameras, GI probes, and any entity that needs a spatial anchor.

LightRef

Type: { type, intensity, colorR, colorG, colorB, range, spotAngle, penumbra, castShadow, indoor, priority }

SoA component for every light in the world. Fields:

· type — Uint8Array, value from LIGHT_TYPE enum (0 = sun, 1 = moon, 2 = hemisphere, 3 = ambient, 4 = point, 5 = spot, 6 = rect area, 7 = emissive)
· intensity — Float32Array, scalar intensity in the light's native units
· colorR, colorG, colorB — Float32Array, linear RGB components in [0, 1]
· range — Float32Array, effect radius for point/spot (0 = infinite)
· spotAngle — Float32Array, cone angle in radians for spot
· penumbra — Float32Array, soft-edge factor for spot
· castShadow — Uint8Array, 0 or 1 flag
· indoor — Uint8Array, 0 or 1 flag, used by interior/exterior blending
· priority — Uint8Array, 0..255, used by light budget sorting

ShadowRef

Type: { cascadeCount, bias, normalBias, softness, atlasTileX, atlasTileY, atlasTileW, atlasTileH }

Per-light shadow configuration. Cascade count is Uint8Array (1–4); bias and normal bias are Float32Array; atlas tile coordinates are Uint16Array (populated by the shadow atlas packer in src/shadows/080_lgt_ShadowAtlasPacker.js).

GIRef

Type: { irradianceR, irradianceG, irradianceB, skyOcclusion, indoorFactor, dirty }

Per-entity global illumination state. dirty is a Uint8Array flag that the GI system clears after re-baking.

AORef

Type: { occlusion, radius, intensity }

Per-entity ambient occlusion state. occlusion is the final AO value used by the shader; radius and intensity are per-entity overrides.

ActiveTag

Type: Uint8Array(MAX_ENTITIES)

Simple active/inactive flag per entity. Reads are O(1), writes only touch a single byte.

world

Type: bitECS world handle

The single bitECS 0.4.0 world created via createWorld({ components, time }). Uses the 0.4.0 API (plain objects with typed arrays), NOT the legacy defineComponent + Types API.

LIGHT_TYPE

Type: frozen enum

Maps symbolic names to integer ids used in LightRef.type. Values: SUN=0, MOON=1, HEMI=2, AMBIENT=3, POINT=4, SPOT=5, RECT=6, EMISSIVE=7.

---

Exported Functions

initializeWorld(options = {})

Returns: the ProceduralWorldCore instance (the object created by createProceduralWorldCore).

Purpose: this is the ONLY way to bring up the world. It:

1. Calls createProceduralWorldCore from world.js with the merged options.
2. Spawns the four engine-level light entities (sun, moon, hemisphere, ambient) via _spawnLightEntities().
3. Runs _syncActiveTag() to mark those entities active.
4. Returns the core.

If called twice it returns the existing core (idempotent). Options include width, height, pixelRatio, enableShadows, shadowResolution, fov, near, far, cameraX, cameraY, cameraZ, biome, biomeSpeed, autoCycle, cycleTime, timeOfDay, daySpeed, usePaletteLight, preloadAllLayers, sceneShadows, seed, ppu, autostart.

bindThreeLights(threeRefs)

Parameters: threeRefs — object with optional sun, moon, hemi, ambient properties, each a THREE.Light instance.

Returns: nothing.

Purpose: downstream code calls this once after creating its Three.js lights, so that syncLightEntitiesToThree can write the ECS state into the actual Three.js objects each frame.

syncLightEntitiesToThree(elapsed)

Parameters: elapsed — total elapsed seconds (unused for now but reserved for future animation).

Returns: nothing.

Purpose: reads the ECS state for the four engine lights and writes it into their bound Three.js instances. Uses module-level scratch Vector3 and Color to avoid per-frame allocations. Called every frame by stepWorld.

stepWorld(dt, elapsed)

Parameters: dt — delta time in seconds; elapsed — total elapsed seconds.

Returns: nothing.

Purpose: the per-frame entry point for the world. Calls _coreInstance.update(dt, elapsed) (which drives the ProceduralWorldCore update) then calls syncLightEntitiesToThree(elapsed).

renderWorld()

Returns: nothing.

Purpose: calls _coreInstance.render(), which invokes renderer.render(scene, camera). The only place in the engine that calls render.

disposeWorld()

Returns: nothing.

Purpose: tears down the ProceduralWorldCore, resets all light entity ids to -1, clears the _initialized flag. After this the engine may call initializeWorld again to bring up a fresh world.

getWorld()

Returns: the bitECS world handle. Use this if you need to call addComponent, query, etc.

getCore()

Returns: the ProceduralWorldCore instance.

getRenderer()

Returns: _coreInstance.renderer or null.

getScene()

Returns: _coreInstance.scene or null.

getCamera()

Returns: _coreInstance.camera or null.

getLightManager()

Returns: _coreInstance.light — the light manager created by ProceduralWorldCore.

getPaletteRGB()

Returns: _coreInstance.palette.rgb — the current blended RGB palette buffer (Float32Array of SLOT_COUNT * 3 floats).

getBiomeWeights()

Returns: _coreInstance.biome.weights — Float32Array of three weights [desert, snow, sea].

getLightEntities()

Returns: the _lightEntities object { sun, moon, hemi, ambient } with numeric entity ids.

getPerfTier()

Returns: the PERF_TIER string ('LOW' | 'MEDIUM' | 'HIGH').

getMobileDprCap()

Returns: the MOBILE_DPR_CAP number (1.25 / 1.75 / 2.0).

isWorldReady()

Returns: boolean — true after initializeWorld has been called.

---

Internal Functions (Not Exported but Documented)

_spawnLightEntities()

Creates the four engine-owned light entities via _makeLightEntity and marks them active in ActiveTag. Runs once at first initializeWorld call.

_makeLightEntity(type, r, g, b, intensity, castShadow)

Returns: a new ECS entity id.

Purpose: creates an entity, initializes every SoA field for it, and attaches the Transform, LightRef, ShadowRef, GIRef, and AORef components via addComponent(world, eid, Component).

_syncActiveTag()

No parameters.

Purpose: iterates the currently-spawned light entities and ensures ActiveTag[eid] = 1. Called once at boot.

---

Re-Exports

For convenience, this module re-exports:

· Biome — biome id enum from world.js
· PaletteSlot — palette slot enum
· Palettes — palette hex/oklab tables
· CoreMath — math utilities from world.js
· ProceduralWorldCore — the class

---

Default Export

The default export is a single frozen object bundling every named export plus the runtime getters (getWorld, getCore, getRenderer, getScene, getCamera, getLightManager, getPaletteRGB, getBiomeWeights, getLightEntities, getPerfTier, getMobileDprCap, isWorldReady, bindThreeLights, syncLightEntitiesToThree, LIGHT_TYPE, the SoA components, Biome, PaletteSlot, Palettes, CoreMath, ProceduralWorldCore, MAX_ENTITIES).

---

Usage Contract

Every downstream lighting module must:

1. Never instantiate a THREE.WebGLRenderer, THREE.Scene, or THREE.PerspectiveCamera directly — pull them from getRenderer(), getScene(), getCamera().
2. Never create a bitECS world — use getWorld().
3. Never resize the SoA component arrays — they are fixed to MAX_ENTITIES.
4. Register any new light it creates with 001_lgt_ThreeLightsOnlyPolicy.js and (optionally) attach it to the engine via bindThreeLights if it is one of the four engine lights.

---
