API Documentation — src/core/015_rnd_AssetManifest.js

File Purpose

This file is the authoritative asset manifest for the anime lighting stack. It is the declaration of every procedural resource the engine depends on — shader chunks, materials, geometry factories, palette tables, light presets, biome definitions, environment presets, task graph definitions, pool presets, and metadata.

The project is deliberately a no-image-texture project. Every "asset" in this manifest is a procedural descriptor, never a PNG/JPG/KTX/WebP/HDR file. The look is generated from colors, noise, and shader math. This module is where that intent is documented and enforced at the declarative level.

The manifest exists to solve five problems:

1. Single source of truth — every subsystem's dependency is declared in one file. No more scattered relative imports or "wait, does this load before or after that?" confusion.
2. Deterministic load order — dependencies are explicit dependsOn edges. The manifest can compute a topological load order without a runtime solver.
3. Per-tier filtering — PERF_TIER 'LOW' devices skip assets tagged as HIGH-only so low-end Android devices do not build geometry they will never render.
4. Per-mode filtering — indoor/exterior/biome tags let the engine pre-warm only what the current scene needs, deferring the rest to streamed load.
5. Symbolic path resolution — downstream subsystems import by symbolic id (manifest.getPath('shader:anime_cel_lighting')) instead of hard-coding relative paths. If a file moves, only this manifest changes.

The manifest also tracks per-asset load state (PENDING, LOADING, LOADED, FAILED, SKIPPED) so the bootstrap can report progress and the adaptive quality controller can react to missing assets.

---

Exported Constants

PERF_TIER

Type: string

Value: 'LOW' | 'MEDIUM' | 'HIGH' — cached from getPerfTier() at module load.

ASSET_KIND

Type: frozen enum

Values:

· SHADER_CHUNK = 0 — a GLSL chunk module.
· SHADER_MATERIAL = 1 — a full ShaderMaterial factory module.
· GEOMETRY_FACTORY = 2 — a procedural geometry builder.
· PALETTE = 3 — a palette table module.
· LIGHT_PRESET = 4 — a light preset table.
· BIOME_DEF = 5 — a biome definition.
· ENV_PRESET = 6 — an environment preset.
· TASK_GRAPH = 7 — a task graph definition.
· POOL_PRESET = 8 — a pool preset.
· METADATA = 9 — pure data, no runtime behavior.
· COUNT = 10

ASSET_KIND_NAME

Type: frozen array

Values: ['shader_chunk', 'shader_material', 'geometry_factory', 'palette', 'light_preset', 'biome_def', 'env_preset', 'task_graph', 'pool_preset', 'metadata'].

ASSET_STATE

Type: frozen enum

Values:

· PENDING = 0 — registered but not yet loaded.
· LOADING = 1 — load in progress.
· LOADED = 2 — load succeeded.
· FAILED = 3 — load failed.
· SKIPPED = 4 — filtered out by tier/tag.

ASSET_STATE_NAME

Type: frozen array

Values: ['pending', 'loading', 'loaded', 'failed', 'skipped'].

ASSET_TIER

Type: frozen enum

Minimum device tier required to load this asset.

Values:

· ANY = 0 — loadable on any device.
· LOW = 1 — requires LOW or higher.
· MEDIUM = 2 — requires MEDIUM or higher.
· HIGH = 3 — requires HIGH only.

ASSET_TAG

Type: frozen object of bitmask flags

Scene/biome/purpose tags. A tag value is a bitwise OR of any of these.

Values:

· NONE = 0
· INDOOR = 1 << 0
· EXTERIOR = 1 << 1
· DESERT = 1 << 2
· SNOW = 1 << 3
· SEA = 1 << 4
· HOUSE = 1 << 5
· CANYON = 1 << 6
· FOREST = 1 << 7
· CRITICAL = 1 << 8
· OPTIONAL = 1 << 9
· DEBUG = 1 << 10
· WORKER = 1 << 11

The CRITICAL tag means the asset always loads regardless of the scene tag filter. The OPTIONAL and DEBUG tags are excluded by default.

MAX_ASSETS

Type: number

Value: 512

The fixed capacity of the asset table.

MAX_LOADERS

Type: number

Value: ASSET_KIND.COUNT (10).

The maximum number of registered loader functions, one per asset kind.

---

Module-Level State (Not Exported Directly)

_defaultManifest

Type: AssetManifest | null

The module-level singleton, created on first getDefaultManifest() call and immediately populated with the canonical lighting manifest.

_now()

Internal helper returning the current high-resolution timestamp.

_tierAllowed(assetTier, deviceTier)

Internal helper. Returns true if the asset tier is allowed on the device tier. ANY is always allowed. LOW requires the device to be at least 'LOW'. MEDIUM requires at least 'MEDIUM'. HIGH requires 'HIGH'.

---

Exported Class — AssetRecord

One instance per registered asset.

Constructor

```
new AssetRecord()
```

Instance Properties

· id — the asset's unique symbolic id (e.g. 'shader:anime_cel_lighting').
· kind — one of ASSET_KIND.
· path — the module path relative to src/.
· tier — one of ASSET_TIER.
· tags — the tag bitmask.
· dependsOn — a frozen array of asset ids, or null.
· state — one of ASSET_STATE.
· loadedAt — the timestamp of the last successful load.
· resolved — the module namespace returned by the loader.
· loader — an optional per-record loader function that overrides the kind's default.
· priority — the load order priority within a topological depth. Lower runs earlier.
· group — a free-form grouping string (e.g. 'shader_base', 'palette', 'core').

Instance Methods

reset()

Returns: nothing. Clears every field.

---

Exported Class — AssetManifest

The main manifest.

Constructor

```
new AssetManifest(options = {})
```

Parameters:

· tier — the device's performance tier. Default PERF_TIER.
· tags — a tag bitmask to filter against. Default ASSET_TAG.NONE (no filter).
· includeOptional — if true, loads OPTIONAL assets too. Default true.
· includeDebug — if true, loads DEBUG assets too. Default false.
· autoLoad — reserved. Default false.
· concurrency — the number of parallel loads. Default 6 on HIGH, 4 on MEDIUM, 2 on LOW.

Constructor work:

1. Allocates records — the array of AssetRecord instances.
2. Initializes count = 0 and indexById (a Map).
3. Allocates loaders — an array of MAX_LOADERS entries, all null.
4. Initializes the stats object.
5. Allocates _listeners (Map).
6. Calls _installDefaultLoaders() which installs the default ESM loader for every kind.

Instance Properties

· options — the merged options.
· capacity — the slot capacity.
· records — the asset record array.
· count — the number of registered assets.
· indexById — the Map from id to array index.
· loaders — the per-kind loader array.
· frame — the frame counter.
· stats — the aggregate stats object.

The stats object has: registered, loaded, failed, skipped, loadMs.

Instance Methods

on(event, fn)

Parameters:

· event — one of 'registered', 'loading', 'loaded', 'failed', 'batch-complete'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

registerLoader(kind, fn)

Parameters:

· kind — one of ASSET_KIND.
· fn — a loader function (record) => Promise<unknown>.

Returns: boolean.

Purpose: overrides the default loader for a kind. By default every kind uses the ESM dynamic-import loader.

_installDefaultLoaders()

Internal. Installs the default ESM loader for every kind. The loader is async (record) => await import(/* @vite-ignore */ record.path).

register(spec)

Parameters: spec — an asset descriptor with these fields:

· id — required. A unique string.
· kind — one of ASSET_KIND.
· path — a module path relative to src/.
· tier — one of ASSET_TIER. Default ANY.
· tags — the tag bitmask. Default NONE.
· dependsOn — an optional array of parent asset ids.
· priority — an integer load priority. Default 100.
· group — an optional group name. Default 'default'.
· loader — an optional per-record loader function.

Returns: the created AssetRecord, or null on failure.

Purpose: registers a single asset. Idempotent — registering the same id twice returns the existing record. Emits registered.

registerMany(specs)

Parameters: specs — an array of specs.

Returns: the number of assets successfully registered.

Purpose: convenience wrapper for bulk registration.

get(id)

Parameters: id — the asset id.

Returns: the AssetRecord, or null.

getPath(id)

Parameters: id — the asset id.

Returns: the module path, or null.

getResolved(id)

Parameters: id — the asset id.

Returns: the loaded module namespace, or null.

getState(id)

Parameters: id — the asset id.

Returns: one of ASSET_STATE.

isLoaded(id)

Parameters: id — the asset id.

Returns: boolean.

resolveLoadOrder()

Returns: an array of asset ids in the correct load order.

Purpose: applies the tier and tag filter to every registered asset, marks filtered assets as SKIPPED, and topologically sorts the remaining ones using their dependsOn edges.

The sort is Kahn's algorithm. Within each depth, assets are ordered by ascending priority for determinism.

If a cycle is detected, the method logs a warning and appends the unresolved assets to the end of the order array. The pipeline still runs, but cyclic dependencies are surfaced in the log.

_passesFilter(rec)

Internal. Applies the tier check via _tierAllowed, the optional filter, the debug filter, and the tag intersection. CRITICAL assets always pass.

load(id)

Parameters: id — the asset id.

Returns: a Promise resolving to the loaded module namespace, or null if the asset is skipped or its loader fails.

Purpose: loads a single asset, first ensuring all its dependencies are loaded. State transitions: PENDING → LOADING → LOADED or FAILED. Emits loading, then loaded or failed.

loadGroup(groupName)

Parameters: groupName — the group name.

Returns: a Promise resolving to the number of assets loaded.

Purpose: loads every asset in a group in dependency order.

loadAll()

Returns: a Promise resolving to the number of assets loaded.

Purpose: loads the full resolved load order.

_loadBatch(ids)

Internal. Runs load() on every id in the array, using concurrency parallel workers. Each worker pulls the next id from a shared cursor. Returns the number of successful loads.

getStats()

Returns: an object with frame, capacity, count, the five aggregate counters, the tier and tag filters, a byKind array with per-kind counts, a byState array with per-state counts, a groups object mapping group names to counts, and perfTier.

reset()

Returns: this. Clears every record and zeroes the counters.

dispose()

Returns: this. Resets and nulls every internal array and clears listeners.

---

Exported Function — buildCanonicalLightingManifest(manifest)

Parameters: manifest — an AssetManifest instance.

Returns: the same manifest (chainable).

Purpose: populates the manifest with the canonical lighting asset tree. The registration is grouped by purpose:

Shader base chunks (group 'shader_base')

· shader:base → 117_rnd_BaseShader.glsl.js (CRITICAL, priority 10)
· shader:color_utils → 039_rnd_ColorPalette.js (CRITICAL, priority 11)
· shader:normal_quant → 064_scn_normalQuantizer.js (CRITICAL, priority 12)
· shader:perceptual_base → 121_rnd_PerceptualBaseEnhancer.glsl.js (CRITICAL, priority 13, depends on shader:base)

Anime cel lighting core (group 'shader_lighting')

· shader:anime_cel_lighting → 296_lgt_AnimeCelLighting.glsl.js (CRITICAL, priority 20, depends on shader:base)
· shader:anime_rim_light → 297_lgt_AnimeRimLight.glsl.js (CRITICAL, priority 21)
· shader:anime_shadow_tint → 298_lgt_AnimeShadowTint.glsl.js (CRITICAL, priority 22)
· shader:anime_specular → 299_lgt_AnimeSpecular.glsl.js (CRITICAL, priority 23)
· shader:anime_emissive → 300_lgt_AnimeEmissive.glsl.js (CRITICAL, priority 24)
· shader:directional_light → 281_lgt_DirectionalLightShader.glsl.js (CRITICAL, priority 30, depends on shader:anime_cel_lighting)
· shader:cast_shadow → 283_lgt_CastShadowShader.glsl.js (CRITICAL, priority 31, depends on shader:directional_light)

Shadow sampling (group 'shader_shadow')

· shader:shadow_pcf → 291_lgt_PCF.glsl.js (CRITICAL, priority 40)
· shader:shadow_pcss → 292_lgt_PCSS.glsl.js (MEDIUM tier, OPTIONAL, priority 41)
· shader:shadow_vsm → 293_lgt_VSM.glsl.js (HIGH tier, OPTIONAL, priority 42)
· shader:cascaded_shadow → 289_lgt_CascadedShadowSampling.glsl.js (MEDIUM, OPTIONAL, priority 43)
· shader:contact_shadow → 290_lgt_ContactShadowSampling.glsl.js (MEDIUM, OPTIONAL, priority 44)

GI / AO (groups 'shader_gi' and 'shader_ao')

· shader:irradiance_volume → 301_lgt_IrradianceVolumeSampling.glsl.js (CRITICAL, priority 50)
· shader:spherical_harmonics → 302_lgt_SphericalHarmonics.glsl.js (MEDIUM, OPTIONAL, priority 51)
· shader:ssao → 305_lgt_ScreenSpaceAmbientOcclusion.glsl.js (MEDIUM, OPTIONAL, priority 52)
· shader:hbao → 306_lgt_HBAO.glsl.js (MEDIUM, OPTIONAL, priority 53)
· shader:gtao → 307_lgt_GTAO.glsl.js (HIGH, OPTIONAL, priority 54)
· shader:multiscatter_ao → 308_lgt_MultiScatterAO.glsl.js (HIGH, OPTIONAL, priority 55)
· shader:ssgi → 309_lgt_ScreenSpaceGI.glsl.js (HIGH, OPTIONAL, priority 56)

Cluster / forward+ / rect-area (group 'shader_cluster')

· shader:clustered_lighting → 285_lgt_ClusteredLighting.glsl.js (CRITICAL, priority 60)
· shader:forward_plus → 287_lgt_ForwardPlusLighting.glsl.js (CRITICAL, priority 61)
· shader:cluster_index → 323_lgt_ClusterIndex.glsl.js (CRITICAL, priority 62)
· shader:light_list → 322_lgt_LightList.glsl.js (CRITICAL, priority 63)
· shader:rect_area_ltc → 326_lgt_RectAreaLightLTC.glsl.js (MEDIUM, OPTIONAL, priority 64)

Environment (group 'shader_env')

· shader:sky → 279_lgt_SkyShader.glsl.js (CRITICAL, priority 70)
· shader:clouds → 280_lgt_CloudsShader.glsl.js (CRITICAL, priority 71)
· shader:sun_rays → 282_lgt_SunRaysShader.glsl.js (CRITICAL, priority 72)
· shader:volumetric_scatter → 314_lgt_VolumetricLightScattering.glsl.js (MEDIUM, OPTIONAL, priority 73)
· shader:sun_disk → 315_lgt_SunDisk.glsl.js (CRITICAL, priority 74)
· shader:moon_phase → 316_lgt_MoonPhase.glsl.js (CRITICAL, priority 75)
· shader:starfield → 317_lgt_Starfield.glsl.js (CRITICAL, priority 76)
· shader:aurora → 318_lgt_Aurora.glsl.js (HIGH, OPTIONAL + SNOW, priority 77)
· shader:cloud_shadow → 319_lgt_CloudShadow.glsl.js (MEDIUM, OPTIONAL, priority 78)

Palettes (group 'palette')

· palette:reference → 338_lgt_ReferenceImagePalettes.js (CRITICAL, priority 100)
· palette:house → 339_lgt_HousePalette.js (CRITICAL + HOUSE, priority 101)
· palette:canyon → 340_lgt_CanyonPalette.js (CRITICAL + CANYON, priority 102)
· palette:desert → 341_lgt_DesertPalette.js (CRITICAL + DESERT, priority 103)
· palette:snow → 342_lgt_SnowPalette.js (CRITICAL + SNOW, priority 104)
· palette:indoor → 343_lgt_IndoorPalette.js (CRITICAL + INDOOR, priority 105)
· palette:outdoor → 344_lgt_OutdoorPalette.js (CRITICAL + EXTERIOR, priority 106)

Presets (group 'preset')

· preset:lights → 337_lgt_LightPresets.js (CRITICAL, priority 120)
· preset:shadow_defaults → 345_lgt_ShadowDefaults.js (CRITICAL, priority 121)
· preset:gi_defaults → 346_lgt_GIDefaults.js (CRITICAL, priority 122)
· preset:ao_defaults → 347_lgt_AODefaults.js (CRITICAL, priority 123)
· preset:light_budgets → 348_lgt_LightBudgets.js (CRITICAL, priority 124)

Biome defs (group 'biome')

· biome:defs → 094_scn_BiomeDefs.js (CRITICAL, priority 140)
· biome:themes → 093_scn_ThemeDefs.js (CRITICAL, priority 141)

Environment presets (group 'env')

· env:house → 216_lgt_HouseEnvironmentPreset.js (HOUSE, priority 160)
· env:canyon → 217_lgt_CanyonEnvironmentPreset.js (CANYON, priority 161)
· env:desert → 218_lgt_DesertEnvironmentPreset.js (DESERT, priority 162)
· env:snow → 219_lgt_SnowEnvironmentPreset.js (SNOW, priority 163)
· env:climate_presets → 209_lgt_ClimatePresets.js (CRITICAL, priority 164)

Task graph and pool presets (group 'core')

· graph:standard → ./008_rnd_TaskGraph.js (CRITICAL, priority 180)
· pool:object → ./010_rnd_ObjectPool.js (CRITICAL, priority 181)
· pool:typed → ./011_rnd_TypedArrayPool.js (CRITICAL, priority 182)
· pool:buffer → ./012_rnd_BufferPool.js (CRITICAL, priority 183)
· pool:render_target → ./013_rnd_RenderTargetPool.js (CRITICAL, priority 184)

Roughly 55 assets in total. Every one is procedural.

---

Exported Functions

getDefaultManifest(options)

Parameters: options — same as the constructor. Only used on first call.

Returns: the module-level singleton AssetManifest, creating it on first call and immediately calling buildCanonicalLightingManifest() on it.

disposeDefaultManifest()

Returns: nothing.

createAssetManifest(options = {})

Returns: a new AssetManifest.

---

Default Export

The default export bundles: AssetManifest, AssetRecord, createAssetManifest, getDefaultManifest, disposeDefaultManifest, buildCanonicalLightingManifest, ASSET_KIND, ASSET_KIND_NAME, ASSET_STATE, ASSET_STATE_NAME, ASSET_TIER, ASSET_TAG, MAX_ASSETS.

---

Usage Pattern

The bootstrap loads the manifest once at startup:

```
import {
  getDefaultManifest,
  ASSET_TAG,
  ASSET_TIER,
} from './src/core/015_rnd_AssetManifest.js';

const manifest = getDefaultManifest({
  tier: 'HIGH',
  tags: ASSET_TAG.CANYON | ASSET_TAG.EXTERIOR,
  includeOptional: true,
});

const loaded = await manifest.loadAll();
console.log(`Loaded ${loaded} assets`);

// Later, look up a shader chunk by name:
const celLightingMod = manifest.getResolved('shader:anime_cel_lighting');
const celChunkGLSL = celLightingMod.GLSL_ANIME_CEL_LIGHTING;
```

A subsystem that wants to load only its own group:

```
const n = await manifest.loadGroup('palette');
```

A debug HUD that wants to display load stats:

```
const stats = manifest.getStats();
console.log(`Loaded ${stats.loaded}/${stats.count}, failed ${stats.failed}, skipped ${stats.skipped}`);
```

The manifest is what guarantees that the engine never accidentally pulls in an image texture, never loads a file before its dependencies, and never re-imports a shader chunk twice.
