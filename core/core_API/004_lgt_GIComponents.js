API Documentation — src/ecs/004_lgt_GIComponents.js

File Purpose

This file provides the bitECS 0.4.0 SoA component definitions and entity factories for the global illumination system. It is the ECS-side data layout for every GI probe, every radiance cache entry, every lightfield sample, every SH coefficient slab, and every bounce contribution that the anime lighting stack maintains.

The problem it solves is specific to GI on Android. A GI probe grid can hold thousands of probes. Each probe stores irradiance in three channels, sky occlusion, indoor weight, bounce direction, ambient decay, and a validity flag. When the GI system bakes probes, when it interpolates between them, when it applies bounce lighting to a surface, and when it decays stale probes, it reads and writes every one of those fields. If the fields were stored as JavaScript properties on per-probe objects, the memory overhead alone would be prohibitive (each object carries a hidden class header of 24–48 bytes plus the property slots), and the per-field access would cost a Map lookup plus a property dereference.

By storing every field in a flat typed array indexed by entity id, the GI system accesses each probe's state with a single integer offset. A million-probe grid becomes a handful of multi-megabyte Float32Array slabs that fit cleanly in the CPU cache and transfer to the GPU without conversion.

The file follows the bitECS 0.4.0 architectural redesign:

· No defineComponent, no Types enum, no separate component stores.
· Components are plain JavaScript objects whose fields are typed arrays sized to MAX_ENTITIES.
· Every typed array is allocated once at module load and never resized.

The components declared here are:

1. GIProbe — the core per-probe state: position, irradiance in RGB, sky occlusion, indoor weight, ambient decay, dirtiness.
2. GIBounce — the per-probe bounce contribution: direction to the primary bounce, magnitude, colour tint, and decay.
3. GIRadianceCache — the per-probe cached radiance from the previous frame: previous irradiance, temporal weight, stability score, validity flag.
4. GISphericalHarmonics — the per-probe SH-9 coefficients: nine floats per channel, used for directional irradiance.
5. GILightfield — the per-probe lightfield state: lightfield index, parameterization window, resolution.
6. GIVolume — the per-volume descriptor for irradiance volumes (bounding box, resolution, update rate, dirty state).
7. GIReflection — the per-probe reflection probe state: reflected colour, roughness bias, mip level, parallax correction offsets.
8. GIBudget — the per-probe cost and priority used by the GI budget manager.
9. GIDirty — a compact dirty mask that tracks which probes were invalidated this frame, along with a reason tag.

And the entity factories that spawn pre-configured GI probes and volumes:

· spawnGIProbe
· spawnGIProbeGrid
· spawnGIProbeVolume
· spawnGIProbeForLight
· spawnGIProbeFromEmissive
· spawnGIReflectionProbe
· spawnGILightfieldProbe

Every factory creates an ECS entity, attaches the nine components, initializes them to sensible defaults, and returns the entity id. The factory never creates a THREE.DataTexture or a render target — those are created by the sync system that reads the ECS state each frame and writes it into the GI render targets.

---

Exported Constants

MAX_ENTITIES

Type: number

Value: 100000

The fixed capacity for every SoA component array. Matches the value used across the entire engine.

GI_PROBE_TYPE

Type: frozen enum

The kind of GI probe.

Values:

· SURFACE = 0 — a probe attached to a mesh surface. Used for character GI and close-range lighting.
· VOLUME = 1 — a probe inside an irradiance volume. The canonical probe type.
· REFLECTION = 2 — a reflection probe. Captures a cubemap for specular GI.
· LIGHTFIELD = 3 — a lightfield probe. Captures directional radiance for parallax-corrected GI.
· SKY = 4 — a sky probe. Captures the sky radiance. One per scene.
· GROUND = 5 — a ground probe. Captures ground bounce. One per scene or per biome.
· EMISSIVE = 6 — an emissive probe. Attached to an emissive object to inject its contribution.
· COUNT = 7

GI_PROBE_TYPE_NAME

Type: frozen array

Values: ['surface', 'volume', 'reflection', 'lightfield', 'sky', 'ground', 'emissive'].

GI_PROBE_STATE

Type: frozen enum

The lifecycle state of a probe.

Values:

· INVALID = 0 — the probe has no data and must not be sampled.
· STALE = 1 — the probe has data that has exceeded its staleness budget.
· PARTIAL = 2 — the probe has data for some directions only.
· READY = 3 — the probe has fresh data and can be sampled.
· FROZEN = 4 — the probe is intentionally frozen and never re-baked.
· COUNT = 5

GI_PROBE_STATE_NAME

Type: frozen array

Values: ['invalid', 'stale', 'partial', 'ready', 'frozen'].

GI_DIRTY_REASON

Type: frozen enum

The reason a probe was invalidated this frame. Used by the GI scheduler to prioritize re-bakes.

Values:

· NONE = 0
· LIGHT_CHANGED = 1 — a light that affects the probe changed.
· GEOMETRY_CHANGED = 2 — nearby geometry changed.
· BIOME_CHANGED = 3 — the biome weights changed.
· DAY_CYCLE_CHANGED = 4 — the day cycle changed.
· INTERIOR_ENTERED = 5 — the probe went from outdoors to indoors.
· INTERIOR_EXITED = 6 — the probe went from indoors to outdoors.
· QUALITY_CHANGED = 7 — the quality level changed.
· MANUAL = 8 — the probe was invalidated by an explicit call.
· COUNT = 9

GI_DIRTY_REASON_NAME

Type: frozen array

Values: ['none', 'light_changed', 'geometry_changed', 'biome_changed', 'day_cycle_changed', 'interior_entered', 'interior_exited', 'quality_changed', 'manual'].

GI_STATE_FLAG

Type: frozen object of bit flags

Per-probe state flags packed into GIProbe.flags.

· ENABLED = 1 << 0
· DIRTY = 1 << 1
· BAKING = 1 << 2
· SYNCED = 1 << 3 — the probe's data has been uploaded to the GPU this frame.
· INDOOR = 1 << 4
· OUTDOOR = 1 << 5
· STATIC = 1 << 6 — the probe is a static probe that only re-bakes on demand.
· DYNAMIC = 1 << 7 — the probe is a dynamic probe that re-bakes every frame.
· OCCLUDED = 1 << 8 — the probe is currently occluded from the camera.
· HIGH_QUALITY = 1 << 9 — the probe should use the high-quality sampling path.
· BOUNCE_ONLY = 1 << 10 — the probe only carries bounce light.
· SKY_ONLY = 1 << 11 — the probe only carries sky light.
· RESERVED_BIT_12 = 1 << 12
· RESERVED_BIT_13 = 1 << 13
· RESERVED_BIT_14 = 1 << 14
· RESERVED_BIT_15 = 1 << 15

GI_BOUNCE_FLAG

Type: frozen object of bit flags

Per-bounce contribution flags packed into GIBounce.flags.

· ACTIVE = 1 << 0
· MULTI_BOUNCE = 1 << 1
· EMISSIVE = 1 << 2
· SPECULAR = 1 << 3
· RESERVED_BIT_4 = 1 << 4
· RESERVED_BIT_5 = 1 << 5
· RESERVED_BIT_6 = 1 << 6
· RESERVED_BIT_7 = 1 << 7

GI_DEFAULT_LATTICE_RES

Type: number

Value: 32 on HIGH, 16 on MEDIUM, 8 on LOW.

The default irradiance volume lattice resolution along each axis. A 32×32×32 lattice holds 32768 probes.

GI_DEFAULT_SPACING

Type: number

Value: 2.0 on HIGH, 4.0 on MEDIUM, 6.0 on LOW.

The default probe spacing in world units. Computed from the RAM class by 019_rnd_AndroidProfile.js.

GI_SH_COEFFICIENTS

Type: number

Value: 9

The number of spherical harmonic coefficients per channel. The engine uses first-order SH (four coefficients) plus five second-order coefficients for a richer directional response. The nine coefficients are stored as nine floats per channel per probe.

GI_RADIANCE_SAMPLES

Type: number

Value: 8

The number of radiance samples per probe stored for the reflection path. Reflection probes store eight samples arranged as a small disk.

GI_DEFAULT_UPDATE_HZ

Type: number

Value: 20 on HIGH, 15 on MEDIUM, 8 on LOW.

The default probe re-bake rate in Hz.

GI_MAX_BOUNCES

Type: number

Value: 3

The maximum number of bounce iterations the GI solver performs per frame. Higher tiers use more bounces.

---

Exported Components

GIProbe

The core per-probe state.

Fields:

· type — Uint8Array. One of GI_PROBE_TYPE.
· state — Uint8Array. One of GI_PROBE_STATE.
· flags — Uint16Array. A bitmask of GI_STATE_FLAG.
· x, y, z — Float32Array. World-space position.
· irradianceR — Float32Array. Red channel of the probe's irradiance.
· irradianceG — Float32Array. Green channel.
· irradianceB — Float32Array. Blue channel.
· skyOcclusion — Float32Array. The probe's sky occlusion in [0, 1]. 0 means fully occluded from the sky, 1 means fully exposed.
· indoorWeight — Float32Array. The probe's interior/exterior blend in [0, 1]. 0 means outdoors, 1 means indoors.
· ambientDecay — Float32Array. The probe's ambient decay factor in [0, 1]. Multiplied into the base ambient contribution.
· lightListHash — Uint32Array. A hash of the light list that last affected this probe. Used for change detection.
· biomeHash — Uint32Array. A hash of the biome weights that last affected this probe.
· lastBakeFrame — Uint32Array. The frame when the probe was last baked.
· bakeCount — Uint32Array. The total number of times the probe has been baked.
· invalidatedAtFrame — Uint32Array. The frame when the probe was last invalidated.
· volumeEid — Int32Array. The entity id of the volume that owns this probe, or -1.
· latticeX, latticeY, latticeZ — Uint16Array. The probe's lattice coordinates within its volume.

GIBounce

The per-probe bounce contribution.

Fields:

· flags — Uint8Array. A bitmask of GI_BOUNCE_FLAG.
· dirX, dirY, dirZ — Float32Array. The direction to the primary bounce contribution.
· magnitude — Float32Array. The bounce magnitude.
· colorR, colorG, colorB — Float32Array. The bounce tint.
· distance — Float32Array. The distance to the bounce source.
· decay — Float32Array. The bounce decay.
· weight — Float32Array. The bounce weight in [0, 1].

GIRadianceCache

The per-probe temporal radiance cache.

Fields:

· prevIrradianceR — Float32Array. The previous frame's red channel.
· prevIrradianceG — Float32Array. The previous frame's green channel.
· prevIrradianceB — Float32Array. The previous frame's blue channel.
· temporalWeight — Float32Array. The temporal blend weight in [0, 1].
· stabilityScore — Float32Array. A rolling stability score. When the score drops below a threshold, the cache is reset.
· accumFrames — Uint16Array. The number of consecutive frames the cache has been valid.
· valid — Uint8Array. 1 if the cache is valid, 0 otherwise.
· lastResetFrame — Uint32Array. The frame of the last cache reset.

GISphericalHarmonics

The per-probe spherical harmonic coefficients. Nine coefficients per channel, stored as three separate Float32Array slabs sized MAX_ENTITIES * 9.

Fields:

· coeffR — Float32Array(MAX_ENTITIES * 9). The nine red-channel coefficients. Indexed as [eid * 9 + i].
· coeffG — Float32Array(MAX_ENTITIES * 9). The nine green-channel coefficients.
· coeffB — Float32Array(MAX_ENTITIES * 9). The nine blue-channel coefficients.
· l0R, l0G, l0B — Float32Array. Cached L0 (DC) coefficient per channel. Redundant with coeffR[0] but stored for fast access.
· directionalBias — Float32Array. A multiplier that boosts the directional response in [0, 2].
· shVersion — Uint32Array. The version counter for the SH coefficients. Incremented whenever the coefficients are updated.

GILightfield

The per-probe lightfield state.

Fields:

· lightfieldIndex — Uint32Array. The index of the lightfield entry within the global lightfield buffer.
· windowWidth — Uint16Array. The width of the parameterization window in probes.
· windowHeight — Uint16Array. The height.
· resolutionU — Uint8Array. The resolution along the U parameter.
· resolutionV — Uint8Array. The resolution along the V parameter.
· parallaxX — Float32Array. The parallax correction offset along X.
· parallaxY — Float32Array. The parallax correction offset along Y.
· parallaxZ — Float32Array. The parallax correction offset along Z.
· depthBias — Float32Array. The depth bias applied during lookup.

GIVolume

The per-volume descriptor. A GI volume is a bounding box that owns a lattice of probes.

Fields:

· minX, minY, minZ — Float32Array. The volume's minimum corner.
· maxX, maxY, maxZ — Float32Array. The volume's maximum corner.
· resX, resY, resZ — Uint16Array. The lattice resolution along each axis.
· spacing — Float32Array. The world-space distance between adjacent probes.
· updateHz — Float32Array. The volume's re-bake rate.
· budgetClass — Uint8Array. The volume's budget class. 0 = always on, 1 = high priority, 2 = normal, 3 = low priority.
· priority — Uint8Array. The volume's priority in [0, 255].
· flags — Uint16Array. A bitmask of GI_STATE_FLAG. Uses the same flag set.
· probeCount — Uint32Array. The number of probes in the volume.
· probeCapacity — Uint32Array. The maximum number of probes.
· lastFullBakeFrame — Uint32Array. The frame of the last full bake.
· lastPartialBakeFrame — Uint32Array. The frame of the last partial bake.
· dirtyProbeCount — Uint32Array. The number of probes in the volume that are currently dirty.

GIReflection

The per-probe reflection state.

Fields:

· reflectionR, reflectionG, reflectionB — Float32Array. The probe's reflected colour.
· roughnessBias — Float32Array. A multiplier applied to the surface roughness during reflection lookup.
· mipLevel — Float32Array. The suggested mip level for the reflection lookup.
· parallaxEnabled — Uint8Array. 1 if parallax correction is enabled.
· parallaxOffsetX, parallaxOffsetY, parallaxOffsetZ — Float32Array. The correction offsets.
· roughnessThreshold — Float32Array. The roughness above which the probe contributes no reflection.

GIBudget

The per-probe budget state.

Fields:

· costEstimate — Float32Array. The estimated per-frame cost of baking the probe.
· currentCost — Float32Array. The cost charged to the probe this frame.
· priority — Uint8Array. The probe's priority in [0, 255].
· budgetClass — Uint8Array. The probe's budget class.
· lastUpdateFrame — Uint32Array. The frame of the last cost update.

GIDirty

The compact dirty mask and reason tag.

Fields:

· mask — Uint32Array. A bit-packed mask of which probes in the grid are dirty. One bit per probe.
· reason — Uint8Array. One of GI_DIRTY_REASON.
· count — Uint32Array. The number of dirty probes.
· firstDirtyIndex — Uint32Array. The lowest dirty probe index. Used to seed the bake scheduler.
· lastInvalidateFrame — Uint32Array. The frame of the last dirty mask update.

The GIDirty component is the fast path for the GI scheduler. Instead of iterating every probe every frame to find the dirty ones, the scheduler reads the dirty mask directly and iterates the set bits.

---

Exported Functions

_initCommonFields(world, eid)

Internal. Initializes every field of every GI component for the given entity to sensible defaults.

Sets:

· GIProbe defaults: type VOLUME, state INVALID, flags 0, position 0, irradiance 0, skyOcclusion 1, indoorWeight 0, ambientDecay 1, hashes 0, timestamps 0, volumeEid -1, lattice coords 0.
· GIBounce defaults: flags 0, direction 0, magnitude 0, colour 0, distance 0, decay 0, weight 0.
· GIRadianceCache defaults: prev irradiance 0, temporalWeight 0.5, stability 0, accum 0, valid 0, last reset 0.
· GISphericalHarmonics defaults: all nine coefficients 0, L0 0, directionalBias 1, version 0.
· GILightfield defaults: index 0, window 0, resolution 0, parallax offsets 0, depth bias 0.
· GIVolume defaults: bounding box 0, resolution 0, spacing 2, updateHz 20, budgetClass 2, priority 128, flags 0, probe counts 0, timestamps 0.
· GIReflection defaults: colour 0, roughness bias 1, mip 0, parallax disabled, offsets 0, threshold 1.
· GIBudget defaults: cost 1, current 0, priority 128, class 2, last update 0.
· GIDirty defaults: mask 0, reason NONE, count 0, first index 0, last frame 0.

spawnGIProbe(world, options = {})

Parameters:

· world — the bitECS world handle.
· options.type — one of GI_PROBE_TYPE. Default VOLUME.
· options.position — a [x, y, z] array. Default [0, 0, 0].
· options.volumeEid — the entity id of the owning volume. Default -1.
· options.latticeX, options.latticeY, options.latticeZ — the probe's lattice coordinates within its volume. Default 0.
· options.spacing — reserved for future use.
· options.flags — an optional bitmask to OR into the initial state.

Returns: the entity id.

Purpose: the canonical single-probe spawn. Used for one-off probes, character GI probes, and probes that do not belong to a full volume. Sets GIProbe.flags to ENABLED | STATIC by default.

spawnGIProbeGrid(world, options = {})

Parameters:

· world — the bitECS world handle.
· options.min — a [x, y, z] array for the volume's minimum corner.
· options.max — a [x, y, z] array for the volume's maximum corner.
· options.resolution — the lattice resolution along each axis. Default GI_DEFAULT_LATTICE_RES.
· options.spacing — the world-space distance between probes. Default GI_DEFAULT_SPACING.
· options.updateHz — the re-bake rate. Default GI_DEFAULT_UPDATE_HZ.
· options.budgetClass — the volume's budget class. Default 2.
· options.priority — the volume's priority. Default 128.

Returns: an object with volumeEid (the volume entity id) and probeEids (an array of every spawned probe entity id).

Purpose: spawns a full irradiance volume with resX * resY * resZ probes. Creates the volume entity first, then iterates the lattice and spawns one probe per cell. Each probe is positioned at its lattice cell's world-space center, given its latticeX/Y/Z coordinates, and linked to the volume via volumeEid. The volume's probeCount and probeCapacity are set to the total.

spawnGIProbeVolume(world, options = {})

Parameters: same as spawnGIProbeGrid, plus:

· options.deferProbes — if true, the volume is created without spawning individual probe entities. Probe spawning is done lazily when the first bake is requested.

Returns: the volume entity id.

Purpose: a lighter-weight volume that spawns only the volume entity. Used when the volume's probe count would exceed the entity budget, or when the volume's probes are streamed in gradually.

spawnGIProbeForLight(world, lightEid, options = {})

Parameters:

· world — the bitECS world handle.
· lightEid — the light entity that the probe should follow.
· options.offset — an optional [x, y, z] offset from the light's position.
· options.type — one of GI_PROBE_TYPE. Default SURFACE.

Returns: the entity id.

Purpose: spawns a probe that tracks a specific light. Used for character GI, magic glow GI, and fire GI. The probe's position is updated by the GI sync system each frame from the light's current position plus the offset.

spawnGIProbeFromEmissive(world, emissiveEid, options = {})

Parameters:

· world — the bitECS world handle.
· emissiveEid — the emissive object's entity id.
· options.radius — the effective radius of the emissive's GI contribution. Default 2.
· options.color — the emissive's colour.

Returns: the entity id.

Purpose: spawns a probe that injects an emissive contribution into the surrounding GI. Sets GIProbe.type to EMISSIVE and marks GIBounce.flags with EMISSIVE. The probe's irradiance fields are set to the emissive's colour scaled by its intensity.

spawnGIReflectionProbe(world, options = {})

Parameters:

· world — the bitECS world handle.
· options.position — the probe's world-space position.
· options.resolution — the reflection cubemap resolution. Default 128.
· options.roughnessThreshold — the roughness above which the probe is not sampled.

Returns: the entity id.

Purpose: spawns a reflection probe. Sets GIProbe.type to REFLECTION and initializes GIReflection. The sync system creates a WebGLCubeRenderTarget for the probe on the next frame.

spawnGILightfieldProbe(world, options = {})

Parameters:

· world — the bitECS world handle.
· options.position — the probe's position.
· options.windowWidth, options.windowHeight — the parameterization window size in probes.
· options.resolutionU, options.resolutionV — the lightfield resolution.

Returns: the entity id.

Purpose: spawns a lightfield probe. Sets GIProbe.type to LIGHTFIELD and initializes GILightfield. Lightfield probes are used for parallax-corrected GI in interior scenes.

setProbeEnabled(world, eid, enabled)

Parameters:

· world — the bitECS world handle.
· eid — the probe's entity id.
· enabled — boolean.

Returns: nothing.

Purpose: sets or clears the ENABLED flag on GIProbe.flags.

setProbePosition(world, eid, x, y, z)

Parameters: the three world-space coordinates.

Returns: nothing.

Purpose: updates the probe's position and sets the DIRTY flag with reason MANUAL.

setProbeIrradiance(world, eid, r, g, b)

Parameters: the three irradiance values.

Returns: nothing.

Purpose: updates the probe's irradiance. Copies the current values into GIRadianceCache before overwriting, so the temporal blend has a previous frame to work with.

setProbeSkyOcclusion(world, eid, occlusion)

Parameters: the occlusion value in [0, 1].

Returns: nothing.

Purpose: updates the probe's sky occlusion.

setProbeIndoorWeight(world, eid, weight)

Parameters: the weight in [0, 1].

Returns: nothing.

Purpose: updates the probe's interior/exterior blend and toggles the INDOOR and OUTDOOR flags accordingly.

setProbeSHCoefficients(world, eid, coeffR, coeffG, coeffB)

Parameters:

· world — the bitECS world handle.
· eid — the probe's entity id.
· coeffR — a nine-element array of red-channel SH coefficients.
· coeffG — a nine-element array of green-channel coefficients.
· coeffB — a nine-element array of blue-channel coefficients.

Returns: nothing.

Purpose: writes the SH coefficients into the per-probe slabs and increments the probe's shVersion.

setProbeVolume(world, eid, volumeEid, latticeX, latticeY, latticeZ)

Parameters:

· world — the bitECS world handle.
· eid — the probe's entity id.
· volumeEid — the entity id of the volume.
· latticeX, latticeY, latticeZ — the lattice coordinates.

Returns: nothing.

Purpose: links a probe to its volume and records its lattice position.

markProbeDirty(world, eid, reason)

Parameters:

· world — the bitECS world handle.
· eid — the probe's entity id.
· reason — one of GI_DIRTY_REASON.

Returns: nothing.

Purpose: sets the DIRTY flag on the probe, records the invalidate frame, and updates the probe's reason tag. Also updates the parent volume's GIDirty mask if the probe has a parent volume.

clearProbeDirty(world, eid)

Parameters: the probe's entity id.

Returns: nothing.

Purpose: clears the DIRTY flag.

clearVolumeDirty(world, volumeEid)

Parameters: the volume's entity id.

Returns: nothing.

Purpose: zeroes the volume's GIDirty.mask and GIDirty.count.

isProbeReady(world, eid)

Parameters: the probe's entity id.

Returns: boolean.

Purpose: true if the probe's state is READY and its ENABLED flag is set.

getProbesInVolume(world, volumeEid, outEids)

Parameters:

· world — the bitECS world handle.
· volumeEid — the volume's entity id.
· outEids — an array to receive matching probe entity ids.

Returns: the number of probes collected.

Purpose: iterates every probe and collects those whose volumeEid matches.

getDirtyProbes(world, outEids)

Parameters:

· world — the bitECS world handle.
· outEids — an array to receive matching probe entity ids.

Returns: the number of probes collected.

Purpose: iterates every probe and collects those with the DIRTY flag set. Used by the GI bake scheduler.

getDirtyProbesInVolume(world, volumeEid, outEids)

Parameters:

· world — the bitECS world handle.
· volumeEid — the volume's entity id.
· outEids — an array to receive matching probe entity ids.

Returns: the number of probes collected.

Purpose: filtered variant of getDirtyProbes.

getReadyProbes(world, outEids)

Parameters:

· world — the bitECS world handle.
· outEids — an array to receive matching probe entity ids.

Returns: the number of probes collected.

Purpose: iterates every probe and collects those whose state is READY. Used by the GI sampler to build the probe list uploaded to the GPU.

updateProbeBudgetCost(world, eid)

Parameters:

· world — the bitECS world handle.
· eid — the probe's entity id.

Returns: the computed cost.

Purpose: updates GIBudget.costEstimate from the probe's type, volume resolution, and SH sample count. Formula: shSampleCount * 9 + bounceCount * bounceSamples, scaled by a type multiplier.

sumGICost(world, volumeEid)

Parameters:

· world — the bitECS world handle.
· volumeEid — an optional volume entity id. If provided, sums only probes in that volume. If -1 or undefined, sums all probes.

Returns: the total cost.

Purpose: the GI budget manager's aggregate estimate.

computeProbeInterpolationWeights(world, x, y, z, outEids, outWeights)

Parameters:

· world — the bitECS world handle.
· x, y, z — the sample position.
· outEids — an array of at least eight entity ids to receive the eight nearest probes.
· outWeights — an array of at least eight floats to receive the corresponding weights.

Returns: the number of probes written.

Purpose: finds the eight probes surrounding the sample position and computes trilinear interpolation weights. Used by the CPU-side GI sampler for character lighting and by the debug tools.

sampleProbeIrradiance(world, x, y, z, outRGB)

Parameters:

· world — the bitECS world handle.
· x, y, z — the sample position.
· outRGB — a three-element array to receive the interpolated irradiance.

Returns: boolean. True if a valid sample was found.

Purpose: the canonical CPU-side irradiance sample. Uses computeProbeInterpolationWeights internally.

getGIStats(world)

Parameters: world — the bitECS world handle.

Returns: an object with:

· total — the total number of GI probes.
· ready — the number of probes in the READY state.
· dirty — the number of probes with the DIRTY flag.
· stale — the number of probes in the STALE state.
· byType — an array of counts indexed by GI_PROBE_TYPE.
· volumeCount — the number of volumes.
· totalProbesInVolumes — the sum of GIVolume.probeCount across all volumes.
· totalCost — the sum of GIBudget.costEstimate across all probes.
· dirtyByReason — an array of counts indexed by GI_DIRTY_REASON.
· avgSkyOcclusion — the mean sky occlusion across all probes.
· avgIndoorWeight — the mean indoor weight across all probes.

Purpose: the debug HUD's primary view of the GI population. Registered as a named source in 025_rnd_StatsCollector.js.

getProbeSnapshotForStats(world, eid)

Parameters:

· world — the bitECS world handle.
· eid — the probe's entity id.

Returns: a plain object with the probe's type, state, position, irradiance, sky occlusion, indoor weight, and dirty status.

Purpose: the per-probe stats snapshot.

clearAllRadianceCaches(world)

Parameters: world — the bitECS world handle.

Returns: the number of probes whose cache was cleared.

Purpose: sets GIRadianceCache.valid to 0 for every probe. Called on context restore, tier change, biome transition, or quality downgrade to force the temporal accumulator to restart.

invalidateAllProbes(world, reason)

Parameters:

· world — the bitECS world handle.
· reason — one of GI_DIRTY_REASON.

Returns: the number of probes invalidated.

Purpose: sets the DIRTY flag and the reason tag on every probe. Used by the biome transition system, the day-cycle system, and the interior/exterior crossfade.

invalidateProbesInVolume(world, volumeEid, reason)

Parameters:

· world — the bitECS world handle.
· volumeEid — the volume's entity id.
· reason — one of GI_DIRTY_REASON.

Returns: the number of probes invalidated.

Purpose: filtered variant of invalidateAllProbes.

updateGIDirtyMask(world, volumeEid)

Parameters:

· world — the bitECS world handle.
· volumeEid — the volume's entity id.

Returns: the number of dirty probes in the volume.

Purpose: rebuilds the volume's GIDirty.mask from the individual probe flags. Called once per frame by the GI scheduler after any invalidation.

---

Exported Default Object

The default export bundles every component, every factory, every setter, every getter, and every aggregate function:

· The nine components: GIProbe, GIBounce, GIRadianceCache, GISphericalHarmonics, GILightfield, GIVolume, GIReflection, GIBudget, GIDirty.
· The enums: GI_PROBE_TYPE, GI_PROBE_STATE, GI_DIRTY_REASON, GI_STATE_FLAG, GI_BOUNCE_FLAG.
· The constants: MAX_ENTITIES, GI_DEFAULT_LATTICE_RES, GI_DEFAULT_SPACING, GI_SH_COEFFICIENTS, GI_RADIANCE_SAMPLES, GI_DEFAULT_UPDATE_HZ, GI_MAX_BOUNCES.
· The seven spawn factories.
· The setters and readers.
· The aggregate functions.
· The interpolation and sampling helpers.

---

Usage Pattern

A subsystem that spawns an irradiance volume and iterates its probes:

```
import {
  spawnGIProbeGrid,
  getProbesInVolume,
  getGIStats,
} from './src/ecs/004_lgt_GIComponents.js';

const { volumeEid, probeEids } = spawnGIProbeGrid(world, {
  min: [-32, 0, -32],
  max: [32, 16, 32],
  resolution: 16,
  spacing: 4.0,
  updateHz: 15,
  budgetClass: 1,
  priority: 200,
});

// Later, when the volume needs to be re-baked:
const dirtyProbes = [];
const n = getProbesInVolume(world, volumeEid, dirtyProbes);
for (let i = 0; i < n; i++) {
  const eid = dirtyProbes[i];
  // read GIProbe.x[eid], GIProbe.irradianceR[eid], etc.
}
```

A subsystem that reads aggregate GI stats:

```
import { getGIStats } from './src/ecs/004_lgt_GIComponents.js';

const stats = getGIStats(world);
console.log(`GI probes: ${stats.ready}/${stats.total} ready`);
console.log(`Dirty: ${stats.dirty}, stale: ${stats.stale}`);
console.log(`Volumes: ${stats.volumeCount}`);
console.log(`Total cost: ${stats.totalCost}`);
```

A subsystem that invalidates every probe when the biome changes:

```
import {
  invalidateAllProbes,
  GI_DIRTY_REASON,
} from './src/ecs/004_lgt_GIComponents.js';

function onBiomeChanged(fromBiome, toBiome) {
  const count = invalidateAllProbes(world, GI_DIRTY_REASON.BIOME_CHANGED);
  console.log(`Invalidated ${count} probes for biome transition`);
}
```

A subsystem that samples the GI at a specific point:

```
import { sampleProbeIrradiance } from './src/ecs/004_lgt_GIComponents.js';

const out = [0, 0, 0];
if (sampleProbeIrradiance(world, 10.5, 2.0, 8.3, out)) {
  console.log(`Irradiance at (10.5, 2.0, 8.3): ${out[0]}, ${out[1]}, ${out[2]}`);
}
```

The GI component definitions are the canonical data layout for the global illumination system. Every subsystem that reads or writes GI state — the volume builder, the probe baker, the SH projector, the bounce calculator, the radiance cache, the interpolation sampler, the interior/exterior crossfade, the biome transition system, the day-cycle system, the budget manager, the reflection probe updater — reads from these typed arrays. Because the layout is fixed and shared, there is exactly one source of truth for each probe's position, irradiance, occlusion, indoor weight, SH coefficients, bounce contribution, temporal history, and budget cost.

The nine-component layout is what makes the GI system's parallelism possible. On a device with multiple cores, each core can bake a disjoint subset of probes without any shared state, because every probe's fields live at a fixed offset in a flat typed array. The shared-array-buffer path in 022_rnd_FeatureDetector.js transfers those slabs to workers with zero copy. On a device without shared memory, the worker receives a copy and returns a copy, but the layout is identical, so the code path is the same.

